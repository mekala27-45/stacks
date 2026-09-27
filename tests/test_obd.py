"""BTS policy reconstruction, sample boundaries and six-file ingestion contracts."""

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from packages.ope import obd


def example_frame():
    index = np.arange(120)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=120, freq="min", tz="UTC").astype(str),
            "item_id": (index // 3) % 4,
            "position": index % 3 + 1,
            "click": (index % 11 == 0).astype(int),
            "propensity_score": np.full(120, 0.25),
        }
    )


def test_vectorized_bts_exactly_matches_upstream_beta_rank_rule():
    alpha, beta = np.array([1.0, 2.0, 4.0, 3.0]), np.array([5.0, 7.0, 2.0, 6.0])
    observed = obd.bts_action_distribution(alpha, beta, simulations=101, seed=12345)
    rng = np.random.RandomState(12345)
    expected = np.zeros((4, 3))
    for _ in range(101):
        selected = rng.beta(alpha, beta).argsort()[::-1][:3]
        for position, action in enumerate(selected):
            expected[action, position] += 1
    np.testing.assert_array_equal(observed, expected / 101)
    np.testing.assert_allclose(observed.sum(axis=0), 1)
    assert np.all(observed.sum(axis=1) <= 1)


@pytest.mark.parametrize(
    "alpha,beta,simulations",
    [
        (np.ones(2), np.ones(2), 10),
        (np.ones(4), np.ones(3), 10),
        (np.zeros(4), np.ones(4), 10),
        (np.ones(4), np.ones(4), 0),
        (np.ones(4), np.ones(4), True),
        (np.full(4, np.nan), np.ones(4), 10),
    ],
)
def test_bts_refuses_empty_or_invalid_policy_parameters(alpha, beta, simulations):
    with pytest.raises(ValueError):
        obd.bts_action_distribution(alpha, beta, simulations, 0)


def test_pinned_prior_refuses_modified_or_empty_source():
    for source in (b"", b"all:\n  alpha:\n    - 1.0\n"):
        with pytest.raises(ValueError, match="checksum"):
            obd.parse_priors(source)


def test_on_policy_interval_accounts_for_zero_click_uncertainty():
    result = obd.on_policy_interval(np.zeros(100))
    assert result["mean"] == 0 and result["high"] > 0
    with pytest.raises(ValueError):
        obd.on_policy_interval(np.array([]))


def test_campaign_uses_shared_cutoff_and_uniform_ips_identity():
    frame = example_frame()
    rows, summary = obd.campaign_evidence("fixture", frame, frame, np.full((4, 3), 0.25), draws=30)
    assert len(rows) == 24
    assert summary["evaluation_rows"] == summary["bts_reference_rows"] == 60
    assert summary["cutoff"] == frame.iloc[60].timestamp
    for row in rows:
        assert row["comparison_identifies_exact_logged_policy"] is False
        if row["estimator"] in ("IPS", "SNIPS"):
            assert row["estimate"]["mean"] == pytest.approx(row["observed_logging"]["mean"])
            assert row["difference_from_bts"]["mean"] == pytest.approx(0)


@pytest.mark.parametrize(
    "field,value",
    [("propensity_score", 0), ("position", 4), ("click", 2), ("item_id", 5), ("timestamp", "invalid")],
)
def test_bad_logged_observation_fails_before_ope(field, value):
    frame = example_frame()
    frame[field] = value
    with pytest.raises(ValueError, match="contract"):
        obd.validate_log(frame, 4)


def test_six_files_are_all_consumed_with_campaign_specific_prior(monkeypatch, tmp_path):
    prior = "".join(
        f"{c}:\n  alpha:\n    - 1.0\n    - 1.0\n    - 1.0\n    - 1.0\n  beta:\n    - 2.0\n    - 2.0\n    - 2.0\n    - 2.0\n"
        for c in obd.CAMPAIGNS
    ).encode()
    monkeypatch.setattr(obd, "PRIOR_SHA", hashlib.sha256(prior).hexdigest())
    consumed = []

    def fake_fetch(cache: Path, name: str, source: str):
        consumed.append(source)
        path = cache / name
        if name == "prior_bts.yaml":
            path.write_bytes(prior)
        elif name == "LICENSE":
            path.write_text("Fixture license")
        else:
            example_frame().to_csv(path, index=False)
        return path, obd.BASE + source

    original = obd.bts_action_distribution
    monkeypatch.setattr(obd, "fetch", fake_fetch)
    monkeypatch.setattr(
        obd,
        "bts_action_distribution",
        lambda alpha, beta, simulations, seed: original(alpha, beta, 100, seed),
    )
    result = obd.run_obd(tmp_path)
    assert len([source for source in consumed if source.endswith(".csv")]) == 6
    assert {row["campaign"] for row in result["rows"]} == set(obd.CAMPAIGNS)
    assert len(result["rows"]) == 72
    assert len(list((tmp_path / "data" / "obd").glob("*.parquet"))) == 6


def test_download_uses_cache_and_rejects_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(obd.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"payload"))
    path, _ = obd.fetch(tmp_path, "data.txt", "data.txt")
    assert path.read_bytes() == b"payload"
    monkeypatch.setattr(obd.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b""))
    assert obd.fetch(tmp_path, "data.txt", "data.txt")[0] == path
    with pytest.raises(ValueError, match="Empty"):
        obd.fetch(tmp_path, "empty.txt", "empty.txt")
