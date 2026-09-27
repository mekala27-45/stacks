"""Six-file Open Bandit benchmark using the official production-prior BTS approximation."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from packages.ope.estimators import estimate_intervals

COMMIT = "8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4"
BASE = f"https://raw.githubusercontent.com/st-tech/zr-obp/{COMMIT}/"
PRIOR_SHA = "81a54fbdc184021ccc62ab97de81ec35461c032e3d7f99ccaa0a9745f732d251"
CAMPAIGNS = ("all", "men", "women")
FloatArray = NDArray[np.float64]
Interval = dict[str, float]


def fetch(cache: Path, name: str, source: str) -> tuple[Path, str]:
    path = cache / name
    if not path.exists():
        request = urllib.request.Request(BASE + source, headers={"User-Agent": "stacks-public-data-demo"})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
        if not payload:
            raise ValueError(f"Empty upstream file: {source}")
        path.write_bytes(payload)
    return path, BASE + source


def parse_priors(source: bytes) -> dict[str, tuple[FloatArray, FloatArray]]:
    """Read the hash-verified file's simple YAML scalar-list schema."""
    if hashlib.sha256(source).hexdigest() != PRIOR_SHA:
        raise ValueError("BTS prior checksum differs from the pinned source")
    values: dict[str, dict[str, list[float]]] = {}
    campaign = parameter = ""
    for line in source.decode("utf-8").splitlines():
        clean = line.strip()
        if not clean:
            continue
        if not line.startswith(" ") and clean.endswith(":"):
            campaign = clean[:-1]
            values[campaign] = {}
        elif clean in ("alpha:", "beta:"):
            parameter = clean[:-1]
            values[campaign][parameter] = []
        elif clean.startswith("- "):
            values[campaign][parameter].append(float(clean[2:]))
        else:
            raise ValueError("Unexpected pinned prior syntax")
    return {
        key: (np.asarray(values[key]["alpha"], dtype=float), np.asarray(values[key]["beta"], dtype=float))
        for key in CAMPAIGNS
    }


def bts_action_distribution(alpha: FloatArray, beta: FloatArray, simulations: int, seed: int) -> FloatArray:
    """Upstream BernoulliTS beta sampling and top-three ranking, without log fitting."""
    if alpha.ndim != 1 or alpha.shape != beta.shape or alpha.size < 3:
        raise ValueError("BTS requires aligned vectors with at least three actions")
    if not np.isfinite(alpha).all() or not np.isfinite(beta).all() or np.any(alpha <= 0) or np.any(beta <= 0):
        raise ValueError("Beta parameters must be finite and positive")
    if isinstance(simulations, bool) or simulations <= 0:
        raise ValueError("BTS requires positive Monte Carlo simulations")
    rng = np.random.RandomState(seed)
    counts = np.zeros((alpha.size, 3), dtype=np.int64)
    for start in range(0, simulations, 10000):
        size = min(10000, simulations - start)
        samples = rng.beta(alpha, beta, size=(size, alpha.size))
        chosen = np.argsort(samples, axis=1)[:, ::-1][:, :3]
        for slot in range(3):
            counts[:, slot] += np.bincount(chosen[:, slot], minlength=alpha.size)
    return np.asarray(counts / simulations, dtype=float)


def validate_log(frame: pd.DataFrame, actions: int) -> None:
    required = {"timestamp", "item_id", "position", "click", "propensity_score"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("OBD requires nonempty records with all logging fields")
    if (
        not frame["propensity_score"].between(0, 1, inclusive="right").all()
        or not frame["click"].isin([0, 1]).all()
        or not frame["position"].isin([1, 2, 3]).all()
        or not frame["item_id"].isin(range(actions)).all()
        or pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).isna().any()
    ):
        raise ValueError("OBD reward, propensity, action, position or timestamp contract failed")


def on_policy_interval(rewards: FloatArray) -> Interval:
    """Wilson uncertainty remains informative for a small sample without clicks."""
    if rewards.size == 0 or not np.isin(rewards, [0, 1]).all():
        raise ValueError("On-policy estimate needs nonempty binary rewards")
    n, rate, z = rewards.size, float(rewards.mean()), 1.959963984540054
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return {"mean": rate, "low": float(max(0, center - half)), "high": float(min(1, center + half))}


def difference_interval(
    rewards: FloatArray,
    actions: NDArray[np.int64],
    propensity: FloatArray,
    target: FloatArray,
    model: FloatArray,
    factual: FloatArray,
    seed: int,
    draws: int,
) -> dict[str, Interval]:
    """Independent row bootstrap for OPE minus empirical on-policy reward."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, rewards.size, size=(draws, rewards.size))
    on_policy = factual[rng.integers(0, factual.size, size=(draws, factual.size))].mean(axis=1)
    weights = target[np.arange(rewards.size), actions] / propensity
    direct = (target * model).sum(axis=1)
    residual = rewards - model[np.arange(rewards.size), actions]
    samples = {
        "IPS": (weights * rewards)[indices].mean(axis=1),
        "SNIPS": (weights * rewards)[indices].sum(axis=1) / weights[indices].sum(axis=1),
        "DM": direct[indices].mean(axis=1),
        "DR": (direct + weights * residual)[indices].mean(axis=1),
    }
    points = estimate_intervals(rewards, actions, propensity, target, model, seed=seed, draws=draws)
    return {
        name: {
            "mean": points[name]["mean"] - float(factual.mean()),
            "low": float(np.quantile(sample - on_policy, 0.025)),
            "high": float(np.quantile(sample - on_policy, 0.975)),
        }
        for name, sample in samples.items()
    }


def campaign_evidence(
    campaign: str,
    random: pd.DataFrame,
    bts: pd.DataFrame,
    probabilities: FloatArray,
    draws: int = 1000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n_actions = probabilities.shape[0]
    validate_log(random, n_actions)
    validate_log(bts, n_actions)
    if not np.allclose(random["propensity_score"].to_numpy(), 1 / n_actions):
        raise ValueError("Uniform logger does not match the declared action universe")
    random = random.sort_values("timestamp", kind="stable")
    bts = bts.sort_values("timestamp", kind="stable")
    train, test = random.iloc[: len(random) // 2], random.iloc[len(random) // 2 :]
    cutoff = str(test.iloc[0]["timestamp"])
    reference = bts[bts["timestamp"] >= cutoff]
    rows: list[dict[str, Any]] = []
    for position in (1, 2, 3):
        fit, heldout = train[train.position == position], test[test.position == position]
        factual = reference[reference.position == position]
        if heldout.empty or factual.empty:
            raise ValueError("Each held-out campaign position requires random and BTS records")
        actions = np.asarray(heldout.item_id.to_numpy(), dtype=np.int64)
        rewards = np.asarray(heldout.click.to_numpy(), dtype=float)
        propensity = np.asarray(heldout.propensity_score.to_numpy(), dtype=float)
        factual_rewards = np.asarray(factual.click.to_numpy(), dtype=float)
        summaries = fit.groupby("item_id").click.agg(["sum", "count"]).reindex(range(n_actions), fill_value=0)
        prediction = np.asarray(
            (summaries["sum"] + 20 * float(train.click.mean())) / (summaries["count"] + 20), dtype=float
        )
        model = np.tile(prediction, (len(heldout), 1))
        for policy, distribution in (
            ("Uniform sanity check", np.full(n_actions, 1 / n_actions)),
            ("Official prior BTS approximation", probabilities[:, position - 1]),
        ):
            target = np.tile(distribution, (len(heldout), 1))
            estimates = estimate_intervals(
                rewards, actions, propensity, target, model, seed=731 + position, draws=draws
            )
            differences = difference_interval(
                rewards, actions, propensity, target, model, factual_rewards, 941 + position, draws
            )
            weights = target[np.arange(len(heldout)), actions] / propensity
            for name, point in estimates.items():
                rows.append(
                    {
                        "campaign": campaign,
                        "policy": policy,
                        "position": position,
                        "population": f"OBD {campaign}; random held-out slot {position}; cutoff {cutoff}",
                        "sample_size": len(heldout),
                        "estimator": name,
                        "estimate": point,
                        "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
                        "observed_logging": on_policy_interval(rewards),
                        "bts_reference": on_policy_interval(factual_rewards),
                        "bts_reference_n": len(factual),
                        "difference_from_bts": differences[name],
                        "comparison_identifies_exact_logged_policy": False,
                    }
                )
    groups = bts.groupby(["item_id", "position"]).propensity_score
    summary = {
        "campaign": campaign,
        "actions": n_actions,
        "logging_rows": len(random),
        "bts_rows": len(bts),
        "training_rows": len(train),
        "evaluation_rows": len(test),
        "bts_reference_rows": len(reference),
        "cutoff": cutoff,
        "on_policy_bts": on_policy_interval(np.asarray(reference.click, dtype=float)),
        "logged_propensity_diagnostic": {
            "max_distinct_values_within_item_position": int(groups.nunique().max()),
            "max_range_within_item_position": float((groups.max() - groups.min()).max()),
            "interpretation": "Logged BTS propensities vary within item and position. The frozen upstream prior is a reproducible benchmark approximation, not an exact per-impression policy reconstruction.",
        },
    }
    return rows, summary


def run_obd(root: Path) -> dict[str, Any]:
    cache, output = root / ".cache" / "obd", root / "data" / "obd"
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    sources: dict[str, tuple[Path, str]] = {}
    sources["prior_bts.yaml"] = fetch(cache, "prior_bts.yaml", "obp/policy/conf/prior_bts.yaml")
    sources["LICENSE"] = fetch(cache, "LICENSE", "LICENSE")
    priors = parse_priors(sources["prior_bts.yaml"][0].read_bytes())
    rows: list[dict[str, Any]] = []
    campaigns: list[dict[str, Any]] = []
    monte_carlo: list[dict[str, Any]] = []
    for campaign in CAMPAIGNS:
        frames: dict[str, pd.DataFrame] = {}
        for policy in ("random", "bts"):
            name = f"{policy}-{campaign}.csv"
            sources[name] = fetch(cache, name, f"obd/{policy}/{campaign}/{campaign}.csv")
            frame = pd.read_csv(sources[name][0])
            validate_log(frame, priors[campaign][0].size)
            frame[["timestamp", "item_id", "position", "click", "propensity_score"]].to_parquet(
                output / f"{policy}-{campaign}.parquet", index=False
            )
            frames[policy] = frame
        sims, seed = 200000, 12345
        probability = bts_action_distribution(*priors[campaign], simulations=sims, seed=seed)
        other = bts_action_distribution(*priors[campaign], simulations=sims, seed=seed + 1)
        monte_carlo.append(
            {
                "campaign": campaign,
                "simulations": sims,
                "seed": seed,
                "maximum_probability_se": float(np.sqrt(probability * (1 - probability) / sims).max()),
                "maximum_independent_seed_difference": float(np.abs(probability - other).max()),
                "probabilities": probability.tolist(),
            }
        )
        measured, summary = campaign_evidence(campaign, frames["random"], frames["bts"], probability)
        rows.extend(measured)
        campaigns.append(summary)
    (output / "LICENSE").write_bytes(sources["LICENSE"][0].read_bytes())
    provenance = {
        "source_commit": COMMIT,
        "source": "https://github.com/st-tech/zr-obp",
        "sample_repository_license": "Apache-2.0",
        "dataset_paper_license": "CC BY 4.0",
        "license_note": "Repository Apache notice retained; dataset paper also specifies CC BY 4.0. Both provenance statements and source attribution are retained.",
        "citation": "Saito, Aihara, Matsutani, and Narita (2020), Open Bandit Dataset and Pipeline",
        "files": [
            {"name": name, "url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, (path, url) in sources.items()
        ],
        "transformation": "All six campaign/policy files; selected logged-action columns exported. No target probabilities inferred from action frequencies.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (output / "bts-probabilities.json").write_text(json.dumps(monte_carlo, indent=2) + "\n", encoding="utf-8")
    (output / "PROVENANCE.md").write_text(
        "# Open Bandit sample\n\n"
        f"Source: https://github.com/st-tech/zr-obp. Pinned commit `{COMMIT}`. "
        "All six random/BTS files for all, men and women are preserved as selected-column Parquet exports. "
        "The source Apache notice is retained; the dataset paper also specifies CC BY 4.0. "
        "Citation: Saito, Aihara, Matsutani, and Narita (2020), Open Bandit Dataset and Pipeline. "
        "Exact file hashes and URLs are in provenance.json. "
        "Positions remain source one-based slots; propensities are per-position marginals, not joint-slate probabilities. "
        "Published beta priors and the official BernoulliTS rule define a Monte Carlo target; "
        "no target probabilities are inferred from action frequencies. "
        "Random-policy logs train the reward model before a shared timestamp cutoff and evaluate OPE after it. "
        "BTS logs after that same cutoff provide empirical on-policy reference rewards. "
        "Varying logged propensities prevent claiming this frozen target exactly matches every BTS logging state. "
        "See ../../docs/ope-benchmark.md for uncertainty and policy-mismatch limitations.\n",
        encoding="utf-8",
    )
    return {
        "status": "measured_six_file_bts_benchmark",
        "source_commit": COMMIT,
        "campaign": "all / men / women",
        "campaigns": campaigns,
        "logging_rows": sum(c["logging_rows"] for c in campaigns),
        "training_rows": sum(c["training_rows"] for c in campaigns),
        "evaluation_rows": sum(c["evaluation_rows"] for c in campaigns),
        "actions": {c["campaign"]: c["actions"] for c in campaigns},
        "target": "Official production-prior BernoulliTS Monte Carlo approximation and uniform sanity check",
        "detail": "All six random/BTS samples. Official campaign beta priors define the BTS target through beta draws ranked into three positions. Random logs after a shared cutoff provide OPE; BTS logs after the same cutoff provide an empirical on-policy benchmark with Wilson intervals. Logged BTS propensities vary within item/slot, so the frozen-prior approximation is not proven identical to the deployed per-impression policy. Differences include target mismatch and sampling error, not pure estimator error. Row bootstrap is conditional on the fitted reward model and Monte Carlo target, ignores repeated-reader dependence, and is not joint-slate inference.",
        "reference_method": BASE + "examples/obd/evaluate_off_policy_estimators.py",
        "policy_source": BASE + "obp/policy/contextfree.py",
        "prior_sha256": PRIOR_SHA,
        "monte_carlo": monte_carlo,
        "rows": rows,
    }
