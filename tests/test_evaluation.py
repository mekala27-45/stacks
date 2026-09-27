import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from packages.evaluation.metrics import (
    benjamini_hochberg,
    bootstrap,
    intra_list_diversity,
    rank_unseen,
    ranking_metrics,
    wilson_interval,
)


def test_sparse_pair_diversity_is_invariant_to_recommendation_order():
    directed = csr_matrix([[0, 1, 0], [0, 0, 0.6], [0.2, 0, 0]], dtype=np.float32)
    expected = 1 - (0.5 + 0.3 + 0.1) / 3
    assert intra_list_diversity(directed, np.array([0, 1, 2])) == pytest.approx(expected)
    assert intra_list_diversity(directed, np.array([2, 1, 0])) == pytest.approx(expected)
    assert intra_list_diversity(directed, np.array([0])) == 0


from packages.ope import estimate, estimate_intervals
from packages.retrieval import train_models
from packages.sim import Simulator


def test_full_ranking_removes_every_training_item_and_breaks_ties_by_id():
    ranked = rank_unseen(np.array([2.0, 1.0, 2.0, 4.0]), {3, 1})
    assert ranked.tolist() == [0, 2]


def test_perfect_ranking_and_monotonicity():
    assert ranking_metrics(np.arange(20), {0, 1, 2})["ndcg"] == 1.0
    scores = [ranking_metrics(np.roll(np.arange(20), index), {0})["ndcg"] for index in range(10)]
    assert all(left > right for left, right in pairwise(scores))
    with pytest.raises(ValueError):
        ranking_metrics(np.arange(20), set())


def test_random_ranker_expected_recall_and_sampled_inflation():
    rng = np.random.default_rng(222)
    recalls, sampled = [], []
    for _ in range(3000):
        scores = rng.random(1000)
        full = rank_unseen(scores, set())
        candidates = np.r_[0, rng.choice(np.arange(1, 1000), size=100, replace=False)]
        short = rank_unseen(scores, set(), candidates)
        recalls.append(ranking_metrics(full, {0})["recall"])
        sampled.append(ranking_metrics(short, {0})["recall"])
    assert abs(np.mean(recalls) - 10 / 1000) < 0.008
    assert abs(np.mean(sampled) - 10 / 101) < 0.025
    assert np.mean(sampled) > 5 * np.mean(recalls)


def test_bh_known_family_and_interval_guards():
    np.testing.assert_allclose(
        benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.2])), [0.04, 0.0533333333, 0.0533333333, 0.2]
    )
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([2.0]))
    with pytest.raises(ValueError):
        bootstrap(np.array([]))
    assert bootstrap(np.ones(20)) == {"mean": 1.0, "low": 1.0, "high": 1.0}
    assert wilson_interval(0, 200)["high"] > 0.018
    assert wilson_interval(200, 200)["low"] < 0.982


def test_als_cosine_are_trained_and_cold_reader_falls_back():
    matrix = csr_matrix(np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
    bundle = train_models(matrix, factors=3, iterations=4)
    assert bundle.cosine[0, 1] == pytest.approx(1)
    assert np.isfinite(bundle.user_factors).all()
    assert np.linalg.norm(bundle.item_factors) > 0
    assert not np.allclose(bundle.score(0)["als"], bundle.score(2)["als"])
    np.testing.assert_array_equal(bundle.score(3)["als"], bundle.popularity)


def test_ope_known_exact_on_policy_case_and_invalid_propensity():
    rewards = np.array([0.0, 1.0, 0.0, 1.0])
    actions = np.array([0, 1, 0, 1])
    propensity = np.full(4, 0.5)
    target = np.full((4, 2), 0.5)
    model = np.tile([0.0, 1.0], (4, 1))
    assert estimate(rewards, actions, propensity, target, model) == {
        "IPS": 0.5,
        "SNIPS": 0.5,
        "DM": 0.5,
        "DR": 0.5,
    }
    with pytest.raises(ValueError, match="propensities"):
        estimate(rewards, actions, np.zeros(4), target, model)
    with pytest.raises(ValueError, match="distribution"):
        estimate(rewards, actions, propensity, np.zeros((4, 2)), model)
    with pytest.raises(ValueError, match="Nonempty"):
        estimate(np.array([]), np.array([], dtype=int), np.array([]), np.empty((0, 2)), np.empty((0, 2)))


def test_ope_simulation_bias_and_misspecification():
    simulator = Simulator()
    measured = {name: [] for name in ("IPS", "SNIPS", "DM", "DR")}
    misspecified = {name: [] for name in measured}
    for seed in range(200):
        for destination, wrong in ((measured, False), (misspecified, True)):
            for name, result in estimate(*simulator.sample(1000, seed, wrong)).items():
                destination[name].append(result)
    for values in measured.values():
        assert abs(np.mean(values) - simulator.value) < 0.015
    assert abs(np.mean(misspecified["DM"]) - simulator.value) > 0.2
    assert abs(np.mean(misspecified["DR"]) - simulator.value) < 0.015
    intervals = estimate_intervals(*simulator.sample(1000, 17), seed=9)
    assert all(value["low"] <= value["mean"] <= value["high"] for value in intervals.values())


def test_committed_protocol_and_manifest_are_honest_when_available():
    root = Path(__file__).resolve().parents[1]
    protocol = (root / "docs" / "evaluation.md").read_text(encoding="utf-8")
    assert "no interaction timestamps" in protocol
    path = root / "results" / "manifest.json"
    if not path.exists():
        pytest.skip("Evidence pipeline has not run")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["ope"]["simulator"]["seeds"] == 200
    model_count = len(manifest["metrics"])
    assert len(manifest["comparisons"]) == model_count * (model_count - 1) // 2
    assert (
        manifest["dataset"]["train_interactions"] + manifest["dataset"]["test_interactions"]
        == manifest["dataset"]["interactions"]
    )
    assert all(0 <= row["ndcg"]["mean"] <= 1 for row in manifest["metrics"])


def test_source_order_split_features_ignore_changed_holdout_ratings():
    import pandas as pd

    from packages.evaluation.split import matrices

    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 1, 2],
            "book_id": [1, 2, 2, 3, 3, 4],
            "rating": [5, 2, 4, 5, 5, 4],
            "source_row": np.arange(6),
        }
    )
    split = (frame.source_row < 4).to_numpy()
    matrix, seen, truth = matrices(frame, split, {1: 0, 2: 1})
    changed = frame.copy()
    changed.loc[~split, "rating"] = 1
    other, other_seen, other_truth = matrices(changed, split, {1: 0, 2: 1})
    np.testing.assert_array_equal(matrix.toarray(), other.toarray())
    assert seen == other_seen
    assert truth != other_truth
    assert 1 in seen[0]  # A disliked training item must still be filtered.


def test_persisted_training_popularity_matches_200_independent_recomputations():
    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    path = root / "results" / "model-artifacts.npz"
    if not path.exists():
        pytest.skip("Evidence pipeline has not run")
    ratings = pd.read_parquet(root / "data" / "goodbooks" / "ratings.parquet")
    manifest = json.loads((root / "results" / "manifest.json").read_text(encoding="utf-8"))
    training = ratings[(ratings.source_row < manifest["protocol"]["cutoff_row"]) & (ratings.rating >= 4)]
    stored = np.load(path)["popularity"]
    counts = training.book_id.value_counts()
    catalog_size = manifest["dataset"]["catalog_size"]
    for item in np.random.default_rng(89).choice(np.arange(1, catalog_size + 1), size=200, replace=False):
        assert stored[item - 1] == counts.get(item, 0)


def test_real_obd_uniform_policy_sanity_when_present():
    root = Path(__file__).resolve().parents[1]
    path = root / "results" / "manifest.json"
    if not path.exists():
        pytest.skip("Evidence pipeline has not run")
    result = json.loads(path.read_text(encoding="utf-8"))["ope"]["open_bandit"]
    if result["status"] not in {"measured_alternative_target", "measured_six_file_bts_benchmark"}:
        pytest.skip(result["detail"])
    for row in result["rows"]:
        if row["policy"] == "Uniform sanity check" and row["estimator"] in {"IPS", "SNIPS"}:
            assert row["estimate"]["mean"] == pytest.approx(row["observed_logging"]["mean"])


def test_simulator_bootstrap_coverage_against_binomial_band_when_present():
    from scipy.stats import binom

    root = Path(__file__).resolve().parents[1]
    path = root / "results" / "manifest.json"
    if not path.exists():
        pytest.skip("Evidence pipeline has not run")
    result = json.loads(path.read_text(encoding="utf-8"))["ope"]["simulator"]
    low, high = np.array(binom.interval(0.95, result["seeds"], 0.95)) / result["seeds"]
    for row in result["rows"]:
        if row["reward_model"] == "oracle":
            assert low <= row["coverage"] <= high


def test_exported_reader_history_preserves_source_order_when_present():
    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    path = root / "web" / "public" / "data" / "readers.json"
    if not path.exists():
        pytest.skip("Evidence pipeline has not run")
    ratings = pd.read_parquet(root / "data" / "goodbooks" / "ratings.parquet")
    manifest = json.loads((root / "results" / "manifest.json").read_text(encoding="utf-8"))
    training = ratings[(ratings.source_row < manifest["protocol"]["cutoff_row"]) & (ratings.rating >= 4)]
    for reader in json.loads(path.read_text(encoding="utf-8")):
        expected = (
            training[training.user_id == reader["id"]].sort_values("source_row").book_id.tail(12).tolist()
        )
        assert reader["history"] == expected
