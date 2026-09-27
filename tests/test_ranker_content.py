import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from packages.content import build_vectors
from packages.ranking.learned import FEATURE_NAMES, ranker_features, train_ranker
from packages.retrieval.models import load_bundle, sparse_item_cosine, train_models


def test_content_retrieves_an_unobserved_book_from_title_and_tags():
    catalog = [
        {"title": "Wizard academy", "author": "A Writer", "tags": ["fantasy", "magic"], "genre": "Fantasy"},
        {
            "title": "Wizard adventures",
            "author": "A Writer",
            "tags": ["fantasy", "magic"],
            "genre": "Fantasy",
        },
        {
            "title": "Economic history",
            "author": "B Author",
            "tags": ["economics", "history"],
            "genre": "Nonfiction",
        },
    ]
    vectors = build_vectors(catalog)
    model = train_models(
        csr_matrix([[1, 0, 0], [1, 0, 0]], dtype=np.float32), factors=2, iterations=2, content_vectors=vectors
    )
    scores = model.score(0)
    assert scores["item_cosine"][1] == 0
    assert scores["content"][1] > scores["content"][2]
    np.testing.assert_array_equal(model.score_history(np.array([], dtype=int))["content"], model.popularity)


def test_sparse_cosine_pruning_is_part_of_the_model():
    matrix = csr_matrix([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.float32)
    cosine = sparse_item_cosine(matrix, neighbors=1)
    assert np.diff(cosine.indptr).max() <= 1
    assert cosine[0, 1] == pytest.approx(1.0)
    assert np.diag(cosine.toarray()).sum() == 0


def test_lambdamart_trains_candidate_groups_and_scores_unseen_catalog():
    rng = np.random.default_rng(71)
    dense = np.zeros((40, 20), dtype=np.float32)
    seen, truth = {}, {}
    for uid in range(40):
        history = rng.choice(20, 6, replace=False)
        dense[uid, history] = 1
        seen[uid] = set(history.tolist())
        truth[uid] = set(rng.choice(np.setdiff1d(np.arange(20), history), 2, replace=False).tolist())
    bundle = train_models(csr_matrix(dense), factors=3, iterations=2)
    before = ranker_features(bundle.score(0), 6)
    ranker, metadata = train_ranker(bundle, seen, truth, max_users=40)
    assert metadata["training_groups"] == 40
    assert metadata["features"] == FEATURE_NAMES
    bundle.ranker = ranker
    scores = bundle.score(0)
    assert scores["lambdamart"].shape == (20,)
    assert np.isfinite(scores["lambdamart"]).all()
    np.testing.assert_array_equal(before, ranker_features(scores, 6))


def test_ranker_refuses_empty_training_evidence():
    bundle = train_models(csr_matrix(np.eye(4, dtype=np.float32)), factors=2, iterations=1)
    with pytest.raises(ValueError, match="Too few"):
        train_ranker(bundle, {0: {0}}, {0: {1}})


def test_shared_inference_roundtrip_and_exported_profile_parity():
    root = Path(__file__).resolve().parents[1]
    if not (root / "results" / "train-positive.npz").exists():
        pytest.skip("Full model artifacts have not been generated")
    bundle = load_bundle(root / "results")
    assert bundle.user_ids is not None
    profiles = json.loads(
        (root / "web" / "public" / "data" / "edge-profiles.json").read_text(encoding="utf-8")
    )
    for profile in profiles[:3]:
        index = int(np.flatnonzero(bundle.user_ids == profile["id"])[0])
        initial = bundle.score(index)
        from_profile = bundle.score_history(
            np.array(profile["positive_indices"]), np.array(profile["user_factor"], dtype=np.float32)
        )
        for model in ("popularity", "item_cosine", "als", "blend", "content", "lambdamart"):
            np.testing.assert_allclose(initial[model], from_profile[model], atol=1e-7)
