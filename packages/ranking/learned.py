from typing import Any

import lightgbm as lgb
import numpy as np

from packages.evaluation.metrics import rank_unseen
from packages.retrieval.models import ModelBundle, _normalize

FEATURE_NAMES = [
    "cosine_normalized",
    "als_normalized",
    "popularity_normalized",
    "blend_score",
    "log_popularity",
    "log_history_count",
    "cosine_raw",
    "als_raw",
    "training_cold_item",
]


def ranker_features(scores: dict[str, np.ndarray], history_count: int) -> np.ndarray:
    popularity = scores["popularity"]
    return np.column_stack(
        [
            _normalize(scores["item_cosine"]),
            _normalize(scores["als"]),
            _normalize(popularity),
            scores["blend"],
            np.log1p(popularity),
            np.full(popularity.size, np.log1p(history_count)),
            scores["item_cosine"],
            scores["als"],
            (popularity == 0).astype(float),
        ]
    ).astype(np.float32)


def train_ranker(
    bundle: ModelBundle, seen: dict[int, set[int]], truth: dict[int, set[int]], max_users: int = 3000
) -> tuple[Any, dict[str, object]]:
    rng = np.random.default_rng(861)
    eligible = np.array([uid for uid in truth if bundle.matrix[uid].nnz >= 5])
    users = np.sort(rng.choice(eligible, size=min(max_users, len(eligible)), replace=False))
    features, labels, groups = [], [], []
    positives = 0
    for user in users:
        scores = bundle.score(int(user))
        union = np.unique(
            np.concatenate(
                [
                    rank_unseen(scores[name], seen.get(int(user), set()))[:200]
                    for name in ("popularity", "item_cosine", "als", "blend")
                ]
            )
        )
        target = np.isin(union, list(truth[int(user)])).astype(np.int32)
        if target.sum() == 0 or target.sum() == target.size:
            continue
        features.append(ranker_features(scores, bundle.matrix[user].nnz)[union])
        labels.append(target)
        groups.append(len(union))
        positives += int(target.sum())
    if len(groups) < 30:
        raise ValueError("Too few nontrivial candidate groups to fit LambdaMART")
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=70,
        learning_rate=0.055,
        num_leaves=15,
        max_depth=5,
        min_child_samples=100,
        reg_lambda=1.0,
        random_state=861,
        n_jobs=2,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(np.concatenate(features), np.concatenate(labels), group=groups, feature_name=FEATURE_NAMES)
    return model.booster_, {
        "status": "trained",
        "objective": "lambdarank",
        "training_groups": len(groups),
        "candidate_rows": int(sum(groups)),
        "relevant_candidates": positives,
        "trees": 70,
        "features": FEATURE_NAMES,
        "candidate_sources": ["popularity", "item_cosine", "als", "blend"],
        "candidate_k_per_source": 200,
        "detail": "Candidate union labels come from an earlier held-out source-order window; every feature uses only its prefix training interactions. Snapshot metadata is not a ranker feature. Hyperparameters fixed before final holdout evaluation. Full-catalog inference, including items outside the candidate union, is published without a selection gate.",
    }
