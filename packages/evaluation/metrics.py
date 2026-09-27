from collections.abc import Iterable

import numpy as np


def rank_unseen(scores: np.ndarray, seen: Iterable[int], candidates: np.ndarray | None = None) -> np.ndarray:
    """Rank eligible zero-based items, with stable ID tie breaks."""
    ids = np.arange(scores.size) if candidates is None else np.asarray(candidates)
    ids = ids[~np.isin(ids, np.asarray(list(seen), dtype=int))]
    return ids[np.lexsort((ids, -scores[ids]))]


def ranking_metrics(ranked: np.ndarray, positives: Iterable[int], k: int = 10) -> dict[str, float]:
    relevant = set(positives)
    if not relevant:
        raise ValueError("Metrics require at least one relevant holdout item")
    hits = np.array([item in relevant for item in ranked[:k]], dtype=float)
    discounts = 1 / np.log2(np.arange(hits.size) + 2)
    ideal = (1 / np.log2(np.arange(min(k, len(relevant))) + 2)).sum()
    return {
        "ndcg": float((hits * discounts).sum() / ideal),
        "recall": float(hits.sum() / len(relevant)),
        "hit_rate": float(hits.any()),
        "recall200": float(sum(item in relevant for item in ranked[:200]) / len(relevant)),
    }


def bootstrap(values: np.ndarray, draws: int = 1000, seed: int = 42) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Intervals require finite nonempty observations")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "low": float(low), "high": float(high)}


def wilson_interval(successes: int, trials: int) -> dict[str, float]:
    """95% Wilson binomial interval, including meaningful boundary uncertainty."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Binomial counts require 0 <= successes <= positive trials")
    rate = successes / trials
    z = 1.959963984540054
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    radius = z * np.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return {"mean": rate, "low": max(0.0, float(center - radius)), "high": min(1.0, float(center + radius))}


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    if np.any(~np.isfinite(pvalues)) or np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError("p-values must be finite probabilities")
    if not pvalues.size:
        return pvalues.copy()
    order = np.argsort(pvalues)
    adjusted = np.minimum.accumulate((pvalues[order] * pvalues.size / np.arange(1, pvalues.size + 1))[::-1])[
        ::-1
    ]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


def paired_comparisons(per_user: dict[str, np.ndarray]) -> list[dict[str, object]]:
    from itertools import combinations

    result: list[dict[str, object]] = []
    for left, right in combinations(per_user, 2):
        difference = per_user[left] - per_user[right]
        signs = np.random.default_rng(91).choice([-1, 1], size=(4999, difference.size))
        null = (signs * difference).mean(axis=1)
        pvalue = (1 + np.count_nonzero(np.abs(null) >= abs(difference.mean()))) / 5000
        result.append({"left": left, "right": right, "difference": bootstrap(difference), "p": float(pvalue)})
    corrected = benjamini_hochberg(np.array([row["p"] for row in result]))
    for row, qvalue in zip(result, corrected, strict=True):
        row["q"] = float(qvalue)
        row["significant"] = bool(qvalue < 0.05)
    return result
