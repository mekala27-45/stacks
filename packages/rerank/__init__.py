"""Business rules, diversity and calibration for the demonstration shelf."""

from collections import Counter
from math import log
from typing import Any

Item = dict[str, Any]


def tag_similarity(left: Item, right: Item) -> float:
    a, b = set(left.get("tags", [])), set(right.get("tags", []))
    if not a or not b:
        return float(left.get("genre") == right.get("genre"))
    return len(a & b) / len(a | b)


def diversity(items: list[Item]) -> float:
    pairs = [1.0 - tag_similarity(a, b) for i, a in enumerate(items) for b in items[i + 1 :]]
    return sum(pairs) / len(pairs) if pairs else 0.0


def distribution(items: list[Item]) -> dict[str, float]:
    counts = Counter(str(item.get("genre", "other")) for item in items)
    total = sum(counts.values())
    return {key: count / total for key, count in counts.items()} if total else {}


def calibration_divergence(items: list[Item], history: list[Item]) -> float:
    """Jensen-Shannon divergence between shelf and history genre distributions."""
    p, q = distribution(items), distribution(history)
    if not p or not q:
        return 0.0
    value = 0.0
    for key in p.keys() | q.keys():
        a, b = p.get(key, 0.0), q.get(key, 0.0)
        midpoint = (a + b) / 2
        if a:
            value += 0.5 * a * log(a / midpoint)
        if b:
            value += 0.5 * b * log(b / midpoint)
    return value


def apply_rules(candidates: list[Item], seen: set[str], max_per_author: int = 2) -> list[Item]:
    counts: Counter[str] = Counter()
    result = []
    unique: set[str] = set()
    for item in candidates:
        item_id, author = str(item["id"]), str(item["author"]).split(",")[0].strip().casefold()
        if (
            item_id in seen
            or item_id in unique
            or item.get("available", True) is False
            or counts[author] >= max_per_author
        ):
            continue
        result.append(item)
        unique.add(item_id)
        counts[author] += 1
    return result


def mmr(candidates: list[Item], k: int, tradeoff: float = 0.25) -> list[Item]:
    if not 0 <= tradeoff <= 1:
        raise ValueError("MMR tradeoff must be in [0, 1]")
    remaining = list(candidates)
    selected: list[Item] = []
    while remaining and len(selected) < k:
        chosen = max(
            remaining,
            key=lambda item: (
                (1 - tradeoff) * float(item["score"])
                - tradeoff * max((tag_similarity(item, old) for old in selected), default=0)
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    # MMR optimizes an objective, not diversity itself; guard the stated contract.
    baseline = candidates[:k]
    return selected if diversity(selected) + 1e-12 >= diversity(baseline) else baseline


def calibrate(
    selected: list[Item], candidates: list[Item], history: list[Item], weight: float = 0.2
) -> list[Item]:
    if not 0 <= weight <= 1:
        raise ValueError("Calibration weight must be in [0, 1]")
    result = list(selected)
    if not history or not result or weight == 0:
        return result

    def objective(items: list[Item]) -> float:
        relevance = sum(float(item["score"]) for item in items) / len(items)
        return weight * calibration_divergence(items, history) - (1 - weight) * relevance

    # Every accepted swap reduces divergence and the relevance-aware objective.
    for _ in range(len(result)):
        current_divergence, current_objective = calibration_divergence(result, history), objective(result)
        best: list[Item] | None = None
        best_objective = current_objective
        included = {str(item["id"]) for item in result}
        for replacement in candidates:
            if str(replacement["id"]) in included:
                continue
            for position in range(len(result)):
                proposal = list(result)
                proposal[position] = replacement
                value = objective(proposal)
                if (
                    value < best_objective - 1e-12
                    and calibration_divergence(proposal, history) < current_divergence - 1e-12
                ):
                    best, best_objective = proposal, value
        if best is None:
            break
        result = best
    return result


def rerank(
    candidates: list[Item],
    seen: set[str],
    history: list[Item],
    k: int,
    diversity_weight: float = 0.25,
    calibration_weight: float = 0.2,
) -> tuple[list[Item], dict[str, Any], list[Item]]:
    ruled = apply_rules(candidates, seen)
    diverse = mmr(ruled, k, diversity_weight)
    calibrated = calibrate(diverse, ruled[:60], history, calibration_weight)
    trace = {
        "rules": {"seen_removed": len(seen), "author_cap": 2, "eligible_count": len(ruled)},
        "diversity": {
            "weight": diversity_weight,
            "before": diversity(ruled[:k]),
            "after": diversity(diverse),
        },
        "calibration": {
            "weight": calibration_weight,
            "before": calibration_divergence(diverse, history),
            "after": calibration_divergence(calibrated, history),
        },
        "before_ids": [item["id"] for item in ruled[:k]],
        "diverse_ids": [item["id"] for item in diverse],
        "calibrated_ids": [item["id"] for item in calibrated],
    }
    return calibrated, trace, ruled
