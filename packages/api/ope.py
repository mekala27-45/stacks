"""Identified final-slot OPE with an immutable click horizon and explicit support."""

import math
from datetime import UTC, datetime
from typing import Any

from packages.api.database import Feedback, Impression

TARGET_POLICY = "final-slot-80-best-20-uniform-v1"
REWARD_HORIZON_SECONDS = 60


def as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def evaluate_logs(
    impressions: list[Impression], feedback: list[Feedback], now: datetime | None = None
) -> dict[str, Any]:
    clock = as_utc(now or datetime.now(UTC))
    pending = invalid = late = 0
    observed: list[tuple[float, float, float, str]] = []
    excluded_test_rows = 0
    for row in impressions:
        if row.policy != "uniform":
            continue
        if row.trace.get("traffic_kind") == "load_test":
            excluded_test_rows += 1
            continue
        config = row.trace.get("exploration", {})
        pool, target = config.get("candidate_pool", []), config.get("target_probabilities", [])
        if (
            config.get("target_policy") != TARGET_POLICY
            or config.get("reward_horizon_seconds") != REWARD_HORIZON_SECONDS
        ):
            invalid += 1
            continue
        if (clock - as_utc(row.impression_at)).total_seconds() < REWARD_HORIZON_SECONDS:
            pending += 1
            continue
        if (
            not pool
            or len(pool) > 20
            or len(pool) != len(target)
            or row.item_id not in pool
            or abs(row.propensity - 1 / len(pool)) > 1e-9
            or any(
                not isinstance(p, (int, float))
                or not math.isfinite(p)
                or abs(p - (0.2 / len(pool) + (0.8 if index == 0 else 0))) > 1e-9
                for index, p in enumerate(target)
            )
        ):
            invalid += 1
            continue
        reward = 0.0
        for event in feedback:
            if event.impression_id == row.id and event.event == "click":
                delay = (as_utc(event.feedback_at) - as_utc(row.impression_at)).total_seconds()
                if delay > REWARD_HORIZON_SECONDS:
                    late += 1
                elif delay >= 0:
                    reward = 1.0
        observed.append(
            (
                target[pool.index(row.item_id)] / row.propensity,
                reward,
                max(target) / row.propensity,
                row.session_hash,
            )
        )
    common: dict[str, Any] = {
        "schema_version": "1.1",
        "target_policy": TARGET_POLICY,
        "scope": "Conditional final-position action value within each recorded candidate pool",
        "horizon_seconds": REWARD_HORIZON_SECONDS,
        "matured": len(observed),
        "pending": pending,
        "invalid_excluded": invalid,
        "load_test_excluded": excluded_test_rows,
        "late_feedback_ignored": late,
        "session_count": len({row[3] for row in observed}),
        "full_slate_identified": False,
    }
    if not observed:
        return {
            **common,
            "status": "awaiting_mature_feedback",
            "estimates": None,
            "detail": "No randomized impression has completed its reward horizon.",
        }
    n = len(observed)
    weighted = sum(weight * reward for weight, reward, _, _ in observed) / n
    denominator = sum(row[0] for row in observed) / n
    # Predetermined support bound remains valid for adaptively chosen future pools.
    bound = 16.2
    radius = bound * math.sqrt(math.log(40) / (2 * n))
    ratio_radius = bound * math.sqrt(math.log(80) / (2 * n))
    dr = 0.05 + sum(weight * (reward - 0.05) for weight, reward, _, _ in observed) / n
    return {
        **common,
        "status": "measured",
        "estimates": {
            "IPS": {"mean": weighted, "low": weighted - radius, "high": weighted + radius},
            "SNIPS": {
                "mean": weighted / denominator,
                "low": max(0, weighted - ratio_radius) / max(0.2, denominator + ratio_radius),
                "high": min(1, (weighted + ratio_radius) / max(0.2, denominator - ratio_radius)),
            },
            "DM": {"mean": 0.05, "low": 0.05, "high": 0.05},
            "DR": {"mean": dr, "low": dr - radius, "high": dr + radius},
        },
        "effective_sample_size": (n * denominator) ** 2 / sum(row[0] ** 2 for row in observed),
        "logging_click_rate": sum(row[1] for row in observed) / n,
        "maximum_importance_weight": max(row[0] for row in observed),
        "interval_method": "Fixed-snapshot 95% bounded concentration intervals for adaptively chosen contexts. SNIPS uses simultaneous numerator/denominator bounds; DM is conditional on a fixed 0.05 prediction.",
        "detail": "Target: 80% best scored remaining candidate plus 20% uniform. This estimates the randomized position under logged prefixes, not changing the whole slate or future traffic. Intervals are not a sequential stopping rule.",
    }
