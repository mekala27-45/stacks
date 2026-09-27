"""Fail-closed promotion eligibility checks; serving activation is separate."""

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    observed: Any
    requirement: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_THRESHOLDS = {
    "recall_at_200": 0.5,
    "coverage": 0.1,
    "long_tail_share": 0.1,
    "calibration_divergence": 0.25,
    "p99_latency_ms": 500.0,
}


def evaluate_gates(
    metrics: dict[str, Any] | None,
    expected_schema: str = "catalog-v1",
    thresholds: dict[str, float] | None = None,
) -> list[GateResult]:
    values = metrics or {}
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def number(name: str) -> float | None:
        raw = values.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (float, int)) or not isfinite(raw):
            return None
        return float(raw)

    lift, q_value = number("ndcg_lift_ci_low"), number("ndcg_q_value")
    result = [
        GateResult(
            "corrected_ndcg_improvement",
            lift is not None and q_value is not None and lift > 0 and 0 <= q_value <= 0.05,
            {"ndcg_lift_ci_low": lift, "ndcg_q_value": q_value},
            "Positive paired-bootstrap lift lower bound and BH-adjusted q <= 0.05",
        )
    ]
    for name in ("recall_at_200", "coverage", "long_tail_share"):
        value = number(name)
        result.append(
            GateResult(
                name, value is not None and limits[name] <= value <= 1, value, f">= {limits[name]} and <= 1"
            )
        )
    for name in ("calibration_divergence", "p99_latency_ms"):
        value = number(name)
        result.append(
            GateResult(
                name, value is not None and 0 <= value <= limits[name], value, f">= 0 and <= {limits[name]}"
            )
        )
    result.append(
        GateResult(
            "feature_schema",
            values.get("feature_schema") == expected_schema,
            values.get("feature_schema"),
            f"Exactly {expected_schema}",
        )
    )
    return result


def promotion_eligible(metrics: dict[str, Any] | None, expected_schema: str = "catalog-v1") -> bool:
    return all(gate.passed for gate in evaluate_gates(metrics, expected_schema))
