import math

import pytest

from packages.registry import evaluate_gates, promotion_eligible


@pytest.fixture
def clean_metrics():
    return {
        "ndcg_lift_ci_low": 0.01,
        "ndcg_q_value": 0.02,
        "recall_at_200": 0.8,
        "coverage": 0.5,
        "long_tail_share": 0.4,
        "calibration_divergence": 0.1,
        "p99_latency_ms": 120,
        "feature_schema": "catalog-v1",
    }


def test_all_gates_accept_clean_evidence(clean_metrics):
    assert promotion_eligible(clean_metrics)
    assert len(evaluate_gates(clean_metrics)) == 7


@pytest.mark.parametrize(
    ("field", "invalid", "gate_name"),
    [
        ("ndcg_lift_ci_low", -0.01, "corrected_ndcg_improvement"),
        ("ndcg_q_value", 0.2, "corrected_ndcg_improvement"),
        ("recall_at_200", 0.1, "recall_at_200"),
        ("coverage", 0.01, "coverage"),
        ("long_tail_share", 0.01, "long_tail_share"),
        ("calibration_divergence", 0.8, "calibration_divergence"),
        ("p99_latency_ms", 800, "p99_latency_ms"),
        ("feature_schema", "different-schema", "feature_schema"),
    ],
)
def test_each_gate_rejects_a_deliberate_violation(clean_metrics, field, invalid, gate_name):
    clean_metrics[field] = invalid
    gates = {gate.name: gate for gate in evaluate_gates(clean_metrics)}
    assert not gates[gate_name].passed
    assert not promotion_eligible(clean_metrics)


@pytest.mark.parametrize("empty", [None, {}])
def test_every_gate_refuses_absent_evidence(empty):
    assert not any(gate.passed for gate in evaluate_gates(empty))
    assert not promotion_eligible(empty)


@pytest.mark.parametrize("bad_number", [math.nan, math.inf, True, "0.9", None])
def test_invalid_numbers_cannot_promote(clean_metrics, bad_number):
    clean_metrics["recall_at_200"] = bad_number
    assert not promotion_eligible(clean_metrics)
