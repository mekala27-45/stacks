"""Estimator contracts that prevent silent broadcasting and unsupported intervals."""

import numpy as np
import pytest

from packages.ope import estimate, estimate_intervals


def fixture():
    return (
        np.array([0.0, 1.0]),
        np.array([0, 1]),
        np.full(2, 0.5),
        np.full((2, 2), 0.5),
        np.array([[0.2, 0.3], [0.2, 0.3]]),
    )


@pytest.mark.parametrize("estimator", [estimate, estimate_intervals])
def test_column_reward_vectors_are_rejected_before_broadcasting(estimator):
    rewards, actions, propensities, target, model = fixture()
    with pytest.raises(ValueError, match="vectors must align"):
        estimator(rewards[:, None], actions, propensities, target, model)


def test_sparse_target_bootstrap_refuses_undefined_snips_interval():
    rewards, actions, propensities, _, model = fixture()
    target = np.array([[1.0, 0.0], [1.0, 0.0]])
    # The point estimate has support, while some bootstrap resamples do not.
    point = estimate(rewards, actions, propensities, target, model)
    assert all(np.isfinite(value) for value in point.values())
    with pytest.raises(ValueError, match="SNIPS bootstrap draw has no observed target support"):
        estimate_intervals(rewards, actions, propensities, target, model, seed=0, draws=200)


def test_no_observed_target_support_refuses_even_point_estimates():
    rewards, _, propensities, _, model = fixture()
    with pytest.raises(ValueError, match="no observed support"):
        estimate(rewards, np.ones(2, dtype=int), propensities, np.array([[1.0, 0.0], [1.0, 0.0]]), model)


@pytest.mark.parametrize("draws", [0, -1, True, 1.5])
def test_empty_or_invalid_bootstrap_draw_counts_refuse(draws):
    with pytest.raises(ValueError, match="positive integer"):
        estimate_intervals(*fixture(), draws=draws)


def test_full_support_bootstrap_remains_finite_and_matches_point_estimates():
    expected = estimate(*fixture())
    observed = estimate_intervals(*fixture(), seed=42)
    for name, result in observed.items():
        assert result["mean"] == expected[name]
        assert all(np.isfinite(value) for value in result.values())
