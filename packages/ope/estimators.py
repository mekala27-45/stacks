import numpy as np


def _terms(
    rewards: np.ndarray,
    actions: np.ndarray,
    propensities: np.ndarray,
    target: np.ndarray,
    reward_model: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rewards, actions, propensities = np.asarray(rewards), np.asarray(actions), np.asarray(propensities)
    n = rewards.size
    if n == 0 or rewards.shape != (n,) or actions.shape != (n,) or propensities.shape != (n,):
        raise ValueError("Nonempty reward, action, and propensity vectors must align")
    if target.ndim != 2 or target.shape != reward_model.shape or target.shape[0] != n:
        raise ValueError("Target and reward model must have shape [observations, actions]")
    if not np.issubdtype(actions.dtype, np.integer) or np.any((actions < 0) | (actions >= target.shape[1])):
        raise ValueError("Actions must be valid integer action indexes")
    if any(not np.isfinite(array).all() for array in [rewards, propensities, target, reward_model]):
        raise ValueError("Estimator inputs must be finite")
    if np.any((propensities <= 0) | (propensities > 1)):
        raise ValueError("Logged propensities must lie in (0, 1]")
    if np.any(target < 0) or not np.allclose(target.sum(axis=1), 1):
        raise ValueError("Target policy rows must be probability distributions")
    weights = target[np.arange(n), actions] / propensities
    if weights.sum() == 0:
        raise ValueError("Target policy has no observed support")
    direct = (target * reward_model).sum(axis=1)
    residual = rewards - reward_model[np.arange(n), actions]
    return weights, rewards * weights, direct, direct + weights * residual


def estimate(
    rewards: np.ndarray,
    actions: np.ndarray,
    propensities: np.ndarray,
    target: np.ndarray,
    reward_model: np.ndarray,
) -> dict[str, float]:
    """Estimate target value under a caller-established overlap assumption.

    Logged chosen-action propensities cannot prove support for unchosen actions.
    Callers must establish that every target action has positive logging-policy
    probability at its context; the simulator and uniform OBD logger do so.
    """
    weights, weighted, direct, robust = _terms(rewards, actions, propensities, target, reward_model)
    return {
        "IPS": float(weighted.mean()),
        "SNIPS": float(weighted.sum() / weights.sum()),
        "DM": float(direct.mean()),
        "DR": float(robust.mean()),
    }


def estimate_intervals(
    rewards: np.ndarray,
    actions: np.ndarray,
    propensities: np.ndarray,
    target: np.ndarray,
    reward_model: np.ndarray,
    seed: int = 0,
    draws: int = 200,
) -> dict[str, dict[str, float]]:
    weights, weighted, direct, robust = _terms(rewards, actions, propensities, target, reward_model)
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("Bootstrap draws must be a positive integer")
    indices = np.random.default_rng(seed).integers(0, rewards.size, size=(draws, rewards.size))
    weight_sums = weights[indices].sum(axis=1)
    if np.any(weight_sums == 0):
        # Dropping these draws would condition on support and silently alter the
        # bootstrap distribution. Refuse the interval instead of publishing NaN.
        raise ValueError(
            "SNIPS bootstrap draw has no observed target support; "
            "collect more supported feedback before reporting intervals"
        )
    bootstrap = {
        "IPS": weighted[indices].mean(axis=1),
        "SNIPS": weighted[indices].sum(axis=1) / weight_sums,
        "DM": direct[indices].mean(axis=1),
        "DR": robust[indices].mean(axis=1),
    }
    points = estimate(rewards, actions, propensities, target, reward_model)
    return {
        name: {
            "mean": points[name],
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
        }
        for name, values in bootstrap.items()
    }
