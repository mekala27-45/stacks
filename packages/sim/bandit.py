from dataclasses import dataclass

import numpy as np

from packages.evaluation.metrics import bootstrap, wilson_interval
from packages.ope import estimate_intervals


@dataclass
class Simulator:
    seed: int = 72

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.rewards = 0.1 + 0.7 * rng.random((12, 6))
        logits = rng.normal(size=(12, 6))
        logging = np.exp(logits - logits.max(axis=1, keepdims=True))
        self.logging = 0.75 * logging / logging.sum(axis=1, keepdims=True) + 0.25 / 6
        target = np.exp(3 * self.rewards)
        self.target = target / target.sum(axis=1, keepdims=True)

    @property
    def value(self) -> float:
        return float((self.target * self.rewards).sum(axis=1).mean())

    def sample(self, n: int, seed: int, misspecified: bool = False) -> tuple[np.ndarray, ...]:
        rng = np.random.default_rng(seed)
        context = rng.integers(0, 12, size=n)
        actions = (rng.random(n)[:, None] > np.cumsum(self.logging[context], axis=1)).sum(axis=1)
        rewards = rng.binomial(1, self.rewards[context, actions]).astype(float)
        model = np.full((n, 6), 0.2) if misspecified else self.rewards[context]
        return rewards, actions, self.logging[context, actions], self.target[context], model


def validate_estimators(seeds: int = 200) -> dict[str, object]:
    simulator = Simulator()
    rows = []
    for n in (250, 1000):
        for misspecified in (False, True):
            samples: dict[str, list[dict[str, float]]] = {name: [] for name in ("IPS", "SNIPS", "DM", "DR")}
            for seed in range(seeds):
                estimates = estimate_intervals(*simulator.sample(n, seed, misspecified), seed=10000 + seed)
                for name, values in estimates.items():
                    samples[name].append(values)
            for name, estimates in samples.items():
                values = np.array([row["mean"] for row in estimates])
                errors = values - simulator.value
                covered = np.array(
                    [row["low"] <= simulator.value <= row["high"] for row in estimates], dtype=float
                )
                variance = float(np.var(values, ddof=1))
                resamples = np.random.default_rng(56).integers(0, seeds, size=(1000, seeds))
                boot_variance = np.var(values[resamples], axis=1, ddof=1)
                boot_rmse = np.sqrt(np.mean(errors[resamples] ** 2, axis=1))
                rmse = float(np.sqrt(np.mean(errors**2)))
                rows.append(
                    {
                        "estimator": name,
                        "sample_size": n,
                        "seeds": seeds,
                        "reward_model": "misspecified constant" if misspecified else "oracle",
                        "truth": simulator.value,
                        "estimate": bootstrap(values),
                        "bias": float(errors.mean()),
                        "bias_interval": bootstrap(errors),
                        "variance": variance,
                        "rmse": rmse,
                        "variance_interval": {
                            "mean": variance,
                            "low": float(np.quantile(boot_variance, 0.025)),
                            "high": float(np.quantile(boot_variance, 0.975)),
                        },
                        "rmse_interval": {
                            "mean": rmse,
                            "low": float(np.quantile(boot_rmse, 0.025)),
                            "high": float(np.quantile(boot_rmse, 0.975)),
                        },
                        "coverage": float(covered.mean()),
                        "coverage_interval": wilson_interval(int(covered.sum()), seeds),
                    }
                )
    return {
        "status": "measured",
        "truth": simulator.value,
        "contexts": 12,
        "actions": 6,
        "bootstrap_draws": 200,
        "seeds": seeds,
        "rows": rows,
    }
