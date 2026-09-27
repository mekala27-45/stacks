"""Crosscheck point estimates against exact pinned OBP arithmetic methods.

This is a source-level reference crosscheck, not an installed-library
compatibility claim. Only four unmodified `_estimate_round_rewards` methods
are compiled. Upstream package initialization, validators, confidence
intervals, and Torch-dependent helpers are not executed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.ope import estimate
from packages.sim import Simulator

COMMIT = "8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4"
SOURCE_URL = f"https://raw.githubusercontent.com/st-tech/zr-obp/{COMMIT}/obp/ope/estimators.py"
EXPECTED_SHA256 = "197b9f0d25e7b5766ce71de3703b142efb561170f4c15ac13ab0fcfd860fe55a"
CLASSES = {
    "IPS": "InverseProbabilityWeighting",
    "SNIPS": "SelfNormalizedInverseProbabilityWeighting",
    "DM": "DirectMethod",
    "DR": "DoublyRobust",
}


def reference_methods(source: bytes) -> dict[str, Callable[..., Any]]:
    if hashlib.sha256(source).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Pinned OBP source hash mismatch; refusing to execute reference")
    tree = ast.parse(source.decode("utf-8"), filename=SOURCE_URL)
    methods: dict[str, Callable[..., Any]] = {}
    for short, class_name in CLASSES.items():
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
        if len(classes) != 1:
            raise ValueError(f"Missing unique reference class: {class_name}")
        definitions = [
            node
            for node in classes[0].body
            if isinstance(node, ast.FunctionDef) and node.name == "_estimate_round_rewards"
        ]
        if len(definitions) != 1:
            raise ValueError(f"Missing unique arithmetic method: {class_name}")
        # The original AST body, arguments, defaults and annotations are unchanged.
        module = ast.Module(body=[definitions[0]], type_ignores=[])
        namespace: dict[str, Any] = {"np": np, "Optional": Optional}
        exec(compile(module, SOURCE_URL, "exec"), namespace)  # noqa: S102 - hash-verified upstream AST only
        methods[short] = namespace["_estimate_round_rewards"]
    return methods


def compare(source: bytes, seeds: int = 200, tolerance: float = 1e-10) -> dict[str, Any]:
    if seeds <= 0:
        raise ValueError("Refusing a crosscheck with no fixtures")
    methods = reference_methods(source)
    simulator = Simulator()
    parameters = SimpleNamespace(lambda_=np.inf)
    rows: list[dict[str, Any]] = []
    for n in (250, 1000):
        for misspecified in (False, True):
            largest = {name: 0.0 for name in CLASSES}
            for seed in range(seeds):
                fixture = simulator.sample(n, seed, misspecified)
                rewards, actions, propensities, target, reward_model = fixture
                actual = estimate(*fixture)
                for name, method in methods.items():
                    reference = float(
                        np.mean(
                            method(
                                parameters,
                                reward=rewards,
                                action=actions,
                                pscore=propensities,
                                action_dist=target[:, :, None],
                                estimated_rewards_by_reg_model=reward_model[:, :, None],
                                position=np.zeros(n, dtype=int),
                            )
                        )
                    )
                    error = abs(actual[name] - reference)
                    if not np.isfinite(error) or error > tolerance:
                        raise AssertionError(
                            f"{name} disagrees with OBP for n={n}, seed={seed}, misspecified={misspecified}: {error}"
                        )
                    largest[name] = max(largest[name], error)
            for name, error in largest.items():
                rows.append(
                    {
                        "estimator": name,
                        "sample_size": n,
                        "reward_model": "misspecified constant" if misspecified else "oracle",
                        "seeds": seeds,
                        "max_absolute_error": error,
                    }
                )
    return {
        "status": "passed_source_reference",
        "detail": "Local IPS, SNIPS, DM and DR point estimates agree with unmodified arithmetic methods extracted from pinned Open Bandit Pipeline source on the same seeded simulator fixtures. This is a source-level crosscheck, not an installed OBP package or confidence-interval crosscheck.",
        "source": SOURCE_URL,
        "source_commit": COMMIT,
        "source_sha256": EXPECTED_SHA256,
        "source_license": "Apache-2.0",
        "license_url": f"https://github.com/st-tech/zr-obp/blob/{COMMIT}/LICENSE",
        "source_copyright": "Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc.",
        "scope": "Four exact _estimate_round_rewards method ASTs; NumPy arrays, lambda_=infinity, one position. Upstream validators and bootstrap code are outside this check.",
        "tolerance": tolerance,
        "fixtures": seeds * 4,
        "estimator_comparisons": seeds * 4 * len(CLASSES),
        "rows": rows,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    cache = ROOT / "data" / "raw" / "obp_reference"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "estimators.py"
    if not path.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            source = response.read()
        if hashlib.sha256(source).hexdigest() != EXPECTED_SHA256:
            raise ValueError("Downloaded source checksum mismatch")
        path.write_bytes(source)
        with urllib.request.urlopen(
            f"https://raw.githubusercontent.com/st-tech/zr-obp/{COMMIT}/LICENSE", timeout=60
        ) as response:
            (cache / "LICENSE").write_bytes(response.read())
    result = compare(path.read_bytes())
    destination = ROOT / "results" / "obp-crosscheck.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Passed {result['estimator_comparisons']} point-estimate comparisons against pinned OBP arithmetic."
    )


if __name__ == "__main__":
    main()
