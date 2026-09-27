"""Measure end-to-end persisted shelf latency using explicitly excluded test traffic."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np


def measure(url: str, samples: int = 30, warmup: int = 3, reader_id: str | None = None) -> dict[str, Any]:
    if not 5 <= samples <= 40 or not 0 <= warmup <= 5:
        raise ValueError("Use 5-40 measured requests and at most 5 warmups to respect demonstration limits")
    with httpx.Client(base_url=url.rstrip("/"), timeout=60) as client:
        health = client.get("/health")
        health.raise_for_status()
        created = client.post("/v1/session", json={"reader_id": reader_id, "traffic_kind": "load_test"})
        created.raise_for_status()
        session = created.json()
        token = session["session_id"]
        timings: list[float] = []
        for index in range(samples + warmup):
            started = time.perf_counter()
            response = client.post(f"/v1/session/{token}/preferences", json={"genre": "All books"})
            response.raise_for_status()
            elapsed = (time.perf_counter() - started) * 1000
            if index >= warmup:
                timings.append(elapsed)
        logs = client.get(f"/v1/session/{token}/logs")
        logs.raise_for_status()
        if any(row["trace"].get("traffic_kind") != "load_test" for row in logs.json()["impressions"]):
            raise ValueError("Measured traffic was not labeled for OPE exclusion")
    values = np.asarray(timings, dtype=float)
    draws = values[np.random.default_rng(712).integers(0, len(values), size=(1000, len(values)))]

    def quantile_interval(level: float) -> dict[str, float]:
        estimates = np.quantile(draws, level, axis=1)
        return {
            "mean": float(np.quantile(values, level)),
            "low": float(np.quantile(estimates, 0.025)),
            "high": float(np.quantile(estimates, 0.975)),
        }

    return {
        "status": "measured",
        "measured_at": datetime.now(UTC).isoformat(),
        "url": url.rstrip("/"),
        "backend": session["backend"],
        "artifact_version": session.get("artifact_version"),
        "model_version": session["model_version"],
        "scoring_mode": session.get("scoring_mode"),
        "requests": samples,
        "warmup_requests": warmup,
        "concurrency": 1,
        "operation": "Persist a recomputed full shelf via session preferences; includes client transport, server scoring and database writes",
        "p50_ms": float(np.quantile(values, 0.5)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "p50": quantile_interval(0.5),
        "p99": quantile_interval(0.99),
        "minimum_ms": float(values.min()),
        "maximum_ms": float(values.max()),
        "raw_ms": timings,
        "traffic_kind": "load_test",
        "detail": "A small sequential warm-service test, not a capacity or production SLO claim. Percentile bootstrap ranges describe this finite latency sample. Test exposures remain labeled and are excluded from own-log OPE.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--reader-id")
    parser.add_argument("--output", type=Path, default=Path("results/latency.json"))
    args = parser.parse_args()
    result = measure(args.url, args.samples, reader_id=args.reader_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: result[name] for name in ("backend", "requests", "p50_ms", "p99_ms")}, indent=2))


if __name__ == "__main__":
    main()
