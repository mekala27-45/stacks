"""Measure pgvector HNSW loss against exact SQL on the same frozen vectors.

Use an isolated local benchmark database, not the live session database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import psycopg

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "pgvector/pgvector@sha256:cf134a767f474095eeba57e0117be8e568e011a63f33fbf252f14c9b760f8e6f"


def vector(values: npt.NDArray[np.floating[Any]]) -> str:
    if not np.isfinite(values).all():
        raise ValueError("Embeddings must be finite")
    return "[" + ",".join(format(float(value), ".9g") for value in values) + "]"


def interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260927)
    means = array[rng.integers(0, len(array), size=(500, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ROOT / "results" / "neural-artifacts.npz")
    parser.add_argument("--queries", type=int, default=200)
    args = parser.parse_args()
    url = os.environ.get("STACKS_VECTOR_DATABASE_URL")
    if not url:
        raise SystemExit("Set STACKS_VECTOR_DATABASE_URL to an isolated benchmark PostgreSQL database")
    artifact = np.load(args.artifact)
    items = np.asarray(artifact["item_embeddings"], dtype=np.float32)
    queries = np.asarray(artifact["query_embeddings"], dtype=np.float32)[: args.queries]
    # The archive carries training-only seen item indices for each evaluated query.
    indptr, indices = artifact["seen_indptr"], artifact["seen_indices"]
    if items.ndim != 2 or queries.ndim != 2 or items.shape[1] != queries.shape[1]:
        raise ValueError("Incompatible item and query embeddings")
    dimensions = int(items.shape[1])
    if not 1 <= dimensions <= 2000:
        raise ValueError("Unsupported vector dimension")
    if len(queries) == 0 or not np.isfinite(queries).all():
        raise ValueError("No valid query vectors")
    query = "SELECT item_id FROM stacks_benchmark.items WHERE NOT(item_id = ANY(%s)) ORDER BY embedding <#> %s::vector LIMIT 200"
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE SCHEMA IF NOT EXISTS stacks_benchmark")
        # A dedicated transaction replaces only this script's isolated benchmark table.
        conn.execute("DROP TABLE IF EXISTS stacks_benchmark.items")
        conn.execute(
            f"CREATE TABLE stacks_benchmark.items (item_id integer PRIMARY KEY, embedding vector({dimensions}) NOT NULL)"
        )
        with conn.cursor().copy("COPY stacks_benchmark.items (item_id, embedding) FROM STDIN") as copy:
            for index, embedding in enumerate(items):
                copy.write_row((index, vector(embedding)))
        conn.execute(
            "CREATE INDEX stacks_benchmark_hnsw ON stacks_benchmark.items USING hnsw(embedding vector_ip_ops) WITH (m=16, ef_construction=128)"
        )
        conn.execute("ANALYZE stacks_benchmark.items")
        version = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
        exact: list[set[int]] = []
        exact_ms: list[float] = []
        payloads: list[tuple[list[int], str]] = []
        conn.execute("SET enable_indexscan=off")
        conn.execute("SET enable_bitmapscan=off")
        for i, embedding in enumerate(queries):
            payload = ([int(v) for v in indices[int(indptr[i]) : int(indptr[i + 1])]], vector(embedding))
            payloads.append(payload)
            started = time.perf_counter()
            exact_rows = conn.execute(query, payload).fetchall()
            exact_ms.append((time.perf_counter() - started) * 1000)
            exact.append({int(row[0]) for row in exact_rows})
        exact_plan = conn.execute("EXPLAIN " + query, payloads[0]).fetchall()
        conn.execute("SET enable_indexscan=on")
        conn.execute("SET enable_seqscan=off")
        conn.execute("SET hnsw.iterative_scan='strict_order'")
        rows: list[dict[str, Any]] = []
        for ef in (200, 400, 800):
            conn.execute(f"SET hnsw.ef_search={ef}")
            plan = conn.execute("EXPLAIN " + query, payloads[0]).fetchall()
            if "stacks_benchmark_hnsw" not in str(plan):
                raise RuntimeError("Benchmark planner did not use the HNSW index")
            recalls: list[float] = []
            latencies: list[float] = []
            for payload, truth in zip(payloads, exact, strict=True):
                started = time.perf_counter()
                returned = {int(row[0]) for row in conn.execute(query, payload).fetchall()}
                latencies.append((time.perf_counter() - started) * 1000)
                recalls.append(len(returned & truth) / len(truth))
            rows.append(
                {
                    "ef_search": ef,
                    "recall_at_200": interval(recalls),
                    "p50_ms": float(np.median(latencies)),
                    "p99_ms": float(np.quantile(latencies, 0.99)),
                    "query_plan": [row[0] for row in plan],
                }
            )
    result = {
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "status": "measured_local_postgres",
        "model": "two_tower",
        "image": IMAGE,
        "pgvector_version": version[0] if version else None,
        "items": len(items),
        "queries": len(queries),
        "dimensions": dimensions,
        "distance": "inner product",
        "training_seen_excluded": True,
        "exact": {
            "p50_ms": float(np.median(exact_ms)),
            "p99_ms": float(np.quantile(exact_ms, 0.99)),
            "query_plan": [row[0] for row in exact_plan],
        },
        "rows": rows,
        "detail": "Local Docker PostgreSQL benchmark with frozen two-tower embeddings. Exact SQL and HNSW use identical training-seen exclusions. Latencies include the local client round trip, are warm sequential requests, and do not describe the hosted D1 API. Recall intervals resample queries.",
    }
    (ROOT / "results" / "pgvector-benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
