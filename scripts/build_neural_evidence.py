"""Add measured neural retrieval and recurrent-session evidence to the core run."""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.evaluation.metrics import bootstrap, paired_comparisons
from packages.evaluation.split import matrices
from packages.retrieval.models import ModelBundle, load_bundle, sparse_item_cosine
from packages.retrieval.two_tower import train_two_tower
from packages.session.rnn import RecurrentModel, train_recurrent
from scripts import build_evidence as core

LABELS = {"two_tower": "Two-tower neural", "session_cosine": "Session cosine", "recurrent": "Elman recurrent"}


def finalize_diagnostics() -> None:
    result_dir = ROOT / "results"
    path = result_dir / "manifest.json"
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    artifacts = np.load(result_dir / "neural-artifacts.npz")
    items, queries = artifacts["item_embeddings"], artifacts["query_embeddings"]
    popularity = np.load(result_dir / "model-artifacts.npz")["popularity"]
    singular = np.linalg.svd(items, compute_uv=False)
    warm = popularity > 0
    geometry = {
        "status": "measured_from_frozen_artifacts",
        "dimensions": items.shape[1],
        "item_mean_active_channels": float((np.abs(items) > 1e-8).sum(axis=1).mean()),
        "query_mean_active_channels": float((np.abs(queries) > 1e-8).sum(axis=1).mean()),
        "item_zero_vector_fraction": float((np.linalg.norm(items, axis=1) == 0).mean()),
        "warm_item_zero_vector_fraction": float((np.linalg.norm(items[warm], axis=1) == 0).mean()),
        "query_zero_vector_fraction": float((np.linalg.norm(queries, axis=1) == 0).mean()),
        "item_training_cold_count": int((~warm).sum()),
        "item_column_standard_deviations": items.std(axis=0).tolist(),
        "singular_values": singular.tolist(),
        "stable_rank": float(np.square(singular).sum() / np.square(singular[0])),
        "training_positive_fraction_sampled": manifest["neural_training"]["two_tower"]["training_pairs"]
        / manifest["dataset"]["positive_training_interactions"],
        "nominal_batch_chance_cross_entropy": float(np.log(128)),
        "detail": "Exact descriptive geometry of the frozen arrays, not confidence intervals. Training-cold item vectors are deliberately zeroed. Low held-out accuracy and training loss near nominal batch chance show weak learning in this finite training budget; the nominal log128 reference ignores duplicate-item masking. Nonzero singular values and active channels do not support a claim of global dead-ReLU collapse. Local ANN recall on these weak embeddings must not be generalized to other models or catalogs.",
    }
    manifest["neural_training"]["geometry"] = geometry
    core.write_json(result_dir / "neural-geometry.json", geometry)
    pg_path = result_dir / "pgvector-benchmark.json"
    if pg_path.exists():
        benchmark = json.loads(pg_path.read_text(encoding="utf-8"))
        actual_hash = hashlib.sha256((result_dir / "neural-artifacts.npz").read_bytes()).hexdigest()
        if benchmark.get("artifact_sha256") == actual_hash:
            manifest["pgvector"] = benchmark
            for module in manifest["modules"]:
                if module["name"] == "pgvector / approximate nearest neighbors":
                    module.update(
                        {
                            "status": "measured_local_postgres",
                            "detail": "Exact SQL and HNSW measured on the frozen two-tower embeddings in local PostgreSQL, with the same training-seen filters. This is separate from the hosted D1 serving path.",
                        }
                    )
            manifest["limitations"] = [
                item.replace("approximate indexing, ", "") for item in manifest["limitations"]
            ]
        else:
            manifest["pgvector"] = {
                "status": "stale_artifact",
                "detail": "The saved ANN benchmark does not match the current neural artifact SHA; rerun the independent benchmark before publishing recall.",
            }
    manifest["protocol"]["evaluation_users"] = manifest["dataset"]["evaluated_users"]
    manifest["limitations"] = [
        item.replace(
            "Snapshot genres and average ratings are descriptive metadata. Their historical availability cannot be verified.",
            "Snapshot titles, authors and tags power the content baseline; their historical availability cannot be verified. Average ratings are descriptive metadata only.",
        )
        for item in manifest["limitations"]
    ]
    core.write_json(path, manifest)
    core.write_json(ROOT / "web" / "public" / "data" / "evidence.json", manifest)


def histories_from(
    ratings: pd.DataFrame, mask: np.ndarray, user_map: dict[int, int]
) -> dict[int, np.ndarray]:
    positive = ratings[mask & (ratings.rating >= 4).to_numpy()].sort_values("source_row")
    return {
        user_map[int(group.user_id.iloc[0])]: group.book_id.to_numpy(dtype=np.int32) - 1
        for _, group in positive.groupby("user_id")
    }


class NeuralBundle(ModelBundle):
    def __init__(
        self,
        base: ModelBundle,
        users: np.ndarray,
        items: np.ndarray,
        rnn: RecurrentModel,
        histories: dict[int, np.ndarray],
    ) -> None:
        super().__init__(base.matrix, base.popularity, base.cosine, users, items)
        self.rnn = rnn
        self.histories = histories

    def score(self, user: int) -> dict[str, np.ndarray]:
        history = self.histories.get(user, np.array([], dtype=np.int32))
        if not history.size:
            return {model: self.popularity.copy() for model in LABELS}
        recent = history[-12:]
        weights = 0.85 ** np.arange(len(recent) - 1, -1, -1)
        session = np.asarray(self.cosine[recent].multiply(weights[:, None]).sum(axis=0)).ravel()
        return {
            "two_tower": self.item_factors @ self.user_factors[user],
            "session_cosine": session,
            "recurrent": self.rnn.outputs @ self.rnn.encode(recent),
        }


def fit_neural(
    base: ModelBundle, histories: dict[int, np.ndarray]
) -> tuple[NeuralBundle, dict[str, Any], dict[str, Any], Any]:
    print("Training dual nonlinear encoders on400000 positive pairs", flush=True)
    tower, tower_metadata = train_two_tower(base.matrix)
    users, items = tower.vectors()
    items[base.popularity == 0] = 0
    print("Training recurrent next-item model on24000 source-order sequences", flush=True)
    rnn, rnn_metadata = train_recurrent(histories, base.matrix.shape[1])
    rnn.outputs[base.popularity == 0] = 0
    return NeuralBundle(base, users, items, rnn, histories), tower_metadata, rnn_metadata, tower


def main() -> None:
    started = time.perf_counter()
    result_dir = ROOT / "results"
    manifest: dict[str, Any] = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset"]["catalog_size"] != 10000:
        raise ValueError("Run the full core pipeline before neural augmentation")
    base = load_bundle(result_dir)
    assert base.user_ids is not None and base.seen is not None
    user_map = {int(uid): index for index, uid in enumerate(base.user_ids)}
    ratings = pd.read_parquet(ROOT / "data" / "goodbooks" / "ratings.parquet")
    mask = (ratings.source_row < manifest["protocol"]["cutoff_row"]).to_numpy()
    _, seen, truth = matrices(ratings, mask, user_map, 10000)
    main_users = core.eligible_users(base.matrix, truth)
    histories = histories_from(ratings, mask, user_map)
    neural, tower_meta, recurrent_meta, tower = fit_neural(base, histories)
    selected_seen = base.seen[main_users]
    np.savez_compressed(
        result_dir / "neural-artifacts.npz",
        item_embeddings=neural.item_factors,
        query_embeddings=neural.user_factors[main_users],
        seen_indptr=selected_seen.indptr,
        seen_indices=selected_seen.indices,
        user_ids=base.user_ids[main_users],
        user_indices=main_users,
    )
    np.savez_compressed(
        result_dir / "two-tower.npz",
        user_embeddings=tower.user_embeddings,
        item_embeddings=tower.item_embeddings,
        user_projection=tower.user_projection,
        item_projection=tower.item_projection,
        user_vectors=neural.user_factors,
        item_vectors=neural.item_factors,
    )
    np.savez_compressed(
        result_dir / "recurrent.npz",
        inputs=neural.rnn.inputs,
        outputs=neural.rnn.outputs,
        recurrent=neural.rnn.recurrent,
        bias=neural.rnn.bias,
    )
    print("NEURAL EMBEDDINGS READY for independent pgvector benchmark", flush=True)
    catalog = json.loads((ROOT / "web" / "public" / "data" / "catalog.json").read_text(encoding="utf-8"))
    genre_names = sorted({item["genre"] for item in catalog})
    genres = np.array([genre_names.index(item["genre"]) for item in catalog])
    core.LABELS = LABELS
    records, lists = core.evaluate(neural, seen, truth, main_users, genres)
    sampled, _ = core.evaluate(neural, seen, truth, main_users, genres, sampled=True)
    print("Training neural shortcut models on independent random training split", flush=True)
    random_mask = np.random.default_rng(93).random(len(ratings)) < 0.8
    random_matrix, random_seen, random_truth = matrices(ratings, random_mask, user_map, 10000)
    random_pop = np.asarray(random_matrix.sum(axis=0)).ravel().astype(np.float32)
    random_cosine = sparse_item_cosine(random_matrix.astype(np.float32))
    random_base = ModelBundle(
        random_matrix, random_pop, random_cosine, np.empty((len(user_map), 16)), np.empty((10000, 16))
    )
    random_histories = histories_from(ratings, random_mask, user_map)
    random_neural, random_tower_meta, random_rnn_meta, _ = fit_neural(random_base, random_histories)
    random_users = core.eligible_users(random_matrix, random_truth)
    random_records, _ = core.evaluate(random_neural, random_seen, random_truth, random_users, genres)
    manifest["metrics"] = [row for row in manifest["metrics"] if row["model"] not in LABELS] + core.summarize(
        records, lists
    )
    old = pd.read_parquet(result_dir / "per-user.parquet")
    old = old[~old.model.isin(LABELS)]
    rows = [
        {
            "user_id": int(base.user_ids[uid]),
            "model": name,
            **{key: float(values[index]) for key, values in metrics.items()},
        }
        for name, metrics in records.items()
        for index, uid in enumerate(main_users)
    ]
    all_users = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    all_users.to_parquet(result_dir / "per-user.parquet", index=False)
    all_users.to_csv(result_dir / "per-user.csv", index=False)
    list_path = result_dir / "recommendation-lists.parquet"
    old_lists = pd.read_parquet(list_path)
    old_lists = old_lists[~old_lists.model.isin(LABELS)]
    new_lists = pd.DataFrame(
        [
            {"user_id": int(base.user_ids[uid]), "model": model, "book_ids": (ranked[index] + 1).tolist()}
            for model, ranked in lists.items()
            for index, uid in enumerate(main_users)
        ]
    )
    pd.concat([old_lists, new_lists], ignore_index=True).to_parquet(list_path, index=False)
    paired = {
        str(model): group.sort_values("user_id").ndcg.to_numpy()
        for model, group in all_users.groupby("model", sort=False)
    }
    manifest["comparisons"] = paired_comparisons(paired)
    leader = max(paired, key=lambda model: float(paired[model].mean()))
    corrected = (
        leader
        if all(row["significant"] for row in manifest["comparisons"] if leader in (row["left"], row["right"]))
        else None
    )
    manifest["winner"] = {"observed": leader, "corrected": corrected}
    for kind in ("split", "sampled"):
        manifest["inflation"][kind] = [
            row for row in manifest["inflation"][kind] if row["model"] not in LABELS
        ]
    for name, label in LABELS.items():
        baseline = records[name]["ndcg"]
        manifest["inflation"]["split"].append(
            {
                "model": name,
                "label": label,
                "full": bootstrap(baseline),
                "shortcut": bootstrap(random_records[name]["ndcg"]),
                "delta": core.independent_difference(baseline, random_records[name]["ndcg"]),
                "ordered_users": len(main_users),
                "random_users": len(random_users),
            }
        )
        manifest["inflation"]["sampled"].append(
            {
                "model": name,
                "label": label,
                "full": bootstrap(baseline),
                "shortcut": bootstrap(sampled[name]["ndcg"]),
                "delta": bootstrap(sampled[name]["ndcg"] - baseline),
                "users": len(main_users),
            }
        )
    manifest["inflation"]["ordered_winner"] = leader
    manifest["inflation"]["random_winner"] = max(
        manifest["inflation"]["split"], key=lambda row: row["shortcut"]["mean"]
    )["model"]
    manifest["inflation"]["sampled_winner"] = max(
        manifest["inflation"]["sampled"], key=lambda row: row["shortcut"]["mean"]
    )["model"]
    cold_users = core.eligible_users(base.matrix, truth, cold=True)
    cold_items = set(np.flatnonzero(base.popularity == 0))
    cold_item_users = np.array([uid for uid in main_users if truth[int(uid)] & cold_items])
    manifest["cold_slices"] = [row for row in manifest["cold_slices"] if row["model"] not in LABELS]
    for label, users, relevant in (
        ("Cold readers (0-4 positive training items)", cold_users, truth),
        (
            "Cold item positives",
            cold_item_users,
            {int(uid): truth[int(uid)] & cold_items for uid in cold_item_users},
        ),
    ):
        if not len(users):
            continue
        cold, _ = core.evaluate(neural, seen, relevant, users, genres)
        for name, metrics in cold.items():
            manifest["cold_slices"].append(
                {
                    "slice": label,
                    "users": len(users),
                    "model": name,
                    "label": LABELS[name],
                    **{key: bootstrap(metrics[key]) for key in ("ndcg", "recall", "recall200")},
                }
            )
    manifest["neural_training"] = {
        "two_tower": tower_meta,
        "recurrent": recurrent_meta,
        "random_shortcut_two_tower": random_tower_meta,
        "random_shortcut_recurrent": random_rnn_meta,
        "evaluation_users": len(main_users),
        "cohort": "Same held-out reader IDs, positives, seen-item filters and full10000-item catalog as every other headline model",
        "session_cosine": {
            "history_length": 12,
            "decay": 0.85,
            "detail": "Source-order positive history, with earlier clicks exponentially downweighted",
        },
        "run_seconds": round(time.perf_counter() - started, 3),
    }
    manifest["modules"] = [row for row in manifest["modules"] if row["name"] != "Two-tower neural retrieval"]
    manifest["modules"].extend(
        [
            {
                "name": "Two-tower neural retrieval",
                "status": "measured",
                "detail": tower_meta["architecture"],
            },
            {
                "name": "Recurrent session model",
                "status": "measured",
                "detail": recurrent_meta["architecture"],
            },
        ]
    )
    manifest["limitations"] = [item.replace("neural models, ", "") for item in manifest["limitations"]]
    manifest["limitations"].append(
        "Two-tower and recurrent models use explicit NumPy training and finite sampled training pairs/sequences. Their losses are optimization diagnostics; held-out results and both shortcuts are reported even when these neural models lose."
    )
    core.write_json(result_dir / "manifest.json", manifest)
    core.write_json(ROOT / "web" / "public" / "data" / "evidence.json", manifest)
    print(
        json.dumps(
            {
                "seconds": manifest["neural_training"]["run_seconds"],
                "winner": manifest["winner"],
                "new_models": [
                    {"model": row["model"], "ndcg": row["ndcg"]}
                    for row in manifest["metrics"]
                    if row["model"] in LABELS
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    if "--diagnostics-only" not in sys.argv:
        main()
    finalize_diagnostics()
