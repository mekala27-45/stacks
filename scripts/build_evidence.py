"""Download pinned public data, fit bounded models, and publish measured evidence.

Run from the project root: python scripts/build_evidence.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gc
import hashlib
import json
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.content import build_vectors
from packages.core import STATEMENT
from packages.evaluation.metrics import (
    bootstrap,
    intra_list_diversity,
    paired_comparisons,
    rank_unseen,
    ranking_metrics,
)
from packages.evaluation.split import matrices
from packages.ope.obd import run_obd
from packages.ranking.learned import FEATURE_NAMES, train_ranker
from packages.retrieval.models import ModelBundle, train_models
from packages.sim import validate_estimators

COMMIT = "6dd165b555a7b47b2dd36743a425776e641ff50c"
SOURCE = f"https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/{COMMIT}/"
SOURCE_ROWS = 5_976_479
CATALOG_SIZE = 10000
CUTOFF = int(SOURCE_ROWS * 0.8)
EVALUATION_USERS = 5000
SEED = 20260926
LABELS = {
    "popularity": "Popularity",
    "item_cosine": "Item cosine",
    "als": "Implicit ALS",
    "blend": "Fixed blend",
    "content": "Content TF-IDF",
    "lambdamart": "LambdaMART",
}
GENRES = {
    "fantasy": "Fantasy",
    "science-fiction": "Science fiction",
    "sci-fi": "Science fiction",
    "mystery": "Mystery",
    "thriller": "Mystery",
    "crime": "Mystery",
    "romance": "Romance",
    "classics": "Classics",
    "classic": "Classics",
    "non-fiction": "Nonfiction",
    "nonfiction": "Nonfiction",
    "biography": "Nonfiction",
    "memoir": "Nonfiction",
    "history": "Nonfiction",
    "poetry": "Poetry",
    "young-adult": "Young adult",
    "childrens": "Young adult",
    "children": "Young adult",
    "historical-fiction": "Fiction",
    "contemporary": "Fiction",
    "literary-fiction": "Fiction",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def cached_download(name: str) -> Path:
    path = ROOT / ".cache" / "goodbooks" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading pinned {name}", flush=True)
        request = urllib.request.Request(SOURCE + name, headers={"User-Agent": "stacks-public-data-demo"})
        with urllib.request.urlopen(request, timeout=120) as response:
            path.write_bytes(response.read())
    return path


def source_ratings() -> Path:
    return cached_download("ratings.csv")


def ingest() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    books = pd.read_csv(cached_download("books.csv"), keep_default_na=False).iloc[:CATALOG_SIZE].copy()
    tags = pd.read_csv(cached_download("tags.csv"))
    book_tags = pd.read_csv(cached_download("book_tags.csv"))
    license_text = cached_download("LICENSE").read_text(encoding="utf-8")
    ratings = pd.read_csv(source_ratings(), dtype={"user_id": "int32", "book_id": "int32", "rating": "int8"})
    if len(ratings) != SOURCE_ROWS:
        raise ValueError("Pinned full ratings row count mismatch; remove cache and retry")
    ratings["source_row"] = np.arange(len(ratings), dtype=np.int32)
    ratings = ratings[ratings.book_id <= CATALOG_SIZE].copy()
    if ratings.duplicated(["user_id", "book_id"]).any() or not ratings.rating.between(1, 5).all():
        raise ValueError("Ratings contract failed")
    tag_names = dict(zip(tags.tag_id, tags.tag_name, strict=True))
    grouped_tags = book_tags[book_tags.goodreads_book_id.isin(books.goodreads_book_id)].groupby(
        "goodreads_book_id"
    )
    catalog: list[dict[str, Any]] = []
    for row in books.to_dict(orient="records"):
        tag_rows = grouped_tags.get_group(row["goodreads_book_id"]).sort_values("count", ascending=False)
        relevant = [
            (tag_names[tag["tag_id"]], int(tag["count"]))
            for tag in tag_rows.to_dict(orient="records")
            if tag_names[tag["tag_id"]] in GENRES
        ]
        genre = GENRES[relevant[0][0]] if relevant else "Fiction"
        try:
            year = int(float(row["original_publication_year"]))
        except (ValueError, TypeError):
            year = None
        catalog.append(
            {
                "id": int(row["book_id"]),
                "title": row["title"],
                "author": row["authors"],
                "year": year,
                "genre": genre,
                "tags": list(dict.fromkeys(name for name, _ in relevant))[:6],
                "popularity": 0,
                "rating": float(row["average_rating"]),
            }
        )
    target = ROOT / "data" / "goodbooks"
    target.mkdir(parents=True, exist_ok=True)
    (target / "LICENSE").write_text(license_text, encoding="utf-8")
    ratings.to_parquet(target / "ratings.parquet", index=False)
    pd.DataFrame(catalog).to_parquet(target / "catalog.parquet", index=False)
    book_tags[book_tags.goodreads_book_id.isin(books.goodreads_book_id)].merge(tags, on="tag_id").to_parquet(
        target / "book_tags.parquet", index=False
    )
    cached = [
        source_ratings(),
        *(cached_download(name) for name in ("books.csv", "tags.csv", "book_tags.csv", "LICENSE")),
    ]
    provenance = {
        "dataset": "goodbooks-10k",
        "author": "Zygmunt Zajac and contributors",
        "source": "https://github.com/zygmuntz/goodbooks-10k",
        "commit": COMMIT,
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "transformations": [
            f"All {SOURCE_ROWS} source rows",
            f"Book IDs 1 through {CATALOG_SIZE}",
            "Discard external cover URLs",
            "Map reader tags to named genre families",
        ],
        "files": [
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in cached
        ],
    }
    write_json(target / "provenance.json", provenance)
    (target / "PROVENANCE.md").write_text(
        "# goodbooks-10k derived data\n\nSource: https://github.com/zygmuntz/goodbooks-10k\n\n"
        f"Pinned commit: `{COMMIT}`. Original author: Zygmunt Zajac and contributors. "
        "The original and these derived data are licensed CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/. "
        "File hashes and transformations are in provenance.json. Download is automated by scripts/build_evidence.py. "
        "The complete upstream ratings file is ingested and hashed without rewriting. "
        "No cover URL is retained. Source rows lack timestamps. Snapshot average ratings appear only as catalog metadata, never in model features.\n",
        encoding="utf-8",
    )
    return books, ratings, catalog


def eligible_users(matrix: csr_matrix, truth: dict[int, set[int]], cold: bool = False) -> np.ndarray:
    counts = np.diff(matrix.indptr)
    eligible = np.array([uid for uid in truth if (counts[uid] < 5 if cold else counts[uid] >= 5)], dtype=int)
    size = min(1000 if cold else EVALUATION_USERS, eligible.size)
    return (
        np.sort(np.random.default_rng(SEED).choice(eligible, size=size, replace=False)) if size else eligible
    )


def evaluate(
    bundle: ModelBundle,
    seen: dict[int, set[int]],
    truth: dict[int, set[int]],
    users: np.ndarray,
    genres: np.ndarray,
    sampled: bool = False,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    records: dict[str, dict[str, list[float]]] = {name: {} for name in LABELS}
    lists: dict[str, list[np.ndarray]] = {name: [] for name in LABELS}
    long_tail = np.argsort(bundle.popularity, kind="stable")[: int(CATALOG_SIZE * 0.8)]
    probabilities = (bundle.popularity + 1) / (bundle.popularity.sum() + CATALOG_SIZE)
    for position, uid in enumerate(users):
        if position and position % 1000 == 0:
            print(
                f"Evaluated {position}/{len(users)} readers ({'sampled' if sampled else 'full catalog'})",
                flush=True,
            )
        candidates = None
        if sampled:
            pool = np.setdiff1d(np.arange(CATALOG_SIZE), list(seen.get(int(uid), set()) | truth[int(uid)]))
            negatives = np.random.default_rng(61 + int(uid)).choice(
                pool, size=min(100, len(pool)), replace=False
            )
            candidates = np.concatenate([np.array(sorted(truth[int(uid)])), negatives])
        for name, scores in bundle.score(int(uid)).items():
            ranked = rank_unseen(scores, seen.get(int(uid), set()), candidates)
            row = ranking_metrics(ranked, truth[int(uid)])
            top = ranked[:10]
            row["long_tail_share"] = float(np.isin(top, long_tail).mean())
            row["novelty"] = float(-np.log2(probabilities[top]).mean())
            row["diversity"] = intra_list_diversity(bundle.cosine, top)
            history_dist = (
                np.bincount(genres[bundle.matrix[int(uid)].indices], minlength=9).astype(float) + 0.5
            )
            top_dist = np.bincount(genres[top], minlength=9).astype(float) + 0.5
            row["calibration"] = float(jensenshannon(history_dist, top_dist, base=2) ** 2)
            for metric, value in row.items():
                records[name].setdefault(metric, []).append(value)
            lists[name].append(top)
    return {
        name: {key: np.array(values) for key, values in metrics.items()} for name, metrics in records.items()
    }, {name: np.array(values) for name, values in lists.items()}


def coverage_interval(lists: np.ndarray) -> dict[str, float | str]:
    rng = np.random.default_rng(42)
    estimates = [
        len(np.unique(lists[rng.integers(0, len(lists), size=len(lists))])) / CATALOG_SIZE
        for _ in range(1000)
    ]
    # Conditional resampling variability is distinct from population confidence.
    return {
        "mean": len(np.unique(lists)) / CATALOG_SIZE,
        "expected_resampled_coverage": float(np.mean(estimates)),
        "low": float(np.quantile(estimates, 0.025)),
        "high": float(np.quantile(estimates, 0.975)),
        "interval_kind": "Observed distinct coverage with a separate 95% user-resampling range, not a confidence interval; the observed point may be outside the range",
    }


def summarize(
    records: dict[str, dict[str, np.ndarray]], lists: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    return [
        {
            "model": name,
            "label": LABELS[name],
            **{key: bootstrap(values) for key, values in metrics.items()},
            "coverage": coverage_interval(lists[name]),
            "observed_coverage": len(np.unique(lists[name])) / CATALOG_SIZE,
        }
        for name, metrics in records.items()
    ]


def independent_difference(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(18)
    samples = right[rng.integers(0, len(right), size=(1000, len(right)))].mean(axis=1) - left[
        rng.integers(0, len(left), size=(1000, len(left)))
    ].mean(axis=1)
    return {
        "mean": float(right.mean() - left.mean()),
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
    }


def evaluate_reranking(
    bundle: ModelBundle,
    catalog: list[dict[str, Any]],
    seen: dict[int, set[int]],
    truth: dict[int, set[int]],
    users: np.ndarray,
) -> dict[str, object]:
    from packages.rerank import apply_rules, calibrate, calibration_divergence, diversity, mmr

    chosen_users = np.sort(np.random.default_rng(914).choice(users, size=min(100, len(users)), replace=False))
    stage_records: dict[str, dict[str, list[float]]] = {
        name: {}
        for name in (
            "Popularity baseline",
            "Blend before author rules",
            "Author rules",
            "MMR diversity",
            "Genre calibration",
        )
    }
    long_tail = np.argsort(bundle.popularity, kind="stable")[: int(CATALOG_SIZE * 0.8)]
    for uid in chosen_users:
        scored = bundle.score(int(uid))
        unseen = seen.get(int(uid), set())
        blend_ids = rank_unseen(scored["blend"], unseen)[:200]
        candidates = [{**catalog[int(item)], "score": float(scored["blend"][item])} for item in blend_ids]
        history = [catalog[int(item)] for item in bundle.matrix[int(uid)].indices]
        ruled = apply_rules(candidates, {str(item + 1) for item in unseen})
        diverse = mmr(ruled, 10, 0.25)
        calibrated = calibrate(diverse, ruled[:60], history, 0.2)
        stages = {
            "Popularity baseline": [
                catalog[int(item)] for item in rank_unseen(scored["popularity"], unseen)[:10]
            ],
            "Blend before author rules": candidates[:10],
            "Author rules": ruled[:10],
            "MMR diversity": diverse,
            "Genre calibration": calibrated,
        }
        for stage, items in stages.items():
            item_ids = np.array([int(item["id"]) - 1 for item in items])
            metrics = ranking_metrics(item_ids, truth[int(uid)])
            metrics.pop("recall200")
            metrics["tag_diversity"] = diversity(items)
            metrics["genre_divergence"] = calibration_divergence(items, history)
            metrics["long_tail_share"] = float(np.isin(item_ids, long_tail).mean())
            for metric, value in metrics.items():
                stage_records[stage].setdefault(metric, []).append(value)
    return {
        "status": "measured",
        "users": len(chosen_users),
        "source": "packages.rerank serving implementation",
        "detail": "100-reader seeded subset; top200 blend candidates, author cap2, MMR weight0.25, calibration weight0.2. Tag Jaccard diversity and unsmoothed natural-log Jensen-Shannon differ from the main table's cosine diversity and base2-smoothed calibration. No exploration slot is evaluated.",
        "rows": [
            {"stage": stage, **{metric: bootstrap(np.array(values)) for metric, values in metrics.items()}}
            for stage, metrics in stage_records.items()
        ],
    }


def analyze_cohorts(
    bundle: ModelBundle,
    seen: dict[int, set[int]],
    truth: dict[int, set[int]],
    warm: np.ndarray,
    cold: np.ndarray,
) -> dict[str, object]:
    rows = []
    strata = []
    for name, users in (("Warm: at least5 training positives", warm), ("Cold: 0-4 training positives", cold)):
        if not len(users):
            continue
        counts = np.array([len(truth[int(uid)]) for uid in users])
        training_counts = np.array([bundle.matrix[int(uid)].nnz for uid in users])
        target_popularity = np.array([np.mean(bundle.popularity[list(truth[int(uid)])]) for uid in users])
        baseline = np.array(
            [
                ranking_metrics(rank_unseen(bundle.popularity, seen.get(int(uid), set())), truth[int(uid)])[
                    "ndcg"
                ]
                for uid in users
            ]
        )
        rows.append(
            {
                "cohort": name,
                "users": len(users),
                "training_positives": bootstrap(training_counts),
                "holdout_positives": bootstrap(counts),
                "target_training_popularity": bootstrap(target_popularity),
                "popularity_ndcg": bootstrap(baseline),
                "no_training_history_share": bootstrap((training_counts == 0).astype(float)),
                "single_holdout_positive_share": bootstrap((counts == 1).astype(float)),
            }
        )
        for label, mask in (("1", counts == 1), ("2-5", (counts >= 2) & (counts <= 5)), ("6+", counts >= 6)):
            if mask.any():
                strata.append(
                    {
                        "cohort": name,
                        "holdout_positive_count": label,
                        "users": int(mask.sum()),
                        "popularity_ndcg": bootstrap(baseline[mask]),
                        "target_training_popularity": bootstrap(target_popularity[mask]),
                    }
                )
    return {
        "status": "measured",
        "cohorts": rows,
        "strata": strata,
        "explanation": "Cold and warm readers are different held-out cohorts. NDCG normalizes by the number of relevant holdout items up to10, while target-book popularity and the unseen candidate sets also differ. Compare the measured positive counts, target popularity, and within-count strata here; a higher cold score does not show that lacking history improves a reader's recommendations.",
    }


def export_edge(
    bundle: ModelBundle, user_ids: np.ndarray, users: np.ndarray, seen: csr_matrix, public: Path
) -> dict[str, object]:
    public.mkdir(parents=True, exist_ok=True)
    arrays = {
        "item-factors.f32": (bundle.item_factors, "<f4"),
        "popularity.f32": (bundle.popularity, "<f4"),
        "cosine-data.f32": (bundle.cosine.data, "<f4"),
        "cosine-indices.u32": (bundle.cosine.indices, "<u4"),
        "cosine-indptr.u32": (bundle.cosine.indptr, "<u4"),
    }
    for name, (array, dtype) in arrays.items():
        (public / name).write_bytes(array.astype(dtype).tobytes(order="C"))
    profiles = [
        {
            "id": int(user_ids[uid]),
            "positive_indices": bundle.matrix[int(uid)].indices.tolist(),
            "seen_indices": seen[int(uid)].indices.tolist(),
            "user_factor": bundle.user_factors[int(uid)].tolist(),
        }
        for uid in users
    ]
    write_json(public / "edge-profiles.json", profiles)
    hashes = {
        name: hashlib.sha256((public / name).read_bytes()).hexdigest()
        for name in (*arrays, "edge-profiles.json")
    }
    version = "goodbooks-full-" + hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    manifest = {
        "schema_version": "1",
        "artifact_version": version,
        "catalog_size": CATALOG_SIZE,
        "factors": bundle.item_factors.shape[1],
        "alpha": 20,
        "regularization": 0.1,
        "training_cutoff_row": CUTOFF,
        "source_rows": SOURCE_ROWS,
        "files": {
            "item_factors": "item-factors.f32",
            "cosine_data": "cosine-data.f32",
            "cosine_indices": "cosine-indices.u32",
            "cosine_indptr": "cosine-indptr.u32",
            "popularity": "popularity.f32",
            "profiles": "edge-profiles.json",
            "catalog": "catalog.json",
        },
        "sha256": hashes,
        "models": ["popularity", "item_cosine", "als", "blend"],
        "blend_weights": {"item_cosine": 0.55, "als": 0.35, "popularity": 0.10},
        "indexing": "Zero-based contiguous catalog index = book_id minus1; little-endian row-major arrays",
        "cosine_neighbors": 100,
    }
    fixtures = []
    for uid in users:
        scores = bundle.score(int(uid))
        fixture: dict[str, Any] = {"id": int(user_ids[uid]), "models": {}}
        for name in manifest["models"]:
            ids = rank_unseen(scores[name], seen[int(uid)].indices)[:20]
            fixture["models"][name] = [
                {"id": int(index + 1), "score": float(scores[name][index])} for index in ids
            ]
        fixtures.append(fixture)
    write_json(public / "edge-manifest.json", manifest)
    write_json(
        public / "edge-parity.json",
        {"artifact_version": version, "absolute_score_tolerance": 0.00002, "readers": fixtures},
    )
    write_json(ROOT / "results" / "serving-manifest.json", manifest)
    return manifest


def main() -> None:
    started = time.perf_counter()
    _, ratings, catalog = ingest()
    user_ids = np.sort(ratings.user_id.unique())
    user_map = {int(uid): index for index, uid in enumerate(user_ids)}
    train_mask = (ratings.source_row < CUTOFF).to_numpy()
    matrix, seen, truth = matrices(ratings, train_mask, user_map, CATALOG_SIZE)
    users = eligible_users(matrix, truth)
    if len(users) < 30:
        raise ValueError("Insufficient eligible readers to publish bootstrap evidence")
    genre_names = sorted({str(item["genre"]) for item in catalog})
    genres = np.array([genre_names.index(str(item["genre"])) for item in catalog])
    print(
        f"Training {matrix.shape[0]} readers, {matrix.shape[1]} books, {matrix.nnz} positives; evaluating {len(users)} readers",
        flush=True,
    )
    content_vectors = build_vectors(catalog)
    print("Training ranker retrieval prefix through source fraction0.60; ranker labels0.60-0.80", flush=True)
    prefix = ratings[train_mask]
    ranker_train_mask = (prefix.source_row < int(SOURCE_ROWS * 0.6)).to_numpy()
    rank_matrix, rank_seen, rank_truth = matrices(prefix, ranker_train_mask, user_map, CATALOG_SIZE)
    rank_bundle = train_models(rank_matrix)
    ranker, ranker_training = train_ranker(rank_bundle, rank_seen, rank_truth)
    ranker_training["feature_cutoff_row"] = int(SOURCE_ROWS * 0.6)
    ranker_training["label_end_row"] = CUTOFF
    del rank_bundle, rank_matrix, rank_seen, rank_truth
    gc.collect()
    print("Training final retrieval models on source fraction0.80", flush=True)
    bundle = train_models(matrix, content_vectors=content_vectors)
    bundle.ranker = ranker
    main_records, main_lists = evaluate(bundle, seen, truth, users, genres)
    sampled_records, _ = evaluate(bundle, seen, truth, users, genres, sampled=True)
    comparisons = paired_comparisons({name: records["ndcg"] for name, records in main_records.items()})
    observed = max(main_records, key=lambda name: main_records[name]["ndcg"].mean())
    corrected: str | None = observed
    for comparison in comparisons:
        if observed in (comparison["left"], comparison["right"]) and not comparison["significant"]:
            corrected = None
    print("Training random-split shortcut models", flush=True)
    random_mask = np.random.default_rng(93).random(len(ratings)) < 0.8
    random_matrix, random_seen, random_truth = matrices(ratings, random_mask, user_map, CATALOG_SIZE)
    random_users = eligible_users(random_matrix, random_truth)
    random_prefix = ratings[random_mask]
    random_rank_mask = np.random.default_rng(94).random(len(random_prefix)) < 0.75
    random_rank_matrix, random_rank_seen, random_rank_truth = matrices(
        random_prefix, random_rank_mask, user_map, CATALOG_SIZE
    )
    random_rank_bundle = train_models(random_rank_matrix)
    random_ranker, _ = train_ranker(random_rank_bundle, random_rank_seen, random_rank_truth)
    del random_rank_bundle, random_rank_matrix, random_rank_seen, random_rank_truth
    gc.collect()
    random_bundle = train_models(random_matrix, content_vectors=content_vectors)
    random_bundle.ranker = random_ranker
    random_records, _ = evaluate(random_bundle, random_seen, random_truth, random_users, genres)
    del random_bundle, random_matrix, random_seen, random_truth, random_prefix
    gc.collect()
    cold_users = eligible_users(matrix, truth, cold=True)
    cold_slices = []
    if len(cold_users):
        cold_records, _ = evaluate(bundle, seen, truth, cold_users, genres)
        for name, records in cold_records.items():
            cold_slices.append(
                {
                    "slice": "Cold readers (0-4 positive training items)",
                    "users": len(cold_users),
                    "model": name,
                    "label": LABELS[name],
                    **{key: bootstrap(records[key]) for key in ("ndcg", "recall", "recall200")},
                }
            )
    cohort_analysis = analyze_cohorts(bundle, seen, truth, users, cold_users)
    cold_items = set(np.flatnonzero(bundle.popularity == 0))
    cold_item_users = np.array([uid for uid in users if truth[int(uid)] & cold_items])
    if len(cold_item_users):
        cold_truth = {int(uid): truth[int(uid)] & cold_items for uid in cold_item_users}
        cold_records, _ = evaluate(bundle, seen, cold_truth, cold_item_users, genres)
        for name, records in cold_records.items():
            cold_slices.append(
                {
                    "slice": "Cold item positives",
                    "users": len(cold_item_users),
                    "model": name,
                    "label": LABELS[name],
                    **{key: bootstrap(records[key]) for key in ("ndcg", "recall", "recall200")},
                }
            )
    print("Validating OPE across 200 seeds and two sample sizes", flush=True)
    ope = validate_estimators()
    print("Measuring serving re-ranking stages on a fixed reader subset", flush=True)
    reranking = evaluate_reranking(bundle, catalog, seen, truth, users)
    print("Evaluating explicit target policies on the real Open Bandit sample", flush=True)
    try:
        obd = run_obd(ROOT)
    except (OSError, ValueError, KeyError) as exc:
        obd = {
            "status": "unavailable",
            "detail": f"Real logged-data evaluation could not complete: {type(exc).__name__}: {exc}",
        }
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    per_user = []
    for name, records in main_records.items():
        for index, uid in enumerate(users):
            per_user.append(
                {
                    "user_id": int(user_ids[uid]),
                    "model": name,
                    **{metric: float(values[index]) for metric, values in records.items()},
                }
            )
    pd.DataFrame(per_user).to_parquet(results / "per-user.parquet", index=False)
    pd.DataFrame(per_user).to_csv(results / "per-user.csv", index=False)
    pd.DataFrame(
        [
            {"user_id": int(user_ids[uid]), "model": model, "book_ids": (ranked[index] + 1).tolist()}
            for model, ranked in main_lists.items()
            for index, uid in enumerate(users)
        ]
    ).to_parquet(results / "recommendation-lists.parquet", index=False)
    np.savez_compressed(
        results / "model-artifacts.npz",
        popularity=bundle.popularity,
        user_factors=bundle.user_factors,
        item_factors=bundle.item_factors,
        user_ids=user_ids,
        catalog_ids=np.arange(1, CATALOG_SIZE + 1),
    )
    save_npz(results / "cosine.npz", bundle.cosine)
    save_npz(results / "train-positive.npz", matrix.astype(np.float32))
    train_rows = ratings[train_mask]
    seen_matrix = csr_matrix(
        (
            np.ones(len(train_rows), dtype=np.uint8),
            (train_rows.user_id.map(user_map), train_rows.book_id - 1),
        ),
        shape=matrix.shape,
    )
    save_npz(results / "train-seen.npz", seen_matrix)
    save_npz(results / "content-vectors.npz", content_vectors)
    ranker.save_model(str(results / "lambdamart.txt"))
    write_json(results / "lambdamart-features.json", {"features": FEATURE_NAMES, "training": ranker_training})
    split_rows, sampled_rows = [], []
    for name, label in LABELS.items():
        base = main_records[name]["ndcg"]
        split_rows.append(
            {
                "model": name,
                "label": label,
                "full": bootstrap(base),
                "shortcut": bootstrap(random_records[name]["ndcg"]),
                "delta": independent_difference(base, random_records[name]["ndcg"]),
                "ordered_users": len(users),
                "random_users": len(random_users),
            }
        )
        sampled_rows.append(
            {
                "model": name,
                "label": label,
                "full": bootstrap(base),
                "shortcut": bootstrap(sampled_records[name]["ndcg"]),
                "delta": bootstrap(sampled_records[name]["ndcg"] - base),
                "users": len(users),
            }
        )
    timestamp = datetime.now(UTC).isoformat()
    manifest: dict[str, Any] = {
        "version": "0.1.0",
        "generated_at": timestamp,
        "statement": STATEMENT,
        "dataset": {
            "name": "goodbooks-10k complete catalog and ratings",
            "backend": "pinned public CSV -> parquet -> NumPy/SciPy exact scoring",
            "source": "https://github.com/zygmuntz/goodbooks-10k",
            "source_commit": COMMIT,
            "license": "CC BY-SA 4.0",
            "source_rows": SOURCE_ROWS,
            "catalog_size": CATALOG_SIZE,
            "interactions": len(ratings),
            "train_interactions": int(train_mask.sum()),
            "test_interactions": int((~train_mask).sum()),
            "positive_training_interactions": int(matrix.nnz),
            "readers": len(user_ids),
            "evaluated_users": len(users),
        },
        "protocol": {
            "name": "Global source-order holdout, full catalog",
            "split": "80/20 source-file order; timestamps unavailable",
            "cutoff_row": CUTOFF,
            "candidate_count": CATALOG_SIZE,
            "bootstrap_draws": 1000,
            "confidence": 0.95,
            "seed": SEED,
            "positive_threshold": 4,
            "description": "All unseen catalog items ranked. Training ratings excluded. Paired user bootstrap and Benjamini-Hochberg correction. Source order is a proxy, not a date-validated temporal split.",
            "diversity_definition": "Symmetric sparse collaborative pair dissimilarity: one minus the average of both directed top100-pruned cosine edges for each unordered pair. Missing pruned edges count as zero. This is not complete unpruned cosine dissimilarity.",
        },
        "metrics": summarize(main_records, main_lists),
        "comparisons": comparisons,
        "winner": {"observed": observed, "corrected": corrected},
        "inflation": {
            "split": split_rows,
            "sampled": sampled_rows,
            "ordered_winner": observed,
            "random_winner": max(random_records, key=lambda name: random_records[name]["ndcg"].mean()),
            "sampled_winner": max(sampled_records, key=lambda name: sampled_records[name]["ndcg"].mean()),
            "note": "Random-split rows also change the eligible cohort. Sampled negatives use identical held-out users and positives. A protocol difference is not proof of online gain or a causal leakage estimate.",
        },
        "cold_slices": cold_slices,
        "cold_reader_analysis": cohort_analysis,
        "ranker_training": ranker_training,
        "reranking": reranking,
        "cold_items": {
            "catalog_items": len(cold_items),
            "eligible_readers": len(cold_item_users),
            "status": "measured"
            if len(cold_item_users)
            else "unavailable: no eligible cold-item positives in the bounded holdout",
        },
        "ope": {
            "simulator": ope,
            "crosscheck": {
                "status": "not_run",
                "detail": "Published-library compatibility is not assumed. See modules for available validation.",
            },
            "open_bandit": obd,
            "demo_logs": {
                "status": "not_estimated",
                "detail": "No meaningful online feedback sample is available at build time; no causal or online benefit claim is made.",
            },
        },
        "modules": [
            {
                "name": "Popularity / cosine / implicit ALS / blend",
                "status": "measured",
                "detail": "Train-only scorers with exact full-catalog offline evaluation and persisted factors.",
            },
            {
                "name": "LambdaMART learned ranker",
                "status": "measured",
                "detail": "Learned over a retrieval candidate union using an earlier source-order window. Evaluated against all baselines over the complete unseen catalog.",
            },
            {
                "name": "Two-tower neural retrieval",
                "status": "not_implemented",
                "detail": "Sparse collaborative and TF-IDF content retrieval are implemented; no neural retrieval result is claimed.",
            },
            {
                "name": "pgvector / approximate nearest neighbors",
                "status": "not_implemented",
                "detail": "Exact NumPy scoring is used. No approximate recall result is claimed.",
            },
            {
                "name": "Content TF-IDF path",
                "status": "measured",
                "detail": "Normalized title, author and tag TF-IDF profiles score all books including training-cold items. Uses undated metadata; no historical metadata availability claim.",
            },
            {
                "name": "OPE simulator",
                "status": "measured",
                "detail": "Local IPS, SNIPS, DM, DR with 200 seeds, two sample sizes and oracle/misspecified reward models.",
            },
            {"name": "Real logged-bandit OPE", "status": obd["status"], "detail": obd["detail"]},
            {
                "name": "Verified temporal features",
                "status": "unavailable",
                "detail": "Source ratings lack timestamps. Source-order train-only features are tested; metadata is a snapshot.",
            },
        ],
        "limitations": [
            "goodbooks has no interaction timestamps; source-order holdout is not verified temporal evaluation.",
            "All source ratings and all10000 catalog items are ingested; headline metrics use a seeded sample of at most5000 eligible readers, not every reader.",
            "Only readers with at least five positive training items enter headline evaluation; cold readers are separate.",
            "Snapshot genres and average ratings are descriptive metadata. Their historical availability cannot be verified.",
            "LambdaMART trains on a candidate union from an earlier source-order window and scores the full unseen catalog; training negatives have a different distribution from full-catalog inference.",
            "Coverage.mean is observed distinct catalog coverage; low/high show a separate conditional user-resampling range, not a confidence interval. The observed point can exceed this range because resampling cannot invent unseen items. expected_resampled_coverage records the average resampled value.",
            "The random-split comparison uses a different eligible cohort, so it is not an isolated estimate of temporal leakage.",
            "Simulator reward models are oracle or deliberately misspecified. This is estimator validation, not reward-model learning.",
            "Contextual BTS target-policy truth, online impact, neural models, approximate indexing, and historical point-in-time metadata remain unmeasured.",
        ],
        "run_seconds": round(time.perf_counter() - started, 3),
    }
    crosscheck_path = results / "obp-crosscheck.json"
    if crosscheck_path.exists():
        manifest["ope"]["crosscheck"] = json.loads(crosscheck_path.read_text(encoding="utf-8"))
    for book in catalog:
        book["popularity"] = int(bundle.popularity[int(book["id"]) - 1])
    pd.DataFrame(catalog).to_parquet(ROOT / "data" / "goodbooks" / "catalog.parquet", index=False)
    similarity = {}
    for index in range(CATALOG_SIZE):
        row = bundle.cosine[index]
        neighbors = row.indices[np.argsort(-row.data, kind="stable")[:20]]
        similarity[str(index + 1)] = [
            {"id": int(other + 1), "score": round(float(bundle.cosine[index, other]), 7)}
            for other in neighbors
            if other != index and bundle.cosine[index, other] > 0
        ]
    training_positives = ratings[train_mask & (ratings.rating >= 4).to_numpy()].sort_values("source_row")
    readers = [
        {
            "id": int(user_ids[uid]),
            "history": [
                int(item)
                for item in training_positives[training_positives.user_id == user_ids[uid]].book_id.tail(12)
            ],
        }
        for uid in users[:24]
    ]
    public = ROOT / "web" / "public" / "data"
    edge_manifest = export_edge(bundle, user_ids, users[:24], seen_matrix, public)
    manifest["serving_artifacts"] = edge_manifest
    for name, value in (
        ("catalog.json", catalog),
        ("readers.json", readers),
        ("similarity.json", similarity),
        ("evidence.json", manifest),
    ):
        if name in {"catalog.json", "similarity.json"}:
            (public / name).write_text(
                json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
            )
        else:
            write_json(public / name, value)
    (public / "LICENSE-goodbooks.txt").write_text(
        (ROOT / "data" / "goodbooks" / "LICENSE").read_text(encoding="utf-8")
        + "\nDerived catalog, reader histories, similarities, and goodbooks evaluation results: Zygmunt Zajac and contributors; adaptations by stacks.\nSource: https://github.com/zygmuntz/goodbooks-10k\n",
        encoding="utf-8",
    )
    (results / "LICENSE-DATA.txt").write_text(
        "goodbooks-derived data and metrics: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/. Original source: Zygmunt Zajac and contributors, https://github.com/zygmuntz/goodbooks-10k. Adaptations: stacks. Code and synthetic simulator output use the repository code license.\n",
        encoding="utf-8",
    )
    write_json(results / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "seconds": manifest["run_seconds"],
                "winner": manifest["winner"],
                "metrics": [{"model": row["model"], "ndcg": row["ndcg"]} for row in manifest["metrics"]],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    if "--export-only" in sys.argv:
        from packages.retrieval.models import load_bundle

        stored = load_bundle(ROOT / "results")
        assert stored.user_ids is not None and stored.seen is not None
        public = ROOT / "web" / "public" / "data"
        reader_ids = [
            reader["id"] for reader in json.loads((public / "readers.json").read_text(encoding="utf-8"))
        ]
        chosen = np.array([int(np.flatnonzero(stored.user_ids == reader_id)[0]) for reader_id in reader_ids])
        edge = export_edge(stored, stored.user_ids, chosen, stored.seen, public)
        similarity_path = public / "similarity.json"
        if similarity_path.exists():
            similarity = json.loads(similarity_path.read_text(encoding="utf-8"))
            similarity_path.write_text(
                json.dumps({key: value[:20] for key, value in similarity.items()}, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
        evidence = json.loads((ROOT / "results" / "manifest.json").read_text(encoding="utf-8"))
        evidence["serving_artifacts"] = edge
        write_json(ROOT / "results" / "manifest.json", evidence)
        write_json(public / "evidence.json", evidence)
        print("Exported edge parity for all reader profiles", flush=True)
    else:
        main()
