"""Download pinned public data, fit bounded models, and publish measured evidence.

Run from the project root: python scripts/build_evidence.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import hashlib
import io
import itertools
import json
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.core import STATEMENT
from packages.evaluation.metrics import bootstrap, paired_comparisons, rank_unseen, ranking_metrics
from packages.evaluation.split import matrices
from packages.ope.obd import run_obd
from packages.retrieval.models import ModelBundle, train_models
from packages.sim import validate_estimators

COMMIT = "6dd165b555a7b47b2dd36743a425776e641ff50c"
SOURCE = f"https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/{COMMIT}/"
SOURCE_ROWS = 300_000
CATALOG_SIZE = 1000
CUTOFF = 240_000
SEED = 20260926
LABELS = {
    "popularity": "Popularity",
    "item_cosine": "Item cosine",
    "als": "Implicit ALS",
    "blend": "Fixed blend",
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
    path = ROOT / ".cache" / "goodbooks" / f"ratings-first-{SOURCE_ROWS}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Streaming first {SOURCE_ROWS:,} source rows", flush=True)
        request = urllib.request.Request(
            SOURCE + "ratings.csv", headers={"User-Agent": "stacks-public-data-demo"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            reader = csv.reader(io.TextIOWrapper(response, encoding="utf-8", newline=""))
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(next(reader))
                writer.writerows(itertools.islice(reader, SOURCE_ROWS))
    return path


def ingest() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    books = pd.read_csv(cached_download("books.csv"), keep_default_na=False).iloc[:CATALOG_SIZE].copy()
    tags = pd.read_csv(cached_download("tags.csv"))
    book_tags = pd.read_csv(cached_download("book_tags.csv"))
    license_text = cached_download("LICENSE").read_text(encoding="utf-8")
    ratings = pd.read_csv(source_ratings())
    if len(ratings) != SOURCE_ROWS:
        raise ValueError("Incomplete cached ratings prefix; remove cache and retry")
    ratings["source_row"] = np.arange(len(ratings))
    ratings = ratings[ratings.book_id <= CATALOG_SIZE].copy()
    if ratings.duplicated(["user_id", "book_id"]).any() or not ratings.rating.between(1, 5).all():
        raise ValueError("Ratings contract failed")
    tag_names = dict(zip(tags.tag_id, tags.tag_name, strict=True))
    grouped_tags = book_tags[book_tags.goodreads_book_id.isin(books.goodreads_book_id)].groupby(
        "goodreads_book_id"
    )
    catalog = []
    for row in books.itertuples(index=False):
        tag_rows = grouped_tags.get_group(row.goodreads_book_id).sort_values("count", ascending=False)
        relevant = [
            (tag_names[tag.tag_id], int(tag.count))
            for tag in tag_rows.itertuples(index=False)
            if tag_names[tag.tag_id] in GENRES
        ]
        genre = GENRES[relevant[0][0]] if relevant else "Fiction"
        try:
            year = int(float(row.original_publication_year))
        except (ValueError, TypeError):
            year = None
        catalog.append(
            {
                "id": int(row.book_id),
                "title": row.title,
                "author": row.authors,
                "year": year,
                "genre": genre,
                "tags": list(dict.fromkeys(name for name, _ in relevant))[:6],
                "popularity": 0,
                "rating": float(row.average_rating),
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
            f"First {SOURCE_ROWS} source rows",
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
        "The rating prefix is parsed and rewritten as CSV before hashing; it is not the complete upstream ratings file. "
        "No cover URL is retained. Source rows lack timestamps. Snapshot average ratings appear only as catalog metadata, never in model features.\n",
        encoding="utf-8",
    )
    return books, ratings, catalog


def eligible_users(matrix: csr_matrix, truth: dict[int, set[int]], cold: bool = False) -> np.ndarray:
    counts = np.diff(matrix.indptr)
    eligible = np.array([uid for uid in truth if (counts[uid] < 5 if cold else counts[uid] >= 5)], dtype=int)
    size = min(200 if cold else 500, eligible.size)
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
    for uid in users:
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
            similarity = bundle.cosine[np.ix_(top, top)]
            row["diversity"] = float(1 - similarity[np.triu_indices(len(top), 1)].mean())
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
    catalog: list[dict[str, object]],
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
    long_tail = np.argsort(bundle.popularity, kind="stable")[:800]
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


def main() -> None:
    started = time.perf_counter()
    _, ratings, catalog = ingest()
    user_ids = np.sort(ratings.user_id.unique())
    user_map = {int(uid): index for index, uid in enumerate(user_ids)}
    train_mask = (ratings.source_row < CUTOFF).to_numpy()
    matrix, seen, truth = matrices(ratings, train_mask, user_map)
    users = eligible_users(matrix, truth)
    if len(users) < 30:
        raise ValueError("Insufficient eligible readers to publish bootstrap evidence")
    genre_names = sorted({str(item["genre"]) for item in catalog})
    genres = np.array([genre_names.index(str(item["genre"])) for item in catalog])
    print(
        f"Training {matrix.shape[0]} readers, {matrix.shape[1]} books, {matrix.nnz} positives; evaluating {len(users)} readers",
        flush=True,
    )
    bundle = train_models(matrix)
    main_records, main_lists = evaluate(bundle, seen, truth, users, genres)
    sampled_records, _ = evaluate(bundle, seen, truth, users, genres, sampled=True)
    comparisons = paired_comparisons({name: records["ndcg"] for name, records in main_records.items()})
    observed = max(main_records, key=lambda name: main_records[name]["ndcg"].mean())
    corrected = observed
    for comparison in comparisons:
        if observed in (comparison["left"], comparison["right"]) and not comparison["significant"]:
            corrected = None
    print("Training random-split shortcut models", flush=True)
    random_mask = np.random.default_rng(93).random(len(ratings)) < 0.8
    random_matrix, random_seen, random_truth = matrices(ratings, random_mask, user_map)
    random_users = eligible_users(random_matrix, random_truth)
    random_bundle = train_models(random_matrix)
    random_records, _ = evaluate(random_bundle, random_seen, random_truth, random_users, genres)
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
    np.savez_compressed(
        results / "model-artifacts.npz",
        popularity=bundle.popularity,
        cosine=bundle.cosine,
        user_factors=bundle.user_factors,
        item_factors=bundle.item_factors,
        user_ids=user_ids,
    )
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
    manifest = {
        "version": "0.1.0",
        "generated_at": timestamp,
        "statement": STATEMENT,
        "dataset": {
            "name": "goodbooks-10k bounded subset",
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
                "name": "Two-tower model and LambdaMART",
                "status": "not_implemented",
                "detail": "The shipped ranking blend is fixed and does not claim learned learning-to-rank weights.",
            },
            {
                "name": "pgvector / approximate nearest neighbors",
                "status": "not_implemented",
                "detail": "Exact NumPy scoring is used. No approximate recall result is claimed.",
            },
            {
                "name": "Content and session path",
                "status": "available",
                "detail": "Interactive fallback uses stored cosine neighbors and tag overlap; no neural weights or independent content offline result.",
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
            "The first 300,000 source records and 1,000 popular catalog items form a bounded, biased demonstration subset.",
            "Only readers with at least five positive training items enter headline evaluation; cold readers are separate.",
            "Snapshot genres and average ratings are descriptive metadata. Their historical availability cannot be verified.",
            "The fixed blend is evaluated directly over the full catalog; a trained two-stage LambdaMART pipeline is not implemented.",
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
        neighbors = np.argsort(-bundle.cosine[index], kind="stable")[:40]
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
    for name, value in (
        ("catalog.json", catalog),
        ("readers.json", readers),
        ("similarity.json", similarity),
        ("evidence.json", manifest),
    ):
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
    main()
