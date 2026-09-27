"""Repair only the headline diversity statistic from frozen evaluated models."""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.evaluation.metrics import bootstrap, intra_list_diversity
from packages.retrieval.models import load_bundle
from packages.session.rnn import RecurrentModel
from scripts.build_evidence import write_json
from scripts.build_neural_evidence import NeuralBundle, histories_from


def top_unseen(scores: np.ndarray, seen: np.ndarray, k: int = 10) -> np.ndarray:
    masked = scores.copy()
    masked[seen] = -np.inf
    threshold = np.partition(masked, -k)[-k]
    candidates = np.flatnonzero(masked >= threshold)
    return candidates[np.lexsort((candidates, -masked[candidates]))][:k]


def main() -> None:
    results = ROOT / "results"
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    bundle = load_bundle(results)
    assert bundle.user_ids is not None and bundle.seen is not None
    mapping = {int(uid): index for index, uid in enumerate(bundle.user_ids)}
    records = pd.read_parquet(results / "per-user.parquet")
    readers = np.sort(records.user_id.unique())
    ratings = pd.read_parquet(ROOT / "data" / "goodbooks" / "ratings.parquet")
    mask = (ratings.source_row < manifest["protocol"]["cutoff_row"]).to_numpy()
    histories = histories_from(ratings, mask, mapping)
    tower = np.load(results / "two-tower.npz")
    recurrent = np.load(results / "recurrent.npz")
    rnn = RecurrentModel(recurrent["inputs"], recurrent["outputs"], recurrent["recurrent"], recurrent["bias"])
    neural = NeuralBundle(bundle, tower["user_vectors"], tower["item_vectors"], rnn, histories)
    values = []
    recommendation_rows = []
    for index, reader in enumerate(readers):
        uid = mapping[int(reader)]
        scores = {**bundle.score(uid), **neural.score(uid)}
        for model, score in scores.items():
            items = top_unseen(score, bundle.seen[uid].indices)
            value = intra_list_diversity(bundle.cosine, items)
            values.append({"user_id": int(reader), "model": model, "corrected_diversity": value})
            recommendation_rows.append(
                {"user_id": int(reader), "model": model, "book_ids": (items + 1).tolist()}
            )
        if index and index % 1000 == 0:
            print(f"Recomputed symmetric diversity for {index}/{len(readers)} readers", flush=True)
    corrected = records.merge(
        pd.DataFrame(values), on=["user_id", "model"], how="left", validate="one_to_one"
    )
    if corrected.corrected_diversity.isna().any():
        raise ValueError("Every recorded model-reader result must have a matching recomputation")
    corrected["diversity"] = corrected.pop("corrected_diversity")
    corrected.to_parquet(results / "per-user.parquet", index=False)
    corrected.to_csv(results / "per-user.csv", index=False)
    pd.DataFrame(recommendation_rows).to_parquet(results / "recommendation-lists.parquet", index=False)
    for row in manifest["metrics"]:
        row["diversity"] = bootstrap(corrected.loc[corrected.model == row["model"], "diversity"].to_numpy())
    manifest["protocol"]["diversity_definition"] = (
        "Symmetric sparse collaborative pair dissimilarity: one minus the average of both directed top100-pruned cosine edges for each unordered pair. Missing pruned edges count as zero. This is not complete unpruned cosine dissimilarity."
    )
    write_json(results / "manifest.json", manifest)
    write_json(ROOT / "web" / "public" / "data" / "evidence.json", manifest)
    print("Updated diversity only; accuracy, ranking comparisons and winners are unchanged", flush=True)


if __name__ == "__main__":
    main()
