"""Serving adapter over the same frozen model bundle used by evaluation."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from packages.retrieval.models import load_bundle


class Catalog:
    def __init__(self, directory: Path, artifacts: Path | None = None) -> None:
        items = json.loads((directory / "catalog.json").read_text(encoding="utf-8"))
        if not isinstance(items, list) or not items:
            raise ValueError("catalog.json must contain a nonempty item list")
        self.items: dict[str, dict[str, Any]] = {}
        for item in items:
            if not all(key in item for key in ("id", "title", "author", "genre", "tags", "popularity")):
                raise ValueError("Catalog metadata schema mismatch")
            normalized = {**item, "id": str(item["id"])}
            if normalized["id"] in self.items:
                raise ValueError("Catalog item IDs must be unique")
            self.items[normalized["id"]] = normalized
        readers = json.loads((directory / "readers.json").read_text(encoding="utf-8"))
        self.readers: dict[str, list[str]] = {
            str(reader["id"]): [str(item) for item in reader["history"] if str(item) in self.items]
            for reader in readers
        }
        default = (
            directory / "artifacts"
            if (directory / "artifacts").is_dir()
            else Path(__file__).resolve().parents[2] / "results"
        )
        self.artifacts = artifacts or Path(os.environ.get("STACKS_ARTIFACT_DIR", str(default)))
        self.bundle = load_bundle(self.artifacts)
        if self.bundle.user_ids is None or self.bundle.catalog_ids is None:
            raise ValueError("Serving artifacts require explicit user and catalog IDs")
        self.user_indices = {str(int(value)): index for index, value in enumerate(self.bundle.user_ids)}
        self.ids = [str(int(value)) for value in self.bundle.catalog_ids]
        self.indices = {value: index for index, value in enumerate(self.ids)}
        if set(self.ids) != set(self.items):
            raise ValueError("Catalog and evaluated artifacts do not identify the same items")
        manifest_path = self.artifacts / "serving-manifest.json"
        self.artifact_version = hashlib.sha256(
            (self.artifacts / "model-artifacts.npz").read_bytes()
        ).hexdigest()[:16]
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.artifact_version = str(manifest.get("artifact_version", self.artifact_version))
            for filename, expected in manifest.get("sha256", {}).items():
                base = self.artifacts if (self.artifacts / filename).is_file() else directory
                path = (base / filename).resolve()
                if (
                    not path.is_relative_to(base.resolve())
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected
                ):
                    raise ValueError("Artifact checksum mismatch")

    def neighbors(self, item_id: str) -> dict[str, float]:
        row = self.bundle.cosine.getrow(self.indices[item_id])
        return {
            self.ids[int(other)]: float(score) for other, score in zip(row.indices, row.data, strict=True)
        }

    def reader_history(self, reader_id: str | None) -> list[str]:
        if reader_id is None:
            return []
        return [self.ids[int(item)] for item in self.bundle.matrix[self.user_indices[reader_id]].indices]

    def reader_seen(self, reader_id: str | None) -> list[str]:
        if reader_id is None:
            return []
        if self.bundle.seen is None:
            raise ValueError("Serving artifacts require all-rating seen exclusions")
        return [self.ids[int(item)] for item in self.bundle.seen[self.user_indices[reader_id]].indices]

    def score(
        self, history: list[str], model: str = "blend", reader_id: str | None = None
    ) -> tuple[list[dict[str, Any]], str]:
        indexes = np.array(
            sorted({self.indices[item] for item in history if item in self.indices}), dtype=int
        )
        exact = reader_id is not None and set(history) == set(self.reader_history(reader_id))
        scores = (
            self.bundle.score(self.user_indices[reader_id])
            if exact and reader_id is not None
            else self.bundle.score_history(indexes)
        )
        if model not in scores:
            raise ValueError("Requested model has no loaded evaluated adapter")
        mode = (
            "evaluated-reader-exact"
            if exact
            else "frozen-item-factor-session-fold-in"
            if indexes.size
            else "cold-popularity"
        )
        rows = [
            {
                **self.items[item_id],
                "score": float(scores[model][index]),
                "scores": {name: float(values[index]) for name, values in scores.items()},
                "explanation": "Training-reader favorites while this demonstration has no reading history."
                if mode == "cold-popularity"
                else f"Ranked by the evaluated {model} artifacts.",
                "source_item_id": None,
            }
            for index, item_id in enumerate(self.ids)
        ]
        return sorted(rows, key=lambda row: (-row["score"], self.indices[row["id"]])), mode
