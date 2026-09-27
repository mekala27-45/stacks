"""Load published bundle data and score an honest, inspectable model blend."""

import json
from math import log1p
from pathlib import Path
from typing import Any


class Catalog:
    def __init__(self, directory: Path):
        with (directory / "catalog.json").open(encoding="utf-8") as stream:
            items = json.load(stream)
        if not isinstance(items, list) or not items:
            raise ValueError("catalog.json must contain a nonempty item list")
        self.items: dict[str, dict[str, Any]] = {}
        for item in items:
            if not all(key in item for key in ("id", "title", "author", "genre", "tags", "popularity")):
                raise ValueError("Every catalog item requires id, title, author, genre, tags and popularity")
            normalized = {**item, "id": str(item["id"])}
            if normalized["id"] in self.items:
                raise ValueError("Catalog item IDs must be unique")
            self.items[normalized["id"]] = normalized
        with (directory / "similarity.json").open(encoding="utf-8") as stream:
            similarities = json.load(stream)
        self.similarities = {
            str(key): {str(row["id"]): float(row["score"]) for row in rows if str(row["id"]) in self.items}
            for key, rows in similarities.items()
        }
        with (directory / "readers.json").open(encoding="utf-8") as stream:
            readers = json.load(stream)
        self.readers = {
            str(reader["id"]): [str(item) for item in reader["history"] if str(item) in self.items]
            for reader in readers
        }
        self.popularity_max = (
            max(log1p(max(float(item["popularity"]), 0.0)) for item in self.items.values()) or 1.0
        )

    def score(self, history: list[str], baseline: bool = False) -> list[dict[str, Any]]:
        result = []
        recent = list(reversed(history[-30:]))
        total_weight = sum(0.85**i for i in range(len(recent))) or 1.0
        for item in self.items.values():
            popularity = log1p(max(float(item["popularity"]), 0.0)) / self.popularity_max
            contributions = [
                (self.similarities.get(old, {}).get(item["id"], 0.0) * 0.85**i, old)
                for i, old in enumerate(recent)
            ]
            cosine = sum(value for value, _ in contributions) / total_weight
            strongest = max(contributions, default=(0.0, ""))
            score = popularity if baseline or not recent else 0.3 * popularity + 0.7 * cosine
            source = self.items.get(strongest[1], {}) if strongest[0] > 0 else {}
            explanation = (
                f"Connected to {source['title']} in the public reading sample; blended with training popularity."
                if source and not baseline
                else "Selected by popularity in the public training sample."
            )
            result.append(
                {
                    **item,
                    "score": score,
                    "explanation": explanation,
                    "scores": {"popularity": popularity, "item_cosine": cosine},
                    "source_item_id": source.get("id"),
                }
            )
        return sorted(result, key=lambda item: (-item["score"], item["id"]))
