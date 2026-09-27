"""Render complete evidence-backed documents, or fail on any generated-file drift."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "ndcg", "recall", "recall200", "hit_rate", "coverage", "novelty",
    "long_tail_share", "diversity", "calibration",
)


class EvidenceError(ValueError):
    """Evidence cannot substantiate a complete publication."""


def _number(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EvidenceError(f"{field} must be a finite number")


def _interval(value: Any, field: str) -> None:
    if not isinstance(value, dict):
        raise EvidenceError(f"{field} must be an estimate with an interval")
    for key in ("mean", "low", "high"):
        _number(value.get(key), f"{field}.{key}")
    if value["low"] > value["high"]:
        raise EvidenceError(f"{field} has reversed interval bounds")


def validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict) or not manifest:
        raise EvidenceError("Refusing to publish empty evidence")
    for name in ("dataset", "protocol", "winner", "inflation", "ope"):
        if not isinstance(manifest.get(name), dict) or not manifest[name]:
            raise EvidenceError(f"Missing evidence section: {name}")
    for name in ("metrics", "modules", "limitations"):
        if not isinstance(manifest.get(name), list) or not manifest[name]:
            raise EvidenceError(f"Missing nonempty evidence list: {name}")
    for name in ("statement", "generated_at"):
        if not isinstance(manifest.get(name), str) or not manifest[name].strip():
            raise EvidenceError(f"Missing publication field: {name}")
    for name in ("catalog_size", "source_rows", "interactions", "train_interactions", "test_interactions", "evaluated_users"):
        value = manifest["dataset"].get(name)
        _number(value, f"dataset.{name}")
        if value <= 0:
            raise EvidenceError(f"dataset.{name} must be positive")
    names: set[str] = set()
    for row in manifest["metrics"]:
        name = row.get("model")
        if not isinstance(name, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) is None:
            raise EvidenceError("Model identifiers must be safe lowercase file names")
        if name in names:
            raise EvidenceError(f"Duplicate model: {name}")
        names.add(name)
        if not isinstance(row.get("label"), str) or not row["label"].strip():
            raise EvidenceError(f"Missing label for {name}")
        for metric in METRICS:
            _interval(row.get(metric), f"metrics.{name}.{metric}")
    if "popularity" not in names:
        raise EvidenceError("A popularity baseline is required")
    for row in manifest.get("comparisons", []):
        _interval(row.get("difference"), "comparison.difference")
        for field in ("p", "q"):
            _number(row.get(field), f"comparison.{field}")
            if not 0 <= row[field] <= 1:
                raise EvidenceError(f"comparison.{field} must be a probability")
    for section in ("simulator", "crosscheck", "open_bandit", "demo_logs"):
        if not isinstance(manifest["ope"].get(section), dict):
            raise EvidenceError(f"Missing OPE evidence: {section}")
        if not manifest["ope"][section].get("status"):
            raise EvidenceError(f"Missing OPE status: {section}")
    return manifest


def interval(value: dict[str, Any], digits: int = 4) -> str:
    formatted = f"{value['mean']:.{digits}f} [{value['low']:.{digits}f}, {value['high']:.{digits}f}]"
    if "interval_kind" in value:
        formatted += " (observed coverage; bracket is a reader-resampling range, not a confidence interval)"
    return formatted


def number(value: float, digits: int = 4) -> str:
    return f"{value:,.{digits}f}"


def publication_context(root: Path) -> dict[str, dict[str, Any]]:
    """Load separately verified operational evidence without changing model evidence."""
    context: dict[str, dict[str, Any]] = {"hosting": {}, "latency": {}}
    for key, name in (("hosting", "deployment-verification.json"), ("latency", "latency.json")):
        path = root / "results" / name
        if not path.exists():
            continue
        try:
            artifact = json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError) as error:
            raise EvidenceError(f"Invalid operational evidence: {name}") from error
        if not isinstance(artifact, dict) or not artifact.get("status"):
            raise EvidenceError(f"Missing operational evidence status: {name}")
        context[key] = artifact
    hosting = context["hosting"]
    if hosting:
        hosting["verified_for_publication"] = (
            hosting.get("status") == "passed" and hosting.get("independent_d1_read_verified") is True
        )
        if hosting["verified_for_publication"]:
            checks = hosting.get("checks")
            if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
                raise EvidenceError("Verified deployment requires a nonempty set of passing checks")
            if not isinstance(hosting.get("url"), str) or not hosting["url"].startswith("https://"):
                raise EvidenceError("Verified deployment requires its public HTTPS URL")
    latency = context["latency"]
    if latency:
        for key in ("requests", "warmup_requests", "concurrency", "p50_ms", "p99_ms"):
            _number(latency.get(key), f"latency.{key}")
        if latency["requests"] <= 0 or latency["concurrency"] <= 0:
            raise EvidenceError("Latency evidence requires a nonempty request population")
        for key in ("p50", "p99"):
            _interval(latency.get(key), f"latency.{key}")
    return context


def build_outputs(manifest: dict[str, Any], root: Path = ROOT) -> dict[Path, str]:
    validate_manifest(manifest)
    environment = Environment(
        loader=FileSystemLoader(root / "report" / "templates"),
        undefined=StrictUndefined,
        autoescape=lambda name: bool(name and ".html" in name),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters.update(estimate=interval, number=number)
    common = {
        "e": manifest,
        "baseline": next(row for row in manifest["metrics"] if row["model"] == "popularity"),
        **publication_context(root),
    }
    outputs: dict[Path, str] = {}
    specs = (
        ("README.md.j2", "README.md"),
        ("RESULTS.md.j2", "RESULTS.md"),
        ("evaluation.md.j2", "report/evaluation.md"),
        ("index.html.j2", "report/index.html"),
    )
    for template, destination in specs:
        rendered = environment.get_template(template).render(**common).rstrip() + "\n"
        if not rendered.strip():
            raise EvidenceError(f"Empty rendered document: {destination}")
        outputs[root / destination] = rendered
    for row in manifest["metrics"]:
        card = environment.get_template("model-card.md.j2").render(model=row, **common).rstrip() + "\n"
        outputs[root / "report" / "cards" / f"{row['model']}.md"] = card
    for source, destination in (
        ("RESULTS.md", "results.md"),
        ("report/evaluation.md", "evaluation.md"),
        ("report/index.html", "evaluation.html"),
    ):
        outputs[root / "web" / "public" / "downloads" / destination] = outputs[root / source]
    for row in manifest["metrics"]:
        outputs[root / "web" / "public" / "downloads" / "cards" / f"{row['model']}.md"] = outputs[root / "report" / "cards" / f"{row['model']}.md"]
    return outputs


def render(manifest_path: Path, root: Path = ROOT, check: bool = False) -> list[str]:
    if not manifest_path.is_file() or not manifest_path.read_text(encoding="utf-8").strip():
        raise EvidenceError(f"Missing or empty manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = build_outputs(manifest, root)
    errors: list[str] = []
    for target, expected in outputs.items():
        if check:
            actual = target.read_text(encoding="utf-8") if target.exists() else ""
            if actual != expected:
                diff = "".join(difflib.unified_diff(actual.splitlines(True), expected.splitlines(True), fromfile=str(target), tofile="rendered evidence"))
                errors.append(diff or f"Missing generated file: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(expected, encoding="utf-8", newline="\n")
    for directory in (root / "report" / "cards", root / "web" / "public" / "downloads" / "cards"):
        for path in directory.glob("*.md"):
            if path not in outputs:
                errors.append(f"Unexpected stale generated card: {path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "manifest.json")
    parser.add_argument("--check", action="store_true", help="Compare complete generated files without writing")
    args = parser.parse_args()
    try:
        errors = render(args.manifest, check=args.check)
    except (EvidenceError, ValueError, KeyError, TypeError) as error:
        print(f"Evidence rendering failed: {error}")
        return 1
    if errors:
        print("\n".join(errors))
        return 1
    print("Generated evidence documents are current." if args.check else "Rendered evidence documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
