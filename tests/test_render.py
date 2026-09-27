"""The publication gate must accept evidence, detect drift, and reject absence."""

import copy
import json
import shutil
from pathlib import Path

import pytest

from scripts.render_reports import EvidenceError, METRICS, ROOT, render


@pytest.fixture
def publication(tmp_path: Path) -> tuple[Path, Path, dict]:
    shutil.copytree(ROOT / "report" / "templates", tmp_path / "report" / "templates")
    estimate = {"mean": 0.2, "low": 0.1, "high": 0.3}
    manifest = {
        "statement": "Demonstration recommendations; no real reader identity.",
        "generated_at": "2026-01-01T00:00:00Z",
        "dataset": {
            "name": "test fixture", "backend": "seeded fixture", "license": "test-only",
            "source_commit": "fixture", "catalog_size": 20, "source_rows": 100,
            "interactions": 100, "train_interactions": 80, "test_interactions": 20,
            "evaluated_users": 10,
        },
        "protocol": {
            "name": "fixture protocol", "split": "source order", "candidate_count": 20,
            "cutoff_row": 80, "bootstrap_draws": 30, "confidence": 0.95, "seed": 1,
        },
        "metrics": [{"model": "popularity", "label": "Popularity", **{key: copy.deepcopy(estimate) for key in METRICS}}],
        "comparisons": [],
        "winner": {"observed": "popularity", "corrected": None},
        "inflation": {"split": [], "sampled": []},
        "ope": {key: {"status": "not evaluated", "detail": "Test fixture only."} for key in ("simulator", "crosscheck", "open_bandit", "demo_logs")},
        "modules": [{"name": "fixture", "status": "test", "detail": "Not scientific evidence."}],
        "limitations": ["This fixture is solely for publication behavior tests."],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, path, manifest


def test_clean_render_matches_every_published_file(publication):
    root, manifest_path, _ = publication
    assert render(manifest_path, root) == []
    assert render(manifest_path, root, check=True) == []
    assert (root / "report" / "cards" / "popularity.md").is_file()
    assert (root / "web" / "public" / "downloads" / "evaluation.html").is_file()


def test_deliberate_metric_change_is_rejected_without_rewriting(publication):
    root, manifest_path, _ = publication
    render(manifest_path, root)
    target = root / "RESULTS.md"
    corrupted = target.read_text(encoding="utf-8").replace("0.2000", "0.9900", 1)
    target.write_text(corrupted, encoding="utf-8")
    errors = render(manifest_path, root, check=True)
    assert errors and "0.9900" in "\n".join(errors)
    assert target.read_text(encoding="utf-8") == corrupted


@pytest.mark.parametrize("body", ["", "{}", "null"])
def test_empty_evidence_is_refused_before_any_publication(publication, body):
    root, manifest_path, _ = publication
    manifest_path.write_text(body, encoding="utf-8")
    with pytest.raises(EvidenceError):
        render(manifest_path, root)
    assert not (root / "README.md").exists()


def test_missing_report_and_stale_model_card_are_rejected(publication):
    root, manifest_path, _ = publication
    render(manifest_path, root)
    (root / "report" / "evaluation.md").unlink()
    (root / "report" / "cards" / "retired.md").write_text("Obsolete claim", encoding="utf-8")
    errors = render(manifest_path, root, check=True)
    assert any("evaluation.md" in error for error in errors)
    assert any("retired.md" in error for error in errors)


def test_nonfinite_metrics_and_missing_baseline_are_refused(publication):
    root, manifest_path, manifest = publication
    manifest["metrics"][0]["ndcg"]["mean"] = float("nan")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(EvidenceError, match="finite"):
        render(manifest_path, root)
    manifest["metrics"][0]["ndcg"]["mean"] = 0.2
    manifest["metrics"][0]["model"] = "candidate"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(EvidenceError, match="popularity baseline"):
        render(manifest_path, root)
