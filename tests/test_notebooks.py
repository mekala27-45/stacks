"""Notebook publication must reject stale, absent and failed execution."""

import json
from pathlib import Path

import pytest

from scripts.check_notebooks import check, manifest_digest


def fixture(root: Path) -> list[Path]:
    (root / "results").mkdir()
    (root / "results/manifest.json").write_text('{"evidence": true}', encoding="utf-8")
    digest = manifest_digest(root / "results/manifest.json")
    (root / "notebooks").mkdir()
    paths = []
    for i in range(4):
        path = root / "notebooks" / f"{i}.ipynb"
        path.write_text(
            json.dumps(
                {
                    "metadata": {"stacks_manifest_sha256": digest},
                    "cells": [
                        {
                            "cell_type": "code",
                            "execution_count": 1,
                            "outputs": [{"output_type": "stream", "text": "computed result"}],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def test_executed_current_notebooks_pass(tmp_path):
    fixture(tmp_path)
    assert check(tmp_path) == 4


def test_empty_notebook_set_fails(tmp_path):
    paths = fixture(tmp_path)
    for path in paths:
        path.unlink()
    with pytest.raises(ValueError, match="Exactly four"):
        check(tmp_path)


def test_changed_manifest_rejects_old_execution(tmp_path):
    fixture(tmp_path)
    (tmp_path / "results/manifest.json").write_text('{"evidence": false}', encoding="utf-8")
    with pytest.raises(ValueError, match="digest is stale"):
        check(tmp_path)


def test_git_line_endings_and_json_formatting_preserve_execution_digest(tmp_path):
    fixture(tmp_path)
    manifest = tmp_path / "results/manifest.json"
    original = manifest_digest(manifest)
    manifest.write_bytes(b'{\r\n  "evidence": true\r\n}\r\n')
    assert manifest_digest(manifest) == original
    assert check(tmp_path) == 4
    manifest.write_bytes(b'{\n  "evidence": true\n}\n')
    assert manifest_digest(manifest) == original
    assert check(tmp_path) == 4
    manifest.write_bytes(b'{\n  "evidence": false\n}\n')
    assert manifest_digest(manifest) != original
    with pytest.raises(ValueError, match="digest is stale"):
        check(tmp_path)


def test_canonical_digest_preserves_changed_string_values(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"caption": "Line one\nLine two", "metric": 0.2}), encoding="utf-8")
    original = manifest_digest(manifest)
    manifest.write_text(json.dumps({"metric": 0.2, "caption": "Line one\nLine two"}), encoding="utf-8")
    assert manifest_digest(manifest) == original
    manifest.write_text(json.dumps({"metric": 0.2, "caption": "Line one\r\nLine two"}), encoding="utf-8")
    assert manifest_digest(manifest) != original


@pytest.mark.parametrize("mutation", ["unexecuted", "error", "no_output", "no_code"])
def test_deliberate_execution_violations_fail(tmp_path, mutation):
    path = fixture(tmp_path)[0]
    content = json.loads(path.read_text(encoding="utf-8"))
    cell = content["cells"][0]
    if mutation == "unexecuted":
        cell["execution_count"] = None
    elif mutation == "error":
        cell["outputs"] = [{"output_type": "error", "ename": "ValueError"}]
    elif mutation == "no_output":
        cell["outputs"] = []
    else:
        content["cells"] = []
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError):
        check(tmp_path)
