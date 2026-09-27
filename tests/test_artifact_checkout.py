"""The released serving bundle must survive Windows and Linux Git checkouts."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from packages.api.catalog import Catalog
from scripts.build_evidence import write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("checkout_autocrlf", ["false", "true"])
def test_real_serving_bundle_survives_git_checkout(tmp_path, checkout_autocrlf):
    """Stage the published profile on Windows, then load either checkout policy.

    A synthetic fixture without a checksum manifest hid the CI startup failure.
    This exercises the actual release's hashes, profiles and trained artifacts.
    """
    source = ROOT / "web/public/data"
    manifest = json.loads((ROOT / "results/serving-manifest.json").read_text(encoding="utf-8"))
    target = tmp_path / "web/public/data"
    target.mkdir(parents=True)
    shutil.copyfile(ROOT / ".gitattributes", tmp_path / ".gitattributes")
    for name in {*manifest["sha256"], "catalog.json", "readers.json"}:
        shutil.copyfile(source / name, target / name)
    profile = target / "edge-profiles.json"
    relative = profile.relative_to(tmp_path).as_posix()

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments], cwd=tmp_path, check=True, capture_output=True
        ).stdout

    git("init", "--quiet")
    git("-c", "core.autocrlf=true", "add", ".gitattributes", relative)
    staged = git("show", f":{relative}")
    expected = manifest["sha256"]["edge-profiles.json"]
    assert hashlib.sha256(staged).hexdigest() == expected
    profile.unlink()
    git("-c", f"core.autocrlf={checkout_autocrlf}", "checkout", "--", relative)
    assert hashlib.sha256(profile.read_bytes()).hexdigest() == expected

    catalog = Catalog(target, artifacts=ROOT / "results")
    assert len(catalog.items) == manifest["catalog_size"]
    assert catalog.artifact_version == manifest["artifact_version"]

    # Content corruption must still fail: this fix never relaxes hash validation.
    profiles = json.loads(profile.read_text(encoding="utf-8"))
    profiles[0]["user_factor"][0] += 1
    write_json(profile, profiles)
    with pytest.raises(ValueError, match="Artifact checksum mismatch"):
        Catalog(target, artifacts=ROOT / "results")


def test_generated_json_has_explicit_lf_bytes(tmp_path):
    path = tmp_path / "profile.json"
    write_json(path, {"reader": [1, 2]})
    assert path.read_bytes() == b'{\n  "reader": [\n    1,\n    2\n  ]\n}\n'
