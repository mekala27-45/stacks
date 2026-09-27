"""Require executed, error-free notebooks tied to the current evidence digest."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(root: Path = ROOT) -> int:
    expected = hashlib.sha256((root / "results/manifest.json").read_bytes()).hexdigest()
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    if len(notebooks) != 4:
        raise ValueError("Exactly four executed evidence notebooks are required")
    for path in notebooks:
        content = json.loads(path.read_text(encoding="utf-8"))
        if content.get("metadata", {}).get("stacks_manifest_sha256") != expected:
            raise ValueError(f"Notebook evidence digest is stale: {path.name}")
        cells = [cell for cell in content.get("cells", []) if cell.get("cell_type") == "code"]
        if not cells or any(cell.get("execution_count") is None for cell in cells):
            raise ValueError(f"Notebook has absent or unexecuted code: {path.name}")
        if any(output.get("output_type") == "error" for cell in cells for output in cell.get("outputs", [])):
            raise ValueError(f"Notebook retained an execution error: {path.name}")
        if not any(cell.get("outputs") for cell in cells):
            raise ValueError(f"Notebook has no computed output: {path.name}")
    return len(notebooks)


if __name__ == "__main__":
    print(f"Verified {check()} executed notebooks against current evidence")
