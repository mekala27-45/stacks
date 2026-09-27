"""Create and execute four evidence notebooks; all quantitative output is computed."""

from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
INTRO = "These are demonstration recommendations on a public dataset; no real reader's identity is present and no recommendation is personalized to a real person."
LOAD = """from pathlib import Path
import json
import pandas as pd
root = Path.cwd()
e = json.loads((root / 'results/manifest.json').read_text(encoding='utf-8'))
print('Population:', e['dataset'])
print('Protocol:', e['protocol']['name'])
"""


def notebook_specs() -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        (
            "01-source-and-population",
            [
                (
                    "markdown",
                    "# Source order is not a calendar\n\n"
                    + INTRO
                    + "\n\nDecision date: 2026-09-27. Inspect the actual population before interpreting a score.",
                ),
                ("code", LOAD),
                (
                    "code",
                    """ratings = pd.read_parquet(root / 'data/goodbooks/ratings.parquet')
print({'rows_read': len(ratings), 'books': int(ratings.book_id.nunique()), 'readers': int(ratings.user_id.nunique())})
print('Columns:', ratings.columns.tolist())
print('Source rows monotonic:', ratings.source_row.is_monotonic_increasing)
""",
                ),
                (
                    "markdown",
                    "## Dead end retained: asking this file for event dates\n\nThe source's stated row order cannot establish calendar boundaries. The attempted column lookup below fails on the actual exported data. We retain the failure rather than inventing dates.",
                ),
                (
                    "code",
                    """try:
    pd.to_datetime(ratings['timestamp'])
except KeyError as error:
    print('Rejected calendar reconstruction:', repr(error))
assert 'timestamp' not in ratings.columns
print('Use the documented source-order proxy, with undated metadata caveats.')
""",
                ),
            ],
        ),
        (
            "02-ranking-and-dead-end-selection",
            [
                (
                    "markdown",
                    "# The largest point estimate is not a promotion decision\n\n"
                    + INTRO
                    + "\n\nDecision date: 2026-09-27.",
                ),
                ("code", LOAD),
                (
                    "code",
                    """table = pd.DataFrame([{'model': r['label'], 'ndcg': r['ndcg']['mean'], 'low': r['ndcg']['low'], 'high': r['ndcg']['high']} for r in e['metrics']])
print(table.to_string(index=False))
print('Observed and corrected selection:', e['winner'])
print(pd.DataFrame(e['comparisons'])[['left', 'right', 'p', 'q', 'significant']].to_string(index=False))
""",
                ),
                (
                    "markdown",
                    "## Dead end retained: promote the initial blend because it sounds richer\n\nThe original bounded release's paired evidence did not establish the blend above ALS. This is an actual earlier measured result, preserved separately from the expanded run. Complexity and a large point estimate were rejected as sufficient promotion evidence.",
                ),
                (
                    "code",
                    """history = json.loads((root / 'results/history/bounded-v0.1.0.json').read_text(encoding='utf-8'))
pair = [r for r in history['comparisons'] if {r['left'], r['right']} == {'als', 'blend'}]
print('Original bounded population:', history['dataset'])
print('Recorded ALS versus blend comparison:', pair)
assert pair and not pair[0]['significant']
for shortcut in ('split', 'sampled'):
    print(shortcut, pd.DataFrame([{'model':r['label'], 'reference':r['full']['mean'], 'shortcut':r['shortcut']['mean'], 'delta':r['delta']['mean']} for r in e['inflation'][shortcut]]).to_string(index=False))
print('Cold-reader investigation:', pd.DataFrame(e['cold_reader_analysis']['cohorts']).to_string(index=False))
print('Neural training:', e['neural_training']['two_tower'])
print('Measured geometry:', e['neural_training']['geometry'])
print('Local pgvector experiment:', e['pgvector'])
print('The neural loss fell, but held-out quality remained weak. Retain this failed architecture choice; do not infer global dead ReLU from a single query vector.')
""",
                ),
            ],
        ),
        (
            "03-ope-and-target-reconstruction",
            [
                (
                    "markdown",
                    "# Reconstruct a policy from its published rule\n\n"
                    + INTRO
                    + "\n\nDecision date: 2026-09-27. The following outputs come from all three Open Bandit campaigns.",
                ),
                ("code", LOAD),
                (
                    "code",
                    """obd = e['ope']['open_bandit']
print(obd['status'])
print(pd.DataFrame([{'campaign': c['campaign'], 'actions': c['actions'], 'random_test': c['evaluation_rows'], 'bts_reference': c['bts_reference_rows'], 'bts_reward': c['on_policy_bts']['mean']} for c in obd['campaigns']]).to_string(index=False))
print(pd.DataFrame([{'campaign':r['campaign'], 'slot':r['position'], 'estimator':r['estimator'], 'estimate':r['estimate']['mean'], 'BTS_empirical':r['bts_reference']['mean'], 'delta':r['difference_from_bts']['mean']} for r in obd['rows'] if r['policy'] == 'Official prior BTS approximation']).to_string(index=False))
""",
                ),
                (
                    "markdown",
                    "## Dead end retained: treating the frozen prior as the exact logged policy\n\nThe official benchmark supplies a prior-based target. The actual log has varying propensities within item and slot, so an exact match cannot be asserted. This diagnostic rejected that stronger interpretation. We use the official approximation and show the mismatch limitation.",
                ),
                (
                    "code",
                    """for campaign in obd['campaigns']:
    diagnostic = campaign['logged_propensity_diagnostic']
    print(campaign['campaign'], diagnostic)
    assert diagnostic['max_distinct_values_within_item_position'] > 1
print('Monte Carlo diagnostics:', [{k:v for k,v in r.items() if k != 'probabilities'} for r in obd['monte_carlo']])
print('Source-reference arithmetic check:', e['ope']['crosscheck']['status'])
""",
                ),
            ],
        ),
        (
            "04-coverage-and-publication-failures",
            [
                (
                    "markdown",
                    "# Keep aggregate coverage and document claims honest\n\n"
                    + INTRO
                    + "\n\nDecision date: 2026-09-27.",
                ),
                ("code", LOAD),
                (
                    "code",
                    """for row in e['metrics']:
    value = row['coverage']
    print(row['label'], value)
print('Resampling a fixed set of reader lists can omit unique items. Its coverage range is not a confidence interval for the observed union or unseen-reader coverage.')
""",
                ),
                (
                    "markdown",
                    "## Dead end retained: correcting a result by editing generated Markdown\n\nThis deliberate mutation uses a temporary copy. The actual renderer rejects the changed claim while leaving source evidence untouched. The correct edit path is the template or calculation, followed by regeneration.",
                ),
                (
                    "code",
                    """import shutil
import tempfile
from scripts.render_reports import render
with tempfile.TemporaryDirectory() as directory:
    sandbox = Path(directory)
    shutil.copytree(root / 'report/templates', sandbox / 'report/templates')
    manifest = sandbox / 'manifest.json'
    manifest.write_text(json.dumps(e), encoding='utf-8')
    assert not render(manifest, sandbox)
    report = sandbox / 'RESULTS.md'
    report.write_text(report.read_text(encoding='utf-8') + '\\nUnsupported promotional claim.\\n', encoding='utf-8')
    failures = render(manifest, sandbox, check=True)
    assert failures
    print('Rejected edited report:', failures[0][-350:])
print('Original repository report remains unchanged.')
""",
                ),
            ],
        ),
    ]


def build(root: Path = ROOT) -> None:
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    target = root / "notebooks"
    target.mkdir(exist_ok=True)
    for name, content in notebook_specs():
        code_cell = cast(Callable[[str], Any], nbformat.v4.new_code_cell)
        markdown_cell = cast(Callable[[str], Any], nbformat.v4.new_markdown_cell)
        new_notebook = cast(Callable[..., Any], nbformat.v4.new_notebook)
        write_notebook = cast(Callable[[Any, Path], None], nbformat.write)
        cells = [code_cell(value) if kind == "code" else markdown_cell(value) for kind, value in content]
        notebook: Any = new_notebook(cells=cells)
        notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
        notebook.metadata.stacks_manifest_sha256 = hashlib.sha256(
            (root / "results/manifest.json").read_bytes()
        ).hexdigest()
        NotebookClient(
            notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(root)}}
        ).execute()
        if any(cell.cell_type == "code" and cell.execution_count is None for cell in notebook.cells):
            raise RuntimeError(f"Unexecuted cell in {name}")
        write_notebook(notebook, target / f"{name}.ipynb")
        print(f"Executed {name}")


if __name__ == "__main__":
    build()
