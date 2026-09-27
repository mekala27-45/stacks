# Contributing

These are demonstration recommendations on a public dataset; no real reader's identity is present and no recommendation is personalized to a real person.

Install locked Python dependencies with `uv sync --frozen` and run `npm ci` in `web`. Preserve source provenance. Do not commit `.env` files or live databases.

Before changing models or estimators, read [the evaluation protocol](docs/evaluation.md). Define the protocol before inspecting a new holdout result, retain popularity and distinguish diagnostics from selection criteria. Cohort and candidate changes alter a metric's meaning.

Run the Python checks from the root:

```sh
uv run ruff check .
uv run mypy
uv run python scripts/check_no_em_dash.py
uv run python scripts/check_vocabulary.py
uv run python scripts/validate_palette.py
uv run pytest --cov=packages --cov-branch --cov-report=term-missing --cov-fail-under=80
uv run python scripts/render_reports.py --check
uv run python scripts/check_notebooks.py
```

Strict typing covers `packages` and `scripts`. The Python branch-coverage floor covers application packages; meaningful Worker and browser tests are separate. Text gates inspect authored source, excluding datasets, downloaded material and generated browser copies. Palette checks read the actual light/dark semantic CSS colors and require normal-text contrast of 4.5 for paper and panel surfaces. This gate is not a claim of complete color-vision or accessibility validation. Each gate has clean, deliberate-failure and empty-input tests.

After `npm ci` in `web`, run `npm run typecheck`, `npm test`, `npx playwright install --with-deps chromium`, `npm run test:e2e` and `npm run build`. From the root, run `node --test packages/edge/test-worker.mjs` under Node 24. Browser tests launch their own API and frontend, checking live persistence, reload/resume and offline fallback. API tests inspect writes through independent connections and a real subprocess.

Edit `report/templates/` to change generated prose. Edit the pipeline and rebuild evidence to change calculations. Regenerate reports and execute the four notebooks after the final manifest changes. `--check` rejects complete-file drift, and the notebook gate rejects stale evidence digests, missing execution and error outputs. Do not patch generated metrics to make a gate pass.

Use a focused commit and describe the problem, resulting behavior and validation. Retain failed model experiments and documented reversals. Report source-order, population, target-policy and hosting limits whenever they affect a changed claim.
