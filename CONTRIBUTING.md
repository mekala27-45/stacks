# Contributing

These are demonstration recommendations on a public dataset; no real reader's
identity is present and no recommendation is personalized to a real person.

Install the locked Python dependencies with `uv sync --frozen`. In `web`, run
`npm ci`. Keep source downloads outside version control, preserve provenance,
and never commit a `.env` file or a live interaction database.

Before changing model or estimator code, read [the evaluation protocol](docs/evaluation.md).
Fix the protocol before inspecting a new test result, retain the popularity
comparison, and distinguish exploratory diagnostics from selection criteria.
Any change to the cohort or candidate universe changes the meaning of a metric.

Run `uv run pytest`, `uv run python scripts/render_reports.py --check`, then
`npm run typecheck` and `npm run build` in `web`. The report checker must reject
deliberate drift and empty input; persistence checks must observe writes beyond
the request's own connection. Run the relevant behavior test before a broad
rebuild, then run the release checks once the change is ready.

Edit `report/templates/` and regenerate when changing generated prose. Edit
pipeline code and rebuild evidence when changing a calculation. Do not manually
patch `README.md`, `RESULTS.md`, or generated reports to make a gate pass.

Use a short branch name, a focused commit, and a pull request explaining the
problem, resulting behavior, and validation. Report source-order, catalog,
simulator, or hosting limitations whenever they affect the claim being changed.
