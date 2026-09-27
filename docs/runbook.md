# Runbook

These are demonstration recommendations on a public dataset; no real reader's
identity is present and no recommendation is personalized to a real person.

## Inspect the committed evidence

Install Python through uv and Node.js, then run from the repository root:

```sh
uv sync --frozen
uv run ruff check .
uv run mypy
uv run pytest --cov=packages --cov-branch --cov-fail-under=80
uv run python scripts/render_reports.py --check
uv run python scripts/check_notebooks.py
```

The default check uses committed evidence. It must not need a data-host account
or cloud database. The renderer compares complete files, including the README,
results, model cards, report and downloadable copies. Missing or empty evidence,
altered prose, and obsolete generated cards fail the check.

## Rebuild scientific evidence

Read [the evaluation protocol](evaluation.md) before changing any split,
feature or candidate rule. Then run:

```sh
uv run python scripts/crosscheck_obp.py
uv run python scripts/build_evidence.py
uv run python scripts/build_neural_evidence.py
uv run python scripts/render_reports.py
uv run python scripts/build_notebooks.py
uv run pytest
uv run python scripts/render_reports.py --check
```

The pipeline downloads declared public files into the ignored raw-data area.
Its manifest records the chosen backend and selection. Network availability,
source hashes, package versions and the seed affect reproducibility. A fixture
or reduced sample must retain its own label; never relabel it as full-source
evidence. Changing source or protocol requires reviewing regenerated results.

The core pipeline trains retrieval and LambdaMART; the second command adds the
learned two-tower, recurrent and recent-history comparison. Optional local
PostgreSQL benchmarking uses `scripts/benchmark_pgvector.py`; pass a local
database URL through `STACKS_VECTOR_DATABASE_URL` and inspect `--help`.
After a new benchmark, `uv run python scripts/build_neural_evidence.py
--diagnostics-only` attaches it only when its frozen-artifact hash matches.
Rebuild the reports and notebooks after that attachment. The four notebooks
record the manifest digest and retain executed failure investigations.

On a Windows machine where compiled scientific libraries cannot run, use the
Linux pipeline container. Do not change operating-system security settings:

```sh
docker build -f Dockerfile.pipeline -t stacks-pipeline .
docker run --rm -v "${PWD}:/app" stacks-pipeline
```

The container keeps Linux Python dependencies under `/opt/stacks-venv`, outside
the mounted checkout. Pipeline generation is separate from the local exposure
database and does not reset a running service or any remote database.

## Serve the bookstore and API

In `web`, run `npm ci`, `npm run typecheck`, then `npm run dev`. The static
application uses committed catalog artifacts. To run a production-like static
build for Pages, set `NEXT_PUBLIC_BASE_PATH=/stacks` when building and serve the
exported `web/out` at that same prefix.

From the repository root, start the optional live service:

```sh
uv run uvicorn packages.api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

`DATABASE_URL` defaults to `sqlite:///./.runtime/stacks.db`.
`STACKS_DATA_DIR` overrides the directory of model-serving catalog files.
Use a persistent `SESSION_HASH_KEY` for repeatable anonymous session hashing.
Set `ADMIN_TOKEN` to enable protected administrative endpoints and configure
`CORS_ORIGINS` with the exact local frontend origin, including its port.
Never embed a database URL or administrative secret in a browser build.
Disable access logs because session-bearing request paths contain credentials.
See [serving](serving.md) for API paths and [logging](logging.md) for exposure,
feedback and propensity semantics.

The default local session key is for development only. Configure service keys
before exposing a service beyond the local machine. A database connection
string selects an existing destination; it does not provision a managed service.

## Public deployment and release

The GitHub Pages application is
[mekala27-45.github.io/stacks](https://mekala27-45.github.io/stacks/).
Configure repository Pages to use GitHub Actions. `Validate stacks` verifies
Python behavior, strict types, lint, branch coverage, text and palette gates,
executed notebooks, edge behavior, browser flows, generated evidence and the
static build.
`Publish bookstore` runs after successful validation, checks out that exact
revision and verifies it again before uploading `web/out`. Manual publication
also runs the same checks. The base path is `/stacks`.

After deployment, open the bookstore in a browser. Check catalog filtering,
recommendation interactions, trace details, evaluation evidence, download links,
keyboard navigation and a narrow viewport. Confirm the displayed backend and
API mode are accurate. Fetch the deployed evidence JSON independently and
compare it with the committed artifact. A successful workflow alone does not
prove that every browser interaction works.

The [public API](https://stacks-recommender-api.mekalaa1.chatgpt.site/health)
runs `packages/edge` on the selected free Sites host with durable D1 storage.
The [sanitized verification](../results/deployment-verification.json) records
passing public health, exact-origin CORS, versioned recommendation, feedback,
session resume and SSE checks. A separate Sites D1 connector query observed the
same selected impression's click feedback after the HTTP request completed.
The artifact records that independent check explicitly.

The host buffers long event-stream responses. The Worker therefore uses
two-second SSE connections, a one-second EventSource retry and Last-Event-ID
replay of persisted events. A new revision is verified on a resumed connection.
This transport does not continuously flush chunks on one long-lived connection.

The Worker/D1 substitution meets the user's strict free-hosting requirement.
The FastAPI reference still runs locally with SQLite or PostgreSQL, and the
pgvector experiment uses an isolated local database. Managed PostgreSQL is not
required by the deployed service. Static browser-only state does not count as a
server exposure log.

Sites receives a portable serving package because the complete research checkout
exceeds its source-upload limit. Run `node scripts/prepare-site-source.mjs`
after committing the intended source revision, then publish the generated
`.cache/site-source` directory through Sites. That package contains the Worker,
schema, frozen serving inputs and `SOURCE.json` identifying the GitHub revision.
GitHub retains the complete data exports, training code, tests, notebooks and
reports; the portable package does not replace that repository. After changing
the Worker or artifacts, repeat live verification and the independent D1 check.

[Public latency evidence](../results/latency.json) records a small sequential
warm-service sample, including transport, shelf recomputation and database
writes. Its percentile bootstrap ranges describe that finite sample, not a
capacity limit or production SLO. Test traffic is labeled and excluded from
own-log OPE. Use `uv run python scripts/load_test.py --url <public-api-url>` to
repeat the documented request experiment. Preserve the measurement date and
model/artifact identifiers when comparing runs.

## Failure and recovery

If a report differs, correct the evidence or its template and regenerate. Never
patch a generated metric. If source validation fails, preserve the downloaded
file and its checksum for diagnosis. If persistence fails, stop claiming a
logged exposure until a fresh connection can observe the committed write.

For a failed website release, rerun a known-good Pages revision or revert the
faulty application change. Preserve the matching catalog and evidence bundle.
Keep a separate backup before intentionally resetting a local interaction
database. Never reset an external database as part of static-site rollback.

The repository owner is the maintainer for this demonstration. Report a failure
with its revision, command, backend, and sanitized error output in the
[GitHub issue tracker](https://github.com/mekala27-45/stacks/issues); exclude
tokens, database URLs, and raw session identifiers.

## Independent estimator reference

`scripts/crosscheck_obp.py` downloads the pinned Apache-licensed Open Bandit
Pipeline estimator source, verifies its exact SHA-256 digest, and compiles only
the original arithmetic method ASTs for IPS, SNIPS, DM and DR. It compares point
estimates on the same simulator samples used by the local calibration harness.
This avoids the upstream package's older Python and Torch dependency stack.
It does not modify the reference method bodies, execute package initialization,
or test upstream input validation or bootstrap intervals. The resulting
`results/obp-crosscheck.json` records the scope, digest, source revision and
observed differences; the evidence rebuild incorporates it into the manifest.

The source cache and its Apache notice live in ignored `data/raw/obp_reference`.
A missing network source or checksum mismatch fails the crosscheck. Existing
recorded output can be reviewed without a network dependency, but it must not be
described as a new crosscheck run. Full installed-package compatibility remains
unverified.
