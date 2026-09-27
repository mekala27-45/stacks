# Serving the demonstration

These are demonstration recommendations on a public dataset; no real reader's identity is present and no recommendation is personalized to a real person.

The published browser application uses the committed bundle by default. The FastAPI service is executable locally and packaged for deployment, but no remote API or database is provisioned by this repository. Set the frontend's API configuration only after deploying and verifying a backend. Browser-only interactions are not durable API exposures.

## Start

Install the locked Python environment with `uv sync`. After generating the data bundle, run:

```sh
uv run uvicorn packages.api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

The OpenAPI specification is at `/openapi.json` and interactive endpoint documentation is at `/docs`. Health requires both the catalog bundle and a working database. A missing bundle returns a service-unavailable response; there is no fabricated fallback inside the API.

`DATABASE_URL` defaults to `sqlite:///./.runtime/stacks.db`. A PostgreSQL URL is accepted and automatically uses psycopg. `STACKS_DATA_DIR` defaults to `web/public/data`, whose required files are `catalog.json`, `similarity.json`, and `readers.json`. `SESSION_HASH_KEY` must be replaced with a persistent secret before remote deployment. `CORS_ORIGINS` is an explicit comma-separated list. `ADMIN_TOKEN` enables administrative writes and monitoring. The complete environment example is `.env.example`.

## Endpoints

| Method | Path | Contract |
| --- | --- | --- |
| GET | /health | Database and catalog readiness. |
| GET | /v1/catalog | Public catalog items and demonstration statement. |
| POST | /v1/session | Body `{ "reader_id": "public-sample-id" }` or `{}`. Creates an anonymous session and returns its initial logged shelf. |
| GET | /v1/recommend/{user_id} | Public sample reader ID or `new`. Optional `session_id` continues existing state; optional `limit` bounds the shelf. Returns a logged shelf. |
| POST | /v1/session/{session_id}/event | Body `{ "impression_id": "...", "event": "click" }`. Also accepts `save` or `rating` with a rating value. Returns the new shelf after one atomic state update. |
| GET | /v1/session/{session_id}/stream | SSE event `shelf`, ID equal to revision. Uses fresh database reads so workers see committed updates. Supports Last-Event-ID and `once=true` for a bounded snapshot. |
| GET | /v1/similar/{item_id} | Item cosine neighbors from the committed exact-scoring bundle. No approximate index is available; `exact=false` still identifies the returned backend accurately. |
| GET | /v1/explain/{impression_id} | Requires owning `session_id` query token. Returns the durable trace, position, propensity, and timestamp. |
| GET | /v1/experiment/assign | Requires `session_id`. Returns the stored deterministic arm and experiment. |
| POST | /v1/log | Admin token required. Body includes `session_id` and the same event fields. Persists delayed feedback and publishes updated state. |
| POST | /v1/registry/evaluate | Admin token required. Body `{ "candidate_version": "...", "metrics": {...} }`. Persists gate decisions; never activates a model. |
| GET | /v1/monitoring | Admin token required. Returns durable exposure and feedback counts with no online performance claim. |

Every shelf contains `session_id`, `revision`, `model_version`, `arm`, `experiment_id`, `backend`, `latency_ms`, `statement`, and `items`. Every item contains catalog metadata, score, explanation, `impression_id`, position, propensity, and trace. A trace contains the retrieval candidate preview, ranker features and weights, reranking steps, exact exploration support, and scores from a shadow model. The latency field measures shelf construction through generation, and excludes network transport and transaction commit. An external load test is needed for an end-to-end latency claim.

## Model and business rules

`popularity-v1` sorts normalized log training popularity. `popularity-cosine-blend-v1` combines that signal with item-cosine similarities to recent history, with recency decay over ordered item IDs. These are explicit arithmetic models, not a deployed LambdaMART model or neural network. Candidate lookup uses the process-cached published bundle. The trace records component scores and the actual serving weights.

Rules remove seen and unavailable items and cap each author. MMR selects a diverse slate; an observed diversity safeguard keeps the original list if the objective would reduce measured diversity. Calibration accepts only swaps that improve genre-distribution divergence and a relevance-aware objective. The final exploration draw may change the final slate's diversity and calibration, so the trace names the stage whose metrics it reports. Cold sessions use popularity plus exploration. A clicked or saved item enters ordered session history; positive ratings also enter it. There is no recurrent model.

Shadow scoring logs the alternate arithmetic model's candidates and scores but returns only the selected arm. The feature-schema, accuracy, coverage, long-tail, calibration, and latency gates reject absent or nonfinite evidence. Their persisted result is promotion **eligibility** only. Deployment and activation require a separately implemented model adapter and an operational review.

## Operational limits

The API has bounded shelf size, per-client process-local request windows, a durable session cap, a durable per-session impression cap, and validation on event ownership and rating fields. For Internet deployment place a global rate limiter before the service; process-local limits do not coordinate across workers. SQLAlchemy revision checks prevent silently overwriting concurrent session updates. SSE is a latest-state stream with replay of the last committed revision, not an event archive; intermediate revisions can be skipped for slow clients. Streaming connections read the current snapshot periodically and heartbeat while idle.

The checked-in Fly configuration is a deployment template and does not establish a free entitlement. Provision an app, choose the host's available plan, and inject database and session secrets explicitly before deployment. PostgreSQL migration history, retention jobs, a globally coordinated limiter, authenticated end-user accounts, and a provisioned production backend are outside this release.

What a product manager would push back on: a popularity shelf may be sufficient, and diversity can lower immediate relevance. This service exposes the alternative shelf, the component scores, and the conditional exploration probabilities. It does not turn a small demonstration log into an effectiveness claim.
