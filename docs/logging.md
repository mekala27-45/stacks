# Exposure and feedback logging

These are demonstration recommendations on a public dataset; no real reader's identity is present and no recommendation is personalized to a real person.

The reference exposure schema is defined in `packages/api/database.py`. SQLite is the local default; SQLAlchemy also accepts PostgreSQL with psycopg. The portable deployed schema is `packages/edge/schema.sql`, using D1. Reference SQLite persistence is tested through a new engine and through an independent process. Edge tests run the SQL transactions against SQLite, and deployment verification must separately confirm D1 persistence. PostgreSQL support is not a claim that a PostgreSQL project was provisioned.

## Records

| Table | Essential fields | Meaning |
| --- | --- | --- |
| sessions | session_hash, created_at, updated_at, history, revision, latest_shelf, arm | Anonymous session state and last committed SSE snapshot. |
| impressions | id, session_hash, item_id, position, propensity, arm, experiment_id, model_version, policy, candidate_pool, impression_at, trace | One row per shelf item actually returned by the API. |
| feedback | id, impression_id, event, rating, feedback_at | Delayed click, save, or rating. Its timestamp is separate from impression time. |
| registry_decisions | candidate_version, metrics, gates, eligible, evaluated_at | Evidence submitted to the gate evaluator; this does not activate a model. |

Every feedback row references an existing impression. SQL checks require positive positions and propensities in the interval `(0, 1]`. An impression without a propensity fails at the database boundary. Feedback is accepted only for the session that owns the impression. Duplicate event types for the same impression are rejected by the API. Ratings require an integer in the supported rating scale. Session events, newly generated impressions, and the updated snapshot commit in one transaction.

## Privacy and request identity

The client receives a random opaque bearer token. The database stores only its HMAC-SHA256 digest using `SESSION_HASH_KEY`; it never stores the raw token, names, email addresses, or IP addresses. The public reader examples are anonymous dataset histories. Histories are item IDs. The key must be stable across API restarts and workers or existing sessions will become inaccessible.

The session token appears in the stream URL because browser EventSource cannot set arbitrary request headers. Production proxies must redact session URL segments and the `session_id` query parameter. The documented uvicorn command and container disable request access logs for this reason. Application log messages contain only hashed sessions and operational metadata. Administrative endpoints require `X-Admin-Token`; they are disabled when `ADMIN_TOKEN` is absent. Public event writes require the opaque session token and a matching impression ID, and have per-process rate limits plus durable per-session storage limits.

## Assignment and propensities

Assignment hashes the experiment ID with the hashed session into `als` or `blend`; the arm is stored with every exposure. These are the evaluated frozen artifacts. Public reader initialization uses saved factors and complete training exclusions. Later events use explicitly labeled frozen-item-factor fold-in. Both arms reserve the final position for uniform exploration after business rules. The live deterministic prefix preserves scorer order; MMR and calibration remain separate offline diagnostics. This assignment shell does not claim online lift.

The deterministic prefix is fixed before exploration and each prefix impression records conditional probability one. The exploration pool is the first eligible candidates remaining after all prefix items have been removed and the author cap applied. The pool is fixed for that request, stored verbatim with the impression, and the draw is uniform. Its probability is exactly the reciprocal of the pool size. If the pool is empty, no exploration impression is invented. Assignment probability and action propensity are separate quantities.

These are conditional **item probabilities at a position**, not joint slate propensities. Deterministic positions provide support only for the chosen action. The own-log endpoint uses exploration rows and the stored target distribution, which mixes the best remaining scored candidate with uniform exploration. Click reward is defined by the immutable horizon in the trace. Pending impressions are censored; late feedback persists but does not change that reward. A click on an earlier shelf remains attached to its original impression and can arrive after subsequent shelves. Saves and ratings do not count as clicks.

Synthetic verification sessions set `traffic_kind` to `load_test`. The flag persists in the shelf and exposure trace and excludes these rows from OPE, position CTR, coverage and shadow monitoring. They remain available in the owning session's export. The default interactive mode does not make a claim that a visitor represents the broader reader population.

The compatible Open Bandit field mapping is `item_id -> action`, `position - 1 -> position`, click within the fixed horizon `-> reward`, and `propensity -> pscore`. Context includes the hashed session, experiment, model, and candidate pool. Only after the horizon ends does a missing click become a zero reward. Support and the target distribution are validated before estimation.

## Persistence verification

Run `python -m pytest tests/test_api.py tests/test_registry.py` and `node --test packages/edge/test-worker.mjs`. The Python tests open a real HTTP API in a subprocess, create a session, read SSE, send clicks, then inspect committed records from a separate process and engine. Edge tests exercise the D1-shaped SQL API, ownership, atomic snapshots, resume, SSE, bounded uploads, horizon censoring, support validation, and all exported reader scoring fixtures. Both verify that replay creates no additional impressions.

The deployed proxy buffers SSE until connection closure. The edge response closes after two seconds, requests a one-second EventSource retry, and uses `Last-Event-ID` to resume without duplicate events. This bounded connection transport reads committed snapshots; receiving or reconnecting to it never writes impressions. `scripts/smoke_deployed.py` verifies an initial event, a click from a separate HTTP client, the next revision on a resumed SSE connection, and exactly twenty impressions plus one feedback row. Its requests are explicitly excluded `load_test` traffic. The resulting `results/deployment-verification.json` contains an impression identifier and counts for an independent D1 inspection, with no session token or hash.

For a disposable local reset, stop the API and remove only `.runtime/stacks.db`. This also deletes sessions and registry audits. Never reset a production database with this development procedure.
