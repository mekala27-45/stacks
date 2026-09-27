# Exposure and feedback logging

These are demonstration recommendations on a public dataset; no real reader's identity is present and no recommendation is personalized to a real person.

The exposure schema is defined in `packages/api/database.py`. SQLite is the local default. The same SQLAlchemy models accept a PostgreSQL connection URL with the psycopg driver. SQLite persistence is tested through a new engine and through an independent process. PostgreSQL is supported by the implementation but has not been exercised against a provisioned database in this release.

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

Assignment hashes the experiment ID with the hashed session into either `baseline` or `diverse`; the arm is stored with every exposure. The baseline arm serves training popularity. The diverse arm serves a popularity and item-cosine blend followed by business rules, MMR, and calibration. Both arms reserve the final shelf position for exploration. This shell does not estimate or claim online lift.

The deterministic prefix is fixed before exploration and each prefix impression records conditional probability one. The exploration pool is the first eligible candidates remaining after all prefix items have been removed and the author cap applied. The pool is fixed for that request, stored verbatim with the impression, and the draw is uniform. Its probability is exactly the reciprocal of the pool size. If the pool is empty, no exploration impression is invented. Assignment probability and action propensity are separate quantities.

These are conditional **item probabilities at a position**, not joint slate propensities. Deterministic positions provide support only for the chosen action. Off-policy evaluation of alternative actions must use exploration rows and restrict target-policy support to the logged pool. Do not treat an unobserved click as a mature negative without a feedback window. A click on an earlier shelf remains attached to its original impression and can arrive after subsequent shelves.

The compatible Open Bandit field mapping is `item_id -> action`, `position - 1 -> position`, click event within a stated maturation window `-> reward`, and `propensity -> pscore`. Context includes the hashed session, experiment, model, and the candidate pool. No automatic join that treats missing delayed feedback as zero is supplied.

## Persistence verification

Run `python -m pytest tests/test_api.py tests/test_registry.py`. The tests open a real HTTP API in a subprocess, create a session, read an SSE event, send clicks, and then inspect committed sessions, exposures, and feedback using a separate Python process and database engine. In-process tests verify foreign keys, position and propensity checks, ownership, timestamps, experiment assignment, and SSE replay without duplicate exposure writes.

For a disposable local reset, stop the API and remove only `.runtime/stacks.db`. This also deletes sessions and registry audits. Never reset a production database with this development procedure.
