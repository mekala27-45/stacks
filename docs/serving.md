# Serving the evaluated artifacts

These are demonstration recommendations on a public dataset; no real reader's identity is present and no recommendation is personalized to a real person.

The portable Fetch handler in `packages/edge/worker.ts` serves evaluated ALS and fixed-blend artifacts with durable D1 sessions and exposure logs. GitHub Pages remains the frontend and its committed bundle remains the fallback. The FastAPI implementation is the Python reference service. Deployment verification and the current public URL belong in the release report; the existence of a Fly configuration alone does not establish a deployment.

## Artifact parity

Both implementations read the evaluation run's frozen factors, training popularity and sparse item-cosine index. The fixed blend uses the same normalized component weights as evaluation: cosine, ALS and popularity. Initial public-reader requests use the saved user factors and complete positive training history. Every observed training item, including disliked items, is excluded. New visitors start with training popularity. Subsequent positive events use ALS fold-in against frozen item factors; this mode is explicitly labeled and is not presented as an unchanged offline reader prediction.

The edge implementation loads little-endian typed arrays described by `web/public/data/edge-manifest.json`, validates their shapes and values, and verifies the manifest's asset SHA256 hashes. The public-reader artifact contains complete positive history and seen exclusions. The portable parity test checks every exported reader's top recommendations and raw scores against the evaluation-generated fixture. The Python service loads `results/model-artifacts.npz` and the associated sparse matrices through the evaluation package's shared loader.

The live experiment assigns sessions deterministically between evaluated ALS and the fixed blend. Rules remove seen and unavailable items, apply the chosen genre, and cap the primary author. The live deterministic prefix preserves scorer order; MMR and calibration are disabled on this path. Their offline diagnostic remains available separately. The final position is a uniform draw from the remaining eligible pool. Its trace stores the actual support, propensity and target distribution. Shadow scoring records the other evaluated scorer without returning a second shelf.

## Python reference startup

```sh
uv sync
uv run uvicorn packages.api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

`DATABASE_URL` defaults to a disposable SQLite database under `.runtime`. PostgreSQL URLs use psycopg. `STACKS_DATA_DIR` defaults to `web/public/data`; `STACKS_ARTIFACT_DIR` defaults to `results`. Set a stable `SESSION_HASH_KEY` before remote operation. `ADMIN_TOKEN` protects administrative writes. `CORS_ORIGINS` is an explicit origin list. Request access logging is disabled because EventSource uses the bearer session token in its URL.

`Dockerfile.api` includes the evaluated artifacts and the OpenMP runtime needed by the learned offline model adapter. `fly.toml` is an optional deployment template; no Fly entitlement or managed PostgreSQL account is inferred from it. The deployed portable path uses D1 instead.

## Stable v1 response contract

Shelf responses have `schema_version: "1.1"`, `session_id`, `revision`, `model_version`, `artifact_version`, `scoring_mode`, `arm`, `experiment_id`, `backend`, `history`, `genre`, `traffic_kind`, `latency_ms`, `statement`, and `items`. Item fields include catalog metadata, score, explanation, impression ID, position, propensity, and trace. Version additions preserve the original shelf fields.

| Method | Path | Contract |
| --- | --- | --- |
| GET | /health | Database and artifact readiness. |
| GET | /v1/catalog | Public catalog. |
| POST | /v1/session | Optional `reader_id`, `genre` and `traffic_kind`; returns a logged initial shelf. |
| GET | /v1/session/{token} | Resume the stored shelf without generating impressions. |
| POST | /v1/session/{token}/preferences | Body `{ "genre": "All books" }`; returns the next logged revision. |
| POST | /v1/session/{token}/event | Owning impression ID plus `click`, `save`, or `rating`; ratings require an integer value. Feedback and the next shelf persist atomically. |
| GET | /v1/session/{token}/stream | SSE `shelf` events with revision IDs. Two-second connections resume with Last-Event-ID after a one-second retry; `once=true` returns one snapshot. |
| GET | /v1/session/{token}/logs | Persisted exposure and feedback export without raw tokens or session hashes. |
| GET | /v1/session/{token}/ope | Conditional randomized-slot estimates on this session's mature logs. |
| GET | /v1/ope | Same estimates over the bounded global log window. |
| GET | /v1/explain/{impression_id} | Requires owning `session_id`; returns the stored trace. |
| GET | /v1/similar/{item_id} | Neighbors from the same stored sparse item-cosine index used by evaluation. |
| GET | /v1/experiment/assign | Returns the persisted assignment for `session_id`. |
| GET | /v1/monitoring | Interactive exposure counts, position CTR, coverage, shadow disagreement and OPE. Python requires an admin token; edge returns aggregates publicly. |
| GET | /v1/registry | Python returns persisted gate decisions; edge describes the reference registry and does not expose activation. |

The Python reference also provides the original recommendation endpoint, administrative feedback ingestion, submitted-evidence gate evaluation, and `POST /v1/registry/check/{candidate}`. The last route reads the measured manifest and latency report and persists the actual gate decisions. An eligible audit does not activate a model. Published rejected candidates remain rejected.

## Reward horizon and identified OPE

Only the random final position enters own-log OPE. Its target is a fixed mixture of the best scorer-ranked remaining candidate and uniform exploration over that impression's stored pool. Every target action has positive support under the logged uniform policy. Deterministic-prefix probabilities are not misused to evaluate a different slate.

The reward is a click received within the immutable horizon recorded with the impression. Clicks before maturity do not make other impressions mature. Late clicks persist but are excluded from that reward definition. Saves and ratings affect history when appropriate; they are not click rewards. The reward model is an explicit fixed probability baseline, not a fitted click model.

IPS, SNIPS, DM and DR point estimates are returned once mature randomized feedback exists. IPS and DR use conservative bounded concentration intervals with a predetermined importance-weight bound, so adaptive session contexts do not require an independent-row bootstrap assumption. SNIPS uses simultaneous bounds on numerator and denominator. DM's point interval conditions on its fixed prediction and does not certify that model's accuracy. These are fixed-snapshot diagnostics, not an optional-stopping decision rule. They identify the random position under its logged prefix and context, not an altered full slate or the traffic it would induce.

## Verification traffic and measurement

`traffic_kind: "load_test"` explicitly marks synthetic operational requests. They persist for auditing but are excluded from OPE, position CTR, coverage and shadow monitoring. The frontend's default is `interactive`. Automated browser and load checks should use the test marker.

`scripts/load_test.py --url URL --reader-id ID` measures completed HTTP requests that recompute and persist shelves. It records sample size, backend, artifact version, concurrency, raw measurements, percentile estimates, and bootstrap ranges. The shelf's internal `latency_ms` remains a computation diagnostic; it is not the end-to-end latency figure.

SQL constraints enforce position, propensity and feedback linkage. Session writes use optimistic revision checks. The edge transaction guards its exposure and feedback inserts with the winning operation ID, preventing concurrent stale updates from leaving orphaned impressions. SSE replays the latest committed state and may skip intermediate revisions for slow clients.

The managed hosting proxy buffers event-stream bytes until the response closes, even with identity encoding and `Cache-Control: no-transform`. The edge service therefore closes each SSE connection after two seconds and advertises a one-second retry. Browser EventSource reconnects with `Last-Event-ID`, which suppresses duplicate revisions. This is bounded SSE reconnection, not continuously flushed chunks on one long connection. Each connection performs at most three D1 reads and creates no new impressions. The Python reference stream is independent of this hosting adaptation.

`scripts/smoke_deployed.py --url URL` verifies public health and exact-origin CORS, creates a `load_test` session, receives its initial event within five seconds, clicks through an independent client, and receives the changed revision through a resumed SSE connection. It checks a separate client's resume and persisted exposure/feedback export. The sanitized deployment report records checks, counts, version identifiers and an impression ID for an independent database read; it excludes session tokens and hashes.

Both backends bound request bodies, shelf sizes and session storage. D1 writes also use a hashed-client rate bucket. Session tokens and administrative credentials must be kept out of proxy request logs. Production retention and account authentication remain outside this public demonstration.

What a product manager would push back on: an offline winner may not improve reader outcomes. The live trace records which evaluated scorer served, and randomized-slot estimates expose their support and uncertainty. No online lift or automatic promotion follows from this demonstration.
