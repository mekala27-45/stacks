"""An executable demo API with hashed sessions and durable exposure logging."""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from packages.api.catalog import Catalog
from packages.api.database import (
    Feedback,
    Impression,
    RegistryDecision,
    SessionRecord,
    create_database,
    utcnow,
)
from packages.registry import evaluate_gates
from packages.rerank import rerank

STATEMENT = "These are demonstration recommendations on a public dataset; no real reader's identity is present and no recommendation is personalized to a real person."
EXPERIMENT = "shelf-diversity-v1"
LOGGER = logging.getLogger("stacks.api")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SessionCreate(StrictModel):
    reader_id: str | None = Field(default=None, max_length=64)


class EventRequest(StrictModel):
    impression_id: str = Field(min_length=1, max_length=36)
    event: Literal["click", "save", "rating"]
    rating: int | None = Field(default=None, ge=1, le=5, strict=True)

    @model_validator(mode="after")
    def validate_rating(self) -> "EventRequest":
        if (self.event == "rating") != (self.rating is not None):
            raise ValueError("Rating is required only for a rating event")
        return self


class LogRequest(EventRequest):
    session_id: str = Field(min_length=32, max_length=128)


class RegistryRequest(StrictModel):
    candidate_version: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9._-]+$")
    metrics: dict[str, float | str | None] = Field(max_length=30)


def session_hash(token: str, key: str) -> str:
    return hmac.new(key.encode(), token.encode(), hashlib.sha256).hexdigest()


def assign_arm(hashed_session: str) -> str:
    digest = hashlib.sha256(f"{EXPERIMENT}:{hashed_session}".encode()).digest()
    return "baseline" if int.from_bytes(digest[:8], "big") % 100 < 50 else "diverse"


def create_app(
    database_url: str | None = None, data_dir: str | Path | None = None, hash_key: str | None = None
) -> FastAPI:
    application = FastAPI(title="stacks demonstration API", version="0.1.0", description=STATEMENT)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:3010,http://localhost:5173,http://localhost:3000").split(",")
            if origin.strip()
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Admin-Token", "Last-Event-ID"],
    )
    engine = create_database(database_url or os.getenv("DATABASE_URL", "sqlite:///./.runtime/stacks.db"))
    key = hash_key or os.getenv("SESSION_HASH_KEY", "stacks-local-development-only")
    directory = Path(
        data_dir or os.getenv("STACKS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "web/public/data"))
    )
    application.state.engine = engine
    application.state.data_dir = directory
    catalog_cache: list[Catalog] = []
    write_lock = threading.RLock()
    request_windows: dict[str, deque[float]] = defaultdict(deque)
    rate_lock = threading.Lock()

    @application.exception_handler(StaleDataError)
    async def concurrent_session_change(_: Request, __: StaleDataError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": "Session changed concurrently; retry this request"})

    @application.exception_handler(IntegrityError)
    async def integrity_conflict(_: Request, __: IntegrityError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": "Conflicting or invalid persistent record"})

    @application.middleware("http")
    async def bounded_body(request: Request, call_next: Any) -> Any:
        if request.method == "POST":
            length = request.headers.get("content-length")
            if length is not None and (not length.isdecimal() or int(length) > 32768):
                return JSONResponse(status_code=413, content={"detail": "Demo request body limit exceeded"})
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 32768:
                    return JSONResponse(status_code=413, content={"detail": "Demo request body limit exceeded"})
            request._body = bytes(body)
        return await call_next(request)

    def catalog() -> Catalog:
        if not catalog_cache:
            try:
                catalog_cache.append(Catalog(directory))
            except (OSError, ValueError, KeyError, TypeError) as error:
                LOGGER.error(json.dumps({"event": "bundle_unavailable", "error_type": type(error).__name__}))
                raise HTTPException(503, "Catalog bundle unavailable; run the data pipeline first") from error
        return catalog_cache[0]

    def limited(request: Request, write: bool = True) -> None:
        # No raw network address enters storage or application logs.
        identity = session_hash(request.client.host if request.client else "unknown", key)
        bucket = f"{identity}:{write}"
        now = time.monotonic()
        with rate_lock:
            queue = request_windows[bucket]
            while queue and queue[0] < now - 60:
                queue.popleft()
            if len(queue) >= (60 if write else 180):
                raise HTTPException(
                    429, "Demo request limit reached; retry after one minute", headers={"Retry-After": "60"}
                )
            queue.append(now)
            if len(request_windows) > 10000:
                for old in [
                    name for name, values in request_windows.items() if not values or values[-1] < now - 60
                ]:
                    del request_windows[old]

    def require_admin(token: str | None) -> None:
        configured = os.getenv("ADMIN_TOKEN")
        if not configured:
            raise HTTPException(503, "Administrative writes are disabled until ADMIN_TOKEN is configured")
        if token is None or not secrets.compare_digest(token, configured):
            raise HTTPException(401, "Valid administrative token required")

    def get_session(db: Session, token: str) -> SessionRecord:
        if len(token) < 32 or len(token) > 128:
            raise HTTPException(404, "Session not found")
        record = db.get(SessionRecord, session_hash(token, key))
        if record is None:
            raise HTTPException(404, "Session not found")
        return record

    def create_session(db: Session, reader_id: str | None = None) -> tuple[str, SessionRecord]:
        source = catalog()
        if reader_id is not None and reader_id not in source.readers:
            raise HTTPException(404, "Public sample reader not found")
        if (db.scalar(select(func.count()).select_from(SessionRecord)) or 0) >= 5000:
            raise HTTPException(429, "Demo session storage limit reached")
        token = secrets.token_urlsafe(32)
        hashed = session_hash(token, key)
        record = SessionRecord(
            session_hash=hashed,
            arm=assign_arm(hashed),
            history=list(source.readers.get(reader_id or "", [])),
            revision=0,
            latest_shelf={},
        )
        db.add(record)
        db.flush()
        return token, record

    def make_shelf(db: Session, record: SessionRecord, limit: int = 10) -> dict[str, Any]:
        started = time.perf_counter()
        exposure_count = (
            db.scalar(
                select(func.count())
                .select_from(Impression)
                .where(Impression.session_hash == record.session_hash)
            )
            or 0
        )
        if exposure_count + limit > 1000:
            raise HTTPException(429, "This demo session reached its impression storage limit")
        source = catalog()
        is_baseline = record.arm == "baseline"
        model = "popularity-v1" if is_baseline else "popularity-cosine-blend-v1"
        seen = set(record.history)
        scored = source.score(record.history, baseline=is_baseline)
        eligible = [item for item in scored if item["id"] not in seen and item.get("available", True)]
        candidates = eligible[:200]
        history = [source.items[item_id] for item_id in record.history if item_id in source.items]
        selected, rerank_trace, ruled = rerank(
            candidates,
            seen,
            history,
            max(0, limit - 1),
            0.0 if is_baseline else 0.25,
            0.0 if is_baseline else 0.2,
        )
        selected_ids = {item["id"] for item in selected}
        exploration_pool = [item for item in ruled if item["id"] not in selected_ids][:20]
        exploration = secrets.choice(exploration_pool) if exploration_pool else None
        if exploration is not None:
            selected.append(exploration)
        shadow = source.score(record.history, baseline=not is_baseline)
        shadow_ids = [item["id"] for item in shadow if item["id"] not in seen][:limit]
        disagreement = 1.0 - len(set(shadow_ids) & {item["id"] for item in selected}) / max(len(selected), 1)
        response_items = []
        for position, item in enumerate(selected, start=1):
            exploring = exploration is not None and item["id"] == exploration["id"]
            pool_ids = [candidate["id"] for candidate in exploration_pool] if exploring else [item["id"]]
            propensity = 1 / len(pool_ids) if exploring else 1.0
            impression_id = str(uuid.uuid4())
            trace = {
                "retrieval": {
                    "backend": "committed-item-cosine-and-training-popularity",
                    "candidate_count": len(candidates),
                    "top_candidates": [{"id": row["id"], "score": row["score"]} for row in candidates[:12]],
                },
                "ranking": {
                    "model_version": model,
                    "score": item["score"],
                    "features": item["scores"],
                    "source_item_id": item["source_item_id"],
                    "weights": {
                        "popularity": 1.0 if is_baseline or not history else 0.3,
                        "item_cosine": 0.0 if is_baseline or not history else 0.7,
                    },
                },
                "reranking": rerank_trace,
                "exploration": {
                    "selected": exploring,
                    "policy": "uniform" if exploring else "deterministic",
                    "candidate_pool": pool_ids,
                    "propensity": propensity,
                    "support": "conditional item probability at this position",
                },
                "shadow": {
                    "model_version": "popularity-cosine-blend-v1" if is_baseline else "popularity-v1",
                    "top_ids": shadow_ids,
                    "top_scores": [
                        {"id": row["id"], "score": row["score"]} for row in shadow if row["id"] in shadow_ids
                    ],
                    "set_disagreement": disagreement,
                    "served": False,
                },
            }
            explanation = (
                "Uniform exploration draw from eligible candidates; its selection probability is logged."
                if exploring
                else item["explanation"]
            )
            db.add(
                Impression(
                    id=impression_id,
                    session_hash=record.session_hash,
                    item_id=item["id"],
                    position=position,
                    propensity=propensity,
                    arm=record.arm,
                    experiment_id=EXPERIMENT,
                    model_version=model,
                    policy="uniform" if exploring else "deterministic",
                    candidate_pool=pool_ids,
                    trace=trace,
                )
            )
            response_items.append(
                {
                    **{
                        name: value
                        for name, value in item.items()
                        if name not in {"scores", "source_item_id"}
                    },
                    "impression_id": impression_id,
                    "position": position,
                    "propensity": propensity,
                    "explanation": explanation,
                    "trace": trace,
                }
            )
        record.revision += 1
        record.updated_at = utcnow()
        result = {
            "revision": record.revision,
            "model_version": model,
            "arm": record.arm,
            "experiment_id": EXPERIMENT,
            "backend": "live-api",
            "items": response_items,
            "statement": STATEMENT,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        record.latest_shelf = result
        db.flush()
        LOGGER.info(
            json.dumps(
                {
                    "event": "shelf_generated",
                    "session_hash": record.session_hash,
                    "revision": record.revision,
                    "model_version": model,
                    "arm": record.arm,
                    "impressions": len(response_items),
                }
            )
        )
        return result

    def store_feedback(db: Session, record: SessionRecord, event: EventRequest) -> Feedback:
        impression = db.get(Impression, event.impression_id)
        if impression is None or impression.session_hash != record.session_hash:
            raise HTTPException(404, "Impression not found in this session")
        if db.scalar(
            select(Feedback.id)
            .where(Feedback.impression_id == event.impression_id, Feedback.event == event.event)
            .limit(1)
        ):
            raise HTTPException(409, "This feedback event was already recorded")
        feedback = Feedback(
            id=str(uuid.uuid4()), impression_id=event.impression_id, event=event.event, rating=event.rating
        )
        db.add(feedback)
        if event.event != "rating" or (event.rating is not None and event.rating >= 4):
            record.history = [*record.history, impression.item_id][-200:]
        db.flush()
        return feedback

    @application.get("/health")
    def health() -> dict[str, Any]:
        source = catalog()
        with engine.connect() as connection:
            connection.execute(select(1))
        return {
            "status": "ok",
            "backend": "live-api",
            "catalog_size": len(source.items),
            "statement": STATEMENT,
        }

    @application.get("/v1/catalog")
    def get_catalog(request: Request) -> dict[str, Any]:
        limited(request, False)
        return {"items": list(catalog().items.values()), "statement": STATEMENT}

    @application.post("/v1/session", status_code=201)
    def open_session(body: SessionCreate, request: Request) -> dict[str, Any]:
        limited(request)
        with write_lock, Session(engine) as db:
            token, record = create_session(db, body.reader_id)
            shelf = make_shelf(db, record)
            db.commit()
            return {"session_id": token, **shelf}

    @application.get("/v1/recommend/{user_id}")
    def recommend(
        user_id: str,
        request: Request,
        session_id: str | None = None,
        limit: Annotated[int, Query(ge=1, le=20)] = 10,
    ) -> dict[str, Any]:
        limited(request)
        if user_id != "new" and user_id not in catalog().readers:
            raise HTTPException(404, "Public sample reader not found")
        with write_lock, Session(engine) as db:
            if session_id is None:
                token, record = create_session(db, None if user_id == "new" else user_id)
            else:
                token, record = session_id, get_session(db, session_id)
            shelf = make_shelf(db, record, limit)
            try:
                db.commit()
            except StaleDataError as error:
                raise HTTPException(409, "Session changed concurrently; retry this request") from error
            return {"session_id": token, **shelf}

    @application.post("/v1/session/{token}/event")
    def session_event(token: str, event: EventRequest, request: Request) -> dict[str, Any]:
        limited(request)
        with write_lock, Session(engine) as db:
            record = get_session(db, token)
            store_feedback(db, record, event)
            shelf = make_shelf(db, record)
            try:
                db.commit()
            except StaleDataError as error:
                raise HTTPException(409, "Session changed concurrently; retry this request") from error
            return {"session_id": token, **shelf}

    @application.get("/v1/session/{token}/stream")
    async def stream(
        token: str,
        request: Request,
        once: bool = False,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        limited(request, False)
        with Session(engine) as db:
            get_session(db, token)
        try:
            previous = int(last_event_id) if last_event_id is not None else -1
        except ValueError as error:
            raise HTTPException(400, "Last-Event-ID must be an integer revision") from error

        async def events() -> Any:
            revision = previous
            heartbeat = time.monotonic()
            while not await request.is_disconnected():
                with Session(engine) as db:
                    record = get_session(db, token)
                    current_revision, shelf = record.revision, record.latest_shelf
                if current_revision > revision:
                    revision = current_revision
                    yield f"id: {revision}\nevent: shelf\ndata: {json.dumps({'session_id': token, **shelf})}\n\n"
                if once:
                    break
                if time.monotonic() - heartbeat > 15:
                    yield ": keepalive\n\n"
                    heartbeat = time.monotonic()
                await asyncio.sleep(0.5)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.get("/v1/similar/{item_id}")
    def similar(
        item_id: str, request: Request, limit: Annotated[int, Query(ge=1, le=20)] = 10, exact: bool = True
    ) -> dict[str, Any]:
        limited(request, False)
        source = catalog()
        if item_id not in source.items:
            raise HTTPException(404, "Item not found")
        rows = sorted(source.similarities.get(item_id, {}).items(), key=lambda row: (-row[1], row[0]))[:limit]
        return {
            "items": [
                {**source.items[identifier], "score": score}
                for identifier, score in rows
                if identifier != item_id
            ],
            "backend": "committed-item-cosine",
            "exact": True,
            "approximate_index_available": False,
            "statement": STATEMENT,
        }

    @application.get("/v1/explain/{impression_id}")
    def explain(impression_id: str, session_id: str, request: Request) -> dict[str, Any]:
        limited(request, False)
        with Session(engine) as db:
            record = get_session(db, session_id)
            impression = db.get(Impression, impression_id)
            if impression is None or impression.session_hash != record.session_hash:
                raise HTTPException(404, "Impression not found in this session")
            return {
                "impression_id": impression.id,
                "item_id": impression.item_id,
                "position": impression.position,
                "propensity": impression.propensity,
                "impression_at": impression.impression_at.isoformat(),
                "arm": impression.arm,
                "trace": impression.trace,
                "statement": STATEMENT,
            }

    @application.get("/v1/experiment/assign")
    def assignment(session_id: str, request: Request) -> dict[str, Any]:
        limited(request, False)
        with Session(engine) as db:
            record = get_session(db, session_id)
            return {
                "experiment_id": EXPERIMENT,
                "arm": record.arm,
                "assignment_probability": 0.5,
                "statement": STATEMENT,
            }

    @application.post("/v1/log", status_code=201)
    def log_feedback(
        body: LogRequest, request: Request, x_admin_token: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        require_admin(x_admin_token)
        limited(request)
        with write_lock, Session(engine) as db:
            record = get_session(db, body.session_id)
            feedback = store_feedback(db, record, body)
            feedback_id = feedback.id
            # Admin ingestion still advances and publishes the session state.
            shelf = make_shelf(db, record)
            db.commit()
            return {"feedback_id": feedback_id, "session_id": body.session_id, **shelf}

    @application.post("/v1/registry/evaluate", status_code=201)
    def registry_evaluate(
        body: RegistryRequest, request: Request, x_admin_token: Annotated[str | None, Header()] = None
    ) -> dict[str, Any]:
        require_admin(x_admin_token)
        limited(request)
        gates = [gate.to_dict() for gate in evaluate_gates(body.metrics)]
        result = {
            "id": str(uuid.uuid4()),
            "candidate_version": body.candidate_version,
            "gates": gates,
            "eligible": all(gate["passed"] for gate in gates),
            "activated": False,
            "statement": STATEMENT,
        }
        with Session(engine) as db:
            db.add(
                RegistryDecision(
                    id=result["id"],
                    candidate_version=body.candidate_version,
                    metrics=body.metrics,
                    gates=gates,
                    eligible=result["eligible"],
                )
            )
            db.commit()
        return result

    @application.get("/v1/monitoring")
    def monitoring(request: Request, x_admin_token: Annotated[str | None, Header()] = None) -> dict[str, Any]:
        require_admin(x_admin_token)
        limited(request, False)
        with Session(engine) as db:
            impressions = db.scalar(select(func.count()).select_from(Impression)) or 0
            feedback = db.scalar(select(func.count()).select_from(Feedback)) or 0
            return {
                "impressions": impressions,
                "feedback_events": feedback,
                "backend": engine.dialect.name,
                "online_lift": None,
                "note": "Counts only. No online effectiveness or causal lift is claimed.",
                "statement": STATEMENT,
            }

    return application


app = create_app()
