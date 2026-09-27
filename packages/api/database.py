"""Portable SQLAlchemy tables for SQLite and PostgreSQL."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class SessionRecord(Base):
    __tablename__ = "sessions"
    session_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    arm: Mapped[str] = mapped_column(String(24))
    history: Mapped[list[str]] = mapped_column(JSON, default=list)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    latest_shelf: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    __mapper_args__ = {"version_id_col": revision, "version_id_generator": False}  # noqa: RUF012


class SessionContext(Base):
    __tablename__ = "session_context"
    session_hash: Mapped[str] = mapped_column(ForeignKey("sessions.session_hash"), primary_key=True)
    reader_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    seen: Mapped[list[str]] = mapped_column(JSON, default=list)
    genre: Mapped[str] = mapped_column(String(80), default="All books")


class Impression(Base):
    __tablename__ = "impressions"
    __table_args__ = (
        CheckConstraint("position > 0", name="valid_position"),
        CheckConstraint("propensity > 0 AND propensity <= 1", name="valid_propensity"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_hash: Mapped[str] = mapped_column(ForeignKey("sessions.session_hash"), index=True)
    item_id: Mapped[str] = mapped_column(String(64), index=True)
    position: Mapped[int] = mapped_column(Integer)
    propensity: Mapped[float] = mapped_column(Float, nullable=False)
    arm: Mapped[str] = mapped_column(String(24))
    experiment_id: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(80))
    policy: Mapped[str] = mapped_column(String(40))
    candidate_pool: Mapped[list[str]] = mapped_column(JSON)
    impression_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    trace: Mapped[dict[str, Any]] = mapped_column(JSON)


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint("rating IS NULL OR (rating >= 1 AND rating <= 5)", name="valid_rating"),
        UniqueConstraint("impression_id", "event", name="one_event_type_per_impression"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    impression_id: Mapped[str] = mapped_column(ForeignKey("impressions.id"), index=True)
    event: Mapped[str] = mapped_column(String(16))
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feedback_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class RegistryDecision(Base):
    __tablename__ = "registry_decisions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    candidate_version: Mapped[str] = mapped_column(String(80))
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON)
    gates: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    eligible: Mapped[bool]


def create_database(url: str) -> Engine:
    parsed = make_url(url)
    options: dict[str, Any] = {"pool_pre_ping": True}
    if parsed.drivername.startswith("sqlite"):
        if parsed.database and parsed.database != ":memory:":
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
        options["connect_args"] = {"check_same_thread": False}
        if parsed.database in (None, "", ":memory:"):
            options["poolclass"] = StaticPool
    elif parsed.drivername == "postgresql":
        parsed = parsed.set(drivername="postgresql+psycopg")
    engine = create_engine(parsed, **options)
    if parsed.drivername.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def enable_foreign_keys(connection: Any, _: Any) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    Base.metadata.create_all(engine)
    return engine
