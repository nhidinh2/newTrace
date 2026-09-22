"""Database engine/session helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from newstrace.config import REPO_ROOT, get_settings
from newstrace.models import Base

_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker[Session] | None = None


def _normalise_sqlite_url(url: str) -> str:
    """Make relative sqlite paths repo-root relative and ensure the dir exists."""
    prefix = "sqlite:///"
    if not url.startswith(prefix) or url.startswith("sqlite:///:memory:"):
        return url
    raw = url[len(prefix) :]
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"{prefix}{path}"


def build_engine(url: str | None = None) -> Engine:
    settings = get_settings()
    resolved = _normalise_sqlite_url(url or settings.database_url)
    connect_args = {"check_same_thread": False} if resolved.startswith("sqlite") else {}
    engine = create_engine(resolved, future=True, connect_args=connect_args)
    if resolved.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            # A refresh writing while the UI or API reads is the normal case,
            # and without a busy timeout SQLite raises "database is locked"
            # immediately instead of waiting for the writer to commit.
            cursor.execute(f"PRAGMA busy_timeout={int(settings.sqlite_busy_timeout_ms)}")
            # WAL already gives durability across process crashes; NORMAL only
            # risks the last commits on a machine-level power loss, and it is
            # what makes bulk ingestion write at a sane rate.
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    return engine


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = build_engine()
    return _ENGINE


def get_session_factory() -> sessionmaker[Session]:
    global _SESSION_FACTORY
    if _SESSION_FACTORY is None:
        _SESSION_FACTORY = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SESSION_FACTORY


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any exception."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def create_all(engine: Engine | None = None) -> None:
    """Create tables directly (tests and the fixture demo); Alembic owns prod."""
    target = engine or get_engine()
    Base.metadata.create_all(target)
    # The FTS5 table has no ORM model, so metadata.create_all cannot know
    # about it; the migration and this call share one DDL string.
    from newstrace.retrieval.fulltext import create_index

    create_index(target)


def reset_engine(url: str | None = None) -> Engine:
    """Rebind the global engine, e.g. to a temporary test database."""
    global _ENGINE, _SESSION_FACTORY
    _ENGINE = build_engine(url)
    _SESSION_FACTORY = sessionmaker(bind=_ENGINE, expire_on_commit=False, future=True)
    return _ENGINE
