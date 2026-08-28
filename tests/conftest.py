"""Shared pytest fixtures.

Tests run entirely offline: the embedder is forced to the deterministic hashing
backend so no model download is required and vectors are byte-identical across
machines.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

os.environ.setdefault("NEWSTRACE_EMBEDDING_BACKEND", "hashing")
os.environ.setdefault("NEWSTRACE_RANDOM_SEED", "549")

FIXTURES = Path(__file__).parent / "fixtures"
BASE_TIME = datetime(2026, 8, 18, 6, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point every test at a temporary database, cache and artifacts directory."""
    from newstrace import config as config_module
    from newstrace.representations import embedder as embedder_module

    monkeypatch.setenv("NEWSTRACE_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("NEWSTRACE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NEWSTRACE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("NEWSTRACE_EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("NEWSTRACE_HTTP_MIN_INTERVAL_SECONDS", "0")
    # Tests pin the clustering threshold so that re-tuning the shipped defaults
    # against the fixture corpus cannot silently change test outcomes. The tuned
    # production values live in newstrace.config.
    monkeypatch.setenv("NEWSTRACE_CLUSTER_THRESHOLD_HASHING", "0.62")
    monkeypatch.setenv("NEWSTRACE_CLUSTER_THRESHOLD", "0.62")
    config_module.reset_caches()
    embedder_module.reset_embedder()
    yield
    config_module.reset_caches()
    embedder_module.reset_embedder()


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    from newstrace.db import create_all, get_session_factory, reset_engine

    reset_engine(f"sqlite:///{tmp_path / 'test.db'}")
    create_all()
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def raw_articles() -> list:
    """Three articles about one event plus one unrelated article and one duplicate."""
    from newstrace.ingestion.base import RawArticle

    return [
        RawArticle(
            url="https://alpha.example/atlas-3-release?utm_source=news",
            title="Northwind Labs releases Atlas 3 agent model",
            description="Northwind Labs said Atlas 3 scores 71.4 percent on the public agent "
            "benchmark. The model is available to developers on Tuesday.",
            source_domain="alpha.example",
            language="English",
            published_at=BASE_TIME,
            topic="ai_models",
            source_adapter="test",
        ),
        RawArticle(
            url="https://beta.example/atlas-3",
            title="Atlas 3 launch: Northwind claims 71.4 percent on agent benchmark",
            description="Northwind Labs announced Atlas 3 on Tuesday. The company said the model "
            "reached 71.4 percent on the public agent benchmark.",
            source_domain="beta.example",
            language="English",
            published_at=BASE_TIME + timedelta(hours=2),
            topic="ai_models",
            source_adapter="test",
        ),
        RawArticle(
            url="https://wire.example/atlas-3-release",
            title="Northwind Labs releases Atlas 3 agent model",
            description="Northwind Labs said Atlas 3 scores 71.4 percent on the public agent "
            "benchmark. The model is available to developers on Tuesday.",
            source_domain="wire.example",
            language="English",
            published_at=BASE_TIME + timedelta(hours=3),
            topic="ai_models",
            source_adapter="test",
        ),
        RawArticle(
            url="https://gamma.example/grid-demand",
            title="Grid operator warns data-centre demand could add 6 gigawatts by 2030",
            description="The regional grid operator said data-centre load could add 6 gigawatts "
            "of demand by 2030 under its high-growth scenario.",
            source_domain="gamma.example",
            language="English",
            published_at=BASE_TIME + timedelta(hours=30),
            topic="chips_and_compute",
            source_adapter="test",
        ),
    ]


@pytest.fixture
def populated_session(session: Session, raw_articles: list) -> Session:
    """A session holding the ingested, embedded and clustered test articles."""
    from newstrace.ingestion.pipeline import persist_articles
    from newstrace.pipeline import index_articles

    result = persist_articles(session, raw_articles)
    session.commit()
    index_articles(session, result.article_ids)
    return session


@pytest.fixture
def api_client(populated_session: Session):
    from fastapi.testclient import TestClient

    from newstrace.api.app import create_app

    with TestClient(create_app()) as client:
        yield client
