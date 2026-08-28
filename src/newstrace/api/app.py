"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from newstrace import __version__
from newstrace.api.routes import (
    experiments,
    health,
    ingestion,
    search,
    stories,
    summaries,
    topics,
)
from newstrace.config import get_settings
from newstrace.db import create_all, get_engine
from newstrace.logging import configure_logging, get_logger
from newstrace.utils import set_global_seed

logger = get_logger(__name__)

DESCRIPTION = """
NewsTrace groups articles that describe the same event, tracks how a story
changes over time, and produces evidence-grounded summaries that link back to the
original reporting.

NewsTrace is **not** a fact checker. It reports repetition, provenance,
duplication and source diversity. It does not judge whether a claim is true or
whether a publisher is trustworthy.
""".strip()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    set_global_seed(settings.random_seed)
    create_all(get_engine())
    logger.info("NewsTrace API ready (env=%s, db=%s)", settings.env, settings.database_url)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="NewsTrace",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    for module in (health, topics, stories, search, ingestion, summaries, experiments):
        app.include_router(module.router)
    return app


app = create_app()
