"""Health and system-status endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newstrace import __version__
from newstrace.config import get_llm_settings, get_settings
from newstrace.db import get_db
from newstrace.ingestion.pipeline import latest_run
from newstrace.models import Article, StoryCluster
from newstrace.representations.embedder import detect_device, get_embedder
from newstrace.representations.registry import available_representations
from newstrace.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(session: Session = Depends(get_db)) -> HealthResponse:
    settings = get_settings()
    llm = get_llm_settings()
    embedder = get_embedder(settings)
    run = latest_run(session)
    article_count = int(session.execute(select(func.count(Article.id))).scalar_one())
    duplicate_count = int(
        session.execute(
            select(func.count(Article.id)).where(Article.is_near_duplicate.is_(True))
        ).scalar_one()
    )
    story_count = int(session.execute(select(func.count(StoryCluster.id))).scalar_one())
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.env,
        database=settings.database_url,
        embedding_model=embedder.name,
        embedding_backend=settings.embedding_backend,
        embedding_dimension=embedder.dimension,
        device=detect_device(),
        llm_provider=llm.llm_provider,
        article_count=article_count,
        story_count=story_count,
        duplicate_count=duplicate_count,
        last_ingestion_run=(
            {
                "id": run.id,
                "source": run.source,
                "status": run.status,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "fetched": run.fetched_count,
                "inserted": run.inserted_count,
                "duplicates": run.duplicate_count,
                "failures": run.failure_count,
            }
            if run
            else None
        ),
        representations=available_representations(session),
    )
