"""Grounded summary endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from newstrace.db import get_db
from newstrace.schemas import SummaryRequest, SummaryResponse
from newstrace.summarization.extractive import ExtractiveSummarizer
from newstrace.utils import utcnow

router = APIRouter(prefix="/summaries", tags=["summaries"])


@router.post("/story/{story_id}", response_model=SummaryResponse)
def summarize_story(
    story_id: int,
    payload: SummaryRequest | None = None,
    session: Session = Depends(get_db),
) -> SummaryResponse:
    request = payload or SummaryRequest()
    try:
        if request.method == "llm":
            from newstrace.summarization.llm import LLMSummarizer

            summarizer = LLMSummarizer(session, max_per_section=request.max_per_section)
        else:
            summarizer = ExtractiveSummarizer(  # type: ignore[assignment]
                session, max_per_section=request.max_per_section
            )
        summary = summarizer.summarize(story_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    summary.generated_at = utcnow()
    return SummaryResponse(
        summary=summary,
        citation_coverage=summary.citation_coverage(),
        statement_count=len(summary.statements),
    )
