"""Story listing, detail, timeline and claims."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from newstrace.api.serializers import (
    claim_to_schema,
    story_detail_to_schema,
    story_to_schema,
    timeline_entry_to_schema,
)
from newstrace.claims.evidence import load_claims
from newstrace.db import get_db
from newstrace.schemas import (
    ClaimsResponse,
    StoryDetailResponse,
    StorySummaryResponse,
    TimelineResponse,
)
from newstrace.stories import build_timeline, list_stories, load_story

router = APIRouter(tags=["stories"])


@router.get("/stories", response_model=list[StorySummaryResponse])
def get_stories(
    session: Session = Depends(get_db),
    topic: str | None = None,
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    min_articles: int = Query(default=1, ge=1),
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> list[StorySummaryResponse]:
    stories = list_stories(
        session,
        topic=topic,
        limit=limit,
        offset=offset,
        min_articles=min_articles,
        start_time=start_time,
        end_time=end_time,
    )
    out: list[StorySummaryResponse] = []
    for story in stories:
        bundle = load_story(session, story.id)
        out.append(story_to_schema(story, bundle))
    return out


@router.get("/stories/{story_id}", response_model=StoryDetailResponse)
def get_story(story_id: int, session: Session = Depends(get_db)) -> StoryDetailResponse:
    bundle = load_story(session, story_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail=f"story {story_id} not found")
    return story_detail_to_schema(bundle)


@router.get("/stories/{story_id}/timeline", response_model=TimelineResponse)
def get_timeline(
    story_id: int,
    session: Session = Depends(get_db),
    include_duplicates: bool = True,
) -> TimelineResponse:
    bundle = load_story(session, story_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail=f"story {story_id} not found")
    entries = build_timeline(bundle, include_duplicates=include_duplicates)
    return TimelineResponse(
        story_id=story_id,
        display_title=bundle.story.display_title,
        entries=[timeline_entry_to_schema(e) for e in entries],
    )


@router.get("/stories/{story_id}/claims", response_model=ClaimsResponse)
def get_claims(story_id: int, session: Session = Depends(get_db)) -> ClaimsResponse:
    bundle = load_story(session, story_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail=f"story {story_id} not found")
    claims = load_claims(session, story_id)
    return ClaimsResponse(
        story_id=story_id,
        claims=[claim_to_schema(session, claim, evidence) for claim, evidence in claims],
    )
