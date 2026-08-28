"""Topic listing (configuration itself stays in YAML)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newstrace.config import load_topics
from newstrace.db import get_db
from newstrace.models import Article, StoryCluster
from newstrace.schemas import TopicResponse

router = APIRouter(tags=["topics"])


@router.get("/topics", response_model=list[TopicResponse])
def list_topics(session: Session = Depends(get_db)) -> list[TopicResponse]:
    topics = load_topics()
    article_counts: dict[str | None, int] = {
        topic: int(count)
        for topic, count in session.execute(
            select(Article.topic, func.count(Article.id)).group_by(Article.topic)
        ).all()
    }
    story_counts: dict[str | None, int] = {
        topic: int(count)
        for topic, count in session.execute(
            select(StoryCluster.topic, func.count(StoryCluster.id)).group_by(StoryCluster.topic)
        ).all()
    }
    return [
        TopicResponse(
            key=key,
            display_name=topic.label(),
            gdelt_query=topic.gdelt_query,
            feed_count=len(topic.feeds),
            article_count=article_counts.get(key, 0),
            story_count=story_counts.get(key, 0),
        )
        for key, topic in topics.items()
    ]
