"""Story-level read services: listing, timelines and source accounting.

Duplicate articles are always visible but never counted as independent sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newstrace.clustering.labeling import keyword_summary, split_sentences
from newstrace.models import Article, ClusterMembership, StoryCluster
from newstrace.utils import ensure_utc, jaccard, token_set, truncate


@dataclass
class StoryArticles:
    """A story with its articles split into independent and duplicate sets."""

    story: StoryCluster
    articles: list[Article] = field(default_factory=list)
    similarity_by_article: dict[int, float] = field(default_factory=dict)

    @property
    def independent(self) -> list[Article]:
        return [a for a in self.articles if not a.is_near_duplicate]

    @property
    def duplicates(self) -> list[Article]:
        return [a for a in self.articles if a.is_near_duplicate]

    @property
    def independent_domains(self) -> list[str]:
        return sorted({a.source_domain for a in self.independent if a.source_domain})

    def sorted_by_time(self, *, include_duplicates: bool = False) -> list[Article]:
        pool = self.articles if include_duplicates else self.independent
        return sorted(
            pool,
            key=lambda a: (ensure_utc(a.published_at) or datetime.max.replace(tzinfo=None), a.id),
        )


def load_story(session: Session, story_id: int) -> StoryArticles | None:
    story = session.get(StoryCluster, story_id)
    if story is None:
        return None
    rows = session.execute(
        select(Article, ClusterMembership.similarity)
        .join(ClusterMembership, ClusterMembership.article_id == Article.id)
        .where(ClusterMembership.story_cluster_id == story_id)
    ).all()
    bundle = StoryArticles(story=story)
    for article, similarity in rows:
        bundle.articles.append(article)
        bundle.similarity_by_article[article.id] = float(similarity)
    return bundle


def list_stories(
    session: Session,
    *,
    topic: str | None = None,
    limit: int = 25,
    offset: int = 0,
    min_articles: int = 1,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> list[StoryCluster]:
    stmt = select(StoryCluster).where(StoryCluster.article_count >= min_articles)
    if topic:
        stmt = stmt.where(StoryCluster.topic == topic)
    if start_time:
        stmt = stmt.where(StoryCluster.last_published_at >= start_time)
    if end_time:
        stmt = stmt.where(StoryCluster.first_published_at <= end_time)
    stmt = stmt.order_by(
        StoryCluster.distinct_domain_count.desc(),
        StoryCluster.last_published_at.desc().nulls_last(),
        StoryCluster.id.desc(),
    )
    return list(session.execute(stmt.offset(offset).limit(limit)).scalars())


def count_stories(session: Session, *, topic: str | None = None) -> int:
    stmt = select(func.count(StoryCluster.id))
    if topic:
        stmt = stmt.where(StoryCluster.topic == topic)
    return int(session.execute(stmt).scalar_one())


@dataclass
class TimelineEntry:
    """One dated step in a story, with the sentences that look new at that point."""

    article: Article
    published_at: datetime | None
    similarity: float
    is_near_duplicate: bool
    duplicate_of_article_id: int | None
    new_sentences: list[str] = field(default_factory=list)
    repeated_sentences: list[str] = field(default_factory=list)
    is_first_report: bool = False


NOVELTY_THRESHOLD = 0.6


def sentence_is_new(
    sentence: str, seen: list[set[str]], threshold: float = NOVELTY_THRESHOLD
) -> bool:
    """A sentence is "potentially new" when it is dissimilar to all prior evidence."""
    tokens = token_set(sentence)
    if len(tokens) < 4:
        return False
    return all(jaccard(tokens, prior) < threshold for prior in seen)


def build_timeline(
    bundle: StoryArticles, *, include_duplicates: bool = True
) -> list[TimelineEntry]:
    """Order a story's articles by publication time and mark new information.

    The earliest article in the dataset is labelled ``is_first_report``; this is
    the earliest *available* report, not proof of who reported it first.
    """
    entries: list[TimelineEntry] = []
    seen: list[set[str]] = []
    ordered = sorted(
        bundle.articles,
        key=lambda a: (
            ensure_utc(a.published_at) or datetime.max.replace(tzinfo=None),
            a.is_near_duplicate,
            a.id,
        ),
    )
    first_independent_seen = False
    for article in ordered:
        if article.is_near_duplicate and not include_duplicates:
            continue
        new_sentences: list[str] = []
        repeated: list[str] = []
        if not article.is_near_duplicate:
            for sentence in split_sentences(article.body_text):
                if sentence_is_new(sentence, seen):
                    new_sentences.append(truncate(sentence, 320))
                    seen.append(token_set(sentence))
                else:
                    repeated.append(truncate(sentence, 320))
        is_first = not article.is_near_duplicate and not first_independent_seen
        if is_first:
            first_independent_seen = True
        entries.append(
            TimelineEntry(
                article=article,
                published_at=ensure_utc(article.published_at),
                similarity=bundle.similarity_by_article.get(article.id, 0.0),
                is_near_duplicate=bool(article.is_near_duplicate),
                duplicate_of_article_id=article.duplicate_of_article_id,
                new_sentences=new_sentences,
                repeated_sentences=repeated,
                is_first_report=is_first,
            )
        )
    return entries


def story_keywords(bundle: StoryArticles, *, top_k: int = 8) -> list[str]:
    return keyword_summary([a.body_text for a in bundle.independent], top_k=top_k)


def story_stats(session: Session, story_id: int) -> dict[str, int]:
    bundle = load_story(session, story_id)
    if bundle is None:
        return {}
    return {
        "article_count": len(bundle.articles),
        "independent_article_count": len(bundle.independent),
        "duplicate_count": len(bundle.duplicates),
        "distinct_domain_count": len(bundle.independent_domains),
    }
