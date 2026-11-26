"""Streaming (single-pass) story clustering.

For each incoming article we consider only clusters active inside a recent time
window, score them with cosine similarity blended with title-token and entity
overlap, and either assign the article or open a new story.  Centroids are
updated incrementally, so memory stays bounded by the number of live clusters
rather than the number of articles seen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.clustering.labeling import extract_entities, story_title
from newstrace.config import Settings, get_settings
from newstrace.logging import get_logger
from newstrace.models import Article, ClusterMembership, StoryCluster
from newstrace.representations.embedder import from_blob, to_blob
from newstrace.utils import ensure_utc, jaccard, token_set, utcnow

logger = get_logger(__name__)

Vector = NDArray[np.float32]

STOP_TITLE_TOKENS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "with",
    "as",
    "at",
    "by",
    "from",
    "is",
    "are",
    "be",
    "it",
    "its",
    "this",
    "that",
    "after",
    "over",
    "says",
    "say",
    "new",
}


def default_threshold(settings: Settings) -> float:
    """The tuned threshold for whichever embedder is actually active.

    Cosine similarity is not comparable across embedding backends: the
    deterministic hashing fallback puts paraphrases at lower similarity than the
    sentence-transformer does, so reusing one threshold for both silently
    over-fragments stories.
    """
    from newstrace.representations.embedder import get_embedder

    name = getattr(get_embedder(settings), "name", "")
    if name.startswith("hashing"):
        return settings.cluster_threshold_hashing
    return settings.cluster_threshold


@dataclass
class Assignment:
    """Result of assigning one article to a story."""

    article_id: int
    story_id: int
    similarity: float
    created_new: bool
    method: str = "online_cosine_title"


def content_tokens(text: str) -> set[str]:
    return {t for t in token_set(text) if t not in STOP_TITLE_TOKENS and len(t) > 2}


def combined_score(
    cosine: float,
    title_overlap: float,
    entity_overlap: float,
    *,
    title_weight: float,
    entity_weight: float,
    gap_days: float = 0.0,
    time_weight: float = 0.0,
) -> float:
    """Blend the three signals, keeping the result in ``[0, 1]``.

    Cosine similarity carries the decision; shared title tokens and shared
    entities act as a *bonus*. Weighting them as a convex mixture instead would
    drag every score down, because token-overlap Jaccards are small even for two
    reports of the same event -- which shifts the threshold away from any
    interpretable cosine value.

    ``gap_days`` (see ``time_gap_days``) is a *penalty*, off by default: the
    candidate window already bounds how far apart a story's articles can be.
    """
    score = (
        cosine
        + title_weight * title_overlap
        + entity_weight * entity_overlap
        - time_weight * gap_days
    )
    return float(min(1.0, max(0.0, score)))


def time_gap_days(when: datetime | None, cluster: StoryCluster) -> float:
    """Days from ``when`` to the nearest end of the story's span; 0 inside it."""
    published = ensure_utc(when)
    first = ensure_utc(cluster.first_published_at)
    last = ensure_utc(cluster.last_published_at)
    if published is None or first is None or last is None:
        return 0.0
    if published > last:
        return (published - last) / timedelta(days=1)
    if published < first:
        return (first - published) / timedelta(days=1)
    return 0.0


class OnlineClusterer:
    """Incremental clusterer operating on one representation at a time."""

    def __init__(
        self,
        session: Session,
        *,
        settings: Settings | None = None,
        threshold: float | None = None,
        window_hours: int | None = None,
        embedding_method: str = "full",
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.threshold = default_threshold(self.settings) if threshold is None else threshold
        self.window_hours = (
            self.settings.cluster_window_hours if window_hours is None else window_hours
        )
        self.embedding_method = embedding_method
        self.title_weight = self.settings.cluster_title_overlap_weight
        self.entity_weight = self.settings.cluster_entity_overlap_weight
        self.time_weight = self.settings.cluster_time_gap_weight

    # -- candidate selection -------------------------------------------------
    def candidate_clusters(self, when: datetime | None, topic: str | None) -> list[StoryCluster]:
        stmt = select(StoryCluster).where(StoryCluster.embedding_method == self.embedding_method)
        if topic:
            stmt = stmt.where((StoryCluster.topic == topic) | (StoryCluster.topic.is_(None)))
        anchor = ensure_utc(when) or utcnow()
        low = anchor - timedelta(hours=self.window_hours)
        high = anchor + timedelta(hours=self.window_hours)
        stmt = stmt.where(
            (StoryCluster.last_published_at.is_(None))
            | ((StoryCluster.last_published_at >= low) & (StoryCluster.first_published_at <= high))
        )
        return list(self.session.execute(stmt).scalars())

    @staticmethod
    def _validated_vectors(vectors: Vector, *, rows: int | None = None) -> Vector:
        """Validate before float32 storage can discard or overflow values."""
        expected_ndim = 1 if rows is None else 2
        if vectors.ndim != expected_ndim or vectors.shape[-1] == 0:
            raise ValueError("clustering vectors must have a nonempty feature dimension")
        if rows is not None and vectors.shape[0] != rows:
            raise ValueError("clustering vectors must have exactly one row per article")
        if vectors.dtype.kind not in "fiu" or not np.isfinite(vectors).all():
            raise ValueError("clustering vectors must contain finite real numbers")
        if (np.abs(vectors.astype(np.float64)) > np.finfo(np.float32).max).any():
            raise ValueError("clustering vectors must be representable in float32 storage")
        return vectors.astype(np.float32, copy=False)

    @staticmethod
    def _centroid(cluster: StoryCluster, dimension: int) -> NDArray[np.float64]:
        if cluster.centroid_dimension not in (None, dimension):
            raise ValueError(
                f"story {cluster.id} centroid dimension {cluster.centroid_dimension} "
                f"does not match vector dimension {dimension}; rebuild clustering "
                "when changing embedding dimensions"
            )
        centroid = from_blob(cluster.centroid_blob, cluster.centroid_dimension or dimension)
        if not np.isfinite(centroid).all():
            raise ValueError(f"story {cluster.id} has a nonfinite centroid; rebuild clustering")
        return centroid.astype(np.float64)

    # -- scoring -------------------------------------------------------------
    @staticmethod
    def article_features(article: Article) -> tuple[set[str], set[str]]:
        return content_tokens(article.title), set(extract_entities(article.body_text))

    def signals(
        self,
        article: Article,
        vector: Vector,
        cluster: StoryCluster,
        *,
        features: tuple[set[str], set[str]] | None = None,
    ) -> tuple[float, float, float, float]:
        """The raw (cosine, title overlap, entity overlap, gap in days) that
        ``score`` blends.

        ``features`` is the article's (title tokens, entities), which ``assign``
        extracts once rather than per candidate: entity extraction dominates the
        cost, and an article is scored against every story in the window."""
        centroid = self._centroid(cluster, vector.shape[0])
        wide_vector = vector.astype(np.float64)
        denom = float(np.linalg.norm(wide_vector) * np.linalg.norm(centroid))
        cosine = float(np.dot(wide_vector, centroid) / denom) if denom > 1e-9 else 0.0
        cluster_title_tokens = set((cluster.extra or {}).get("title_tokens") or [])
        cluster_entities = set((cluster.extra or {}).get("entities") or [])
        title_tokens, entities = features or self.article_features(article)
        return (
            cosine,
            jaccard(title_tokens, cluster_title_tokens),
            jaccard(entities, cluster_entities),
            time_gap_days(article.published_at, cluster),
        )

    def score(
        self,
        article: Article,
        vector: Vector,
        cluster: StoryCluster,
        *,
        features: tuple[set[str], set[str]] | None = None,
    ) -> float:
        """Score one candidate; see ``signals`` for ``features``."""
        cosine, title_overlap, entity_overlap, gap_days = self.signals(
            article, vector, cluster, features=features
        )
        return combined_score(
            cosine,
            title_overlap,
            entity_overlap,
            title_weight=self.title_weight,
            entity_weight=self.entity_weight,
            gap_days=gap_days,
            time_weight=self.time_weight,
        )

    # -- mutation ------------------------------------------------------------
    def _new_cluster(self, article: Article, vector: Vector) -> StoryCluster:
        published = ensure_utc(article.published_at)
        cluster = StoryCluster(
            display_title=story_title([article.title]),
            topic=article.topic,
            centroid_version=1,
            centroid_blob=to_blob(vector),
            centroid_dimension=int(vector.shape[0]),
            embedding_method=self.embedding_method,
            article_count=0,
            distinct_domain_count=0,
            first_published_at=published,
            last_published_at=published,
            extra={
                "title_tokens": sorted(content_tokens(article.title)),
                "entities": sorted(extract_entities(article.body_text)),
                "domains": [],
            },
        )
        self.session.add(cluster)
        self.session.flush()
        return cluster

    def _update_cluster(self, cluster: StoryCluster, article: Article, vector: Vector) -> None:
        """Incremental centroid update: c <- c + (x - c)/n over non-duplicates."""
        dim = cluster.centroid_dimension or int(vector.shape[0])
        centroid = self._centroid(cluster, dim)
        n_effective = max(1, cluster.article_count)
        if not article.is_near_duplicate:
            centroid = centroid + (vector.astype(np.float64) - centroid) / float(n_effective + 1)
            norm = float(np.linalg.norm(centroid))
            if norm > 1e-9:
                centroid = centroid / norm
            cluster.centroid_blob = to_blob(centroid.astype(np.float32))
            cluster.centroid_version += 1

        cluster.article_count += 1
        extra = dict(cluster.extra or {})
        domains = set(extra.get("domains") or [])
        if not article.is_near_duplicate:
            domains.add(article.source_domain)
        extra["domains"] = sorted(domains)
        extra["title_tokens"] = sorted(
            set(extra.get("title_tokens") or []) | content_tokens(article.title)
        )[:64]
        extra["entities"] = sorted(
            set(extra.get("entities") or []) | set(extract_entities(article.body_text))
        )[:64]
        cluster.extra = extra
        cluster.distinct_domain_count = len(domains)

        published = ensure_utc(article.published_at)
        if published is not None:
            first = ensure_utc(cluster.first_published_at)
            last = ensure_utc(cluster.last_published_at)
            cluster.first_published_at = min(first, published) if first else published
            cluster.last_published_at = max(last, published) if last else published
        cluster.updated_at = utcnow()

    def assign(self, article: Article, vector: Vector) -> Assignment:
        """Assign one article to the best candidate story, or create a new one."""
        vector = self._validated_vectors(vector)
        existing = (
            self.session.execute(
                select(ClusterMembership).where(ClusterMembership.article_id == article.id)
            )
            .scalars()
            .first()
        )
        if existing is not None:
            return Assignment(article.id, existing.story_cluster_id, existing.similarity, False)

        # Duplicates always follow the article they duplicate.
        if article.is_near_duplicate and article.duplicate_of_article_id:
            parent = (
                self.session.execute(
                    select(ClusterMembership).where(
                        ClusterMembership.article_id == article.duplicate_of_article_id
                    )
                )
                .scalars()
                .first()
            )
            if parent is not None:
                cluster = self.session.get(StoryCluster, parent.story_cluster_id)
                if cluster is not None:
                    self._record(article, cluster, vector, 1.0, "duplicate_link")
                    return Assignment(article.id, cluster.id, 1.0, False, "duplicate_link")

        best_cluster: StoryCluster | None = None
        best_score = -1.0
        features = self.article_features(article)
        for cluster in self.candidate_clusters(article.published_at, article.topic):
            score = self.score(article, vector, cluster, features=features)
            if score > best_score:
                best_cluster, best_score = cluster, score

        if best_cluster is not None and best_score >= self.threshold:
            self._record(article, best_cluster, vector, best_score, "online_cosine_title")
            return Assignment(article.id, best_cluster.id, best_score, False)

        cluster = self._new_cluster(article, vector)
        self._record(article, cluster, vector, 1.0, "new_cluster")
        return Assignment(article.id, cluster.id, 1.0, True, "new_cluster")

    def _record(
        self,
        article: Article,
        cluster: StoryCluster,
        vector: Vector,
        score: float,
        method: str,
    ) -> None:
        # Duplicate links bypass scoring, so check the destination before adding
        # a pending membership or updating any story metadata.
        self._centroid(cluster, vector.shape[0])
        self.session.add(
            ClusterMembership(
                article_id=article.id,
                story_cluster_id=cluster.id,
                similarity=float(score),
                assignment_method=method,
                assigned_at=utcnow(),
            )
        )
        self._update_cluster(cluster, article, vector)
        if not article.is_near_duplicate and cluster.article_count <= 1:
            cluster.display_title = story_title([article.title])
        self.session.flush()

    def assign_many(
        self, articles: Sequence[Article], vectors: NDArray[np.float32]
    ) -> list[Assignment]:
        """Assign in publication order after validating the whole input matrix.

        Matrix shape/value errors do not partially assign a batch. Stored-state
        incompatibilities and database/runtime failures still require the caller
        to roll back its transaction.
        """
        vectors = self._validated_vectors(vectors, rows=len(articles))
        order = sorted(
            range(len(articles)),
            key=lambda i: (ensure_utc(articles[i].published_at) or utcnow(), articles[i].id),
        )
        assignments: list[Assignment] = []
        for i in order:
            assignments.append(self.assign(articles[i], vectors[i]))
        return assignments
