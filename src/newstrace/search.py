"""Search service: representation selection, filtering, reranking, explanation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.config import Settings, get_settings
from newstrace.logging import get_logger
from newstrace.models import Article, ClusterMembership
from newstrace.representations.projection import load_projector
from newstrace.representations.registry import load_vectors
from newstrace.retrieval.exact import DenseRetriever, ScoredHit, TfidfRetriever
from newstrace.retrieval.explain import evidence_excerpt, explain_hit
from newstrace.retrieval.rerank import blend_pagerank, rerank_with_signals
from newstrace.utils import ensure_utc

logger = get_logger(__name__)


@dataclass
class SearchFilters:
    topic: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    source_domain: str | None = None
    language: str | None = None
    entity: str | None = None
    include_duplicates: bool = False


@dataclass
class SearchRequest:
    query: str
    filters: SearchFilters = field(default_factory=SearchFilters)
    method: str = "full"
    dimension: int | None = None
    fit_version: str | None = None
    rerank_with_pagerank: bool = False
    apply_signal_rerank: bool = True
    top_k: int = 10


@dataclass
class SearchResultItem:
    article: Article
    hit: ScoredHit
    story_id: int | None
    excerpt: str
    explanation: str


@dataclass
class SearchResponse:
    query: str
    method: str
    items: list[SearchResultItem] = field(default_factory=list)
    took_ms: float = 0.0
    candidates_considered: int = 0
    notes: list[str] = field(default_factory=list)


def _matching_article_ids(session: Session, filters: SearchFilters) -> set[int]:
    stmt = select(Article.id)
    if filters.topic:
        stmt = stmt.where(Article.topic == filters.topic)
    if filters.start_time:
        stmt = stmt.where(Article.published_at >= filters.start_time)
    if filters.end_time:
        stmt = stmt.where(Article.published_at <= filters.end_time)
    if filters.source_domain:
        stmt = stmt.where(Article.source_domain == filters.source_domain.lower())
    if filters.language:
        stmt = stmt.where(Article.language == filters.language.lower())
    if not filters.include_duplicates:
        stmt = stmt.where(Article.is_near_duplicate.is_(False))
    ids = set(session.execute(stmt).scalars())
    if filters.entity:
        needle = filters.entity.lower()
        keep: set[int] = set()
        for article in session.execute(select(Article).where(Article.id.in_(ids))).scalars():
            if needle in article.body_text.lower():
                keep.add(article.id)
        ids = keep
    return ids


def _story_map(session: Session, article_ids: list[int]) -> dict[int, int]:
    rows = session.execute(
        select(ClusterMembership.article_id, ClusterMembership.story_cluster_id).where(
            ClusterMembership.article_id.in_(article_ids)
        )
    ).all()
    return {int(article_id): int(story_id) for article_id, story_id in rows}


def search(
    session: Session, request: SearchRequest, *, settings: Settings | None = None
) -> SearchResponse:
    """Run one search end to end and return explained results."""
    settings = settings or get_settings()
    started = time.perf_counter()
    notes: list[str] = []
    allowed = _matching_article_ids(session, request.filters)
    if not allowed:
        return SearchResponse(
            request.query, request.method, [], 0.0, 0, ["no articles matched the filters"]
        )

    if request.method == "tfidf":
        articles = list(session.execute(select(Article).where(Article.id.in_(allowed))).scalars())
        retriever: Any = TfidfRetriever([a.id for a in articles], [a.body_text for a in articles])
        hits = retriever.search(request.query, k=max(request.top_k * 4, request.top_k))
    else:
        vectors = load_vectors(
            session,
            method=request.method,
            fit_version=request.fit_version,
            dimension=request.dimension,
        )
        if len(vectors) == 0:
            if request.method != "full":
                notes.append(
                    f"no stored '{request.method}' vectors (dimension={request.dimension}); "
                    "falling back to the full-dimensional baseline"
                )
                vectors = load_vectors(session, method="full")
            if len(vectors) == 0:
                return SearchResponse(
                    request.query, request.method, [], 0.0, 0, ["no embeddings available"]
                )
        projector = _load_projector_for(session, vectors.method, vectors.fit_version)
        retriever = DenseRetriever(
            vectors.article_ids, vectors.matrix, method=vectors.method, projector=projector
        )
        hits = retriever.search(
            request.query, k=max(request.top_k * 4, request.top_k), allowed_ids=allowed
        )

    hits = [h for h in hits if h.article_id in allowed]
    candidates = len(hits)
    articles_by_id = {
        a.id: a
        for a in session.execute(
            select(Article).where(Article.id.in_([h.article_id for h in hits]))
        ).scalars()
    }

    if request.apply_signal_rerank:
        hits = rerank_with_signals(hits, articles_by_id)

    if request.rerank_with_pagerank and hits:
        from newstrace.graph.build import build_graph
        from newstrace.graph.pagerank import personalized_pagerank, seed_from_articles

        pool_ids = list(articles_by_id)
        pool = [articles_by_id[i] for i in pool_ids]
        graph = build_graph(session, articles=pool, include_claims=False)
        seeds = seed_from_articles([h.article_id for h in hits], [max(h.score, 0.0) for h in hits])
        result = personalized_pagerank(graph, personalization=seeds)
        hits = blend_pagerank(hits, result.article_scores())
        notes.append(
            f"personalized PageRank over {graph.number_of_nodes()} nodes, "
            f"{result.iterations} iterations (converged={result.converged})"
        )

    hits = hits[: request.top_k]
    story_by_article = _story_map(session, [h.article_id for h in hits])

    items: list[SearchResultItem] = []
    for hit in hits:
        article = articles_by_id.get(hit.article_id)
        if article is None:
            continue
        items.append(
            SearchResultItem(
                article=article,
                hit=hit,
                story_id=story_by_article.get(article.id),
                excerpt=evidence_excerpt(article, request.query),
                explanation=explain_hit(hit, article, request.query),
            )
        )

    return SearchResponse(
        query=request.query,
        method=hits[0].method if hits else request.method,
        items=items,
        took_ms=(time.perf_counter() - started) * 1000.0,
        candidates_considered=candidates,
        notes=notes,
    )


def _load_projector_for(session: Session, method: str, fit_version: str) -> Any:
    """Load the saved projector matching a stored representation, if any."""
    if method == "full":
        return None
    from pathlib import Path

    from newstrace.models import ProjectionArtifact

    artifact = session.execute(
        select(ProjectionArtifact).where(ProjectionArtifact.fit_version == fit_version)
    ).scalar_one_or_none()
    if artifact is None:
        logger.warning(
            "No saved projector for fit_version=%s; queries may be mis-scaled", fit_version
        )
        return None
    path = Path(artifact.path)
    if not path.exists():
        logger.warning("Projector artifact missing on disk: %s", path)
        return None
    return load_projector(path)


def query_latency_benchmark(
    session: Session, queries: list[str], *, method: str = "full", dimension: int | None = None
) -> dict[str, float]:
    """p50/p95 latency for the given representation."""
    vectors = load_vectors(session, method=method, dimension=dimension)
    if len(vectors) == 0:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "queries": 0.0}
    retriever = DenseRetriever(vectors.article_ids, vectors.matrix, method=method)
    stats = retriever.benchmark(queries)
    stats["index_memory_mib"] = retriever.memory_bytes() / (1024 * 1024)
    return stats


def newest_article_time(session: Session) -> datetime | None:
    value = session.execute(
        select(Article.published_at).order_by(Article.published_at.desc().nulls_last()).limit(1)
    ).scalar_one_or_none()
    return ensure_utc(value)


def as_float(value: Any) -> float:
    return float(np.asarray(value).item())
