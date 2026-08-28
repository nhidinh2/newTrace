"""Reranking of candidate hits.

Two rerankers are provided:

* :func:`rerank_with_signals` -- transparent blend of retrieval score, recency
  and a duplication penalty.
* :func:`rerank_with_pagerank` -- personalized PageRank over the news graph,
  seeded by the retrieved candidates (see :mod:`newstrace.graph`).

No publisher-reputation score is used anywhere: it would be unexplainable.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from newstrace.models import Article
from newstrace.retrieval.exact import ScoredHit, recency_weight
from newstrace.utils import ensure_utc, utcnow


def rerank_with_signals(
    hits: Sequence[ScoredHit],
    articles: dict[int, Article],
    *,
    now: datetime | None = None,
    recency_half_life_hours: float = 48.0,
    recency_weight_factor: float = 0.25,
    duplicate_penalty: float = 0.5,
) -> list[ScoredHit]:
    """Blend retrieval score with recency and a duplicate penalty."""
    now = ensure_utc(now) or utcnow()
    out: list[ScoredHit] = []
    for hit in hits:
        article = articles.get(hit.article_id)
        if article is None:
            continue
        rec = recency_weight(ensure_utc(article.published_at), now, recency_half_life_hours)
        penalty = duplicate_penalty if article.is_near_duplicate else 1.0
        combined = (
            (1.0 - recency_weight_factor) * hit.score + recency_weight_factor * rec
        ) * penalty
        signals = dict(hit.signals)
        signals.update({"recency": rec, "duplicate_penalty": penalty, "combined": combined})
        out.append(
            ScoredHit(
                article_id=hit.article_id,
                score=float(combined),
                rank=hit.rank,
                method=f"{hit.method}+signals",
                signals=signals,
            )
        )
    out.sort(key=lambda h: (-h.score, h.article_id))
    for rank, hit in enumerate(out, start=1):
        hit.rank = rank
    return out


def blend_pagerank(
    hits: Sequence[ScoredHit],
    pagerank: dict[int, float],
    *,
    alpha: float = 0.7,
) -> list[ScoredHit]:
    """Convex blend of the retrieval score with a personalized-PageRank mass.

    ``alpha`` is the weight on the retrieval score; PageRank masses are min-max
    scaled across the candidate set so the blend is scale free.
    """
    if not hits:
        return []
    masses = [pagerank.get(h.article_id, 0.0) for h in hits]
    lo, hi = min(masses), max(masses)
    span = (hi - lo) or 1.0
    out: list[ScoredHit] = []
    for hit, mass in zip(hits, masses, strict=True):
        scaled = (mass - lo) / span
        combined = alpha * hit.score + (1.0 - alpha) * scaled
        signals = dict(hit.signals)
        signals.update({"pagerank": mass, "pagerank_scaled": scaled, "combined": combined})
        out.append(
            ScoredHit(
                article_id=hit.article_id,
                score=float(combined),
                rank=hit.rank,
                method=f"{hit.method}+ppr",
                signals=signals,
            )
        )
    out.sort(key=lambda h: (-h.score, h.article_id))
    for rank, hit in enumerate(out, start=1):
        hit.rank = rank
    return out
