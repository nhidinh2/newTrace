"""Human-readable explanations of why a result was retrieved.

Every search result carries one of these.  Explanations describe *ranking
signals only*; they never assert that a source is reliable or that a claim is
true.
"""

from __future__ import annotations

from newstrace.models import Article
from newstrace.retrieval.exact import ScoredHit
from newstrace.utils import token_set, truncate

SIGNAL_LABELS = {
    "cosine": "semantic similarity",
    "tfidf": "lexical term overlap",
    "pagerank": "graph centrality (personalized PageRank)",
    "recency": "recency",
    "duplicate_penalty": "duplicate penalty",
    "combined": "combined score",
}


def query_term_overlap(query: str, article: Article) -> list[str]:
    shared = token_set(query) & token_set(article.body_text)
    return sorted(t for t in shared if len(t) > 2)[:8]


def explain_hit(hit: ScoredHit, article: Article, query: str) -> str:
    """Build a one-line, source-neutral explanation of the ranking."""
    parts: list[str] = []
    ordered = sorted(hit.signals.items(), key=lambda kv: -abs(kv[1]))
    for name, value in ordered:
        label = SIGNAL_LABELS.get(name, name)
        parts.append(f"{label}={value:.3f}")
    overlap = query_term_overlap(query, article)
    if overlap:
        parts.append("shared terms: " + ", ".join(overlap))
    if article.is_near_duplicate:
        parts.append("marked as a near-duplicate of another article (not counted as independent)")
    return f"rank {hit.rank} via {hit.method}: " + "; ".join(parts)


def evidence_excerpt(article: Article, query: str, *, limit: int = 280) -> str:
    """Pick the excerpt sentence with the highest query-term overlap."""
    from newstrace.clustering.labeling import split_sentences

    text = article.excerpt or article.description or article.title
    sentences = split_sentences(text) or [text]
    q_tokens = token_set(query)
    best = max(
        sentences,
        key=lambda s: (len(q_tokens & token_set(s)), -len(s)),
        default=text,
    )
    return truncate(best, limit)
