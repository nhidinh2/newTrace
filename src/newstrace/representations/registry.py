"""Persistence and lookup of embedding vectors.

Vectors live in a single ``LargeBinary`` column per record (never one column per
dimension) and are keyed by ``model + preprocessing_version + normalized_text``
so re-ingesting the same article never recomputes an embedding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.config import Settings, get_settings
from newstrace.logging import get_logger
from newstrace.models import Article, EmbeddingRecord
from newstrace.representations.embedder import (
    Embedder,
    cache_key,
    from_blob,
    get_embedder,
    to_blob,
)

logger = get_logger(__name__)

Matrix = NDArray[np.float32]

FULL_FIT_VERSION = "identity"


@dataclass
class VectorSet:
    """A matrix of vectors aligned with ``article_ids``."""

    article_ids: list[int]
    matrix: Matrix
    method: str
    dimension: int
    fit_version: str

    def __len__(self) -> int:
        return len(self.article_ids)

    @property
    def index_by_article(self) -> dict[int, int]:
        return {aid: i for i, aid in enumerate(self.article_ids)}

    def memory_bytes(self) -> int:
        return int(self.matrix.nbytes)


def embed_text(text: str, embedder: Embedder | None = None) -> Matrix:
    embedder = embedder or get_embedder()
    return embedder.encode([text])[0]


def ensure_embeddings(
    session: Session,
    articles: Sequence[Article],
    *,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
) -> list[EmbeddingRecord]:
    """Compute and store full-dimensional embeddings for articles that lack one."""
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)

    existing = {
        record.article_id: record
        for record in session.execute(
            select(EmbeddingRecord).where(
                EmbeddingRecord.article_id.in_([a.id for a in articles]),
                EmbeddingRecord.method == "full",
                EmbeddingRecord.model_name == embedder.name,
            )
        ).scalars()
    }

    pending = [a for a in articles if a.id not in existing]
    records: list[EmbeddingRecord] = [existing[a.id] for a in articles if a.id in existing]
    if not pending:
        return records

    texts = [a.body_text for a in pending]
    vectors = embedder.encode(texts)
    for article, text, vector in zip(pending, texts, vectors, strict=True):
        record = EmbeddingRecord(
            article_id=article.id,
            model_name=embedder.name,
            method="full",
            dimension=int(vector.shape[0]),
            fit_version=FULL_FIT_VERSION,
            cache_key=cache_key(text, embedder.name, settings.preprocessing_version),
            vector_blob=to_blob(vector),
        )
        session.add(record)
        records.append(record)
    session.flush()
    logger.info("Embedded %s new articles with %s", len(pending), embedder.name)
    return records


def store_vectors(
    session: Session,
    article_ids: Sequence[int],
    matrix: Matrix,
    *,
    model_name: str,
    method: str,
    fit_version: str,
    settings: Settings | None = None,
) -> int:
    """Persist projected vectors, replacing any previous fit with the same id."""
    settings = settings or get_settings()
    dimension = int(matrix.shape[1])
    existing = {
        record.article_id: record
        for record in session.execute(
            select(EmbeddingRecord).where(
                EmbeddingRecord.article_id.in_(list(article_ids)),
                EmbeddingRecord.method == method,
                EmbeddingRecord.fit_version == fit_version,
                EmbeddingRecord.dimension == dimension,
            )
        ).scalars()
    }
    written = 0
    for article_id, vector in zip(article_ids, matrix, strict=True):
        blob = to_blob(vector)
        record = existing.get(article_id)
        if record is not None:
            record.vector_blob = blob
        else:
            session.add(
                EmbeddingRecord(
                    article_id=article_id,
                    model_name=model_name,
                    method=method,
                    dimension=dimension,
                    fit_version=fit_version,
                    cache_key=f"{fit_version}:{article_id}",
                    vector_blob=blob,
                )
            )
        written += 1
    session.flush()
    return written


def load_vectors(
    session: Session,
    *,
    method: str = "full",
    fit_version: str | None = None,
    dimension: int | None = None,
    article_ids: Sequence[int] | None = None,
    model_name: str | None = None,
) -> VectorSet:
    """Load a :class:`VectorSet` for the requested representation."""
    stmt = select(EmbeddingRecord).where(EmbeddingRecord.method == method)
    if fit_version is not None:
        stmt = stmt.where(EmbeddingRecord.fit_version == fit_version)
    if dimension is not None:
        stmt = stmt.where(EmbeddingRecord.dimension == dimension)
    if article_ids is not None:
        stmt = stmt.where(EmbeddingRecord.article_id.in_(list(article_ids)))
    if model_name is not None:
        stmt = stmt.where(EmbeddingRecord.model_name == model_name)
    stmt = stmt.order_by(EmbeddingRecord.article_id)

    records = list(session.execute(stmt).scalars())
    if not records:
        dim = dimension or 0
        return VectorSet([], np.zeros((0, dim), dtype=np.float32), method, dim, fit_version or "")

    dim = records[0].dimension
    ids = [r.article_id for r in records]
    matrix = np.vstack([from_blob(r.vector_blob, r.dimension) for r in records]).astype(np.float32)
    return VectorSet(ids, matrix, method, dim, records[0].fit_version)


def available_representations(session: Session) -> list[dict[str, object]]:
    """List every (method, dimension, fit_version) present in the database."""
    rows = session.execute(
        select(
            EmbeddingRecord.method,
            EmbeddingRecord.dimension,
            EmbeddingRecord.fit_version,
            EmbeddingRecord.model_name,
        ).distinct()
    ).all()
    return [{"method": m, "dimension": d, "fit_version": f, "model_name": n} for m, d, f, n in rows]
