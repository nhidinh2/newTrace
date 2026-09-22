"""Process-level cache of materialised retrieval indexes.

Building an index means reading one ``LargeBinary`` per article, stacking the
blobs into a matrix and normalising it.  On a six-thousand article corpus that
is roughly 230 ms -- against a 6 ms search -- so doing it per request made the
serving path forty times slower than the retrieval it was serving.

The cache holds one :class:`~newstrace.retrieval.exact.DenseRetriever` per
``(method, dimension, fit_version)`` and keeps it current in two ways:

* a *signature* -- the row count and the highest ``embedding_records`` primary
  key for that representation -- is re-read before every use.  It is one
  aggregate query (sub-millisecond, index-only) and it catches the ordinary
  case of another process ingesting articles while the UI is open.  When only
  new rows appeared, the cached matrix is *extended* with those rows rather
  than rebuilt.
* :func:`invalidate` is called by the writers, because a vector rewritten in
  place (re-running a sweep under an existing ``fit_version``) changes neither
  the count nor the maximum id.

Both are needed: the signature covers other processes, the explicit call
covers this one.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newstrace.config import Settings, get_settings
from newstrace.logging import get_logger
from newstrace.models import EmbeddingRecord, ProjectionArtifact
from newstrace.representations.embedder import from_blob
from newstrace.representations.projection import load_projector
from newstrace.retrieval.exact import DenseRetriever

logger = get_logger(__name__)

# The database is part of the key: one process can rebind the engine (tests
# do, and so does anything pointed at a second corpus), and a matrix from the
# previous database would otherwise be served for the new one.
IndexKey = tuple[str, str, int | None, str | None]


def _bind_key(session: Session) -> str:
    bind = session.get_bind()
    return str(getattr(bind, "url", bind))


@dataclass
class _Entry:
    retriever: DenseRetriever
    count: int
    max_record_id: int
    dimension: int
    fit_version: str


_CACHE: OrderedDict[IndexKey, _Entry] = OrderedDict()
_LOCK = threading.Lock()
_STATS = {"hits": 0, "misses": 0, "extends": 0}


def _filtered(stmt: Any, method: str, dimension: int | None, fit_version: str | None) -> Any:
    stmt = stmt.where(EmbeddingRecord.method == method)
    if dimension is not None:
        stmt = stmt.where(EmbeddingRecord.dimension == dimension)
    if fit_version is not None:
        stmt = stmt.where(EmbeddingRecord.fit_version == fit_version)
    return stmt


def _signature(
    session: Session, method: str, dimension: int | None, fit_version: str | None
) -> tuple[int, int]:
    stmt = _filtered(
        select(func.count(EmbeddingRecord.id), func.max(EmbeddingRecord.id)),
        method,
        dimension,
        fit_version,
    )
    count, max_id = session.execute(stmt).one()
    return int(count or 0), int(max_id or 0)


def _load_rows(
    session: Session,
    method: str,
    dimension: int | None,
    fit_version: str | None,
    *,
    after_record_id: int = 0,
) -> tuple[list[int], list[Any], int, str]:
    stmt = _filtered(select(EmbeddingRecord), method, dimension, fit_version)
    if after_record_id:
        stmt = stmt.where(EmbeddingRecord.id > after_record_id)
    records = list(session.execute(stmt.order_by(EmbeddingRecord.id)).scalars())
    if not records:
        return [], [], dimension or 0, fit_version or ""
    ids = [int(r.article_id) for r in records]
    vectors = [from_blob(r.vector_blob, r.dimension) for r in records]
    return ids, vectors, int(records[0].dimension), str(records[0].fit_version)


def _projector_for(session: Session, method: str, fit_version: str) -> Any:
    """Load the saved projector for a stored representation, if there is one."""
    if method in ("full", "tfidf"):
        return None
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


def get_index(
    session: Session,
    *,
    method: str = "full",
    dimension: int | None = None,
    fit_version: str | None = None,
    settings: Settings | None = None,
) -> DenseRetriever | None:
    """Return a ready-to-search index, building or extending it only if needed.

    ``None`` means the representation has no stored vectors at all.
    """
    settings = settings or get_settings()
    key: IndexKey = (_bind_key(session), method, dimension, fit_version)

    with _LOCK:
        count, max_id = _signature(session, method, dimension, fit_version)
        if count == 0:
            _CACHE.pop(key, None)
            return None

        entry = _CACHE.get(key)
        if entry is not None and (count, max_id) == (entry.count, entry.max_record_id):
            _STATS["hits"] += 1
            _CACHE.move_to_end(key)
            return entry.retriever

        # Rows were only appended: grow the matrix instead of rebuilding it.
        if entry is not None and max_id > entry.max_record_id and count > entry.count:
            ids, vectors, _, _ = _load_rows(
                session, method, dimension, fit_version, after_record_id=entry.max_record_id
            )
            if len(ids) == count - entry.count:
                entry.retriever.extend(ids, np.vstack(vectors).astype(np.float32))
                entry.count, entry.max_record_id = count, max_id
                _STATS["extends"] += 1
                _CACHE.move_to_end(key)
                return entry.retriever

        ids, vectors, dim, resolved_fit = _load_rows(session, method, dimension, fit_version)
        if not ids:
            _CACHE.pop(key, None)
            return None

        matrix = np.vstack(vectors).astype(np.float32)
        retriever = DenseRetriever(
            ids,
            matrix,
            method=method,
            projector=_projector_for(session, method, resolved_fit),
        )
        _CACHE[key] = _Entry(retriever, count, max_id, dim, resolved_fit)
        _CACHE.move_to_end(key)
        _STATS["misses"] += 1
        while len(_CACHE) > max(1, settings.retrieval_cache_entries):
            evicted, _ = _CACHE.popitem(last=False)
            logger.debug("Evicted cached index %s", evicted)
        logger.info(
            "Built %s index: %s vectors, %s-d, %.1f MiB",
            method,
            len(ids),
            dim,
            retriever.memory_bytes() / (1024 * 1024),
        )
        return retriever


def invalidate(method: str | None = None) -> None:
    """Drop cached indexes after vectors are written in place."""
    with _LOCK:
        if method is None:
            _CACHE.clear()
            return
        for key in [k for k in _CACHE if k[1] == method]:
            del _CACHE[key]


def cache_stats() -> dict[str, int]:
    """Hit/miss/extend counters, exposed on the health endpoint."""
    with _LOCK:
        return {**_STATS, "entries": len(_CACHE)}


def describe() -> list[dict[str, Any]]:
    """One row per resident index (method, dimension, vectors, memory)."""
    with _LOCK:
        return [
            {
                "method": key[1],
                "dimension": entry.dimension,
                "fit_version": entry.fit_version,
                "vectors": len(entry.retriever.article_ids),
                "memory_mib": round(entry.retriever.memory_bytes() / (1024 * 1024), 3),
            }
            for key, entry in _CACHE.items()
        ]
