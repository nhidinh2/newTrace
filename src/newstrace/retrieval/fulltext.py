"""SQLite FTS5 index over article headlines and excerpts.

Two things used to scan the whole corpus in Python: the ``entity`` search
filter (``needle in article.body_text.lower()`` over every matching row) and
the TF-IDF baseline (a ``TfidfVectorizer`` re-fitted per query). Both are text
retrieval, and SQLite already ships a text retrieval engine.

The index carries its own copy of the text -- a few megabytes -- rather than
using an external-content table, so a rebuild can never be silently out of
step with a table it no longer matches. It is written on insert, on
enrichment, and by ``newstrace reindex-text``.

Everything here degrades to ``None``/empty rather than raising when the
virtual table is absent: a database created before this migration, or a
SQLite build without FTS5, must still serve search.
"""

from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlalchemy.orm import Session

from newstrace.logging import get_logger
from newstrace.models import Article
from newstrace.retrieval.exact import ScoredHit

logger = get_logger(__name__)

FTS_TABLE = "articles_fts"

# Kept beside the reader so the migration, ``create_all`` and the rebuild
# command cannot drift into three different table definitions.
CREATE_FTS_SQL = (
    f"CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} "
    "USING fts5(title, body, tokenize = 'unicode61 remove_diacritics 2')"
)


def create_index(engine: object) -> bool:
    """Create the FTS5 table on a SQLite engine; False when unsupported."""
    from sqlalchemy import Engine

    if not isinstance(engine, Engine) or engine.dialect.name != "sqlite":
        return False
    try:
        with engine.begin() as connection:
            connection.execute(text(CREATE_FTS_SQL))
    except DatabaseError as exc:  # pragma: no cover - SQLite built without FTS5
        logger.warning("Full-text index unavailable: %s", exc)
        return False
    reset_availability()
    return True


# FTS5 treats a bare query as an expression: bare "AND", a stray quote or a
# trailing "*" is a syntax error, and a colon makes a column filter. Queries
# arrive from a text box, so every token is quoted and combined explicitly.
_TOKEN = re.compile(r"[0-9A-Za-z_]+", re.UNICODE)


# Presence is a property of the database, not of the request: looking it up in
# sqlite_master on every insert would cost more than the insert.
_AVAILABLE: dict[str, bool] = {}


def _bind_key(session: Session) -> str:
    bind = session.get_bind()
    return str(getattr(bind, "url", bind))


def available(session: Session) -> bool:
    """True when the FTS5 table exists in this database."""
    key = _bind_key(session)
    cached = _AVAILABLE.get(key)
    if cached is not None:
        return cached
    try:
        row = session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
            {"n": FTS_TABLE},
        ).first()
    except DatabaseError:  # pragma: no cover - non-SQLite backends
        _AVAILABLE[key] = False
        return False
    _AVAILABLE[key] = row is not None
    return row is not None


def reset_availability() -> None:
    """Forget cached probes (a migration may have just created the table)."""
    _AVAILABLE.clear()


def _match_expression(query: str, *, operator: str = "OR") -> str:
    tokens = [t.lower() for t in _TOKEN.findall(query or "")]
    return f" {operator} ".join(f'"{t}"' for t in tokens)


def phrase_expression(query: str) -> str:
    """A quoted phrase, so ``entity=\"Northwind Labs\"`` matches the phrase."""
    tokens = [t.lower() for t in _TOKEN.findall(query or "")]
    return f'"{" ".join(tokens)}"' if tokens else ""


def index_article_text(session: Session, article: Article) -> bool:
    """Insert or replace one article's row in the full-text index."""
    if article.id is None or not available(session):
        return False
    try:
        session.execute(
            text(f"INSERT OR REPLACE INTO {FTS_TABLE}(rowid, title, body) VALUES (:i, :t, :b)"),
            {"i": int(article.id), "t": article.title or "", "b": article.excerpt or ""},
        )
    except OperationalError as exc:  # pragma: no cover - corrupt or missing index
        logger.warning("Full-text index write failed for article %s: %s", article.id, exc)
        return False
    return True


def delete_article_text(session: Session, article_id: int) -> None:
    if not available(session):
        return
    session.execute(text(f"DELETE FROM {FTS_TABLE} WHERE rowid = :i"), {"i": int(article_id)})


def rebuild(session: Session, *, batch_size: int = 1000) -> int:
    """Repopulate the whole index from ``articles``. Returns the row count."""
    if not available(session):
        return 0
    session.execute(text(f"DELETE FROM {FTS_TABLE}"))
    written = 0
    offset = 0
    while True:
        rows = session.execute(
            text("SELECT id, title, excerpt FROM articles ORDER BY id LIMIT :limit OFFSET :offset"),
            {"limit": batch_size, "offset": offset},
        ).all()
        if not rows:
            break
        session.execute(
            text(f"INSERT INTO {FTS_TABLE}(rowid, title, body) VALUES (:i, :t, :b)"),
            [{"i": int(r[0]), "t": r[1] or "", "b": r[2] or ""} for r in rows],
        )
        written += len(rows)
        offset += batch_size
    session.commit()
    logger.info("Rebuilt the full-text index over %s articles", written)
    return written


def match_ids(session: Session, query: str, *, phrase: bool = False) -> set[int] | None:
    """Article ids matching ``query``; ``None`` when the index is unavailable."""
    if not available(session):
        return None
    expression = phrase_expression(query) if phrase else _match_expression(query, operator="AND")
    if not expression:
        return set()
    rows = session.execute(
        text(f"SELECT rowid FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH :q"), {"q": expression}
    ).scalars()
    return {int(r) for r in rows}


def bm25_hits(
    session: Session,
    query: str,
    *,
    k: int = 10,
    allowed_ids: set[int] | None = None,
    overfetch: int = 10,
) -> list[ScoredHit] | None:
    """Top ``k`` articles by Okapi BM25, or ``None`` when FTS5 is unavailable.

    BM25 is the standard lexical baseline, and unlike the TF-IDF retriever it
    does not re-fit a vectoriser over the corpus on every query. Scores are
    negated because SQLite returns bm25() as "smaller is better".

    Filters are applied to an over-fetched candidate list rather than pushed
    into the SQL: the allowed set is usually most of the corpus, and binding
    thousands of ids costs more than reading a few hundred extra rows.
    """
    if not available(session):
        return None
    expression = _match_expression(query)
    if not expression:
        return []
    limit = max(k, k * overfetch) if allowed_ids is not None else k
    # A title match is worth more than a body match; 2.0/1.0 are column
    # weights, not a tuned ranking model.
    rows = session.execute(
        text(
            f"SELECT rowid, -bm25({FTS_TABLE}, 2.0, 1.0) AS score "
            f"FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH :q "
            "ORDER BY score DESC LIMIT :k"
        ),
        {"q": expression, "k": int(limit)},
    ).all()

    hits: list[ScoredHit] = []
    for row in rows:
        article_id = int(row[0])
        if allowed_ids is not None and article_id not in allowed_ids:
            continue
        score = float(row[1])
        hits.append(
            ScoredHit(
                article_id=article_id,
                score=score,
                rank=len(hits) + 1,
                method="bm25",
                signals={"bm25": score},
            )
        )
        if len(hits) >= k:
            break
    return hits
