"""The SQLite FTS5 index: BM25 retrieval and the entity filter."""

from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from newstrace.ingestion.base import RawArticle
from newstrace.ingestion.pipeline import persist_articles
from newstrace.models import Article
from newstrace.retrieval import fulltext
from newstrace.search import SearchFilters, SearchRequest, search
from tests.conftest import BASE_TIME


def test_index_exists_after_create_all(session: Session) -> None:
    assert fulltext.available(session)


def test_persisted_articles_are_searchable(populated_session: Session) -> None:
    ids = fulltext.match_ids(populated_session, "Atlas")
    assert ids
    titles = [
        a.title
        for a in populated_session.execute(select(Article).where(Article.id.in_(ids))).scalars()
    ]
    assert any("Atlas" in title for title in titles)


def test_bm25_ranks_the_matching_article_first(populated_session: Session) -> None:
    hits = fulltext.bm25_hits(populated_session, "gigawatts data centre demand", k=3)
    assert hits
    top = populated_session.get(Article, hits[0].article_id)
    assert top is not None
    assert "gigawatts" in f"{top.title} {top.excerpt}".lower()
    assert hits[0].method == "bm25"


def test_bm25_respects_the_allowed_set(populated_session: Session) -> None:
    everything = fulltext.bm25_hits(populated_session, "Atlas agent benchmark", k=10)
    assert everything
    allowed = {everything[-1].article_id}
    filtered = fulltext.bm25_hits(
        populated_session, "Atlas agent benchmark", k=10, allowed_ids=allowed
    )
    assert filtered is not None
    assert {h.article_id for h in filtered} <= allowed


def test_query_syntax_cannot_break_the_index(populated_session: Session) -> None:
    """FTS5 treats its input as an expression; the reader must quote it."""
    for query in ['Atlas AND "', "NEAR(", "column:value", "*", "", "-- drop"]:
        assert fulltext.bm25_hits(populated_session, query, k=3) is not None


def test_entity_filter_uses_a_phrase(populated_session: Session) -> None:
    response = search(
        populated_session,
        SearchRequest(
            query="agent benchmark",
            filters=SearchFilters(entity="Northwind Labs"),
            top_k=5,
        ),
    )
    assert response.items
    for item in response.items:
        assert "northwind labs" in item.article.body_text.lower()


def test_entity_filter_excludes_non_matching_articles(populated_session: Session) -> None:
    response = search(
        populated_session,
        SearchRequest(
            query="anything",
            filters=SearchFilters(entity="Nonexistent Corporation"),
            top_k=5,
        ),
    )
    assert response.items == []


def test_rebuild_repopulates_the_index(populated_session: Session) -> None:
    populated_session.execute(text(f"DELETE FROM {fulltext.FTS_TABLE}"))
    populated_session.commit()
    assert fulltext.match_ids(populated_session, "Atlas") == set()

    count = fulltext.rebuild(populated_session)
    assert count > 0
    assert fulltext.match_ids(populated_session, "Atlas")


def test_search_falls_back_when_the_index_is_missing(
    populated_session: Session, monkeypatch
) -> None:
    monkeypatch.setattr(fulltext, "available", lambda session: False)
    response = search(populated_session, SearchRequest(query="agent benchmark", method="bm25"))
    assert any("TF-IDF" in note for note in response.notes)
    assert response.items


def test_enrichment_updates_the_indexed_text(session: Session) -> None:
    sparse = RawArticle(
        url="https://alpha.example/atlas-3",
        title="Northwind Labs releases Atlas 3",
        description="",
        source_domain="alpha.example",
        language="English",
        published_at=BASE_TIME,
        source_adapter="test",
    )
    persist_articles(session, [sparse])
    session.commit()
    assert fulltext.match_ids(session, "quadrupled") == set()

    richer = RawArticle(
        url="https://alpha.example/atlas-3",
        title="Northwind Labs releases Atlas 3",
        description="Throughput quadrupled on the public agent benchmark.",
        source_domain="alpha.example",
        language="English",
        published_at=BASE_TIME,
        source_adapter="test",
    )
    persist_articles(session, [richer])
    session.commit()
    assert fulltext.match_ids(session, "quadrupled")
