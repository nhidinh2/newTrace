"""End-to-end ingestion: idempotency, audit records and failure isolation."""

from __future__ import annotations

import itertools

import httpx
import pytest
import respx
from sqlalchemy.orm import Session

from newstrace.ingestion.base import RawArticle
from newstrace.ingestion.gdelt import GDELT_DOC_ENDPOINT, GdeltAdapter
from newstrace.ingestion.http import HttpFetcher, ResponseCache
from newstrace.ingestion.pipeline import IngestOptions, persist_articles, run_ingestion
from newstrace.models import Article, IngestionRun
from tests.conftest import BASE_TIME


def test_fixture_ingestion_is_idempotent(session: Session) -> None:
    first_run, first = run_ingestion(session, IngestOptions(source="fixtures", index=False))
    assert first_run.status.startswith("completed")
    assert first.inserted > 0
    count_after_first = session.query(Article).count()

    second_run, second = run_ingestion(session, IngestOptions(source="fixtures", index=False))
    assert second.inserted == 0
    assert second.duplicates == 0
    assert session.query(Article).count() == count_after_first
    assert second_run.id != first_run.id


def test_run_records_configuration_and_counts(session: Session) -> None:
    run, result = run_ingestion(session, IngestOptions(source="fixtures", index=False))
    stored = session.get(IngestionRun, run.id)
    assert stored is not None
    assert stored.random_seed == 549
    assert stored.config["source"] == "fixtures"
    assert stored.config["cluster_threshold"] is not None
    assert stored.fetched_count == result.fetched
    assert stored.inserted_count == result.inserted
    assert stored.finished_at is not None


def test_articles_link_back_to_their_run(session: Session) -> None:
    run, _ = run_ingestion(session, IngestOptions(source="fixtures", index=False))
    assert session.query(Article).filter(Article.ingestion_run_id == run.id).count() > 0


def test_unusable_records_are_skipped_not_fatal(session: Session) -> None:
    result = persist_articles(
        session,
        [
            RawArticle(url="", title="no url"),
            RawArticle(
                url="https://ok.example/a", title="A usable article", published_at=BASE_TIME
            ),
        ],
    )
    session.commit()
    assert result.skipped == 1
    assert result.inserted == 1
    assert session.query(Article).count() == 1


@respx.mock
def test_live_fetch_failure_does_not_corrupt_the_database(session: Session) -> None:
    run_ingestion(session, IngestOptions(source="fixtures", index=False))
    before = session.query(Article).count()

    respx.get(GDELT_DOC_ENDPOINT).mock(return_value=httpx.Response(503))
    run, result = run_ingestion(session, IngestOptions(source="gdelt", index=False))

    assert run.status == "failed"
    assert run.errors
    assert result.failures == 1
    assert session.query(Article).count() == before, "no partial writes on failure"


@respx.mock
def test_gdelt_adapter_ingests_live_payload(session: Session, tmp_path) -> None:
    payload = {
        "articles": [
            {
                "url": "https://live.example/story",
                "title": "Live headline about Atlas 3",
                "domain": "live.example",
                "language": "English",
                "seendate": "20260818T060000Z",
            }
        ]
    }
    respx.get(GDELT_DOC_ENDPOINT).mock(return_value=httpx.Response(200, json=payload))
    fetcher = HttpFetcher(cache=ResponseCache(tmp_path / "c"), sleep=lambda _: None)
    articles = list(GdeltAdapter(fetcher).fetch(query="Atlas", topic="ai_models", max_records=10))
    assert len(articles) == 1

    result = persist_articles(session, articles)
    session.commit()
    assert result.inserted == 1
    stored = session.query(Article).one()
    assert stored.canonical_url == "https://live.example/story"
    assert stored.topic == "ai_models"
    assert stored.source_adapter == "gdelt"


def test_rss_enriches_an_article_already_seen_via_gdelt(session: Session) -> None:
    """GDELT's artlist mode has no description; the feed supplies one later."""
    gdelt_view = RawArticle(
        url="https://alpha.example/a",
        title="Atlas 3 released",
        source_adapter="gdelt",
        published_at=BASE_TIME,
    )
    persist_articles(session, [gdelt_view])
    session.commit()
    assert session.query(Article).one().excerpt == ""

    feed_view = RawArticle(
        url="https://alpha.example/a",
        title="Atlas 3 released",
        description="Northwind Labs said the model is available on Tuesday.",
        source_adapter="rss",
        published_at=BASE_TIME,
    )
    result = persist_articles(session, [feed_view])
    session.commit()
    assert result.skipped == 1
    assert session.query(Article).count() == 1
    assert "available on Tuesday" in session.query(Article).one().excerpt


def test_unknown_source_is_rejected(session: Session) -> None:
    run, result = run_ingestion(session, IngestOptions(source="carrier-pigeon", index=False))
    assert run.status == "failed"
    assert result.failures == 1


def test_full_index_pass_produces_stories_claims_and_embeddings(session: Session) -> None:
    from newstrace.models import Claim, ClusterMembership, EmbeddingRecord, StoryCluster

    run, result = run_ingestion(session, IngestOptions(source="fixtures", index=True))
    assert run.status.startswith("completed")
    assert session.query(EmbeddingRecord).count() == session.query(Article).count()
    assert session.query(ClusterMembership).count() == session.query(Article).count()
    assert session.query(StoryCluster).count() > 1
    assert session.query(Claim).count() > 0
    assert result.article_ids


@pytest.mark.network
def test_live_gdelt_is_reachable() -> None:  # pragma: no cover - excluded by default
    """Deselected by default; run with `pytest -m network`."""
    from newstrace.ingestion.gdelt import build_params

    response = httpx.get(
        GDELT_DOC_ENDPOINT,
        params=build_params("(OpenAI OR Anthropic)", timespan="1h", max_records=5),
        timeout=30,
        headers={"User-Agent": "NewsTraceResearch/0.1"},
    )
    assert response.status_code == 200


@respx.mock
def test_windowed_sweep_issues_one_request_per_window(session: Session, tmp_path) -> None:
    """`--windows N` must produce N non-overlapping, timespan-free requests.

    A single artlist response carries at most 250 records, so a wide query is
    sampled window by window. GDELT rejects `timespan` alongside an explicit
    window, and overlapping windows would re-fetch the same articles.
    """
    from datetime import UTC, datetime

    from newstrace.config import TopicConfig
    from newstrace.ingestion.gdelt import GDELT_DATETIME_FORMAT

    def payload(request: httpx.Request) -> httpx.Response:
        start = request.url.params["startdatetime"]
        return httpx.Response(
            200,
            json={
                "articles": [
                    {
                        "url": f"https://live.example/{start}",
                        "title": f"Headline from window {start}",
                        "domain": "live.example",
                        "language": "English",
                        "seendate": f"{start[:8]}T{start[8:]}Z",
                    }
                ]
            },
        )

    route = respx.get(GDELT_DOC_ENDPOINT).mock(side_effect=payload)
    fetcher = HttpFetcher(cache=ResponseCache(tmp_path / "c"), sleep=lambda _: None)
    topics = {"ai_models": TopicConfig(key="ai_models", gdelt_query="Atlas")}

    articles = list(
        GdeltAdapter(fetcher).fetch_topics(
            topics,
            timespan="24h",
            max_records=250,
            windows=4,
            end=datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
        )
    )

    assert route.call_count == 4, "one request per window"
    assert len(articles) == 4, "every window contributed"

    bounds = []
    for call in route.calls:
        params = call.request.url.params
        assert "timespan" not in params, "GDELT rejects timespan alongside a window"
        bounds.append(
            (
                datetime.strptime(params["startdatetime"], GDELT_DATETIME_FORMAT),
                datetime.strptime(params["enddatetime"], GDELT_DATETIME_FORMAT),
            )
        )

    bounds.sort()
    assert bounds[0][0] == datetime(2026, 8, 23, 12, 0)
    assert bounds[-1][1] == datetime(2026, 8, 24, 12, 0)
    for earlier, later in itertools.pairwise(bounds):
        assert earlier[1] == later[0], "windows must tile without gaps or overlap"

    result = persist_articles(session, articles)
    session.commit()
    assert result.inserted == 4
