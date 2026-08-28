"""GDELT and RSS payload parsing, including malformed input."""

from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from newstrace.ingestion.gdelt import (
    GDELT_DOC_ENDPOINT,
    build_params,
    parse_articles,
    parse_timespan,
    time_windows,
    window_params,
)
from newstrace.ingestion.rss import parse_feed

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_build_params_encodes_query_without_concatenation() -> None:
    params = build_params('("artificial intelligence" OR OpenAI)', timespan="12h", max_records=10)
    assert params["query"] == '("artificial intelligence" OR OpenAI)'
    assert params["format"] == "json"
    assert params["timespan"] == "12h"
    assert params["maxrecords"] == 10
    assert GDELT_DOC_ENDPOINT.startswith("https://api.gdeltproject.org")


def test_build_params_clamps_max_records() -> None:
    assert build_params("x", max_records=10_000)["maxrecords"] == 250


def test_parse_gdelt_fixture() -> None:
    payload = json.loads((FIXTURES / "gdelt_sample.json").read_text(encoding="utf-8"))
    articles = parse_articles(payload, topic="ai_models")
    assert len(articles) == 4
    first = articles[0]
    assert first.url.startswith("https://")
    assert first.title
    assert first.topic == "ai_models"
    assert first.source_adapter == "gdelt"
    assert first.published_at == datetime(2026, 8, 18, 6, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "payload",
    [{}, {"articles": None}, {"articles": []}, [], "nonsense", None, {"other": [1, 2]}],
)
def test_parse_gdelt_handles_malformed_payloads(payload: object) -> None:
    assert parse_articles(payload) == []


def test_parse_gdelt_skips_unusable_records() -> None:
    payload = json.loads((FIXTURES / "gdelt_malformed.json").read_text(encoding="utf-8"))
    assert parse_articles(payload) == []


def test_parse_rss_fixture() -> None:
    articles = parse_feed(
        (FIXTURES / "rss_sample.xml").read_text(encoding="utf-8"), topic="ai_models"
    )
    assert articles
    assert all(a.url for a in articles)
    assert all(a.source_adapter == "rss" for a in articles)
    assert any(a.description for a in articles)


def test_parse_rss_strips_html() -> None:
    xml = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
    <item><title>T</title><link>https://example.com/a</link>
    <description>&lt;p&gt;Hello &lt;b&gt;world&lt;/b&gt;&lt;/p&gt;</description>
    <pubDate>Tue, 18 Aug 2026 06:00:00 +0000</pubDate></item></channel></rss>"""
    articles = parse_feed(xml)
    assert articles[0].description == "Hello world"


def test_parse_rss_handles_garbage() -> None:
    assert parse_feed("not xml at all") == []
    assert parse_feed("") == []


def test_parse_timespan_accepts_gdelt_units() -> None:
    assert parse_timespan("24h") == timedelta(hours=24)
    assert parse_timespan("30min") == timedelta(minutes=30)
    assert parse_timespan(" 7D ") == timedelta(days=7)


@pytest.mark.parametrize("bad", ["", "24", "h", "24 hours ago", "-1h"])
def test_parse_timespan_rejects_nonsense(bad: str) -> None:
    with pytest.raises(ValueError, match="Unrecognised GDELT timespan"):
        parse_timespan(bad)


def test_time_windows_tile_the_timespan_without_gaps() -> None:
    end = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    windows = time_windows("24h", 4, end=end)
    assert len(windows) == 4
    assert windows[0][0] == end - timedelta(hours=24)
    assert windows[-1][1] == end
    for earlier, later in itertools.pairwise(windows):
        assert earlier[1] == later[0], "windows must tile without gaps or overlap"


def test_time_windows_degenerate_counts_return_one_window() -> None:
    end = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    for count in (0, 1, -3):
        assert time_windows("6h", count, end=end) == [(end - timedelta(hours=6), end)]


def test_window_params_are_gdelt_formatted_utc() -> None:
    start = datetime(2026, 8, 23, 6, 30, tzinfo=UTC)
    end = datetime(2026, 8, 24, 6, 30, tzinfo=UTC)
    assert window_params(start, end) == {
        "startdatetime": "20260823063000",
        "enddatetime": "20260824063000",
    }


def test_build_params_carries_the_window_through() -> None:
    params = build_params(
        "q",
        extra=window_params(datetime(2026, 8, 23, tzinfo=UTC), datetime(2026, 8, 24, tzinfo=UTC)),
    )
    assert params["startdatetime"] == "20260823000000"
    assert params["enddatetime"] == "20260824000000"
