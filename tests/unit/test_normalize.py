"""URL, text and timestamp normalisation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from newstrace.ingestion.base import RawArticle
from newstrace.ingestion.normalize import (
    canonicalize_url,
    extract_domain,
    normalize_article,
    normalize_language,
)
from newstrace.utils import normalize_unicode, to_utc


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://www.Example.com/story/", "https://example.com/story"),
        ("http://example.com/story", "https://example.com/story"),
        ("https://example.com/story?utm_source=x&id=7", "https://example.com/story?id=7"),
        ("https://example.com/story?fbclid=abc", "https://example.com/story"),
        ("https://example.com/story#section", "https://example.com/story"),
        ("https://example.com:443/story", "https://example.com/story"),
        ("https://example.com/story?b=2&a=1", "https://example.com/story?a=1&b=2"),
        ("example.com/story", "https://example.com/story"),
        ("https://example.com", "https://example.com/"),
    ],
)
def test_canonicalize_url(raw: str, expected: str) -> None:
    assert canonicalize_url(raw) == expected


def test_canonicalize_url_is_idempotent() -> None:
    once = canonicalize_url("https://www.example.com/a/?utm_medium=rss&x=1#frag")
    assert canonicalize_url(once) == once


def test_canonicalize_empty_url() -> None:
    assert canonicalize_url("") == ""


@pytest.mark.parametrize(
    ("url", "domain"),
    [
        ("https://news.bbc.co.uk/a", "bbc.co.uk"),
        ("https://www.example.com/a", "example.com"),
        ("https://sub.deep.example.org/a", "example.org"),
    ],
)
def test_extract_domain(url: str, domain: str) -> None:
    assert extract_domain(url) == domain


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("20260818T060000Z", datetime(2026, 8, 18, 6, 0, tzinfo=UTC)),
        ("20260818060000", datetime(2026, 8, 18, 6, 0, tzinfo=UTC)),
        ("2026-08-18T06:00:00Z", datetime(2026, 8, 18, 6, 0, tzinfo=UTC)),
        ("Tue, 18 Aug 2026 06:00:00 +0000", datetime(2026, 8, 18, 6, 0, tzinfo=UTC)),
    ],
)
def test_timestamp_normalisation(raw: str, expected: datetime) -> None:
    assert to_utc(raw) == expected


def test_naive_timestamps_become_utc() -> None:
    parsed = to_utc(datetime(2026, 8, 18, 6, 0))
    assert parsed is not None and parsed.tzinfo is UTC


def test_offset_timestamps_convert_to_utc() -> None:
    assert to_utc("2026-08-18T08:00:00+02:00") == datetime(2026, 8, 18, 6, 0, tzinfo=UTC)


def test_unparseable_timestamp_is_none() -> None:
    assert to_utc("not a date at all") is None
    assert to_utc("") is None


def test_unicode_normalisation_collapses_whitespace_and_controls() -> None:
    # Deliberately mixes a fullwidth letter, a zero-width space and a control code.
    assert normalize_unicode("  Ｈello​   world \x07 ") == "Hello world"  # noqa: RUF001


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("English", "english"), ("en-GB", "en"), (None, "unknown"), ("", "unknown")],
)
def test_normalize_language(raw: str | None, expected: str) -> None:
    assert normalize_language(raw) == expected


def test_normalize_article_populates_hashes_and_domain() -> None:
    article = normalize_article(
        RawArticle(
            url="https://www.Example.com/story/?utm_source=x",
            title="  A   Title ",
            description="Some description.",
            published_at="20260818T060000Z",
            language="English",
        )
    )
    assert article.canonical_url == "https://example.com/story"
    assert article.title == "A Title"
    assert article.source_domain == "example.com"
    assert article.language == "english"
    assert article.published_at == datetime(2026, 8, 18, 6, 0, tzinfo=UTC)
    assert len(str(article.normalized_text_hash)) == 64
    assert len(str(article.normalized_title_hash)) == 64


def test_normalize_article_rejects_urlless_records() -> None:
    with pytest.raises(ValueError, match="no usable URL"):
        normalize_article(RawArticle(url=""))


def test_excerpt_is_truncated() -> None:
    article = normalize_article(
        RawArticle(url="https://example.com/x", description="word " * 500),
        max_excerpt_chars=100,
    )
    assert len(str(article.excerpt)) <= 100
