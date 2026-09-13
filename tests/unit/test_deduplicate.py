"""Exact and near-duplicate detection."""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy.orm import Session

from newstrace.ingestion.base import RawArticle
from newstrace.ingestion.deduplicate import (
    DuplicateDetector,
    hamming_distance,
    simhash,
    simhash_similarity,
    strip_site_suffix,
    title_similarity,
)
from newstrace.ingestion.normalize import normalize_article
from newstrace.ingestion.pipeline import persist_articles
from newstrace.models import Article
from tests.conftest import BASE_TIME


def test_simhash_is_deterministic_and_similarity_bounded() -> None:
    text = "Northwind Labs releases Atlas 3 agent model"
    assert simhash(text) == simhash(text)
    assert hamming_distance(simhash(text), simhash(text)) == 0
    assert simhash_similarity(text, text) == 1.0
    assert 0.0 <= simhash_similarity(text, "completely unrelated words here") <= 1.0


def test_simhash_of_empty_text() -> None:
    assert simhash("") == 0


def test_title_similarity_ranks_paraphrases_above_unrelated() -> None:
    a = "Northwind Labs releases Atlas 3 agent model"
    near = "Northwind Labs releases the Atlas 3 agent model"
    far = "Grid operator warns of data centre demand"
    assert title_similarity(a, a) == 1.0
    assert title_similarity(a, near) > title_similarity(a, far)


def test_exact_canonical_url_duplicate(session: Session) -> None:
    raw = RawArticle(url="https://alpha.example/a", title="Title one", published_at=BASE_TIME)
    persist_articles(session, [raw])
    session.commit()
    detector = DuplicateDetector(session)
    verdict = detector.check(normalize_article(RawArticle(url="https://www.alpha.example/a/")))
    assert verdict.is_duplicate
    assert verdict.reason == "canonical_url"


def test_identical_title_from_another_domain_is_a_duplicate(session: Session) -> None:
    persist_articles(
        session,
        [
            RawArticle(
                url="https://alpha.example/a", title="Same headline here", published_at=BASE_TIME
            )
        ],
    )
    session.commit()
    verdict = DuplicateDetector(session).check(
        normalize_article(
            RawArticle(
                url="https://wire.example/b", title="Same headline here", published_at=BASE_TIME
            )
        )
    )
    assert verdict.is_duplicate
    assert verdict.reason in {"title_hash", "normalized_text_hash"}


def test_unrelated_article_is_not_a_duplicate(session: Session) -> None:
    persist_articles(
        session,
        [
            RawArticle(
                url="https://alpha.example/a", title="Atlas 3 model release", published_at=BASE_TIME
            )
        ],
    )
    session.commit()
    verdict = DuplicateDetector(session).check(
        normalize_article(
            RawArticle(
                url="https://gamma.example/x",
                title="Grid operator warns of transmission upgrades",
                published_at=BASE_TIME,
            )
        )
    )
    assert not verdict.is_duplicate
    assert verdict.duplicate_of_id is None


def test_duplicates_are_preserved_and_linked(session: Session, raw_articles: list) -> None:
    result = persist_articles(session, raw_articles)
    session.commit()
    assert result.duplicates == 1
    duplicate = session.query(Article).filter(Article.is_near_duplicate.is_(True)).one()
    original = session.get(Article, duplicate.duplicate_of_article_id)
    assert original is not None
    assert original.source_domain != duplicate.source_domain
    # Preserved, not deleted.
    assert session.query(Article).count() == len(raw_articles)


def test_publication_time_window_limits_candidates(session: Session) -> None:
    persist_articles(
        session,
        [
            RawArticle(
                url="https://alpha.example/a",
                title="A distinctive rare headline about widgets",
                description="Body text about widgets and the widget market.",
                published_at=BASE_TIME,
            )
        ],
    )
    session.commit()
    detector = DuplicateDetector(session, window_hours=1)
    far_away = normalize_article(
        RawArticle(
            url="https://beta.example/b",
            title="A distinctive rare headline about widgets today",
            description="Body text about widgets and the widget market.",
            published_at=BASE_TIME + timedelta(days=30),
        )
    )
    assert not detector.check(far_away).is_duplicate


# One wire story, nine co-owned mastheads. The headline is byte-identical; only
# the appended masthead differs, and these feeds carry no excerpt, so the
# simhash fallback has nothing to compare. Without suffix stripping the pair
# scores about 0.65 against a 0.9 threshold and each copy is counted as an
# independent source.
SYNDICATED = [
    "AI chatbot travel tips fail Aussie tourists at border crossing | Moree Champion",
    "AI chatbot travel tips fail Aussie tourists at border crossing | The Queanbeyan Age",
    "AI chatbot travel tips fail Aussie tourists at border crossing | The Advertiser - Cessnock",
    "AI chatbot travel tips fail Aussie tourists at border crossing | Daily Liberal",
]


@pytest.mark.parametrize("title", SYNDICATED)
def test_strip_site_suffix_removes_mastheads(title: str) -> None:
    assert (
        strip_site_suffix(title) == "AI chatbot travel tips fail Aussie tourists at border crossing"
    )


def test_syndicated_copies_score_as_duplicates() -> None:
    for other in SYNDICATED[1:]:
        assert title_similarity(SYNDICATED[0], other) >= 0.9


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # Trailing clauses that are not mastheads: stripping either of these
        # would merge two stories that report opposite outcomes.
        (
            "Fed holds rates steady - Powell signals cuts",
            "Fed holds rates steady - Powell rules out cuts",
        ),
        ("Nvidia beats estimates - shares jump 12%", "Nvidia beats estimates - shares fall 4%"),
        ("OpenAI ships new model | The Verge", "Anthropic ships new model | The Verge"),
    ],
)
def test_different_stories_are_not_merged_by_stripping(left: str, right: str) -> None:
    assert title_similarity(left, right) < 0.9


@pytest.mark.parametrize(
    "title",
    [
        "Fed holds rates steady - Powell signals cuts",  # tail is a clause, not a masthead
        "Nvidia beats estimates - shares jump 12%",  # digits
        "Cats - Dogs",  # head too short to strip safely
    ],
)
def test_strip_site_suffix_leaves_headlines_alone(title: str) -> None:
    assert strip_site_suffix(title) == title


def test_detector_flags_syndicated_copy(session: Session) -> None:
    persist_articles(
        session,
        [
            RawArticle(
                url="https://moreechampion.com.au/story/1",
                title=SYNDICATED[0],
                description="",
                published_at=BASE_TIME,
            )
        ],
    )
    session.commit()
    detector = DuplicateDetector(session)
    copy = normalize_article(
        RawArticle(
            url="https://queanbeyanage.com.au/story/1",
            title=SYNDICATED[1],
            description="",
            published_at=BASE_TIME + timedelta(hours=1),
        )
    )
    verdict = detector.check(copy)
    assert verdict.is_duplicate
    assert verdict.reason == "title_similarity"
