"""Blocking keys: the candidate lookup must not change a single verdict.

The point of ``article_signatures`` is that duplicate detection stops reading
a 96-hour window of articles and starts reading an index. That is only a win
if the index is *complete* for the rules it serves, so these tests check the
guarantee itself, not just that the fast path returns something.
"""

from __future__ import annotations

import random

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.ingestion.base import RawArticle
from newstrace.ingestion.deduplicate import (
    BAND_BITS,
    KIND_BAND,
    KIND_TITLE,
    SIMHASH_BANDS,
    DuplicateDetector,
    band_keys,
    hamming_distance,
    index_article,
    signature_values,
    simhash,
    title_blocking_tokens,
    title_similarity,
    to_signed64,
    to_unsigned64,
)
from newstrace.ingestion.normalize import normalize_article
from newstrace.ingestion.pipeline import persist_articles
from newstrace.models import Article, ArticleSignature
from newstrace.utils import tokenize
from tests.conftest import BASE_TIME


def _flip_bits(value: int, count: int, rng: random.Random) -> int:
    for bit in rng.sample(range(64), count):
        value ^= 1 << bit
    return value


@pytest.mark.parametrize("differing_bits", [0, 1, 3, 5, 7])
def test_band_lookup_finds_every_pair_the_rule_could_accept(differing_bits: int) -> None:
    """Eight bands guarantee a shared band up to seven differing bits.

    The excerpt rule accepts at >= 0.92 similarity, i.e. at most five flipped
    bits out of 64, so the guarantee covers it with margin.
    """
    rng = random.Random(549)
    for _ in range(200):
        left = rng.getrandbits(64)
        right = _flip_bits(left, differing_bits, rng)
        assert hamming_distance(left, right) <= differing_bits
        assert set(band_keys(left)) & set(band_keys(right))


def test_band_keys_cover_the_whole_digest() -> None:
    keys = band_keys(simhash("Northwind Labs releases Atlas 3 agent model today"))
    assert len(keys) == SIMHASH_BANDS
    assert {key.split(":")[0] for key in keys} == {str(i) for i in range(SIMHASH_BANDS)}
    assert all(0 <= int(key.split(":")[1]) < 2**BAND_BITS for key in keys)


def test_signed_round_trip_survives_sqlite() -> None:
    for value in (0, 1, 2**63 - 1, 2**63, 2**64 - 1):
        signed = to_signed64(value)
        assert -(2**63) <= signed < 2**63
        assert to_unsigned64(signed) == value


def test_title_prefix_filter_is_complete_for_the_threshold() -> None:
    """Any pair the title rule could accept shares a prefix token.

    A blended score of 0.9 needs a token Jaccard of at least 0.8, so the
    filter is applied at 0.8 -- these random edits check that the looser bound
    is the right one.
    """
    rng = random.Random(7)
    words = [f"word{i}" for i in range(40)]
    for _ in range(300):
        base = rng.sample(words, rng.randint(6, 12))
        edited = list(base)
        if rng.random() < 0.7:
            edited[rng.randrange(len(edited))] = rng.choice(words)
        left, right = " ".join(base), " ".join(edited)
        if title_similarity(left, right) < 0.9:
            continue
        assert title_blocking_tokens(left) & title_blocking_tokens(right), (left, right)


def test_masthead_variant_is_reachable() -> None:
    """A syndicated copy must be findable through the stripped headline too."""
    wire = "Northwind Labs releases Atlas 3 agent model"
    syndicated = f"{wire} | The Moree Champion"
    assert title_blocking_tokens(wire) & title_blocking_tokens(syndicated)


def test_signature_values_skip_bands_without_an_excerpt() -> None:
    kinds = {kind for kind, _ in signature_values("A headline with several words", "")}
    assert kinds == {KIND_TITLE}
    kinds = {kind for kind, _ in signature_values("A headline with several words", "Body text.")}
    assert kinds == {KIND_TITLE, KIND_BAND}


def test_index_article_is_idempotent(session: Session) -> None:
    article = Article(
        url="https://alpha.example/a",
        canonical_url="https://alpha.example/a",
        title="Northwind Labs releases Atlas 3 agent model",
        excerpt="Northwind Labs said Atlas 3 scores 71.4 percent on the public benchmark.",
        source_domain="alpha.example",
        normalized_text_hash="x",
        normalized_title_hash="y",
    )
    session.add(article)
    session.flush()

    first = index_article(session, article)
    session.flush()
    rows = session.execute(
        select(ArticleSignature).where(ArticleSignature.article_id == article.id)
    ).scalars()
    assert len({(r.kind, r.value) for r in rows}) == first
    assert article.simhash == to_signed64(simhash(article.excerpt))

    index_article(session, article)
    session.flush()
    again = session.execute(
        select(ArticleSignature).where(ArticleSignature.article_id == article.id)
    ).scalars()
    assert len(list(again)) == first


def test_index_article_rewrites_keys_when_the_text_changes(session: Session) -> None:
    article = Article(
        url="https://alpha.example/b",
        canonical_url="https://alpha.example/b",
        title="Grid operator warns of data centre demand",
        excerpt="The operator said demand could add six gigawatts by 2030.",
        source_domain="alpha.example",
        normalized_text_hash="x",
        normalized_title_hash="y",
    )
    session.add(article)
    session.flush()
    index_article(session, article)
    session.flush()
    before = {
        (r.kind, r.value)
        for r in session.execute(
            select(ArticleSignature).where(ArticleSignature.article_id == article.id)
        ).scalars()
    }

    article.title = "Chip maker reports record lithography yields"
    article.excerpt = "The manufacturer reported a record yield on its newest line."
    index_article(session, article)
    session.flush()
    after = {
        (r.kind, r.value)
        for r in session.execute(
            select(ArticleSignature).where(ArticleSignature.article_id == article.id)
        ).scalars()
    }
    assert before != after
    assert article.simhash == to_signed64(simhash(article.excerpt))


def test_persisted_articles_get_blocking_keys(session: Session, raw_articles: list) -> None:
    persist_articles(session, raw_articles)
    session.commit()
    articles = list(session.execute(select(Article)).scalars())
    assert articles
    for article in articles:
        keys = session.execute(
            select(ArticleSignature).where(ArticleSignature.article_id == article.id)
        ).scalars()
        assert list(keys), f"article {article.id} has no blocking keys"


def test_blocking_reproduces_the_exhaustive_scan(session: Session) -> None:
    """The candidate set never drops a pair the full scan would have accepted."""
    rng = random.Random(11)
    subjects = [
        "Northwind Labs releases Atlas 3 agent model",
        "Grid operator warns data centre demand could add six gigawatts",
        "Chip maker reports record lithography yields this quarter",
        "Regulator opens consultation on model evaluations",
    ]
    raws = []
    for index in range(40):
        subject = subjects[index % len(subjects)]
        tokens = subject.split()
        if rng.random() < 0.5:
            tokens.insert(rng.randrange(len(tokens)), "reportedly")
        raws.append(
            RawArticle(
                url=f"https://pub{index}.example/story-{index}",
                title=" ".join(tokens),
                description=f"{subject}. Filing number {index % 7} was published on Tuesday.",
                source_domain=f"pub{index}.example",
                language="English",
                published_at=BASE_TIME,
                topic="ai_models",
                source_adapter="test",
            )
        )
    persist_articles(session, raws)
    session.commit()

    detector = DuplicateDetector(session, window_hours=96)
    stored = list(session.execute(select(Article)).scalars())
    live = [a for a in stored if not a.is_near_duplicate]

    for article in stored:
        probe = normalize_article(
            RawArticle(
                url=article.url,
                title=article.title,
                description=article.excerpt,
                source_domain=article.source_domain,
                language="English",
                published_at=article.published_at,
                source_adapter="test",
            )
        )
        blocked = set(detector._candidate_ids(probe))
        exhaustive = {
            other.id
            for other in live
            if other.id != article.id and title_similarity(probe.title, other.title) >= 0.9
        }
        assert exhaustive <= blocked | {article.id}


def test_candidate_lookup_stays_small(session: Session) -> None:
    """Blocking exists to keep the candidate list from growing with the corpus."""
    raws = [
        RawArticle(
            url=f"https://pub{i}.example/unrelated-{i}",
            title=f"Unrelated headline about {tokenize('subject')[0]} number {i} in the archive",
            description=f"An unrelated report number {i} with its own distinct wording entirely.",
            source_domain=f"pub{i}.example",
            language="English",
            published_at=BASE_TIME,
            source_adapter="test",
        )
        for i in range(60)
    ]
    persist_articles(session, raws)
    session.commit()

    probe = normalize_article(
        RawArticle(
            url="https://new.example/story",
            title="A completely different headline about quantum error correction research",
            description="Nothing here overlaps with the archive's wording at all.",
            source_domain="new.example",
            language="English",
            published_at=BASE_TIME,
            source_adapter="test",
        )
    )
    detector = DuplicateDetector(session, window_hours=96)
    assert len(detector._candidate_ids(probe)) < 60


def test_max_candidates_is_respected(session: Session, raw_articles: list) -> None:
    persist_articles(session, raw_articles)
    session.commit()
    detector = DuplicateDetector(session, window_hours=96, max_candidates=1)
    probe = normalize_article(
        RawArticle(
            url="https://delta.example/atlas-3",
            title="Northwind Labs releases Atlas 3 agent model",
            description="Northwind Labs said Atlas 3 scores 71.4 percent.",
            source_domain="delta.example",
            language="English",
            published_at=BASE_TIME,
            source_adapter="test",
        )
    )
    assert len(detector._candidate_ids(probe)) <= 1
