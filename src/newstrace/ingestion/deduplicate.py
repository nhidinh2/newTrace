"""Duplicate and near-duplicate detection.

Duplicates are never deleted.  They are stored, linked to the article they
duplicate, and excluded from independent-source counts: copied or syndicated
reporting is not independent confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.ingestion.normalize import NormalizedArticle
from newstrace.models import Article
from newstrace.utils import ensure_utc, jaccard, tokenize

SIMHASH_BITS = 64


def char_shingles(text: str, size: int = 4) -> set[str]:
    compact = text.replace(" ", "")
    if len(compact) <= size:
        return {compact} if compact else set()
    return {compact[i : i + size] for i in range(len(compact) - size + 1)}


def simhash(text: str, bits: int = SIMHASH_BITS) -> int:
    """Charikar SimHash over word tokens (deterministic, hash-seed independent)."""
    import hashlib

    tokens = tokenize(text)
    if not tokens:
        return 0
    vector = [0] * bits
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(bits):
            vector[bit] += 1 if (value >> bit) & 1 else -1
    out = 0
    for bit in range(bits):
        if vector[bit] > 0:
            out |= 1 << bit
    return out


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def simhash_similarity(a: str, b: str, bits: int = SIMHASH_BITS) -> float:
    return 1.0 - hamming_distance(simhash(a, bits), simhash(b, bits)) / bits


# Mastheads a syndicating publisher appends to an otherwise identical headline:
# "Wire headline | Moree Champion". Both separators and the check below stay
# deliberately narrow -- a false strip would merge two genuinely different
# stories that happen to share an opening clause.
_TITLE_SEPARATORS = (" | ", " — ", " – ", " - ", " · ", " :: ")
_PUBLICATION_STOPWORDS = frozenset(
    {"a", "an", "and", "at", "de", "du", "for", "in", "la", "le", "of", "on", "the"}
)


def _looks_like_publication(tail: str) -> bool:
    """True when ``tail`` reads as a masthead rather than part of the headline.

    A masthead is short, capitalised and free of the digits and sentence
    punctuation that a trailing headline clause almost always carries.
    """
    tokens = tail.split()
    if not 1 <= len(tokens) <= 5:
        return False
    if any(character.isdigit() for character in tail):
        return False
    if any(character in tail for character in ".!?%:;,\"'"):
        return False
    alphabetic = [token for token in tokens if token[:1].isalpha()]
    if not alphabetic:
        return False
    return all(
        token[:1].isupper() or token.lower() in _PUBLICATION_STOPWORDS for token in alphabetic
    )


def strip_site_suffix(title: str, max_strips: int = 2) -> str:
    """Drop trailing mastheads so syndicated copies compare as the same headline.

    Two passes, because a masthead can itself be compound: "... | The
    Advertiser - Cessnock". The headline must keep enough words to stay
    meaningful, otherwise the title is returned untouched.
    """
    result = title.strip()
    for _ in range(max_strips):
        cut = -1
        width = 0
        for separator in _TITLE_SEPARATORS:
            position = result.rfind(separator)
            if position > cut:
                cut, width = position, len(separator)
        if cut <= 0:
            break
        head, tail = result[:cut].strip(), result[cut + width :].strip()
        if len(head.split()) < 3 or len(head) < 15 or not _looks_like_publication(tail):
            break
        result = head
    return result


def title_similarity(a: str, b: str) -> float:
    """Blend token Jaccard with 4-gram character Jaccard for robustness.

    Scored twice, with and without trailing mastheads, and the better score
    wins: stripping may only rescue a match that the masthead was hiding, never
    weaken one.
    """

    def score(left: str, right: str) -> float:
        token_score = jaccard(set(tokenize(left)), set(tokenize(right)))
        char_score = jaccard(char_shingles(left), char_shingles(right))
        return 0.5 * token_score + 0.5 * char_score

    raw = score(a, b)
    stripped_a, stripped_b = strip_site_suffix(a), strip_site_suffix(b)
    if stripped_a == a and stripped_b == b:
        return raw
    return max(raw, score(stripped_a, stripped_b))


@dataclass
class DuplicateVerdict:
    """Outcome of duplicate checking for one candidate article."""

    is_duplicate: bool
    duplicate_of_id: int | None = None
    reason: str | None = None
    similarity: float = 0.0


class DuplicateDetector:
    """Multi-signal duplicate detection against already-persisted articles."""

    def __init__(
        self,
        session: Session,
        *,
        title_threshold: float = 0.9,
        excerpt_threshold: float = 0.92,
        window_hours: int = 96,
    ) -> None:
        self.session = session
        self.title_threshold = title_threshold
        self.excerpt_threshold = excerpt_threshold
        self.window_hours = window_hours

    def _candidates(self, article: NormalizedArticle) -> list[Article]:
        published = ensure_utc(article.published_at)
        stmt = select(Article).where(Article.is_near_duplicate.is_(False))
        if published is not None:
            low = published - timedelta(hours=self.window_hours)
            high = published + timedelta(hours=self.window_hours)
            stmt = stmt.where(
                (Article.published_at.is_(None))
                | ((Article.published_at >= low) & (Article.published_at <= high))
            )
        return list(self.session.execute(stmt.limit(2000)).scalars())

    def check(self, article: NormalizedArticle) -> DuplicateVerdict:
        """Return a verdict for ``article`` against stored articles."""
        # 1. Exact canonical URL -- same document, not a near duplicate.
        existing = self.session.execute(
            select(Article).where(Article.canonical_url == article.canonical_url)
        ).scalar_one_or_none()
        if existing is not None:
            return DuplicateVerdict(True, existing.id, "canonical_url", 1.0)

        # 2. Exact normalized-text hash.
        text_match = (
            self.session.execute(
                select(Article).where(Article.normalized_text_hash == article.normalized_text_hash)
            )
            .scalars()
            .first()
        )
        if text_match is not None:
            return DuplicateVerdict(True, text_match.id, "normalized_text_hash", 1.0)

        # 3. Exact normalized-title hash.
        title_match = (
            self.session.execute(
                select(Article).where(
                    Article.normalized_title_hash == article.normalized_title_hash
                )
            )
            .scalars()
            .first()
        )
        if title_match is not None:
            return DuplicateVerdict(True, title_match.id, "title_hash", 1.0)

        # 4/5. Near-duplicate title or excerpt within the publication-time window.
        title = article.title
        excerpt = article.excerpt
        best: DuplicateVerdict | None = None
        for candidate in self._candidates(article):
            score = title_similarity(title, candidate.title)
            reason = "title_similarity"
            if score < self.title_threshold and excerpt and candidate.excerpt:
                excerpt_score = simhash_similarity(excerpt, candidate.excerpt)
                if excerpt_score >= self.excerpt_threshold:
                    score, reason = excerpt_score, "excerpt_simhash"
            if score >= min(self.title_threshold, self.excerpt_threshold) and (
                best is None or score > best.similarity
            ):
                best = DuplicateVerdict(True, candidate.id, reason, float(score))
        if best is not None:
            return best
        return DuplicateVerdict(False)
