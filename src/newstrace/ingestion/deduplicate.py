"""Duplicate and near-duplicate detection.

Duplicates are never deleted.  They are stored, linked to the article they
duplicate, and excluded from independent-source counts: copied or syndicated
reporting is not independent confirmation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from newstrace.ingestion.normalize import NormalizedArticle
from newstrace.models import Article, ArticleSignature
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


# --- Blocking keys -----------------------------------------------------------
#
# Eight 8-bit bands. Two SimHashes that differ in at most 7 bits must agree on
# at least one whole band (pigeonhole), and the excerpt rule accepts a pair only
# at >= 0.92 similarity, i.e. at most 5 differing bits -- so band lookup returns
# every pair the rule could accept, with margin. Fewer, wider bands would be
# cheaper and would start missing duplicates.

SIMHASH_BANDS = 8
BAND_BITS = SIMHASH_BITS // SIMHASH_BANDS
BAND_MASK = (1 << BAND_BITS) - 1

KIND_BAND = "simhash_band"
KIND_TITLE = "title_prefix"

# A blended title score of 0.9 needs a token Jaccard of at least 0.8, because
# the character-shingle half of the blend cannot exceed 1.0. The prefix filter
# is applied at that looser bound so it can never drop a pair the threshold
# would have accepted.
TITLE_PREFIX_JACCARD = 0.8


def to_signed64(value: int) -> int:
    """SQLite integers are signed; SimHash is not."""
    value &= (1 << 64) - 1
    return value - (1 << 64) if value >= (1 << 63) else value


def to_unsigned64(value: int | None) -> int:
    return 0 if value is None else value & ((1 << 64) - 1)


def band_keys(digest: int) -> list[str]:
    """The band blocking keys of one SimHash, as ``"<band>:<byte>"``."""
    unsigned = to_unsigned64(digest)
    return [
        f"{band}:{(unsigned >> (band * BAND_BITS)) & BAND_MASK}" for band in range(SIMHASH_BANDS)
    ]


def _prefix_tokens(tokens: set[str], threshold: float = TITLE_PREFIX_JACCARD) -> list[str]:
    """The prefix-filter tokens of one title.

    Tokens are ordered by a fixed, corpus-independent hash rather than by
    frequency, because the filter only needs *some* global order that indexing
    and probing agree on -- and a frequency order would change as the corpus
    grows, silently invalidating rows written earlier.

    For Jaccard ``t`` the first ``|A| - ceil(t|A|) + 1`` tokens suffice: two
    sets whose prefixes are disjoint cannot overlap enough to reach ``t``.
    """
    if not tokens:
        return []
    ordered = sorted(tokens, key=lambda token: (blake_order(token), token))
    size = len(ordered)
    keep = max(1, size - math.ceil(threshold * size) + 1)
    return ordered[:keep]


def blake_order(token: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def title_blocking_tokens(title: str) -> set[str]:
    """Prefix tokens for both the raw and the masthead-stripped headline.

    ``title_similarity`` scores the pair twice and keeps the better score, so a
    candidate that only matches after stripping must still be reachable.
    """
    keys: set[str] = set()
    for variant in {title, strip_site_suffix(title)}:
        keys.update(_prefix_tokens(set(tokenize(variant))))
    return keys


def signature_values(title: str, excerpt: str, digest: int | None = None) -> list[tuple[str, str]]:
    """Every ``(kind, value)`` blocking key for one article."""
    values: list[tuple[str, str]] = []
    if excerpt:
        code = simhash(excerpt) if digest is None else to_unsigned64(digest)
        values.extend((KIND_BAND, key) for key in band_keys(code))
    values.extend((KIND_TITLE, token) for token in title_blocking_tokens(title))
    return values


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


def index_article(session: Session, article: Article) -> int:
    """Write (or rewrite) the SimHash and blocking keys for a stored article.

    Called after every insert and whenever an article's text changes, so the
    keys can never describe an older version of the text.
    """
    digest = simhash(article.excerpt) if article.excerpt else None
    article.simhash = to_signed64(digest) if digest is not None else None

    wanted = set(signature_values(article.title or "", article.excerpt or "", digest))
    current = {
        (row.kind, row.value): row
        for row in session.execute(
            select(ArticleSignature).where(ArticleSignature.article_id == article.id)
        ).scalars()
    }
    for key, row in current.items():
        if key not in wanted:
            session.delete(row)
    for kind, value in sorted(wanted - set(current)):
        session.add(ArticleSignature(article_id=article.id, kind=kind, value=value))
    return len(wanted)


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
        max_candidates: int = 400,
    ) -> None:
        self.session = session
        self.title_threshold = title_threshold
        self.excerpt_threshold = excerpt_threshold
        self.window_hours = window_hours
        self.max_candidates = max_candidates
        self.last_candidate_count = 0

    def _candidate_ids(self, article: NormalizedArticle) -> list[int]:
        """Article ids sharing a blocking key with ``article``.

        Both blocking schemes are *complete* for the rules they serve: a pair
        the SimHash rule would accept always shares a band, and a pair the
        title rule would accept always shares a prefix token. So this narrows
        the work without changing a single verdict -- unlike the fixed
        ``LIMIT 2000`` window scan it replaces, which quietly compared each
        article against an arbitrary subset once the corpus outgrew the cap.
        """
        keys = signature_values(article.title, article.excerpt)
        if not keys:
            return []
        conditions = [
            (ArticleSignature.kind == kind) & (ArticleSignature.value == value)
            for kind, value in keys
        ]
        stmt = (
            select(ArticleSignature.article_id)
            .where(or_(*conditions))
            .group_by(ArticleSignature.article_id)
            # Ordering by shared-key count first means that if the guard below
            # ever truncates, it keeps the most promising candidates rather
            # than whichever rows the planner happened to emit.
            .order_by(func.count(ArticleSignature.id).desc(), ArticleSignature.article_id)
            .limit(self.max_candidates)
        )
        return [int(article_id) for article_id in self.session.execute(stmt).scalars()]

    def _candidates(self, article: NormalizedArticle) -> list[Article]:
        ids = self._candidate_ids(article)
        self.last_candidate_count = len(ids)
        if not ids:
            return []
        stmt = select(Article).where(Article.id.in_(ids), Article.is_near_duplicate.is_(False))
        published = ensure_utc(article.published_at)
        if published is not None:
            low = published - timedelta(hours=self.window_hours)
            high = published + timedelta(hours=self.window_hours)
            stmt = stmt.where(
                (Article.published_at.is_(None))
                | ((Article.published_at >= low) & (Article.published_at <= high))
            )
        return list(self.session.execute(stmt).scalars())

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
        # Hashed once per incoming article rather than once per candidate pair:
        # SimHash walks every token across 64 bit positions, and recomputing it
        # on both sides of every comparison was the single most expensive thing
        # ingestion did.
        incoming_hash = simhash(excerpt) if excerpt else None
        best: DuplicateVerdict | None = None
        for candidate in self._candidates(article):
            score = title_similarity(title, candidate.title)
            reason = "title_similarity"
            if score < self.title_threshold and incoming_hash is not None and candidate.excerpt:
                candidate_hash = (
                    to_unsigned64(candidate.simhash)
                    if candidate.simhash is not None
                    else simhash(candidate.excerpt)
                )
                excerpt_score = 1.0 - hamming_distance(incoming_hash, candidate_hash) / SIMHASH_BITS
                if excerpt_score >= self.excerpt_threshold:
                    score, reason = excerpt_score, "excerpt_simhash"
            if score >= min(self.title_threshold, self.excerpt_threshold) and (
                best is None or score > best.similarity
            ):
                best = DuplicateVerdict(True, candidate.id, reason, float(score))
        if best is not None:
            return best
        return DuplicateVerdict(False)
