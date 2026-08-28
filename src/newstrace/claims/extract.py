"""Rule-based claim extraction.

A "claim" here is a candidate factual statement lifted verbatim from retrieved
text: a sentence carrying an entity, a quantity, a date, or an attributed
statement.  Extraction never asserts that a claim is true -- only that a source
reported it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from newstrace.clustering.labeling import extract_entities, extract_numbers, split_sentences
from newstrace.utils import normalize_for_hash, normalize_unicode, sha256_text, tokenize

ATTRIBUTION_VERBS = {
    "said",
    "says",
    "told",
    "announced",
    "reported",
    "confirmed",
    "denied",
    "claimed",
    "stated",
    "warned",
    "argued",
    "added",
    "wrote",
    "disclosed",
    "acknowledged",
    "alleged",
    "estimated",
    "projected",
    "declined",
}

DATE_HINTS = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|"
    r"april|may|june|july|august|september|october|november|december|today|yesterday|"
    r"this week|last week|next week|\d{4})\b",
    re.IGNORECASE,
)

NEGATION = re.compile(r"\b(?:not|no|never|denied|denies|deny|without|rejects?|rejected)\b", re.I)

HEDGE = re.compile(
    r"\b(?:may|might|could|reportedly|allegedly|appears?|seems?|expected to|"
    r"is set to|plans? to|according to)\b",
    re.IGNORECASE,
)

MIN_CLAIM_TOKENS = 6
MAX_CLAIM_TOKENS = 60


@dataclass
class ExtractedClaim:
    """One candidate claim with its span in the source text."""

    text: str
    normalized: str
    claim_hash: str
    start: int
    end: int
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    confidence: float = 0.0
    entities: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    has_attribution: bool = False
    is_negated: bool = False
    is_hedged: bool = False
    extraction_method: str = "rule_v1"


def _svo(sentence: str) -> tuple[str | None, str | None, str | None]:
    """Very rough subject/predicate/object split around an attribution verb."""
    tokens = sentence.split()
    lowered = [t.strip(",.;:").lower() for t in tokens]
    for i, token in enumerate(lowered):
        if token in ATTRIBUTION_VERBS:
            subject = " ".join(tokens[:i]).strip(" ,;:")
            obj = " ".join(tokens[i + 1 :]).strip(" ,;:")
            return (subject or None, token, obj or None)
    if len(tokens) >= 3:
        return (" ".join(tokens[:2]), tokens[2].lower(), " ".join(tokens[3:]) or None)
    return (None, None, None)


def score_claim(
    *, entities: list[str], numbers: list[str], has_attribution: bool, has_date: bool
) -> float:
    """Heuristic confidence in *being a claim*, not in the claim being true."""
    score = 0.2
    score += 0.25 if entities else 0.0
    score += 0.2 if numbers else 0.0
    score += 0.2 if has_attribution else 0.0
    score += 0.15 if has_date else 0.0
    return round(min(1.0, score), 3)


def extract_claims(text: str, *, max_claims: int = 12) -> list[ExtractedClaim]:
    """Extract candidate claims with character offsets into ``text``."""
    cleaned = normalize_unicode(text)
    if not cleaned:
        return []
    claims: list[ExtractedClaim] = []
    cursor = 0
    for sentence in split_sentences(cleaned):
        start = cleaned.find(sentence, cursor)
        if start < 0:
            start = cursor
        end = start + len(sentence)
        cursor = end

        tokens = tokenize(sentence)
        if not (MIN_CLAIM_TOKENS <= len(tokens) <= MAX_CLAIM_TOKENS):
            continue
        entities = extract_entities(sentence, max_entities=6)
        numbers = extract_numbers(sentence)
        has_attribution = any(t in ATTRIBUTION_VERBS for t in tokens)
        has_date = bool(DATE_HINTS.search(sentence))
        if not (entities or numbers or has_attribution or has_date):
            continue

        subject, predicate, obj = _svo(sentence)
        normalized = normalize_for_hash(sentence)
        claims.append(
            ExtractedClaim(
                text=sentence,
                normalized=normalized,
                claim_hash=sha256_text(normalized),
                start=start,
                end=end,
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=score_claim(
                    entities=entities,
                    numbers=numbers,
                    has_attribution=has_attribution,
                    has_date=has_date,
                ),
                entities=entities,
                numbers=numbers,
                has_attribution=has_attribution,
                is_negated=bool(NEGATION.search(sentence)),
                is_hedged=bool(HEDGE.search(sentence)),
            )
        )
        if len(claims) >= max_claims:
            break
    return claims
