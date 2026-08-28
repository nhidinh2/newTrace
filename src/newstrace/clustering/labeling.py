"""Story titles and lightweight entity extraction.

Entity extraction is deliberately rule based: capitalised token runs, tickers,
and numeric/date expressions.  It costs nothing, needs no model download, and is
good enough for overlap features and graph edges.  It is *not* a NER system and
the limitations doc says so.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

from newstrace.utils import normalize_unicode, tokenize

_CAPS_RUN = re.compile(r"\b([A-Z][\w&.'-]*(?:\s+(?:of|the|and|for)?\s*[A-Z][\w&.'-]*)*)\b")
_NUMBER = re.compile(r"\b(?:\$|€|£)?\d[\d,.]*\s?(?:%|percent|billion|million|trillion|bn|m)?\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")

LOWERCASE_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "it",
    "its",
    "this",
    "that",
    "after",
    "over",
    "new",
    "says",
    "said",
    "will",
    "has",
    "have",
}

_LEADING_STOP = re.compile(r"^(?:The|A|An|But|And|In|On|At|For|Of|To|As|After|Over)\s+")

# Calendar words and generic nouns are capitalised constantly in headlines but
# identify nothing, so they must never count as a shared entity.
CALENDAR_WORDS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "today",
    "yesterday",
    "tomorrow",
}
GENERIC_CAPITALS = {
    "report",
    "reports",
    "review",
    "regulators",
    "researchers",
    "analysts",
    "sources",
    "developers",
    "company",
    "two",
    "three",
    "four",
    "five",
    "under",
    "what",
    "state",
    "officials",
    "construction",
    "pricing",
    "availability",
    "security",
    "an",
    "a",
    "the",
}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences without pulling in a heavyweight NLP stack."""
    cleaned = normalize_unicode(text)
    if not cleaned:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(cleaned) if s.strip()]


def extract_entities(text: str, *, max_entities: int = 24) -> list[str]:
    """Return candidate entity strings (capitalised runs, org-like names).

    A single capitalised word is only kept when it also appears away from the
    start of a sentence. Otherwise every headline contributes its first word --
    "Regulators", "Review", "Two" -- and two unrelated stories look like they
    share entities.

    This is a heuristic, not named-entity recognition; ``docs/limitations.md``
    says so.
    """
    cleaned = normalize_unicode(text)
    if not cleaned:
        return []

    sentence_starts: set[int] = {0}
    for match in re.finditer(r"[.!?]\s+", cleaned):
        sentence_starts.add(match.end())

    found: list[str] = []
    seen_mid_sentence: set[str] = set()
    for match in _CAPS_RUN.finditer(cleaned):
        raw = match.group(1)
        candidate = _LEADING_STOP.sub("", raw).strip(" .,'\"-")
        for suffix in ("'s", "\u2019s"):
            if candidate.endswith(suffix):
                candidate = candidate[: -len(suffix)]
        if not candidate or len(candidate) < 2:
            continue
        lowered = candidate.lower()
        if lowered in LOWERCASE_STOPWORDS or lowered in CALENDAR_WORDS:
            continue
        if lowered in GENERIC_CAPITALS:
            continue
        if candidate.isupper() and len(candidate) > 6:
            continue
        if match.start() not in sentence_starts:
            seen_mid_sentence.add(candidate)
        found.append(candidate)

    kept = [name for name in found if " " in name or name in seen_mid_sentence]
    counts = Counter(kept)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ranked[:max_entities]]


def extract_numbers(text: str) -> list[str]:
    return [m.group(0).strip() for m in _NUMBER.finditer(normalize_unicode(text))]


def story_title(titles: Sequence[str]) -> str:
    """Pick a representative story title (longest informative headline)."""
    candidates = [normalize_unicode(t) for t in titles if t and t.strip()]
    if not candidates:
        return "Untitled story"
    return max(candidates, key=lambda t: (len(tokenize(t)), -len(t)))


def keyword_summary(texts: Iterable[str], *, top_k: int = 8) -> list[str]:
    """Most frequent content tokens across a story, used for UI chips."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(t for t in tokenize(text) if t not in LOWERCASE_STOPWORDS and len(t) > 3)
    return [word for word, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]]
