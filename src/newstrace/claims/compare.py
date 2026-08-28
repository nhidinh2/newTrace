"""Cautious comparison of claims across sources.

The vocabulary is deliberately narrow:

* ``repeated``      -- reported by more than one *independent* (non-duplicate) domain
* ``single_source`` -- currently supported by exactly one independent source
* ``unclear``       -- attributed statements that cannot be reconciled from the
  available evidence (e.g. one source negates what another asserts, or two
  sources give different numbers for the same quantity)

None of these labels says a claim is true, false, or that a publisher is
trustworthy.  Copied or syndicated text is never counted as confirmation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from newstrace.claims.extract import ExtractedClaim
from newstrace.utils import jaccard, tokenize

SIMILARITY_THRESHOLD = 0.55
CONFLICT_MIN_SIMILARITY = 0.18
CONFLICT_MIN_SHARED_ENTITIES = 1
NEGATION_MIN_SIMILARITY = 0.75
CONFLICT_MIN_FRAME_OVERLAP = 0.45
CONFLICT_MIN_ENTITY_AGREEMENT = 0.5

_QUANTITY = re.compile(
    r"(?:\$|€|£)?\s?(\d[\d,]*(?:\.\d+)?)\s*(percent|%|billion|million|trillion|bn|jobs|"
    r"gigawatts?|people|positions)?",
    re.IGNORECASE,
)

_UNIT_ALIASES = {
    "%": "percent",
    "bn": "billion",
    "gigawatt": "gigawatts",
    "position": "jobs",
    "positions": "jobs",
    "people": "jobs",
}


@dataclass
class ClaimOccurrence:
    """One article's rendition of a claim."""

    article_id: int
    source_domain: str
    is_near_duplicate: bool
    claim: ExtractedClaim


@dataclass
class ClaimGroup:
    """A set of occurrences judged to express the same claim."""

    representative: ExtractedClaim
    occurrences: list[ClaimOccurrence] = field(default_factory=list)

    @property
    def independent_domains(self) -> set[str]:
        return {
            o.source_domain for o in self.occurrences if not o.is_near_duplicate and o.source_domain
        }

    @property
    def duplicate_count(self) -> int:
        return sum(1 for o in self.occurrences if o.is_near_duplicate)

    def status(self) -> str:
        if self.is_conflicting():
            return "unclear"
        return "repeated" if len(self.independent_domains) > 1 else "single_source"

    def is_conflicting(self) -> bool:
        """Whether independent occurrences of this claim disagree with each other.

        Two tests, both deliberately conservative:

        * the same *unit* is given different values (``$4.2 billion`` versus
          ``$3.8 billion``) -- comparing bare numbers would treat a year and a
          dollar figure as a contradiction;
        * one occurrence negates what another asserts, and the two are near
          paraphrases. Without the similarity floor, an incidental "did not
          respond to a request for comment" flips the whole group to disputed.
        """
        independent = [o for o in self.occurrences if not o.is_near_duplicate]
        if len(independent) < 2:
            return False

        by_unit: dict[str, set[float]] = {}
        for occurrence in independent:
            for unit, values in quantities(occurrence.claim).items():
                if unit in by_unit and by_unit[unit].isdisjoint(values):
                    return True
                by_unit.setdefault(unit, set()).update(values)

        for i, left in enumerate(independent):
            for right in independent[i + 1 :]:
                if left.claim.is_negated == right.claim.is_negated:
                    continue
                if claim_similarity(left.claim, right.claim) >= NEGATION_MIN_SIMILARITY:
                    return True
        return False

    def confidence(self) -> float:
        base = max((o.claim.confidence for o in self.occurrences), default=0.0)
        bonus = 0.1 * (len(self.independent_domains) - 1)
        hedged = any(o.claim.is_hedged for o in self.occurrences)
        return round(max(0.0, min(1.0, base + bonus - (0.1 if hedged else 0.0))), 3)


def quantities(claim: ExtractedClaim) -> dict[str, set[float]]:
    """Map a unit ("percent", "billion", ...) to the values asserted with it.

    Comparing bare numbers across sentences produces nonsense conflicts -- a year
    is not a dollar figure -- so a disagreement is only considered when two
    passages state *different values for the same unit*.
    """
    out: dict[str, set[float]] = {}
    for match in _QUANTITY.finditer(claim.text):
        raw, unit = match.group(1), (match.group(2) or "").lower()
        if not unit:
            continue
        unit = _UNIT_ALIASES.get(unit, unit)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        out.setdefault(unit, set()).add(value)
    return out


def claim_similarity(a: ExtractedClaim, b: ExtractedClaim) -> float:
    """Token Jaccard with an entity-overlap bonus."""
    token_score = jaccard(set(tokenize(a.normalized)), set(tokenize(b.normalized)))
    entity_score = jaccard({e.lower() for e in a.entities}, {e.lower() for e in b.entities})
    return 0.75 * token_score + 0.25 * entity_score


@dataclass
class ConflictPair:
    """Two claim groups that cannot be reconciled from the available evidence."""

    left: ClaimGroup
    right: ClaimGroup
    reason: str
    detail: str = ""

    def independent_domains(self) -> set[str]:
        return self.left.independent_domains | self.right.independent_domains


def _shared_entities(a: ClaimGroup, b: ClaimGroup) -> set[str]:
    left = {e.lower() for o in a.occurrences for e in o.claim.entities}
    right = {e.lower() for o in b.occurrences for e in o.claim.entities}
    return left & right


def frame_overlap(a: ExtractedClaim, b: ExtractedClaim) -> float:
    """Token overlap ignoring numbers: do these say the same thing apart from the figure?"""

    def frame(claim: ExtractedClaim) -> set[str]:
        return {t for t in tokenize(claim.normalized) if not any(c.isdigit() for c in t)}

    return jaccard(frame(a), frame(b))


def _denied_value(negated: ExtractedClaim, asserted: ExtractedClaim) -> tuple[str, float] | None:
    """Return a (unit, value) the negated passage repeats from the asserted one."""
    negated_q, asserted_q = quantities(negated), quantities(asserted)
    for unit in set(negated_q) & set(asserted_q):
        shared_values = negated_q[unit] & asserted_q[unit]
        if shared_values:
            return unit, sorted(shared_values)[0]
    return None


def find_conflicts(
    groups: Sequence[ClaimGroup], *, min_similarity: float = CONFLICT_MIN_SIMILARITY
) -> list[ConflictPair]:
    """Find claim groups that disagree, across groups rather than inside one.

    Two sources reporting the same disagreement rarely phrase it alike -- one
    asserts a figure, the other denies it -- so they land in different groups
    and a within-group check can never see the disagreement.

    Two conservative rules, both requiring at least one shared entity:

    * **denied_value** -- one passage is negated and repeats a value the other
      asserts for the same unit ("does not score 71.4 percent" against "scores
      71.4 percent"). This is the strongest available signal and needs no
      lexical-similarity guess.
    * **different_values_for_same_unit** -- both passages assert, give different
      values for the same unit, and are otherwise the same sentence frame. The
      frame test is what stops two genuinely different facts about one company
      ($1.7bn for one plant, $5.9bn for another) from being called a
      contradiction.

    Neither rule says which side is correct. Both say the evidence available to
    NewsTrace cannot reconcile the two passages.
    """
    conflicts: list[ConflictPair] = []
    for i, left in enumerate(groups):
        for right in groups[i + 1 :]:
            if len(left.independent_domains | right.independent_domains) < 2:
                continue
            if len(_shared_entities(left, right)) < CONFLICT_MIN_SHARED_ENTITIES:
                continue
            if claim_similarity(left.representative, right.representative) >= SIMILARITY_THRESHOLD:
                continue  # near paraphrases belong to the within-group check

            lc, rc = left.representative, right.representative
            if lc.is_negated != rc.is_negated:
                negated, asserted = (lc, rc) if lc.is_negated else (rc, lc)
                found = _denied_value(negated, asserted)
                if found is not None:
                    unit, value = found
                    conflicts.append(
                        ConflictPair(
                            left,
                            right,
                            "denied_value",
                            f"one source denies {value:g} {unit} that another asserts",
                        )
                    )
                continue

            if lc.is_negated or rc.is_negated:
                continue
            if frame_overlap(lc, rc) < CONFLICT_MIN_FRAME_OVERLAP:
                continue
            # Same frame, different subject: "$1.7bn for the Northgate plant" and
            # "$5.9bn for the Sandhill facility" read alike but are two facts, not
            # a contradiction. Require the two passages to name the same things.
            if (
                jaccard({e.lower() for e in lc.entities}, {e.lower() for e in rc.entities})
                < CONFLICT_MIN_ENTITY_AGREEMENT
            ):
                continue
            left_q, right_q = quantities(lc), quantities(rc)
            for unit in sorted(set(left_q) & set(right_q)):
                if left_q[unit] and right_q[unit] and left_q[unit].isdisjoint(right_q[unit]):
                    conflicts.append(
                        ConflictPair(
                            left,
                            right,
                            "different_values_for_same_unit",
                            f"{sorted(left_q[unit])} vs {sorted(right_q[unit])} ({unit})",
                        )
                    )
                    break
    conflicts.sort(
        key=lambda c: (c.left.representative.claim_hash, c.right.representative.claim_hash)
    )
    return conflicts


def group_claims(
    occurrences: Sequence[ClaimOccurrence], *, threshold: float = SIMILARITY_THRESHOLD
) -> list[ClaimGroup]:
    """Greedy single-pass grouping, ordered for determinism."""
    ordered = sorted(
        occurrences,
        key=lambda o: (-o.claim.confidence, o.article_id, o.claim.claim_hash),
    )
    groups: list[ClaimGroup] = []
    for occurrence in ordered:
        best: ClaimGroup | None = None
        best_score = 0.0
        for group in groups:
            if occurrence.claim.claim_hash == group.representative.claim_hash:
                best, best_score = group, 1.0
                break
            score = claim_similarity(occurrence.claim, group.representative)
            if score > best_score:
                best, best_score = group, score
        if best is not None and best_score >= threshold:
            best.occurrences.append(occurrence)
        else:
            groups.append(ClaimGroup(representative=occurrence.claim, occurrences=[occurrence]))
    groups.sort(
        key=lambda g: (-len(g.independent_domains), -g.confidence(), g.representative.claim_hash)
    )
    return groups
