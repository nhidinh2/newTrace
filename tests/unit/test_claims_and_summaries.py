"""Claim extraction, cautious comparison, and evidence-grounded summaries."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from newstrace.claims.compare import (
    ClaimOccurrence,
    ConflictPair,
    find_conflicts,
    group_claims,
    quantities,
)
from newstrace.claims.evidence import load_claims, persist_claims
from newstrace.claims.extract import extract_claims
from newstrace.clustering.labeling import extract_entities, split_sentences, story_title
from newstrace.evaluation.summaries import evaluate_summary, statement_supported
from newstrace.models import Claim, ClaimEvidence
from newstrace.stories import load_story
from newstrace.summarization.base import SummarySection, SummaryStatement
from newstrace.summarization.extractive import ExtractiveSummarizer


def test_split_sentences() -> None:
    assert split_sentences("One thing. Two things! Three?") == [
        "One thing.",
        "Two things!",
        "Three?",
    ]
    assert split_sentences("") == []


def test_entity_extraction_skips_calendar_and_sentence_starts() -> None:
    entities = extract_entities(
        "Regulators opened a review of Larkspur Robotics on Wednesday. "
        "Larkspur Robotics declined to comment."
    )
    assert "Larkspur Robotics" in entities
    assert "Wednesday" not in entities
    assert "Regulators" not in entities


def test_story_title_prefers_the_informative_headline() -> None:
    assert story_title(["Short", "A considerably longer and more informative headline"]) == (
        "A considerably longer and more informative headline"
    )
    assert story_title([]) == "Untitled story"


def test_extract_claims_finds_quantities_and_attribution() -> None:
    claims = extract_claims(
        "Northwind Labs said Atlas 3 scores 71.4 percent on the public agent benchmark. "
        "It was nice weather."
    )
    assert claims
    first = claims[0]
    assert "71.4 percent" in first.text
    assert first.has_attribution
    assert first.numbers
    assert 0 < first.confidence <= 1.0
    assert first.claim_hash and len(first.claim_hash) == 64


def test_extract_claims_records_offsets() -> None:
    text = "Padding sentence here that is long enough. Northwind Labs said Atlas 3 scores 71.4 percent today."
    claims = extract_claims(text)
    for claim in claims:
        assert text[claim.start : claim.end] == claim.text


def test_extract_claims_ignores_short_or_empty_text() -> None:
    assert extract_claims("") == []
    assert extract_claims("Too short.") == []


def test_negation_and_hedging_are_detected() -> None:
    claims = extract_claims(
        "Northwind Labs said Atlas 3 does not score 71.4 percent on the public benchmark."
    )
    assert claims[0].is_negated
    hedged = extract_claims(
        "Northwind Labs may release the Atlas 4 model in September, according to two people."
    )
    assert hedged[0].is_hedged


def test_quantities_groups_values_by_unit() -> None:
    claim = extract_claims(
        "Meridian Semiconductor said the plant will cost $4.2 billion and employ 1,800 people."
    )[0]
    units = quantities(claim)
    assert units["billion"] == {4.2}
    assert units["jobs"] == {1800.0}


def _occurrence(article_id: int, domain: str, text: str, *, duplicate: bool = False):
    claims = extract_claims(text)
    assert claims, f"no claim extracted from {text!r}"
    return ClaimOccurrence(article_id, domain, duplicate, claims[0])


def test_repeated_claims_need_two_independent_domains() -> None:
    text = "Northwind Labs said Atlas 3 scores 71.4 percent on the public agent benchmark."
    groups = group_claims(
        [_occurrence(1, "alpha.example", text), _occurrence(2, "beta.example", text)]
    )
    assert len(groups) == 1
    assert groups[0].status() == "repeated"
    assert groups[0].independent_domains == {"alpha.example", "beta.example"}


def test_syndicated_copies_are_not_independent_confirmation() -> None:
    text = "Northwind Labs said Atlas 3 scores 71.4 percent on the public agent benchmark."
    groups = group_claims(
        [
            _occurrence(1, "alpha.example", text),
            _occurrence(2, "wire.example", text, duplicate=True),
        ]
    )
    assert groups[0].status() == "single_source"
    assert groups[0].independent_domains == {"alpha.example"}
    assert groups[0].duplicate_count == 1


def test_denied_value_is_reported_as_a_disagreement() -> None:
    asserted = _occurrence(
        1,
        "alpha.example",
        "Northwind Labs said Atlas 3 scores 71.4 percent on the public agent benchmark.",
    )
    denied = _occurrence(
        2,
        "beta.example",
        "Two analysts said Atlas 3 does not score 71.4 percent on the public agent benchmark, "
        "putting the comparable figure closer to 64 percent.",
    )
    conflicts = find_conflicts(group_claims([asserted, denied]))
    assert conflicts
    assert conflicts[0].reason == "denied_value"
    assert "71.4" in conflicts[0].detail


def test_different_facts_about_one_company_are_not_a_disagreement() -> None:
    left = _occurrence(
        1,
        "alpha.example",
        "Regulators opened a review of Larkspur Robotics's $1.7 billion Northgate "
        "fabrication plant agreement on a recent date.",
    )
    right = _occurrence(
        2,
        "beta.example",
        "Regulators opened a review of Larkspur Robotics's $5.9 billion Sandhill test "
        "facility agreement on a recent date.",
    )
    assert find_conflicts(group_claims([left, right])) == []


def test_persisted_claims_are_idempotent(populated_session: Session) -> None:
    session = populated_session
    story_id = session.query(Claim.story_cluster_id).first()[0]
    bundle = load_story(session, story_id)
    assert bundle is not None
    before_claims = session.query(Claim).count()
    before_evidence = session.query(ClaimEvidence).count()
    persist_claims(session, story_id, bundle.articles)
    session.commit()
    assert session.query(Claim).count() == before_claims
    assert session.query(ClaimEvidence).count() == before_evidence


def test_loaded_claims_carry_evidence(populated_session: Session) -> None:
    story_id = populated_session.query(Claim.story_cluster_id).first()[0]
    claims = load_claims(populated_session, story_id)
    assert claims
    assert all(evidence for _, evidence in claims)


def test_summary_statements_must_cite_evidence() -> None:
    with pytest.raises(ValidationError, match="at least one evidence id"):
        SummaryStatement(section=SummarySection.NEW_DEVELOPMENTS, text="x", evidence_ids=[])


def test_every_summary_statement_resolves_to_evidence(populated_session: Session) -> None:
    session = populated_session
    story_id = session.query(Claim.story_cluster_id).first()[0]
    summary = ExtractiveSummarizer(session).summarize(story_id)
    assert summary.statements
    assert summary.validate_citations() == []
    assert summary.citation_coverage() == 1.0
    known = set(summary.evidence_index())
    for statement in summary.statements:
        assert statement.evidence_ids
        assert set(statement.evidence_ids) <= known


def test_summary_evidence_excerpt_supports_its_statement(populated_session: Session) -> None:
    session = populated_session
    story_id = session.query(Claim.story_cluster_id).first()[0]
    summary = ExtractiveSummarizer(session).summarize(story_id)
    index = summary.evidence_index()
    for statement in summary.statements:
        excerpts = [index[eid].excerpt for eid in statement.evidence_ids]
        assert statement_supported(statement.text, excerpts) or statement.is_disputed


def test_summary_metrics_reported(populated_session: Session) -> None:
    session = populated_session
    story_id = session.query(Claim.story_cluster_id).first()[0]
    metrics = evaluate_summary(ExtractiveSummarizer(session).summarize(story_id))
    assert metrics.citation_coverage == 1.0
    assert metrics.redundancy_rate == 0.0
    assert metrics.statements > 0


def test_summarizer_rejects_unknown_story(populated_session: Session) -> None:
    with pytest.raises(LookupError, match="not found"):
        ExtractiveSummarizer(populated_session).summarize(999_999)


def test_conflict_pair_reports_union_of_domains() -> None:
    left = _occurrence(1, "alpha.example", "Northwind Labs said Atlas 3 scores 71.4 percent today.")
    right = _occurrence(2, "beta.example", "Northwind Labs said Atlas 3 scores 64.0 percent today.")
    groups = group_claims([left, right])
    pair = ConflictPair(groups[0], groups[-1], "test")
    assert "alpha.example" in pair.independent_domains()
