"""Claim reads retain complete evidence without per-claim database queries."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.orm import Session

from newstrace.api.app import create_app
from newstrace.api.serializers import claim_to_schema
from newstrace.claims.evidence import load_claims
from newstrace.db import get_engine
from newstrace.models import Article, Claim, ClaimEvidence, StoryCluster


@contextmanager
def count_selects() -> Iterator[list[str]]:
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    engine = get_engine()
    event.listen(engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", record)


def seed_claims(session: Session, count: int) -> tuple[int, list[int], dict[int, list[int]]]:
    story = StoryCluster(display_title="Synthetic claim loading regression")
    other = StoryCluster(display_title="Another story")
    articles = [
        Article(
            url=f"https://{domain}/{i}",
            canonical_url=f"https://{domain}/{i}",
            source_domain=domain,
            title=f"Report {i}",
            normalized_text_hash=str(i),
            normalized_title_hash=str(i),
            is_near_duplicate=i == 2,
        )
        for i, domain in enumerate(["alpha.example", "beta.example", "wire.example"])
    ]
    session.add_all([story, other, *articles])
    session.flush()
    claims = [
        Claim(
            story_cluster_id=story.id,
            claim_hash=str(i),
            normalized_claim=f"Synthetic claim {i}",
            confidence=(i % 3) / 3,
        )
        for i in range(count)
    ]
    foreign = Claim(story_cluster_id=other.id, claim_hash="foreign", normalized_claim="Other")
    session.add_all([*claims, foreign])
    session.flush()
    expected_evidence: dict[int, list[int]] = {c.id: [] for c in claims}
    # Insert across claims in reverse order: evidence IDs must not reorder claims.
    for i, claim in reversed(list(enumerate([*claims, foreign]))):
        if i == 0:
            continue  # Retain claims that have no evidence.
        sources = [articles[0], articles[0]] if i == 1 else articles
        if i == 2:
            sources = [articles[0], articles[2]]
        for j, article in enumerate(sources):
            ev = ClaimEvidence(
                claim_id=claim.id,
                article_id=article.id,
                excerpt=f"Evidence {i}/{j}",
                excerpt_start=j * 10,
                stance="disputes" if i == 4 and j == 1 else "reports",
                confidence=0.75,
            )
            session.add(ev)
            session.flush()
            if claim.id in expected_evidence:
                expected_evidence[claim.id].append(ev.id)
    ids = [c.id for c in sorted(claims, key=lambda c: (-c.confidence, c.id))]
    story_id = story.id
    session.commit()
    return story_id, ids, expected_evidence


@pytest.mark.parametrize("count", [0, 1, 40, 1201])
def test_claim_loading_has_bounded_queries_and_preserves_rows(session: Session, count: int) -> None:
    story_id, ids, expected_evidence = seed_claims(session, count)
    # A fresh session prevents the identity map from hiding lazy-load queries.
    with Session(get_engine()) as reader, count_selects() as statements:
        loaded = load_claims(reader, story_id)
        schemas = [claim_to_schema(claim, evidence) for claim, evidence in loaded]
        assert [c.id for c, _ in loaded] == ids
        assert {c.id: [e.id for e in evidence] for c, evidence in loaded} == expected_evidence
        assert [s.id for s in schemas] == ids
        assert len(statements) <= (2 if count else 1), len(statements)


@pytest.mark.parametrize("count", [0, 1, 40])
def test_claims_endpoint_has_bounded_queries_and_preserves_evidence(
    session: Session, count: int
) -> None:
    story_id, ids, expected_evidence = seed_claims(session, count)
    with TestClient(create_app()) as client:
        with count_selects() as statements:
            response = client.get(f"/stories/{story_id}/claims")
        assert response.status_code == 200
        body = response.json()
        assert [c["id"] for c in body["claims"]] == ids
        assert len(statements) <= (3 if count else 2), len(statements)
        by_text = {c["normalized_claim"]: c for c in body["claims"]}
        assert sum(len(c["evidence"]) for c in body["claims"]) == sum(
            len(e) for e in expected_evidence.values()
        )
        if count:
            assert by_text["Synthetic claim 0"]["evidence"] == []
        if count > 4:
            for index, status, independent, duplicates in [
                (1, "single_source", 1, 0),
                (2, "single_source", 1, 1),
                (3, "repeated", 2, 1),
                (4, "unclear", 2, 1),
            ]:
                claim = by_text[f"Synthetic claim {index}"]
                assert claim["status"] == status
                assert claim["independent_domain_count"] == independent
                assert claim["duplicate_evidence_count"] == duplicates
                assert [e["excerpt"] for e in claim["evidence"]] == [
                    f"Evidence {index}/{j}" for j in range(len(claim["evidence"]))
                ]
                assert all(e["url"].startswith("https://") for e in claim["evidence"])
                assert all(e["confidence"] == 0.75 for e in claim["evidence"])
        with count_selects() as statements:
            assert client.get("/stories/999999/claims").status_code == 404
        assert len(statements) == 1
