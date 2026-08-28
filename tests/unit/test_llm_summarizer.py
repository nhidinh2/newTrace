"""The optional LLM summarizer: structured output, citation validation, fallback."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from newstrace.models import StoryCluster
from newstrace.summarization.llm import (
    LLMClient,
    LLMSummarizer,
    LLMSummaryPayload,
    _extract_json,
)


class _StubClient(LLMClient):
    def __init__(self, response: str, *, provider: str = "anthropic", available: bool = True):
        self._response = response
        self._provider = provider
        self._available = available
        self.calls: list[tuple[str, str]] = []

    @property
    def provider(self) -> str:
        return self._provider

    def available(self) -> bool:
        return self._available

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._response


def _story_id(session: Session) -> int:
    return int(session.query(StoryCluster.id).first()[0])


def test_no_provider_returns_the_extractive_summary(populated_session: Session) -> None:
    summarizer = LLMSummarizer(populated_session, client=_StubClient("", available=False))
    summary = summarizer.summarize(_story_id(populated_session))
    assert summary.method == "extractive"
    assert any("unavailable" in note for note in summary.notes)


def test_valid_llm_output_is_accepted(populated_session: Session) -> None:
    story_id = _story_id(populated_session)
    from newstrace.summarization.extractive import ExtractiveSummarizer

    base = ExtractiveSummarizer(populated_session).summarize(story_id)
    evidence_id = base.evidence[0].evidence_id
    payload = json.dumps(
        {
            "statements": [
                {
                    "section": "repeated_reporting",
                    "text": "Two outlets reported the same benchmark figure.",
                    "evidence_ids": [evidence_id],
                    "is_single_source": False,
                    "is_disputed": False,
                }
            ]
        }
    )
    summary = LLMSummarizer(populated_session, client=_StubClient(payload)).summarize(story_id)
    assert summary.method.startswith("llm:anthropic")
    assert len(summary.statements) == 1
    assert summary.validate_citations() == []


def test_statements_with_unknown_evidence_ids_are_dropped(populated_session: Session) -> None:
    payload = json.dumps(
        {
            "statements": [
                {
                    "section": "new_developments",
                    "text": "Invented statement.",
                    "evidence_ids": ["ev-does-not-exist"],
                }
            ]
        }
    )
    summary = LLMSummarizer(populated_session, client=_StubClient(payload)).summarize(
        _story_id(populated_session)
    )
    assert summary.method == "extractive"
    assert any("lacked valid evidence" in note for note in summary.notes)


def test_uncited_statement_fails_schema_validation() -> None:
    with pytest.raises(ValidationError):
        LLMSummaryPayload.model_validate(
            {"statements": [{"section": "new_developments", "text": "x", "evidence_ids": []}]}
        )


def test_malformed_json_falls_back(populated_session: Session) -> None:
    summary = LLMSummarizer(
        populated_session, client=_StubClient("I am not JSON at all")
    ).summarize(_story_id(populated_session))
    assert summary.method == "extractive"
    assert any("rejected" in note for note in summary.notes)


def test_api_failure_falls_back(populated_session: Session) -> None:
    class _Boom(_StubClient):
        def complete(self, system: str, user: str) -> str:
            raise RuntimeError("provider exploded")

    summary = LLMSummarizer(populated_session, client=_Boom("")).summarize(
        _story_id(populated_session)
    )
    assert summary.method == "extractive"
    assert any("failed" in note or "rejected" in note for note in summary.notes)


def test_prompt_contains_only_evidence_and_metadata(populated_session: Session) -> None:
    story_id = _story_id(populated_session)
    client = _StubClient(json.dumps({"statements": []}))
    LLMSummarizer(populated_session, client=client).summarize(story_id)
    _, user = client.calls[0]
    payload = json.loads(user)
    assert set(payload) == {
        "story_title",
        "independent_domain_count",
        "duplicate_article_count",
        "sections",
        "evidence",
        "schema",
    }
    for item in payload["evidence"]:
        assert set(item) == {
            "evidence_id",
            "title",
            "source_domain",
            "published_at",
            "excerpt",
            "is_near_duplicate",
        }


def test_extract_json_handles_fenced_output() -> None:
    assert json.loads(_extract_json('```json\n{"a": 1}\n```')) == {"a": 1}
    assert json.loads(_extract_json('Sure! {"a": 2} hope that helps')) == {"a": 2}
    with pytest.raises(ValueError, match="no JSON object"):
        _extract_json("nothing here")
