"""Optional LLM summarizer.

Contract:

* the model sees **only** retrieved evidence and metadata;
* it must return JSON that validates against :class:`LLMSummaryPayload`;
* every statement must cite evidence ids that exist;
* statements with unknown or missing evidence ids are dropped;
* any API, parsing or validation failure falls back to the extractive summarizer.

The default provider is ``none``: NewsTrace never requires an API key.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from newstrace.config import LLMSettings, get_llm_settings
from newstrace.logging import get_logger
from newstrace.stories import StoryArticles, load_story
from newstrace.summarization.base import StorySummary, SummarySection, SummaryStatement
from newstrace.summarization.extractive import ExtractiveSummarizer, drop_uncited

logger = get_logger(__name__)

PROMPT_VERSION = "v1"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You summarise news coverage for an evidence-tracking system.

Rules:
1. Use ONLY the evidence provided. Never add outside knowledge.
2. Every statement must cite one or more evidence_id values from the evidence list.
3. Do not state that a claim is true or false, or that a publisher is reliable.
4. Mark a statement single_source when only one non-duplicate domain supports it.
5. Mark a statement disputed when the evidence cannot be reconciled.
6. Return ONLY JSON matching the requested schema. No prose, no markdown.
"""


class LLMStatement(BaseModel):
    section: SummarySection
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    is_single_source: bool = False
    is_disputed: bool = False


class LLMSummaryPayload(BaseModel):
    statements: list[LLMStatement] = Field(default_factory=list)


def build_prompt(bundle: StoryArticles, evidence: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "story_title": bundle.story.display_title,
            "independent_domain_count": len(bundle.independent_domains),
            "duplicate_article_count": len(bundle.duplicates),
            "sections": [s.value for s in SummarySection],
            "evidence": evidence,
            "schema": LLMSummaryPayload.model_json_schema(),
        },
        indent=2,
    )


class LLMClient:
    """Thin provider adapter. Returns raw text; parsing happens upstream."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or get_llm_settings()

    @property
    def provider(self) -> str:
        return self.settings.llm_provider

    def available(self) -> bool:
        if self.provider == "anthropic":
            return bool(self.settings.anthropic_api_key)
        if self.provider == "openai":
            return bool(self.settings.openai_api_key)
        return False

    def complete(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            return self._complete_anthropic(system, user)
        if self.provider == "openai":
            return self._complete_openai(system, user)
        raise RuntimeError(f"LLM provider {self.provider!r} is not configured")

    def _complete_anthropic(self, system: str, user: str) -> str:
        import anthropic
        from anthropic.types import TextBlock

        client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        message = client.messages.create(
            model=self.settings.llm_model or DEFAULT_ANTHROPIC_MODEL,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # A response can carry non-text blocks (thinking, tool use); only text
        # blocks contribute to the JSON payload we are about to validate.
        return "".join(block.text for block in message.content if isinstance(block, TextBlock))

    def _complete_openai(self, system: str, user: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.settings.llm_model or DEFAULT_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""


class LLMSummarizer:
    """LLM-assisted summarizer that always degrades to the extractive summary."""

    name = "llm"

    def __init__(
        self,
        session: Session,
        *,
        client: LLMClient | None = None,
        max_per_section: int = 5,
    ) -> None:
        self.session = session
        self.client = client or LLMClient()
        self.fallback = ExtractiveSummarizer(session, max_per_section=max_per_section)

    def summarize(self, story_id: int) -> StorySummary:
        bundle = load_story(self.session, story_id)
        if bundle is None:
            raise LookupError(f"story {story_id} not found")

        base = self.fallback.summarize_bundle(bundle)
        if not self.client.available():
            base.notes.append(
                f"LLM provider '{self.client.provider}' is unavailable; "
                "returned the deterministic extractive summary."
            )
            return base

        evidence_payload = [
            {
                "evidence_id": e.evidence_id,
                "title": e.title,
                "source_domain": e.source_domain,
                "published_at": e.published_at.isoformat() if e.published_at else None,
                "excerpt": e.excerpt,
                "is_near_duplicate": e.is_near_duplicate,
            }
            for e in base.evidence
        ]
        try:
            raw = self.client.complete(SYSTEM_PROMPT, build_prompt(bundle, evidence_payload))
            payload = LLMSummaryPayload.model_validate_json(_extract_json(raw))
        except (ValidationError, ValueError, RuntimeError) as exc:
            logger.warning("LLM summary rejected (%s); falling back to extractive", exc)
            base.notes.append(f"LLM output rejected ({exc.__class__.__name__}); used extractive.")
            return base
        except Exception as exc:
            logger.warning("LLM call failed (%s); falling back to extractive", exc)
            base.notes.append(f"LLM call failed ({exc.__class__.__name__}); used extractive.")
            return base

        known = set(base.evidence_index())
        statements: list[SummaryStatement] = []
        dropped = 0
        for item in payload.statements:
            cited = [eid for eid in item.evidence_ids if eid in known]
            if not cited:
                dropped += 1
                continue
            domains = {
                base.evidence_index()[eid].source_domain
                for eid in cited
                if not base.evidence_index()[eid].is_near_duplicate
            }
            statements.append(
                SummaryStatement(
                    section=item.section,
                    text=item.text,
                    evidence_ids=cited,
                    distinct_domains=len(domains),
                    is_single_source=item.is_single_source or len(domains) <= 1,
                    is_disputed=item.is_disputed,
                    confidence=0.0,
                )
            )
        if not statements:
            base.notes.append("Every LLM statement lacked valid evidence ids; used extractive.")
            return base

        base.statements = statements
        base.method = f"llm:{self.client.provider}:{PROMPT_VERSION}"
        base.notes.append(
            f"LLM statements: {len(statements)} kept, {dropped} dropped for missing evidence."
        )
        return drop_uncited(base)


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found in the LLM response")
    return stripped[start : end + 1]
