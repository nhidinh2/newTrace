"""Application configuration.

All settings are environment driven (prefix ``NEWSTRACE_``) with an optional
``.env`` file.  Nothing here may require a paid API key: the defaults must give
a fully working offline system.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

EmbeddingBackend = Literal["auto", "sentence-transformers", "hashing"]
LLMProvider = Literal["none", "anthropic", "openai"]


class Settings(BaseSettings):
    """Runtime configuration for every NewsTrace entry point."""

    model_config = SettingsConfigDict(
        env_prefix="NEWSTRACE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "development"
    database_url: str = "sqlite:///./newstrace.db"
    log_level: str = "INFO"
    random_seed: int = 549

    topics_file: Path = Path("configs/topics.example.yaml")
    sources_file: Path = Path("configs/sources.example.yaml")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_backend: EmbeddingBackend = "auto"
    embedding_dimension_fallback: int = 384
    preprocessing_version: str = "v1"

    cluster_window_hours: int = 72
    # Selected on the fit+validation window of the committed fixtures with the
    # sentence-transformer embedder (see docs/evaluation.md). The hashing
    # fallback produces a different cosine scale, so it carries its own value.
    cluster_threshold: float = 0.88
    cluster_threshold_hashing: float = 0.74
    cluster_title_overlap_weight: float = 0.2
    cluster_entity_overlap_weight: float = 0.1

    near_duplicate_title_threshold: float = 0.9
    near_duplicate_window_hours: int = 96
    # Candidates are now retrieved by SimHash band, so the cap is a guard
    # against a pathological band collision rather than the sampling step it
    # used to be. See newstrace.ingestion.deduplicate.DuplicateDetector.
    near_duplicate_max_candidates: int = 400

    # How many (method, dimension, fit_version) matrices the process keeps
    # resident. Each costs ~4 bytes x articles x dimension: 9 MiB for 6k
    # articles at 384-d, and a fortieth of that for a 32-d projection.
    retrieval_cache_entries: int = 4
    sqlite_busy_timeout_ms: int = 5000

    http_user_agent: str = "NewsTraceResearch/0.1"
    http_timeout_seconds: float = 20.0
    http_max_retries: int = 4
    http_backoff_base_seconds: float = 0.5
    http_min_interval_seconds: float = 1.0

    cache_dir: Path = Path("data/raw/cache")
    artifacts_dir: Path = Path("artifacts")
    fixtures_dir: Path = Path("data/fixtures")

    def resolve(self, path: Path) -> Path:
        """Resolve a possibly-relative configured path against the repo root."""
        p = Path(path)
        return p if p.is_absolute() else (REPO_ROOT / p)


class LLMSettings(BaseSettings):
    """LLM configuration.

    Separate from :class:`Settings` because these variables are *not* prefixed
    and because the default (``none``) must never touch the network.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: LLMProvider = "none"
    llm_model: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""


class TopicConfig(BaseModel):
    """One followed topic."""

    key: str
    display_name: str = ""
    gdelt_query: str = ""
    languages: list[str] = Field(default_factory=lambda: ["english"])
    feeds: list[str] = Field(default_factory=list)

    def label(self) -> str:
        return self.display_name or self.key.replace("_", " ").title()


class ExtractionPolicy(BaseModel):
    enabled: bool = False
    allowlist: list[str] = Field(default_factory=list)
    max_excerpt_chars: int = 1200

    def allows(self, domain: str) -> bool:
        return self.enabled and domain.lower() in {d.lower() for d in self.allowlist}


class SourcePolicy(BaseModel):
    extraction: ExtractionPolicy = Field(default_factory=ExtractionPolicy)
    rate_limits: dict[str, Any] = Field(default_factory=dict)

    def min_interval_for(self, domain: str, default: float) -> float:
        limits = self.rate_limits or {}
        per_domain = limits.get("per_domain") or {}
        if domain in per_domain:
            return float(per_domain[domain])
        return float(limits.get("default_min_interval_seconds", default))


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@functools.lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    return LLMSettings()


def load_topics(path: Path | None = None) -> dict[str, TopicConfig]:
    """Load the topic registry from YAML, returning an ordered mapping."""
    settings = get_settings()
    target = settings.resolve(path or settings.topics_file)
    if not target.exists():
        return {}
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    topics: dict[str, TopicConfig] = {}
    for key, value in (raw.get("topics") or {}).items():
        payload = dict(value or {})
        payload["key"] = key
        topics[key] = TopicConfig.model_validate(payload)
    return topics


def load_source_policy(path: Path | None = None) -> SourcePolicy:
    settings = get_settings()
    target = settings.resolve(path or settings.sources_file)
    if not target.exists():
        return SourcePolicy()
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return SourcePolicy.model_validate(raw)


def reset_caches() -> None:
    """Clear cached settings (used by tests that patch the environment)."""
    get_settings.cache_clear()
    get_llm_settings.cache_clear()
