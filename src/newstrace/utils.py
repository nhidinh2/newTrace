"""Small shared helpers: hashing, time handling, seeding, git metadata."""

from __future__ import annotations

import hashlib
import os
import random
import re
import subprocess
import unicodedata
from datetime import UTC, datetime
from typing import Any

import numpy as np
from dateutil import parser as date_parser

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    import json

    return sha256_text(json.dumps(obj, sort_keys=True, default=str))


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip()


def normalize_unicode(text: str) -> str:
    """NFKC-normalise, drop control characters and collapse whitespace."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    cleaned = "".join(ch for ch in normalized if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    return normalize_whitespace(cleaned)


def normalize_for_hash(text: str) -> str:
    """Aggressive normalisation used for duplicate detection only."""
    lowered = normalize_unicode(text).lower()
    return " ".join(_TOKEN_RE.findall(lowered))


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_unicode(text).lower())


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def to_utc(value: datetime | str | None) -> datetime | None:
    """Parse and normalise any timestamp to timezone-aware UTC."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            # GDELT uses compact YYYYMMDDTHHMMSSZ stamps.
            if re.fullmatch(r"\d{8}T\d{6}Z", text):
                parsed = datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
            elif re.fullmatch(r"\d{14}", text):
                parsed = datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
            else:
                parsed = date_parser.parse(text)
        except (ValueError, OverflowError, date_parser.ParserError):
            return None
    else:
        parsed = value
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def ensure_utc(value: datetime | None) -> datetime | None:
    """Attach UTC to naive datetimes read back from SQLite."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def utcnow() -> datetime:
    return datetime.now(UTC)


def isoformat(value: datetime | None) -> str | None:
    v = ensure_utc(value)
    return v.isoformat() if v else None


def set_global_seed(seed: int) -> None:
    """Seed every RNG we rely on so experiments are reproducible."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit or None


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    text = normalize_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix
