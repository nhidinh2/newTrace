"""Structured-ish logging setup shared by the CLI, API and scripts."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging(level: str | None = None, *, force: bool = False) -> None:
    """Configure root logging once per process."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    resolved = (level or os.environ.get("NEWSTRACE_LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(resolved)
    logging.getLogger("httpx").setLevel(max(logging.WARNING, root.level))
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
