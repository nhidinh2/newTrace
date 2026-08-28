"""Fixture adapter: replays committed JSON/XML payloads with no network access."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from newstrace.config import get_settings
from newstrace.ingestion.base import RawArticle, SourceAdapter
from newstrace.ingestion.gdelt import parse_articles as parse_gdelt
from newstrace.ingestion.rss import parse_feed
from newstrace.logging import get_logger

logger = get_logger(__name__)


class FixtureAdapter(SourceAdapter):
    """Load ``*.gdelt.json`` and ``*.rss.xml`` fixtures from a directory."""

    name = "fixtures"

    def __init__(self, directory: Path | None = None) -> None:
        settings = get_settings()
        self.directory = settings.resolve(directory or settings.fixtures_dir)

    def _topic_for(self, path: Path) -> str | None:
        stem = path.name
        for suffix in (".gdelt.json", ".rss.xml", ".json", ".xml"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)] or None
        return path.stem or None

    def fetch(self, *, topic: str | None = None, **_: Any) -> Iterable[RawArticle]:
        if not self.directory.exists():
            logger.warning("Fixture directory %s does not exist", self.directory)
            return []
        articles: list[RawArticle] = []
        for path in sorted(self.directory.glob("*.gdelt.json")):
            fixture_topic = topic or self._topic_for(path)
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
            parsed = parse_gdelt(payload, topic=fixture_topic)
            for article in parsed:
                article.source_adapter = "fixtures"
                article.raw.setdefault("fixture", path.name)
            articles.extend(parsed)
        for path in sorted(self.directory.glob("*.rss.xml")):
            fixture_topic = topic or self._topic_for(path)
            parsed_rss = parse_feed(
                path.read_text(encoding="utf-8"), topic=fixture_topic, feed_url=path.name
            )
            for article in parsed_rss:
                article.source_adapter = "fixtures"
                article.raw.setdefault("fixture", path.name)
            articles.extend(parsed_rss)
        logger.info("Loaded %s fixture articles from %s", len(articles), self.directory)
        return articles
