"""HTTP client with retries, jittered backoff, rate limiting and response cache.

The cache exists for reproducibility: a run can be replayed from disk without
hitting the publisher or GDELT again.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import httpx

from newstrace.config import Settings, SourcePolicy, get_settings, load_source_policy
from newstrace.logging import get_logger
from newstrace.utils import sha256_obj

logger = get_logger(__name__)

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class RateLimiter:
    """Per-host minimum-interval limiter.

    ``per_host`` overrides the default interval for hosts that publish a
    stricter policy.  GDELT, for example, answers 429 with "limit requests to
    one every 5 seconds", so its interval must come from configuration rather
    than from the global default.
    """

    def __init__(self, min_interval: float = 1.0, per_host: dict[str, float] | None = None) -> None:
        self.min_interval = max(0.0, min_interval)
        self.per_host = {
            str(host).lower(): max(0.0, float(interval))
            for host, interval in (per_host or {}).items()
        }
        self._last: dict[str, float] = {}

    def interval_for(self, host: str) -> float:
        """Minimum seconds between requests to ``host``."""
        return self.per_host.get(host.lower(), self.min_interval)

    def wait(self, host: str, *, sleep: Any = time.sleep) -> None:
        interval = self.interval_for(host)
        if interval <= 0:
            return
        now = time.monotonic()
        previous = self._last.get(host)
        if previous is not None:
            delta = interval - (now - previous)
            if delta > 0:
                sleep(delta)
        self._last[host] = time.monotonic()


class ResponseCache:
    """Content-addressed cache of raw responses keyed by URL + params."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @staticmethod
    def key_for(url: str, params: dict[str, Any] | None) -> str:
        return sha256_obj({"url": url, "params": params or {}})

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def read(self, key: str) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Discarding unreadable cache entry %s", path)
            return None

    def write(self, key: str, payload: dict[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path


class HttpFetcher:
    """Thin wrapper over ``httpx.Client`` implementing the project's HTTP rules."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
        cache: ResponseCache | None = None,
        use_cache: bool = True,
        sleep: Any = time.sleep,
        rng: random.Random | None = None,
        policy: SourcePolicy | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._owns_client = client is None
        self.client = client or httpx.Client(
            timeout=self.settings.http_timeout_seconds,
            headers={"User-Agent": self.settings.http_user_agent},
            follow_redirects=True,
        )
        self.cache = cache or ResponseCache(self.settings.resolve(self.settings.cache_dir))
        self.use_cache = use_cache
        self.policy = policy if policy is not None else load_source_policy()
        limits = self.policy.rate_limits or {}
        self.limiter = RateLimiter(
            float(
                limits.get("default_min_interval_seconds", self.settings.http_min_interval_seconds)
            ),
            {
                str(host): float(interval)
                for host, interval in (limits.get("per_domain") or {}).items()
            },
        )
        self._sleep = sleep
        self._rng = rng or random.Random(self.settings.random_seed)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> HttpFetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _backoff(self, attempt: int) -> float:
        base = self.settings.http_backoff_base_seconds * (2**attempt)
        return base + self._rng.uniform(0, base * 0.5)

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        refresh: bool = False,
    ) -> str:
        """GET a URL and return the body text, using cache + retry policy."""
        key = ResponseCache.key_for(url, params)
        if self.use_cache and not refresh:
            cached = self.cache.read(key)
            if cached is not None:
                logger.debug("Cache hit for %s", url)
                return str(cached.get("text", ""))

        host = httpx.URL(url).host or "unknown"
        last_error: Exception | None = None
        for attempt in range(self.settings.http_max_retries):
            self.limiter.wait(host, sleep=self._sleep)
            try:
                # params are encoded by httpx -- never string-concatenated.
                response = self.client.get(url, params=params)
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning("HTTP error for %s (attempt %s): %s", url, attempt + 1, exc)
            else:
                if response.status_code in RETRY_STATUS:
                    last_error = httpx.HTTPStatusError(
                        f"retryable status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    logger.warning(
                        "Retryable status %s for %s (attempt %s)",
                        response.status_code,
                        url,
                        attempt + 1,
                    )
                elif response.status_code >= 400:
                    response.raise_for_status()
                else:
                    text = response.text
                    if self.use_cache:
                        self.cache.write(
                            key,
                            {
                                "url": str(response.url),
                                "params": params or {},
                                "status_code": response.status_code,
                                "text": text,
                            },
                        )
                    return text
            if attempt < self.settings.http_max_retries - 1:
                self._sleep(self._backoff(attempt))
        assert last_error is not None
        raise last_error

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        refresh: bool = False,
    ) -> Any:
        text = self.get_text(url, params=params, refresh=refresh)
        if not text.strip():
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON payload from {url}: {exc}") from exc
