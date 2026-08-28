"""Retry, backoff, caching and rate-limiting behaviour of the HTTP fetcher."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from newstrace.config import SourcePolicy, get_settings
from newstrace.ingestion.http import HttpFetcher, RateLimiter, ResponseCache

URL = "https://api.example/doc"


def _fetcher(tmp_path: Path, **kwargs: object) -> HttpFetcher:
    sleeps: list[float] = []
    fetcher = HttpFetcher(
        get_settings(),
        cache=ResponseCache(tmp_path / "cache"),
        sleep=lambda s: sleeps.append(s),
        **kwargs,  # type: ignore[arg-type]
    )
    fetcher.recorded_sleeps = sleeps  # type: ignore[attr-defined]
    return fetcher


@respx.mock
def test_successful_request_is_cached(tmp_path: Path) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, json={"articles": []}))
    fetcher = _fetcher(tmp_path)
    first = fetcher.get_json(URL, params={"q": "x"})
    second = fetcher.get_json(URL, params={"q": "x"})
    assert first == second == {"articles": []}
    assert route.call_count == 1, "second call must be served from the cache"


@respx.mock
def test_refresh_bypasses_the_cache(tmp_path: Path) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, text="{}"))
    fetcher = _fetcher(tmp_path)
    fetcher.get_json(URL)
    fetcher.get_json(URL, refresh=True)
    assert route.call_count == 2


@respx.mock
def test_cache_key_separates_different_params(tmp_path: Path) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, text="{}"))
    fetcher = _fetcher(tmp_path)
    fetcher.get_json(URL, params={"q": "a"})
    fetcher.get_json(URL, params={"q": "b"})
    assert route.call_count == 2


@respx.mock
@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retryable_statuses_are_retried_then_succeed(tmp_path: Path, status: int) -> None:
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(status), httpx.Response(200, text='{"ok": true}')]
    )
    fetcher = _fetcher(tmp_path)
    assert fetcher.get_json(URL) == {"ok": True}
    assert route.call_count == 2
    assert fetcher.recorded_sleeps  # backoff happened


@respx.mock
def test_transport_errors_are_retried(tmp_path: Path) -> None:
    route = respx.get(URL).mock(
        side_effect=[httpx.ConnectError("boom"), httpx.Response(200, text="{}")]
    )
    fetcher = _fetcher(tmp_path)
    assert fetcher.get_json(URL) == {}
    assert route.call_count == 2


@respx.mock
def test_retries_are_exhausted_and_raise(tmp_path: Path) -> None:
    respx.get(URL).mock(return_value=httpx.Response(503))
    fetcher = _fetcher(tmp_path)
    with pytest.raises(httpx.HTTPStatusError):
        fetcher.get_json(URL)


@respx.mock
def test_client_errors_are_not_retried(tmp_path: Path) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(404))
    fetcher = _fetcher(tmp_path)
    with pytest.raises(httpx.HTTPStatusError):
        fetcher.get_json(URL)
    assert route.call_count == 1


@respx.mock
def test_backoff_grows_between_attempts(tmp_path: Path) -> None:
    respx.get(URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(503), httpx.Response(200, text="{}")]
    )
    fetcher = _fetcher(tmp_path)
    fetcher.get_json(URL)
    backoffs = [s for s in fetcher.recorded_sleeps if s > 0]
    assert len(backoffs) >= 2
    assert backoffs[1] > backoffs[0]


@respx.mock
def test_malformed_json_raises_a_clear_error(tmp_path: Path) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, text="not json"))
    fetcher = _fetcher(tmp_path)
    with pytest.raises(ValueError, match="Malformed JSON"):
        fetcher.get_json(URL)


@respx.mock
def test_params_are_encoded_by_the_client(tmp_path: Path) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, text="{}"))
    fetcher = _fetcher(tmp_path)
    fetcher.get_json(URL, params={"query": '("a b" OR c) & d'})
    request = route.calls[0].request
    assert "%22a+b%22" in str(request.url) or "%22a%20b%22" in str(request.url)
    assert request.url.params["query"] == '("a b" OR c) & d'


@respx.mock
def test_user_agent_is_sent(tmp_path: Path) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, text="{}"))
    fetcher = _fetcher(tmp_path)
    fetcher.get_json(URL)
    assert route.calls[0].request.headers["user-agent"].startswith("NewsTrace")


def test_rate_limiter_waits_between_calls_to_one_host() -> None:
    slept: list[float] = []
    limiter = RateLimiter(1.0)
    limiter.wait("example.com", sleep=slept.append)
    limiter.wait("example.com", sleep=slept.append)
    assert any(s > 0 for s in slept)


def test_rate_limiter_applies_a_per_host_override() -> None:
    limiter = RateLimiter(1.0, {"API.GDELTProject.org": 5.5})
    assert limiter.interval_for("api.gdeltproject.org") == 5.5
    assert limiter.interval_for("example.com") == 1.0

    slept: list[float] = []
    limiter.wait("api.gdeltproject.org", sleep=slept.append)
    limiter.wait("api.gdeltproject.org", sleep=slept.append)
    assert slept and slept[-1] > 5.0, "the publisher-stated interval must be honoured"


def test_per_host_limits_come_from_the_source_policy(tmp_path: Path) -> None:
    policy = SourcePolicy.model_validate(
        {
            "rate_limits": {
                "default_min_interval_seconds": 2.0,
                "per_domain": {"api.gdeltproject.org": 5.5},
            }
        }
    )
    fetcher = HttpFetcher(
        get_settings(),
        cache=ResponseCache(tmp_path / "cache"),
        sleep=lambda _s: None,
        policy=policy,
    )
    assert fetcher.limiter.interval_for("api.gdeltproject.org") == 5.5
    assert fetcher.limiter.interval_for("feeds.example.com") == 2.0


def test_rate_limiter_disabled_at_zero_interval() -> None:
    slept: list[float] = []
    limiter = RateLimiter(0.0)
    limiter.wait("example.com", sleep=slept.append)
    limiter.wait("example.com", sleep=slept.append)
    assert slept == []


def test_unreadable_cache_entry_is_discarded(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    key = ResponseCache.key_for(URL, None)
    cache.write(key, {"text": "ok"})
    cache.path_for(key).write_text("{ broken", encoding="utf-8")
    assert cache.read(key) is None
