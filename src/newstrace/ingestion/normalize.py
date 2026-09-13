"""URL, text and timestamp normalisation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import tldextract

from newstrace.ingestion.base import RawArticle
from newstrace.utils import (
    normalize_for_hash,
    normalize_unicode,
    sha256_text,
    to_utc,
    truncate,
)

TRACKING_PREFIXES = ("utm_", "pk_", "mc_", "hsa_", "at_")
TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "dclid",
    "gbraid",
    "wbraid",
    "msclkid",
    "igshid",
    "mkt_tok",
    "ref",
    "ref_src",
    "referrer",
    "source",
    "cmpid",
    "campaign_id",
    "spm",
    "yclid",
    "_hsenc",
    "_hsmi",
    "icid",
    "ncid",
    "sh",
    "smid",
}

# ``include_psl_private_domains`` keeps the tenant label on hosts the Public
# Suffix List marks as private (``someone.github.io``, ``blog.blogspot.com``).
_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)

# Publishing platforms the PSL does not list as private suffixes. Each tenant is
# a separate publisher, so collapsing them to the platform would let two
# unrelated newsletters count as one source when stories report corroboration.
MULTI_TENANT_HOSTS = frozenset(
    {
        "beehiiv.com",
        "ghost.io",
        "medium.com",
        "substack.com",
        "svbtle.com",
        "tumblr.com",
        "typepad.com",
        "wordpress.com",
    }
)


def is_tracking_param(name: str) -> bool:
    lowered = name.lower()
    return lowered in TRACKING_PARAMS or lowered.startswith(TRACKING_PREFIXES)


def canonicalize_url(url: str) -> str:
    """Return a stable canonical form of a URL.

    Lowercases scheme/host, strips ``www.``, default ports, tracking parameters,
    fragments and trailing slashes, and sorts the remaining query parameters.
    """
    if not url:
        return ""
    raw = normalize_unicode(url)
    if "://" not in raw:
        raw = "https://" + raw.lstrip("/")
    parts = urlsplit(raw)
    scheme = (parts.scheme or "https").lower()
    if scheme == "http":
        scheme = "https"
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    netloc = host
    if parts.port and parts.port not in (80, 443):
        netloc = f"{host}:{parts.port}"
    query_pairs = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if not is_tracking_param(k)
    ]
    query = urlencode(sorted(query_pairs))
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    if not path:
        path = "/"
    return urlunsplit((scheme, netloc, path, query, ""))


def extract_domain(url: str) -> str:
    """Publisher domain (``sub.example.co.uk`` -> ``example.co.uk``).

    Registrable domain, except on a multi-tenant publishing platform, where the
    tenant label is kept (``terrytao.wordpress.com``) because the tenant, not
    the platform, is the publisher.
    """
    if not url:
        return ""
    host = urlsplit(url if "://" in url else f"https://{url}").hostname or ""
    result = _EXTRACTOR(host)
    if result.domain and result.suffix:
        registrable = f"{result.domain}.{result.suffix}".lower()
        if registrable in MULTI_TENANT_HOSTS:
            # The tenant is the label directly left of the platform domain, so
            # ``feeds.terrytao.wordpress.com`` is still Tao's blog.
            tenant = result.subdomain.lower().rsplit(".", 1)[-1]
            if tenant and tenant != "www":
                return f"{tenant}.{registrable}"
        return registrable
    return host.lower().removeprefix("www.")


def normalize_language(language: str | None) -> str:
    if not language:
        return "unknown"
    lowered = normalize_unicode(language).lower().replace("_", "-")
    return lowered.split("-")[0] or "unknown"


@dataclass(frozen=True, slots=True)
class NormalizedArticle:
    """Normalised article payload ready for persistence."""

    url: str
    canonical_url: str
    title: str
    description: str
    excerpt: str
    source_domain: str
    language: str
    published_at: datetime | None
    raw_payload_hash: str
    normalized_text_hash: str
    normalized_title_hash: str
    normalized_title: str
    extraction_method: str
    source_adapter: str
    topic: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_article(raw: RawArticle, *, max_excerpt_chars: int = 1200) -> NormalizedArticle:
    """Normalise one raw article; raises ``ValueError`` when unusable."""
    canonical = canonicalize_url(raw.url)
    if not canonical:
        raise ValueError("article has no usable URL")

    title = normalize_unicode(raw.title)
    description = truncate(normalize_unicode(raw.description), max_excerpt_chars)
    excerpt = truncate(normalize_unicode(raw.excerpt or raw.description), max_excerpt_chars)
    domain = raw.source_domain or extract_domain(canonical)
    published: datetime | None = to_utc(raw.published_at)
    normalized_title = normalize_for_hash(title)
    normalized_text = normalize_for_hash(f"{title} {excerpt}")

    return NormalizedArticle(
        url=normalize_unicode(raw.url),
        canonical_url=canonical,
        title=title,
        description=description,
        excerpt=excerpt,
        source_domain=extract_domain(domain) or domain.lower(),
        language=normalize_language(raw.language),
        topic=raw.topic,
        published_at=published,
        raw_payload_hash=sha256_text(repr(sorted(raw.raw.items()))) if raw.raw else "",
        normalized_text_hash=sha256_text(normalized_text),
        normalized_title_hash=sha256_text(normalized_title),
        normalized_title=normalized_title,
        extraction_method=raw.extraction_method,
        source_adapter=raw.source_adapter,
        extra={"raw_keys": sorted(raw.raw.keys())} if raw.raw else {},
    )
