"""Score the story clusterer against a public, human-labelled news stream.

Clustering is otherwise measured on 135 fixture articles whose labels were
written for this repository, and the live corpus has no labels at all -- the
same "the system grades its own homework" problem the BEIR harness fixed for
retrieval. This module runs the *production* :class:`OnlineClusterer`, unchanged,
over the dataset released with

    Miranda, Znotins, Cohen, Barzdins. "Multilingual Clustering of Streaming
    News." EMNLP 2018. https://aclanthology.org/D18-1483

which was built from the Event Registry data of Rupnik et al. (2016). Each
record carries a title, body, timestamp, source and a gold story label, so the
stream can be replayed in publication order exactly as live ingestion sees it.

The paper tunes its threshold on the dev split and reports pairwise F1 on the
test split, per language; this harness does the same, so its numbers sit next
to the published ones in :data:`PUBLISHED_EN`. Nothing here touches the
NewsTrace database: each run clusters into a throwaway in-memory SQLite.

Files are read from ``<data_dir>/dataset.{dev,test}.json`` (a JSON list of
``{"id", "title", "text", "date", "source", "lang", "cluster", ...}``).
``download`` fetches them from Priberam's public FTP; it is opt-in and the only
networked step.
"""

from __future__ import annotations

import ftplib
import json
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from newstrace.clustering.online import OnlineClusterer, default_threshold
from newstrace.config import Settings, get_settings
from newstrace.evaluation.clustering import evaluate_clustering
from newstrace.logging import get_logger
from newstrace.models import Article, Base
from newstrace.representations.embedder import Embedder, get_embedder
from newstrace.utils import set_global_seed

logger = get_logger(__name__)

FTP_HOST = "ftp.priberam.pt"
# The server expects the proxy-style "host|user" login its own download script uses.
FTP_USER = "ftp.priberam.pt|anonymous"
FTP_DIR = "/SUMMAPublic/Corpora/Clustering/2018.0/dataset"
SPLITS = ("dev", "test")

# Table 2 of Miranda et al. (2018), English test split: pairwise F1/P/R in %.
PUBLISHED_EN: list[dict[str, Any]] = [
    {"system": "CluStream (Aggarwal & Yu 2006)", "f1": 79.0, "precision": 98.6, "recall": 65.9},
    {
        "system": "Miranda et al.: tokens+lemmas+entities",
        "f1": 92.7,
        "precision": 92.9,
        "recall": 92.5,
    },
    {"system": "Miranda et al.: + timestamps", "f1": 94.1, "precision": 98.2, "recall": 90.3},
]

DEFAULT_GRID = tuple(round(x, 3) for x in np.arange(0.50, 0.975, 0.025))


@dataclass
class ClusterBenchConfig:
    data_dir: str = "data/cluster_bench"
    language: str = "eng"
    thresholds: list[float] = field(default_factory=lambda: list(DEFAULT_GRID))
    tune: bool = True
    window_hours: int | None = None
    max_documents: int | None = None
    seed: int = 549

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchDocument:
    doc_id: str
    title: str
    text: str
    published_at: datetime
    source: str
    cluster: str


def data_path(config: ClusterBenchConfig, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    root = Path(config.data_dir)
    return root if root.is_absolute() else settings.resolve(root)


def download(
    config: ClusterBenchConfig, settings: Settings | None = None, *, attempts: int = 8
) -> Path:
    """Fetch the dev and test splits. Requires network access.

    The server drops long transfers partway through, so each file is resumed
    from its current size (FTP ``REST``) until it matches the remote size.
    """
    target = data_path(config, settings)
    target.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        name = f"dataset.{split}.json"
        path = target / name
        for attempt in range(1, attempts + 1):
            try:
                with ftplib.FTP(FTP_HOST, timeout=120) as ftp:
                    ftp.login(FTP_USER, "")
                    ftp.cwd(FTP_DIR)
                    ftp.voidcmd("TYPE I")
                    remote = ftp.size(name) or 0
                    have = path.stat().st_size if path.exists() else 0
                    if remote and have == remote:
                        break
                    logger.info(
                        "Fetching %s (%s/%s bytes, attempt %s)", name, have, remote, attempt
                    )
                    with path.open("ab") as handle:
                        ftp.retrbinary(f"RETR {name}", handle.write, rest=have or None)
                if path.stat().st_size == remote:
                    break
            except (ftplib.Error, OSError) as exc:
                logger.warning("%s: transfer interrupted (%s); resuming", name, exc)
        else:
            raise RuntimeError(f"could not download {name} after {attempts} attempts")
        logger.info("%s ready at %s", name, path)
    return target


def load_split(
    config: ClusterBenchConfig, split: str, settings: Settings | None = None
) -> list[BenchDocument]:
    """Read one split for one language, in stream (publication) order."""
    path = data_path(config, settings) / f"dataset.{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/run_cluster_bench.py --download` "
            "first (this step needs the network)."
        )
    records = json.loads(path.read_text(encoding="utf-8"))
    documents = [
        BenchDocument(
            doc_id=str(r["id"]),
            title=(r.get("title") or "").strip(),
            text=(r.get("text") or "").strip(),
            published_at=datetime.strptime(r["date"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC),
            source=(r.get("source") or "unknown").strip() or "unknown",
            cluster=str(r["cluster"]),
        )
        for r in records
        if r.get("lang") == config.language
    ]
    documents.sort(key=lambda d: (d.published_at, d.doc_id))
    if config.max_documents:
        # A prefix of the stream, not a sample: sampling would thin every story
        # and make the clustering problem easier than the one being measured.
        documents = documents[: config.max_documents]
    return documents


def _embed(documents: Sequence[BenchDocument], embedder: Embedder) -> np.ndarray:
    # The same text the live pipeline embeds: Article.body_text, headline first.
    texts = [_as_article(i, d).body_text for i, d in enumerate(documents)]
    blocks = []
    for start in range(0, len(texts), 256):
        blocks.append(np.asarray(embedder.encode(texts[start : start + 256]), dtype=np.float32))
        logger.info("Embedded %s/%s documents", min(start + 256, len(texts)), len(texts))
    return np.vstack(blocks)


def _as_article(row: int, document: BenchDocument) -> Article:
    url = f"benchmark://priberam-2018/{document.doc_id}"
    return Article(
        id=row + 1,
        url=url,
        canonical_url=url,
        title=document.title,
        excerpt=document.text,
        source_domain=document.source,
        language="en",
        topic=None,
        published_at=document.published_at,
        normalized_text_hash="",
        normalized_title_hash="",
        source_adapter="cluster_bench",
    )


def cluster_stream(
    documents: Sequence[BenchDocument],
    vectors: np.ndarray,
    *,
    threshold: float,
    settings: Settings,
    window_hours: int | None = None,
) -> tuple[dict[int, int], float]:
    """Replay the stream through the production clusterer; return labels and seconds."""
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as session:
            articles = [_as_article(i, d) for i, d in enumerate(documents)]
            session.add_all(articles)
            session.flush()
            clusterer = OnlineClusterer(
                session, settings=settings, threshold=threshold, window_hours=window_hours
            )
            started = time.perf_counter()
            # Documents are already in publication order, which is the order
            # live ingestion assigns them in.
            predicted = {
                article.id: clusterer.assign(article, vectors[i]).story_id
                for i, article in enumerate(articles)
            }
            return predicted, time.perf_counter() - started
    finally:
        engine.dispose()


def _truth(documents: Sequence[BenchDocument]) -> dict[int, int]:
    ids: dict[str, int] = {}
    return {i + 1: ids.setdefault(d.cluster, len(ids)) for i, d in enumerate(documents)}


def _score(
    documents: Sequence[BenchDocument],
    vectors: np.ndarray,
    threshold: float,
    settings: Settings,
    config: ClusterBenchConfig,
) -> dict[str, Any]:
    predicted, seconds = cluster_stream(
        documents, vectors, threshold=threshold, settings=settings, window_hours=config.window_hours
    )
    metrics = evaluate_clustering(_truth(documents), predicted).as_dict()
    return {
        "threshold": round(float(threshold), 4),
        **metrics,
        "cluster_seconds": round(seconds, 2),
        "articles_per_second": round(len(documents) / seconds, 1) if seconds else None,
    }


def _describe(documents: Sequence[BenchDocument]) -> dict[str, Any]:
    sizes = Counter(d.cluster for d in documents)
    return {
        "documents": len(documents),
        "clusters": len(sizes),
        "median_cluster_size": float(np.median(list(sizes.values()))) if sizes else 0.0,
        "largest_cluster": max(sizes.values()) if sizes else 0,
        "sources": len({d.source for d in documents}),
        "first": documents[0].published_at.isoformat() if documents else None,
        "last": documents[-1].published_at.isoformat() if documents else None,
    }


def run_cluster_bench(
    config: ClusterBenchConfig | None = None, *, settings: Settings | None = None
) -> dict[str, Any]:
    """Tune the threshold on dev (optionally), then score shipped and tuned on test."""
    settings = settings or get_settings()
    config = config or ClusterBenchConfig()
    set_global_seed(config.seed)
    embedder = get_embedder(settings)
    shipped = default_threshold(settings)

    test = load_split(config, "test", settings)
    if not test:
        raise ValueError(f"no {config.language!r} documents in the test split")
    started = time.perf_counter()
    test_vectors = _embed(test, embedder)
    embed_seconds = time.perf_counter() - started

    sweep: list[dict[str, Any]] = []
    tuned: float | None = None
    dev: list[BenchDocument] = []
    if config.tune:
        dev = load_split(config, "dev", settings)
        dev_vectors = _embed(dev, embedder)
        for threshold in config.thresholds:
            row = _score(dev, dev_vectors, threshold, settings, config)
            sweep.append(row)
            logger.info(
                "dev threshold=%.3f pairwise F1=%.4f (%s clusters)",
                threshold,
                row["pairwise_f1"],
                row["predicted_clusters"],
            )
        # Pairwise F1 is what the paper selects and reports on; ties go to the
        # higher threshold, which fragments rather than merges.
        tuned = max(sweep, key=lambda r: (r["pairwise_f1"], r["threshold"]))["threshold"]

    results = [{"setting": "shipped", **_score(test, test_vectors, shipped, settings, config)}]
    if tuned is not None and abs(tuned - shipped) > 1e-9:
        results.append(
            {"setting": "dev-tuned", **_score(test, test_vectors, tuned, settings, config)}
        )
    for row in results:
        logger.info(
            "test %s threshold=%.3f pairwise F1=%.4f B-cubed F1=%.4f",
            row["setting"],
            row["threshold"],
            row["pairwise_f1"],
            row["bcubed_f1"],
        )

    return {
        "dataset": {
            "name": "priberam-2018",
            "language": config.language,
            "test": _describe(test),
            "dev": _describe(dev) if dev else None,
            "label_source": "human (Rupnik et al. 2016 / Miranda et al. 2018)",
            "embedding_model": embedder.name,
            "embed_seconds": round(embed_seconds, 2),
        },
        "config": config.as_dict(),
        "clusterer": {
            "shipped_threshold": shipped,
            "dev_tuned_threshold": tuned,
            "window_hours": config.window_hours or settings.cluster_window_hours,
            "title_overlap_weight": settings.cluster_title_overlap_weight,
            "entity_overlap_weight": settings.cluster_entity_overlap_weight,
        },
        "results": results,
        "dev_sweep": sweep,
        "published": PUBLISHED_EN if config.language == "eng" else [],
        "notes": [
            "Pairwise P/R/F1 over all test documents of the language, as in Miranda et "
            "al. (2018), so the published rows are directly comparable.",
            "The published systems are supervised (an SVM ranker trained on the train "
            "split); NewsTrace fits one threshold on dev and nothing else.",
            "The embedder sees the headline and the start of the body only: "
            "all-MiniLM-L6-v2 truncates input at 256 word pieces.",
            "No confidence intervals: resampling stories does not give a clean "
            "bootstrap for pairwise metrics, and the paper reports none to compare with.",
        ],
    }


def render_table(metrics: dict[str, Any]) -> str:
    """Markdown summary: NewsTrace rows beside the published ones."""
    dataset = metrics.get("dataset", {})
    test = dataset.get("test") or {}
    lines = [
        f"# Story clustering: {dataset.get('name', '?')} ({dataset.get('language', '?')})",
        "",
        f"{test.get('documents', 0)} test documents in {test.get('clusters', 0)} human-labelled "
        f"stories from {test.get('sources', 0)} sources, replayed in publication order; "
        f"embedder `{dataset.get('embedding_model', '?')}`.",
        "",
        "| System | threshold | pairwise F1 | P | R | B-cubed F1 | stories predicted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.get("results", []):
        lines.append(
            f"| NewsTrace, {row['setting']} | {row['threshold']:.3f} | "
            f"{100 * row['pairwise_f1']:.1f} | {100 * row['pairwise_precision']:.1f} | "
            f"{100 * row['pairwise_recall']:.1f} | {100 * row['bcubed_f1']:.1f} | "
            f"{int(row['predicted_clusters'])} |"
        )
    for row in metrics.get("published", []):
        lines.append(
            f"| {row['system']} | - | {row['f1']:.1f} | {row['precision']:.1f} | "
            f"{row['recall']:.1f} | - | - |"
        )
    sweep = metrics.get("dev_sweep") or []
    if sweep:
        lines += [
            "",
            "Dev sweep (threshold selected on pairwise F1):",
            "",
            "| threshold | pairwise F1 | P | R | stories predicted |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in sweep:
            lines.append(
                f"| {row['threshold']:.3f} | {100 * row['pairwise_f1']:.1f} | "
                f"{100 * row['pairwise_precision']:.1f} | {100 * row['pairwise_recall']:.1f} | "
                f"{int(row['predicted_clusters'])} |"
            )
    lines.append("")
    for note in metrics.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"
