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
import math
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from newstrace.clustering.online import OnlineClusterer, default_threshold
from newstrace.config import Settings, get_settings
from newstrace.evaluation.clustering import evaluate_clustering
from newstrace.logging import get_logger
from newstrace.models import Article, Base, StoryCluster
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

# Wide enough for both blends: learned weights put more of the score in the
# overlap bonuses, which moves the best threshold down (0.425 on the English dev split).
DEFAULT_GRID = tuple(round(x, 3) for x in np.arange(0.30, 0.975, 0.025))


@dataclass
class ClusterBenchConfig:
    data_dir: str = "data/cluster_bench"
    language: str = "eng"
    thresholds: list[float] = field(default_factory=lambda: list(DEFAULT_GRID))
    tune: bool = True
    learn_weights: bool = True
    window_hours: int | None = None
    max_documents: int | None = None
    seed: int = 549

    def __post_init__(self) -> None:
        if self.tune and (
            not self.thresholds or not all(math.isfinite(t) for t in self.thresholds)
        ):
            raise ValueError("thresholds must be a non-empty list of finite numbers when tuning")

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


@dataclass
class LearnedWeights:
    """Blend weights fitted on labelled pairs, plus the fit they came from."""

    title_weight: float
    entity_weight: float
    time_weight: float
    coefficients: list[float]
    intercept: float
    implied_threshold: float
    pairs: int
    positives: int

    def as_dict(self) -> dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


def oracle_signals(
    documents: Sequence[BenchDocument],
    vectors: np.ndarray,
    *,
    settings: Settings,
    window_hours: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the stream with the gold assignments and record, for every article,
    the clusterer's (cosine, title, entity, gap in days) signals against each
    story in its window, labelled by whether that story is the article's own.

    Stories evolve exactly as the production clusterer evolves them -- centroid,
    title tokens, entities, time span -- but membership comes from the labels, so
    the training pairs do not inherit the clusterer's own mistakes. Miranda et al.
    train their ranker the same way.
    """
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    rows: list[tuple[float, float, float, float]] = []
    labels: list[bool] = []
    try:
        with Session(engine) as session:
            articles = [_as_article(i, d) for i, d in enumerate(documents)]
            session.add_all(articles)
            session.flush()
            clusterer = OnlineClusterer(session, settings=settings, window_hours=window_hours)
            stories: dict[str, StoryCluster] = {}
            for i, (document, article) in enumerate(zip(documents, articles, strict=True)):
                own = stories.get(document.cluster)
                features = clusterer.article_features(article)
                for cluster in clusterer.candidate_clusters(article.published_at, article.topic):
                    rows.append(clusterer.signals(article, vectors[i], cluster, features=features))
                    labels.append(own is not None and cluster.id == own.id)
                # The clusterer's own state updates, driven by the label instead
                # of by its decision.
                if own is None:
                    own = stories[document.cluster] = clusterer._new_cluster(article, vectors[i])
                clusterer._record(article, own, vectors[i], 1.0, "oracle")
    finally:
        engine.dispose()
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4), np.asarray(labels, dtype=bool)


def fit_weights(signals: np.ndarray, labels: np.ndarray) -> LearnedWeights:
    """Fit the title, entity and time weights of ``combined_score`` by logistic
    regression on ``oracle_signals`` rows.

    The model's boundary ``a*cos + b*title + c*entity + g*gap + d = 0`` is the
    blend ``cos + (b/a)*title + (c/a)*entity - (-g/a)*gap = -d/a``, so the
    ratios are the weights in
    the units the threshold is already in, and the production scorer needs no
    change to use them. The threshold is still swept on dev rather than taken
    from ``-d/a``: the pairwise boundary is not the clustering decision, which
    takes the best candidate first and thresholds second.
    """
    if not labels.any() or labels.all():
        raise ValueError("fitting blend weights needs both matching and non-matching pairs")
    model = LogisticRegression(C=1e4, max_iter=5000).fit(signals, labels)
    cosine, title, entity, gap = (float(c) for c in model.coef_[0])
    intercept = float(model.intercept_[0])
    if cosine <= 0:
        raise ValueError(
            f"cosine got a non-positive coefficient ({cosine:.3f}); "
            "the blend cannot express this fit"
        )
    return LearnedWeights(
        title_weight=title / cosine,
        entity_weight=entity / cosine,
        # A penalty, so positive when a longer gap makes a match less likely.
        time_weight=-gap / cosine,
        coefficients=[cosine, title, entity, gap],
        intercept=intercept,
        implied_threshold=-intercept / cosine,
        pairs=int(labels.size),
        positives=int(labels.sum()),
    )


def _with_weights(
    settings: Settings, weights: LearnedWeights, *, time_only: bool = False
) -> Settings:
    """``settings`` with the fitted weights; ``time_only`` keeps the shipped
    title and entity weights, so the time signal is measured on its own."""
    update: dict[str, float] = {"cluster_time_gap_weight": weights.time_weight}
    if not time_only:
        update["cluster_title_overlap_weight"] = weights.title_weight
        update["cluster_entity_overlap_weight"] = weights.entity_weight
    return settings.model_copy(update=update)


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
        "title_weight": round(settings.cluster_title_overlap_weight, 4),
        "entity_weight": round(settings.cluster_entity_overlap_weight, 4),
        "time_weight": round(settings.cluster_time_gap_weight, 4),
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


def _sweep(
    documents: Sequence[BenchDocument],
    vectors: np.ndarray,
    settings: Settings,
    config: ClusterBenchConfig,
    name: str,
) -> tuple[list[dict[str, Any]], float]:
    sweep = []
    for threshold in config.thresholds:
        row = _score(documents, vectors, threshold, settings, config)
        sweep.append(row)
        logger.info(
            "dev %s threshold=%.3f pairwise F1=%.4f (%s clusters)",
            name,
            threshold,
            row["pairwise_f1"],
            row["predicted_clusters"],
        )
    # Pairwise F1 is what the paper selects and reports on; ties go to the
    # higher threshold, which fragments rather than merges.
    best = max(sweep, key=lambda r: (r["pairwise_f1"], r["threshold"]))["threshold"]
    return sweep, best


def run_cluster_bench(
    config: ClusterBenchConfig | None = None, *, settings: Settings | None = None
) -> dict[str, Any]:
    """Tune on dev (optionally), then score the shipped and dev-fitted settings on test.

    ``dev-tuned`` keeps the shipped blend weights and fits only the threshold.
    ``dev-learned-time`` adds only the fitted time weight to the shipped blend;
    ``dev-learned`` uses every fitted weight (``fit_weights``). Each variant has
    its threshold swept on dev separately.
    """
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
    learned: LearnedWeights | None = None
    # setting name -> (fitted settings, dev sweep, selected threshold)
    variants: dict[str, tuple[Settings, list[dict[str, Any]], float]] = {}
    dev: list[BenchDocument] = []
    if config.tune:
        dev = load_split(config, "dev", settings)
        dev_vectors = _embed(dev, embedder)
        sweep, tuned = _sweep(dev, dev_vectors, settings, config, "shipped weights")
        if config.learn_weights:
            signals, labels = oracle_signals(
                dev, dev_vectors, settings=settings, window_hours=config.window_hours
            )
            learned = fit_weights(signals, labels)
            logger.info(
                "learned weights on %s dev pairs: title=%.3f entity=%.3f time=%.3f/day",
                learned.pairs,
                learned.title_weight,
                learned.entity_weight,
                learned.time_weight,
            )
            for name, fitted in (
                ("dev-learned-time", _with_weights(settings, learned, time_only=True)),
                ("dev-learned", _with_weights(settings, learned)),
            ):
                variants[name] = (fitted, *_sweep(dev, dev_vectors, fitted, config, name))

    results = [{"setting": "shipped", **_score(test, test_vectors, shipped, settings, config)}]
    if tuned is not None and abs(tuned - shipped) > 1e-9:
        results.append(
            {"setting": "dev-tuned", **_score(test, test_vectors, tuned, settings, config)}
        )
    for name, (fitted, _, threshold) in variants.items():
        results.append({"setting": name, **_score(test, test_vectors, threshold, fitted, config)})
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
            "learned_weights": learned.as_dict() if learned else None,
            "dev_learned_thresholds": {name: v[2] for name, v in variants.items()},
        },
        "results": results,
        "dev_sweep": sweep,
        "dev_sweep_learned": {name: v[1] for name, v in variants.items()},
        "published": PUBLISHED_EN if config.language == "eng" else [],
        "notes": [
            "Pairwise P/R/F1 over all test documents of the language, as in Miranda et "
            "al. (2018), so the published rows are directly comparable.",
            "The published systems are supervised (an SVM ranker trained on the train "
            "split). NewsTrace fits a threshold on dev (dev-tuned), or blend weights "
            "and then a threshold (dev-learned*); nothing is fitted on test.",
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
        "| System | title w | entity w | time w/day | threshold | pairwise F1 | P | R "
        "| B-cubed F1 | stories predicted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.get("results", []):
        lines.append(
            f"| NewsTrace, {row['setting']} | {row.get('title_weight', float('nan')):.2f} | "
            f"{row.get('entity_weight', float('nan')):.2f} | "
            f"{row.get('time_weight', 0.0):.3f} | {row['threshold']:.3f} | "
            f"{100 * row['pairwise_f1']:.1f} | {100 * row['pairwise_precision']:.1f} | "
            f"{100 * row['pairwise_recall']:.1f} | {100 * row['bcubed_f1']:.1f} | "
            f"{int(row['predicted_clusters'])} |"
        )
    for row in metrics.get("published", []):
        lines.append(
            f"| {row['system']} | - | - | - | - | {row['f1']:.1f} | {row['precision']:.1f} | "
            f"{row['recall']:.1f} | - | - |"
        )
    sweeps = {"shipped weights": metrics.get("dev_sweep") or []}
    sweeps.update(metrics.get("dev_sweep_learned") or {})
    for label, sweep in sweeps.items():
        if not sweep:
            continue
        lines += [
            "",
            f"Dev sweep, {label} (threshold selected on pairwise F1):",
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
