"""Run the compression sweep against a public IR benchmark.

The live sweep's honest conclusion is that *23 silver-labelled queries cannot
separate these representations*. That is a statement about the judgments, not
about the representations, and no amount of re-running it on the same corpus
fixes it: the labels are derived from the system's own clustering, and there
are too few of them for an interval to exclude zero.

A BEIR dataset supplies what the live corpus cannot -- hundreds of queries
with human relevance judgments, on text with real paraphrase -- and the
pipeline is representation-agnostic, so the identical projectors can be
measured there. Nothing here touches the NewsTrace database: it is an external
check on the compression claim, deliberately kept separate from the live
system's own evaluation.

Datasets are read in BEIR's published layout::

    <root>/<dataset>/corpus.jsonl      {"_id", "title", "text"}
    <root>/<dataset>/queries.jsonl     {"_id", "text"}
    <root>/<dataset>/qrels/test.tsv    query-id \\t corpus-id \\t score

``scripts/run_beir.py --download`` fetches and unpacks one; the fetch is the
only part of this project that needs the network, and it is opt-in.
"""

from __future__ import annotations

import csv
import json
import time
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from newstrace.config import Settings, get_settings
from newstrace.evaluation.retrieval import bootstrap_intervals, evaluate_rankings
from newstrace.logging import get_logger
from newstrace.representations.embedder import Embedder, get_embedder
from newstrace.representations.projection import make_projector
from newstrace.retrieval.exact import DenseRetriever, TfidfRetriever
from newstrace.utils import set_global_seed

logger = get_logger(__name__)

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

# Small enough to embed on a laptop CPU, and all three carry human judgments.
SUGGESTED = ("scifact", "nfcorpus", "arguana")


@dataclass
class BeirConfig:
    dataset: str = "scifact"
    data_dir: str = "data/beir"
    methods: list[str] = field(default_factory=lambda: ["full", "svd", "gaussian_rp", "sparse_rp"])
    dimensions: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    split: str = "test"
    top_k: int = 10
    max_documents: int | None = None
    max_queries: int | None = None
    include_tfidf: bool = True
    fit_fraction: float = 0.6
    seed: int = 549

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BeirDataset:
    name: str
    doc_ids: list[str]
    documents: list[str]
    query_ids: list[str]
    queries: list[str]
    # query id -> {doc id: graded relevance}
    qrels: dict[str, dict[str, int]]

    def __len__(self) -> int:
        return len(self.doc_ids)


def dataset_dir(config: BeirConfig, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    root = Path(config.data_dir)
    if not root.is_absolute():
        root = settings.resolve(root)
    return root / config.dataset


def download(config: BeirConfig, settings: Settings | None = None) -> Path:
    """Fetch and unpack one BEIR dataset. Requires network access."""
    import httpx

    target = dataset_dir(config, settings)
    target.parent.mkdir(parents=True, exist_ok=True)
    if (target / "corpus.jsonl").exists():
        logger.info("%s is already present at %s", config.dataset, target)
        return target

    url = BEIR_URL.format(dataset=config.dataset)
    archive = target.parent / f"{config.dataset}.zip"
    logger.info("Downloading %s", url)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with archive.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                handle.write(chunk)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target.parent)
    archive.unlink(missing_ok=True)
    logger.info("Unpacked %s to %s", config.dataset, target)
    return target


def load_dataset(config: BeirConfig, settings: Settings | None = None) -> BeirDataset:
    """Read a BEIR dataset from disk."""
    root = dataset_dir(config, settings)
    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_path = root / "qrels" / f"{config.split}.tsv"
    for path in (corpus_path, queries_path, qrels_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python scripts/run_beir.py --download "
                f"--dataset {config.dataset}` first (this step needs the network)."
            )

    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header and header[0].strip() in ("query-id", "query_id"):
            pass  # header consumed
        else:
            handle.seek(0)
            reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            query_id, doc_id, score = row[0].strip(), row[1].strip(), row[2].strip()
            try:
                graded = int(float(score))
            except ValueError:
                continue
            if graded > 0:
                qrels.setdefault(query_id, {})[doc_id] = graded

    judged_queries = set(qrels)
    query_ids: list[str] = []
    queries: list[str] = []
    with queries_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["_id"] in judged_queries:
                query_ids.append(record["_id"])
                queries.append(record.get("text", ""))
    if config.max_queries:
        query_ids, queries = query_ids[: config.max_queries], queries[: config.max_queries]
        judged_queries = set(query_ids)
        qrels = {q: v for q, v in qrels.items() if q in judged_queries}

    # Every judged document stays in the pool even when --max-documents trims
    # the corpus: dropping a relevant document would inflate every method's
    # score identically and quietly change what the numbers mean.
    required = {doc for query in qrels.values() for doc in query}
    doc_ids: list[str] = []
    documents: list[str] = []
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            doc_id = record["_id"]
            title = (record.get("title") or "").strip()
            body = (record.get("text") or "").strip()
            over_budget = config.max_documents and len(doc_ids) >= config.max_documents
            if over_budget and doc_id not in required:
                continue
            doc_ids.append(doc_id)
            documents.append(f"{title}. {body}" if title else body)

    return BeirDataset(config.dataset, doc_ids, documents, query_ids, queries, qrels)


def _embed(texts: Sequence[str], embedder: Embedder, *, batch_size: int = 256) -> np.ndarray:
    blocks = []
    for start in range(0, len(texts), batch_size):
        block = embedder.encode(list(texts[start : start + batch_size]))
        blocks.append(np.asarray(block, dtype=np.float32))
        logger.info("Embedded %s/%s documents", min(start + batch_size, len(texts)), len(texts))
    return np.vstack(blocks).astype(np.float32) if blocks else np.zeros((0, 0), dtype=np.float32)


def run_beir(
    config: BeirConfig | None = None, *, settings: Settings | None = None
) -> dict[str, Any]:
    """Run the sweep on a BEIR dataset and return the metrics document."""
    settings = settings or get_settings()
    config = config or BeirConfig()
    set_global_seed(config.seed)

    dataset = load_dataset(config, settings)
    embedder = get_embedder(settings)
    logger.info(
        "%s: %s documents, %s judged queries, embedder=%s",
        dataset.name,
        len(dataset),
        len(dataset.query_ids),
        embedder.name,
    )

    started = time.perf_counter()
    matrix = _embed(dataset.documents, embedder)
    embed_seconds = time.perf_counter() - started

    row_by_doc = {doc_id: index for index, doc_id in enumerate(dataset.doc_ids)}
    # Judgments are keyed by query *text* because that is what
    # ``evaluate_rankings`` keys on, and graded scores are binarised: the
    # aggregate metrics take relevance as a set.
    judgments: dict[str, set[int]] = {}
    for query_id, query_text in zip(dataset.query_ids, dataset.queries, strict=True):
        relevant = {
            row_by_doc[doc_id] for doc_id in dataset.qrels.get(query_id, {}) if doc_id in row_by_doc
        }
        if relevant:
            judgments[query_text] = relevant
    query_texts = [q for q in dataset.queries if q in judgments]
    if not query_texts:
        raise ValueError("no judged queries survived; check the qrels split")

    # The projectors are fitted on a held-out slice of the corpus, as they are
    # on the live corpus -- fitting on everything would let the projection see
    # the documents it is later scored on.
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(dataset.doc_ids))
    fit_rows = order[: max(2, int(config.fit_fraction * len(order)))]
    fit_matrix = matrix[fit_rows]

    results: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] | None = None
    baseline_rankings: dict[str, list[int]] | None = None
    row_ids = list(range(len(dataset.doc_ids)))

    for method in config.methods:
        dimensions = (
            [0] if method == "full" else [d for d in config.dimensions if d < matrix.shape[1]]
        )
        for dimension in dimensions:
            projector = make_projector(method, dimension, random_seed=config.seed)
            fit_start = time.perf_counter()
            projector.fit(fit_matrix)
            fit_seconds = time.perf_counter() - fit_start
            projected = projector.transform(matrix)

            retriever = DenseRetriever(row_ids, projected, method=method, projector=projector)
            vectors = [retriever.embed_query(q) for q in query_texts]
            rankings: dict[str, list[int]] = {}
            latency: list[float] = []
            for query, vector in zip(query_texts, vectors, strict=True):
                start = time.perf_counter()
                hits = retriever.search(vector, k=config.top_k)
                latency.append((time.perf_counter() - start) * 1000.0)
                rankings[query] = [h.article_id for h in hits]

            metrics, rows = evaluate_rankings(rankings, judgments, baseline=baseline_rankings)
            if method == "full":
                baseline_rows, baseline_rankings = rows, rankings
            intervals = bootstrap_intervals(
                rows, baseline_rows=None if method == "full" else baseline_rows, seed=config.seed
            )
            results.append(
                {
                    "method": method,
                    "dimension": int(projector.dimension)
                    if method != "full"
                    else int(matrix.shape[1]),
                    **metrics.as_dict(),
                    "index_memory_mib": round(retriever.memory_bytes() / 1048576, 4),
                    "p95_ms": round(float(np.percentile(latency, 95)), 4),
                    "fit_seconds": round(fit_seconds, 3),
                    "intervals": intervals,
                }
            )
            logger.info("%s d=%s nDCG@10=%.4f", method, projector.dimension, metrics.ndcg_at_10)

    if config.include_tfidf:
        lexical = TfidfRetriever(row_ids, dataset.documents)
        rankings = {}
        latency = []
        for query in query_texts:
            start = time.perf_counter()
            hits = lexical.search(query, k=config.top_k)
            latency.append((time.perf_counter() - start) * 1000.0)
            rankings[query] = [h.article_id for h in hits]
        metrics, rows = evaluate_rankings(rankings, judgments, baseline=baseline_rankings)
        results.append(
            {
                "method": "tfidf",
                "dimension": 0,
                **metrics.as_dict(),
                "index_memory_mib": round(lexical.memory_bytes() / 1048576, 4),
                "p95_ms": round(float(np.percentile(latency, 95)), 4),
                "fit_seconds": 0.0,
                "intervals": bootstrap_intervals(
                    rows, baseline_rows=baseline_rows, seed=config.seed
                ),
            }
        )

    return {
        "dataset": {
            "name": dataset.name,
            "documents": len(dataset),
            "queries": len(query_texts),
            "judgment_source": "human (BEIR qrels, binarised)",
            "embedding_model": embedder.name,
            "embed_seconds": round(embed_seconds, 2),
        },
        "config": config.as_dict(),
        "results": results,
        "notes": [
            "Relevance is binarised: a graded qrel of 2 counts the same as 1.",
            "Projectors are fitted on a random 60% of the corpus and scored on every "
            "document, mirroring the chronological fit used on the live corpus.",
        ],
    }


def render_table(metrics: dict[str, Any]) -> str:
    """Markdown summary, in the same shape as the live results table."""
    dataset = metrics.get("dataset", {})
    lines = [
        f"# BEIR: {dataset.get('name', '?')}",
        "",
        f"{dataset.get('documents', 0)} documents, {dataset.get('queries', 0)} judged queries, "
        f"embedder `{dataset.get('embedding_model', '?')}`.",
        "",
        "| Retrieval | dim | nDCG@10 | Recall@10 | top-10 overlap | Index | p95 | "
        "delta nDCG vs full (95% CI) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.get("results", []):
        delta = (row.get("intervals") or {}).get("ndcg_at_10_delta")
        ci = (
            f"{delta['mean']:+.3f} [{delta['lo']:+.3f}, {delta['hi']:+.3f}]"
            if delta
            else "baseline"
        )
        lines.append(
            f"| {row['method']} | {row['dimension'] or '-'} | {row['ndcg_at_10']:.3f} | "
            f"{row['recall_at_10']:.3f} | {row['topk_overlap_at_10']:.2f} | "
            f"{row['index_memory_mib']:.2f} MiB | {row['p95_ms']:.2f} ms | {ci} |"
        )
    lines.append("")
    for note in metrics.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"
