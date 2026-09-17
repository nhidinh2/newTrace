# NewsTrace

Real-time news **story** tracker: it ingests AI and technology news, groups the
articles that cover the same event into one evolving story, and answers queries
with summaries where every statement links back to the reporting it came from.

Built for MCS 549 (algorithms for massive data), so the interesting part is the
measurement: **how much can you compress the embeddings of a live news stream
before retrieval quality suffers?** On a 30-day live corpus, the answer was
"more than expected" — a 12× smaller index with no measurable loss.

| | |
| --- | --- |
| ![Stories](docs/screenshots/stories.png) | ![Story](docs/screenshots/story.png) |
| **Stories** — article and *independent*-source counts | **Story** — timeline, earliest report, near-duplicates marked |
| ![Search](docs/screenshots/search.png) | ![Experiments](docs/screenshots/experiments.png) |
| **Search** — every hit carries provenance and a "why this ranked here" line | **Experiments** — full vs. compressed retrieval, from saved metrics |

That story page is the duplicate handling doing real work: nine articles, eight
of them regional syndications of a single wire story, counted as **one**
independent source rather than nine.

## Quickstart

No API key, no GPU, no network — the demo runs on committed fixtures.

```bash
make setup    # uv sync + .env
make db       # alembic upgrade head
make demo     # ingest fixtures
make ui       # Streamlit on :8501   (make api for the FastAPI service)
```

Live ingestion once that works:

```bash
uv run newstrace ingest --source gdelt --topic ai_models --timespan 24h
NEWSTRACE_TOPICS_FILE=configs/topics.live.yaml uv run newstrace ingest --source rss
uv run newstrace search "AI agent security evaluations" --top-k 10
```

`make help` lists every target; `make check` runs lint, types and tests.

## Results

From [`docs/experiments/live-30d`](docs/experiments/live-30d) — 3,191 articles
ingested from GDELT DOC 2.0 and 36 publisher feeds across 292 domains, swept
over the most recent 30 days (1,120 articles, 23 silver-labelled queries).

| Retrieval | dim | nDCG@10 | Recall@10 | Index | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full embeddings | 384 | 0.833 | 0.913 | 1.64 MiB | 0.64 ms |
| Truncated SVD | 32 | 0.875 | 0.913 | 0.14 MiB | 0.09 ms |
| Gaussian RP | 32 | 0.843 | 0.913 | 0.14 MiB | 0.09 ms |
| Sparse RP | 32 | 0.817 | 0.913 | 0.14 MiB | 0.10 ms |

**92% less embedding memory and 7× faster p95 queries, with no measurable loss
in ranking quality.** "No measurable loss" is the defensible claim: SVD@32 beat
the baseline on nDCG, but the paired bootstrap interval on that difference is
[0.000, 0.112] and contains zero — as does every other interval in the run.
Compression cost nothing detectable on 23 queries; it did not help.

Also measured: clustering F1, PageRank convergence and damping sensitivity,
ingestion throughput, and 100% citation coverage on generated summaries. Details
and caveats in [`docs/evaluation.md`](docs/evaluation.md).

## How it works

```
GDELT + RSS → normalize & dedupe → embed (MiniLM-L6-v2) → project (SVD / RP)
   → online clustering (72h window) → claims & evidence → graph + PageRank
   → FastAPI + Streamlit
```

Python 3.11+, SQLite/SQLAlchemy, FastAPI, Streamlit, sentence-transformers,
NumPy/scikit-learn, NetworkX. CPU-only by default; LLM summaries are optional
and off (`LLM_PROVIDER=none`) — the default summaries are extractive.

## What it does not do

It reports **agreement, provenance, duplication and source diversity**. It is
not a fact checker: it never scores a source's trustworthiness or bias, never
declares a claim true or false, and never treats a syndicated copy as
independent confirmation. Similar wording can mean copying rather than
corroboration, and source counts do not establish truth. Full list in
[`docs/limitations.md`](docs/limitations.md).

## Docs

- [Architecture](docs/architecture.md) — data flow, module map, design decisions
- [Evaluation](docs/evaluation.md) — protocol, metrics, uncertainty, reproduction
- [Limitations](docs/limitations.md) · [Model card](docs/model_card.md)
- [Experiments](docs/experiments) — committed runs on [fixtures](docs/experiments/baseline) and [live data](docs/experiments/live-30d)
- [Specification](docs/spec.md) — the original design brief

<details>
<summary><b>Course topics → implementation</b></summary>

| MCS 549 topic | In NewsTrace | Measured by |
| --- | --- | --- |
| High-dimensional geometry | Cosine-neighbour preservation under compression | Pairwise distortion, top-*k* overlap |
| Singular value decomposition | Truncated SVD of article embeddings | nDCG/Recall vs. memory and latency |
| Random projections | Gaussian and sparse projections | Empirical distortion vs. JL bounds |
| Clustering | Online grouping of same-event articles | Pairwise F1, B-cubed F1 |
| Streaming algorithms | Incremental ingest, dedupe, centroid updates | Throughput, update latency, peak memory |
| Random graphs | Article–source–claim–entity graph | Degree distribution, components |
| Random walks | Story-seeded Personalized PageRank | nDCG change vs. cosine-only |
| Markov chains | PageRank as a stationary distribution | Convergence iterations, damping sensitivity |

</details>
