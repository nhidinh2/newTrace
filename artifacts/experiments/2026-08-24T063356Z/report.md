# NewsTrace experiment report

Generated from `metrics.json` in this directory. Every number below is a
measurement produced by `scripts/run_experiments.py`; nothing is estimated.

## Task

Given a short keyword query built from a held-out article, retrieve the OTHER articles covering the same event. The source article is excluded from both the candidate pool and the relevant set.

## Setup

- Articles: **1120** (retrieval pool: 1120), drawn from a stored corpus of 3191 by keeping only the last 30 days (cutoff `2026-07-25T06:33:56.758143+00:00`)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (d=384)
- Chronological split: fit=672, validation=224, test=224
- Fit cutoff: `2026-08-22T19:01:01+00:00` — projectors saw **no** test-period article
- Queries: **43**
- Relevance judgments: `silver_from_full_dim_clustering`

## Observed results

### Retrieval quality

| method | dim | Recall@10 | nDCG@10 | MRR | top-10 overlap |
| --- | --- | --- | --- | --- | --- |
| full | 384 | 0.9167 | 0.9167 | 0.9167 | 1.0000 |
| svd | 32 | 0.9167 | 0.9167 | 0.9167 | 0.4875 |
| svd | 64 | 0.9167 | 0.9167 | 0.9167 | 0.6417 |
| svd | 128 | 0.9167 | 0.9167 | 0.9167 | 0.7500 |
| svd | 256 | 0.9167 | 0.9167 | 0.9167 | 0.9333 |
| gaussian_rp | 32 | 0.9167 | 0.9013 | 0.8958 | 0.2833 |
| gaussian_rp | 64 | 0.9167 | 0.9167 | 0.9167 | 0.4292 |
| gaussian_rp | 128 | 0.9167 | 0.9167 | 0.9167 | 0.5458 |
| gaussian_rp | 256 | 0.9167 | 0.9167 | 0.9167 | 0.6125 |
| sparse_rp | 32 | 0.9167 | 0.9013 | 0.8958 | 0.2875 |
| sparse_rp | 64 | 0.9167 | 0.8898 | 0.8819 | 0.3625 |
| sparse_rp | 128 | 0.9167 | 0.9167 | 0.9167 | 0.4875 |
| sparse_rp | 256 | 0.9167 | 0.9167 | 0.9167 | 0.5958 |

### Computational cost

| method | dim | memory MiB | bytes/vector | p50 ms | p95 ms | fit s | transform s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | 384 | 1.6406 | 1536.0 | 0.3655 | 0.4345 | 0.0000 | 0.0033 |
| svd | 32 | 0.1367 | 128.0000 | 0.0725 | 0.0814 | 0.0245 | 0.0011 |
| svd | 64 | 0.2734 | 256.0000 | 0.0907 | 0.0993 | 0.0053 | 0.0005 |
| svd | 128 | 0.5469 | 512.0000 | 0.1203 | 0.1455 | 0.0091 | 0.0009 |
| svd | 256 | 1.0938 | 1024.0 | 0.1764 | 0.1982 | 0.0162 | 0.0019 |
| gaussian_rp | 32 | 0.1367 | 128.0000 | 0.0698 | 0.0781 | 0.0005 | 0.0007 |
| gaussian_rp | 64 | 0.2734 | 256.0000 | 0.0913 | 0.1268 | 0.0005 | 0.0006 |
| gaussian_rp | 128 | 0.5469 | 512.0000 | 0.1335 | 0.1458 | 0.0008 | 0.0010 |
| gaussian_rp | 256 | 1.0938 | 1024.0 | 0.2104 | 0.2364 | 0.0013 | 0.0016 |
| sparse_rp | 32 | 0.1367 | 128.0000 | 0.0695 | 0.0791 | 0.0027 | 0.0021 |
| sparse_rp | 64 | 0.2734 | 256.0000 | 0.0911 | 0.0999 | 0.0006 | 0.0019 |
| sparse_rp | 128 | 0.5469 | 512.0000 | 0.1211 | 0.1349 | 0.0010 | 0.0023 |
| sparse_rp | 256 | 1.0938 | 1024.0 | 0.2161 | 0.2473 | 0.0017 | 0.0034 |

### Geometry: distance and cosine distortion

| method | dim | mean |Δd|/d | p95 |Δd|/d | mean |Δcos| |
| --- | --- | --- | --- | --- |
| full | 384 | 0.0000 | 0.0000 | 0.0000 |
| svd | 32 | 0.1052 | 0.2576 | 0.1539 |
| svd | 64 | 0.0572 | 0.1572 | 0.0905 |
| svd | 128 | 0.0264 | 0.0740 | 0.0407 |
| svd | 256 | 0.0052 | 0.0151 | 0.0078 |
| gaussian_rp | 32 | 0.0841 | 0.2184 | 0.1424 |
| gaussian_rp | 64 | 0.0583 | 0.1426 | 0.0999 |
| gaussian_rp | 128 | 0.0402 | 0.0968 | 0.0654 |
| gaussian_rp | 256 | 0.0271 | 0.0684 | 0.0458 |
| sparse_rp | 32 | 0.0910 | 0.2289 | 0.1441 |
| sparse_rp | 64 | 0.0587 | 0.1409 | 0.0992 |
| sparse_rp | 128 | 0.0416 | 0.1043 | 0.0680 |
| sparse_rp | 256 | 0.0292 | 0.0726 | 0.0498 |

### Clustering

_No labelled clusters available; clustering metrics were not computed._

### Lexical baseline (TF-IDF)

| metric | value |
| --- | --- |
| recall_at_5 | 0.9167 |
| recall_at_10 | 0.9167 |
| precision_at_5 | 0.3500 |
| precision_at_10 | 0.1750 |
| mrr | 0.9167 |
| ndcg_at_10 | 0.9167 |
| topk_overlap_at_10 | 0.3375 |
| queries | 24.0000 |

### Graph reranking (personalized PageRank)

| metric | value |
| --- | --- |
| recall_at_5 | 0.9167 |
| recall_at_10 | 0.9167 |
| precision_at_5 | 0.3500 |
| precision_at_10 | 0.1750 |
| mrr | 0.9167 |
| ndcg_at_10 | 0.9167 |
| topk_overlap_at_10 | 0.7708 |
| queries | 24.0000 |

Graph: 8367 nodes, 20518 edges, 84 components (largest holds 93.9% of nodes).

Convergence: mean 115.8 iterations, 100% of queries converged, max |sum(p) - 1| = 9.77e-15.

| damping | iterations | top-10 overlap with first |
| --- | --- | --- |
| 0.50 | 28 | 0.00 |
| 0.70 | 53 | 1.00 |
| 0.85 | 115 | 1.00 |
| 0.95 | 355 | 1.00 |

### Summaries and evidence

| metric | value |
| --- | --- |
| stories_evaluated | 10 |
| citation_coverage | 0.8000 |
| citation_precision | 0.8000 |
| unsupported_claim_rate | 0.0000 |
| redundancy_rate | 0.0000 |
| source_domain_diversity | 1.2000 |
| statements | 3.9000 |
| single_source_statements | 3.8000 |
| disputed_statements | 0.0000 |

## Interpretation

- `svd` at d=32: retained **100.0%** of baseline nDCG@10 with a 12.00× smaller index; p95 query latency 5.33× the baseline's (>1 = faster than full).
- `svd` at d=64: retained **100.0%** of baseline nDCG@10 with a 6.00× smaller index; p95 query latency 4.38× the baseline's (>1 = faster than full).
- `svd` at d=128: retained **100.0%** of baseline nDCG@10 with a 3.00× smaller index; p95 query latency 2.99× the baseline's (>1 = faster than full).
- `svd` at d=256: retained **100.0%** of baseline nDCG@10 with a 1.50× smaller index; p95 query latency 2.19× the baseline's (>1 = faster than full).
- `gaussian_rp` at d=32: retained **98.3%** of baseline nDCG@10 with a 12.00× smaller index; p95 query latency 5.56× the baseline's (>1 = faster than full).
- `gaussian_rp` at d=64: retained **100.0%** of baseline nDCG@10 with a 6.00× smaller index; p95 query latency 3.43× the baseline's (>1 = faster than full).
- `gaussian_rp` at d=128: retained **100.0%** of baseline nDCG@10 with a 3.00× smaller index; p95 query latency 2.98× the baseline's (>1 = faster than full).
- `gaussian_rp` at d=256: retained **100.0%** of baseline nDCG@10 with a 1.50× smaller index; p95 query latency 1.84× the baseline's (>1 = faster than full).
- `sparse_rp` at d=32: retained **98.3%** of baseline nDCG@10 with a 12.00× smaller index; p95 query latency 5.49× the baseline's (>1 = faster than full).
- `sparse_rp` at d=64: retained **97.1%** of baseline nDCG@10 with a 6.00× smaller index; p95 query latency 4.35× the baseline's (>1 = faster than full).
- `sparse_rp` at d=128: retained **100.0%** of baseline nDCG@10 with a 3.00× smaller index; p95 query latency 3.22× the baseline's (>1 = faster than full).
- `sparse_rp` at d=256: retained **100.0%** of baseline nDCG@10 with a 1.50× smaller index; p95 query latency 1.76× the baseline's (>1 = faster than full).

## What was skipped

_Nothing was skipped; every requested (method, dimension) pair ran._

## Statistical uncertainty

Percentile bootstrap over the 43 per-query results in `per_query_metrics.parquet` (or `.csv`), 2000 resamples, seeded. The delta column is a *paired* bootstrap of the per-query difference against the full-dimensional baseline: the same queries are easy or hard for every method, so pairing is what makes the comparison readable. `separable from full?` is *no* whenever the delta interval contains zero — that is, whenever this corpus cannot distinguish the method from the baseline.

| method | dim | nDCG@10 | 95% CI | Recall@10 | 95% CI | ΔnDCG vs full | Δ 95% CI | separable from full? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 384 | 0.9167 | [0.7917, 1.0000] | 0.9167 | [0.7917, 1.0000] | baseline | — | — |
| svd | 32 | 0.9167 | [0.7917, 1.0000] | 0.9167 | [0.7917, 1.0000] | +0.0000 | [0.0000, 0.0000] | no |
| svd | 64 | 0.9167 | [0.7917, 1.0000] | 0.9167 | [0.7917, 1.0000] | +0.0000 | [0.0000, 0.0000] | no |
| svd | 128 | 0.9167 | [0.7917, 1.0000] | 0.9167 | [0.7917, 1.0000] | +0.0000 | [0.0000, 0.0000] | no |
| svd | 256 | 0.9167 | [0.7917, 1.0000] | 0.9167 | [0.7917, 1.0000] | +0.0000 | [0.0000, 0.0000] | no |
| gaussian_rp | 32 | 0.9013 | [0.7763, 1.0000] | 0.9167 | [0.7917, 1.0000] | -0.0154 | [-0.0461, 0.0000] | no |
| gaussian_rp | 64 | 0.9167 | [0.7917, 1.0000] | 0.9167 | [0.7917, 1.0000] | +0.0000 | [0.0000, 0.0000] | no |
| gaussian_rp | 128 | 0.9167 | [0.7917, 1.0000] | 0.9167 | [0.7917, 1.0000] | +0.0000 | [0.0000, 0.0000] | no |
| gaussian_rp | 256 | 0.9167 | [0.7917, 1.0000] | 0.9167 | [0.7917, 1.0000] | +0.0000 | [0.0000, 0.0000] | no |
| sparse_rp | 32 | 0.9013 | [0.7763, 1.0000] | 0.9167 | [0.7917, 1.0000] | -0.0154 | [-0.0461, 0.0000] | no |
| sparse_rp | 64 | 0.8898 | [0.7648, 1.0000] | 0.9167 | [0.7917, 1.0000] | -0.0268 | [-0.0805, 0.0000] | no |
| sparse_rp | 128 | 0.9167 | [0.7917, 1.0000] | 0.9167 | [0.7917, 1.0000] | +0.0000 | [0.0000, 0.0000] | no |
| sparse_rp | 256 | 0.9167 | [0.7917, 1.0000] | 0.9167 | [0.7917, 1.0000] | +0.0000 | [0.0000, 0.0000] | no |

A bootstrap interval describes sampling variability in *this* query set only. It cannot repair a biased corpus, silver labels, or a query set too small to represent the task.

## Limitations

- Relevance judgments came from `silver_from_full_dim_clustering`. Silver labels derived from the
  system's own clustering flatter the system: they measure self-consistency, not
  human-judged relevance.
- GDELT and the configured feeds do not represent all journalism.
- Publication timestamps do not establish who reported a claim first.
- Similar wording can indicate copying rather than independent corroboration.
- Source counts do not establish truth. NewsTrace is not a fact checker.

## Plots

- `quality_vs_dimension.png`
- `quality_vs_memory.png`
- `quality_vs_latency.png`
