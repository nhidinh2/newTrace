# NewsTrace experiment report

Generated from `metrics.json` in this directory. Every number below is a
measurement produced by `scripts/run_experiments.py`; nothing is estimated.

## Task

Given a short keyword query built from a held-out article, retrieve the OTHER articles covering the same event. The source article is excluded from both the candidate pool and the relevant set.

## Setup

- Articles: **1120** (retrieval pool: 1120), drawn from a stored corpus of 3191 by keeping only the last 30 days (cutoff `2026-07-25T06:36:58.261066+00:00`)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (d=384)
- Chronological split: fit=672, validation=224, test=224
- Fit cutoff: `2026-08-22T19:01:01+00:00` — projectors saw **no** test-period article
- Queries: **42**
- Relevance judgments: `silver_from_full_dim_clustering_excluding_near_duplicates`

## Observed results

### Retrieval quality

| method | dim | Recall@10 | nDCG@10 | MRR | top-10 overlap |
| --- | --- | --- | --- | --- | --- |
| full | 384 | 0.9130 | 0.8326 | 0.8022 | 1.0000 |
| svd | 32 | 0.9130 | 0.8753 | 0.8587 | 0.4870 |
| svd | 64 | 0.9130 | 0.8487 | 0.8239 | 0.6391 |
| svd | 128 | 0.9130 | 0.8487 | 0.8239 | 0.7478 |
| svd | 256 | 0.9130 | 0.8487 | 0.8239 | 0.9304 |
| gaussian_rp | 32 | 0.9130 | 0.8432 | 0.8152 | 0.2913 |
| gaussian_rp | 64 | 0.9130 | 0.8326 | 0.8022 | 0.4348 |
| gaussian_rp | 128 | 0.9130 | 0.8487 | 0.8239 | 0.5565 |
| gaussian_rp | 256 | 0.9130 | 0.8326 | 0.8022 | 0.6217 |
| sparse_rp | 32 | 0.9130 | 0.8166 | 0.7804 | 0.2957 |
| sparse_rp | 64 | 0.9130 | 0.8207 | 0.7877 | 0.3609 |
| sparse_rp | 128 | 0.9130 | 0.8487 | 0.8239 | 0.4957 |
| sparse_rp | 256 | 0.9130 | 0.8487 | 0.8239 | 0.5913 |

### Computational cost

| method | dim | memory MiB | bytes/vector | p50 ms | p95 ms | fit s | transform s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | 384 | 1.6406 | 1536.0 | 0.4103 | 0.5911 | 0.0000 | 0.0058 |
| svd | 32 | 0.1367 | 128.0000 | 0.0757 | 0.0830 | 0.0173 | 0.0005 |
| svd | 64 | 0.2734 | 256.0000 | 0.0916 | 0.1005 | 0.0046 | 0.0005 |
| svd | 128 | 0.5469 | 512.0000 | 0.1102 | 0.1766 | 0.0090 | 0.0008 |
| svd | 256 | 1.0938 | 1024.0 | 0.1991 | 0.2365 | 0.0166 | 0.0018 |
| gaussian_rp | 32 | 0.1367 | 128.0000 | 0.0723 | 0.0798 | 0.0004 | 0.0006 |
| gaussian_rp | 64 | 0.2734 | 256.0000 | 0.0985 | 0.1311 | 0.0005 | 0.0005 |
| gaussian_rp | 128 | 0.5469 | 512.0000 | 0.1208 | 0.1567 | 0.0008 | 0.0008 |
| gaussian_rp | 256 | 1.0938 | 1024.0 | 0.2079 | 0.2461 | 0.0013 | 0.0018 |
| sparse_rp | 32 | 0.1367 | 128.0000 | 0.0742 | 0.0829 | 0.0006 | 0.0014 |
| sparse_rp | 64 | 0.2734 | 256.0000 | 0.0988 | 0.1272 | 0.0006 | 0.0015 |
| sparse_rp | 128 | 0.5469 | 512.0000 | 0.1161 | 0.1336 | 0.0010 | 0.0024 |
| sparse_rp | 256 | 1.0938 | 1024.0 | 0.2129 | 0.2614 | 0.0023 | 0.0034 |

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
| recall_at_5 | 0.9130 |
| recall_at_10 | 0.9130 |
| precision_at_5 | 0.2000 |
| precision_at_10 | 0.1000 |
| mrr | 0.8696 |
| ndcg_at_10 | 0.8744 |
| topk_overlap_at_10 | 0.3435 |
| queries | 23.0000 |

### Graph reranking (personalized PageRank)

| metric | value |
| --- | --- |
| recall_at_5 | 0.9130 |
| recall_at_10 | 0.9130 |
| precision_at_5 | 0.2000 |
| precision_at_10 | 0.1000 |
| mrr | 0.8696 |
| ndcg_at_10 | 0.8744 |
| topk_overlap_at_10 | 0.7696 |
| queries | 23.0000 |

Graph: 8367 nodes, 20518 edges, 84 components (largest holds 93.9% of nodes).

Convergence: mean 115.7 iterations, 100% of queries converged, max |sum(p) - 1| = 9.77e-15.

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
| stories_with_statements | 8 |
| stories_without_statements | 2 |
| citation_coverage_micro | 1.0000 |
| citation_coverage | 1.0000 |
| citation_precision | 1.0000 |
| unsupported_claim_rate | 0.0000 |
| redundancy_rate | 0.0000 |
| source_domain_diversity | 1.5000 |
| statements | 4.8750 |
| single_source_statements | 4.7500 |
| disputed_statements | 0.0000 |

## Interpretation

- `svd` at d=32: retained **105.1%** of baseline nDCG@10 with a 12.00× smaller index; p95 query latency 7.12× the baseline's (>1 = faster than full).
- `svd` at d=64: retained **101.9%** of baseline nDCG@10 with a 6.00× smaller index; p95 query latency 5.88× the baseline's (>1 = faster than full).
- `svd` at d=128: retained **101.9%** of baseline nDCG@10 with a 3.00× smaller index; p95 query latency 3.35× the baseline's (>1 = faster than full).
- `svd` at d=256: retained **101.9%** of baseline nDCG@10 with a 1.50× smaller index; p95 query latency 2.50× the baseline's (>1 = faster than full).
- `gaussian_rp` at d=32: retained **101.3%** of baseline nDCG@10 with a 12.00× smaller index; p95 query latency 7.41× the baseline's (>1 = faster than full).
- `gaussian_rp` at d=64: retained **100.0%** of baseline nDCG@10 with a 6.00× smaller index; p95 query latency 4.51× the baseline's (>1 = faster than full).
- `gaussian_rp` at d=128: retained **101.9%** of baseline nDCG@10 with a 3.00× smaller index; p95 query latency 3.77× the baseline's (>1 = faster than full).
- `gaussian_rp` at d=256: retained **100.0%** of baseline nDCG@10 with a 1.50× smaller index; p95 query latency 2.40× the baseline's (>1 = faster than full).
- `sparse_rp` at d=32: retained **98.1%** of baseline nDCG@10 with a 12.00× smaller index; p95 query latency 7.13× the baseline's (>1 = faster than full).
- `sparse_rp` at d=64: retained **98.6%** of baseline nDCG@10 with a 6.00× smaller index; p95 query latency 4.65× the baseline's (>1 = faster than full).
- `sparse_rp` at d=128: retained **101.9%** of baseline nDCG@10 with a 3.00× smaller index; p95 query latency 4.43× the baseline's (>1 = faster than full).
- `sparse_rp` at d=256: retained **101.9%** of baseline nDCG@10 with a 1.50× smaller index; p95 query latency 2.26× the baseline's (>1 = faster than full).

## What was skipped

_Nothing was skipped; every requested (method, dimension) pair ran._

## Statistical uncertainty

Percentile bootstrap over the 42 per-query results in `per_query_metrics.parquet` (or `.csv`), 2000 resamples, seeded. The delta column is a *paired* bootstrap of the per-query difference against the full-dimensional baseline: the same queries are easy or hard for every method, so pairing is what makes the comparison readable. `separable from full?` is *no* whenever the delta interval contains zero — that is, whenever this corpus cannot distinguish the method from the baseline.

| method | dim | nDCG@10 | 95% CI | Recall@10 | 95% CI | ΔnDCG vs full | Δ 95% CI | separable from full? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 384 | 0.8326 | [0.7022, 0.9519] | 0.9130 | [0.7826, 1.0000] | baseline | — | — |
| svd | 32 | 0.8753 | [0.7449, 0.9840] | 0.9130 | [0.7826, 1.0000] | +0.0427 | [0.0000, 0.1121] | no |
| svd | 64 | 0.8487 | [0.7125, 0.9679] | 0.9130 | [0.7826, 1.0000] | +0.0160 | [0.0000, 0.0481] | no |
| svd | 128 | 0.8487 | [0.7125, 0.9679] | 0.9130 | [0.7826, 1.0000] | +0.0160 | [0.0000, 0.0481] | no |
| svd | 256 | 0.8487 | [0.7125, 0.9679] | 0.9130 | [0.7826, 1.0000] | +0.0160 | [0.0000, 0.0481] | no |
| gaussian_rp | 32 | 0.8432 | [0.7071, 0.9519] | 0.9130 | [0.7826, 1.0000] | +0.0106 | [-0.0481, 0.0800] | no |
| gaussian_rp | 64 | 0.8326 | [0.7022, 0.9519] | 0.9130 | [0.7826, 1.0000] | +0.0000 | [0.0000, 0.0000] | no |
| gaussian_rp | 128 | 0.8487 | [0.7125, 0.9679] | 0.9130 | [0.7826, 1.0000] | +0.0160 | [0.0000, 0.0481] | no |
| gaussian_rp | 256 | 0.8326 | [0.7022, 0.9519] | 0.9130 | [0.7826, 1.0000] | +0.0000 | [0.0000, 0.0000] | no |
| sparse_rp | 32 | 0.8166 | [0.6822, 0.9356] | 0.9130 | [0.7826, 1.0000] | -0.0160 | [-0.0481, 0.0000] | no |
| sparse_rp | 64 | 0.8207 | [0.6777, 0.9412] | 0.9130 | [0.7826, 1.0000] | -0.0119 | [-0.0840, 0.0481] | no |
| sparse_rp | 128 | 0.8487 | [0.7125, 0.9679] | 0.9130 | [0.7826, 1.0000] | +0.0160 | [0.0000, 0.0481] | no |
| sparse_rp | 256 | 0.8487 | [0.7125, 0.9679] | 0.9130 | [0.7826, 1.0000] | +0.0160 | [0.0000, 0.0481] | no |

A bootstrap interval describes sampling variability in *this* query set only. It cannot repair a biased corpus, silver labels, or a query set too small to represent the task.

## Limitations

- Relevance judgments came from `silver_from_full_dim_clustering_excluding_near_duplicates`. Silver labels derived from the
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
