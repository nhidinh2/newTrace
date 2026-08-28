# NewsTrace experiment report

Generated from `metrics.json` in this directory. Every number below is a
measurement produced by `scripts/run_experiments.py`; nothing is estimated.

## Task

Given a short keyword query built from a held-out article, retrieve the OTHER articles covering the same event. The source article is excluded from both the candidate pool and the relevant set.

## Setup

- Articles: **135** (retrieval pool: 135)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (d=384)
- Chronological split: fit=81, validation=27, test=27
- Fit cutoff: `2026-08-22T05:30:00+00:00` — projectors saw **no** test-period article
- Queries: **27**
- Relevance judgments: `synthetic_fixture_labels`

## Observed results

### Retrieval quality

| method | dim | Recall@10 | nDCG@10 | MRR | top-10 overlap |
| --- | --- | --- | --- | --- | --- |
| full | 384 | 1.0000 | 0.9144 | 0.8846 | 1.0000 |
| svd | 32 | 0.9904 | 0.8664 | 0.8244 | 0.7462 |
| svd | 64 | 1.0000 | 0.9033 | 0.8782 | 0.8654 |
| svd | 80 | 1.0000 | 0.9045 | 0.8782 | 0.8769 |
| gaussian_rp | 32 | 0.9615 | 0.8642 | 0.8365 | 0.6731 |
| gaussian_rp | 64 | 0.9615 | 0.8681 | 0.8462 | 0.7731 |
| gaussian_rp | 128 | 1.0000 | 0.9197 | 0.8974 | 0.7769 |
| gaussian_rp | 256 | 1.0000 | 0.9144 | 0.8846 | 0.8192 |
| sparse_rp | 32 | 0.8558 | 0.6693 | 0.6236 | 0.5731 |
| sparse_rp | 64 | 0.9904 | 0.8929 | 0.8782 | 0.6962 |
| sparse_rp | 128 | 1.0000 | 0.9082 | 0.8837 | 0.7577 |
| sparse_rp | 256 | 1.0000 | 0.9205 | 0.8910 | 0.8077 |

### Computational cost

| method | dim | memory MiB | bytes/vector | p50 ms | p95 ms | fit s | transform s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | 384 | 0.1978 | 1536.0 | 0.0501 | 0.0531 | 0.0000 | 0.0001 |
| svd | 32 | 0.0165 | 128.0000 | 0.0278 | 0.0304 | 0.0036 | 0.0001 |
| svd | 64 | 0.0330 | 256.0000 | 0.0301 | 0.0323 | 0.0036 | 0.0001 |
| svd | 80 | 0.0412 | 320.0000 | 0.0316 | 0.0362 | 0.0023 | 0.0001 |
| gaussian_rp | 32 | 0.0165 | 128.0000 | 0.0275 | 0.0303 | 0.0003 | 0.0001 |
| gaussian_rp | 64 | 0.0330 | 256.0000 | 0.0308 | 0.0331 | 0.0004 | 0.0001 |
| gaussian_rp | 128 | 0.0659 | 512.0000 | 0.0338 | 0.0349 | 0.0006 | 0.0001 |
| gaussian_rp | 256 | 0.1318 | 1024.0 | 0.0398 | 0.0470 | 0.0012 | 0.0001 |
| sparse_rp | 32 | 0.0165 | 128.0000 | 0.0272 | 0.0293 | 0.0017 | 0.0004 |
| sparse_rp | 64 | 0.0330 | 256.0000 | 0.0312 | 0.0376 | 0.0006 | 0.0001 |
| sparse_rp | 128 | 0.0659 | 512.0000 | 0.0340 | 0.0413 | 0.0009 | 0.0001 |
| sparse_rp | 256 | 0.1318 | 1024.0 | 0.0397 | 0.0471 | 0.0015 | 0.0002 |

### Geometry: distance and cosine distortion

| method | dim | mean |Δd|/d | p95 |Δd|/d | mean |Δcos| |
| --- | --- | --- | --- | --- |
| full | 384 | 0.0000 | 0.0000 | 0.0000 |
| svd | 32 | 0.0477 | 0.1899 | 0.0530 |
| svd | 64 | 0.0222 | 0.0928 | 0.0285 |
| svd | 80 | 0.0205 | 0.0854 | 0.0241 |
| gaussian_rp | 32 | 0.0965 | 0.2293 | 0.1433 |
| gaussian_rp | 64 | 0.0640 | 0.1574 | 0.0900 |
| gaussian_rp | 128 | 0.0399 | 0.0969 | 0.0595 |
| gaussian_rp | 256 | 0.0314 | 0.0775 | 0.0442 |
| sparse_rp | 32 | 0.1030 | 0.2579 | 0.1425 |
| sparse_rp | 64 | 0.0583 | 0.1439 | 0.0854 |
| sparse_rp | 128 | 0.0468 | 0.1226 | 0.0654 |
| sparse_rp | 256 | 0.0321 | 0.0775 | 0.0465 |

### Clustering

| method | dim | pairwise F1 | B-cubed F1 | ARI | fragmentation | merge errors |
| --- | --- | --- | --- | --- | --- | --- |
| full | 384 | 0.9070 | 0.9441 | 0.9052 | 0.1463 | 0.0426 |
| svd | 32 | 0.7703 | 0.8717 | 0.7649 | 0.0732 | 0.2647 |
| svd | 64 | 0.7742 | 0.8744 | 0.7691 | 0.1463 | 0.1951 |
| svd | 80 | 0.7899 | 0.8853 | 0.7853 | 0.1463 | 0.1667 |
| gaussian_rp | 32 | 0.7928 | 0.8996 | 0.7883 | 0.1463 | 0.1163 |
| gaussian_rp | 64 | 0.8010 | 0.9070 | 0.7967 | 0.1463 | 0.0909 |
| gaussian_rp | 128 | 0.8449 | 0.9153 | 0.8417 | 0.1463 | 0.1163 |
| gaussian_rp | 256 | 0.8696 | 0.9277 | 0.8669 | 0.1220 | 0.0909 |
| sparse_rp | 32 | 0.8457 | 0.9176 | 0.8425 | 0.1220 | 0.1190 |
| sparse_rp | 64 | 0.8125 | 0.8989 | 0.8085 | 0.1463 | 0.1395 |
| sparse_rp | 128 | 0.8189 | 0.9054 | 0.8150 | 0.1463 | 0.1136 |
| sparse_rp | 256 | 0.8571 | 0.9218 | 0.8542 | 0.1463 | 0.0889 |

### Lexical baseline (TF-IDF)

| metric | value |
| --- | --- |
| recall_at_5 | 1.0000 |
| recall_at_10 | 1.0000 |
| precision_at_5 | 0.4615 |
| precision_at_10 | 0.2308 |
| mrr | 0.9212 |
| ndcg_at_10 | 0.9364 |
| topk_overlap_at_10 | 0.7154 |
| queries | 26.0000 |

### Graph reranking (personalized PageRank)

| metric | value |
| --- | --- |
| recall_at_5 | 0.9808 |
| recall_at_10 | 1.0000 |
| precision_at_5 | 0.4538 |
| precision_at_10 | 0.2308 |
| mrr | 0.8410 |
| ndcg_at_10 | 0.8809 |
| topk_overlap_at_10 | 0.8885 |
| queries | 26.0000 |

Graph: 575 nodes, 2128 edges, 1 components (largest holds 100.0% of nodes).

Convergence: mean 112.7 iterations, 100% of queries converged, max |sum(p) - 1| = 2.00e-15.

| damping | iterations | top-10 overlap with first |
| --- | --- | --- |
| 0.50 | 27 | 0.00 |
| 0.70 | 51 | 0.70 |
| 0.85 | 109 | 0.40 |
| 0.95 | 315 | 0.00 |

### Summaries and evidence

| metric | value |
| --- | --- |
| stories_evaluated | 10 |
| citation_coverage | 1.0000 |
| citation_precision | 0.9889 |
| unsupported_claim_rate | 0.0111 |
| redundancy_rate | 0.0000 |
| source_domain_diversity | 4.0000 |
| statements | 8.1000 |
| single_source_statements | 7.0000 |
| disputed_statements | 0.2000 |

## Interpretation

- `svd` at d=32: retained **94.8%** of baseline nDCG@10 with a 12.00× smaller index; p95 query latency 1.75× the baseline's (>1 = faster than full).
- `svd` at d=64: retained **98.8%** of baseline nDCG@10 with a 6.00× smaller index; p95 query latency 1.65× the baseline's (>1 = faster than full).
- `svd` at d=80: retained **98.9%** of baseline nDCG@10 with a 4.80× smaller index; p95 query latency 1.47× the baseline's (>1 = faster than full).
- `gaussian_rp` at d=32: retained **94.5%** of baseline nDCG@10 with a 12.00× smaller index; p95 query latency 1.75× the baseline's (>1 = faster than full).
- `gaussian_rp` at d=64: retained **94.9%** of baseline nDCG@10 with a 6.00× smaller index; p95 query latency 1.61× the baseline's (>1 = faster than full).
- `gaussian_rp` at d=128: retained **100.6%** of baseline nDCG@10 with a 3.00× smaller index; p95 query latency 1.52× the baseline's (>1 = faster than full).
- `gaussian_rp` at d=256: retained **100.0%** of baseline nDCG@10 with a 1.50× smaller index; p95 query latency 1.13× the baseline's (>1 = faster than full).
- `sparse_rp` at d=32: retained **73.2%** of baseline nDCG@10 with a 12.00× smaller index; p95 query latency 1.81× the baseline's (>1 = faster than full).
- `sparse_rp` at d=64: retained **97.6%** of baseline nDCG@10 with a 6.00× smaller index; p95 query latency 1.41× the baseline's (>1 = faster than full).
- `sparse_rp` at d=128: retained **99.3%** of baseline nDCG@10 with a 3.00× smaller index; p95 query latency 1.29× the baseline's (>1 = faster than full).
- `sparse_rp` at d=256: retained **100.7%** of baseline nDCG@10 with a 1.50× smaller index; p95 query latency 1.13× the baseline's (>1 = faster than full).

## What was skipped

- svd: requested d=128 clamped to d=80 by the fit-set size (81 articles)
- svd: requested d=256 collapsed onto d=80 (TruncatedSVD is capped at n_fit_samples - 1 = 80); not run twice

## Statistical uncertainty

Percentile bootstrap over the 27 per-query results in `per_query_metrics.parquet` (or `.csv`), 2000 resamples, seeded. The delta column is a *paired* bootstrap of the per-query difference against the full-dimensional baseline: the same queries are easy or hard for every method, so pairing is what makes the comparison readable. `separable from full?` is *no* whenever the delta interval contains zero — that is, whenever this corpus cannot distinguish the method from the baseline.

| method | dim | nDCG@10 | 95% CI | Recall@10 | 95% CI | ΔnDCG vs full | Δ 95% CI | separable from full? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 384 | 0.9144 | [0.8410, 0.9781] | 1.0000 | [1.0000, 1.0000] | baseline | — | — |
| svd | 32 | 0.8664 | [0.7760, 0.9527] | 0.9904 | [0.9712, 1.0000] | -0.0545 | [-0.1025, -0.0066] | **yes** |
| svd | 64 | 0.9033 | [0.8264, 0.9678] | 1.0000 | [1.0000, 1.0000] | -0.0129 | [-0.0245, -0.0013] | **yes** |
| svd | 80 | 0.9045 | [0.8273, 0.9698] | 1.0000 | [1.0000, 1.0000] | -0.0116 | [-0.0232, 0.0000] | no |
| gaussian_rp | 32 | 0.8642 | [0.7568, 0.9551] | 0.9615 | [0.8846, 1.0000] | -0.0612 | [-0.1195, -0.0029] | **yes** |
| gaussian_rp | 64 | 0.8681 | [0.7602, 0.9531] | 0.9615 | [0.8846, 1.0000] | -0.0572 | [-0.1113, -0.0031] | **yes** |
| gaussian_rp | 128 | 0.9197 | [0.8458, 0.9808] | 1.0000 | [1.0000, 1.0000] | +0.0079 | [-0.0151, 0.0310] | no |
| gaussian_rp | 256 | 0.9144 | [0.8410, 0.9781] | 1.0000 | [1.0000, 1.0000] | +0.0000 | [0.0000, 0.0000] | no |
| sparse_rp | 32 | 0.6693 | [0.5341, 0.7986] | 0.8558 | [0.7212, 0.9615] | -0.2505 | [-0.3817, -0.1192] | **yes** |
| sparse_rp | 64 | 0.8929 | [0.8102, 0.9608] | 0.9904 | [0.9712, 1.0000] | -0.0221 | [-0.0726, 0.0284] | no |
| sparse_rp | 128 | 0.9082 | [0.8222, 0.9777] | 1.0000 | [1.0000, 1.0000] | -0.0067 | [-0.0394, 0.0260] | no |
| sparse_rp | 256 | 0.9205 | [0.8409, 0.9835] | 1.0000 | [1.0000, 1.0000] | +0.0102 | [-0.0228, 0.0432] | no |

A bootstrap interval describes sampling variability in *this* query set only. It cannot repair a biased corpus, silver labels, or a query set too small to represent the task.

## Limitations

- Relevance judgments came from `synthetic_fixture_labels`. Silver labels derived from the
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
