# BEIR: scifact

5183 documents, 300 judged queries, embedder `sentence-transformers/all-MiniLM-L6-v2`.

| Retrieval | dim | nDCG@10 | Recall@10 | top-10 overlap | Index | p95 | delta nDCG vs full (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 384 | 0.648 | 0.788 | 1.00 | 7.59 MiB | 0.14 ms | baseline |
| svd | 32 | 0.461 | 0.606 | 0.47 | 0.63 MiB | 0.07 ms | -0.187 [-0.229, -0.147] |
| svd | 64 | 0.559 | 0.703 | 0.65 | 1.27 MiB | 0.07 ms | -0.089 [-0.119, -0.061] |
| svd | 128 | 0.619 | 0.770 | 0.81 | 2.53 MiB | 0.07 ms | -0.030 [-0.046, -0.015] |
| svd | 256 | 0.647 | 0.790 | 0.96 | 5.06 MiB | 0.08 ms | -0.002 [-0.007, +0.003] |
| gaussian_rp | 32 | 0.321 | 0.451 | 0.22 | 0.63 MiB | 0.07 ms | -0.327 [-0.374, -0.281] |
| gaussian_rp | 64 | 0.468 | 0.601 | 0.37 | 1.27 MiB | 0.07 ms | -0.180 [-0.220, -0.142] |
| gaussian_rp | 128 | 0.566 | 0.715 | 0.51 | 2.53 MiB | 0.08 ms | -0.083 [-0.113, -0.054] |
| gaussian_rp | 256 | 0.602 | 0.731 | 0.64 | 5.06 MiB | 0.19 ms | -0.046 [-0.070, -0.023] |
| sparse_rp | 32 | 0.283 | 0.401 | 0.20 | 0.63 MiB | 0.07 ms | -0.365 [-0.413, -0.317] |
| sparse_rp | 64 | 0.457 | 0.597 | 0.35 | 1.27 MiB | 0.07 ms | -0.192 [-0.230, -0.154] |
| sparse_rp | 128 | 0.554 | 0.720 | 0.49 | 2.53 MiB | 0.07 ms | -0.094 [-0.123, -0.067] |
| sparse_rp | 256 | 0.597 | 0.745 | 0.62 | 5.06 MiB | 0.07 ms | -0.051 [-0.074, -0.031] |
| tfidf | - | 0.642 | 0.779 | 0.32 | 12.68 MiB | 2.44 ms | -0.007 [-0.048, +0.034] |

- Relevance is binarised: a graded qrel of 2 counts the same as 1.
- Projectors are fitted on a random 60% of the corpus and scored on every document, mirroring the chronological fit used on the live corpus.
