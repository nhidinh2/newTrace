# Approximate retrieval sweep

6320 vectors, 384-d, 60 queries (judgments: silver_from_full_dim_clustering_excluding_near_duplicates).

| Index | dim | probe | recall@10 vs exact | nDCG@10 | Index | p95 | corpus covered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact | 384 | - | 1.00 | 0.487 | 9.26 MiB | 0.65 ms | 100% |
| exact:svd@32 | 32 | - | 0.54 | 0.373 | 0.77 MiB | 0.17 ms | 100% |
| exact:gaussian_rp@32 | 32 | - | 0.36 | 0.369 | 0.77 MiB | 0.17 ms | 100% |
| exact:sparse_rp@32 | 32 | - | 0.40 | 0.372 | 0.77 MiB | 0.16 ms | 100% |
| ivf64_flat | 384 | 1 | 0.68 | 0.348 | 9.35 MiB | 0.13 ms | 100% |
| ivf64_flat | 384 | 4 | 0.82 | 0.392 | 9.35 MiB | 0.25 ms | 100% |
| ivf64_flat | 384 | 8 | 0.89 | 0.441 | 9.35 MiB | 0.34 ms | 100% |
| ivf64_flat | 384 | 16 | 0.94 | 0.457 | 9.35 MiB | 0.51 ms | 100% |
| ivf64_pq48x8 | 384 | 1 | 0.61 | 0.344 | 0.76 MiB | 0.40 ms | 100% |
| ivf64_pq48x8 | 384 | 4 | 0.69 | 0.388 | 0.76 MiB | 1.24 ms | 100% |
| ivf64_pq48x8 | 384 | 8 | 0.73 | 0.440 | 0.76 MiB | 1.84 ms | 100% |
| ivf64_pq48x8 | 384 | 16 | 0.75 | 0.454 | 0.76 MiB | 3.56 ms | 100% |
| ivf64_pq96x8 | 384 | 1 | 0.66 | 0.358 | 1.05 MiB | 0.70 ms | 100% |
| ivf64_pq96x8 | 384 | 4 | 0.79 | 0.402 | 1.05 MiB | 2.05 ms | 100% |
| ivf64_pq96x8 | 384 | 8 | 0.83 | 0.450 | 1.05 MiB | 3.87 ms | 100% |
| ivf64_pq96x8 | 384 | 16 | 0.86 | 0.469 | 1.05 MiB | 6.68 ms | 100% |
