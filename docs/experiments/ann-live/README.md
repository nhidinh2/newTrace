# Approximate retrieval on the live corpus

A committed copy of one `make ann SINCE=30` run over the live corpus (6,320
articles, 384-d MiniLM embeddings, 60 silver-labelled queries). Source
directory: `artifacts/ann/2026-09-22T054623Z`.

The compression sweep asks what happens when the vectors get smaller. This
asks what happens when the scan stops being exhaustive -- the other way to buy
the same memory -- and reports both against the same exact baseline.

## What it says

At roughly equal memory (about 1 MiB, a 9x reduction), quantising the scan
keeps far more of the exact ranking than reducing the dimension does:

| Index | recall@10 vs exact | nDCG@10 | Index memory | p95 |
| --- | ---: | ---: | ---: | ---: |
| Exact, 384-d | 1.00 | 0.487 | 9.26 MiB | 0.65 ms |
| IVF-PQ, 96 subvectors, 16 probes | 0.86 | 0.469 | 1.05 MiB | 6.68 ms |
| IVF-PQ, 48 subvectors, 16 probes | 0.75 | 0.454 | 0.76 MiB | 3.56 ms |
| Exact SVD@32 | 0.54 | 0.373 | 0.77 MiB | 0.17 ms |
| Exact Gaussian RP@32 | 0.36 | 0.369 | 0.77 MiB | 0.17 ms |

**0.86 against 0.54 recall at the same cost.** The two techniques are not
interchangeable: PQ throws away precision *within* a preserved geometry, while
a 32-d projection throws away the geometry.

The latency column is the other half. IVF-PQ is *slower* than the exhaustive
scan at this corpus size -- 6.68 ms against 0.65 ms -- because at 6,320
vectors a BLAS matrix-vector product beats a Python-driven quantised lookup,
and the compressed exact indexes are the fastest thing here. At this scale ANN
buys memory, not latency. That crossover moves with corpus size, and this run
does not locate it.

`recall@10 vs exact` is measured against the exhaustive scan over the same
vectors, so it needs no relevance judgments. That matters: the nDCG column
uses the same 60 silver queries the live sweep uses, and inherits every caveat
in [`../../limitations.md`](../../limitations.md). The recall column does not.

## Caveats

- One IVF configuration (64 lists) and one seed. `--lists` sweeps it.
- The index is rebuilt from scratch per run; nothing in the serving path uses
  it yet. `retrieval/index.py` still serves exact search.
- Silver judgments, so the nDCG column separates nothing on its own.

## Reproducing

```bash
make experiment SINCE=30 QUERIES=200   # refresh the compressed representations first
make ann SINCE=30
```
