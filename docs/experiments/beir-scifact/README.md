# BEIR / SciFact

A committed copy of one `make beir DATASET=scifact` run. Source directory:
`artifacts/beir/scifact-2026-09-22T055023Z` (regenerated directories under
`artifacts/` are gitignored).

This run exists to answer the open question left by
[`../live-30d`](../live-30d): **that sweep could not separate the
representations, and 23 silver-labelled queries were never going to.** SciFact
supplies 300 queries with human relevance judgments over 5,183 documents, and
the pipeline is representation-agnostic, so the identical projectors are
scored there.

## What it says

| Retrieval | dim | nDCG@10 | Recall@10 | top-10 overlap | Index | delta nDCG vs full (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full embeddings | 384 | 0.648 | 0.788 | — | 7.59 MiB | baseline |
| Truncated SVD | 256 | 0.647 | 0.790 | 0.96 | 5.06 MiB | −0.002 [−0.007, +0.003] |
| Truncated SVD | 128 | 0.619 | 0.770 | 0.81 | 2.53 MiB | −0.030 [−0.046, −0.015] |
| Truncated SVD | 64 | 0.559 | 0.703 | 0.65 | 1.27 MiB | −0.089 [−0.119, −0.061] |
| Truncated SVD | 32 | 0.461 | 0.606 | 0.47 | 0.63 MiB | −0.187 [−0.229, −0.147] |
| Gaussian RP | 32 | 0.321 | 0.451 | 0.22 | 0.63 MiB | −0.327 [−0.374, −0.281] |
| Sparse RP | 32 | 0.283 | 0.401 | 0.20 | 0.63 MiB | −0.365 [−0.413, −0.317] |
| TF-IDF, lexical | — | 0.642 | 0.779 | 0.32 | 12.68 MiB | −0.007 [−0.048, +0.034] |

Every dimension except 256 is **worse than the baseline by an interval that
excludes zero**. SVD@32 -- the configuration the live sweep could not
distinguish from full -- loses 0.19 nDCG@10 here, and the interval is nowhere
near zero.

The honest reading is that the live result was a statement about the
judgments. On a corpus where relevance is known and there are 300 queries
instead of 23, 12x compression is not free: it costs about 29% of nDCG@10, and
only 1.5x compression (256 dimensions) is indistinguishable from the
baseline. Nothing about the geometry changed between the two runs; the
measuring instrument did.

The random projections are far worse than SVD at every dimension here, which
the live corpus also could not show. SciFact is a paraphrase-heavy corpus
where the ranking depends on structure that a data-independent projection
destroys and a data-dependent one keeps.

## Caveats

- Relevance is binarised: a graded qrel of 2 counts the same as 1.
- Projectors are fitted on a random 60% of the corpus and scored on all of it,
  mirroring the chronological fit used on the live corpus. SciFact has no time
  axis to split on.
- SciFact is scientific claim verification, not news. TF-IDF matching the
  dense baseline here (−0.007 [−0.048, +0.034]) is a known property of the
  dataset, and it echoes what the live corpus showed. It is not evidence that
  embeddings are useless for news.
- One embedder (`all-MiniLM-L6-v2`), one dataset, one seed. `nfcorpus` and
  `arguana` are one flag away and would make this a pattern rather than a
  point.

## Reproducing

```bash
scripts/run.sh python scripts/run_beir.py --download --dataset scifact
scripts/run.sh python scripts/run_beir.py --dataset scifact --dimensions 32,64,128,256
```
