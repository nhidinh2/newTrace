# Story clustering on human labels

A committed copy of one `make cluster-bench` run. Source directory:
`artifacts/cluster_bench/eng-2026-09-29T011355Z` (regenerated directories under
`artifacts/` are gitignored).

Until this run, clustering was measured only on the 135-article fixture corpus,
whose labels were written for this repository, and the live corpus has no
clustering score at all. This is the clustering counterpart of
[`../beir-scifact`](../beir-scifact): the production `OnlineClusterer`,
unchanged, replayed over a public news stream with human story labels.

**Dataset.** Miranda et al., *Multilingual Clustering of Streaming News*,
EMNLP 2018 ([paper](https://aclanthology.org/D18-1483),
[code](https://github.com/priberam/news-clustering)), built from the Event
Registry data of Rupnik et al. (2016). English only here:

| split | documents | stories | sources | span |
| --- | ---: | ---: | ---: | --- |
| `dataset.dev.json` (the paper's *train* split) | 12,233 | 593 | 2,651 | 2013-12 → 2014-11 |
| `dataset.test.json` | 8,726 | 222 | 1,935 | 2014-11 → 2015-08 |

The splits are chronological and do not overlap. Test counts match Table 1 of
the paper exactly.

## What it says

| System | threshold | pairwise F1 | P | R | B-cubed F1 | stories predicted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NewsTrace, shipped | 0.88 | **10.2** | 99.7 | 5.3 | 33.4 | 3,215 |
| NewsTrace, threshold fitted on dev | 0.55 | **82.8** | 97.7 | 71.9 | 86.1 | 748 |
| CluStream (Aggarwal & Yu 2006), as published | — | 79.0 | 98.6 | 65.9 | — | — |
| Miranda et al., tokens+lemmas+entities, as published | — | 92.7 | 92.9 | 92.5 | — | — |
| Miranda et al., + timestamps, as published | — | 94.1 | 98.2 | 90.3 | — | — |

The metric is the paper's own: pairwise precision, recall and F1 over every
test document, so the published rows are directly comparable.

**The shipped threshold does not transfer.** At 0.88 the clusterer almost never
merges wrong stories (99.7% precision, a 0.4% merge-error rate) and almost never
merges right ones either: 92% of the true stories are split, into 3,215 pieces
for 222 stories.

**With one fitted number it clears the classic baseline.** Refitting only the
threshold on the earlier split gives 82.8 F1, above CluStream's 79.0 at similar
precision. The distance to Miranda et al. is recall: 71.9 against 90.3. Their
system learns its feature weights with an SVM trained on that same earlier
split and uses timestamp features; NewsTrace's weights are hand-set
(`cosine + 0.2·title + 0.1·entity`) and time enters only as the 72-hour
candidate window.

The dev sweep is smooth and peaks broadly between 0.55 and 0.60 (86.6 → 86.0),
so the fitted value is not a knife-edge:

| threshold | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.875 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev pairwise F1 | 85.2 | **86.6** | 86.0 | 84.4 | 80.3 | 65.2 | 52.1 | 30.0 |

Full sweep in [`results.md`](results.md).

## Why 0.55 is not now the shipped value

The two corpora embed different text. Benchmark articles carry the full body
(the embedder sees the headline plus roughly the first 200 words before
MiniLM's 256-word-piece cut-off); live articles carry a headline and an RSS
description, **median 169 characters** across the 6,320-article live corpus.
Short texts about the same event score higher cosine against each other than
long ones do, so the 0.88 tuned on live-like fixtures and the 0.55 fitted here
are each right for their own inputs. The finding that transfers is the shape:
the clusterer is precision-heavy and recall-limited, and a threshold tuned on
one text length is badly wrong on another.

## Caveats

- No confidence intervals. Pairwise metrics do not bootstrap cleanly over
  resampled stories, and the paper reports none to compare against.
- One language. The dataset also has German and Spanish (`--language deu|spa`),
  but MiniLM is an English model; those runs would measure the embedder, not
  the clusterer.
- The published rows are copied from the paper, not re-run.

## Reproduce

```bash
make cluster-bench DOWNLOAD=1    # ~107 MB from Priberam's FTP, once
make cluster-bench               # offline thereafter; ~10 minutes on a laptop CPU
```
