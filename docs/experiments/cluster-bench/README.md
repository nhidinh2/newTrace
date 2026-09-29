# Story clustering on human labels

A committed copy of one `make cluster-bench` run. Source directory:
`artifacts/cluster_bench/eng-2026-09-29T043344Z` (regenerated directories under
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

| System | title w | entity w | threshold | pairwise F1 | P | R | B-cubed F1 | stories predicted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NewsTrace, shipped | 0.20 | 0.10 | 0.88 | **10.2** | 99.7 | 5.3 | 33.4 | 3,215 |
| NewsTrace, threshold fitted on dev | 0.20 | 0.10 | 0.425 | **92.2** | 97.4 | 87.4 | 91.0 | 502 |
| NewsTrace, weights and threshold fitted on dev | 0.43 | 0.52 | 0.425 | **91.3** | 97.0 | 86.2 | 90.7 | 478 |
| CluStream (Aggarwal & Yu 2006), as published | — | — | — | 79.0 | 98.6 | 65.9 | — | — |
| Miranda et al., tokens+lemmas+entities, as published | — | — | — | 92.7 | 92.9 | 92.5 | — | — |
| Miranda et al., + timestamps, as published | — | — | — | 94.1 | 98.2 | 90.3 | — | — |

The metric is the paper's own: pairwise precision, recall and F1 over every
test document, so the published rows are directly comparable. Nothing is
fitted on test.

**The shipped threshold does not transfer.** At 0.88 the clusterer almost never
merges wrong stories (99.7% precision, a 0.4% merge-error rate) and almost never
merges right ones either: 92% of the true stories are split, into 3,215 pieces
for 222 stories.

**With one fitted number the hand-set blend is close to the published system.**
Refitting only the threshold on the earlier split gives 92.2 F1: above
CluStream's 79.0, level with Miranda et al.'s supervised ranker without
timestamps (92.7), and 1.9 below their best. NewsTrace's blend is
`cosine + 0.2·title + 0.1·entity`; time enters only as the 72-hour candidate
window.

**Learning the blend weights does not help.** `fit_weights` fits a logistic
regression on 151,739 dev pairs (every article against every story in its
window, with stories grown from the gold labels) and rewrites its boundary as
the same blend. It more than doubles both overlap weights (title 0.43, entity
0.52), then selects the same threshold, and scores 91.3 on test, 0.9 below the
hand-set weights. On dev the two blends are within 0.2 of each other at their
best. The hand-set weights stay.

**The threshold is not pinned down, and test F1 is sensitive to it.** The dev
sweep with the shipped weights is flat across a wide band, and not unimodal:

| threshold | 0.35 | 0.40 | 0.425 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev pairwise F1 | 84.5 | 86.3 | **86.8** | 86.5 | 85.2 | 86.6 | 86.0 | 84.4 | 80.3 | 52.1 |

Dev F1 stays between 85.1 and 86.8 from 0.40 to 0.60, but test F1 at
the two local peaks is 92.2 (0.425) and 82.8 (0.55). Pairwise F1 counts pairs,
so it is dominated by the largest stories (the biggest test story has 819
articles): one of them merged or split moves the score by points. Read the
headline row as "around 90", not as a precise gap to the published systems.

*Correction.* The first committed version of this run swept thresholds from
0.50 up only, selected 0.55 and reported 82.8 as the fitted result. The grid
now starts at 0.30 (`DEFAULT_GRID`), which is what found 0.425.

Full sweeps for both blends in [`results.md`](results.md).

## Why 0.425 is not now the shipped value

The two corpora embed different text. Benchmark articles carry the full body
(the embedder sees the headline plus roughly the first 200 words before
MiniLM's 256-word-piece cut-off); live articles carry a headline and an RSS
description, **median 169 characters** across the 6,320-article live corpus.
Short texts about the same event score higher cosine against each other than
long ones do, so the 0.88 tuned on live-like fixtures and the 0.425 fitted here
are each right for their own inputs. The finding that transfers is the shape:
the clusterer is precision-heavy and recall-limited, and a threshold tuned on
one text length is badly wrong on another.

## Caveats

- No confidence intervals. Pairwise metrics do not bootstrap cleanly over
  resampled stories, and the paper reports none to compare against. The
  threshold sensitivity above is the practical substitute: differences of a
  point or two between rows are not evidence of anything.
- One language. The dataset also has German and Spanish (`--language deu|spa`),
  but MiniLM is an English model; those runs would measure the embedder, not
  the clusterer.
- The published rows are copied from the paper, not re-run.

## Reproduce

```bash
make cluster-bench DOWNLOAD=1    # ~107 MB from Priberam's FTP, once
make cluster-bench               # offline thereafter; ~35 minutes on a laptop CPU
```
