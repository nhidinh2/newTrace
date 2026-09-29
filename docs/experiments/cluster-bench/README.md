# Story clustering on human labels

A committed copy of one `make cluster-bench` run. Source directory:
`artifacts/cluster_bench/eng-2026-09-29T053547Z` (regenerated directories under
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

| System | title w | entity w | time w/day | threshold | pairwise F1 | P | R | B-cubed F1 | stories predicted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NewsTrace, shipped | 0.20 | 0.10 | 0 | 0.88 | **10.2** | 99.7 | 5.3 | 33.4 | 3,215 |
| NewsTrace, threshold fitted on dev | 0.20 | 0.10 | 0 | 0.425 | **92.2** | 97.4 | 87.4 | 91.0 | 502 |
| NewsTrace, + fitted time penalty | 0.20 | 0.10 | 0.218 | 0.55 | **83.8** | 97.9 | 73.2 | 86.1 | 880 |
| NewsTrace, all weights fitted, with time | 0.52 | 0.64 | 0.218 | 0.40 | **96.9** | 97.5 | 96.2 | 92.6 | 534 |
| CluStream (Aggarwal & Yu 2006), as published | — | — | — | — | 79.0 | 98.6 | 65.9 | — | — |
| Miranda et al., tokens+lemmas+entities, as published | — | — | — | — | 92.7 | 92.9 | 92.5 | — | — |
| Miranda et al., + timestamps, as published | — | — | — | — | 94.1 | 98.2 | 90.3 | — | — |

The metric is the paper's own: pairwise precision, recall and F1 over every
test document, so the published rows are directly comparable. Weights and
thresholds are fitted on dev only. The two middle rows disagree by 13 points
for a reason that has nothing to do with their weights; read the threshold
section before comparing them.

**The shipped threshold does not transfer.** At 0.88 the clusterer almost never
merges wrong stories (99.7% precision, a 0.4% merge-error rate) and almost never
merges right ones either: 92% of the true stories are split, into 3,215 pieces
for 222 stories.

**With one fitted number the hand-set blend is close to the published system.**
Refitting only the threshold on dev gives 92.2 F1: above CluStream's 79.0,
level with Miranda et al.'s supervised ranker without timestamps (92.7), and
1.9 below their best. The blend is `cosine + 0.2·title + 0.1·entity`.

**Time is the signal that helps.** `combined_score` now takes an optional
penalty per day between an article and the nearest end of a story's span
(`cluster_time_gap_weight`, 0 by default, so shipped behaviour is unchanged).
`fit_weights` fits it with the other weights: a logistic regression over
151,739 dev pairs (every article against every story in its window, stories
grown from the gold labels), its boundary rewritten as the same blend. It puts
0.22 on each day of gap. The best dev F1 rises from 86.8 to 88.6 with the
hand-set overlap weights, and to 88.3 with all weights fitted.

**Refitting the overlap weights does not help.** Without time, the fit more
than doubles them (title 0.43, entity 0.52) and scored 91.3 on test against
92.2 for the hand-set blend, in the run committed before this one. With time
they rise further (0.52, 0.64) and match, but do not beat, the hand-set blend
plus time in the sensitivity table below. The hand-set overlap weights stay.

## The threshold is the weak point

Dev F1 is flat across a wide band of thresholds, so dev cannot choose between
them, and test F1 is not flat across the same band. Dev sweeps:

| threshold | 0.30 | 0.35 | 0.40 | 0.425 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hand-set, no time | 76.7 | 84.5 | 86.3 | **86.8** | 86.5 | 85.2 | 86.6 | 86.0 | 84.4 | 80.3 |
| hand-set + time | 86.8 | 88.0 | 88.1 | 88.2 | 87.2 | 85.5 | **88.6** | 86.5 | 80.5 | 76.1 |
| all fitted + time | 87.5 | 87.9 | **88.3** | 88.2 | 86.6 | 83.4 | 84.7 | 82.8 | 76.9 | 69.9 |

To see what that costs, each setting was also scored on **test** at every
threshold from 0.35 to 0.60 ([`test_threshold_sensitivity.json`](test_threshold_sensitivity.json)).
This is a diagnostic, not a selection: nothing in the table above was chosen
from it.

| threshold | 0.35 | 0.375 | 0.40 | 0.425 | 0.45 | 0.475 | 0.50 | 0.55 | 0.60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hand-set, no time | 90.0 | 90.3 | 92.3 | 92.2 | 91.9 | 90.1 | 89.9 | 82.8 | 67.4 |
| hand-set + time | **96.8** | **96.9** | **96.8** | **96.9** | 95.2 | 87.4 | 86.8 | 83.8 | 70.9 |
| all fitted + time | 96.0 | 96.8 | 96.9 | 96.8 | 96.8 | 84.6 | 77.2 | 71.5 | 61.8 |

- **Time adds 4 to 7 points at every threshold from 0.35 to 0.45**, where all
  three settings are at their best. Its 83.8 in the headline table is dev
  selecting 0.55, on the far side of a collapse that starts near 0.47 on test
  and does not show on dev.
- **The collapse is a few large stories merging.** Pairwise F1 counts pairs,
  so the largest stories dominate it (the biggest test story has 819
  articles); one wrong merge of two of them costs points.
- **The honest claim is the difference, not the level.** "About 97, above the
  published 94.1" rests on a threshold band identified by looking at test.
  "Time adds 4 to 7 points across the band where the clusterer works" does not
  depend on which threshold in it is picked.

*Correction.* The first committed version of this run swept thresholds from
0.50 up only, selected 0.55 and reported 82.8 as the fitted result. The grid
now starts at 0.30 (`DEFAULT_GRID`), which is what found 0.425.

Full dev sweeps for every setting in [`results.md`](results.md).

## Why none of this is now the shipped setting

The two corpora embed different text. Benchmark articles carry the full body
(the embedder sees the headline plus roughly the first 200 words before
MiniLM's 256-word-piece cut-off); live articles carry a headline and an RSS
description, **median 169 characters** across the 6,320-article live corpus.
Short texts about the same event score higher cosine against each other than
long ones do, so the 0.88 tuned on live-like fixtures and the 0.425 fitted here
are each right for their own inputs. The finding that transfers is the shape:
the clusterer is precision-heavy and recall-limited, and a threshold tuned on
one text length is badly wrong on another.

The time penalty stays off for the same reason. Its size is relative to the
cosine scale, which differs between the two kinds of text, and live stories
have their own rhythm. Whether it helps live clustering, and at what threshold,
needs live articles labelled by story.

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
make cluster-bench               # offline thereafter; ~50 minutes on a laptop CPU
```
