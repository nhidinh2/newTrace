# Evaluation

Every number on this page is a measurement from one of two committed runs.
Nothing is estimated. Run the sweep yourself and the numbers will move with your
machine and your corpus.

| run | corpus | labels | file |
| --- | --- | --- | --- |
| **baseline** | 135 synthetic fixture articles | curated fixture story labels | [`docs/experiments/baseline/metrics.json`](experiments/baseline/metrics.json) |
| **live-30d** | 1,120 real articles (30-day window of a 3,191-article corpus, 292 domains) | silver labels from the system's own clustering | [`docs/experiments/live-30d/metrics.json`](experiments/live-30d/metrics.json) |

Sections below are the baseline run unless they say otherwise. The fixture
corpus is reproducible offline and carries labels a human wrote; the live corpus
is real but can only be labelled by the system itself. Neither alone is enough,
which is why both are kept.

## Protocol

**Chronological split.** Articles are ordered by publication time and cut
60 / 20 / 20 into fit, validation and test. Projectors and the clustering
threshold are fitted on the fit (and validation) period only;
`ChronologicalSplit.assert_no_leakage()` runs before every experiment and the
test asserts it too.

On the fixture corpus that is 81 / 27 / 27 articles, with a fit cutoff of
`2026-08-22T05:30:00Z`.

**The retrieval task.** For each held-out article, a short keyword query is built
from its headline's content words. The system must return *the other* articles
covering the same event: the source article is removed from both the candidate
pool and the relevant set. The pool is the whole corpus (135 articles), because
a live index holds everything ingested so far.

Using the full headline as the query and letting the article match itself makes
every method score identically — the first version of this sweep did exactly
that and returned 0.952 nDCG@10 for all twelve configurations. The task above is
what produces a signal.

**Relevance judgments** come, in order of preference, from reviewer labels
(`data/labels/retrieval_judgments.csv`), the synthetic fixture story labels, or
silver labels derived from the system's own clustering. The report records which
was used. This run used `synthetic_fixture_labels`.

## Retrieval quality versus cost

Baseline embedding: `all-MiniLM-L6-v2`, d=384, 27 queries over 135 articles.

| method | d | Recall@5 | Recall@10 | nDCG@10 | index MiB | mean abs cosine error |
| --- | --- | --- | --- | --- | --- | --- |
| full | 384 | 1.000 | 1.000 | **0.914** | 0.198 | 0 |
| svd | 32 | 0.952 | 0.990 | 0.866 | 0.016 | 0.055 |
| svd | 64 | 0.981 | 1.000 | 0.903 | 0.033 | 0.030 |
| svd | 80 | 0.990 | 1.000 | 0.904 | 0.041 | 0.025 |
| gaussian_rp | 32 | 0.942 | 0.962 | 0.864 | 0.016 | 0.143 |
| gaussian_rp | 64 | 0.962 | 0.962 | 0.868 | 0.033 | 0.092 |
| gaussian_rp | 128 | 1.000 | 1.000 | **0.920** | 0.066 | 0.060 |
| gaussian_rp | 256 | 1.000 | 1.000 | 0.914 | 0.132 | 0.045 |
| sparse_rp | 32 | 0.721 | 0.856 | 0.669 | 0.016 | 0.141 |
| sparse_rp | 64 | 0.962 | 0.990 | 0.893 | 0.033 | 0.087 |
| sparse_rp | 128 | 1.000 | 1.000 | 0.908 | 0.066 | 0.066 |
| sparse_rp | 256 | 1.000 | 1.000 | **0.920** | 0.132 | 0.046 |

Against the README's targets:

- **"Preserve ≥95% of full Recall@10 at a lower dimension."** Met. SVD at d=64
  and both projections at d=128 reach 100% of baseline Recall@10.
- **"Reduce embedding storage by ≥3×."** Met. d=128 is a 3.0× reduction and
  d=64 is 6.0×, at 98.8% (SVD d=64) of baseline nDCG@10.
- **"Improve p95 exact-search latency."** **Not demonstrated on this corpus.**
  With 135 vectors every configuration sits at 0.03–0.06 ms and the differences
  are scheduler noise, even averaged over 5 repeats per query. It *is*
  demonstrated on the live corpus below, where 1,120 vectors are enough for the
  measurement to mean something: 0.64 ms at d=384 against 0.09 ms at d=32.

**Geometry.** Distance and cosine distortion fall monotonically with dimension
for both random projections — the JL-style behaviour the course predicts — and
SVD distorts least at equal dimension because it keeps the directions of
greatest variance rather than a random subspace. Sparse RP at d=32 is the one
configuration that breaks down badly (0.669 nDCG@10): at that width its sparse
matrix retains too little of the original geometry.

**TruncatedSVD could not be swept past d=80.** It is capped at
`n_fit_samples - 1`, and the fit period holds 81 articles. The requested d=128
was clamped and d=256 collapsed onto the same fit, so it was skipped rather than
reported twice. The report's "What was skipped" section says so. The live run's
672-article fit period sweeps the full 32/64/128/256 ladder with nothing
clamped, which is one concrete reason to run on more than fixtures.

## The live corpus

3,191 articles ingested from GDELT DOC 2.0 and 36 public publisher feeds across
292 domains; the sweep evaluates the last 30 days (1,120 articles) with
`--since-days 30`. Several feeds are vendor blog archives reaching back to 2015,
and without a window a decade-old post would sit in the fit period while this
morning's wire story sat in the test period. Split: 672 / 224 / 224.

| method | d | nDCG@10 | 95% CI | Recall@10 | index MiB | p95 ms |
| --- | --- | --- | --- | --- | --- | --- |
| full | 384 | 0.833 | [0.702, 0.952] | 0.913 | 1.641 | 0.64 |
| svd | 32 | **0.875** | [0.745, 0.984] | 0.913 | 0.137 | **0.09** |
| svd | 64 | 0.849 | [0.713, 0.968] | 0.913 | 0.273 | 0.12 |
| svd | 128 | 0.849 | [0.713, 0.968] | 0.913 | 0.547 | 0.16 |
| svd | 256 | 0.849 | [0.713, 0.968] | 0.913 | 1.094 | 0.21 |
| gaussian_rp | 32 | 0.843 | [0.707, 0.952] | 0.913 | 0.137 | 0.08 |
| gaussian_rp | 64 | 0.833 | [0.702, 0.952] | 0.913 | 0.273 | 0.10 |
| gaussian_rp | 128 | 0.849 | [0.713, 0.968] | 0.913 | 0.547 | 0.18 |
| gaussian_rp | 256 | 0.833 | [0.702, 0.952] | 0.913 | 1.094 | 0.23 |
| sparse_rp | 32 | 0.817 | [0.682, 0.936] | 0.913 | 0.137 | 0.10 |
| sparse_rp | 64 | 0.821 | [0.678, 0.941] | 0.913 | 0.273 | 0.11 |
| sparse_rp | 128 | 0.849 | [0.713, 0.968] | 0.913 | 0.547 | 0.19 |
| sparse_rp | 256 | 0.849 | [0.713, 0.968] | 0.913 | 1.094 | 0.21 |

Read the confidence intervals before the point estimates. **Every paired
bootstrap interval against the full baseline contains zero**, across all twelve
configurations. SVD at d=32 scoring above full is not a finding that compression
improves retrieval; it is 23 queries failing to separate thirteen systems. What
the run does establish is the *cost* side, which needs no labels: a 12× smaller
index and 7× lower p95 latency, measured rather than argued.

**Duplicates are excluded from the relevant set.** A syndicated copy is not
independent coverage (README rule 9), so retrieving one earns no credit. With
copies left in, all thirteen configurations scored an identical 0.917 — the task
had become "find the near-identical article", which no representation can fail.
That degeneracy is why the exclusion exists, and it is the single largest effect
seen in this project so far.

**Only 23 queries survived.** The window holds 59 multi-article stories, of which
32 have two or more non-duplicate articles, and queries are drawn from the
224-article test period alone. Genuine multi-source events are rare in a
one-month feed sample; this is the binding constraint on the live evaluation,
not compute.

**PageRank reranking helps here and hurt on the fixtures**: nDCG@10 0.874 against
the 0.833 cosine baseline, the reverse of the fixture result (0.881 against
0.914). The live graph holds 8,367 nodes and 20,518 edges over real syndication
patterns, in 84 components with 93.9% of nodes in the largest; the fixture graph
is 575 nodes of generated text in one component. Convergence is sound in both:
100% of queries, mean 115.8 iterations, stationary mass error 9.8e-15. Two
corpora disagreeing about a reranker is a reason to withhold the claim, not to
pick the flattering half.

**Summaries on real text.** Citation coverage stays at 1.000 (statement-weighted
and per story) with citation precision 1.000, across the 8 of 10 largest stories
that produced statements at all; the other 2 produced none, which is a coverage
question and not a citation failure. Source-domain diversity drops to 1.5 from
4.0 on the fixtures — most live stories are single-outlet.

## Statistical uncertainty

Both reports carry a percentile bootstrap over the per-query file: 2,000
seeded resamples per metric, plus a **paired** bootstrap of the per-query
difference against the full-dimensional baseline. Pairing matters, because the
same queries are easy or hard for every method; two independent intervals would
be far wider and would hide differences that pairing exposes.

On the fixture corpus the paired intervals do separate some configurations:
sparse_rp at d=32 loses 0.251 nDCG@10 with an interval of [-0.382, -0.119], and
SVD at d=32 and d=64 sit clearly below full. On the live corpus nothing
separates from full at all.

A bootstrap interval describes sampling variability in the query set it was
computed on. It cannot repair a biased corpus, silver labels, or a query set too
small to represent the task — and on both corpora the query set is small.

## Lexical baseline

TF-IDF scores **0.9364 nDCG@10** — *above* the full-dimensional dense baseline
(0.914). This is an honest negative result for the dense pipeline on this corpus
and should be read carefully: the queries are content words lifted from
headlines, and same-event articles in the fixtures share proper nouns, which is
exactly the regime where lexical matching is strongest. It is evidence that the
fixture corpus is lexically easy, not that embeddings are useless. A corpus with
more paraphrase and less name overlap would be needed to separate them.

## Clustering

Threshold tuned on fit+validation only (`newstrace.evaluation.tuning`), selecting
**0.88** for the sentence-transformer backend and **0.74** for the hashing
fallback. Cosine similarity is not comparable across backends, so they carry
separate values.

Held-out clustering quality, full-dimensional representation:

| metric | value |
| --- | --- |
| B-cubed F1 | 0.944 |
| pairwise F1 | 0.864 (whole corpus) |
| ARI | 0.860 (whole corpus) |
| fragmentation rate | 0.098 |
| merge error rate | 0.095 |
| clusters predicted / true | 42 / 41 |

Compressed representations lose little: gaussian_rp at d=256 matches the
full-dimensional B-cubed F1 to three decimals, and even d=32 stays near 0.887.

## Graph and random walks

The graph over the 135-article corpus has **575 nodes and 2,128 edges** in a
single connected component.

Personalized PageRank converges for **100%** of queries in **112.7 iterations**
on average, with a maximum stationary-mass error of **2.0e-15** — the
distribution sums to one.

Damping sensitivity (top-10 overlap is against the damping=0.5 ranking):

| damping | iterations | top-10 overlap with first |
| --- | --- | --- |
| 0.50 | 27 | — |
| 0.70 | 51 | 0.70 |
| 0.85 | 109 | 0.40 |
| 0.95 | 315 | 0.00 |

Iterations grow sharply with damping, as the theory says, and the ranking is
genuinely sensitive to it: at 0.95 the top ten shares nothing with the top ten at
0.5. Any PageRank result is a claim about a *specific* damping factor.

**PPR reranking did not help here**: 0.881 nDCG@10 against 0.914 for cosine
alone. On a corpus this size, graph centrality mostly re-surfaces well-connected
hubs, which is not the same as relevance. Reported as measured.

## Summaries and evidence

Ten largest stories, deterministic extractive summarizer:

| metric | value |
| --- | --- |
| citation coverage | **1.000** |
| citation precision | 0.989 |
| unsupported-claim rate | 0.011 |
| redundancy rate | 0.000 |
| source-domain diversity | 4.0 |
| statements per story | 8.1 |
| single-source statements | 7.0 |
| disputed statements | 0.2 |

Citation coverage of 1.000 is by construction: `drop_uncited` removes any
statement whose evidence ids do not resolve, and evidence ids are per
*(article, excerpt)* rather than per article, so a statement always points at
the exact passage it was lifted from. Citation precision below 1.0 comes from
cross-source disagreement statements, which quote two passages and therefore do
not match either one word for word.

An earlier build cited evidence per *article*, which scored 0.52 citation
precision and 0.36 redundancy. Both were real defects, found by these metrics.

## Ingestion throughput

Fixture ingestion of 270 records (135 articles seen twice, via the GDELT-shaped
payload and the feed): **666 articles/second** end to end, including
normalisation and duplicate detection. Per-article streaming cluster-update
latency is logged by `pipeline.index_articles` on every run.

Live ingestion is bounded by the network and by publisher rate limits, not by
the pipeline:

| run | source | records | seconds | records/s |
| --- | --- | --- | --- | --- |
| 1 | rss (12 feeds) | 332 | 21.7 | 15.3 |
| 2 | gdelt (1 topic) | 250 | 18.1 | 13.8 |
| 3 | gdelt (windowed sweep) | 0 | 55.3 | — (throttled, run recorded `failed`) |
| 4 | rss (36 feeds) | 2,942 | 138.0 | 21.3 |

Indexing alone — embed, assign, extract claims — ran at **118 articles/second**
for the 2,609 new articles of run 4, with p50 cluster-update latency 1.1 ms and
p95 18.5 ms. The gap between 21 and 118 is entirely HTTP: 36 feeds fetched
politely with a per-host interval.

Duplicate detection no longer scales with the corpus. It used to compare each
incoming article against every non-duplicate article in a 96-hour window,
capped at 2,000 rows, recomputing a 64-bit SimHash for both sides of every
pair. It now retrieves candidates by shared blocking key. Measured on the
6,320-article live corpus, over a 250-article sample:

| | Window scan | Blocking |
| --- | ---: | ---: |
| Candidates per article | 644 | 25 |
| Scoring time per article | 20.8 ms | 1.9 ms |
| Verdicts that differ | — | 0 |

The scoring-time column compares like with like: both sides used the stored
SimHash. Against the original code, which hashed both excerpts per pair at
0.57 ms a pair, the old path cost hundreds of milliseconds an article. The
throughput figures above predate the change and are unaffected by it -- they
were measured on 135 articles, where the window scan had almost nothing to
scan. The point of the change is what happens at 6,320 and beyond.

Run 3 is kept in the table deliberately. GDELT answers 429 with "limit requests
to one every 5 seconds" and applies a longer-term per-IP allowance on top; the
windowed sweep exhausted its retries, the run was recorded as `failed`, and the
stored corpus was untouched. That is the acceptance criterion for Milestone 1
("live ingestion failure does not corrupt the database") being exercised for
real rather than in a test.

## Approximate retrieval

`scripts/run_ann.py` measures the other way of buying memory and latency: an
inverted file over the full-dimensional vectors, with or without product
quantisation, against the exhaustive scan over the same vectors.

It reports two different things, deliberately:

- **recall@10 against exact retrieval** -- of the ten documents the exhaustive
  scan returns, how many does the approximate index return. This needs no
  relevance judgments and is a property of the index alone, which is what
  makes it usable on this corpus at all.
- **nDCG@10 against the silver judgments** -- the same weak signal the
  compression sweep uses, reported so the two arms can be compared, and read
  with the same caution.

Every row carries the fraction of the corpus its vectors cover. A compressed
representation stored by an earlier sweep covers only the articles that
existed then, and scoring it against an exact baseline over the whole corpus
reads as a quality collapse when it is really a stale index; re-run
`scripts/run_experiments.py` before comparing those rows.

```bash
make ann                       # defaults: 64 lists, probes 1/4/8/16, PQ 0/48/96
python scripts/run_ann.py --lists 128 --probes 1,8,32 --subvectors 96 --since-days 30
```

## Public benchmark (BEIR)

The live sweep's conclusion is that 23 silver queries cannot separate the
representations. `scripts/run_beir.py` runs the identical projectors on a
dataset that has human judgments and enough queries for a paired interval to
mean something:

```bash
python scripts/run_beir.py --download --dataset scifact
python scripts/run_beir.py --dataset scifact --dimensions 32,64,128,256
```

The output table is the same shape as the live one -- nDCG@10, Recall@10,
top-10 overlap with the full-dimensional baseline, index memory, p95 -- plus a
paired bootstrap interval on the nDCG delta against `full`. Projectors are
fitted on a random 60% of the corpus and scored on all of it, mirroring the
chronological fit used on the live corpus; relevance is binarised.

Nothing in this section has a committed run. The harness is tested on a
synthetic dataset in `tests/unit/test_beir.py`; the numbers are for whoever
runs it.

## Reproducing

```bash
make demo          # ingest fixtures, embed, cluster, extract claims
make experiment    # dimension sweep + graph analysis + report
make report        # re-render report.md from the saved metrics.json
```

The live run, against a separate database so the fixture demo stays intact:

```bash
export NEWSTRACE_DATABASE_URL=sqlite:///./newstrace_live.db
export NEWSTRACE_TOPICS_FILE=configs/topics.live.yaml
scripts/run.sh alembic upgrade head
scripts/run.sh newstrace ingest --source rss
scripts/run.sh newstrace ingest --source gdelt --timespan 7d --windows 7
scripts/run.sh python scripts/run_experiments.py --since-days 30 --max-queries 200 --seed 549
```

Seeded runs reproduce exactly on a *fixed* corpus. The live corpus is not
fixed — the feeds move — so re-ingesting will not reproduce the live numbers
above, only their shape.

Each run writes `config.json`, `environment.json` (Python, package versions,
platform, seed, git commit), `metrics.json`, per-query and per-configuration
tables, three plots and `report.md`. `report.md` is rendered *from*
`metrics.json`, so the prose cannot drift from the measurements.

## What these numbers do not show

The fixture test period holds 27 articles in 10 stories, and the live run scored
23 queries. Both query sets are small enough that the bootstrap intervals are
wide, and on the live corpus nothing separates from the baseline at all — the
correct conclusion there is "not measured", not "no difference".

Neither corpus has human relevance judgments. The fixture labels were written
alongside the fixtures; the live labels come from the system's own clustering
and therefore flatter the representation that produced them. Templates for real
labelling are in `data/labels/`, and closing that gap would do more for these
results than any additional method.

Clustering quality is only measured on the fixtures: the live corpus has no
labels to measure it against, so the live report's clustering table is empty
rather than filled with the system grading its own homework.

See [limitations](limitations.md) for what the data itself does not cover.
