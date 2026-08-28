# Model card — NewsTrace

## Overview

NewsTrace is a research system that groups news articles describing the same
event, tracks how a story develops, and produces summaries assembled from quoted
passages with links to the original reporting. It was built as an applied
MCS 549 project.

**It is not a fact checker.** It reports repetition, provenance, duplication and
source diversity. It does not judge truth, accuracy, bias, or publisher quality.

## Intended use

- Research and coursework on streaming clustering, dimensionality reduction and
  graph-based ranking over a live text stream.
- Following how coverage of an AI or technology event accumulates and changes.
- Seeing which claims appear across several independent domains, which rest on
  one source, and where sources cannot be reconciled.

## Out of scope

- Deciding whether a claim is true or false.
- Scoring publisher trustworthiness or political bias.
- Automated trading or any investment recommendation.
- Circumventing paywalls, robots.txt or publisher terms.
- Any consequential decision about a person or organisation.

## Components

| Component | Model / method | Notes |
| --- | --- | --- |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-d) | Optional extra. CPU by default; GPU detected, never required. |
| Embedding fallback | Deterministic feature hashing over word and 4-gram features | Used when the extra is absent. Weaker; carries its own tuned threshold. |
| Compression | TruncatedSVD, Gaussian RP, sparse RP | Fitted on the chronological fit period only; seeded. |
| Clustering | Single-pass windowed assignment; cosine boosted by title-token and entity overlap | Threshold tuned on fit+validation. |
| Deduplication | Canonical URL, normalised text/title hashes, title similarity, SimHash over excerpts | Duplicates preserved and linked, never deleted. |
| Claims | Rule-based sentence selection (entity / quantity / date / attribution) | Not semantic parsing. |
| Ranking | Exact cosine, TF-IDF, recency, duplicate penalty, personalized PageRank | No publisher-reputation term. |
| Summaries | Deterministic extractive; optional LLM behind a validated JSON schema | Default `LLM_PROVIDER=none`. |

## Training data

NewsTrace trains nothing. The embedding model is used as published. The only
fitted artefacts are the projectors and the clustering threshold, both fitted on
the fit/validation window of whatever corpus is ingested, with the seed and fit
metadata recorded in `ProjectionArtifact` and `EvaluationRun`.

## Evaluation

See [evaluation.md](evaluation.md). Two runs are committed.

**Synthetic fixtures** (135 articles, 41 stories, 26 held-out queries, curated
labels):

- Retrieval: 0.914 nDCG@10 full-dimensional; 0.920 for gaussian_rp at d=128
  (3× smaller index); TF-IDF 0.936.
- Clustering: 0.944 B-cubed F1 on the full-dimensional representation.
- Summaries: 1.000 citation coverage, 0.989 citation precision.
- PageRank: converges for 100% of queries, stationary mass within 2e-15 of 1;
  reranking *lowers* nDCG@10 to 0.881.

**Live corpus** (1,120 articles in a 30-day window of 3,191 from 292 domains,
23 queries, silver labels from the system's own clustering):

- Retrieval: 0.833 nDCG@10 full-dimensional; 0.875 for SVD at d=32 (12× smaller
  index, p95 latency 0.09 ms against 0.64 ms).
- **No configuration is statistically separable from the baseline**: every
  paired bootstrap interval contains zero.
- Summaries: 1.000 citation coverage and precision over the 8 of 10 largest
  stories that produced statements; source-domain diversity 1.5.
- PageRank: reranking *raises* nDCG@10 to 0.874 — the opposite of the fixture
  result, so the claim is unsettled.

The fixture numbers come from generated text and do not predict real-world
performance. The live numbers come from real text but from a convenience sample
of feeds, with labels the system produced itself, and a query set of 23. Quote
either only with its caveat attached.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Readers treat repetition as verification | Every surface states that agreement is not truth; syndicated copies are excluded from independent-source counts and labelled in the UI |
| Generated prose replaces the source | Source links are always shown; summaries are quoted passages, not generated text; the LLM path drops any statement whose citations do not resolve |
| A wrong cluster creates a false disagreement | Conflict rules require shared entities plus either a denied value or an identical sentence frame; the merge-error rate is measured and published |
| "Earliest report" read as "originator" | The timeline note and API response both state that the dataset's earliest article is not proof of origination |
| Over-collection of publisher content | Metadata and short excerpts only; full-text extraction off by default and allowlist-gated |
| Silent quality regression | 211 automated tests, chronological split with a leakage assertion, and every reported number generated from a saved `metrics.json` |

## Maintenance

Re-tune the clustering threshold (`newstrace.evaluation.tuning`) whenever the
embedding backend, corpus or topic mix changes — the shipped values were selected
on this fixture corpus and will not transfer. Re-run `make experiment` to
regenerate every published number.

## Contact

This is a coursework and portfolio project. Report issues in the repository.
