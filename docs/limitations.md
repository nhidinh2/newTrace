# Limitations

NewsTrace reports **repetition, provenance, duplication and source diversity**.
It is not a fact checker. Nothing in this system establishes that a claim is
true, or that a publisher is reliable, and it must not be presented as if it
does.

## What the system cannot tell you

**Source counts do not establish truth.** Five outlets reporting the same figure
means five outlets reported it. It does not make the figure correct, and it is
routine for many outlets to repeat one wire story or one press release.

**Similar wording can indicate copying, not corroboration.** NewsTrace marks
verbatim and near-verbatim copies and excludes them from independent-source
counts, but partial rewrites of a single source will still read as independent
coverage. Independence is inferred from text and domain, and both can be wrong.

**Publication timestamps do not establish who reported first.** The timeline
labels the earliest report *in this dataset*. GDELT's `seendate` is when GDELT
saw the article, not when the publisher released it; feeds backdate and
re-timestamp; and the true first report may never have been ingested. The API
and UI both say so wherever the timeline appears.

**Disagreement detection is narrow and conservative.** Only two patterns are
reported: one passage denying a value another asserts, and two passages with the
same sentence frame giving different values for the same unit. Contradictions
expressed through framing, omission, or context are invisible to it. Silence is
not agreement.

**Merge errors cascade into false disagreements.** When the clusterer wrongly
merges two events, claims from both land in one story and their unrelated figures
can look like a contradiction. The measured merge-error rate on the fixtures is
0.095, so this happens.

**"Entities" are capitalised word runs, not named entities.** No NER model is
used. `extract_entities` drops calendar words and single capitalised words that
only ever appear at a sentence start, but it will still miss entities and invent
some. It is a feature for overlap scoring, not a knowledge base.

**Claim extraction is sentence selection.** A "claim" is a sentence that carries
an entity, a quantity, a date or an attribution verb. It is not semantic parsing;
the subject/predicate/object split is a heuristic around attribution verbs.

## What the data does not cover

**GDELT and the configured feeds are not all journalism.** Coverage skews by
language, region, publisher and what GDELT's crawler reaches. Any statement about
"what was reported" means "what was reported in this sample".

**The live feed list is a convenience sample.** `configs/topics.live.yaml` holds
36 public feeds chosen because they are public and they work, not because they
represent anything. English-language technology outlets dominate it, and vendor
blogs (OpenAI, DeepMind, Hugging Face, NVIDIA) contribute a large share of the
volume while corroborating nothing — they are one source reporting on itself.
Several of those feeds are archives spanning back to 2015, which is why the live
sweep windows the corpus rather than using all of it.

**GDELT throttles, and a throttled sweep is a biased sample.** The API asks for
one request every five seconds and applies a longer-term per-IP allowance on
top. A sweep that is cut short mid-way does not fail cleanly into "less data" —
it fails into data skewed toward whichever topics and windows were fetched
first. The ingestion run is recorded as `failed` when this happens; check the
run table before treating a corpus as complete.

**The committed corpus is synthetic.** Every fixture headline, sentence and
publisher (`*.example`) was written for this repository so the demo can run
offline without redistributing anyone's reporting. That makes results
reproducible and makes them **not** evidence about real news. Notably, the
background stories are generated from shared sentence frames, which makes
same-frame merge errors more likely than they would be on real text.

**Only metadata and short excerpts are stored.** Article bodies are not
archived. Full-text extraction is off by default and, when enabled, is limited to
an explicit publisher allowlist in `configs/sources.yaml`. Robots.txt, terms,
rate limits and paywalls are to be respected; the system will not work around
them.

**Results are from AI and technology news.** They should not be assumed to carry
over to politics, medicine, sport or local news, where event structure, source
ecosystems and language differ.

**Market relevance is retrospective.** Nothing here is investment advice, and no
point-in-time protocol is implemented. Any market-related evaluation would need
strict as-of-time discipline that this system does not yet enforce.

## Where the measurements are weak

- The fixture chronological test period holds **27 articles in 10 stories**.
  Perfect clustering scores on that slice reflect its size.
- **Neither corpus has human relevance judgments.** The fixture labels were
  written alongside the fixtures; the live labels are derived from the system's
  own clustering and so flatter the representation that produced them. Templates
  for real labelling are in `data/labels/`.
- Retrieval latency is **not** meaningfully measured at 135 vectors; the
  fixture differences are scheduler noise. The live run's 1,120 vectors are
  enough to measure it, and that is the only place a latency claim is made.
- **The live run separates nothing.** Every paired bootstrap interval against
  the full-dimensional baseline contains zero. Read those results as "23 queries
  could not tell these apart", never as "the methods are equivalent".
- Only **23 live queries** survived, because the 30-day window holds 59
  multi-article stories of which 32 have two or more non-duplicate articles.
  Genuine multi-source events are rare in a one-month feed sample.
- TF-IDF outscoring the dense baseline is a property of this lexically easy
  corpus and should not be generalised.
- **PageRank reranking helps on one corpus and hurts on the other** (+0.041
  nDCG@10 live, -0.033 on fixtures). The claim is unsettled and is reported as
  unsettled.
- Clustering quality is measured on the fixtures only; the live corpus has no
  labels to measure it against.

## Models and determinism

The default embedder is `all-MiniLM-L6-v2` (English, general-domain). Without the
`embeddings` extra, NewsTrace falls back to a deterministic hashing embedder,
which is weaker — the tuned clustering threshold differs between the two, and
retrieval quality with the fallback has not been benchmarked.

Optional LLM summaries are off by default (`LLM_PROVIDER=none`). When enabled,
the model sees only retrieved evidence, every statement must cite resolvable
evidence ids, uncited statements are dropped, and any failure falls back to the
deterministic extractive summary. An LLM can still omit nuance or mis-frame a
statement whose individual citations are all valid.
