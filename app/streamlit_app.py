"""NewsTrace Streamlit interface.

Deliberately diagnostic rather than pretty: source links are always visible and
never hidden behind generated prose.
"""

from __future__ import annotations

import contextlib
import html
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import func, select

from newstrace import __version__
from newstrace.claims.evidence import load_claims
from newstrace.config import get_llm_settings, get_settings, load_topics
from newstrace.db import create_all, session_scope
from newstrace.ingestion.deduplicate import strip_site_suffix
from newstrace.ingestion.pipeline import latest_run
from newstrace.models import Article, EvaluationRun, IngestionRun, StoryCluster
from newstrace.representations.embedder import detect_device, get_embedder
from newstrace.representations.registry import available_representations
from newstrace.search import SearchFilters, SearchRequest
from newstrace.search import search as run_search
from newstrace.stories import build_timeline, list_stories, load_story, story_keywords
from newstrace.summarization.extractive import ExtractiveSummarizer
from newstrace.utils import truncate

st.set_page_config(page_title="NewsTrace", layout="wide")

DISCLAIMER = (
    "NewsTrace reports repetition, provenance, duplication and source diversity. "
    "It is **not** a fact checker: it does not judge whether a claim is true or "
    "whether a publisher is reliable. Copied or syndicated articles are never "
    "counted as independent confirmation."
)

# The sidebar carries the disclaimer in full on every page, so the reading pages
# only need the one line that changes how you read the numbers on them.
SHORT_DISCLAIMER = "Counts corroboration and provenance, never accuracy."


# Card styling. Streamlit has no chip or eyebrow primitive, so the story card is
# rendered as one escaped HTML block; everything here is scoped to ``ns-`` class
# names rather than Streamlit's internal test ids, which change between releases.
CARD_CSS = """
<style>
.ns-eyebrow {
  font-size: 0.72rem; letter-spacing: 0.09em; text-transform: uppercase;
  color: #8A8073; margin-bottom: 0.35rem;
}
.ns-headline {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 1.42rem; line-height: 1.28; font-weight: 600;
  color: #1A1A1A; margin-bottom: 0.7rem;
}
.ns-signal { font-size: 0.92rem; color: #3A3A3A; margin-bottom: 0.55rem; }
.ns-signal b { font-weight: 700; }
.ns-dots { letter-spacing: 0.14em; color: #8C2F1E; margin-right: 0.45rem; }
.ns-dots .off { color: #D8D0C2; }
.ns-warn { color: #8C2F1E; font-weight: 600; }
.ns-chips { margin-bottom: 0.5rem; line-height: 2.1; }
.ns-chip {
  border: 1px solid #E4DED3; background: #F7F4EE; border-radius: 3px;
  padding: 0.16rem 0.46rem; margin-right: 0.3rem; font-size: 0.78rem; color: #4A443B;
  white-space: nowrap;
}
.ns-kw { font-size: 0.8rem; color: #8A8073; }
</style>
"""

MAX_DOTS = 8


@st.cache_resource
def _init() -> bool:
    create_all()
    return True


def fmt_time(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M UTC") if value else "unknown time"


# Internal keys are snake_case; nothing in the interface should be. Columns and
# metrics are mapped explicitly rather than guessed at, because a generic
# prettifier turns "ndcg_at_10" into "Ndcg At 10".
COLUMN_LABELS = {
    "article_id": "Article",
    "articles": "Articles",
    "artifact_path": "Artifact",
    "dimension": "Dimension",
    "distortion_mean_abs_cosine_error": "Mean absolute cosine error",
    "domain": "Source",
    "duplicate_of": "Copy of",
    "duplicates": "Duplicates",
    "failures": "Failures",
    "feeds": "Feeds",
    "fetched": "Fetched",
    "fit_version": "Fit version",
    "gdelt_query": "GDELT query",
    "id": "Run",
    "inserted": "Inserted",
    "kind": "Kind",
    "method": "Method",
    "model_name": "Model",
    "near_duplicate": "Near-duplicate",
    "published": "Published",
    "retrieval_ndcg_at_10": "nDCG at 10",
    "retrieval_recall_at_10": "Recall at 10",
    "retrieval_topk_overlap_at_10": "Top-k overlap at 10",
    "source": "Source",
    "started": "Started",
    "status": "Status",
    "system_index_memory_mib": "Index memory (MiB)",
    "system_p95_query_latency_ms": "p95 latency (ms)",
    "title": "Title",
    "topic": "Topic",
    "url": "Link",
}


def column_config(frame: pd.DataFrame) -> dict[str, object]:
    """Readable headers, and links rendered as links rather than raw URLs."""
    config: dict[str, object] = {}
    for column in frame.columns:
        label = COLUMN_LABELS.get(str(column), str(column).replace("_", " ").capitalize())
        if str(column) in {"url", "link"}:
            config[column] = st.column_config.LinkColumn(label, display_text="open")
        else:
            config[column] = st.column_config.Column(label)
    return config


METHOD_LABELS = {
    "full": "Full",
    "gaussian_rp": "Gaussian RP",
    "sparse_rp": "Sparse RP",
    "svd": "SVD",
    "tfidf": "TF-IDF",
}


def method_label(method: str) -> str:
    """Readable name for a representation method."""
    return METHOD_LABELS.get(method, method.replace("_", " "))


def headline(title: str) -> str:
    """A story title without the publisher's masthead tacked on the end."""
    return strip_site_suffix(title)


def topic_label(key: str | None) -> str:
    """The configured display name for a topic key, never the raw key."""
    if not key:
        return "No topic"
    topic = load_topics().get(key)
    return topic.label() if topic else key.replace("_", " ").capitalize()


def fmt_date_short(value: datetime | None) -> str:
    return value.strftime("%d %b %Y").lstrip("0") if value else "undated"


def dots(count: int, cap: int = MAX_DOTS) -> str:
    """A filled/empty dot run for ``count``, so diversity reads pre-attentively."""
    filled = min(count, cap)
    markup = "●" * filled + f'<span class="off">{"●" * (cap - filled)}</span>'
    # The overflow marker stays inside the span: outside it, the run's trailing
    # margin falls before the "+" and it reads as part of the count ("+9").
    return f'<span class="ns-dots">{markup}{"+" if count > cap else ""}</span>'


def chips(values: list[str], cap: int = 6) -> str:
    shown = [f'<span class="ns-chip">{html.escape(v)}</span>' for v in values[:cap]]
    if len(values) > cap:
        shown.append(f'<span class="ns-chip">+{len(values) - cap} more</span>')
    return f'<div class="ns-chips">{"".join(shown)}</div>'


WINDOWS = {"24 hours": 24, "3 days": 72, "7 days": 24 * 7, "30 days": 24 * 30, "All time": 0}


def page_top_stories() -> None:
    st.header("Stories")
    st.caption(SHORT_DISCLAIMER)
    with session_scope() as session:
        keys = list(load_topics())
        # Topics as pills: there are six of them, and a pill row shows every
        # choice at once where a select box hides five of the six.
        chosen = st.pills(
            "Topic", ["All topics", *keys], format_func=topic_label, default="All topics"
        )
        topic = None if chosen in (None, "All topics") else chosen
        window = st.segmented_control("Published within", list(WINDOWS), default="All time")
        with st.popover("More filters"):
            limit = st.slider("Stories to show", 5, 60, 20)
            min_articles = st.slider("Minimum articles per story", 1, 6, 1)
        hours = WINDOWS.get(window or "All time", 0)
        start = datetime.now(UTC) - timedelta(hours=hours) if hours else None

        stories = list_stories(
            session,
            topic=topic,
            limit=limit,
            min_articles=min_articles,
            start_time=start,
        )
        if not stories:
            st.info("No stories yet. Run `make demo` to ingest the committed fixtures.")
            return
        for story in stories:
            bundle = load_story(session, story.id)
            if bundle is None:
                continue
            with st.container(border=True):
                sources = bundle.independent_domains
                near_dupes = len(bundle.duplicates)
                # Independent sources is the number the tool exists to report, so
                # it carries the dots and the bold; everything else is secondary.
                dupe_note = (
                    f' · <span class="ns-warn">{near_dupes} near-duplicate'
                    f"{'' if near_dupes == 1 else 's'}</span>"
                    if near_dupes
                    else " · no near-duplicates"
                )
                st.markdown(
                    f'<div class="ns-eyebrow">{html.escape(topic_label(story.topic))}'
                    f" &nbsp;·&nbsp; {fmt_date_short(story.last_published_at)}</div>"
                    f'<div class="ns-headline">{html.escape(headline(story.display_title))}</div>'
                    f'<div class="ns-signal">{dots(len(sources))}'
                    f"<b>{len(sources)}</b> independent source"
                    f"{'' if len(sources) == 1 else 's'}"
                    f" · {story.article_count} article"
                    f"{'' if story.article_count == 1 else 's'}{dupe_note}</div>"
                    f"{chips(sources)}",
                    unsafe_allow_html=True,
                )
                keywords = story_keywords(bundle)
                if keywords:
                    st.markdown(
                        f'<div class="ns-kw">{html.escape(" · ".join(keywords))}</div>',
                        unsafe_allow_html=True,
                    )
                if st.button("Open story", key=f"open-{story.id}"):
                    # ``story_id`` is the story page's select box key, so writing
                    # it here is what preselects the story once that page runs.
                    st.session_state["story_id"] = story.id
                    st.switch_page(STORY_PAGE)


def page_story_detail() -> None:
    st.header("Story")
    st.caption(SHORT_DISCLAIMER)
    with session_scope() as session:
        # Newest first, and capped: the picker is for getting back to something
        # you were reading, not for scrolling four thousand rows.
        recent = session.execute(
            select(StoryCluster)
            .order_by(StoryCluster.last_published_at.desc().nulls_last(), StoryCluster.id.desc())
            .limit(200)
        ).scalars()
        headlines = {s.id: s.display_title for s in recent}
        # A story reached by link or by the Stories page may be older than the
        # cap, so it is added rather than silently swapped for another story.
        held = st.session_state.get("story_id")
        for candidate in (held, int(query_param("story") or 0)):
            if isinstance(candidate, int) and candidate and candidate not in headlines:
                story = session.get(StoryCluster, candidate)
                if story is not None:
                    headlines[story.id] = story.display_title
        ids = list(headlines)
        if not ids:
            st.info("No stories yet. Run `make demo` first.")
            return
        requested = query_param("story")
        from_url = int(requested) if requested.isdigit() and int(requested) in ids else None
        # ``?story=`` is written back on every render, so it only wins when it
        # changes; otherwise it would override the story the "Open story" button
        # just selected. The selectbox is keyed on ``story_id`` so that widget
        # state and the button write to the same place.
        if from_url is not None and from_url != st.session_state.get("_story_from_url"):
            st.session_state["story_id"] = from_url
        if st.session_state.get("story_id") not in ids:
            st.session_state["story_id"] = ids[0]
        story_id = st.selectbox(
            "Story",
            ids,
            key="story_id",
            format_func=lambda i: truncate(headline(headlines.get(i, str(i))), 90),
        )
        set_query_param("story", str(story_id))
        st.session_state["_story_from_url"] = story_id
        bundle = load_story(session, story_id)
        if bundle is None:
            st.error("Story not found.")
            return

        sources = bundle.independent_domains
        near_dupes = len(bundle.duplicates)
        dupe_note = (
            f' · <span class="ns-warn">{near_dupes} near-duplicate'
            f"{'' if near_dupes == 1 else 's'} not counted as confirmation</span>"
            if near_dupes
            else " · no near-duplicates"
        )
        st.markdown(
            f'<div class="ns-eyebrow">{html.escape(topic_label(bundle.story.topic))}'
            f" &nbsp;·&nbsp; {fmt_date_short(bundle.story.last_published_at)}</div>"
            f'<div class="ns-headline">{html.escape(headline(bundle.story.display_title))}</div>'
            f'<div class="ns-signal">{dots(len(sources))}'
            f"<b>{len(sources)}</b> independent source"
            f"{'' if len(sources) == 1 else 's'}"
            f" · {len(bundle.articles)} article"
            f"{'' if len(bundle.articles) == 1 else 's'}{dupe_note}</div>"
            f"{chips(sources, cap=10)}",
            unsafe_allow_html=True,
        )

        tab_timeline, tab_summary, tab_claims, tab_sources = st.tabs(
            ["Timeline", "Grounded summary", "Claims", "Sources"]
        )

        with tab_timeline:
            st.caption(
                "Ordering reflects publication timestamps in this dataset only. The earliest "
                "available report is not proof of who reported a claim first."
            )
            for entry in build_timeline(bundle):
                label = f"{fmt_time(entry.published_at)} — {entry.article.source_domain}"
                if entry.is_first_report:
                    label += "  ⟵ earliest available report"
                if entry.is_near_duplicate:
                    label += "  (near-duplicate, not counted as independent)"
                with st.expander(label, expanded=entry.is_first_report):
                    st.markdown(f"**{entry.article.title}**")
                    st.markdown(f"[{entry.article.url}]({entry.article.url})")
                    if entry.new_sentences:
                        st.markdown("**Potentially new in this report**")
                        for sentence in entry.new_sentences:
                            st.markdown(f"- {sentence}")
                    if entry.repeated_sentences:
                        st.markdown("**Already reported**")
                        for sentence in entry.repeated_sentences[:3]:
                            st.markdown(f"- {sentence}")

        with tab_summary:
            summary = ExtractiveSummarizer(session).summarize_bundle(bundle)
            index = summary.evidence_index()
            st.caption(
                f"Method: {summary.method.replace('_', ' ')} · "
                f"every statement cited: {summary.citation_coverage():.0%}"
            )
            sections = {
                "new_developments": "New developments",
                "repeated_reporting": "Repeated across independent sources",
                "single_source_claims": "Single-source claims",
                "disagreements_or_uncertainty": "Disagreements or uncertainty",
            }
            for key, heading in sections.items():
                items = [s for s in summary.statements if s.section.value == key]
                if not items:
                    continue
                st.markdown(f"### {heading}")
                for statement in items:
                    st.markdown(f"- {statement.text}")
                    for eid in statement.evidence_ids:
                        ev = index[eid]
                        flag = " *(near-duplicate)*" if ev.is_near_duplicate else ""
                        st.caption(
                            f"    ↳ {ev.source_domain} · {fmt_time(ev.published_at)} · "
                            f"[link]({ev.url}){flag}"
                        )
            for note in summary.notes:
                st.info(note)

        with tab_claims:
            claims = load_claims(session, story_id)
            if not claims:
                st.info("No claims extracted for this story.")
            for claim, evidence in claims[:30]:
                domains = {e.article_id: e for e in evidence}
                independent = len(
                    {
                        a.source_domain
                        for a in bundle.articles
                        if a.id in domains and not a.is_near_duplicate
                    }
                )
                status = "Repeated" if independent > 1 else "Single source"
                if any(e.stance in ("disputes", "unclear") for e in evidence):
                    status = "Disputed or unclear"
                with st.expander(f"{status} — {truncate(claim.normalized_claim, 110)}"):
                    st.caption(
                        f"Extraction: {claim.extraction_method.replace('_', ' ')} · "
                        f"confidence {claim.confidence:.2f} · "
                        f"{independent} independent source"
                        f"{'' if independent == 1 else 's'}"
                    )
                    for ev in evidence:
                        article = next((a for a in bundle.articles if a.id == ev.article_id), None)
                        if article is None:
                            continue
                        dup = " *(near-duplicate)*" if article.is_near_duplicate else ""
                        st.markdown(
                            f"- **{article.source_domain}** ({ev.stance}){dup}: {ev.excerpt}  \n"
                            f"  [{article.url}]({article.url})"
                        )

        with tab_sources:
            by_id = {a.id: a for a in bundle.articles}
            rows = [
                {
                    "domain": a.source_domain,
                    "published": fmt_time(a.published_at),
                    "near_duplicate": bool(a.is_near_duplicate),
                    # The domain it copies says more than the article's row id.
                    "duplicate_of": (
                        by_id[a.duplicate_of_article_id].source_domain
                        if a.duplicate_of_article_id in by_id
                        else ""
                    ),
                    "title": a.title,
                    "url": a.url,
                }
                for a in bundle.sorted_by_time(include_duplicates=True)
            ]
            frame = pd.DataFrame(rows)
            st.dataframe(frame, width="stretch", column_config=column_config(frame))


def page_search() -> None:
    st.header("Search")
    st.caption(SHORT_DISCLAIMER)
    with session_scope() as session:
        query = st.text_input(
            "Query",
            value=query_param("q") or "What changed today regarding new AI regulations?",
        )
        left, right = st.columns([3, 1])
        keys = list(load_topics())
        chosen = left.selectbox(
            "Topic", ["All topics", *keys], format_func=topic_label, label_visibility="collapsed"
        )
        topic = None if chosen == "All topics" else chosen
        top_k = right.number_input("Results", min_value=1, max_value=30, value=10)

        # Everything below changes how retrieval runs rather than what you are
        # asking for, and the experiment dashboard is where it is compared.
        with st.popover("Retrieval settings"):
            reps = available_representations(session)
            methods = [
                "full",
                "tfidf",
                *sorted({str(r["method"]) for r in reps if r["method"] != "full"}),
            ]
            method = st.selectbox("Method", methods, format_func=method_label)
            dims = sorted({int(r["dimension"]) for r in reps if r["method"] == method})
            dimension = (
                st.selectbox("Dimension", dims)
                if dims and method not in ("full", "tfidf")
                else None
            )
            domain = st.text_input("Limit to source domain", value="")
            pagerank = st.checkbox("Rerank with PageRank", value=False)
            include_dupes = st.checkbox("Include near-duplicates", value=False)

        if st.button("Search", type="primary") or query:
            request = SearchRequest(
                query=query,
                filters=SearchFilters(
                    topic=topic,
                    source_domain=domain or None,
                    include_duplicates=include_dupes,
                ),
                method=method,
                dimension=dimension,
                rerank_with_pagerank=pagerank,
                top_k=top_k,
            )
            response = run_search(session, request)
            st.caption(
                f"{len(response.items)} results · {response.method.replace('_', ' ')} · "
                f"{response.took_ms:.1f} ms · {response.candidates_considered} candidates"
            )
            for note in response.notes:
                st.warning(note)
            for item in response.items:
                with st.container(border=True):
                    st.markdown(f"**{item.article.title}**")
                    st.markdown(
                        f"{item.article.source_domain} · {fmt_time(item.article.published_at)} · "
                        f"[{item.article.url}]({item.article.url})"
                    )
                    st.markdown(f"> {item.excerpt}")
                    st.caption(f"Why this ranked here — {item.explanation}")
                    if item.story_id:
                        st.caption(f"Part of story {item.story_id}")


def page_topics() -> None:
    st.header("Topics")
    st.caption("Read-only here. Edit configs/topics.live.yaml and ingest again to change it.")
    with session_scope() as session:
        counts = dict(
            session.execute(
                select(Article.topic, func.count(Article.id)).group_by(Article.topic)
            ).all()
        )
        rows = [
            {
                "topic": topic.label(),
                "feeds": len(topic.feeds),
                "articles": counts.get(key, 0),
                "gdelt_query": topic.gdelt_query or "RSS only",
            }
            for key, topic in load_topics().items()
        ]
    frame = pd.DataFrame(rows)
    st.dataframe(frame, width="stretch", hide_index=True, column_config=column_config(frame))


def page_experiments() -> None:
    st.header("Experiments")
    settings = get_settings()
    root = settings.resolve(settings.artifacts_dir) / "experiments"
    runs = sorted((p for p in root.glob("*") if (p / "metrics.json").exists()), reverse=True)
    if not runs:
        st.info("No experiments yet. Run `make experiment`.")
        return
    choice = st.selectbox("Experiment", runs, format_func=lambda p: p.name)
    metrics = json.loads((Path(choice) / "metrics.json").read_text(encoding="utf-8"))

    split = metrics.get("split", {})
    cols = st.columns(4)
    cols[0].metric("Articles", metrics.get("article_count", 0))
    cols[1].metric("Queries", metrics.get("query_count", 0))
    cols[2].metric("Test articles", split.get("test_count", 0))
    cols[3].metric("Judgments", metrics.get("judgment_source", "—"))

    rows = metrics.get("results", [])
    if rows:
        frame = pd.DataFrame(rows)
        st.subheader("Quality versus cost")
        shown = frame[
            [
                c
                for c in [
                    "method",
                    "dimension",
                    "retrieval_recall_at_10",
                    "retrieval_ndcg_at_10",
                    "retrieval_topk_overlap_at_10",
                    "system_index_memory_mib",
                    "system_p95_query_latency_ms",
                    "distortion_mean_abs_cosine_error",
                ]
                if c in frame.columns
            ]
        ]
        shown = shown.copy()
        if "method" in shown.columns:
            shown["method"] = shown["method"].map(method_label)
        st.dataframe(shown, width="stretch", hide_index=True, column_config=column_config(shown))
        for filename, caption in (
            ("quality_vs_dimension.png", "Quality versus dimension"),
            ("quality_vs_memory.png", "Quality versus index memory"),
            ("quality_vs_latency.png", "Quality versus p95 latency"),
        ):
            path = Path(choice) / filename
            if path.exists():
                st.image(str(path), caption=caption)

    pagerank = metrics.get("pagerank")
    if pagerank:
        st.subheader("Graph reranking")
        st.json(
            {
                k: v
                for k, v in pagerank.items()
                if k in ("retrieval", "mean_iterations", "converged_fraction", "max_abs_mass_error")
            }
        )
        st.dataframe(pd.DataFrame(pagerank.get("damping_sensitivity", [])), width="stretch")

    report = Path(choice) / "report.md"
    if report.exists():
        with st.expander("Written report"):
            st.markdown(report.read_text(encoding="utf-8"))


def page_health() -> None:
    st.header("Health")
    settings = get_settings()
    llm = get_llm_settings()
    with session_scope() as session:
        embedder = get_embedder(settings)
        run = latest_run(session)
        cols = st.columns(4)
        cols[0].metric(
            "Articles", int(session.execute(select(func.count(Article.id))).scalar_one())
        )
        cols[1].metric(
            "Stories", int(session.execute(select(func.count(StoryCluster.id))).scalar_one())
        )
        cols[2].metric(
            "Near-duplicates",
            int(
                session.execute(
                    select(func.count(Article.id)).where(Article.is_near_duplicate.is_(True))
                ).scalar_one()
            ),
        )
        cols[3].metric("Version", __version__)

        st.subheader("Models and configuration")
        settings_frame = pd.DataFrame(
            [
                {"Setting": "Environment", "Value": settings.env},
                {"Setting": "Database", "Value": settings.database_url},
                {"Setting": "Embedding model", "Value": embedder.name},
                {"Setting": "Embedding backend", "Value": settings.embedding_backend},
                {"Setting": "Embedding dimension", "Value": embedder.dimension},
                {"Setting": "Device", "Value": detect_device()},
                {"Setting": "Cluster threshold", "Value": settings.cluster_threshold},
                {"Setting": "Cluster window (hours)", "Value": settings.cluster_window_hours},
                {"Setting": "Random seed", "Value": settings.random_seed},
                {"Setting": "Language model", "Value": llm.llm_provider},
            ]
        )
        st.dataframe(settings_frame, width="stretch", hide_index=True)

        st.subheader("Stored representations")
        reps_frame = pd.DataFrame(available_representations(session))
        if "method" in reps_frame.columns:
            reps_frame["method"] = reps_frame["method"].map(method_label)
        st.dataframe(
            reps_frame,
            width="stretch",
            hide_index=True,
            column_config=column_config(reps_frame),
        )

        st.subheader("Recent ingestion runs")
        runs = session.execute(
            select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(10)
        ).scalars()
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "id": r.id,
                        "source": r.source,
                        "status": r.status,
                        "started": fmt_time(r.started_at),
                        "fetched": r.fetched_count,
                        "inserted": r.inserted_count,
                        "duplicates": r.duplicate_count,
                        "failures": r.failure_count,
                    }
                    for r in runs
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        if run and run.errors:
            st.warning(f"Last run errors: {run.errors}")

        st.subheader("Recent evaluation runs")
        evals = session.execute(
            select(EvaluationRun).order_by(EvaluationRun.started_at.desc()).limit(5)
        ).scalars()
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "id": e.id,
                        "kind": e.kind,
                        "status": e.status,
                        "started": fmt_time(e.started_at),
                        "artifact_path": e.artifact_path,
                    }
                    for e in evals
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def query_param(key: str) -> str:
    """Read one query parameter, tolerating Streamlit versions without them."""
    try:
        value = st.query_params.get(key)
    except Exception:  # pragma: no cover - depends on the Streamlit version
        return ""
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def set_query_param(key: str, value: str) -> None:
    """Write a query parameter only when it changes, so the app cannot loop."""
    if query_param(key) == value:
        return
    with contextlib.suppress(Exception):  # depends on the Streamlit version
        st.query_params[key] = value


# Pages are declared once and Streamlit routes them, which is what retires the
# hand-rolled ``?page=`` handling this file used to carry: each page owns a real
# URL (``/stories``), the sidebar is generated, and "Open story" is a
# ``switch_page`` rather than a target parked in session state for the next run.
STORIES_PAGE = st.Page(page_top_stories, title="Stories", default=True)
STORY_PAGE = st.Page(page_story_detail, title="Story", url_path="story")
SEARCH_PAGE = st.Page(page_search, title="Search", url_path="search")
TOPICS_PAGE = st.Page(page_topics, title="Topics", url_path="topics")
EXPERIMENTS_PAGE = st.Page(page_experiments, title="Experiments", url_path="experiments")
HEALTH_PAGE = st.Page(page_health, title="Health", url_path="health")

# Reading the news is one job and inspecting the system is another; the grouping
# says so, rather than presenting six equal-weight pages.
NAVIGATION = {
    "Read": [STORIES_PAGE, STORY_PAGE, SEARCH_PAGE],
    "Inspect": [TOPICS_PAGE, EXPERIMENTS_PAGE, HEALTH_PAGE],
}


def main() -> None:
    _init()
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.sidebar.title("NewsTrace")
    st.sidebar.caption(f"Version {__version__}")
    page = st.navigation(NAVIGATION)
    st.sidebar.markdown("---")
    st.sidebar.caption(DISCLAIMER)
    page.run()


main()
