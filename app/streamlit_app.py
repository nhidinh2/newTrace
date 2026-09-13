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
from newstrace.ingestion.pipeline import latest_run
from newstrace.models import Article, EvaluationRun, IngestionRun, StoryCluster
from newstrace.representations.embedder import detect_device, get_embedder
from newstrace.representations.registry import available_representations
from newstrace.search import SearchFilters, SearchRequest
from newstrace.search import search as run_search
from newstrace.stories import build_timeline, list_stories, load_story, story_keywords
from newstrace.summarization.extractive import ExtractiveSummarizer
from newstrace.utils import truncate

st.set_page_config(page_title="NewsTrace", page_icon="📰", layout="wide")

DISCLAIMER = (
    "NewsTrace reports repetition, provenance, duplication and source diversity. "
    "It is **not** a fact checker: it does not judge whether a claim is true or "
    "whether a publisher is reliable. Copied or syndicated articles are never "
    "counted as independent confirmation."
)


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


def page_top_stories() -> None:
    st.header("Top stories")
    st.caption(DISCLAIMER)
    with session_scope() as session:
        topics = ["(all)", *load_topics()]
        col1, col2, col3 = st.columns(3)
        topic = col1.selectbox("Topic", topics, index=0)
        limit = col2.slider("Stories", 5, 60, 20)
        min_articles = col3.slider("Minimum articles", 1, 6, 1)
        window = st.select_slider(
            "Time window", options=["24h", "72h", "7d", "30d", "all"], value="all"
        )
        start = None
        if window != "all":
            hours = {"24h": 24, "72h": 72, "7d": 24 * 7, "30d": 24 * 30}[window]
            start = datetime.now(UTC) - timedelta(hours=hours)

        stories = list_stories(
            session,
            topic=None if topic == "(all)" else topic,
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
                    f'<div class="ns-eyebrow">{html.escape(story.topic or "untopiced")}'
                    f" &nbsp;·&nbsp; {fmt_date_short(story.last_published_at)}</div>"
                    f'<div class="ns-headline">{html.escape(story.display_title)}</div>'
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
                if st.button("Open story  →", key=f"open-{story.id}"):
                    # ``page`` and ``story_id`` are widget keys, and Streamlit
                    # forbids writing a widget's state once it is instantiated,
                    # so park the target for main() to apply on the next run.
                    st.session_state["_goto"] = ("Story detail", story.id)
                    st.rerun()


def page_story_detail() -> None:
    st.header("Story detail")
    st.caption(DISCLAIMER)
    with session_scope() as session:
        ids = [
            s.id for s in session.execute(select(StoryCluster).order_by(StoryCluster.id)).scalars()
        ]
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
        story_id = st.selectbox("Story", ids, key="story_id")
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
            f'<div class="ns-eyebrow">{html.escape(bundle.story.topic or "untopiced")}'
            f" &nbsp;·&nbsp; {fmt_date_short(bundle.story.last_published_at)}</div>"
            f'<div class="ns-headline">{html.escape(bundle.story.display_title)}</div>'
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
                f"Method: `{summary.method}` · citation coverage: {summary.citation_coverage():.0%}"
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
                status = "repeated" if independent > 1 else "single source"
                if any(e.stance in ("disputes", "unclear") for e in evidence):
                    status = "unclear / disputed"
                with st.expander(f"[{status}] {truncate(claim.normalized_claim, 110)}"):
                    st.caption(
                        f"extraction: {claim.extraction_method} · "
                        f"confidence: {claim.confidence:.2f} · "
                        f"independent domains: {independent}"
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
            rows = [
                {
                    "article_id": a.id,
                    "domain": a.source_domain,
                    "published": fmt_time(a.published_at),
                    "near_duplicate": a.is_near_duplicate,
                    "duplicate_of": a.duplicate_of_article_id,
                    "title": a.title,
                    "url": a.url,
                }
                for a in bundle.sorted_by_time(include_duplicates=True)
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch")


def page_search() -> None:
    st.header("Search")
    st.caption(DISCLAIMER)
    with session_scope() as session:
        query = st.text_input(
            "Query",
            value=query_param("q") or "What changed today regarding new AI regulations?",
        )
        col1, col2, col3, col4 = st.columns(4)
        reps = available_representations(session)
        methods = [
            "full",
            "tfidf",
            *sorted({str(r["method"]) for r in reps if r["method"] != "full"}),
        ]
        method = col1.selectbox("Method", methods)
        dims = sorted({int(r["dimension"]) for r in reps if r["method"] == method})
        dimension = (
            col2.selectbox("Dimension", dims) if dims and method not in ("full", "tfidf") else None
        )
        top_k = col3.slider("Top k", 1, 30, 10)
        pagerank = col4.checkbox("Rerank with PageRank", value=False)

        col5, col6, col7 = st.columns(3)
        topic = col5.selectbox("Topic", ["(all)", *load_topics()])
        domain = col6.text_input("Source domain", value="")
        include_dupes = col7.checkbox("Include near-duplicates", value=False)

        if st.button("Search", type="primary") or query:
            request = SearchRequest(
                query=query,
                filters=SearchFilters(
                    topic=None if topic == "(all)" else topic,
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
                f"{len(response.items)} results · {response.method} · "
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
                        st.caption(f"Part of story #{item.story_id}")


def page_topics() -> None:
    st.header("Topic configuration")
    st.caption(
        "Read-only in the MVP. Edit `configs/topics.yaml` and re-run ingestion to change it."
    )
    with session_scope() as session:
        counts = dict(
            session.execute(
                select(Article.topic, func.count(Article.id)).group_by(Article.topic)
            ).all()
        )
        rows = [
            {
                "key": key,
                "display_name": topic.label(),
                "gdelt_query": topic.gdelt_query,
                "feeds": len(topic.feeds),
                "articles": counts.get(key, 0),
            }
            for key, topic in load_topics().items()
        ]
    st.dataframe(pd.DataFrame(rows), width="stretch")


def page_experiments() -> None:
    st.header("Experiment dashboard")
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
        st.dataframe(
            frame[
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
            ],
            width="stretch",
        )
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
        with st.expander("report.md"):
            st.markdown(report.read_text(encoding="utf-8"))


def page_health() -> None:
    st.header("System health")
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
        st.json(
            {
                "environment": settings.env,
                "database_url": settings.database_url,
                "embedding_model": embedder.name,
                "embedding_backend": settings.embedding_backend,
                "embedding_dimension": embedder.dimension,
                "device": detect_device(),
                "cluster_threshold": settings.cluster_threshold,
                "cluster_window_hours": settings.cluster_window_hours,
                "random_seed": settings.random_seed,
                "llm_provider": llm.llm_provider,
            }
        )

        st.subheader("Stored representations")
        st.dataframe(pd.DataFrame(available_representations(session)), width="stretch")

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
        )


def page_slug(name: str) -> str:
    """URL-friendly form of a page name."""
    return name.lower().replace(" ", "-")


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


PAGES = {
    "Top stories": page_top_stories,
    "Story detail": page_story_detail,
    "Search": page_search,
    "Topic configuration": page_topics,
    "Experiment dashboard": page_experiments,
    "System health": page_health,
}


def requested_page() -> str | None:
    """The page named by ``?page=`` if it matches one we serve."""
    requested = query_param("page")
    if not requested:
        return None
    for name in PAGES:
        if requested.lower() in {name.lower(), page_slug(name)}:
            return name
    return None


def main() -> None:
    _init()
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.sidebar.title("📰 NewsTrace")
    st.sidebar.caption(f"v{__version__}")
    names = list(PAGES)

    # Apply in-app navigation parked by a button before any widget is created,
    # then let it flow through the query parameters like any other navigation.
    goto = st.session_state.pop("_goto", None)
    if goto is not None:
        target_page, target_story = goto
        st.session_state["page"] = target_page
        st.session_state["story_id"] = target_story
        set_query_param("page", page_slug(target_page))
        set_query_param("story", str(target_story))

    # ``?page=`` is written back on every render, so it only wins when it
    # changes; comparing against the raw value would let the URL from the
    # previous run override in-app navigation such as the "Open story" button.
    requested = requested_page()
    if requested and requested != st.session_state.get("_page_from_url"):
        st.session_state["page"] = requested
    if st.session_state.get("page") not in names:
        st.session_state["page"] = names[0]
    page = st.sidebar.radio("Page", names, key="page")
    set_query_param("page", page_slug(page))
    # Remember what we put in the URL, not what we read: recording the stale
    # value would make the next sidebar click look like an external ?page=
    # change and revert it.
    st.session_state["_page_from_url"] = page
    st.sidebar.markdown("---")
    st.sidebar.caption(DISCLAIMER)
    PAGES[page]()


main()
