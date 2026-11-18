"""Source coverage, matching totals and pagination across story entry points."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from newstrace.api.app import create_app
from newstrace.cli import app as cli_app
from newstrace.models import Article, ClusterMembership, StoryCluster
from newstrace.stories import count_stories, list_stories, load_story
from tests.conftest import BASE_TIME


@pytest.fixture
def feed_session(populated_session: Session) -> Session:
    """Add enough stories to exercise cutoff, tie, time and zero-source cases."""
    for title, domains, articles, topic, first, last in [
        ("Older broad coverage", 4, 5, "ai_models", -10, -8),
        ("Current broad coverage", 3, 4, "ai_models", -2, 2),
        ("Current broad coverage tie", 3, 4, "ai_models", -2, 2),
        ("Many articles one domain", 1, 50, "ai_models", 0, 3),
        ("Undated broad coverage", 3, 3, "ai_models", None, None),
        ("Other topic", 3, 4, "chips_and_compute", -2, 2),
        ("Only duplicates", 0, 9, "ai_models", 0, 0),
    ]:
        story = StoryCluster(
            display_title=title,
            distinct_domain_count=domains,
            article_count=articles,
            topic=topic,
            first_published_at=BASE_TIME + timedelta(days=first) if first is not None else None,
            last_published_at=BASE_TIME + timedelta(days=last) if last is not None else None,
        )
        populated_session.add(story)
        populated_session.flush()
        originals = []
        for i in range(articles):
            # Repeat a domain across original reports and add copies from other
            # domains. Neither pattern should inflate the stored source count.
            is_copy = domains == 0 or i >= max(domains, 3)
            domain = f"wire-{i}.example" if is_copy else f"source-{i % domains}.example"
            url = f"https://{domain}/story-{story.id}-report-{i}"
            article = Article(
                url=url,
                canonical_url=url,
                title=f"Report {i} on {title}",
                source_domain=domain,
                is_near_duplicate=is_copy,
                normalized_text_hash=url,
                normalized_title_hash=url,
                published_at=story.last_published_at,
                duplicate_of_article_id=originals[0] if is_copy and originals else None,
            )
            populated_session.add(article)
            populated_session.flush()
            if not is_copy:
                originals.append(article.id)
            populated_session.add(
                ClusterMembership(
                    article_id=article.id,
                    story_cluster_id=story.id,
                    similarity=0.9,
                )
            )
    populated_session.commit()
    return populated_session


def test_duplicate_reports_do_not_satisfy_the_domain_filter(populated_session: Session) -> None:
    # These are real ingested and clustered reports, including one wire copy.
    stories = list_stories(populated_session, limit=200)
    for story in stories:
        bundle = load_story(populated_session, story.id)
        assert bundle is not None
        assert story.distinct_domain_count == len(bundle.independent_domains)
    assert any(load_story(populated_session, s.id).duplicates for s in stories)
    expected = [s.id for s in stories if s.distinct_domain_count >= 2]
    assert expected
    assert [s.id for s in list_stories(populated_session, min_domains=2)] == expected
    assert list_stories(populated_session, min_domains=100) == []


@pytest.mark.parametrize(
    "filters",
    [
        {"min_domains": 0},
        {"min_domains": 2},
        {"min_domains": 3, "min_articles": 4, "topic": "ai_models"},
        {"min_domains": 3, "start_time": BASE_TIME, "end_time": BASE_TIME},
        {"min_domains": 3, "start_time": BASE_TIME + timedelta(days=20)},
        {"min_domains": 99},
    ],
)
def test_totals_and_pages_share_all_filters(feed_session: Session, filters: dict) -> None:
    all_ids = [s.id for s in list_stories(feed_session, limit=200, **filters)]
    assert count_stories(feed_session, **filters) == len(all_ids)
    paged_ids = [
        s.id
        for offset in range(0, len(all_ids) + 2, 2)
        for s in list_stories(feed_session, limit=2, offset=offset, **filters)
    ]
    assert paged_ids == all_ids
    with TestClient(create_app()) as client:
        params = {k: v.isoformat() if hasattr(v, "isoformat") else v for k, v in filters.items()}
        response = client.get("/stories", params=params | {"limit": 2, "offset": 2})
        assert response.status_code == 200
        assert int(response.headers["X-Total-Count"]) == len(all_ids)
        assert [s["id"] for s in response.json()] == all_ids[2:4]
        # A page past the end must still report the matching total.
        past_end = client.get("/stories", params=params | {"offset": 200})
        assert past_end.json() == []
        assert int(past_end.headers["X-Total-Count"]) == len(all_ids)


def test_domain_filter_combines_with_story_time_overlap(feed_session: Session) -> None:
    matches = list_stories(
        feed_session, min_domains=3, topic="ai_models", start_time=BASE_TIME, end_time=BASE_TIME
    )
    assert [s.display_title for s in matches] == [
        "Current broad coverage tie",
        "Current broad coverage",
    ]


@pytest.mark.parametrize("value", ["-1", "1.5", "invalid", str(2**63)])
def test_api_rejects_invalid_domain_threshold(feed_session: Session, value: str) -> None:
    with TestClient(create_app()) as client:
        assert client.get("/stories", params={"min_domains": value}).status_code == 422


def test_default_counts_preserve_empty_clusters(feed_session: Session) -> None:
    before = count_stories(feed_session)
    feed_session.add(StoryCluster(display_title="Empty historical cluster", article_count=0))
    feed_session.commit()
    assert count_stories(feed_session) == before + 1
    assert count_stories(feed_session, min_articles=1) == before
    assert len(list_stories(feed_session, limit=200)) == before


def test_cli_supports_source_filter_and_pages(
    feed_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from newstrace.cli import console

    monkeypatch.setattr(console, "width", 160)
    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        [
            "stories",
            "--min-domains",
            "3",
            "--min-articles",
            "4",
            "--topic",
            "ai_models",
            "--limit",
            "1",
            "--offset",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Current broad coverage tie" in result.output
    assert "Older broad coverage" not in result.output
    assert "Many articles one domain" not in result.output
    assert "Showing 2" in result.output and "of 3 matching stories" in result.output
    empty = runner.invoke(cli_app, ["stories", "--min-domains", "100"])
    assert empty.exit_code == 0
    assert "No stories match" in empty.output
    assert runner.invoke(cli_app, ["stories", "--min-domains", "-1"]).exit_code == 2


def _feed_app():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(
        str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"),
        default_timeout=30,
    ).run()


def _button(app, label: str):
    return next(b for b in app.button if b.label == label)


def _slider(app, label: str):
    return next(s for s in app.slider if s.label == label)


def test_feed_ui_paging_and_filter_resets(feed_session: Session) -> None:
    app = _feed_app()
    assert not app.exception
    assert _button(app, "Previous page").disabled
    _slider(app, "Stories per page").set_value(5).run()
    assert not app.exception
    assert "Showing 1–5 of 9 matching stories" in [c.value for c in app.caption]
    assert _button(app, "Previous page").disabled
    assert not _button(app, "Next page").disabled
    first_page = [m.value for m in app.markdown if 'class="ns-headline"' in m.value]
    assert len(first_page) == 5
    _button(app, "Next page").click().run()
    assert not app.exception
    assert "Showing 6–9 of 9 matching stories" in [c.value for c in app.caption]
    assert _button(app, "Next page").disabled
    last_page = [m.value for m in app.markdown if 'class="ns-headline"' in m.value]
    assert len(last_page) == 4
    assert not set(first_page) & set(last_page)
    _button(app, "Previous page").click().run()
    assert [m.value for m in app.markdown if 'class="ns-headline"' in m.value] == first_page
    _button(app, "Next page").click().run()
    app.number_input[0].set_value(3).run()
    assert not app.exception
    assert "Showing 1–5 of 5 matching stories" in [c.value for c in app.caption]
    assert _button(app, "Previous page").disabled
    assert _button(app, "Next page").disabled
    cards = [m.value for m in app.markdown if 'class="ns-headline"' in m.value]
    assert all("Many articles one domain" not in c and "Only duplicates" not in c for c in cards)
    # Filter changes on a later page reset the offset; changing article minimum
    # and page size also use the same reset path.
    _slider(app, "Minimum articles per story").set_value(6).run()
    assert not app.exception
    assert "No stories match these filters" in app.info[0].value
    assert "make demo" not in app.info[0].value
    app.number_input[0].set_value(0).run()
    assert "Showing 1–2 of 2 matching stories" in [c.value for c in app.caption]


def test_feed_ui_clamps_a_page_after_the_corpus_shrinks(feed_session: Session) -> None:
    app = _feed_app()
    _slider(app, "Stories per page").set_value(5).run()
    _button(app, "Next page").click().run()
    keep = [s.id for s in list_stories(feed_session, limit=2)]
    feed_session.execute(delete(StoryCluster).where(StoryCluster.id.not_in(keep)))
    feed_session.commit()
    app.run()
    assert not app.exception
    assert "Showing 1–2 of 2 matching stories" in [c.value for c in app.caption]
    assert _button(app, "Previous page").disabled
    assert _button(app, "Next page").disabled


def test_feed_ui_topic_and_time_filters_reset_the_page(feed_session: Session) -> None:
    app = _feed_app()
    _slider(app, "Stories per page").set_value(5).run()
    _button(app, "Next page").click().run()
    app.pills[0].set_value("chips_and_compute").run()
    assert not app.exception
    assert "Showing 1–2 of 2 matching stories" in [c.value for c in app.caption]
    app.pills[0].set_value("All topics").run()
    _button(app, "Next page").click().run()
    app.segmented_control[0].set_value("24 hours").run()
    assert not app.exception
    assert "No stories match these filters" in app.info[0].value
    app.segmented_control[0].set_value("All time").run()
    assert not app.exception
    assert "Showing 1–5 of 9 matching stories" in [c.value for c in app.caption]


def test_feed_ui_explains_an_empty_database(session: Session) -> None:
    app = _feed_app()
    assert not app.exception
    assert "No stories yet" in app.info[0].value
    assert "make demo" in app.info[0].value
