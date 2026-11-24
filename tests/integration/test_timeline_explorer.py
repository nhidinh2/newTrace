"""Timeline exploration retains full-story evidence and publication order."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.api.app import create_app
from newstrace.models import Article, ClusterMembership, StoryCluster
from tests.conftest import BASE_TIME

REPEATED = "The council approved a new public transport plan."
UPDATE = "Engineers inspected damaged tunnels before reopening central stations."
UNDATED = "Independent auditors discovered missing records during financial inspections."


@pytest.fixture
def timeline_story(session: Session) -> tuple[int, list[int]]:
    story = StoryCluster(
        display_title="Transit recovery",
        article_count=18,
        distinct_domain_count=2,
        first_published_at=BASE_TIME,
        last_published_at=BASE_TIME + timedelta(hours=15),
    )
    session.add(story)
    session.flush()
    specs = [
        ("alpha.example", 0, REPEATED, False),
        ("beta.example", 1, REPEATED, False),
        ("wire.example", 2, REPEATED, True),
        ("beta.example", 3, UPDATE, False),
        ("beta.example", None, REPEATED, False),
        ("", None, UNDATED, False),
        *[("alpha.example", hour, REPEATED, False) for hour in range(4, 16)],
    ]
    ids = []
    for index, (domain, hour, text, duplicate) in enumerate(specs):
        url = f"https://{domain or 'unknown.example'}/{index}"
        article = Article(
            url=url,
            canonical_url=url,
            title=f"Report {index}",
            description=text,
            source_domain=domain,
            published_at=BASE_TIME + timedelta(hours=hour) if hour is not None else None,
            is_near_duplicate=duplicate,
            normalized_text_hash=url,
            normalized_title_hash=url,
            duplicate_of_article_id=ids[0] if duplicate else None,
        )
        session.add(article)
        session.flush()
        ids.append(article.id)
        session.add(
            ClusterMembership(story_cluster_id=story.id, article_id=article.id, similarity=0.8)
        )
    session.commit()
    return story.id, ids


def test_timeline_api_filters_keep_prior_evidence_and_first_label(
    session: Session, timeline_story
) -> None:
    story_id, ids = timeline_story
    with TestClient(create_app()) as client:
        full = client.get(f"/stories/{story_id}/timeline").json()
        filtered = client.get(
            f"/stories/{story_id}/timeline", params={"source_domain": "beta.example"}
        ).json()
        updates = client.get(
            f"/stories/{story_id}/timeline",
            params={"source_domain": "beta.example", "only_updates": True},
        ).json()
    assert full["total_entries"] == full["matching_entries"] == 18
    assert len(full["entries"]) == 18 and full["limit"] is None
    assert filtered["total_entries"] == 18 and filtered["matching_entries"] == 3
    assert [e["article"]["id"] for e in filtered["entries"]] == [ids[1], ids[3], ids[4]]
    assert filtered["entries"] == [
        e for e in full["entries"] if e["article"]["source_domain"] == "beta.example"
    ]
    assert not any(e["is_first_report"] for e in filtered["entries"])
    assert filtered["entries"][0]["new_sentences"] == []
    assert filtered["entries"][0]["repeated_sentences"]
    assert [e["article"]["id"] for e in updates["entries"]] == [ids[3]]
    assert updates["matching_entries"] == 1
    assert "lexical heuristic against the full story" in updates["note"]


def test_timeline_api_paginates_after_filtering_and_handles_empty_pages(
    session: Session, timeline_story
) -> None:
    story_id, ids = timeline_story
    with TestClient(create_app()) as client:
        url = f"/stories/{story_id}/timeline"
        whole = client.get(url, params={"include_duplicates": False}).json()
        pages = [
            client.get(
                url, params={"include_duplicates": False, "limit": 4, "offset": offset}
            ).json()
            for offset in range(0, 20, 4)
        ]
        missing = client.get(
            url, params={"source_domain": "missing.example", "offset": 4, "limit": 2}
        ).json()
        unknown = client.get(url, params={"source_domain": ""}).json()
    assert [e for page in pages for e in page["entries"]] == whole["entries"]
    assert whole["matching_entries"] == 17
    assert all(
        p["total_entries"] == 18 and p["matching_entries"] == 17 and p["limit"] == 4 for p in pages
    )
    assert pages[-1]["offset"] == 16
    assert [e["article"]["id"] for e in whole["entries"]][-2:] == [ids[4], ids[5]]
    assert missing["entries"] == [] and missing["matching_entries"] == 0
    assert missing["total_entries"] == 18 and missing["offset"] == 4
    assert [e["article"]["id"] for e in unknown["entries"]] == [ids[5]]


@pytest.mark.parametrize(
    "params", [{"limit": 0}, {"limit": 201}, {"offset": -1}, {"only_updates": "invalid"}]
)
def test_timeline_api_rejects_invalid_controls(session: Session, timeline_story, params) -> None:
    with TestClient(create_app()) as client:
        assert (
            client.get(f"/stories/{timeline_story[0]}/timeline", params=params).status_code == 422
        )


def test_timeline_api_empty_story_and_missing_story(session: Session) -> None:
    story = StoryCluster(display_title="Empty")
    session.add(story)
    session.commit()
    with TestClient(create_app()) as client:
        result = client.get(
            f"/stories/{story.id}/timeline", params={"only_updates": True, "limit": 10}
        ).json()
        assert result["total_entries"] == result["matching_entries"] == 0
        assert result["entries"] == []
        assert (
            client.get("/stories/999999/timeline", params={"only_updates": True}).status_code == 404
        )


def timeline_app():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    path = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
    return AppTest.from_string(
        f"import runpy\nrunpy.run_path({str(path)!r})['page_story_detail']()",
        default_timeout=30,
    ).run()


def widget(app, kind: str, label: str):
    return next(w for w in getattr(app, kind) if w.label == label)


def map_data(app):
    charts = app.get("vega_lite_chart")
    assert len(charts) == 1
    spec = json.loads(charts[0].proto.spec)
    import pyarrow as pa

    frame = pa.ipc.open_stream(charts[0].proto.data.data).read_all().to_pandas()
    return spec, frame


def test_timeline_ui_map_and_pagination_preserve_context(session: Session, timeline_story) -> None:
    _, ids = timeline_story
    app = timeline_app()
    assert not app.exception
    assert "Showing 1–10 of 18 matching reports · 18 total reports" in [
        c.value for c in app.caption
    ]
    spec, frame = map_data(app)
    assert spec["encoding"]["x"]["scale"]["type"] == "utc"
    assert len(frame) == 16
    assert set(frame["Text signal"]) == {
        "Potentially new text",
        "No new-text candidates",
        "Near-duplicate",
    }
    assert ids[4] not in set(frame["Article ID"]) and ids[5] not in set(frame["Article ID"])
    widget(app, "button", "Next reports").click().run()
    assert not app.exception
    assert "Showing 11–18 of 18 matching reports · 18 total reports" in [
        c.value for c in app.caption
    ]
    assert len(map_data(app)[1]) == 16
    assert widget(app, "button", "Next reports").disabled
    widget(app, "selectbox", "Timeline source").set_value("beta.example").run()
    assert not app.exception
    assert "Showing 1–3 of 3 matching reports · 18 total reports" in [c.value for c in app.caption]
    assert widget(app, "button", "Previous reports").disabled
    assert not any("earliest available report" in e.label for e in app.expander)
    assert list(map_data(app)[1]["Article ID"]) == [ids[1], ids[3]]
    widget(app, "checkbox", "Only potentially new text").check().run()
    assert not app.exception
    assert "Showing 1–1 of 1 matching reports · 18 total reports" in [c.value for c in app.caption]
    assert list(map_data(app)[1]["Article ID"]) == [ids[3]]
    assert any(UPDATE in m.value for m in app.markdown)


def test_timeline_ui_duplicates_undated_and_empty_filters(session: Session, timeline_story) -> None:
    app = timeline_app()
    widget(app, "checkbox", "Include near-duplicates").uncheck().run()
    assert not app.exception
    assert len(map_data(app)[1]) == 15
    widget(app, "selectbox", "Timeline source").set_value("").run()
    assert not app.exception
    assert not app.get("vega_lite_chart")
    assert any("no publication dates" in i.value for i in app.info)
    assert any(UNDATED in m.value for m in app.markdown)
    widget(app, "selectbox", "Timeline source").set_value("wire.example").run()
    assert not app.exception
    assert any("No reports match" in i.value for i in app.info)
    widget(app, "checkbox", "Include near-duplicates").check().run()
    assert not app.exception
    assert list(map_data(app)[1]["Text signal"]) == ["Near-duplicate"]
    widget(app, "checkbox", "Only potentially new text").check().run()
    assert any("No reports match" in i.value for i in app.info)


def test_timeline_ui_clamps_shrinking_story_and_resets_page_size(
    session: Session, timeline_story
) -> None:
    story_id, ids = timeline_story
    app = timeline_app()
    widget(app, "button", "Next reports").click().run()
    widget(app, "selectbox", "Reports per page").set_value(25).run()
    assert not app.exception
    assert "Showing 1–18 of 18 matching reports · 18 total reports" in [
        c.value for c in app.caption
    ]
    widget(app, "selectbox", "Reports per page").set_value(10).run()
    widget(app, "button", "Next reports").click().run()
    for membership in session.scalars(
        select(ClusterMembership).where(
            ClusterMembership.story_cluster_id == story_id,
            ClusterMembership.article_id.not_in(ids[:2]),
        )
    ):
        session.delete(membership)
    session.commit()
    app.run()
    assert not app.exception
    assert "Showing 1–2 of 2 matching reports · 2 total reports" in [c.value for c in app.caption]
    assert widget(app, "button", "Previous reports").disabled


def test_timeline_ui_story_switch_resets_controls_and_handles_empty_story(
    session: Session, timeline_story
) -> None:
    app = timeline_app()
    widget(app, "button", "Next reports").click().run()
    widget(app, "selectbox", "Timeline source").set_value("beta.example").run()
    other = StoryCluster(display_title="Another story without reports")
    session.add(other)
    session.commit()
    app.run()
    widget(app, "selectbox", "Story").set_value(other.id).run()
    assert not app.exception
    assert widget(app, "selectbox", "Timeline source").value is None
    assert not app.get("vega_lite_chart")
    assert any("No reports in this story" in i.value for i in app.info)


def test_timeline_map_uses_utc_and_never_invents_dates(session: Session, timeline_story) -> None:
    _, ids = timeline_story
    # SQLite stores naive UTC values. The shared timeline normalizes them before plotting.
    article = session.get(Article, ids[0])
    assert article is not None
    article.published_at = BASE_TIME.replace(hour=23, minute=45)
    session.commit()
    app = timeline_app()
    assert not app.exception
    spec, frame = map_data(app)
    original = frame.loc[frame["Article ID"] == ids[0]].iloc[0]
    assert original["Published"] == "2026-08-18T23:45:00+00:00"
    assert original["Published (UTC)"] == "2026-08-18 23:45 UTC"
    assert {"field": "Published (UTC)", "type": "nominal"} in spec["encoding"]["tooltip"]
    assert not frame["Published"].isna().any()
