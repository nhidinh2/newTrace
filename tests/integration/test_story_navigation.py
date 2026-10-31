"""Story deep links against the real page and an isolated SQLite database.

Install the ``ui`` extra to run these Streamlit interaction regressions.
AppTest versions expose query parameter values as either strings or lists.
"""

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from newstrace.models import StoryCluster

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"


@pytest.fixture
def story_page(session: Session) -> AppTest:
    session.add_all(
        [
            StoryCluster(id=1, display_title="Earlier story"),
            StoryCluster(id=2, display_title="Later story"),
        ]
    )
    session.commit()
    # Mount the story page directly; exercise its real widgets and data services.
    source = APP.read_text().rsplit("\nmain()", 1)[0]
    return AppTest.from_string(source + "\npage_story_detail()\n")


@pytest.mark.parametrize(
    "requested",
    ["", "not-a-story", "1.5", "²", "9" * 5000, str(2**63), "-1", "0", "999"],
    ids=[
        "empty",
        "text",
        "decimal",
        "unicode-digit",
        "long-integer",
        "sqlite-overflow",
        "negative",
        "zero",
        "unknown",
    ],
)
def test_invalid_story_link_falls_back_to_picker(story_page: AppTest, requested: str) -> None:
    story_page.query_params["story"] = requested
    story_page.run(timeout=20)

    assert not story_page.exception
    assert story_page.selectbox[0].value == 2
    assert story_page.query_params["story"] in ("2", ["2"])


def test_valid_story_link_and_picker_remain_navigable(story_page: AppTest) -> None:
    story_page.query_params["story"] = "1"
    story_page.run(timeout=20)
    assert not story_page.exception
    assert story_page.selectbox[0].value == 1

    story_page.selectbox[0].select(2).run(timeout=20)
    assert not story_page.exception
    assert story_page.selectbox[0].value == 2
    assert story_page.query_params["story"] in ("2", ["2"])

    story_page.query_params["story"] = "bad-link"
    story_page.run(timeout=20)
    assert not story_page.exception
    assert story_page.selectbox[0].value == 2


def test_valid_story_link_outside_recent_picker_limit(
    story_page: AppTest, session: Session
) -> None:
    session.add_all(StoryCluster(id=i, display_title=f"Story {i}") for i in range(3, 203))
    session.commit()
    story_page.query_params["story"] = "1"
    story_page.run(timeout=20)

    assert not story_page.exception
    assert story_page.selectbox[0].value == 1
    assert story_page.query_params["story"] in ("1", ["1"])
