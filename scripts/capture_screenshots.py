"""Capture Streamlit UI screenshots for the documentation.

Milestone 7 asks for screenshots.  Taking them by hand makes them impossible to
refresh honestly, so they are scripted: the script starts nothing and assumes a
Streamlit server is already serving the database you want pictured.

    make ui                      # in one shell
    make screenshots             # in another

Playwright is an optional extra (``uv sync --extra screenshots``) because the
application, tests and experiments must never depend on a browser.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

DEFAULT_BASE_URL = "http://localhost:8501"
DEFAULT_OUTDIR = Path("docs/screenshots")

# (filename, query string, wait-for-this-text, extra settle selector)
SHOTS: tuple[tuple[str, str, str], ...] = (
    ("top-stories.png", "page=top-stories", "Top stories"),
    ("story-detail.png", "page=story-detail", "Story detail"),
    ("search.png", "page=search", "Search"),
    ("experiments.png", "page=experiment-dashboard", "Experiment dashboard"),
    ("health.png", "page=system-health", "System health"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--story", default="", help="Story id for the detail page")
    parser.add_argument(
        "--settle-ms",
        type=int,
        default=2500,
        help="Extra wait after the heading appears, for tables and plots",
    )
    args = parser.parse_args(argv)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is not installed. Install the optional extra:\n"
            "  uv sync --extra screenshots && uv run playwright install chromium",
            file=sys.stderr,
        )
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        for filename, query, heading in SHOTS:
            if filename == "story-detail.png" and args.story:
                query = f"{query}&story={args.story}"
            url = f"{args.base_url}/?{query}"
            page.goto(url, wait_until="load")
            try:
                page.get_by_text(heading, exact=False).first.wait_for(timeout=30_000)
            except Exception as exc:  # a page that never renders is a failure worth seeing
                print(f"! {filename}: never showed {heading!r} ({exc})", file=sys.stderr)
                continue
            # Streamlit shows a "Running" status widget while a script executes;
            # shooting before it clears captures an empty page.
            with contextlib.suppress(Exception):
                page.locator('[data-testid="stStatusWidget"]').wait_for(
                    state="hidden", timeout=60_000
                )
            page.wait_for_timeout(args.settle_ms)
            target = args.outdir / filename
            page.screenshot(path=str(target), full_page=True)
            written.append(target)
            print(f"wrote {target}")
        browser.close()

    if not written:
        print("No screenshots captured; is the Streamlit server running?", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
