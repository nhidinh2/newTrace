#!/usr/bin/env bash
# Pull fresh articles from the live feeds, embed and cluster them.
#
# Both adapters run with --refresh on purpose. The HTTP layer keeps a
# content-addressed response cache (see ingestion/http.py) that never
# expires: it exists so an experiment can be replayed from disk, which is
# the opposite of what a live pull wants. Without --refresh an identical
# GDELT query replays the stored response and inserts nothing.
#
#   scripts/refresh.sh                  # every configured topic
#   scripts/refresh.sh ai_models        # one topic
#   TIMESPAN=7d scripts/refresh.sh      # widen the GDELT window
#
# The Streamlit UI reads the database per request, so a browser refresh is
# enough to see the new stories; leave `make ui` running.

set -uo pipefail

cd "$(dirname "$0")/.."

TIMESPAN="${TIMESPAN:-24h}"
TOPIC="${1:-}"

# The Makefile assumes uv. Fall back to the virtualenv so the script still
# works on a checkout where uv was never installed.
if command -v uv >/dev/null 2>&1; then
  run() { uv run "$@"; }
elif [ -x .venv/bin/newstrace ]; then
  run() { .venv/bin/"$@"; }
else
  echo "Found neither uv nor .venv/bin/newstrace. Run 'make setup' first." >&2
  exit 1
fi

# `newstrace ingest` exits 0 even when the run fails -- a rate-limited GDELT
# fetch reports status=failed and still returns success -- so the exit code
# cannot be trusted here. The run record is the source of truth.
last_run_status() {
  run python -c "
from newstrace.db import session_scope
from newstrace.models import IngestionRun
with session_scope() as s:
    r = s.query(IngestionRun).order_by(IngestionRun.id.desc()).first()
    # One line, always: the caller reads this with a single \`read\`.
    detail = str(r.errors or '').replace('\n', ' ') if r else ''
    print(f'{r.status} {r.inserted_count} {detail}' if r else 'missing 0 ')
"
}

failed=0

ingest() {
  local label="$1"; shift
  echo "==> ${label}"
  run newstrace ingest "$@" ${TOPIC:+--topic "$TOPIC"}
  read -r status inserted errors <<<"$(last_run_status)"
  if [ "$status" != "completed" ]; then
    # Don't abort: a throttled GDELT fetch should not cost us the RSS pull,
    # and vice versa. Report it and carry on.
    echo "    ${label} run ${status}: ${errors:-no detail}" >&2
    failed=1
  else
    echo "    ${label}: ${inserted} new articles"
  fi
}

# RSS carries the reporting that corroborates; GDELT widens the source set
# beyond the feeds listed in configs/topics.live.yaml. Worth having both.
ingest RSS --source rss --refresh
ingest "GDELT (${TIMESPAN})" --source gdelt --refresh --timespan "$TIMESPAN"

if [ "$failed" -ne 0 ]; then
  echo "At least one adapter failed; the database still has whatever succeeded." >&2
  exit 1
fi
