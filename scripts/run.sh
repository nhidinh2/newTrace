#!/usr/bin/env bash
# Run a command in the project environment.
#
# The Makefile used to hard-code `uv run`, which turns a missing uv into
# "make: uv: No such file or directory" on every target -- on a checkout that
# has a perfectly good .venv. This prefers uv when it is installed and falls
# back to the virtualenv when it is not, the same way scripts/refresh.sh does.
#
#   scripts/run.sh pytest -q
#   scripts/run.sh python scripts/run_experiments.py
#   scripts/run.sh --sync            # install/refresh dependencies

set -uo pipefail

cd "$(dirname "$0")/.."

# Keep this in step with the optional-dependency groups in pyproject.toml; the
# pip path has no lockfile to read them from.
EXTRAS="embeddings,extraction,ui,reports,llm,screenshots,dev"

have_uv() { command -v uv >/dev/null 2>&1; }

if [ "${1:-}" = "--sync" ]; then
  if have_uv; then
    exec uv sync --all-extras
  fi
  if [ ! -x .venv/bin/python ]; then
    echo "Creating .venv (uv is not installed)" >&2
    python3 -m venv .venv || exit 1
  fi
  echo "uv is not installed; installing with pip into .venv" >&2
  .venv/bin/python -m pip install --upgrade pip >/dev/null || exit 1
  exec .venv/bin/python -m pip install -e ".[${EXTRAS}]"
fi

if [ "$#" -eq 0 ]; then
  echo "usage: scripts/run.sh <command> [args...]   (or --sync)" >&2
  exit 2
fi

if have_uv; then
  exec uv run "$@"
fi

command="$1"
shift
if [ -x ".venv/bin/${command}" ]; then
  exec ".venv/bin/${command}" "$@"
fi

if [ -x .venv/bin/python ]; then
  # Not every tool installs a console script; python -m covers the rest.
  exec .venv/bin/python -m "$command" "$@"
fi

echo "Found neither uv nor .venv/bin/python. Run 'make setup' first." >&2
exit 1
