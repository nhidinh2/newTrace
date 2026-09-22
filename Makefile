# NewsTrace convenience targets.
#
# Commands run through scripts/run.sh, which uses uv when it is installed and
# the project virtualenv when it is not -- a missing uv used to fail every
# target on a checkout that already had a working .venv.

PY ?= scripts/run.sh

.DEFAULT_GOAL := help
.PHONY: help setup db demo api ui refresh ingest experiment ann beir backfill report labels screenshots test coverage lint format typecheck check fixtures clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies and create .env (uv if present, pip into .venv otherwise)
	$(PY) --sync
	@test -f .env || cp .env.example .env
	@echo "Setup complete. Run 'make db' next."

db:  ## Apply database migrations
	$(PY) alembic upgrade head

demo:  ## Ingest committed fixtures and list stories (no network, no API keys)
	$(PY) python scripts/ingest_demo.py --source fixtures

api:  ## Serve the FastAPI application
	$(PY) uvicorn newstrace.api.app:app --reload

ui:  ## Serve the Streamlit interface
	$(PY) streamlit run app/streamlit_app.py

refresh:  ## Pull fresh RSS + GDELT articles (override TOPIC/TIMESPAN)
	TIMESPAN=$(or $(TIMESPAN),24h) scripts/refresh.sh $(TOPIC)

ingest:  ## Live GDELT ingestion (override TOPIC/TIMESPAN/MAX)
	$(PY) newstrace ingest --source gdelt $(if $(TOPIC),--topic $(TOPIC),) \
		--timespan $(or $(TIMESPAN),24h) --max-records $(or $(MAX),250)

experiment:  ## Run the dimensionality-reduction sweep
	$(PY) python scripts/run_experiments.py \
		--methods full,svd,gaussian_rp,sparse_rp \
		--dimensions 32,64,128,256 \
		--seed 549

ann:  ## Sweep approximate retrieval (IVF / IVF-PQ) against exact search
	$(PY) python scripts/run_ann.py $(if $(SINCE),--since-days $(SINCE),)

beir:  ## Run the compression sweep on a BEIR dataset (DATASET=scifact, DOWNLOAD=1 first time)
	$(PY) python scripts/run_beir.py --dataset $(or $(DATASET),scifact) \
		$(if $(DOWNLOAD),--download,)

backfill:  ## Build blocking keys and the full-text index for an existing database
	$(PY) python scripts/backfill_search_indexes.py

report:  ## Re-render the report for the latest experiment
	$(PY) python scripts/export_report.py

labels:  ## Build a labelling queue of borderline article pairs
	$(PY) python scripts/build_label_queue.py

screenshots:  ## Capture docs/screenshots from a running `make ui` (needs the screenshots extra)
	$(PY) python scripts/capture_screenshots.py $(if $(STORY),--story $(STORY),)

fixtures:  ## Regenerate the synthetic fixtures
	$(PY) python scripts/make_fixtures.py

test:  ## Run the test suite (network tests excluded)
	$(PY) pytest -q

coverage:  ## Run the test suite with a coverage report
	$(PY) pytest -q --cov=newstrace --cov-report=term-missing

lint:  ## Lint and check formatting
	$(PY) ruff format --check .
	$(PY) ruff check .

format:  ## Apply formatting and autofixes
	$(PY) ruff format .
	$(PY) ruff check --fix .

typecheck:  ## Static type check
	$(PY) mypy src

check: lint typecheck test  ## Lint, typecheck and test

clean:  ## Remove caches and the local database
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f newstrace.db newstrace.db-wal newstrace.db-shm
