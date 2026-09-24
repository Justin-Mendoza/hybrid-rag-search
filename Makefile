.PHONY: setup dev backend-dev frontend-dev stack-up stack-down stack-status stack-logs stack-reset db-upgrade db-downgrade db-current db-revision db-seed db-test tokenizer-setup ingestion-recover index-rebuild-start index-rebuild-promote opensearch-lexical-test opensearch-dense-test opensearch-hybrid-test cohere-smoke format format-check lint typecheck test build check

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/pip install -e 'backend[dev]'
	npm --prefix frontend ci
	.venv/bin/pre-commit install

dev: setup
	@set -eu; \
	.venv/bin/uvicorn hybrid_rag_search.main:app --app-dir backend/src --reload --port 8000 & backend_pid=$$!; \
	npm --prefix frontend run dev & frontend_pid=$$!; \
	trap 'kill $$backend_pid $$frontend_pid 2>/dev/null || true' EXIT INT TERM; \
	printf 'Waiting for the frontend'; \
	until curl --fail --silent http://localhost:3000 >/dev/null 2>&1; do printf '.'; sleep 1; done; \
	printf '\nOpening http://localhost:3000\n'; \
	open http://localhost:3000; \
	wait

backend-dev:
	.venv/bin/uvicorn hybrid_rag_search.main:app --app-dir backend/src --reload --port 8000

frontend-dev:
	npm --prefix frontend run dev

stack-up:
	docker compose up --build --detach --wait

stack-down:
	docker compose down

stack-status:
	docker compose ps

stack-logs:
	docker compose logs --follow --tail=100

stack-reset:
	@if [ "$(CONFIRM)" != "1" ]; then \
		printf '%s\n' 'Refusing to delete local stack data.' 'Run: make stack-reset CONFIRM=1'; \
		exit 1; \
	fi
	docker compose down --volumes --remove-orphans

db-upgrade:
	.venv/bin/alembic --config backend/alembic.ini upgrade head

db-downgrade:
	.venv/bin/alembic --config backend/alembic.ini downgrade -1

db-current:
	.venv/bin/alembic --config backend/alembic.ini current

db-revision:
	@test -n "$(MESSAGE)" || (printf '%s\n' 'Usage: make db-revision MESSAGE="describe change"'; exit 1)
	.venv/bin/alembic --config backend/alembic.ini revision --autogenerate --message "$(MESSAGE)"

db-seed:
	.venv/bin/python -m hybrid_rag_search.seed

db-test: db-upgrade
	.venv/bin/pytest backend -m integration --no-cov

tokenizer-setup:
	.venv/bin/python -m hybrid_rag_search.tokenizer_setup

ingestion-recover:
	.venv/bin/python -m hybrid_rag_search.ingestion_recovery

index-rebuild-start:
	@test -n "$(BUILD_ID)" || (printf '%s\n' 'Usage: make index-rebuild-start BUILD_ID=20260920'; exit 1)
	.venv/bin/python -m hybrid_rag_search.index_rebuild start "$(BUILD_ID)"

index-rebuild-promote:
	@test -n "$(INDEX)" || (printf '%s\n' 'Usage: make index-rebuild-promote INDEX=hybrid-rag-chunks-v2-20260920'; exit 1)
	.venv/bin/python -m hybrid_rag_search.index_rebuild promote "$(INDEX)"

opensearch-lexical-test:
	.venv/bin/pytest backend/tests/test_lexical_retrieval_integration.py -m integration --no-cov

opensearch-dense-test:
	.venv/bin/pytest backend/tests/test_dense_retrieval_integration.py -m integration --no-cov

opensearch-hybrid-test:
	.venv/bin/pytest backend/tests/test_hybrid_retrieval_integration.py -m integration --no-cov

cohere-smoke:
	RUN_LIVE_COHERE_TESTS=1 .venv/bin/pytest backend/tests/test_cohere_live.py -m live --no-cov -s

format:
	.venv/bin/ruff format backend
	.venv/bin/ruff check --fix backend
	npm --prefix frontend run format

format-check:
	.venv/bin/ruff format --check backend
	npm --prefix frontend run format:check

lint:
	.venv/bin/ruff check backend
	npm --prefix frontend run lint

typecheck:
	.venv/bin/mypy backend/src
	npm --prefix frontend run typecheck

test:
	.venv/bin/pytest backend
	npm --prefix frontend test

build:
	npm --prefix frontend run build

check: format-check lint typecheck test build
