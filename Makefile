.PHONY: setup dev backend-dev frontend-dev format format-check lint typecheck test build check

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
