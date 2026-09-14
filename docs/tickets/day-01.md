# Day 01 — Scaffold the monorepo and development standards

GitHub: [Issue #3](https://github.com/Justin-Mendoza/hybrid-rag-search/issues/3)
Depends on: none

## Problem

Every later search, ingestion, and evaluation ticket needs the same runnable project boundaries and automated quality checks. Without that foundation, setup choices drift and contributors cannot distinguish application failures from tooling failures.

## Outcome

A contributor on the target Apple Silicon development environment can install dependencies, run the API and web app independently, and execute all static and test checks through stable repository commands.

## P0 scope

- Establish `backend`, `frontend`, `infra`, `datasets`, and `docs` ownership boundaries.
- Add a typed FastAPI app with a `/health` endpoint and smoke test.
- Add a strict TypeScript Next.js app with a render smoke test.
- Configure formatting, linting, type checking, tests, pre-commit, and CI.
- Provide safe environment-variable examples and ignore secrets/generated files.
- Document Python, Node, npm, and Apple Silicon expectations.

## Non-goals

- Docker services belong to Day 2.
- Database models and migrations belong to Day 3.
- Product workflows, retrieval, and production deployment are intentionally absent.

## Acceptance criteria

- [x] `make backend-dev` serves `GET /health` successfully.
- [x] `make frontend-dev` serves the Next.js shell successfully.
- [x] `make test` passes one backend and one frontend smoke test.
- [x] `make check` verifies formatting, linting, types, tests, and builds.
- [x] CI is configured for backend checks on Python 3.12 and 3.13 and frontend checks on Node 22.
- [x] `.env` and generated dependency/build directories are ignored; examples contain placeholders only.
- [x] setup and Apple Silicon assumptions are documented.

## Verification

Run `make setup`, then `make check`. Start each application independently with the development commands in the root README.
