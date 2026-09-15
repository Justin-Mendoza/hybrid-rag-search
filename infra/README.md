# Infrastructure

This boundary owns local infrastructure configuration and, later, observability assets. The root `compose.yaml` is the entry point so standard `docker compose` commands work without extra flags.

Day 2 pins native ARM64 images for Python 3.12.14, Node.js 22.23.2, PostgreSQL 17.11, Redis 8.10.1, and OpenSearch 3.6.0 LTS. OpenSearch security is disabled only for this local environment, and all published ports bind to `127.0.0.1`.

Use the root Makefile to manage the environment:

- `make stack-up` builds, starts, and waits for every health check.
- `make stack-down` removes containers while preserving named volumes.
- `make stack-status` and `make stack-logs` expose health and diagnostics.
- `make stack-reset CONFIRM=1` removes the containers and all named volumes.
