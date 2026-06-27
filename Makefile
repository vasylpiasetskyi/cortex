SHELL := /bin/bash

.PHONY: lint
lint:  ## ruff lint check.
	uv run ruff check .

.PHONY: format
format:  ## format project code.
	uv run black .
	uv run ruff check --fix .

.PHONY: ci-lint
ci-lint:  ## lint check for CI (no auto-fix).
	uv run ruff check .
	uv run black --check .

.PHONY: test
test:  ## run tests.
	uv run pytest tests/ -v

.PHONY: install
install:  ## install dev dependencies.
	uv sync --extra dev

.PHONY: run
run:  ## start dev server.
	uv run uvicorn app.main:app --reload --port 8000

.PHONY: docker-up
docker-up:  ## start Redis and Postgres.
	docker compose up redis postgres -d

.PHONY: docker-down
docker-down:  ## stop all containers.
	docker compose down
