VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
FL      := $(VENV)/bin/frigate-learn
PYTHON  ?= python3
CONFIG  ?= config.yaml
HOST    ?= 127.0.0.1
PORT    ?= 8080

.PHONY: help setup venv install install-dev install-web install-ml \
	install-all db-init db-migrate db-stats web test test-web lint \
	fmt fmt-check run status clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: venv install-web ## Create venv and install dev + web extras (dashboard)
	$(PIP) install -e ".[dev,web]"

setup-ml: venv install-all ## Create venv and install all extras incl. training

venv: $(VENV)/bin/python ## Create the virtualenv if missing

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)

install: $(VENV)/bin/python ## Editable install with base dependencies
	$(PIP) install -e "."

install-dev: $(VENV)/bin/python ## + dev extras (pytest, ruff)
	$(PIP) install -e ".[dev]"

install-web: $(VENV)/bin/python ## + web extras (fastapi, uvicorn)
	$(PIP) install -e ".[web]"

install-ml: $(VENV)/bin/python ## + ml extras (torch, ultralytics)
	$(PIP) install -e ".[ml]"

install-all: $(VENV)/bin/python ## All extras (dev + web + ml)
	$(PIP) install -e ".[dev,web,ml]"

db-init: ## Create the database and apply all migrations
	$(FL) --config $(CONFIG) db init

db-migrate: ## Apply pending migrations
	$(FL) --config $(CONFIG) db migrate

db-stats: ## Show database statistics
	$(FL) --config $(CONFIG) db stats

web: ## Run the dashboard (HOST/PORT overridable)
	$(FL) web --host $(HOST) --port $(PORT)

test: ## Run the full test suite
	$(PY) -m pytest -p no:warnings

test-web: ## Run webapp tests only
	$(PY) -m pytest -p no:warnings tests/test_webapp.py -v

lint: ## Ruff lint check
	$(VENV)/bin/ruff check src tests

fmt: ## Auto-format with ruff
	$(VENV)/bin/ruff format src tests

fmt-check: ## Check formatting without changing files
	$(VENV)/bin/ruff format --check src tests

run: ## Run the pipeline (dry-run default); pass RUN_ARGS e.g. RUN_ARGS="--real --until train"
	$(FL) run $(RUN_ARGS)

status: ## Overview of collected data and recent jobs
	$(FL) --config $(CONFIG) status

clean: ## Remove bytecode and cache dirs (keeps data/)
	find src tests -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache