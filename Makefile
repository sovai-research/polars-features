.DEFAULT_GOAL := help

BASE ?= main
PY ?= python3

# --- Installer selection: uv when it is usable, pip otherwise -----------------
# `uv pip` is a drop-in, ~10-100x faster replacement for `python -m pip`, and CI
# (.github/workflows/*.yml) already installs everything with it. Locally we use
# it only when uv is on PATH *and* there is an environment for it to target
# (an active $VIRTUAL_ENV, or a ./.venv created by `make venv`), because
# `uv pip` deliberately refuses to touch an externally-managed system
# interpreter. Everywhere else we fall back to plain pip, so this Makefile keeps
# working with no uv installed. Force the fallback with `make edit UV=`.
UV ?= $(shell command -v uv 2>/dev/null)
UV_TARGET := $(strip $(VIRTUAL_ENV)$(wildcard .venv))

ifeq ($(strip $(UV)),)
  PIP := $(PY) -m pip
  UNINSTALL := $(PY) -m pip uninstall -y
  BUILD := $(PY) -m build
else ifeq ($(UV_TARGET),)
  PIP := $(PY) -m pip
  UNINSTALL := $(PY) -m pip uninstall -y
  BUILD := uv build
else
  PIP := uv pip
  UNINSTALL := uv pip uninstall
  BUILD := uv build
endif

.PHONY: venv edit build build-test wheel clean rebuild test lint fmt typecheck

venv:  ## Create ./.venv with uv (falls back to python -m venv)
	@if [ -n "$(UV)" ]; then uv venv; else $(PY) -m venv .venv; fi
	@echo "Activate it with:  source .venv/bin/activate"

edit:  ## Editable install (pure-Python; no compiler needed)
	$(PIP) install -e .

build:  ## Install the package
	$(PIP) install .

build-test:  ## Install with dev extras
	$(PIP) install ".[dev]"

wheel:  ## Build the universal py3-none-any wheel + sdist
	$(BUILD)

test:  ## Run the test suite (skips the slow forecasting file)
	$(PY) -m pytest -q --ignore=tests/test_forecasting.py

lint:  ## ruff check + ruff format --check (same versions as CI / pre-commit)
	ruff check .
	ruff format --check .

fmt:  ## Auto-fix lint findings and format the tree
	ruff check --fix .
	ruff format .

typecheck:  ## mypy over the package (config in pyproject.toml)
	$(PY) -m mypy polars_features

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	$(UNINSTALL) polars_features

rebuild: clean build

# --- Documentation (mkdocs-material; config in mkdocs.yml, sources in docs/) ---
# `docs` collides with the docs/ directory name, so these must be .PHONY or
# make would consider the target already up to date.
.PHONY: docs docs-serve docs-deps docs-clean

docs-deps:  ## Install the documentation toolchain (the `docs` extra)
	$(PIP) install -e ".[docs]"

docs:  ## Build the static docs site into ./site (fails on any warning)
	$(PY) -m mkdocs build --strict

docs-serve:  ## Live-reloading docs preview on http://127.0.0.1:8000
	$(PY) -m mkdocs serve -a 127.0.0.1:8000

docs-clean:  ## Remove the built docs site
	rm -rf site/

.PHONY: help
help:  ## Display this help screen
	@printf '\033[1mAvailable commands:\033[0m\n'
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
