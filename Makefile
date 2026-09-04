.DEFAULT_GOAL := help

BASE ?= main
PY ?= python3

edit:  ## Editable install (pure-Python; no compiler needed)
	$(PY) -m pip install -e .

build:  ## Install the package
	$(PY) -m pip install .

build-test:  ## Install with dev extras
	$(PY) -m pip install ".[dev]"

wheel:  ## Build the universal py3-none-any wheel + sdist
	$(PY) -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	$(PY) -m pip uninstall polars_features -y

rebuild: clean build

.PHONY: help
help:  ## Display this help screen
  @echo -e '\033[1mAvailable commands:\033[0m'
  @grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
