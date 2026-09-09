# Contributing to Panelary

Thank you for considering contributing to **Panelary**! We value every contribution — bug
reports, feature requests, docs, and code. Panelary is built on the foundations of
[functime](https://github.com/functime-org/functime) (Apache-2.0, no longer maintained) and
is licensed under Apache-2.0.

> **Note:** the project/brand is Panelary; the current import/PyPI package is `panelary`.
> The public rename to `panelary` is deferred — please don't rename packages or public APIs.

## Where contributions land: Tiers

Panelary is Rust + Polars under the hood, but **most contributions land in Python** — the Rust
barrier is intentionally low for new contributors.

- **Tier A — Rust kernels.** Performance-critical primitives in `src/`. Higher bar; usually
  reserved for hot loops that can't be expressed efficiently as Polars expressions.
- **Tier B — Polars-native Python.** New transforms, extractors, labelers, selectors, and
  comparisons written as **Polars expressions** in Python. This is the sweet spot: fast,
  lazy, no Rust required. Most new features belong here.
- **Tier C — Python glue & API.** Pipelines, `PanelFrame`, validation, docs, tests, examples.
  Great for first-time contributors.

If you're new, start with **Tier B or C**. You can do a lot without writing any Rust.

## The correctness contract: `panel_safe` & `leakage_safe`

Panelary's whole value proposition is correctness-by-construction. Any operation you add must
respect two contracts (see [docs/leakage.md](./docs/leakage.md)):

- **`panel_safe`** — within-entity operations stay inside their entity and run in time order
  (`.over(entity)` on a time-sorted panel). Never let one entity's data leak into another's.
- **`leakage_safe`** — cross-sectional ops are per-date (`.over(time)`), and anything with a
  `fit` step is fit per-fold on training data only — never globally, never on the test fold.

When you open a PR, please state which contract(s) your change touches and how you verified it
(e.g. a test that fails under a deliberately leaky implementation). PRs that introduce a way to
silently leak the future or the test set will not be merged.

## Good first issues

We label beginner-friendly work `good first issue`. Good entry points:

- Port a `tsfresh`-style feature to a Polars-native, `panel_safe` extractor (Tier B).
- Add a leak-safe transform or cross-sectional comparison (Tier B).
- Improve docs, examples, or the quickstart (Tier C).
- Add tests, including leakage regression tests (Tier C).

Clean-room contributions only: e.g. catch22 features must be reimplemented from
Lubba et al. (2019), **not** copied from the GPL `pycatch22`.

## Getting started

1. Fork and clone the repo.
2. Set up the dev environment. We use [**uv**](https://docs.astral.sh/uv/) — it is the same
   installer CI uses, and it is an order of magnitude faster than pip:

   ```bash
   # one-time: install uv (https://docs.astral.sh/uv/getting-started/installation/)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   uv venv                                    # create ./.venv
   source .venv/bin/activate
   uv pip install -e ".[dev,recommended]"     # editable install + dev tooling
   uv run pre-commit install --install-hooks
   ```

   No uv? Everything still works with plain pip (`python3 -m pip install -e ".[dev,recommended]"`).
   The [Makefile](./Makefile) picks uv automatically when it is available and falls back to pip
   otherwise: `make venv && make edit`, `make test`, `make lint`, `make typecheck`.
3. Run the test suite and `pre-commit` hooks before pushing:

   ```bash
   uv run pytest -q --ignore=tests/test_forecasting.py   # the 40-min forecasting suite is nightly
   uv run pre-commit run --all-files
   ```
4. Open a PR describing the change, the tier, and the leakage/panel safety implications.

For the longer developer walkthrough, see
[docs/developer-guide/contributing.md](./docs/developer-guide/contributing.md).
