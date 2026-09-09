# Contributing to Panelary

Thank you for considering contributing to **Panelary**! We value every contribution — bug
reports, feature requests, docs, and code. Panelary is built on the foundations of
[functime](https://github.com/functime-org/functime) (Apache-2.0) and is licensed under
Apache-2.0.

> **Names.** There is exactly one: the project is **Panelary**, the PyPI distribution is
> `panelary`, and the import is `panelary` — conventionally `import panelary as pn`. The
> pre-0.4.0 names `polars_features` and `PanelKit` are retired; please don't reintroduce them.

## Where contributions land: tiers

Panelary is **pure Python** — the Rust extension was dropped in 0.4.0 and the distribution is
a single universal `py3-none-any` wheel. There is no compiler in the loop and no build step
beyond `pip install -e .`, so the barrier for a first contribution is deliberately low.

- **Tier A — Polars-native operators.** New transforms, extractors, labelers, selectors and
  comparisons written as **Polars expressions**. This is the sweet spot: fast, lazy, and the
  home of most new features. Performance comes from expressing the work as a Polars
  expression, not from dropping to a lower-level language — never reach for `rolling_map`
  (measured ~249× penalty); `map_batches` per group is the escape hatch.
- **Tier B — NumPy kernels.** Numerical routines that genuinely cannot be expressed as a
  Polars expression, written in vectorised NumPy against float64 and driven per group.
- **Tier C — Python glue & API.** Pipelines, `PanelFrame`, validation, docs, tests, examples.
  Great for first-time contributors.

If you're new, start with **Tier A or C**.

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

- Port a `tsfresh`-style feature to a Polars-native, `panel_safe` extractor (Tier A).
- Add a leak-safe transform or cross-sectional comparison (Tier A).
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
   otherwise: `make venv && make edit`, then `make check` (lint + typecheck + test) or the
   individual `make lint`, `make typecheck`, `make test`.
3. Run the test suite and `pre-commit` hooks before pushing:

   ```bash
   uv run pytest -q -n auto --dist loadfile --ignore=tests/test_forecasting.py
   uv run pre-commit run --all-files
   ```
4. Open a PR describing the change, the tier, and the leakage/panel safety implications.

For the longer developer walkthrough, see
[docs/developer-guide/contributing.md](./docs/developer-guide/contributing.md).
