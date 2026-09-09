# AGENTS.md

Instructions for AI coding agents working on **PanelKit** (import/PyPI package
`polars_features`, version 0.4.0). Human-facing docs live in `README.md`,
`CONTRIBUTING.md` and `docs/`; this file is the agent-facing contract.

## Project overview

PanelKit is a leak-safe, Polars-native feature-engineering and ML toolkit for
**panel data** (many entities observed over time). It is pure Python (the Rust
extension was dropped in 0.4.0 — the distribution is a single universal
`py3-none-any` wheel). It is derived from [functime](https://github.com/functime-org/functime)
(Apache-2.0) and retains that license; see `NOTICE`.

- **Brand name:** PanelKit. **Import name:** `polars_features`. These differ on
  purpose. The public rename to `panelkit` is **deferred** — do not rename the
  package, modules, or public APIs.
- Import convention used throughout the codebase and docs: `import polars_features as pk`.
- Requires Python **>=3.10**. Mandatory runtime deps are only `numpy` and
  `polars>=1.0.0`; everything heavier is an optional extra.

## Setup

CI uses `uv`; do the same locally.

```bash
uv venv
uv pip install -e ".[dev,recommended]"
```

`dev` brings ruff, mypy, pytest, pre-commit, build. `recommended` restores the
batteries-included dependency set (scikit-learn, scipy, …). `all` installs
every extra. There are 22 extras in total (`ml`, `scipy`, `lightgbm`,
`catboost`, `xgboost`, `forecasting`, `automl`, `seasonality`, `dimreduce`,
`explain`, `llm`, `cafe`, `viz`, `signatures`, `fast`, `progress`, `ann`,
`gpu`, `docs`, `dev`, `recommended`, `all`).

There is **no `uv.lock`** checked in — this is a library, so resolution is
unpinned by design.

## Build, lint, type-check, test

Run all of these before proposing any change. The `Makefile` wraps them
(`make lint`, `make typecheck`, `make test`, `make fmt`), but there is
currently **no single aggregate `make check`** — run them in sequence.

```bash
make lint        # ruff check . && ruff format --check .   (~0.1s)
make typecheck   # mypy over polars_features (config in pyproject.toml)
make test        # pytest -q --ignore=tests/test_forecasting.py
```

Equivalently, without make:

```bash
ruff check . && ruff format --check .
mypy
pytest -q -m "not slow"            # 1636 of 1664 tests
```

Timing notes for iterating efficiently:

- `ruff check` is effectively instant. Always run it.
- Collection alone is ~6.5s (1664 tests). **The full suite takes well over 15
  minutes locally.** Do not run it on every edit.
- **Run the narrowest file first**: e.g. `pytest tests/test_factor.py -q`
  (~1s of test time). Only run the full suite before finishing.
- `pytest-xdist` is *not* in the `dev` extra, so `-n auto` is unavailable
  unless you install it yourself.

`make venv` creates `./.venv`; the Makefile then prefers `uv pip` when `uv` is
on PATH and a venv is active, falling back to plain pip otherwise. Docs targets
are `make docs` / `make docs-serve`.

## Tests

- Flat layout: `tests/test_<area>.py`. Follow the existing naming when adding
  files (e.g. `tests/test_detect_bsadf.py`, `tests/test_econ_ivx.py`).
- `testpaths = ["tests"]`, `addopts = "-ra --strict-markers --strict-config"`.
  `--strict-markers` means **an unregistered marker is an error** — register
  new markers in `[tool.pytest.ini_options] markers` before using one.
- Registered markers: `slow`, `benchmark`, `multivariate`.
- Packaging guardrails are ordinary pytest files and run locally:
  `tests/test_import_hygiene.py`, `tests/test_dependency_drift.py`,
  `tests/test_wheel_guardrails.py`. If you add a dependency or change
  packaging, run these.

## The correctness contract (the thing that matters most)

PanelKit's value proposition is correctness-by-construction. Every operation
must respect two contracts (see `docs/leakage.md`, `CONTRIBUTING.md`):

- **`panel_safe`** — within-entity operations stay inside their entity and run
  in time order: `.over(entity_col)` on a time-sorted panel. One entity's data
  must never leak into another's.
- **`leakage_safe`** — cross-sectional ops are per-date (`.over(time_col)`),
  and anything with a `fit` step is fit **per fold on training data only** —
  never globally, never on the test fold.

Public transformers subclass `polars_features.core.protocol.PanelTransformer`
and **must set `panel_safe` / `leakage_safe` class attributes**.

Additional hard invariants observed by the newer subpackages (see
`plans/detect-build-contract.md`, which documents them explicitly):

1. **Prefix invariance** — `f(x[:T])[t] == f(x[:T+k])[t]` for all `t <= T`. No
   quantity may depend on `len(x)`: not window sizes, thresholds, lag orders,
   nor normalisation constants.
2. **Determinism** — no unseeded RNG; RNG takes an explicit `seed: int`.
3. **float64 everywhere** — upcast Float32 Polars columns before accumulating.
4. `np.linalg.solve(A, b[..., None])[..., 0]` — never `solve(A, b)` (NumPy 2.0
   mis-solves when `p == batch size`).
5. **Never `rolling_map`** (measured ~249x penalty). Prefer native rolling
   expressions; `map_batches` per group is the escape hatch.

When you change anything in this area, state which contract it touches and add
a test that fails under a deliberately leaky implementation. There are existing
leakage regression suites to model on: `tests/test_leakage.py`,
`tests/test_detect_leak_safety.py`, `tests/test_explain_leakage.py`,
`tests/test_econ_features_leakage.py`, `tests/test_reduce_factor_leakage.py`.

## Conventions

- **Panel keys.** `PanelFrame(df, entity="ticker", time="date")`; accessors are
  `.entity_col` / `.time_col`. Use those names for parameters and locals — the
  codebase says `entity` / `time` (constructor kwargs) and `entity_col` /
  `time_col` (attributes), not `id`/`ts`/`group`.
- **Expression namespaces.** Operators are exposed as
  `pl.col(...).<namespace>.<name>(...)` in the `panel`, `xs`, `ts` and `factor`
  namespaces. `.panel` implies `.over(entity)`; `.xs` implies `.over(time)`.
  Typed stubs live in `polars_features/namespaces/*.pyi` and ship in the wheel.
- **Operator registry.** New operators should register a `FeatureSpec` via
  `polars_features.registry.register_feature` (56 specs registered today:
  `ts`=42, `xs`=7, `factor`=4, `panel`=3). The spec carries the safety contract
  *and* provenance/license, and `registry.audit()` enforces permissive
  licensing. This is the machine-readable catalogue of the library — keep it
  populated.
- **Optional dependencies.** Import them **lazily, inside the function that
  needs them**, and route the import through `polars_features._deps.require`:

  ```python
  from polars_features._deps import require
  sklearn = require("sklearn", feature="mrmr selection")
  ```

  `require()` turns a missing dep into an actionable
  `pip install 'polars-features[ml]'` message. `_deps.py` must keep its
  zero-third-party-import property. Do **not** add a top-level
  `import sklearn` / `import scipy` to any module.
- **Typing.** The package ships `py.typed`. Annotate all public signatures;
  current coverage is 97.4% of public parameters and 84.4% of return types, and
  mypy is a blocking CI gate.
- **Docstrings.** NumPy style with `Parameters` / `Returns` / `Raises`. 79.9% of
  public callables have one; new public code should.

## Pre-commit

`.pre-commit-config.yaml` runs `ruff`, `ruff-format`, `check-yaml`,
`check-toml`, `end-of-file-fixer`, `trailing-whitespace`, `nbstripout`,
`validate-pyproject`, and `mypy`. Two things will trip you up:

- The **mypy hook is `language: system`** — it invokes whatever `mypy` is on
  `PATH`. If you installed into a venv that isn't active, the hook fails with
  "No module named mypy". Activate the venv or install mypy.
- **commitizen-branch** runs at `pre-push` stage and enforces
  [Conventional Commits](https://www.conventionalcommits.org/) on branch
  commits. Use `feat(scope): …`, `fix(scope): …`, `docs: …`, `chore: …`.

## Planning convention

Larger work is specified as a markdown plan in `plans/`, moved from
`plans/todo/` to `plans/done/` when implemented. `plans/detect-build-contract.md`
is a good template: hard invariants, a per-file ownership table, and explicit
dependency limits. If you are handed a plan file, treat it as the spec and only
touch the files it assigns to you.

## What NOT to do

- Do **not** rename `polars_features` to `panelkit`, or rename public APIs.
- Do **not** add a mandatory dependency. `numpy` + `polars` is the entire
  required footprint; anything else goes behind an extra and `require()`.
- Do **not** import optional deps at module top level — it breaks the bare-core
  install that `tests/test_import_hygiene.py` and the CI extras matrix defend.
- Do **not** copy code from copyleft (GPL/AGPL) upstreams. Contributions are
  **clean-room** — e.g. catch22 features are reimplemented from Lubba et al.
  (2019), not taken from `pycatch22`. Record `source` and `license` on the
  `FeatureSpec`.
- Do **not** reintroduce Rust / a compiled extension. 0.4.0 is deliberately
  pure-Python and `tests/test_wheel_guardrails.py` asserts the wheel is
  `py3-none-any` with no compiled artifacts.
- Do **not** use `rolling_map`, unseeded RNG, or any window/threshold that
  depends on total series length.
- Do **not** fit a scaler, selector, or estimator on the full sample when it
  will be evaluated per fold.
