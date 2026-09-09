# Repository layout — build contract (issued 2026-09-09)

This is the layout standard for `panelary/`. It is not a proposal: the rule
below is what the package already does, what the test suite already enforces,
and what a change must not break. Where the package does not yet conform, the
gap is named in [Not done, and why](#not-done-and-why) rather than papered over.

Every number and every claim on this page was measured or read at the file:line
cited. Nothing here is inferred.

## The filing rule

One rule, applied to every directory under `panelary/`:

```
pkg/__init__.py     public re-exports + an explicit __all__, and nothing else
pkg/_impl.py        implementation; free to split, rename, merge, move
pkg/public.py       ONLY when that module path is itself documented API
```

`panelary/validation/` is the reference implementation: `__init__.py` is a
docstring plus four `from panelary.validation._<x> import (...)` blocks
(`panelary/validation/__init__.py:44-88`) and an `__all__`; all the code is in
`_bootstrap.py`, `_cv.py`, `_forecast_tests.py`, `_selection_stats.py`.

The third line is the exception, and it is narrow. A module keeps a public name
only when something outside the package names that path. There are exactly two
sources of such naming, and both are checked in CI:

1. a mkdocstrings identifier in `docs/` (42 of them; `make docs` runs
   `mkdocs build --strict` at `Makefile:86`, and `.github/workflows/docs.yml:11-12`
   turns every unresolved identifier into a build failure);
2. `tests/test_public_api.py`, which derives the surface from the filesystem —
   see [Enforcement](#enforcement).

## Hard invariants

1. **The public surface is filesystem-derived.** `tests/test_public_api.py:42-50`
   enumerates every non-underscore directory containing an `__init__.py`;
   `:119-127` enumerates every non-underscore top-level `.py`. Each must resolve
   as `pn.<name>` and be in `__all__` or `_LAZY_SUBMODULES`. **Creating a
   public-named file or directory creates a public API obligation.** Underscore
   names are excluded outright — that exclusion is the entire licence for
   `_internal/` and for module→package conversion.
2. **A module may become a package; a public module may not be renamed.**
   `MIGRATING.md:14-15` and `CHANGELOG.md:39` promise no compatibility shim,
   ever, and `MIGRATING.md:144` explains why. `X.py` → `X/__init__.py` preserves
   `panelary.X` and every `from panelary.X import y` exactly, so it is free.
   `X.py` → `Y/X.py` is a break with no shim available to soften it, and
   `AGENTS.md` ("What NOT to do") classes it as an explicit decision, not a refactor.
3. **A private filename is not private API.** See
   [The subtlety](#the-subtlety-a-private-filename-is-not-private-api).
4. **Light core.** `import panelary` must load no optional dependency
   (`tests/test_import_hygiene.py:94-102`), and each feature module must stay
   clean when imported on its own (`:105-124`). Design ceiling: **85 ms**.
   The automated ratchets are deliberately looser so a slow CI runner cannot
   fail the build on noise — `IMPORT_BUDGET_MS = 300`, scaled by the machine's
   own `import polars` cost (`tests/test_import_hygiene.py:42-47, 184-217`), and
   Panelary's overhead *net of polars* ≤ 150 ms (`:216-233`). Treat 85 ms as the
   number to defend and 300 ms as the number that fails the build.
5. **`forecasting`, `llm`, `plotting`, `backtesting` stay lazy.**
   `panelary/__init__.py:325-339` documents the measured cost of eager-importing
   each (`forecasting` 828 ms; `backtesting` took the cold import from 80 ms to
   779 ms; `plotting` 143 ms; `llm` raises without its extra) and resolves them
   through the PEP 562 `__getattr__` at `:342-348`. They are absent from
   `__all__` on purpose so `from panelary import *` cannot trigger them. No move
   may make any of the four eager.
6. **`_deps` keeps its path and its zero-third-party-import property.**
   `tests/test_import_hygiene.py:140-160` loads `panelary/_internal/_deps.py`
   straight off disk by that literal path (`:147`) and asserts no third-party
   import. This is the only filesystem-path assertion against a module inside
   `panelary/`. `AGENTS.md` ("Conventions -> Optional dependencies") and `llms.txt:17` name
   `panelary._internal._deps.require` as the documented convention.
7. **The verb block stays last in `__init__.py`.** `panelary/__init__.py:280-284`
   states the reason: `select`, `cluster` and `reduce` are subpackage names *and*
   verb names, and the verb must win the collision. The import sits at `:294-306`.

## Enforcement

| Invariant | Enforced by | Failure mode if you break it |
|---|---|---|
| Public dirs reachable + advertised | `tests/test_public_api.py:53-79` | test failure, named module |
| Public top-level `.py` reachable | `tests/test_public_api.py:114-127` | test failure, named module |
| `_LAZY_SUBMODULES` entries are real | `tests/test_public_api.py:82-88` | test failure |
| Documented module paths resolve | `mkdocs build --strict` (`Makefile:86`) | docs job fails |
| No eager optional dependency | `tests/test_import_hygiene.py:94-124` | test failure, named dep |
| `_deps.py` stdlib-only, at that path | `tests/test_import_hygiene.py:140-160` | test failure |
| Import time | `tests/test_import_hygiene.py:184-233` | test failure |
| Wheel ships only `panelary/`, pure-Python | `tests/test_wheel_guardrails.py:107-138` | test failure |

The one deliberate hole is `tests/test_public_api.py:27-31`: `_PRIVATE = {"evolve"}`
opts a directory out of invariant 1. It is an escape hatch for work in progress,
not a place to park a public subpackage; see
[Not done, and why](#not-done-and-why).

## The subtlety: a private filename is not private API

The underscore excludes a module from `tests/test_public_api.py`. It does **not**
excuse it from `mkdocs build --strict`. Four modules are named directly by
mkdocstrings despite their underscores, and renaming any of them breaks the docs
build:

| Module | Directive |
|---|---|
| `panelary/validation/_cv.py` | `docs/api-reference/validation.md:64` |
| `panelary/validation/_selection_stats.py` | `docs/api-reference/validation.md:77` |
| `panelary/validation/_forecast_tests.py` | `docs/api-reference/validation.md:94` |
| `panelary/validation/_bootstrap.py` | `docs/api-reference/validation.md:116` |

These four, plus the *public*-named module paths below, are the complete set of
submodule paths that may not move. Everything else in the tree is free.

| Frozen public submodule path | Named at |
|---|---|
| `panelary.base.forecaster` | `::: panelary.base.forecaster.Forecaster` |
| `panelary.core.model_selection` | five `:::` directives |
| `panelary.core.panel_frame` | `::: panelary.core.panel_frame` |
| `panelary.econ.features` | `::: panelary.econ.features` |
| `panelary.forecasting.automl` | `::: panelary.forecasting.automl.AutoForecaster` |
| `panelary.metrics.multi_objective` | `::: panelary.metrics.multi_objective` |
| `panelary.metrics.point` | `::: panelary.metrics.point` |

**Correction to the previous draft.** That draft asserted that
`panelary/reduce/_estimators.py` also holds mkdocstrings-published API. It does
not. The only `:::` directive touching `reduce` is `::: panelary.reduce`
(the package), and `_estimators` appears nowhere in `docs/`, `mkdocs.yml` or
`llms.txt`. Its symbols reach the public surface through
`panelary/reduce/__init__.py:62`, which is re-export, not path exposure.
`reduce/_estimators.py` is free to move. The `validation/_*` four are not.

## Placement table

Where a new file goes, by directory. "Private layer" counts `_*.py` excluding
`__init__.py`; "public-named" lists module paths that carry a public name today.

| Directory | Private | Public-named modules | Status |
|---|---|---|---|
| `_internal/` | 9 | — | conforms; the private helper layer |
| `catch22/` | 4 | — | conforms |
| `conformal/` | 2 | — | conforms |
| `detect/` | 5 | — | conforms |
| `econ/` | 7 | — | conforms |
| `evolve/` | 8 | — | conforms structurally; unwired (below) |
| `explain/` | 7 | — | conforms |
| `feature_extractors/` | 5 | — | conforms |
| `label/` | 1 | — | conforms |
| `plotting/` | 1 | — | conforms |
| `preprocessing/` | 8 | — | conforms |
| `select/` | 2 | — | conforms |
| `validation/` | 4 | — | conforms; four paths frozen by docs |
| `base/` | 0 | `forecaster` (frozen), `metric`, `model`, `transformer` | 3 of 4 unjustified |
| `core/` | 0 | `model_selection`, `panel_frame` (both frozen), `pipeline`, `protocol` | 2 of 4 unjustified |
| `metrics/` | 0 | `multi_objective`, `point` (both frozen) | justified |
| `seasonality/` | 0 | `calendar`, `fourier` | unjustified |
| `llm/` | 0 | `api`, `common`, `formatting` | unjustified |
| `transform/` | 0 | `frac_diff`, `neutralize`, `rank`, `scaling` | unjustified |
| `namespaces/` | 1 | `panel`, `xs` | unjustified |
| `factor/` | 1 | `ic`, `neutralize`, `portfolio` | unjustified |
| `cluster/` | 2 | `cross_sectional`, `kshape` | unjustified |
| `reduce/` | 7 | `factors`, `pca`, `xs` | unjustified |
| `forecasting/` | 5 | 11 modules, `automl` frozen | 10 unjustified |

"Unjustified" means the module carries a public name that nothing outside the
package cites — it is a candidate to be renamed `_*` behind its `__init__.py`,
under invariant 2 (this is renaming a *submodule*, not a public top-level
module, and is only free where no `:::` directive and no user code path names
it). It is not a defect to fix in a hurry; it is the backlog the rule generates.

Top level, `panelary/*.py`, is eight public modules plus `__init__.py`:
`backtesting`, `cross_validation`, `evaluation`, `imputation`, `models`,
`offsets`, `registry`, `testing`. Every one is either a `:::` target or listed
in `_INTERNAL_MODULES` (`tests/test_public_api.py:33-38`, which holds exactly
`registry`). **Do not add a ninth without deciding it is public API** — invariant
1 makes that decision for you the moment the file lands.

## Dependency limits

- `panelary/_internal/` is a leaf. Nothing in it may import from a public
  subpackage. There is one violation today: `_internal/_ranges.py:9` imports
  `panelary.offsets._strip_freq_alias`. Every other file in the directory has
  zero module-level `panelary` imports.
- `_internal/__init__.py` re-exports nothing, by design, so the package costs
  zero at `import panelary` time. Keep it that way — importing
  `panelary._internal` must never be a way to reach anything.
- Optional dependencies are imported inside the function that needs them, via
  `panelary._internal._deps.require` (`AGENTS.md`, "Conventions -> Optional dependencies"). No module-level
  `import sklearn` / `scipy` / `pandas` anywhere.
- `base/` must not import from `backtesting`, `conformal`, `cross_validation`
  or `forecasting`. It does today, from inside function bodies; this is the one
  outstanding architectural defect and it is scoped separately below.

## Measured invariants (2026-09-09)

Measured on this checkout at commit `20ce293` plus the working tree, after the
five in-flight module->package conversions landed. Re-measured at the end of the
session, not the start; the numbers below are the ones that stand.

| Quantity | Value | How |
|---|---|---|
| `import panelary` cold, best of 7 | **104 ms** | fresh subprocess, `time.perf_counter()` around the import |
| `import polars` cold, same conditions | 63 ms | as above — 61% of the total |
| Panelary overhead net of polars | 41 ms | difference; ratchet is 150 ms |
| Optional deps pulled by `import panelary` | **0** | `-X importtime`: no scipy / sklearn / pandas / plotly |
| `len(panelary.__all__)` | **54** | measured |
| `len(panelary.registry.to_records())` | **56** | measured; matches `AGENTS.md` |
| Top-level `.py` in `panelary/` | 9 (8 public + `__init__`) | `ls panelary/*.py` |
| Subpackage directories | 24 (23 public-named + `_internal`) | `find panelary -maxdepth 1 -type d` |
| Total modules | 161 | `find panelary -name '*.py'` |
| mkdocstrings identifiers | 42 | `grep -rh '^::: ' docs/` |
| Import-graph SCC, module granularity | 14 modules | AST scan, Tarjan |
| Import-graph SCC, package granularity | 6 packages | as above, collapsed |
| `ruff check .` | **All checks passed** | measured |
| `ruff format --check .` | **252 files already formatted** | measured |
| `mypy panelary` | 1028 errors in 65 files (161 checked) | measured |
| git pack | 235.72 MiB | `git count-objects -vH` |
| tracked `data/` | 141.2 MB / 63 files | `git ls-files data` + `cat-file -s` |

Two rows need a caveat rather than trust.

**Import time.** 104 ms exceeds the 85 ms design ceiling in invariant 4, but the
machine was at load average 11.6 with six agents running concurrently, and
`import polars` measured 63 ms against the 70 ms reference the budget is
calibrated to. Under the test's own scaling that is a ~271 ms budget against a
104 ms measurement — no risk of failing. The structural half of the guarantee is
intact and is the half that matters: `-X importtime` shows the whole 104 ms is
polars (63 ms) plus numpy plus ~38 ms of Panelary's own module-level work, with
**no optional dependency loaded at all**. The previous draft recorded 79 ms on a
quiet machine. **Re-measure on an idle machine before treating the gap as a
regression** — and if it survives, `panelary.core.model_selection` is where to
look: it accounts for 94 ms of the 122 ms cumulative `-X importtime` total,
almost all of it its own `polars` and `numpy` imports.

**mypy.** 1028 errors is a real count on a clean tree, down from the 1040 the
previous draft recorded. It is a baseline, not a pass: `ci.yml` gates on
`MYPY_BASELINE`, so the number must not grow.

The 14-module SCC is `{backtesting, base, base.forecaster, forecasting._ar,
forecasting._evaluate, forecasting._reduction, metrics, metrics.multi_objective,
metrics.point, preprocessing, preprocessing._detrend, seasonality,
seasonality.calendar, seasonality.fourier}` — six packages: `backtesting`,
`base`, `forecasting`, `metrics`, `preprocessing`, `seasonality`. The previous
draft measured 12 modules; the difference is entirely an artefact of
`preprocessing.py` and `seasonality` having become multi-module packages, which
makes more *files* visible to the scan without changing the package-level cycle
at all. That cycle is unchanged, and its cause is unchanged — see
[Not done, and why](#not-done-and-why). The only other cycle in the tree is
`{evolve, evolve._genome, evolve._search}`, which is self-contained.

### Corrections to the record

The previous draft listed eight defects, B1-B8. Seven were real and were fixed
in `20ce293`. **B4 was wrong and is struck.** It claimed
`::: panelary.label.fixed_horizon` and `::: panelary.label.meta_label`
(`docs/user-guide/labeling.md:248-249`) were dead directives naming
non-existent symbols. Both symbols exist — `panelary/label/_barriers.py:214`
and `:289` — and both are in `panelary.label.__all__` (verified by import).
The directives resolve and the docs build is correct as written. Do not delete
them.

## Not done, and why

### 1. `base/forecaster.py` reaches up into its own consumers

`panelary/base/forecaster.py` defers four imports into higher layers from inside
function bodies: `forecasting._ar` at `:203`, `backtesting` at `:280`,
`cross_validation` at `:281`, `conformal` at `:332`. A base class importing its
own consumers is inverted, and the deferral is not a fix — it only hides the
cycle from the interpreter.

This was verified causally, not asserted: stripping those four imports and
re-running the SCC scan collapses the 14-module / 6-package cycle to **nothing**
(the only remaining SCC is `evolve`'s internal three). The file is the sole
cause.

Not done because the fix is a design change, not a move: the orchestration in
those four call sites has to be inverted — pushed into `forecasting/`, or
injected as callbacks — and that changes behaviour-bearing code in a base class
that every forecaster subclasses. It is the highest-value and highest-risk item
in the tree and needs its own change, its own tests, and its own review. Doing
it inside a layout pass would bury it.

### 2. The 235.72 MiB git pack

`20ce293` untracked ~101 MB of unreferenced parquet; `data/` is now 63 tracked
files totalling 141.2 MB, down from 906 files / 242 MB. The pack is
**unchanged at 235.72 MiB** and will stay that way: `git rm --cached` stops the
tree growing, it does not rewrite history. Shrinking the pack requires a
`filter-repo` rewrite, which invalidates every existing clone, every open PR and
every commit SHA in the changelog.

Not done because that is a project decision with external blast radius, not a
refactor, and nobody has made it. Recording it here so the number is not
mistaken for a regression.

### 3. `panelary/evolve/` is a finished subpackage that is not wired up

`tests/test_public_api.py:27-31` lists `evolve` in `_PRIVATE` with the comment
"it has no `__init__.py` and nothing imports it yet". **That comment is stale.**
Measured state:

- 9 modules, 9,843 lines (`_fitness.py` 1,949, `_honest.py` 1,501).
- `panelary/evolve/__init__.py` exists, imports cleanly, and exports **38 names**
  in `__all__` with a full NumPy-style module docstring.
- 349 lines of tests across `tests/test_evolve.py` and
  `tests/test_evolve_leakage.py`, both entered through
  `pytest.importorskip("panelary.evolve")`.
- It ships in the wheel and roughly doubled it: `tests/test_wheel_guardrails.py:30`
  records ~0.29 MB in 0.4.0 → ~0.65 MB now, attributing the growth to
  `panelary/evolve/`.
- Zero references in `docs/`, `mkdocs.yml`, `llms.txt` or `README.md`; zero
  importers anywhere in `panelary/`; absent from `panelary/__init__.py` entirely.
  In a fresh interpreter, `hasattr(pn, "evolve")` is `False`, `"evolve" in
  pn.__all__` is `False`, and `"evolve" in dir(pn)` is `False`.

So the accurate characterisation is: **structurally complete, tested, shipped to
users, and reachable only by `import panelary.evolve`.** The `_PRIVATE` entry is
the only thing keeping the first row of the enforcement table green — remove it
and `test_subpackage_is_reachable[evolve]` and `test_subpackage_is_advertised[evolve]`
both fail immediately.

Not done because wiring it up is a public-API addition, not a layout fix. It
needs an owner's decision on three things this contract cannot make for them:
whether the 38 names are the intended surface, whether it is eager or lazy
(it costs wheel size, but its import cost has not been measured), and a
`docs/api-reference/` page — without which `mkdocs build --strict` gains nothing
and users gain a subpackage they cannot find. Until then the `_PRIVATE` entry
should stay, but **its comment must be corrected** — a stale justification is
how an opt-out becomes permanent.

## Gate

After any change to layout: `make check` (lint, typecheck, test), **plus**
`mkdocs build --strict` — the docs job is the only gate that catches a moved
module path — plus a re-measure of the import time and the `__all__` / registry
counts in the table above.
