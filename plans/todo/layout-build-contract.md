# Repository layout — build contract (drafted 2026-09-09)

Goal: one legible, **standardised** layout for `panelary/`, arrived at without
breaking the public API. Today the package is 141 modules — 18 well-formed
subpackages plus **24 loose top-level `.py` files**, several of them very large
(`feature_extractors.py` 3487 lines, `_verbs.py` 1264, `preprocessing.py` 1321,
`catch22.py` 1066).

This contract is tiered by **risk**, not by appeal. Tier 0-2 are safe and should
be done. Tier 3-4 are breaking or architectural and need an explicit decision —
`AGENTS.md` says renaming a public API "needs an explicit decision, not a
refactor", and `MIGRATING.md` commits the project to shipping **no compatibility
shims**. Both constraints are respected below.

## Measured baseline (must not regress)

| Quantity | Today | Source |
|---|---|---|
| `import panelary` cold | **79 ms** | measured |
| `len(panelary.__all__)` | **54** | measured |
| git pack size | **235.7 MiB** | `git count-objects -vH` |
| tracked `data/` | **242 MB / 906 files** | `git ls-files data` |
| mypy baseline | 1040 errors | `ci.yml` `MYPY_BASELINE` |
| test files deep-importing | **70 of 81** | AST scan |

The 79 ms is load-bearing: `__init__.py:325-338` documents a "light-core
guarantee" and keeps `forecasting` (828 ms), `backtesting`, `plotting` (143 ms)
and `llm` behind a PEP 562 `__getattr__`. **No move may make a heavy dependency
eager.**

## The standard (what "standardised" means here)

One rule, applied everywhere:

```
pkg/__init__.py     public re-exports + explicit __all__, and nothing else
pkg/_impl.py        implementation; free to move, rename, split
pkg/public.py       ONLY when the module path itself is documented API
```

Conformance today is uneven, and that unevenness is the actual disorder:

- **Clean** (8): `detect/`, `econ/`, `econ/features/`, `explain/`, `evolve/`,
  `label/`, `select/`, `validation/`
- **Mixed** (4): `cluster/`, `factor/`, `reduce/`, `forecasting/`
- **No private layer at all** (7): `base/`, `core/`, `llm/`, `metrics/`,
  `namespaces/`, `seasonality/`, `transform/`

Caveat that must be honoured: **a private filename is not automatically
private API.** `reduce/_estimators.py` and all four `validation/_*` modules hold
documented, mkdocstrings-published API (`docs/api-reference/validation.md:64-116`).
Renaming those breaks `mkdocs build --strict`.

## Hard invariants (every tier)

1. **Public surface is filesystem-derived.** `tests/test_public_api.py:41-49`
   enumerates every non-underscore directory with an `__init__.py`, and
   `:120` every non-underscore top-level `.py`, asserting each is reachable as
   `pn.<name>`. Any new public-named file or dir becomes required public API.
   Underscore-prefixed names are excluded — this is what makes Tier 1 free.
2. **No shim.** `MIGRATING.md:14` and `CHANGELOG.md:39` promise none. A move
   that needs a forwarding module is a Tier 3 decision, not a refactor.
3. **Light core.** `import panelary` stays <= 85 ms and pulls no optional dep.
   `tests/test_import_hygiene.py` is the ratchet.
4. **`_deps.py` stays put.** `panelary._deps.require` is named as the documented
   convention in `AGENTS.md:150-153` and `llms.txt:17`, and its path is asserted
   on the filesystem by `tests/test_import_hygiene.py:147`. It is public by
   documentation despite the underscore. Moving it is churn for no gain.
5. **`_verbs.py` stays top-level and last.** `__init__.py:294-301` requires the
   verb block to load after the `select`/`cluster`/`reduce` subpackages, because
   the verb must win the name collision.

## Findings that are bugs regardless of layout

These are defects, not preferences. Fix them whether or not any file moves.

| # | Defect | Evidence |
|---|---|---|
| B1 | `metrics/probabilistic.py` is **0 bytes** — imported by nothing, re-exported by nothing | `wc -c` = 0 |
| B2 | `metrics/multi_objective.py` (166 lines) is **unreachable**: `hasattr(pn.metrics, "multi_objective")` is `False`, yet `docs/api-reference/multi-objective.md` documents `Metrics` / `score_forecast` / `score_backtest` and `mkdocs.yml` navs it | measured |
| B3 | False docstring: `core/model_selection.py:777` says it "thinly wraps" `panelary.cross_validation.expanding_window_split`. It does not — it calls its own `_window_panel_split` reimplementation | read |
| B4 | Dead mkdocstrings directives `panelary.label.fixed_horizon` / `.meta_label` — neither symbol exists | `docs/user-guide/labeling.md:248-249` |
| B5 | `pytest --ignore` **silently accepts non-existent paths** (verified). CI's bare-core job carries a hand-maintained 15-file ignore list and the Makefile another; renaming any listed test silently re-enables it in the bare-core matrix, where it fails on a missing optional dep | measured |
| B6 | `.benchmarks/` is absent from `.gitignore` | read |
| B7 | Test fixtures read `data/...` by **relative path** (`tests/conftest.py:76-80,246-249,327`), so the suite only passes when run from the repo root | read |
| B8 | **`make typecheck` cannot run on Python 3.12+.** `[tool.mypy] python_version = "3.10"` (`pyproject.toml:211`) makes mypy reject numpy's own bundled stubs (`numpy/__init__.pyi:737`, a PEP 695 `type` statement) and abort with "errors prevented further checking". CI hides this by pinning 3.10 for that job (`ci.yml:90`), but the repo supports up to 3.13, so `make check` — the documented gate in `CLAUDE.md` and `AGENTS.md` — is broken for any contributor on a modern interpreter | measured on 3.13 |

## Real duplication (three sites)

Distinct from the many *apparent* duplicates, which are clean layering and must
be left alone: `plotting.py`/`_plotting.py`, `_ffd.py`/`transform/frac_diff.py`,
`cluster/kshape.py`/`_kshape.py`, and `factor/neutralize.py`/`transform/neutralize.py`
(different math — per-date OLS on named exposures vs. Gram-Schmidt of a whole
block; only the filename collides).

- **D1 — CAFE imputation, implemented twice.** `imputation.CafeImputer`
  (a `PanelTransformer`, 98 lines) and `preprocessing.cafe_impute`
  (a functime `@transformer` closure, `preprocessing.py:559-698`) wrap the same
  `cafe.impute` for the same purpose, each with its own `_require_cafe()`
  (`imputation.py:24`, `preprocessing.py:546`). `_verbs.impute` imports **both**.
  The `preprocessing` one is richer (uncertainty / anomaly / missingness
  by-products). Collapse onto one kernel; keep both public names.
- **D2 — cross-sectional neutralization, implemented twice.**
  `transform/neutralize.py:164` (`_residual_for_group`) and
  `namespaces/xs.py:242` (`_neutralize_residual`) run the *same* per-date
  `np.linalg.lstsq` with one-hot categoricals, intercept and validity mask, with
  no shared kernel. They can silently diverge. Extract one kernel; both call it.
- **D3 — walk-forward splitting, implemented twice.** `cross_validation.py`
  (row-based `(train, test)` frames) and `core/model_selection.py:692-835`
  (`_window_panel_split`, PanelFrame folds on the unique-time axis). Same
  algorithm, two semantics, and B3's docstring already claims they are one.

## Tier 0 — zero-risk (no import path changes at all)

Nothing moves; no public name changes. Do this first and independently.

| Task | Files |
|---|---|
| Delete the empty module | `metrics/probabilistic.py` |
| Decide B2: re-export from `metrics/__init__.py`, **or** delete the module and its docs page | `metrics/__init__.py`, `docs/api-reference/multi-objective.md`, `mkdocs.yml` |
| Fix the false docstring (B3) | `core/model_selection.py:777` |
| Drop the two dead directives (B4) | `docs/user-guide/labeling.md:248-249` |
| Add `.benchmarks/` (B6) | `.gitignore` |
| Add a guard test: every `--ignore` path in `ci.yml` and the `Makefile` must exist on disk (B5) | new `tests/test_ci_guardrails.py` |
| Make fixture paths root-relative via a `conftest` anchor (B7) | `tests/conftest.py` |
| Unbreak `make check` on 3.12+ (B8): raise `python_version`, or scope the numpy stubs out. **Do this first** — until it lands, one of the three documented gates is unverifiable locally, and the 1040/235 counts above cannot be re-measured by a contributor | `pyproject.toml:211` |

### Data hygiene (largest single win, and it is not a layout change)

`data/` is **242 MB across 906 tracked files** — `.gitignore:12-15` already bans
the pattern and admits in a comment that "the repo already carries a large
tracked `data/` tree from before this rule". It is *not* dead weight
wholesale: `tests/conftest.py` genuinely reads from it. But a precise audit
shows **~101 MB is referenced nowhere**:

| File | Size | Only mention |
|---|---|---|
| `m5_X_train.parquet` | 51 M | commented-out block, `conftest.py:349` |
| `m5_y_train.parquet` | 45 M | commented-out block, `conftest.py:350` |
| `m5_X_test.parquet` | 2.5 M | commented-out block, `conftest.py:352` |
| `m5_y_test.parquet` | 1.7 M | — |
| `m4_1h_train.parquet` | 820 K | none |
| `m4_1h_test.parquet` | 64 K | none |
| `tourism.parquet` | 12 K | none |

Only the `m5_*_sample` variants are actually loaded. `git rm --cached` these
seven stops the tree growing and shrinks the working copy; note it does **not**
shrink the 235 MiB pack — history rewriting is a separate, disruptive decision
that should not be bundled into a layout change.

## Tier 1 — group the private helper layer (no public surface change)

Seven loose `_*.py` files at the package root are the visible mess and are
excluded from the public-API test by the underscore rule (invariant 1).

```
panelary/_internal/
    __init__.py        (empty; no re-exports, to keep import cost at zero)
    _compat.py         Polars version shim      (1 importer)
    _numpy_stats.py    NumPy drop-ins for SciPy (4 importers)
    _progress.py       tqdm-or-passthrough      (3 importers)
    _utils.py          one decorator, 25 lines  (1 importer)
```

Deliberately **not** moved:
- `_deps.py` — invariant 4.
- `_verbs.py` — invariant 5.
- `_ffd.py` — it is a leaf *by design* so `namespaces/panel.py` can use it
  without importing `transform/`; its own docstring documents this. Moving it is
  safe but pointless. It has one genuine wart: a deferred upward import into
  `econ/features/_longmemory` for `estimate_ffd_order`, which is the only reason
  a latent `{_ffd, econ}` cycle exists. **Move `estimate_ffd_order` out of
  `_ffd.py`** and that cycle disappears. Do this instead of relocating the file.
- `_plotting.py` — belongs with `plotting.py` in Tier 3, not here.

Blast radius, all mechanical: ~8 test call sites (`test_numpy_stats_parity.py`,
`test_perf_parity_vectorization.py`, `test_forecasting_light.py`,
`test_import_hygiene.py`'s string list at `:105-160`), and nothing in `docs/`.

**While here, normalize one misleading spelling.** `catch22.py:48`,
`feature_extractors.py:15` and `llm/common.py:8` write `from panelary import
_numpy_stats` / `_deps` — importing a *submodule through the parent package*.
An AST scan reports this as an `__init__` cycle; it is **not** one
(`import panelary.catch22` standalone works, and neither module uses any name
defined in `__init__.py`). Spell it `from panelary._internal._numpy_stats import ...`
so the graph reads true.

## Tier 2 — collapse the three duplications

D1, D2, D3 above. Each is behaviour-preserving and each keeps every public name.
Each needs a test proving the two former implementations agree before one is
deleted. D3 additionally retires B3's false docstring.

## Tier 3 — feature-layer grouping (BREAKING — needs an explicit decision)

The two biggest files are both feature libraries and both are top-level public
modules named in `mkdocs.yml`, `README.md` and `llms.txt`:

```
panelary/features/
    __init__.py      re-export extract_features, FeatureExtractor, catch22_*
    _kernels.py      feature_extractors.py L1-278    vendored kernels
    _stats.py        feature_extractors.py L279-2160 ~55 general features
    _finance.py      feature_extractors.py L2162-2303 finance block
    _namespace.py    feature_extractors.py L2305-3282 the `.ts` expr class
    _catalogue.py    feature_extractors.py L3283-3487 registry specs
    catch22.py       catch22.py (already banner-sectioned; splits trivially)
panelary/plotting/
    __init__.py      the 7 plot_* functions
    _display.py      _plotting.py (TimeSeriesDisplay)
```

Argument for: `feature_extractors.py` alone carries **235 of the 1040** mypy
baseline errors (per the `ci.yml:97-99` comment — not re-measured here, see B8); the `.ts` namespace class is ~65 one-line delegations that are
mechanically derivable; and 3487 lines is the single worst legibility problem in
the repo.

Argument against, and it is strong: `pn.feature_extractors` and `pn.catch22` are
in `__all__`, in `README.md`'s "What's inside" table, in `llms.txt`, and are
mkdocstrings targets. Moving them **is** a public API break, and the project has
promised no shims. The honest options are (a) leave the module paths exactly
where they are and split only *behind* them — `feature_extractors.py` becomes a
thin re-export of `features/_*` while keeping its own name — or (b) accept a
breaking change in a minor version with a `MIGRATING.md` entry.

**Recommendation: (a).** It captures the entire legibility and mypy win at zero
API cost, and it is not a shim — the module keeps its real name and identity.

`plotting.py` + `_plotting.py` → `plotting/` is genuinely free by comparison:
the package name is unchanged, `_plotting` is private, and only
`tests/test_plotting.py` reaches inside.

## Tier 4 — the one real architectural smell

`base/forecaster.py` defers imports to **four higher layers** — `backtesting`,
`conformal`, `cross_validation`, `forecasting._ar` — from inside function
bodies. This single file is what creates the 12-module latent SCC
(`base` <-> `forecasting` <-> `metrics` <-> `preprocessing` <-> `seasonality`).
A base class should not reach up into its own consumers. Inverting this
(callbacks, or moving the orchestration into `forecasting/`) is the only change
here that improves the *architecture* rather than the filing.

Scope it separately. It is the highest-value and highest-risk item on this page.

## Ordering

Tier 0 → Tier 2 → Tier 1 → Tier 3(a) → Tier 4. Tier 2 before Tier 1 so the
duplication is gone before files move; Tier 3 last among the safe work because
it is the largest diff.

Gate after every tier: `make check`, plus `mkdocs build --strict` (the docs job
is what catches the 42 mkdocstrings identifiers), plus a re-measure of the 79 ms
/ 54-name baseline.
