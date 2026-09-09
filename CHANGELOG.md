# Changelog

All notable changes to Panelary are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is `0`, breaking changes land in a minor bump
(`0.4` → `0.5`).

Panelary is built on the foundations of [functime](https://github.com/functime-org/functime)
(Apache-2.0), which is actively maintained (v1.0.0, May 2026). Panelary reuses and credits
functime's feature-extraction and forecasting engine and adds a panel-first, leak-safe layer
on top — it interoperates with functime and Nixtla rather than replacing them.

**On names:** the project was published to PyPI as `polars-features` and imported as
`polars_features` through 0.1.7. Version 0.5.0 renames both to `panelary`. Entries for
0.4.0 and earlier keep the names that were true when they were written — see the note
above the 0.4.0 entry, and [MIGRATING.md](MIGRATING.md) for how to move.

## [Unreleased]

Nothing yet.

## [0.5.0] — 2026-09-09

The rename to **Panelary**, and the largest feature round so far: five design plans
implemented together. Four new capability pillars land (`detect`, `explain`,
`validation`, `econ`, plus factor extraction in `reduce`), a top-level verb API goes
in front of them, two long-standing numerical bugs are closed, and the light-core
gains from 0.4.0 are locked behind CI guardrails. No change to the mandatory
footprint: `[project.dependencies]` is still exactly `{numpy, polars}`.

### Changed — BREAKING: renamed to `panelary`

- **The import path and the PyPI distribution are both now `panelary`.**
  `import polars_features` and `pip install polars-features` no longer reach this
  project. The brand is Panelary; the canonical alias in the docs is
  `import panelary as pn`.
- **No compatibility shim ships, and none is planned.** A forwarding
  `polars_features` package would advertise a compatibility that has never been
  true. The last release on PyPI under the old name was `polars-features` 0.1.7 — a
  maintained fork of functime, with none of the panel API. 0.2.x, 0.3.x and 0.4.0
  were never published to PyPI under any name, so there is no released
  `polars_features` surface to forward to. See [MIGRATING.md](MIGRATING.md).
- **The public attribution hook is renamed `panelkit_shap_values` →
  `panelary_shap_values`.** This is the one rename that can break code you wrote:
  if you implemented that hook on a custom model so `explain.TreeAttributor` would
  pick it up, rename the method. It affects people tracking a git ref only — the
  hook never appeared in a PyPI release.
- **The test-guard environment overrides are renamed `PANELKIT_*` → `PANELARY_*`**:
  `PANELARY_IMPORT_BUDGET_MS`, `PANELARY_WHEEL_BUDGET_MB`, `PANELARY_DETECT_STRICT`.
- The repository, documentation site and package metadata all move to the
  `panelary` name.

### Added — a top-level verb API

- The capability pillars are now reachable as top-level verbs on the package:
  `pn.impute`, `pn.select`, `pn.features`, `pn.bubbles`, `pn.cluster`,
  `pn.regression`, `pn.causal` and `pn.reduce`. The underlying subpackages remain
  importable and unchanged; the verbs are a shorter front door, not a replacement.
  See the API reference for signatures.

### Added — `detect`: causal bubble, regime and changepoint detection

A fifth capability pillar. Every statistic is a function of data up to `t` and
nothing after it: recomputing a feature with more data appended returns the
**bitwise identical** value, asserted across fifteen expanding cut points and
every public entry point.

- **`bsadf_sequence` / `bsadf_panel`** — the backward sup-ADF (Phillips, Shi &
  Yu 2015). Built on a prefix-sum sufficient-statistic engine (`cumulative_moments`,
  `window_adf`): every one of the `~T²/2` nested windows costs `O(1)` to assemble,
  and `SSR = s − β̂'b` removes the residual pass entirely. `T = 1000` exhaustive
  (452,676 windows) in **~0.02 s of pure NumPy**, against 0.78 s for the fastest
  published C++ implementation, which is still `O(T³)` because it forms residuals
  in its inner loop. Agreement with per-window OLS: 1e-14.
- **`mc_table`** — one-pass nested critical values. On a driftless null the BSADF
  sequence is itself point-in-time, so paths nest and one simulation yields `cv[t]`
  for every `t`; draws are time-major so a wider table slices bitwise to a narrower
  one. Reproduces the published finite-sample 95% GSADF value at `T = 400`,
  `r0 = 0.10` as **2.2197**. A shipped fixed-point table serves the default call in
  0.4 ms, fingerprint-guarded against kernel drift.
  `kurozumi_boundary` and `training_max_cv` simulate nothing at all.
- **`focus`** — functional-pruning CUSUM: provably equivalent to running Page's
  CUSUM at every magnitude and every window length at once, no tuning parameter,
  `O(log n)` amortised. Exact against brute-force GLR (0.0), 100k points in 0.075 s.
- **`page_cusum`** in closed form (Lindley's recursion is a reflection, so it is two
  cumulative aggregations — `page_cusum_expr` gives the Polars expression),
  `shiryaev_roberts` in log space, `hb_cusum` with an analytic Chu–Stinchcombe–White
  boundary, `end_of_sample_S`, and one-sided `spot_variance` / `volatility_rescale`.
- **`breadth`, `residualise`, `panel_features`** — cross-sectional aggregation by
  participation rather than by mean, and backward-looking factor residualisation
  with betas frozen at `t`.

Deliberately refused: full-sample `GSADF` as a per-row feature, and episode peak /
end / duration. `min_window` is a required absolute integer — the conventional
`⌊T(0.01 + 1.8/√T)⌋` rule revises 41.2% of already-published cells when the sample
grows. No bootstrap critical values ship, because the standard wild bootstrap fits
its null on the whole sample.

### Added — `reduce`: leak-safe latent factor extraction

- `PCAFactors`, `HFAFactors`, `ICAFactors`, `RobustPCAFactors` plus the
  `pca_factors` / `hfa_factors` / `ica_factors` / `robust_pca_factors` functional
  cores. All emit `factor_1..r` columns from train-fit, sign-fixed loadings.
- **HFA** (higher-order multi-cumulant factor analysis), `order=3|4` — eigenanalysis
  of a cumulant matrix rather than the covariance matrix, recovering weak and
  Gaussian-masked non-Gaussian factors where PCA fails. Pure NumPy, with a blocked
  accumulation path (`block_rows=`) bounding peak memory at O(block·n) and a hard row
  guard. Clean-room from the published equations. `order=4` applies an Isserlis-derived
  Gaussian correction (`3·tr(S²)S² + 6·S⁴`) — without it the order-4 matrix is a
  polynomial in the covariance and merely re-finds the masking factor.
- Shared factor-count selectors `n_factors` / `bai_ng` (Bai–Ng IC_p1, IC_p2) /
  `eigenvalue_ratio` (Ahn–Horenstein). `n_factors=None` resolves on training rows only.

### Added — `explain`: leak-safe, panel-aware feature attribution

- `TimeAwareBackground` makes the SHAP reference set a fold-bound, past-only
  (`t' < t`, optional embargo), deterministically sampled, auditable object —
  closing the background-set leak that every other SHAP library leaves open.
- `TreeAttributor` (`PanelTransformer`, `panel_safe`/`leakage_safe`) dispatches to the
  models' own exact native TreeSHAP (XGBoost/LightGBM/CatBoost) and emits
  `shap_<feature>` columns keyed by `(entity, time)`, or a tidy long frame.
  `.check_efficiency()` verifies `E[f] + Σφ = f(x)` on the raw margin scale.
- Panel-native aggregation: `group_shap` (additive, exact) vs `joint_group_shap`
  (groups as coalition players), and `window_shap` on each entity's own calendar.
- `attribution_stability` / `background_sensitivity` / `attribution_drift` separate
  reference-induced oscillation from genuine regime drift.
- `interaction_values` / `interaction_matrix`: any-order Shapley interactions via
  `shapiq` interop (`max_order=k`).
- Custom models can supply their own exact attributions through the
  `panelary_shap_values` hook (renamed from `panelkit_shap_values`, see above).
- New optional extra `explain` (`shap`, `shapiq`). The `TreeAttributor` fast path
  needs neither.

### Added — `validation`: the honest validation & selection layer

- CPCV backtest-path reconstruction, purged walk-forward and conformal-calibration
  splits, Probabilistic/Deflated Sharpe, PBO, Romano-Wolf stepdown (FWER),
  Benjamini-Hochberg/Yekutieli (FDR), Diebold-Mariano (HAC + Harvey-Leybourne-Newbold),
  Hansen's SPA, the Model Confidence Set, CRPS/pinball/interval score/PIT, and
  moving-block/circular/stationary/wild/sieve bootstraps that never resample across a
  fold boundary. Pure NumPy + Polars — SPA and MCS are implemented natively rather than
  wrapped from `arch`, so the fold-boundary guarantee extends to them.
- `conformal`: time-series conformal prediction — ACI (Gibbs-Candès), Conformal-PID,
  NexCP and CQR, plus `conformal_calibration_split` so the calibration block is purged
  and embargoed. Coverage now holds under drift where naive split conformal collapses
  (0.89 vs 0.52 on a volatility ramp).

### Added — `econ`: panel & time-series econometrics

- **HDFE** — N-way fixed-effect absorption by alternating projections (Gaure/reghdfe)
  with classical, HC1, one-way and two-way (Cameron-Gelbach-Miller) clustered and
  Driscoll-Kraay standard errors; matches dense dummy-variable OLS to machine precision.
  `HDFETransformer` emits leak-controlled partialled-out features.
- **Heterogeneous panels** — Mean Group, Pesaran CCE-MG/CCEP and Pooled Mean Group,
  plus LLC / IPS / CIPS panel unit-root tests and the Pesaran CD test. Critical values
  are simulated from the estimator pipeline itself, not tabulated.
- **Fama-MacBeth** — per-date cross-sectional regression with Newey-West standard
  errors; winsorising and standardising are per-date, never global.
- **Diebold-Yilmaz connectedness** — generalised-FEVD spillover networks over a rolling
  VAR; total/directional/net connectedness as cross-entity features. Clean-room.
- **IVX** — predictive-regression inference that stays correctly sized under
  near-unit-root, endogenous predictors (5.6–7.1% empirical size at nominal 5%, vs
  8–27% for OLS); the IVX-Wald statistic doubles as a screening score (`IVXSelector`).
- **Double ML** — cross-fitted partially-linear DML over Panelary's purged and embargoed
  splitters, plus post-double-selection LASSO with the rigorous (plug-in) penalty.

### Added — `econ.features`: causal econometric feature generators

- Unit-root battery (ADF/KPSS/PP/DF-GLS/Ng-Perron/Zivot-Andrews with embedded
  critical-value tables), GPH & local-Whittle long-memory estimators, HAR-RV /
  bipower / jumps, Nelson-Siegel(-Svensson) curve factors, Amihud/Roll/Amivest
  liquidity, EVT Hill/POT-GPD VaR-ES, and a causal (trailing-window) seasonal-trend
  decomposition. A two-sided STL leaks and is deliberately not offered.
- `StationarityDifferencer`, `AutoFracDiff`, `HARModel`, `NelsonSiegel` and
  `CausalSeasonalDecomposer` — `PanelTransformer`s that learn on train and freeze.
- `panelary._ffd.estimate_ffd_order` — data-driven fractional-differencing
  order, wiring the long-memory estimators into the existing frac-diff filter.

### Added — the "interactions" theme

- `docs/concepts/interactions.md`: the shared functional-decomposition narrative tying
  `reduce`/HFA (higher-order structure in the *data*, `order=k`) to `explain`/Shapley
  interactions (higher-order structure in the *model*, `max_order=k`).

### Added — packaging guardrails

- Import-hygiene, dependency-drift and wheel-shape/size tests
  (`tests/test_import_hygiene.py`, `test_dependency_drift.py`,
  `test_wheel_guardrails.py`), plus a CI extras matrix that installs `[]`,
  `[recommended]` and `[all]` and runs the suite in each. Measured: 85 ms import,
  0.46 MB wheel. The budgets are overridable with `PANELARY_IMPORT_BUDGET_MS` and
  `PANELARY_WHEEL_BUDGET_MB`.
- `benchmarks/bench_hotspots.py` (ranked per-operation profile) and
  `benchmarks/show_capabilities.py` (optional-feature probe).
- `_numpy_stats.chebyshev_neighbour_counts`: exact, SciPy-free L-inf neighbour counts,
  so `sample_entropy` / `approximate_entropy` now work on the bare numpy+polars core
  (SciPy stays the faster backend when installed). Bare-core `test_tsfresh.py` went
  from 8 failed / 221 passed to 229 passed.

### Changed

- catch22 is 1.2×–2.0× faster (bit-exact): the remaining Python scans in
  `FC_LocalSimple_*`, `SB_MotifThree`, `SB_TransitionMatrix`, `_longest_run`,
  `CO_FirstMin_ac`, `CO_f1ecac`, `PD_PeriodicityWang` and `DN_OutlierInclude_*`
  are vectorised.
- k-Shape clustering is 4×–9× faster to fit and 8×–14× faster to score (bit-exact):
  forward FFTs are computed once per series/centroid instead of once per pair.
- `reduce.CrossSectionalPCA.transform` is 1.3×–2.0× faster (one array conversion
  instead of one per date).
- The test suite runs with `-n auto --dist loadfile` in CI and `make test`:
  **15+ minutes serial → 67 seconds** on 8 cores (1585 passed, 8 pre-existing
  failures in `tests/test_cusum_pure.py`). Those 8 were resolved later in the
  same cycle: their Rust-parity baselines had been captured from the extension
  removed in 0.4.0 and cannot be regenerated, so they now skip with that reason
  rather than fail. The file stands at 4 passed, 6 skipped.
- The mypy job no longer pretends. It reported ~1040 errors while configured as
  a hard gate, so it had never passed; it now ratchets against a baseline and
  fails only when the count rises.
- Every CI job has a `timeout-minutes`.
- `make check` runs lint, type-check and tests as one command.
- CI/release workflows dropped the dead Rust/maturin steps; releases now build a
  single universal `py3-none-any` wheel.
- Documentation is published by CI. GitHub Pages had never been enabled and the
  declared `site_url` returned 404; `docs/CNAME` pointed at `docs.functime.ai`,
  a domain this project does not control.

### Fixed

- **`pn.detect` raised `AttributeError`.** `detect` — 0.4.0's headline feature,
  37 public symbols — was never added to the package `__init__.py`. Nine
  modules were unreachable in total (`detect`, `forecasting`, `llm`, `metrics`,
  `backtesting`, `conformal`, `cross_validation`, `evaluation`, `plotting`);
  seven more resolved only by import side-effect and were absent from
  `__all__`. All are now reachable, and `tests/test_public_api.py` fails the
  build if a public subpackage is ever left unwired again.
- **`panelary.backtesting` could not be imported by any path.** A cycle
  (`backtesting` → `forecasting._reduction` → `forecasting/__init__` →
  `elite` → `backtesting`) meant every import of it raised `ImportError`.
  `forecasting/elite.py` now imports `backtest` inside the function that uses
  it, matching the pattern `base/forecaster.py` already used.
- **Broken `pip install` hints.** `forecasting/__init__` advertised the extras
  `lgb`, `cat` and `xgb`; none exist (the real names are `lightgbm`,
  `catboost`, `xgboost`), so following the hint produced "no matches found".
  Those names were also bound to an `ImportError` *instance*, making
  `fc.lightgbm` a value rather than something that raised, so failures
  surfaced later as a confusing "not callable" `TypeError`.
- **`import panelary.llm` required an API key.** The OpenAI client was
  constructed at module scope and raised `ValueError` when `OPENAI_API_KEY`
  was unset. It is now built on first use, and the module's imports route
  through `_deps.require` like every other optional dependency (its old hint,
  `polars_features[llm]`, was not a pasteable distribution name — under the new
  name the import name and the distribution name coincide, so the whole class of
  mistake is gone).
- `panelary/forecasting/elite.py` no longer imports scikit-learn at
  module scope, which was defeating the light-core guarantee that the extras
  matrix exists to defend.
- **frac-diff divergence** (one of two known numerical bugs): `ffd_weights` now rejects
  `d < 0`, whose binomial weights are not summable and made the truncated fixed-width
  filter diverge — at `d=-0.5` it ran to the 100 000-term safety cap and nulled every
  row of a normal-length series. Two further defects found in the same function:
  `max_width` was checked *after* appending (so `max_width=1` returned two weights),
  and a legal `d` with an unreachable `threshold` silently returned a 100 000-tap
  kernel instead of raising. Regression tests carry hand-computed golden weights.
- **Deflated Sharpe `N`/`V`** (the second known bug): the formula itself was verified
  correct against Bailey & López de Prado (2014) and pinned with independently computed
  golden values. The real defect was the `(N, V)` *pairing* in `_reconstruct_paths`,
  which combined the user's trial count `N` with the variance of one strategy's CPCV
  path Sharpes as a proxy for `V`. `cross_validate` / `validate.cpcv` now accept
  `trial_sharpes=`, deriving both from the same sample.
- **catch22 is now SciPy-independent.** `catch22.PD_PeriodicityWang_th0_01` returned
  *different values* depending on whether SciPy was installed: its
  `LSQUnivariateSpline` detrend degraded silently to a zero spline on the bare core
  (140.0 → 0.0 on a 512-point random walk). The least-squares cubic spline is now
  fitted in pure NumPy (`catch22._lsq_spline_fit`, a Cox-de Boor B-spline design
  matrix + `lstsq`), matching SciPy to ~1e-14 over n = 8..2048 and reproducing every
  SciPy-captured golden value exactly. All 24 features now compute on numpy + polars
  alone.
- **The README quickstart documented a library that was never shipped.** None of
  `pk.col(...)`, `pk.transform.winsorize`, `pk.models.lgbm_classifier` or
  `pk.model_selection.validate.cpcv` exists, and the sample used a
  `.panel.pct_change` method that does not exist and passed
  `triple_barrier(max_holding="10d")` where the real parameter is an integer.
  The quickstart has been replaced with one that was actually executed against
  the package.
- **The canonical alias is now `pn`.** `import panelary as pn` is used
  consistently across the README, every documentation page, `llms.txt`,
  `AGENTS.md` and the site. The old `pk` alias was short for the pre-launch
  codename and is retired.
- **README badges advertised Python 3.8+**; `pyproject.toml` requires `>=3.10`.
  The badges now match the metadata.
- **`CONTRIBUTING.md` described a project that no longer exists** — "Rust +
  Polars under the hood", with a `src/` directory of Rust kernels to contribute
  to. There is no `src/`, and the package has been pure Python since 0.4.0.

### Removed

- The `signatures` extra (documented RESERVED/UNUSED; its only reference
  anywhere was the `_deps` mapping), the `rust` keyword and the `maturin` dev
  dependency (no Rust extension since 0.4.0), the dead `[tool.hypothesis]`
  block (Hypothesis does not read `pyproject.toml`, so `deadline = 0` had no
  effect), a tracked 4.5 MB pre-0.4.0 Windows wheel, and `functime_rename.py`
  (a one-shot fork-migration script that rewrote every `.py` in the repo).
- The dangling `narwhals` → `interop` row in `_deps._MODULE_TO_EXTRA` (the `interop`
  extra went away in 0.4.0) and the corresponding docs entry.
- **Third-party analytics on the documentation site.** The site was reporting to a
  Google Analytics property (`G-CYGTL9FJ4R`) that belongs to the upstream functime
  docs and was inherited wholesale in the original import commit — it was never
  ours, and readers were being measured by someone else. Analytics is now disabled.

### Deferred

Every milestone that assumed a compiled extension was dropped, since 0.4.0 ships a
pure-Python wheel: the native Arrow/Polars TreeSHAP kernel and the sparse
Möbius/Fourier (SPEX) engine (`explain` M6/M7), the Rust HFA cumulant contraction
(`reduce` M6), vendoring the `tsecon` crates and its golden-fixture CI harness
(econometrics M7), and `augurs`-backed MSTL. DCC-GARCH, vine copulas and Bayesian
VAR/FAVAR remain out of scope by design.

---

> **Everything below this line predates the rename.** These entries were written
> while the package was published as `polars-features` and imported as
> `polars_features`, and those names are left exactly as they were — they record
> what existed at the time and are not instructions to follow today. (The *brand*
> in the prose has been normalised to Panelary; the project's working name changed
> more than once before launch, and reviving the dead codenames would help nobody.)
> For anything you actually run, use `panelary` and read [MIGRATING.md](MIGRATING.md).

---

## [0.4.0] — 2026-09-04

*Never published to PyPI.* A "light package" release: a numpy + polars core,
~10× faster import, and a pure-Python distribution (the Rust extension is gone).
Two **breaking** structural changes are called out below.

### Changed — BREAKING: slim hard dependencies

- **`pip install polars-features` now installs only `numpy` + `polars`.**
  `scikit-learn`, `scipy`, `flaml`, `holidays`, and `tqdm` moved out of the hard
  dependencies into optional extras (`ml`, `scipy`, `forecasting`, `seasonality`,
  `progress`). Feature code imports them lazily via `polars_features._deps.require`,
  which raises a single actionable `pip install 'polars-features[<extra>]'` message
  when one is missing. **To restore the previous batteries-included behavior:**
  `pip install 'polars-features[recommended]'` (ml + scipy + seasonality + cafe) or
  `[all]`.

### Changed — BREAKING: pure-Python distribution (Rust extension removed)

- **The compiled Rust extension (`src/`, Cargo, maturin) has been removed.** Panelary
  now ships as a single universal `py3-none-any` wheel — no compiler, no per-platform
  wheels, trivial installs everywhere. An audit found the crate was net-negative: its
  fractional-differencing kernel was already dead, its least-squares kernel was 35–500×
  *slower* than numpy, and Lempel–Ziv showed no gain. The only real win was CUSUM, now
  reimplemented in pure Python with an optional `numba` fast-path via the new `fast`
  extra (`pip install 'polars-features[fast]'`). Build backend switched maturin → hatchling.

### Changed — packaging & lighter import

- **Lighter `import polars_features`.** `scipy`, `scikit-learn`, and `cloudpickle`
  are no longer imported at `import polars_features` time; heavy back-ends are now
  loaded lazily at first use via `polars_features._deps.require(...)`, which raises a
  single actionable `pip install 'polars-features[<extra>]'` hint when a dependency
  is missing.
- **Dropped `cloudpickle`** from the hard dependencies — the only use (persisting a
  fitted sklearn regressor in the deseasonalize/reseasonalize path) is now handled by
  the stdlib `pickle` module. No functional change; the serialized objects are
  top-level library classes that `pickle` round-trips identically.

### Added — optional extras

- New named extras so users and tools can already target slim installs and so the
  `require(...)` install hints resolve to real extras:
  `ml` (`scikit-learn`), `scipy`, `progress` (`tqdm`), `seasonality` (`holidays`),
  `forecasting` (`flaml` + `tqdm`), `automl` (`flaml[automl]` + `lightgbm`),
  `lightgbm`, `catboost`, `xgboost`, and `ann` (`pylance`, for the `forecasting.lance`
  ANN reduction).
- The booster / ANN extras (`lightgbm`, `catboost`, `xgboost`, `ann`) previously did
  not exist even though `forecasting` error messages pointed at them — this makes
  `pip install 'polars-features[lightgbm]'` (etc.) actually work.
- `recommended` extra — a batteries-included bundle (`ml` + `scipy` + `seasonality`
  + `cafe`) that reproduces the old effective feature set for users who want
  `pip install` to "just work".
- Refreshed the `all` extra to union every real optional feature set.

### Removed

- Dead `interop` extra (`narwhals`) — `narwhals` is not imported anywhere in the
  package (it is only duck-typed in comments).

### Notes

- The `signatures` extra (`iisignature`) is retained but **reserved/unused** — no
  code imports `iisignature` yet and its license must be verified before it is
  advertised. (Removed outright in 0.5.0.)

## [0.3.0] — 2026-09-04

*Never published to PyPI.* Unsupervised toolkit (clustering, dimensionality
reduction, unsupervised feature selection) ported leak-safely from the Sov.ai SDK,
plus a cross-sectional/factor evaluation layer and four confirmed correctness fixes.
Every learned method fits on the training fold only; genuinely offline methods are
gated behind `leakage_safe=False`.

### Added — clustering (`polars_features.cluster`)

- `KShapeClusterer` — leak-safe k-Shape time-series clustering (FFT normalized
  cross-correlation + Shape-Based Distance), centroids learned on train and frozen;
  emits per-`(entity, time)` distance columns + a hard label; static / expanding /
  fixed-window modes. Pure-NumPy engine, no torch.
- `CrossSectionalClusterer` — clusters entities by their same-date feature vectors
  (KMeans / agglomerative), scaler fit on train only.
- Public utilities `sbd`, `ncc`, `k_from_n_entities`.

### Added — dimensionality reduction (`polars_features.reduce`)

- `PanelPCA` and `PanelSVD` / `PanelFactorAnalysis` / `PanelRandomProjection` /
  `PanelKernelPCA` / `PanelNMF` — leak-safe reducers (scaler + rotation + auto
  `n_components` fit on train only, deterministic component sign-fixing).
- `StatisticalFactors` — K statistical factors from a returns panel (train-fit
  loadings); `scope="global"` is gated `leakage_safe=False`.
- `CrossSectionalPCA` — per-date cross-sectional reduction.
- Optional `PanelUMAP` via the new `dimreduce` extra (`umap-learn`).

### Added — unsupervised feature selection (`polars_features.select`)

- `pfa` / `PFASelector` — Principal Feature Analysis (PCA loadings clustered, one
  representative feature per cluster; unsupervised, complements `mrmr`).
- `variance`, `correlation`, `select_top`, `projection_importance`, and
  `VarianceSelector` / `CorrelationSelector` transformers.

### Added — cross-sectional / factor evaluation (`polars_features.factor`)

- `forward_return` — the single audited, leak-safe forward-return aligner
  (backward per-entity shift with a gap guard).
- `ic` / `ic_summary` — per-date information coefficient (rank / Pearson), ICIR, t-stat.
- `portfolio_sort` — per-date quantile buckets, long-short spread, monotonicity.
- `orthogonalize` — per-date feature-set de-correlation (Gram-Schmidt / QR).
- `.xs.standardize` and `.xs.rank(normalize="unit"|"centered"|"uniform")`.

### Fixed

- **Deflated Sharpe Ratio**: `cross_validate` / `validate.cpcv` no longer deflate
  against `n_paths` (CV geometry) with a fallback variance — they now take an explicit
  `n_trials` (warning + `n_paths` floor when omitted) and use the empirical
  cross-trial variance `V = var(path_sharpes, ddof=1)`. New NaN-safe, annualization-aware
  `_sharpe`; PBO can rank by Sharpe (`statistic="sharpe"`). (Superseded by the
  `trial_sharpes=` fix in 0.5.0.)
- **`select.mda`**: permutation is now within-entity by default
  (`permute_within="entity"`) instead of a global cross-entity/time shuffle that
  produced off-manifold rows and inflated importances.
- **Fractional differencing**: consolidated to a single `ffd_weights` / `frac_diff_expr`
  source of truth (`polars_features._ffd`); all five Python surfaces now agree bit-for-bit
  (parity-tested), removing the Rust `threshold=0.0` divergence from the Python path.
- **`preprocessing.impute`**: `bfill` / `interpolate` now emit a `LeakageWarning`
  (opt out with `allow_leaky=True`); steers users to the point-in-time CAFE path.

## [0.2.0] — 2026-09-02

*Never published to PyPI.* First cut of the Panelary panel core, plus the leak-safe
feature-engineering, labeling, and validation layer landing on top of the
functime-derived engine.

### Added — panel core (shipped in-repo, experimental)

- `PanelFrame` — typed, lazy view over a long-format `(entity, time, *features)` panel.
- Leak-safe `Pipeline` with sklearn ergonomics and `panel_safe` / `leakage_safe` contracts
  (`PanelTransformer` / `PanelEstimator`).
- `.panel` (per-entity, causal) and `.xs` (cross-sectional) Polars expression **and**
  frame-level namespaces, registered as an import side effect.
- Finance-grade, leak-safe cross-validation: `PurgedKFold` and `CombinatorialPurgedCV`
  (CPCV), plus panel-aware `expanding_window_split` / `sliding_window_split`
  (López de Prado, *Advances in Financial Machine Learning*).
- Backtest-overfitting metrics: `deflated_sharpe_ratio` and
  `probability_of_backtest_overfitting` (PBO via CSCV).

### Added — landing this release

- CAFE imputation: `cafe_impute` (`polars_features.preprocessing`) and `CafeImputer`
  (`polars_features.imputation`), powered by the optional `cafe-impute` dependency
  (MIT, Sov.ai) via the new `cafe` extra.
- Bulk `extract_features` (`polars_features.feature_extractors`).
- Cross-sectional `.xs` operations.
- Triple-barrier labeling: `triple_barrier` (`polars_features.label`).
- catch22 feature set (`polars_features.catch22`) — clean-room from Lubba et al. 2019.
- Leakage verifier `assert_no_lookahead` (`polars_features.testing`).
- CPCV runner: `cross_validate` / `validate` / `CVReport`
  (`polars_features.core.model_selection`).

### Changed

- Bumped version to `0.2.0` (past the live PyPI `polars-features` 0.1.7); aligned
  `Cargo.toml`.
- Fixed `__version__` (was functime's `0.9.5`) and the package docstring to describe
  the project.
- Top-level exports now use narrow `try/except ImportError` with a `warnings.warn`
  instead of blanket `contextlib.suppress(Exception)` (which silently hid real import bugs).
- Repositioned the project story: functime is **actively maintained**; Panelary interoperates
  with functime/Nixtla rather than replacing them. Corrected `README.md`, `CHANGELOG.md`,
  `NOTICE`, and docs.
- Packaging hygiene: removed the committed prebuilt wheel from git and ignore `*.whl`;
  fixed the stale `Makefile` (uninstall `polars_features`, `BASE ?= main`, `[dev]` extra);
  fixed user-facing `pip install functime[...]` strings to `polars_features[...]`.
- Documentation: mkdocstrings API pages now target `polars_features.*`; `mkdocs.yml`
  site/repo URLs point at the project's own locations.

## [0.1.x] — the "maintained functime" wedge

The 0.1 line is the only one that ever reached PyPI, and it did so under the name
**`polars-features`**: 0.1.0, 0.1.5, 0.1.6 and 0.1.7. It was a drop-in, *maintained*
functime with modern packaging — none of the panel API described above existed in it.
**0.1.7 is the last release under the old name**; there is no 0.1.8, and no
`polars_features` release will ever contain the current API.

### Added

- Verified compatibility with current Polars (`polars>=1.29`).
- Published wheels for Linux/macOS/Windows on supported Python versions.
- Green CI, test suite passing, reproducible builds.
- A rebranded documentation site retaining the functime API surface.
- A migration note for functime users (same APIs, new home).

### Notes

- Out of scope for the 0.1 line, and delivered later: `PanelFrame` / leak-safe
  `Pipeline` / CPCV / labeling and the full panel module taxonomy (0.2.0), the
  clean-room catch22 feature set (0.2.0), and the public package rename (0.5.0).

## [Phase 0] — modernization & foundation

Pre-0.1 groundwork, recorded for completeness.

### Added

- `NOTICE` file crediting functime as the upstream this work is derived from, with
  Apache-2.0 third-party notices and a clean-room statement for catch22 (reimplemented
  from Lubba et al. 2019, not vendored from GPL `pycatch22`).
- `llms.txt` discovery file (per the llms.txt convention) summarizing the project, its
  module taxonomy, and core concepts for coding agents.
- `CHANGELOG.md` (this file).
- Own-branded documentation: `docs/quickstart.md` and `docs/leakage.md`
  (correctness-by-construction narrative).

### Changed

- Repositioned the project as a named product — "leak-safe, Rust-fast feature engineering
  and ML for panel data, built on Polars" — across `README.md`, `mkdocs.yml`, and docs.
  (The "Rust-fast" claim stopped being true in 0.4.0, which removed the extension.)
- Rewrote `README.md` with the product one-liner/tagline, prominent functime credit,
  install guidance (note: import name remains `polars_features`), a 60-second quickstart
  of the target panel API, an honest shipped-vs-roadmap table, and ecosystem positioning.
- Updated `mkdocs.yml` site name/description and navigation.
- Updated `CONTRIBUTING.md` to describe the Tier B/C (Python-first, low Rust barrier)
  contribution path, the `panel_safe`/`leakage_safe` contracts, and good-first-issue posture.

### Notes

- Infrastructure in progress at the time: modern packaging via `pyproject.toml`
  (maturin + abi3 wheels), CI and multi-platform wheel builds, and core panel protocol
  scaffolding (`PanelFrame`, leak-safe `Pipeline` contracts) in design phase.
- No public Python or Rust APIs were renamed in Phase 0; the package rename was
  deferred. It finally landed in 0.5.0, as `panelary`.
- The functime-derived engine (feature extractors, forecasting, preprocessing, seasonality,
  cross-validation, metrics, LLM analysis) was unchanged and remained importable from
  `polars_features`.
