# Changelog

All notable changes to PanelKit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

PanelKit is built on the foundations of [functime](https://github.com/functime-org/functime)
(Apache-2.0), which is actively maintained (v1.0.0, May 2026). PanelKit reuses and credits
functime's feature-extraction and forecasting engine and adds a panel-first, leak-safe layer
on top — it interoperates with functime and Nixtla rather than replacing them. The import/PyPI
package is currently `polars_features`; the public rename to `panelkit` is planned but not yet
effective.

## [Unreleased]

Five design plans implemented in one round. Four new capability pillars land
(`explain`, `validation`, `econ`, factor extraction in `reduce`), two long-standing
numerical bugs are closed, and the light-core gains from 0.4.0 are locked behind
CI guardrails. No change to the mandatory footprint: `[project.dependencies]` is
still exactly `{numpy, polars}`.

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
- **Double ML** — cross-fitted partially-linear DML over PanelKit's purged and embargoed
  splitters, plus post-double-selection LASSO with the rigorous (plug-in) penalty.

### Added — `econ.features`: causal econometric feature generators

- Unit-root battery (ADF/KPSS/PP/DF-GLS/Ng-Perron/Zivot-Andrews with embedded
  critical-value tables), GPH & local-Whittle long-memory estimators, HAR-RV /
  bipower / jumps, Nelson-Siegel(-Svensson) curve factors, Amihud/Roll/Amivest
  liquidity, EVT Hill/POT-GPD VaR-ES, and a causal (trailing-window) seasonal-trend
  decomposition. A two-sided STL leaks and is deliberately not offered.
- `StationarityDifferencer`, `AutoFracDiff`, `HARModel`, `NelsonSiegel` and
  `CausalSeasonalDecomposer` — `PanelTransformer`s that learn on train and freeze.
- `polars_features._ffd.estimate_ffd_order` — data-driven fractional-differencing
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
  0.46 MB wheel.
- `benchmarks/bench_hotspots.py` (ranked per-operation profile) and
  `benchmarks/show_capabilities.py` (optional-feature probe).
- `_numpy_stats.chebyshev_neighbour_counts`: exact, SciPy-free L-inf neighbour counts,
  so `sample_entropy` / `approximate_entropy` now work on the bare numpy+polars core
  (SciPy stays the faster backend when installed). Bare-core `test_tsfresh.py` went
  from 8 failed / 221 passed to 229 passed.

### Fixed

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

### Changed

- catch22 is 1.2×–2.0× faster (bit-exact): the remaining Python scans in
  `FC_LocalSimple_*`, `SB_MotifThree`, `SB_TransitionMatrix`, `_longest_run`,
  `CO_FirstMin_ac`, `CO_f1ecac`, `PD_PeriodicityWang` and `DN_OutlierInclude_*`
  are vectorised.
- k-Shape clustering is 4×–9× faster to fit and 8×–14× faster to score (bit-exact):
  forward FFTs are computed once per series/centroid instead of once per pair.
- `reduce.CrossSectionalPCA.transform` is 1.3×–2.0× faster (one array conversion
  instead of one per date).
- CI/release workflows dropped the dead Rust/maturin steps; releases now build a
  single universal `py3-none-any` wheel.
- Removed the dangling `narwhals` -> `interop` row from `_deps._MODULE_TO_EXTRA`
  (the `interop` extra went away in 0.4.0) and the corresponding docs entry.

### Fixed — catch22 is now SciPy-independent

- `catch22.PD_PeriodicityWang_th0_01` returned *different values* depending on whether
  SciPy was installed: its `LSQUnivariateSpline` detrend degraded silently to a zero
  spline on the bare core (140.0 -> 0.0 on a 512-point random walk). The least-squares
  cubic spline is now fitted in pure NumPy (`catch22._lsq_spline_fit`, a Cox-de Boor
  B-spline design matrix + `lstsq`), matching SciPy to ~1e-14 over n = 8..2048 and
  reproducing every SciPy-captured golden value exactly. This closes the last SciPy
  hole in catch22: all 24 features now compute on numpy + polars alone.

### Deferred from the implemented plans

Every milestone that assumed a compiled extension was dropped, since 0.4.0 ships a
pure-Python wheel: the native Arrow/Polars TreeSHAP kernel and the sparse
Möbius/Fourier (SPEX) engine (`explain` M6/M7), the Rust HFA cumulant contraction
(`reduce` M6), vendoring the `tsecon` crates and its golden-fixture CI harness
(econometrics M7), and `augurs`-backed MSTL. DCC-GARCH, vine copulas and Bayesian
VAR/FAVAR remain out of scope by design.

## [0.4.0] — 2026-09-04

A "light package" release: a numpy + polars core, ~10× faster import, and a
pure-Python distribution (the Rust extension is gone). Two **breaking** structural
changes are called out below.

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

- **The compiled Rust extension (`src/`, Cargo, maturin) has been removed.** PanelKit
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
  advertised.
- The hard-dependency slim flagged here as planned is **now done** in this 0.4.0
  release (see "BREAKING: slim hard dependencies" above).

## [0.3.0] — 2026-09-04

Unsupervised toolkit (clustering, dimensionality reduction, unsupervised feature
selection) ported leak-safely from the Sov.ai SDK, plus a cross-sectional/factor
evaluation layer and four confirmed correctness fixes. Every learned method fits on
the training fold only; genuinely offline methods are gated behind `leakage_safe=False`.

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

### Fixed — correctness
- **Deflated Sharpe Ratio**: `cross_validate` / `validate.cpcv` no longer deflate
  against `n_paths` (CV geometry) with a fallback variance — they now take an explicit
  `n_trials` (warning + `n_paths` floor when omitted) and use the empirical
  cross-trial variance `V = var(path_sharpes, ddof=1)`. New NaN-safe, annualization-aware
  `_sharpe`; PBO can rank by Sharpe (`statistic="sharpe"`).
- **`select.mda`**: permutation is now within-entity by default
  (`permute_within="entity"`) instead of a global cross-entity/time shuffle that
  produced off-manifold rows and inflated importances.
- **Fractional differencing**: consolidated to a single `ffd_weights` / `frac_diff_expr`
  source of truth (`polars_features._ffd`); all five Python surfaces now agree bit-for-bit
  (parity-tested), removing the Rust `threshold=0.0` divergence from the Python path.
- **`preprocessing.impute`**: `bfill` / `interpolate` now emit a `LeakageWarning`
  (opt out with `allow_leaky=True`); steers users to the point-in-time CAFE path.

## [0.2.0] — 2026-09-02

First release of the PanelKit panel core, plus the leak-safe feature-engineering,
labeling, and validation layer landing on top of the functime-derived engine.

### Added — panel core (shipped, experimental)
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
- Bumped version to `0.2.0` (past the live PyPI `0.1.7`); aligned `Cargo.toml`.
- Fixed `__version__` (was functime's `0.9.5`) and the package docstring to describe PanelKit.
- Top-level exports now use narrow `try/except ImportError` with a `warnings.warn`
  instead of blanket `contextlib.suppress(Exception)` (which silently hid real import bugs).
- Repositioned the project story: functime is **actively maintained**; PanelKit interoperates
  with functime/Nixtla rather than replacing them. Corrected `README.md`, `CHANGELOG.md`,
  `NOTICE`, and docs.
- Packaging hygiene: removed the committed prebuilt wheel from git and ignore `*.whl`;
  fixed the stale `Makefile` (uninstall `polars_features`, `BASE ?= main`, `[dev]` extra);
  fixed user-facing `pip install functime[...]` strings to `polars_features[...]`.
- Documentation: mkdocstrings API pages now target `polars_features.*`; `mkdocs.yml`
  site/repo URLs point at the PanelKit / sovai-research locations.

## [Unreleased]

### Phase 0 — Modernization & foundation

#### Added
- `NOTICE` file crediting functime as the upstream this work is derived from, with
  Apache-2.0 third-party notices and a clean-room statement for catch22 (reimplemented
  from Lubba et al. 2019, not vendored from GPL `pycatch22`).
- `llms.txt` discovery file (per the llms.txt convention) summarizing PanelKit, its
  module taxonomy, and core concepts for coding agents.
- `CHANGELOG.md` (this file).
- PanelKit-branded documentation: `docs/quickstart.md` and `docs/leakage.md`
  (correctness-by-construction narrative).

#### Changed
- Repositioned the project as **PanelKit** — "leak-safe, Rust-fast feature engineering
  and ML for panel data, built on Polars" — across `README.md`, `mkdocs.yml`, and docs.
- Rewrote `README.md` with the PanelKit one-liner/tagline, prominent functime credit,
  install guidance (note: import name remains `polars_features`), a 60-second quickstart
  of the target panel API, an honest shipped-vs-roadmap table, and ecosystem positioning.
- Updated `mkdocs.yml` site name/description and navigation for PanelKit.
- Updated `CONTRIBUTING.md` to describe the Tier B/C (Python-first, low Rust barrier)
  contribution path, the `panel_safe`/`leakage_safe` contracts, and good-first-issue posture.

#### Infrastructure (in progress)
- Modern packaging via `pyproject.toml` (maturin + abi3 wheels).
- CI and multi-platform wheel builds.
- Core panel protocol scaffolding (`PanelFrame`, leak-safe `Pipeline` contracts) — design phase.

#### Notes
- No public Python/Rust APIs were renamed in Phase 0. The `panelkit` rename is deferred.
- The functime-derived engine (feature extractors, forecasting, preprocessing, seasonality,
  cross-validation, metrics, LLM analysis) is unchanged and remains importable from
  `polars_features`.

## [Planned] [0.1.0] — "Maintained functime" wedge release

The first PanelKit release: a drop-in, **maintained** functime with modern packaging.

### Planned
- Verified compatibility with current Polars (`polars>=1.29`).
- Published wheels for Linux/macOS/Windows on supported Python versions.
- Green CI, test suite passing, reproducible builds.
- Documentation site rebranded to PanelKit while retaining the functime API surface.
- Clear migration note for functime users (same APIs, new home).

### Out of scope for 0.1.0 (later phases)
- `PanelFrame` / leak-safe `Pipeline` / CPCV / labeling and the full panel module taxonomy.
- Clean-room catch22 feature set.
- Public `panelkit` package rename.

[Unreleased]: https://github.com/sovai-research/polars-features/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/sovai-research/polars-features/releases/tag/v0.2.0
