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

## [Unreleased] — 0.4.0-dev

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
- **Planned (breaking):** a future release will slim the hard dependency set —
  `flaml`, `holidays`, `scikit-learn`, `scipy`, and `tqdm` will move out of
  `[project.dependencies]` into the extras above. This has **not** happened yet;
  those packages are still installed by a bare `pip install polars-features`.

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
