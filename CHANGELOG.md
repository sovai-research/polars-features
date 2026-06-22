# Changelog

All notable changes to PanelKit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

PanelKit is built on the foundations of [functime](https://github.com/functime-org/functime)
(Apache-2.0, no longer maintained). The import/PyPI package is currently `polars_features`;
the public rename to `panelkit` is planned but not yet effective.

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

[Unreleased]: https://github.com/functime-org/functime/compare/main...HEAD
