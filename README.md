<div align="center">

# PanelKit

**Leak-safe, fast feature engineering and ML for panel data, built on Polars.**

*Built on the foundations of [functime](https://github.com/functime-org/functime). Panel-native data science in Polars: transform → extract → label → select → model → validate, with no lookahead, ever.*

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](./LICENSE)
[![Polars](https://img.shields.io/badge/built%20on-Polars-CD792C.svg)](https://pola.rs/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

</div>

---

> **Your backtest is lying to you.** Most panel-data feature engineering leaks the future into the past — a single careless `.shift`, a global `StandardScaler`, a cross-sectional rank computed over the whole sample — and your beautiful Sharpe ratio evaporates in production. PanelKit makes leakage-proof feature engineering the *default*, not an afterthought.

## What is PanelKit?

PanelKit is a feature-engineering and machine-learning toolkit for **panel data** (many entities observed over time — stocks, customers, sensors, regions). It is:

- **sklearn-familiar** — `fit` / `transform` / `Pipeline`, the API you already know.
- **Polars-native & lazy** — every transform is a Polars expression; nothing computes until you `.collect()`.
- **Panel/cross-section as a first-class object** — the panel (entity × time) is the unit of work, not a bag of rows.
- **Correct by construction** — operations are panel-aware and leak-safe. No lookahead, ever.

> **Heritage.** PanelKit is built on the foundations of [**functime**](https://github.com/functime-org/functime), an excellent Polars-native time-series library (Apache-2.0). functime is actively maintained (v1.0.0, May 2026); PanelKit reuses and credits its feature-extraction and forecasting engine, retains its license, and adds a panel-first, leak-safe feature-engineering / labeling / validation layer on top. See [`NOTICE`](./NOTICE).

## Installation

```bash
pip install polars_features
```

> **Note on names.** The project/brand is **PanelKit**. The current PyPI/import package is `polars_features` (the public rename to `panelkit` is planned but not yet effective). Import it as:
>
> ```python
> import polars_features  # PanelKit
> ```

Optional extras (LLM analysis, CAFE imputation, GPU, plotting, path signatures):

```bash
pip install "polars_features[llm,cafe]"   # or [all] for everything
```

## 60-second quickstart

> **Status:** the panel core — `PanelFrame`, the leak-safe `Pipeline`, `PurgedKFold` / `CombinatorialPurgedCV`, the de Prado overfitting metrics, and the `.panel` / `.xs` namespaces — is **shipped (experimental)** today. Labeling (`triple_barrier`), bulk `extract_features`, CAFE imputation, catch22, the `assert_no_lookahead` verifier, and the `cross_validate` / `validate` CPCV runner are **landing in this release**. The exact call signatures below (e.g. `pk.transform.winsorize`, `pk.select.mrmr`, `pk.models.lgbm_classifier`) show the target ergonomics and may still shift — see [What's shipped today](#whats-shipped-today).

```python
import polars as pl
import polars_features as pk  # panel core shipped (experimental)

# A panel: many entities (e.g. tickers) observed over time.
prices = pl.read_parquet("prices.parquet")  # columns: ticker, date, close, volume, ...
panel = pk.PanelFrame(prices, entity="ticker", time="date")

# --- Panel feature engineering: operations are leak-safe by construction ---
feats = panel.with_columns(
    # within-entity transform: fractional differencing computed per ticker, in time order
    pk.col("close").panel.frac_diff(d=0.4).over("ticker").alias("close_fd"),

    # cross-sectional rank: rank each date across all tickers (no lookahead)
    pk.col("close").xs.rank().over("date").alias("xs_rank"),

    # rolling momentum, per entity
    pk.col("close").panel.pct_change(periods=21).over("ticker").alias("mom_1m"),
)

# --- Labels: triple-barrier (López de Prado), leak-safe horizons ---
labels = feats.label.triple_barrier(
    price="close", entity="ticker", time="date",
    pt=2.0, sl=1.0, max_holding="10d",
)

# --- A leak-safe pipeline: fit only sees the past of each fold ---
pipe = pk.Pipeline([
    pk.transform.winsorize(limits=0.01),     # per-fold, per-date — never global
    pk.select.mrmr(k=20),                     # feature selection inside the fold
    pk.models.lgbm_classifier(),
])

# --- Validation: Combinatorial Purged Cross-Validation (CPCV) ---
report = pk.model_selection.validate.cpcv(
    pipe, feats, labels,
    n_splits=6, n_test_groups=2,
    embargo="5d",   # purge + embargo prevent train/test leakage around fold boundaries
)
print(report.summary())
```

Every step above respects two contracts: **`panel_safe`** (within-entity ops stay inside their entity and in time order) and **`leakage_safe`** (cross-sectional and fit-based ops never see the future or the test fold).

## What's shipped today

The functime-derived engine is **available now** under the `polars_features` import. You can use it for production forecasting and feature extraction over large panels:

```python
import polars as pl
from polars_features.feature_extractors import binned_entropy
from polars_features.forecasting import linear_model
from polars_features.cross_validation import train_test_split
from polars_features.metrics import mase

y = pl.read_parquet(
    "https://github.com/functime-org/functime/raw/main/data/commodities.parquet"
)

# Panel-aware feature extraction via the `ts` namespace
features = y.group_by(y.columns[0]).agg(
    binned_entropy=pl.col(y.columns[2]).ts.binned_entropy(bin_count=10),
    longest_streak_above_mean=pl.col(y.columns[2]).ts.longest_streak_above_mean(),
)

# Forecasting + backtesting + metrics
y_train, y_test = y.pipe(train_test_split(test_size=3))
y_pred = linear_model(freq="1mo", lags=24)(y=y_train, fh=3)
scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)
```

| Area | Status |
| --- | --- |
| 100+ Polars-native feature extractors (`ts` namespace) | Shipped (from functime) |
| Forecasting (linear, GBM, conformal, censored) | Shipped (from functime) |
| Preprocessing, seasonality, cross-validation, metrics | Shipped (from functime) |
| LLM forecast analysis | Shipped (from functime) |
| Modern packaging, CI, wheels (Phase 0) | Shipped |
| `PanelFrame` panel/cross-section object | Shipped (experimental) |
| Leak-safe `Pipeline` + `panel_safe` / `leakage_safe` contracts | Shipped (experimental) |
| `.panel` / `.xs` expression + frame namespaces | Shipped (experimental) |
| `PurgedKFold`, `CombinatorialPurgedCV` (CPCV) splitters | Shipped (experimental) |
| Deflated Sharpe ratio, Probability of Backtest Overfitting (PBO) | Shipped (experimental) |
| CAFE imputation (`cafe_impute` / `CafeImputer`) | Landing this release |
| Bulk feature extraction (`extract_features`) | Landing this release |
| Cross-sectional `.xs` ops | Landing this release |
| Labeling: triple-barrier (`triple_barrier`) | Landing this release |
| Leakage verifier (`assert_no_lookahead`) | Landing this release |
| CPCV runner (`cross_validate` / `validate` / `CVReport`) | Landing this release |
| catch22 (clean-room from Lubba et al. 2019, Apache-2.0) | Landing this release |
| Module taxonomy (`transform`, `interact`, `map`, `extract`, `select`, `synthesize`, `compare`, `label`, `signal_eval`, `models`, `model_selection`, `neutralize`) | In progress |

See [`CHANGELOG.md`](./CHANGELOG.md) and the docs for the full roadmap.

## How PanelKit fits in the ecosystem

PanelKit is opinionated about **panel-native correctness and ML workflow**, and deliberately *depends on* the ecosystem rather than competing with it:

- **`functime`** — PanelKit is built on functime's Polars-native feature-extraction and forecasting engine, and functime is actively maintained (v1.0.0, May 2026). We reuse and credit it under Apache-2.0 and add the panel object, leakage safety, labeling, and validation on top — we interoperate with functime, we don't replace it.
- **`polars-ds` / `polars_ta`** — we build on and recommend these for general Polars-native data-science and technical-analysis primitives; PanelKit focuses on the panel object, leakage safety, labeling, and validation that sit *above* them.
- **Nixtla (`statsforecast`, `mlforecast`, ...)** — Nixtla owns forecasting; PanelKit's center of gravity is leak-safe **feature engineering, labeling, selection, and cross-sectional ML** for panels. We interoperate, we don't reinvent forecasting.
- **`tsfresh` / `pycatch22`** — PanelKit's extractors are Polars-native and far faster; catch22 features are being clean-room reimplemented from the paper (not vendored from GPL `pycatch22`).

## Documentation

- [Quickstart](./docs/quickstart.md)
- [Leakage & correctness-by-construction](./docs/leakage.md)
- [Contributing](./CONTRIBUTING.md)
- [`llms.txt`](./llms.txt) (for coding agents)
- Legacy functime docs: <https://docs.functime.ai/>

## License

PanelKit is distributed under the **Apache License 2.0**, retained from functime. functime is credited as the upstream this work is derived from — see [`NOTICE`](./NOTICE) and [`LICENSE`](./LICENSE).

CAFE imputation (`cafe_impute` / `CafeImputer`) is powered by [`cafe-impute`](https://pypi.org/project/cafe-impute/) (MIT, Sov.ai), an optional dependency installed via the `cafe` extra.
