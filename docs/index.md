# Panelary

![Panelary](./img/banner.png)

**The best methods for panel data.**

Panelary is a Polars-native toolkit for **panel data** — many entities observed over time
(stocks, customers, sensors, regions). It treats the `(entity, time)` panel as a first-class,
lazy, leak-safe object and gives you the whole quant-ML workflow —
**transform → extract → label → select → model → validate** — with no lookahead, ever.

Install it with `pip install panelary` and import it as `import panelary as pn`.

It is built on the foundations of
[functime](https://github.com/functime-org/functime) (Apache-2.0, actively maintained) and
interoperates with functime and Nixtla rather than replacing them.

## Why Panelary

- **Leak-safe by construction.** Within-entity ops stay inside their entity and in time
  order (`panel_safe`); cross-sectional and fit-based ops never see the future or the test
  fold (`leakage_safe`). Ship the guarantee with a one-line test:
  [`assert_no_lookahead`](#a-60-second-tour).
- **Polars-native & lazy.** Every transform is a Polars expression via the `.panel` / `.xs`
  namespaces — nothing computes until you `.collect()`, so the engine optimizes the whole
  plan.
- **Polars-fast.** Panel feature generation runs **10–19× faster than pandas 3.0** on the same
  workloads, verified for exact numerical equality — see the
  [benchmarks](./benchmarks/vs_pandas.md).
- **Panel-first & quant-ML shaped.** The panel is the unit of work: cross-sectional
  ranking/neutralization, triple-barrier labels, purged & combinatorial cross-validation,
  and backtest-overfitting diagnostics (Deflated Sharpe, PBO) are built in.

## A 60-second tour

```python
import polars as pl
import panelary as pn

# A panel: many entities (tickers) observed over time.
prices = pl.read_parquet("prices.parquet")   # columns: ticker, day, close, volume
panel = pn.PanelFrame(prices, entity="ticker", time="day")

# Leak-safe feature engineering — the .panel / .xs expression namespaces.
feats = panel.with_columns(
    # within-entity, causal: fractional differencing per ticker, in time order
    pl.col("close").panel.frac_diff(d=0.4).over("ticker").alias("close_fd"),
    # within-entity, causal: trailing rolling z-score
    pl.col("close").panel.zscore(window=10).over("ticker").alias("close_z"),
    # cross-sectional: rank each day across all tickers (no lookahead)
    pl.col("close").xs.rank(normalize=True).over("day").alias("xs_rank"),
)

# Prove a feature never looks ahead (raises if it leaks).
pn.assert_no_lookahead(
    pl.col("close").panel.zscore(window=10).over("ticker").alias("z"),
    panel,
)
```

Continue with the [Quickstart](./quickstart.md) for the full path — bulk feature extraction,
triple-barrier labels, CAFE imputation, and combinatorial purged cross-validation — every
step runnable.

## The golden path

Every stage of the workflow has one obvious entry point on the top-level `pn` namespace, so
you rarely need to remember which submodule a method lives in. Reach for the underlying
classes when you want the full parameter surface.

| Stage | Verb | What it covers |
| --- | --- | --- |
| Fill gaps | `pn.impute` | Point-in-time imputation, CAFE and simpler baselines |
| Engineer | `pn.features` | Bulk and per-column leak-safe feature generation |
| Narrow | `pn.select` | Feature selection under purged cross-validation |
| Compress | `pn.reduce` | Latent factors and dimensionality reduction |
| Group | `pn.cluster` | Time-series and cross-sectional clustering |
| Fit | `pn.regression` | Panel regression and panel-aware ML estimators |
| Explain | `pn.causal` | Causal and econometric panel estimators |
| Detect | `pn.bubbles` | Explosive-behaviour and change-point detection |

## The benchmark headline

Panel feature generation (many entities × time) is Panelary's core workload. Each computation
is run **both** ways and the results asserted equal (within `1e-6`) *before* timing, so no
speedup comes from computing something cheaper.

| Workload (2.5M rows) | Speedup vs pandas 3.0 |
| --- | ---: |
| Rolling z-score, per entity (window 21) | **10.7×** |
| Cross-sectional rank, per date | **17.8×** |
| Bulk: 10 tsfresh-style features per entity | **12.5×** |

Full methodology and 0.5M-row numbers in the [benchmarks](./benchmarks/vs_pandas.md).

## Who is this for

- **Quant researchers** building leak-safe factor and signal pipelines over cross-sections of
  assets, and validating them with purged / combinatorial CV before they trust a backtest.
- **ML engineers on panel data** — customers, devices, regions over time — who want
  sklearn-shaped `fit` / `transform` / `Pipeline` ergonomics that respect entity boundaries.
- **Anyone moving off pandas** for panel feature engineering who wants Polars speed without
  hand-rolling leak-safety.

## What's shipped today

The functime-derived engine (100+ `ts` feature extractors, forecasting, preprocessing,
seasonality, cross-validation, metrics) is available now under the `panelary` import.
On top of it, Panelary adds the leak-safe panel layer:

| Area | Status |
| --- | --- |
| `PanelFrame` panel/cross-section object | Shipped (experimental) |
| `.panel` / `.xs` expression + frame namespaces | Shipped (experimental) |
| Leak-safe `Pipeline` + `panel_safe` / `leakage_safe` contracts | Shipped (experimental) |
| Bulk feature extraction (`extract_features`) | Shipped (experimental) |
| Triple-barrier labeling (`triple_barrier`) | Shipped (experimental) |
| CAFE imputation (`cafe_impute` / `CafeImputer`, `cafe` extra) | Shipped (experimental) |
| Leakage verifier (`assert_no_lookahead`) | Shipped (experimental) |
| `PurgedKFold`, `CombinatorialPurgedCV` + `cross_validate` / `validate` runner | Shipped (experimental) |
| Deflated Sharpe ratio, Probability of Backtest Overfitting (PBO) | Shipped (experimental) |
| Feature selection (`select.mrmr` / `mda` / `mdi`), panel models | Shipped (experimental) |
| catch22 feature set (clean-room) | Shipped (experimental) |
| Latent factors (`reduce.HFAFactors` / `PCAFactors` / `ICAFactors`) | Shipped (experimental) |
| Leak-safe SHAP attribution (`explain.TreeAttributor`, `TimeAwareBackground`) | Shipped (experimental) |
| Honest validation (`validation`: CPCV, Romano-Wolf, SPA/MCS, bootstraps) | Shipped (experimental) |
| Time-series conformal (ACI / Conformal-PID / NexCP / CQR) | Shipped (experimental) |
| Panel econometrics (`econ`: HDFE, CCE/MG/PMG, Fama-MacBeth, IVX, DML) | Shipped (experimental) |
| Causal econometric features (`econ.features`: unit roots, HAR-RV, EVT, ...) | Shipped (experimental) |

## Where to next

- **[Installation](./installation.md)** — pip, optional extras, and version notes.
- **[Quickstart](./quickstart.md)** — a runnable 5-minute tour end to end.
- **[Leakage & correctness-by-construction](./leakage.md)** — the contracts, in depth.
- **[Interactions](./concepts/interactions.md)** — higher-order structure in the data (HFA) and in the model (Shapley interactions).

## License

Panelary is distributed under the **Apache License 2.0**, retained from functime, which is
credited as the upstream this work derives from. CAFE imputation is powered by the optional
`cafe-impute` dependency (MIT, Sov.ai), installed via the `cafe` extra.
