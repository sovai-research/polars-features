# PanelKit

![PanelKit](./img/banner.png)

## Leak-safe, fast feature engineering and ML for panel data

**PanelKit** (import package `polars_features`) is a Polars-native toolkit for **panel data** —
many entities observed over time (stocks, customers, sensors, regions). It treats the
`(entity, time)` panel as a first-class, lazy, leak-safe object and gives you the whole
quant-ML workflow — **transform → extract → label → select → model → validate** — with no
lookahead, ever.

It is built on the foundations of
[functime](https://github.com/functime-org/functime) (Apache-2.0, actively maintained) and
interoperates with functime and Nixtla rather than replacing them.

!!! info "Naming"
    The project/brand is **PanelKit**. The current import/PyPI package is `polars_features`
    — the public rename to `panelkit` is planned but not yet effective. Import it as
    `import polars_features as pk`.

## Why PanelKit

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
import polars_features as pk

# A panel: many entities (tickers) observed over time.
prices = pl.read_parquet("prices.parquet")   # columns: ticker, day, close, volume
panel = pk.PanelFrame(prices, entity="ticker", time="day")

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
pk.assert_no_lookahead(
    pl.col("close").panel.zscore(window=10).over("ticker").alias("z"),
    panel,
)
```

Continue with the [Quickstart](./quickstart.md) for the full path — bulk feature extraction,
triple-barrier labels, CAFE imputation, and combinatorial purged cross-validation — every
step runnable.

## The benchmark headline

Panel feature generation (many entities × time) is PanelKit's core workload. Each computation
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
seasonality, cross-validation, metrics) is available now under the `polars_features` import.
On top of it, PanelKit adds the leak-safe panel layer:

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

## Where to next

- **[Installation](./installation.md)** — pip, optional extras, and version notes.
- **[Quickstart](./quickstart.md)** — a runnable 5-minute tour end to end.
- **[Leakage & correctness-by-construction](./leakage.md)** — the contracts, in depth.

## License

PanelKit is distributed under the **Apache License 2.0**, retained from functime, which is
credited as the upstream this work derives from. CAFE imputation is powered by the optional
`cafe-impute` dependency (MIT, Sov.ai), installed via the `cafe` extra.
