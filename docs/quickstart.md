# Quickstart

**PanelKit** is leak-safe, Rust-fast feature engineering and ML for panel data, built on
[Polars](https://pola.rs/). It is built on the foundations of
[functime](https://github.com/functime-org/functime) (Apache-2.0, no longer maintained).

!!! info "Naming"
    The project/brand is **PanelKit**. The current import/PyPI package is `polars_features`
    — the public rename to `panelkit` is planned but not yet effective. Import it as
    `import polars_features as pk`.

## Install

```bash
pip install polars_features
```

Optional extras (carried over from functime):

```bash
pip install "polars_features[lgb,xgb,cat,llm]"
```

## 60-second quickstart — the target panel API

!!! warning "Roadmap (coming soon)"
    The high-level `PanelFrame` / `Pipeline` / `triple_barrier` / `validate.cpcv` API shown
    here is the **target ergonomics** for PanelKit and is **not yet shipped**. For what runs
    today, see [What works today](#what-works-today). The example shows where we are going.

```python
import polars as pl
import polars_features as pk  # roadmap API

# A panel: many entities (e.g. tickers) observed over time.
prices = pl.read_parquet("prices.parquet")  # ticker, date, close, volume, ...
panel = pk.PanelFrame(prices, entity="ticker", time="date")

# Leak-safe feature engineering — operations are panel-aware by construction.
feats = panel.with_columns(
    pk.col("close").panel.frac_diff(d=0.4).over("ticker").alias("close_fd"),  # within entity
    pk.col("close").xs.rank().over("date").alias("xs_rank"),                  # cross-sectional
    pk.col("close").panel.pct_change(periods=21).over("ticker").alias("mom_1m"),
)

# Labels: triple-barrier, leak-safe horizons.
labels = feats.label.triple_barrier(
    price="close", entity="ticker", time="date",
    pt=2.0, sl=1.0, max_holding="10d",
)

# A leak-safe pipeline: fit only ever sees the past of each fold.
pipe = pk.Pipeline([
    pk.transform.winsorize(limits=0.01),  # per-fold, per-date — never global
    pk.select.mrmr(k=20),                 # selection inside the fold
    pk.models.lgbm_classifier(),
])

# Combinatorial Purged Cross-Validation.
report = pk.model_selection.validate.cpcv(
    pipe, feats, labels, n_splits=6, n_test_groups=2, embargo="5d",
)
print(report.summary())
```

Two contracts hold throughout: **`panel_safe`** (within-entity ops stay inside their entity,
in time order) and **`leakage_safe`** (cross-sectional and fit-based ops never see the future
or the test fold). Read [Leakage & correctness](leakage.md) for the why.

## What works today

The functime-derived engine is available now under the `polars_features` import.

### Feature extraction (Polars `ts` namespace)

```python
import polars as pl
from polars_features.feature_extractors import binned_entropy

y = pl.read_parquet(
    "https://github.com/functime-org/functime/raw/main/data/commodities.parquet"
)
entity_col, time_col, value_col = y.columns

# Across many series (per entity)
features = y.group_by(entity_col).agg(
    binned_entropy=pl.col(value_col).ts.binned_entropy(bin_count=10),
    longest_streak_above_mean=pl.col(value_col).ts.longest_streak_above_mean(),
)
```

### Forecasting, backtesting, metrics

```python
from polars_features.cross_validation import train_test_split
from polars_features.forecasting import linear_model
from polars_features.metrics import mase

y_train, y_test = y.pipe(train_test_split(test_size=3))
y_pred = linear_model(freq="1mo", lags=24)(y=y_train, fh=3)
scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)
```

See the [User Guide](user-guide/forecasting.md) and [API Reference](api-reference/forecasting.md)
for the full shipped surface.

## Next steps

- [Leakage & correctness-by-construction](leakage.md)
- [Installation](installation.md)
- [Contributing](developer-guide/contributing.md)
