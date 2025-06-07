# polars_features

**Fast-Speed Time-series Machine Learning At Scale**


---
Built on `functime`, `polars_features` is a powerful Python library for production-ready forecasting and time-series feature extraction on large panel datasets, compatible with `polars==1.29.0`.

It includes preprocessing tools (Box-Cox, differencing), cross-validation (expanding/sliding windows), forecast metrics (MASE, SMAPE, etc.), and more. All components are optimized as lazy Polars transformations.

## Highlights

- Fast: Feature extraction and forecasting across 100k+ time series in seconds
- Efficient: Parallelized, memory-safe pipelines built on Polars
- Reliable: ML components proven in competitions and production
- Full support for exogenous features in any forecaster
- Expanding/sliding window backtesting
- Optional automated lag selection and hyperparameter tuning via FLAML
- Integrated LLM module for interpreting and analyzing forecasts

## Installation

Basic install:

```bash
pip install polars_features
```

Install with optional extras:

```bash
pip install "polars_features[llm,lgb]"
```

- `cat`: enable CatBoost support
- `xgb`: enable XGBoost support
- `lgb`: enable LightGBM support
- `llm`: enable LLM-based forecast analysis

## Forecasting Example

```python
import polars as pl
from polars_features.cross_validation import train_test_split
from polars_features.seasonality import add_fourier_terms
from polars_features.forecasting import linear_model
from polars_features.preprocessing import scale
from polars_features.metrics import mase

y = pl.read_parquet("https://github.com/functime-org/functime/raw/main/data/commodities.parquet")
entity_col, time_col = y.columns[:2]

y_train, y_test = y.pipe(train_test_split(test_size=3))

forecaster = linear_model(freq="1mo", lags=24)
forecaster.fit(y=y_train)
y_pred = forecaster.predict(fh=3)

# One-liner
y_pred = linear_model(freq="1mo", lags=24)(y=y_train, fh=3)

# Evaluate forecasts
scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)

# Forecast with transforms
forecaster = linear_model(
    freq="1mo",
    lags=24,
    target_transform=scale(),
    feature_transform=add_fourier_terms(sp=12, K=6)
)

# Forecast with exogenous regressors
X = (
    y.select([entity_col, time_col])
    .pipe(add_fourier_terms(sp=12, K=6)).collect()
)
X_train, X_future = y.pipe(train_test_split(test_size=3))
forecaster = linear_model(freq="1mo", lags=24)
forecaster.fit(y=y_train, X=X_train)
y_pred = forecaster.predict(fh=3, X=X_future)
```

Full guide: [https://docs.functime.ai/forecasting/](https://docs.functime.ai/forecasting/)

## Feature Extraction Example

```python
import polars as pl
import numpy as np
from polars_features.feature_extractors import FeatureExtractor, binned_entropy

y = pl.read_parquet("https://github.com/functime-org/functime/raw/main/data/commodities.parquet")
entity_col, time_col, value_col = y.columns

# Example on raw Series
entropy_val = binned_entropy(
    pl.Series(np.random.normal(0, 1, size=10)),
    bin_count=10
)

# Example on LazyFrame
features = (
    pl.LazyFrame({
        "index": list(range(10)),
        "value": np.random.normal(0, 1, size=10)
    })
    .select(
        binned_entropy=pl.col("value").ts.binned_entropy(bin_count=10),
        lempel_ziv_complexity=pl.col("value").ts.lempel_ziv_complexity(threshold=3),
        longest_streak_above_mean=pl.col("value").ts.longest_streak_above_mean(),
    )
)

# Grouped features
features = (
    y.group_by(entity_col)
    .agg(
        binned_entropy=pl.col(value_col).ts.binned_entropy(bin_count=10),
        lempel_ziv_complexity=pl.col(value_col).ts.lempel_ziv_complexity(threshold=3),
        longest_streak_above_mean=pl.col(value_col).ts.longest_streak_above_mean(),
    )
)

# Rolling window features
features = (
    y.group_by_dynamic(
        time_col,
        every="12mo",
        by=entity_col,
    )
    .agg(
        binned_entropy=pl.col(value_col).ts.binned_entropy(bin_count=10),
        lempel_ziv_complexity=pl.col(value_col).ts.lempel_ziv_complexity(threshold=3),
        longest_streak_above_mean=pl.col(value_col).ts.longest_streak_above_mean(),
    )
)
```

More feature extraction docs: [https://docs.functime.ai/feature-extraction/](https://docs.functime.ai/feature-extraction/)

## Related Projects

We recommend checking out [`polars-ds`](https://github.com/abstractqqq/polars_ds_extension) — a blazing fast extension library for common data science utilities in Polars, created by one of the core maintainers of `polars_features`.

## License

This project is distributed under the terms of the Apache 2.0 License.
