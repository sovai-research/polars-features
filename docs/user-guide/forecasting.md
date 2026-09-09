# Global Forecasting Walkthrough

`panelary.forecasting` fits **global** forecasters: one model across a whole
collection of series. Every forecaster shares the same API, so swapping
`linear_model` for `lightgbm` or `auto_lasso` is a one-line change. See the
[forecasting API reference](../api-reference/forecasting.md) for the full list.

Two benchmark forecasters are implemented as pure Polars queries:

!!! info "Benchmark forecasters"
    - `naive` — random-walk forecaster.
    - `snaive` — seasonal-naive forecaster.

## Quickstart

Want to go straight into code? Run through every forecasting example with the
following script:

??? tip "`quickstart.py`"

    ```python
    --8<-- "docs/code/quickstart.py"
    ```

## Prepare data

Load a collection of time series — panel data — into a
[`polars.LazyFrame`](https://docs.pola.rs/py-polars/html/reference/lazyframe/index.html)
(recommended) or a `polars.DataFrame`, then split it into train and test
subsets.

```python
import polars as pl
from panelary.cross_validation import train_test_split
from panelary.metrics import mase
from panelary.seasonality import add_calendar_effects

# Load data
y = pl.read_parquet(
    "https://github.com/functime-org/functime/raw/main/data/commodities.parquet"
)
entity_col, time_col = y.columns[:2]
X = (
    y.select([entity_col, time_col])
    .pipe(add_calendar_effects(["month"]))
    .collect()
)

# Train-test splits
test_size = 3
freq = "1mo"
y_train, y_test = train_test_split(test_size)(y)
X_train, X_test = train_test_split(test_size)(X)
```

!!! info "Supported data schemas"
    `X` and `y` (each a `polars.LazyFrame` or `polars.DataFrame`) must contain at
    least three columns:

    - column 0 — the `entity` / `series_id` dimension;
    - column 1 — the `time` dimension, as an integer, `pl.Date`, or
      `pl.Datetime` series;
    - remaining columns — features.

## Fit / predict / score

Forecasters expose scikit-learn-compatible `.fit` and `.predict` methods.
`panelary.metrics` holds scoring functions for both point and probabilistic
forecasts.

```python
from panelary.forecasting import linear_model
from panelary.metrics import mase

# Fit
forecaster = linear_model(lags=24, freq="1mo")
forecaster.fit(y=y_train)

# Predict
y_pred = forecaster.predict(fh=3)

# Score
scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)
```

!!! tip "Transformers and splitters are curried"
    Every `transformer` and `splitter` is a
    [curried function](https://www.composingprograms.com/pages/16-higher-order-functions.html#currying):
    call it with its parameters to get a callable, then apply that with
    `.pipe(...)`.

    ```python
    from panelary.preprocessing import boxcox, impute

    # Chain operations with df.pipe
    X_splits: pl.LazyFrame = (
        X.pipe(boxcox(method="mle"))
        .pipe(impute(method="ffill"))
    )
    # Call .collect to execute the query
    X_new = X_splits.collect()
    ```

    A `forecaster` is callable too, which runs fit-then-predict in one line:

    ```python
    from panelary.forecasting import linear_model

    y_pred = linear_model(lags=24, freq="1mo")(
        y=y_train,
        fh=3,
        X=X_train,
        X_future=X_test,
    )
    ```

!!! warning "Everything is lazy"
    The transformers and splitters in `cross_validation`, `feature_extraction`
    and `preprocessing` return `LazyFrame`s — a query plan against the input
    frame. **Nothing is computed until you call `.collect()`.**

    Preprocess `X` and `y` lazily for the best performance: Polars can then
    optimise every operation at once, and series-by-series transforms
    (`boxcox`, `impute`, `diff`) are chained into a *single* `group_by` instead
    of one `group_by`-aggregate per transform.

## Global forecasting

Every forecaster exposes a scikit-learn `fit` / `predict` API. `fit` takes `y`
and an optional `X`; `predict` takes the forecast horizon `fh: int`, the
frequency alias `freq: str`, and an optional `X`.

!!! info "Supported frequency aliases"
    `1s` (second) · `1m` (minute) · `30m` · `1h` (hour) · `1d` (day) ·
    `1w` (week) · `1mo` (calendar month) · `3mo` (calendar quarter) ·
    `1y` (calendar year) · `1i` (index count)

```python
from panelary.forecasting import linear_model
from panelary.metrics import mase

# Fit
forecaster = linear_model(lags=24, freq="1mo")
forecaster.fit(y=y_train)

# Predict
y_pred = forecaster.predict(fh=3)

# Score
scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)
```

!!! question "Global vs local forecasting"
    **Panelary supports global forecasters only.** A global forecaster fits and
    predicts a whole collection of series with a *single* model; a local
    forecaster (ARIMA, ETS, Theta) fits one model per series. Collections of
    related series — panel data — are everywhere:

    - sales across products in a retail store,
    - churn rates across customer segments,
    - sensor readings across devices in a factory,
    - delivery times across trucks in a logistics fleet.

    Global forecasters trained on a collection of similar series consistently
    outperform local ones.[^1] All top 50 competitors in the M5 Forecasting
    Competition used a global LightGBM model.[^2]

    [^1]: Montero-Manso, P., & Hyndman, R. J. (2021). Principles and algorithms for forecasting groups of time series: Locality and globality. *International Journal of Forecasting*, 37(4), 1632-1653.

    [^2]: Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting*.

!!! tip "You probably do not need a cluster"
    Local forecasting scales by brute force: one fit-predict per series, farmed
    out across a distributed cluster. Every `forecaster`, `transformer`,
    `splitter` and `metric` in Panelary instead operates *globally* across the
    whole collection, as multi-threaded Polars queries — so a panel of a few
    thousand series is ordinary single-machine work.

## Benchmark forecasters

Naive and seasonal-naive forecasts are surprisingly hard to beat, so always run
them as benchmarks. Panelary implements both as pure Polars queries, executed in
lazy streaming mode for speed and memory efficiency.

```python
from panelary.forecasting import naive, snaive

y_pred_naive = naive(freq="1mo")(y=y_train, fh=12)
# sp = seasonal periods (length of one seasonal cycle)
y_pred_snaive = snaive(freq="1mo", sp=12)(y=y_train, fh=12)
```

## Exogenous regressors

Every forecaster accepts exogenous regressors. Pass them as `X` at `fit` time
and as `X` at `predict` time (or `X_future` in the one-line call form).

```python
from panelary.forecasting import linear_model

forecaster = linear_model(lags=24, fit_intercept=False, freq="1mo")
forecaster.fit(y=y_train, X=X_train)
y_pred = forecaster.predict(fh=3, X=X_test)
```

## Transformations and preprocessing

Every forecaster takes two optional parameters, `target_transform` and
`feature_transform`. Each accepts a single Panelary transformer — for example
`diff(order=1, fill_strategy="backward")` or `detrend(freq="1mo",
method="linear")` — or a list of them.

- **`target_transform`** transforms `y` before fit and predict, then applies the
  **inverse** after predict, so the forecast comes back on the original scale.
  It therefore only accepts *invertible* transformers (`diff`, `detrend`,
  `scale`, `boxcox`, `yeojohnson`, `deseasonalize_fourier`).
- **`feature_transform`** transforms `X` before fit and predict.

Use both rather than transforming by hand: they are what keeps the training and
prediction pipelines identical, which is the usual source of inconsistent
feature engineering and leakage. See the API reference for
[preprocessing](../api-reference/preprocessing.md) and
[feature extractors](../api-reference/feature-extractors.md).

### Target transform

```python
from panelary.forecasting import linear_model
from panelary.preprocessing import boxcox, diff, scale

# First differences
forecaster = linear_model(
    freq="1mo", lags=12,
    target_transform=diff(order=1, fill_strategy="backward"),
)

# Or local standardisation
forecaster = linear_model(freq="1mo", lags=12, target_transform=scale())

# Or Box-Cox
forecaster = linear_model(freq="1mo", lags=12, target_transform=boxcox())

# Or chain: first differences, then Box-Cox
forecaster = linear_model(
    freq="1mo",
    lags=12,
    target_transform=[
        diff(order=1, fill_strategy="backward"),
        boxcox(),
    ],
)
```

### Feature transform

```python
from panelary.forecasting import linear_model
from panelary.preprocessing import roll
from panelary.seasonality import add_fourier_terms

# Fourier terms to model complex seasonality
forecaster = linear_model(
    freq="1mo",
    lags=12,
    feature_transform=add_fourier_terms(sp=12, K=3),
)

# Moving-average features on the exogenous regressors
forecaster = linear_model(
    freq="1mo",
    lags=12,
    feature_transform=roll(window_sizes=[6, 12], stats=["mean", "std"], freq="1mo"),
)

# Or both
forecaster = linear_model(
    freq="1mo",
    lags=12,
    feature_transform=[
        add_fourier_terms(sp=12, K=3),
        roll(window_sizes=[6, 12], stats=["mean", "std"], freq="1mo"),
    ],
)
```

### Target and feature transform together

```python
from panelary.forecasting import linear_model
from panelary.preprocessing import scale
from panelary.seasonality import add_fourier_terms

forecaster = linear_model(
    freq="1mo",
    lags=12,
    target_transform=scale(),
    feature_transform=add_fourier_terms(sp=12, K=3),
)
```

## Forecast strategies

Three strategies are supported: `recursive`, `direct` multi-step, and an
`ensemble` of both.

```python
from panelary.forecasting import linear_model

fh = 3
max_horizons = 12   # number of direct models

# Recursive (default)
recursive_forecaster = linear_model(lags=12, freq="1mo", strategy="recursive")
y_pred_rec = recursive_forecaster(y=y_train, fh=fh)

# Direct
direct_forecaster = linear_model(
    lags=12, freq="1mo", strategy="direct", max_horizons=max_horizons,
)
y_pred_dir = direct_forecaster(y=y_train, fh=fh)

# Ensemble
ensemble_forecaster = linear_model(
    lags=12, freq="1mo", strategy="ensemble", max_horizons=max_horizons,
)
y_pred_ens = ensemble_forecaster(y=y_train, fh=fh)
```

`max_horizons` is the number of horizon-specific models fitted under the
`direct` and `ensemble` strategies. With `max_horizons=12`, twelve forecasters
are fitted in total: one for the 1-step-ahead forecast, one for 2-steps-ahead,
and so on through 12.

## Censored forecasts

Real-world e-commerce and logistics targets are full of zeros — periods with no
sales. `censored_model` handles this by training a binary classifier plus two
forecasters: the classifier predicts whether the target falls above or below a
threshold, and the final forecast is the probability-weighted average of the
above-threshold and below-threshold forecasters.

```python
import polars as pl
from panelary.forecasting import censored_model

# The M5 competition Walmart dataset
m5_y_train = pl.read_parquet("data/m5_y_train_sample.parquet")
m5_X_train = pl.read_parquet("data/m5_X_train_sample.parquet")
m5_X_future = pl.read_parquet("data/m5_X_future_sample.parquet")

# Fit-predict with threshold = 0.0
y_pred = censored_model(lags=3, threshold=0.0, freq="1d")(
    y=m5_y_train, X=m5_X_train, fh=28, X_future=m5_X_future,
)
```

!!! tip "Custom classifier and regressor"
    By default `censored_model` uses sklearn's `HistGradientBoostingClassifier`
    and `HistGradientBoostingRegressor`. To supply your own, pass functions that
    take `X` and `y` numpy arrays and return a *fitted* sklearn-compatible
    estimator.

    ```python
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPRegressor


    def regress(X: np.ndarray, y: np.ndarray):
        regressor = MLPRegressor()
        regressor.fit(X=X, y=y)
        return regressor


    def classify(X: np.ndarray, y: np.ndarray):
        classifier = RandomForestClassifier()
        classifier.fit(X=X, y=y)
        return classifier


    forecaster = censored_model(
        lags=3,
        threshold=0.0,
        freq="1d",
        classify=classify,
        regress=regress,
    )
    y_pred = forecaster(y=m5_y_train, X=m5_X_train, fh=28, X_future=m5_X_future)
    ```

## Automated parameter tuning

The `auto_*` forecasters tune the number of lagged regressors *and* the model's
hyperparameters (e.g. `alpha` for Lasso). Cross-validation, lag tuning and
hyperparameter tuning run simultaneously for efficiency.

### Optimal lag length

`auto_*` forecasters search over lag counts between `min_lags` and `max_lags`
by cross-validation, and keep the model with the lowest average RMSE across
splits.

```python
from panelary.forecasting import auto_linear_model

# Fit, then predict
forecaster = auto_linear_model(min_lags=20, max_lags=24, freq="1mo")
forecaster.fit(y=y_train, X=X_train)
y_pred = forecaster.predict(fh=3, X=X_test)

# Or fit and predict in one call
y_pred = auto_linear_model(min_lags=20, max_lags=24, freq="1mo")(
    y=y_train,
    fh=3,
    X=X_train,
    X_future=X_test,
)
```

### Hyperparameter tuning

Panelary uses [FLAML](https://microsoft.github.io/FLAML/docs/getting-started)
under the hood. Pass a `search_space`, initial `points_to_evaluate`, and a
`time_budget` in seconds.

!!! tip "Sane defaults"
    Leaving `search_space` or `points_to_evaluate` as `None` uses default search
    spaces drawn from industry practice, top Kaggle solutions, and research.

```python
from flaml import tune
from panelary.forecasting import auto_lightgbm

max_depth = -1
DEFAULT_TREE_DEPTH = 6

search_space = {
    "reg_alpha": tune.loguniform(1e-08, 10.0),
    "reg_lambda": tune.loguniform(1e-08, 10.0),
    "num_leaves": tune.randint(
        2, 2**max_depth if max_depth > 0 else 2**DEFAULT_TREE_DEPTH
    ),
    "colsample_bytree": tune.uniform(0.4, 1.0),
    "subsample": tune.uniform(0.4, 1.0),
    "subsample_freq": tune.randint(1, 7),
    "min_child_samples": tune.qlograndint(5, 100, 5),
}
points_to_evaluate = [
    {
        "num_leaves": 31,
        "colsample_bytree": 1.0,
        "subsample": 1.0,
        "min_child_samples": 20,
    }
]

forecaster = auto_lightgbm(
    freq="1mo",
    min_lags=20,
    max_lags=24,
    time_budget=420,
    search_space=search_space,
    points_to_evaluate=points_to_evaluate,
)
forecaster.fit(y=y_train)

# Best lags and model hyperparameters
best_params = forecaster.best_params
```

!!! note "Optional dependency"
    `auto_lightgbm` and the other FLAML-driven paths need the `automl` extra:
    `pip install 'panelary[automl]'`. Without it the name is bound to a
    placeholder that raises an actionable `ImportError` when called.

## Backtesting

Every forecaster and auto-forecaster has a `backtest` method, built on
`expanding_window_split` and `sliding_window_split` from
[`panelary.cross_validation`](../api-reference/cross-validation.md).

```python
from panelary.forecasting import linear_model

forecaster = linear_model(lags=24, fit_intercept=False, freq="1mo")
y_preds, y_resids = forecaster.backtest(
    y=y_train,
    X=X_train,
    test_size=6,
    step_size=1,
    n_splits=3,
    window_size=1,          # only applies when strategy="sliding"
    strategy="expanding",
    # Raises ValueError if drop_short=False and some entities are too short
    drop_short=True,
)
```

!!! warning "These splitters are not purged"
    `expanding_window_split` and `sliding_window_split` respect time order but
    do **not** purge or embargo overlapping labels. When your target spans
    several periods — an `h`-step forward return, a triple-barrier label — use
    the purged splitters described in [Validation](validation.md) instead.

## Probabilistic forecasts

Two ways to produce prediction intervals.

### Quantile regression

Supported by the LightGBM, XGBoost and CatBoost forecasters and their automated
equivalents.

```python
from panelary.forecasting import auto_lightgbm

# Forecasts at the 10th and 90th percentile
y_pred_10 = auto_lightgbm(alpha=0.1, freq="1d")(y=y_train, fh=28)
y_pred_90 = auto_lightgbm(alpha=0.9, freq="1d")(y=y_train, fh=28)
```

### Conformal prediction

`panelary.conformal.conformalize` implements batch prediction intervals (EnbPI)
from [Conformal prediction interval for dynamic time-series](https://arxiv.org/abs/2010.09107).
It needs the point forecast, the backtest predictions, and the backtest
residuals.

```python
from panelary.conformal import conformalize
from panelary.forecasting import linear_model

forecaster = linear_model(lags=24, fit_intercept=False, freq="1mo")
y_preds, y_resids = forecaster.backtest(y=y_train, X=X_train)
y_pred = forecaster.predict(fh=3, X=X_test)

y_pred_quantiles = conformalize(
    y_pred=y_pred,
    y_preds=y_preds,
    y_resids=y_resids,
    alphas=[0.1, 0.9],
)
```

For the time-series-aware conformal methods that survive drift — adaptive
conformal (ACI), conformal PID, NexCP and CQR — see
[Honest validation](honest-validation.md#7-prediction-intervals-that-survive-drift).
