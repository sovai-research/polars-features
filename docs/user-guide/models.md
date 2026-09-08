# Panel ML Models

PanelKit ships a small estimator layer that terminates a modelling pipeline with
a **leak-safe, panel-aware** supervised model. Each estimator is a thin wrapper
around an ordinary scikit-learn-style estimator that:

- learns its parameters in `fit` from **only** the rows you hand it (the training
  fold), and never re-fits at `predict` time, and
- re-aligns its predictions back to the panel's `(entity, time)` keys, so the
  output is itself a panel you can join, evaluate, or feed downstream.

Because a wrapper fits solely on the `fit` panel, it is both `panel_safe` and
`leakage_safe`: dropping one at the end of a `Pipeline` keeps the whole chain
leak-safe under any purged / walk-forward split.

Two families live in `polars_features.models`:

| Class | Backend | Notes |
| --- | --- | --- |
| `PanelSklearnRegressor` / `PanelSklearnClassifier` | any sklearn estimator | Defaults to `HistGradientBoosting{Regressor,Classifier}` — no extra dependency. |
| `PanelLGBMRegressor` / `PanelLGBMClassifier` | LightGBM | Lazy import; falls back to the sklearn gradient booster (with a warning) if LightGBM is not installed. |

!!! tip "GBDT is the sane default for panel/tabular data"
    Gradient-boosted decision trees handle mixed-scale features, non-linear
    interactions, and missing values with almost no preprocessing, and they
    consistently beat linear models on tabular panels. That is why the default
    estimator — when you pass none — is a histogram gradient booster.

## The data model

Every estimator works on a **bare long-format panel**: one row per
`(entity, time)`, feature columns, and a target column. You point the estimator
at the keys with `entity=` / `time=` and at the label with `target=`. Any column
that is neither a key, the target, nor the sample-weight column is treated as a
feature unless you pass an explicit `features=` list.

```python
import numpy as np
import polars as pl

rng = np.random.default_rng(0)
rows = []
for e in range(6):
    level = rng.normal()
    for t in range(40):
        mom = rng.normal()
        vol = rng.normal()
        ret_fwd = 2.0 * mom - 1.0 * vol + level + rng.normal() * 0.1
        rows.append(
            {"ticker": f"stock_{e}", "date": t,
             "mom": mom, "vol": vol, "ret_fwd": ret_fwd}
        )
df = pl.DataFrame(rows)

# A simple time-based split: train on the past, predict the future.
train = df.filter(pl.col("date") < 30)
test = df.filter(pl.col("date") >= 30)
print(df.shape)   # (240, 5)
```

## Regression: features → model → predictions

The primary path needs nothing beyond scikit-learn. With no `estimator=`, you
get a `HistGradientBoostingRegressor`.

```python
from polars_features.models import PanelSklearnRegressor

model = PanelSklearnRegressor(
    target="ret_fwd",
    features=["mom", "vol"],   # omit to use every non-key, non-target column
    entity="ticker",
    time="date",
)
model.fit(train)

preds = model.predict(test).collect()
print(preds.columns)   # ['ticker', 'date', 'prediction']
print(preds.head(3))
```

```text
shape: (3, 3)
┌─────────┬──────┬────────────┐
│ ticker  ┆ date ┆ prediction │
│ ---     ┆ ---  ┆ ---        │
│ str     ┆ i64  ┆ f64        │
╞═════════╪══════╪════════════╡
│ stock_0 ┆ 30   ┆ 0.393746   │
│ stock_0 ┆ 31   ┆ 0.520052   │
│ stock_0 ┆ 32   ┆ -0.733444  │
└─────────┴──────┴────────────┘
```

### Where predictions land

`predict` returns a **`PanelFrame`** carrying only the `(entity, time)` keys plus
one prediction column (default name `"prediction"`). Call `.collect()` to get a
plain polars `DataFrame`. The output is keyed the same way as the input, so you
can join it straight back onto the truth column to score it:

```python
scored = preds.join(
    test.select("ticker", "date", "ret_fwd"),
    on=["ticker", "date"],
)
mae = (scored["prediction"] - scored["ret_fwd"]).abs().mean()
print(round(mae, 3))
```

After `fit`, the resolved columns are available for inspection:

```python
print(model.target_)     # 'ret_fwd'
print(model.features_)    # ['mom', 'vol']
print(type(model.estimator_).__name__)   # 'HistGradientBoostingRegressor'
```

### Wrapping any sklearn estimator

Pass any estimator with the sklearn `fit` / `predict` API as the first argument.
Rename the output column with `prediction_col=`.

```python
from sklearn.linear_model import LinearRegression

linear = PanelSklearnRegressor(
    LinearRegression(),
    target="ret_fwd",
    features=["mom", "vol"],
    prediction_col="yhat",
    entity="ticker",
    time="date",
)
linear.fit(train)
print(linear.predict(test).collect().columns)   # ['ticker', 'date', 'yhat']
```

A fresh clone of the estimator is made on every `fit`, so re-using one wrapper
instance across cross-validation folds never leaks state from a previous fold.

### Sample weights

Point `sample_weight=` at a column of per-row weights. It is forwarded to the
underlying estimator's `fit`. If the estimator does not accept `sample_weight`,
the wrapper warns and refits without it rather than crashing.

```python
weighted = train.with_columns(
    # e.g. down-weight older observations
    (pl.col("date") / 30.0).alias("w")
)
model = PanelSklearnRegressor(
    target="ret_fwd",
    features=["mom", "vol"],
    sample_weight="w",
    entity="ticker",
    time="date",
).fit(weighted)
```

## Classification and probabilities

`PanelSklearnClassifier` predicts discrete labels and additionally exposes
`predict_proba` when the underlying estimator supports it. The default is a
`HistGradientBoostingClassifier`.

```python
from polars_features.models import PanelSklearnClassifier

# Binary label: was the forward return positive?
train_c = train.with_columns((pl.col("ret_fwd") > 0).cast(pl.Int64).alias("up"))
test_c = test.with_columns((pl.col("ret_fwd") > 0).cast(pl.Int64).alias("up"))

clf = PanelSklearnClassifier(
    target="up",
    features=["mom", "vol"],
    entity="ticker",
    time="date",
).fit(train_c)

labels = clf.predict(test_c).collect()          # 'prediction' column of class labels
proba = clf.predict_proba(test_c).collect()
print(proba.columns)
# ['ticker', 'date', 'prediction_proba_0', 'prediction_proba_1']
print(clf.classes_)   # [0 1]
```

One probability column per class is emitted, named
`f"{prediction_col}_proba_{class}"`. Calling `predict_proba` on an estimator
without one raises `AttributeError`.

## LightGBM with graceful fallback

`PanelLGBMRegressor` / `PanelLGBMClassifier` are convenience constructors that
lazily import LightGBM and forward any keyword arguments as LightGBM
hyper-parameters. They subclass the sklearn wrappers, so everything above
(`target=`, `features=`, `sample_weight=`, `prediction_col=`, `predict_proba`)
applies unchanged.

```python
from polars_features.models import PanelLGBMRegressor

gbm = PanelLGBMRegressor(
    target="ret_fwd",
    features=["mom", "vol"],
    n_estimators=100,
    learning_rate=0.05,
    verbosity=-1,
    entity="ticker",
    time="date",
).fit(train)

print(gbm.predict(test).collect().head(3))
```

!!! note "Fallback when LightGBM is missing"
    If `import lightgbm` fails, the constructor does **not** error. It emits a
    `UserWarning` —

    ```text
    lightgbm is not installed; falling back to sklearn HistGradientBoostingRegressor.
    Install lightgbm (`pip install lightgbm`) to use the gradient booster, or use
    PanelSklearn{Regressor,Classifier} directly.
    ```

    — and transparently uses `HistGradientBoosting{Regressor,Classifier}` with
    the same hyper-parameters instead. Your code keeps working without the
    optional dependency; install `lightgbm` to get the real booster.

## Leak-safety in a nutshell

- Parameters come **only** from the `fit` panel; `predict` never re-fits.
- Rows are never mixed across entities during fitting.
- Both `panel_safe` and `leakage_safe` are `True`, so these estimators can end a
  `Pipeline` and stay leak-safe under a purged / walk-forward CV splitter — fit
  on each training fold, predict the held-out block.
