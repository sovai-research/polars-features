# Forecasting

`panelary.forecasting` is the global-forecasting layer: one model is fit across **all**
entities at once on lagged features, rather than one model per series. It ships both
individual forecasters (`lasso`, `ridge`, `lightgbm`, `xgboost`, `catboost`, `knn`, `ann`,
`naive`, `snaive`, …) and `auto_*` variants that tune the hyperparameters **and** the number
of lags. All of them implement the same `Forecaster` API, so they are interchangeable.

Recursive forecasting is causal by construction: features for horizon `h` are built from
observations at or before the forecast origin, and each step feeds its own prediction forward
rather than reading the realised value. Fitting is likewise confined to the rows handed to
`fit`, so a forecaster refit per fold under an
[expanding- or sliding-window split](cross-validation.md) never sees its own test period.

The `auto_*` forecasters use [`FLAML`](https://github.com/microsoft/FLAML) for hyperparameter
search via the CFO (Frugal Optimization for Cost-related Hyperparameters[^1]) algorithm. The
search **is** a fit: run it inside the training fold, never over the whole panel, or the
selected configuration encodes the test period.

## What's here

| Family | Forecasters |
| --- | --- |
| Baselines | `naive`, `snaive` |
| Linear | `linear_model`, `lasso`, `ridge`, `elastic_net` (+ `*_cv` variants) |
| Gradient boosting | `lightgbm`, `xgboost`, `catboost`, `flaml_lightgbm` |
| Neighbours & networks | `knn`, `ann` |
| Tuned (`FLAML`) | `auto_linear_model`, `auto_lasso`, `auto_ridge`, `auto_elastic_net`, `auto_knn`, `auto_lightgbm` |
| Specialised | `censored_model`, `zero_inflated_model`, `elite` |

Each back-end is an optional extra — `pip install 'panelary[forecasting,lightgbm]'` and
friends — imported lazily, so importing this module pulls in nothing you have not asked for.

## See also

- [Forecasting guide](../user-guide/forecasting.md) — the narrative walkthrough.
- [`cross_validation`](cross-validation.md) — the rolling-origin splits to backtest under.
- [`evaluation`](evaluation.md) / [`metrics`](metrics.md) — scoring the result.

## API

### `Forecaster`

::: panelary.base.forecaster.Forecaster

### `AutoForecaster`

::: panelary.forecasting.automl.AutoForecaster

### Forecasters

::: panelary.forecasting

[^1]: <https://arxiv.org/abs/2005.01571>
