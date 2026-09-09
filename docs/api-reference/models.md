# Models

Concrete, leak-safe panel estimators that terminate a `Pipeline`. These are
thin, panel-aware wrappers around ordinary sklearn-style estimators that honour
the `PanelEstimator` contract: they learn parameters in `fit` from the training
rows they are handed (and only those rows), and produce predictions in `predict`
re-aligned to the panel's `(entity, time)` keys. Because a wrapper fits solely on
the data passed to `fit` and never re-fits at predict time, it is both
`panel_safe` and `leakage_safe` under any purged / walk-forward split.

Two families live here: `PanelSklearnRegressor` / `PanelSklearnClassifier` wrap
**any** sklearn-shaped estimator (defaulting to dependency-free
`HistGradientBoosting`), and `PanelLGBMRegressor` / `PanelLGBMClassifier` are
convenience constructors that lazily import LightGBM with a graceful sklearn
fallback.

## What's here

| Your problem | Entry point |
| --- | --- |
| Wrap any sklearn-shaped regressor / classifier | `PanelSklearnRegressor`, `PanelSklearnClassifier` |
| LightGBM with a graceful sklearn fallback | `PanelLGBMRegressor`, `PanelLGBMClassifier` |

## See also

- [Panel ML Models guide](../user-guide/models.md) — the narrative walkthrough.
- [`explain`](explain.md) — leak-safe attribution for a fitted model.
- [`validation`](validation.md) — evaluating one honestly.

## API

::: panelary.models
