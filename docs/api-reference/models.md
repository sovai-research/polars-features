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

::: polars_features.models
