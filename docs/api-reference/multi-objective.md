# Multi-objective scoring

`panelary.metrics.multi_objective` scores forecasts against **several** metrics at once.
Real forecast selection is rarely single-objective: accuracy, bias and scale-robustness
trade against each other, and a model chosen on `rmse` alone is usually systematically
biased. `score_forecast` computes the whole metric set per entity for one forecast,
`score_backtest` does the same for stacked predictions across CV splits, and
`summarize_scores` collapses that per-entity frame into a single `Metrics` record.

The metric set is fixed: MAE, MASE, MSE, overforecast, RMSE, RMSSE, SMAPE and
underforecast. SMAPE stands in for MAPE to avoid divide-by-zero on near-zero actuals.

Scoring is a pure function of forecasts and actuals — it learns nothing and therefore leaks
nothing. Leak-safety here is inherited entirely from the split that produced the forecasts, so
score the output of a purged backtest, not of a globally-fitted model.

## What's here

| Entry point | Purpose |
| --- | --- |
| `score_forecast` | Score one forecast frame against actuals, per entity |
| `score_backtest` | Score predictions stacked across CV splits, optionally aggregating overlapping forecasts first |
| `summarize_scores` | Reduce the per-entity score frame to one `Metrics` record (mean or median) |
| `Metrics` | Frozen dataclass holding one aggregated score per metric |

All four are re-exported from `panelary.metrics`, so `pn.metrics.score_forecast` and
`pn.metrics.multi_objective.score_forecast` are the same function.

## See also

- [`metrics`](metrics.md) — the individual point metrics being combined.

## API

::: panelary.metrics.multi_objective
