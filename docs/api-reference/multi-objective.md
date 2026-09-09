# Multi-objective scoring

`panelary.metrics.multi_objective` computes, compares and optimises forecasts against
**several** metrics at once. Real forecast selection is rarely single-objective: accuracy,
bias and scale-robustness trade against each other, and a model chosen on `rmse` alone is
usually systematically biased. `Metrics` bundles a metric set, `score_forecast` and
`score_backtest` apply it to a single horizon or to a whole backtest, and `summarize_scores`
reduces the per-entity frame to a comparable summary.

Scoring is a pure function of forecasts and actuals — it learns nothing and therefore leaks
nothing. Leak-safety here is inherited entirely from the split that produced the forecasts, so
score the output of a purged backtest, not of a globally-fitted model.

## What's here

| Entry point | Purpose |
| --- | --- |
| `Metrics` | The metric bundle to score against |
| `score_forecast` | Score one forecast frame against actuals |
| `score_backtest` | Score every fold of a backtest |
| `summarize_scores` | Reduce per-entity scores to a comparable summary |

## See also

- [`metrics`](metrics.md) — the individual point metrics being combined.

## API

::: panelary.metrics.multi_objective
