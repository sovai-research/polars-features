# Forecast & residual evaluation

`panelary.evaluation` ranks backtested forecasts and interrogates their residuals. Given a
frame of forecasts and one of actuals, it scores every entity, sorts the panel by the metric
you care about, and surfaces the entities whose errors are worst — or whose residuals are
still structured, which is the signal that a feature is missing.

Ranking is a **post-hoc, out-of-sample** report: it is computed from backtest output that has
already been produced under a leak-safe split, and nothing here feeds back into a model. The
residual diagnostics (`acf`, `ljung_box_test`, `normality_test`) are whole-series statistics by
construction, so read them as summaries of a completed backtest, never as per-row features.

## What's here

| Question | Entry point |
| --- | --- |
| Which entities are forecast worst? | `rank_point_forecasts` |
| Whose residuals still carry structure? | `rank_residuals` |
| Did the model beat the naive baseline? | `rank_fva` (forecast value added) |
| Residual autocorrelation and its confidence band | `acf`, `acf_formula`, `acf_confint_formula` |
| Is the residual white noise / Gaussian? | `ljung_box_test`, `normality_test` |

## See also

- [`metrics`](metrics.md) — the underlying point-forecast error metrics.
- [`plotting`](plotting.md) — `plot_comet` and `plot_fva` visualise these rankings.

## API

::: panelary.evaluation
