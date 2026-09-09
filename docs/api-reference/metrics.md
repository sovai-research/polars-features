# Point-forecast metrics

`panelary.metrics.point` is the error-metric layer: scale-dependent, percentage and scaled
metrics, all vectorised in Polars and all computed **per entity** so a panel score is a frame
of per-entity errors rather than one number that hides the cross-section.

Metrics are pure functions of `(y_true, y_pred)` and learn nothing, so they cannot leak on
their own. The scaled metrics are the exception worth naming: `mase` and `rmsse` divide by an
in-sample naive error, which must come from the **training** rows — passing the full series
denominator quietly imports test-period information into the score.

## What's here

| Family | Metrics |
| --- | --- |
| Absolute error | `mae`, `mse`, `rmse` |
| Percentage error | `mape`, `smape`, `smape_original` |
| Scaled error | `mase`, `rmsse` |
| Bias | `mfe`, `overforecast`, `underforecast` |

## See also

- [`multi_objective`](multi-objective.md) — combining several of these into one score.
- [`evaluation`](evaluation.md) — ranking entities by any of them.

## API

::: panelary.metrics.point
