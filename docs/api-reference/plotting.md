# Plotting

`panelary.plotting` renders panels, forecasts, backtests and residual diagnostics as
interactive Plotly figures. Every function takes tidy panel frames — the same
`(entity, time, value)` shape the rest of Panelary works in — samples a manageable number of
entities, and returns a `plotly.graph_objects.Figure` you can further customise or export.

Plotting is presentation-only: it reads data and returns a figure, never a feature, so it sits
outside the leak-safety contract entirely. Plot whatever you like, including whole-sample
views — just do not turn a chart's whole-sample summary into a model input.

Plotly is an **optional** dependency, imported lazily: `pip install 'panelary[viz]'`.

## What's here

| Question | Entry point |
| --- | --- |
| What do these series look like? | `plot_entities`, `plot_panel` |
| How did the forecast track the actuals? | `plot_forecasts` |
| How did it do across backtest folds? | `plot_backtests` |
| Are the residuals structured? | `plot_residuals` |
| Which entities are hard to forecast, and how volatile are they? | `plot_comet` |
| Did the model beat the naive baseline? | `plot_fva` |

## See also

- [`evaluation`](evaluation.md) — the rankings `plot_comet` and `plot_fva` visualise.

## API

::: panelary.plotting
