# Seasonality features

`panelary.seasonality` generates deterministic calendar, holiday and Fourier regressors from
the time column alone. Because they are functions of the *timestamp* and nothing else, they
carry no information from any observation and are unconditionally leak-safe: a day-of-week
dummy for a future date is knowable today, which is exactly why the `make_future_*` helpers
can build the exogenous frame a forecast horizon needs.

`add_calendar_effects` and `add_fourier_terms` are dependency-free; the holiday calendars
require the optional `holidays` package (`pip install 'panelary[seasonality]'`).

Fourier terms are the compact way to encode a long or non-integer seasonal period — a handful
of sine/cosine pairs replaces hundreds of dummies — and the period comes from
[`freq_to_sp`](offsets.md) so it matches the panel's declared frequency.

## What's here

| Task | Entry point |
| --- | --- |
| Calendar dummies (day-of-week, month, quarter, …) | `add_calendar_effects` |
| Holiday indicators for a country / region | `add_holiday_effects` |
| Smooth harmonic seasonal terms | `add_fourier_terms` |
| The same features over a future horizon | `make_future_calendar_effects`, `make_future_holiday_effects` |

## See also

- [Seasonality guide](../user-guide/seasonality.md) — the narrative walkthrough.
- [`offsets`](offsets.md) — seasonal periods per frequency alias.
- [`econ.features`](econ-features.md) — `causal_seasonal_decompose` for *estimated* seasonality.

## API

::: panelary.seasonality
