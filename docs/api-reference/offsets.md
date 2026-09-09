# Frequency offsets

`panelary.offsets` maps a Polars offset alias (`"1d"`, `"3mo"`, `"1w"`, …) to the seasonal
periods that alias implies. Seasonality helpers, Fourier term generation and seasonal-naive
forecasters all need to know "how many observations is one cycle?", and this module is the one
place that question is answered, so a daily panel means the same thing everywhere in Panelary.

It is a pure lookup over the frequency string — no data is read, so nothing here can leak.

## What's here

| Entry point | Purpose |
| --- | --- |
| `freq_to_sp` | Seasonal periods implied by an offset alias (Hyndman's table) |

## See also

- [`seasonality`](seasonality.md) — calendar, holiday and Fourier features that consume these periods.

## API

::: panelary.offsets
