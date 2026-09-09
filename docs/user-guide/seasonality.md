# Seasonality and Holiday Effects

Seasonality is structure that repeats on a known calendar: a weekly retail
cycle, a monthly billing cycle, an annual weather cycle. Because the pattern is
a function of the *timestamp alone*, it can be encoded as ordinary exogenous
columns — which is exactly what `panelary.seasonality` does. Every helper here
reads only the time column, so none of them can look ahead.

```python
from panelary.seasonality import (
    add_calendar_effects, add_fourier_terms, add_holiday_effects,
)
from panelary.offsets import freq_to_sp
```

Like the rest of [`panelary.preprocessing`](preprocessing.md), these are curried
and lazy: call the helper with its parameters, apply it with `X.pipe(...)`, and
`.collect()` when you are ready.

## Modelling seasonality

### Seasonal periods

Given a Polars offset alias, `freq_to_sp` returns the seasonal periods that
belong to that frequency:

```python
from panelary.offsets import freq_to_sp

freq_to_sp("1mo")   # [12]
freq_to_sp("1d")    # [7, 365]
```

The full table:

| `freq` | Seasonal periods |
| --- | --- |
| `1s` | 60, 3 600, 86 400, 604 800, 31 557 600 |
| `1m` | 60, 1 440, 10 080, 525 960 |
| `30m` | 48, 336, 17 532 |
| `1h` | 24, 168, 8 766 |
| `1d` | 7, 365 |
| `1w` | 52 |
| `1mo` | 12 |
| `3mo` | 4 |
| `1y` | 1 |

### Method 1 — dummy variables / categoricals

`add_calendar_effects` extracts calendar attributes from the time column. There
are two ways to model them as discrete features: as a **categorical** column
(useful for learners with native categorical support, e.g. LightGBM) or as
**one-hot binary** columns. See
[Chapter 7.4: Seasonal dummy variables](https://otexts.com/fpp3/useful-predictors.html#seasonal-dummy-variables)
for a primer.

Supported attributes:

- `minute` — 0…59 (within the hour)
- `hour` — 0…23 (within the day)
- `day` — 1…31 (within the month)
- `weekday` — 1…7 (within the week)
- `week` — 1…53 (ISO week within the year)
- `month` — 1…12
- `quarter` — 1…4
- `year` — 1999, 2000, …

Each is read straight off the time column with the corresponding Polars
`.dt` accessor and cast to `Categorical`.

```python
from panelary.seasonality import add_calendar_effects

# One categorical column "month" with values 1, 2, ..., 12
X_new = X.pipe(add_calendar_effects(["month"])).collect()

# One-hot encoded instead: binary columns "month_1", "month_2", ..., "month_12"
X_new = X.pipe(add_calendar_effects(["month"], as_dummies=True)).collect()
```

!!! warning "The dummy variable trap"
    If you include *every* dummy column alongside an intercept the design matrix
    is rank-deficient. Either drop one level or set `fit_intercept=False` on the
    downstream regressor.

### Method 2 — Fourier terms

Fourier terms approximate a continuous periodic signal with a handful of sine
and cosine columns, which makes them the practical way to model **multiple** or
**long** seasonal periods — a weekly series has period 365.25 / 7 ≈ 52.179, which
no dummy encoding handles cleanly.
[Chapter 12.1: Complex Seasonality](https://otexts.com/fpp3/complexseasonality.html)
is a good practical introduction.

For each seasonal period `sp` and each order `k = 1, …, K`, `add_fourier_terms`
appends two columns, `sin_{sp}_{k}` and `cos_{sp}_{k}`. With `sp=12` and `K=3`
you get `sin_12_1`, `cos_12_1`, `sin_12_2`, `cos_12_2`, `sin_12_3`, `cos_12_3`
alongside the original columns.

```python
from panelary.offsets import freq_to_sp
from panelary.seasonality import add_fourier_terms

sp = freq_to_sp("1mo")[0]                              # 12
X_new = X.pipe(add_fourier_terms(sp=sp, K=3)).collect()
```

`K` must not exceed `sp`; a larger `K` raises `ValueError`.

## Modelling holidays and special events

`add_holiday_effects` wraps the [`holidays`](https://pypi.org/project/holidays/)
package to produce one categorical column per country, named
`holiday__<CODE>`. Dates with no holiday are null.

```python
from panelary.seasonality import add_holiday_effects

# Two categorical columns: "holiday__US" and "holiday__CA"
north_america = add_holiday_effects(country_codes=["US", "CA"])
X_new = X.pipe(north_america).collect()

# One-hot encoded instead (e.g. "holiday__US_christmas")
north_america = add_holiday_effects(country_codes=["US", "CA"], as_dummies=True)
X_new = X.pipe(north_america).collect()
```

!!! tip "Custom events"
    For your own special events — a promotion, a product launch, a maintenance
    window — build the
    [dummy variables](https://otexts.com/fpp3/useful-predictors.html#dummy-variables)
    yourself as a Polars
    [boolean expression](https://docs.pola.rs/user-guide/expressions/casting/)
    and join them on the time column.

## Forecasting into the future

Exogenous seasonal features have to exist for the *forecast* horizon too, not
just the training window. `make_future_calendar_effects` and
`make_future_holiday_effects` build them from a panel's index: they take the
last timestamp of each entity, extend it `fh` steps at frequency `freq`, and
apply the corresponding transformer to that future index.

```python
from panelary.seasonality import make_future_calendar_effects

X_future = make_future_calendar_effects(
    idx=y_train.select(entity_col, time_col), attrs=["month"], fh=3, freq="1mo",
)
```

## See also

- [Preprocessing](preprocessing.md) — the other curried panel transformers.
- [Forecasting](forecasting.md) — passing these columns as exogenous regressors.
