# Preprocessing

`panelary.preprocessing` is a family of **curried, lazy, panel-aware**
transformers. Each one takes a long-format panel — the first two columns are the
entity and time keys, the rest are features — and transforms every entity's
series independently, as one parallelised `group_by` over the panel.

The usual reasons to reach for them: stabilise a series (`boxcox`,
`yeojohnson`), make it [stationary](https://otexts.com/fpp3/stationarity.html)
(`diff`, `detrend`, `fractional_diff`), or derive rolling features (`roll`).
Several are **invertible** — `diff`, `detrend`, `scale` and the power transforms
all expose `.invert(...)`, which is what lets a forecaster transform its target,
predict, and map the forecast back to the original scale.

```python
from panelary.preprocessing import diff, detrend, scale, boxcox, roll, impute
```

!!! info "Curried and lazy"
    Every transformer is a *curried* function: calling it with its parameters
    returns a transformer object, which you then apply with `X.pipe(...)`. The
    result is a `polars.LazyFrame`, so nothing is computed until you call
    `.collect()`. Chain as many `.pipe(...)` calls as you like and Polars
    optimises the whole graph at once.

See the [preprocessing API reference](../api-reference/preprocessing.md) for the
full parameter list of each transformer.

## Differencing

Apply `order`-th differences within each entity. Invertible.

```python
from panelary.preprocessing import diff

transformer = diff(order=1)
X_new = X.pipe(transformer).collect()
X_original = transformer.invert(X_new)
```

### Seasonal differencing

The same, shifted by `sp` periods. Invertible.

```python
from panelary.preprocessing import diff

# X is a monthly panel with seasonal period 12
transformer = diff(order=1, sp=12)
X_new = X.pipe(transformer).collect()
X_original = transformer.invert(X_new)
```

## Detrending

`detrend` removes a trend from every entity's series. `freq` is the panel's
Polars offset alias and is **required**; `method` chooses what is removed.
Invertible.

```python
from panelary.preprocessing import detrend

# Linear: subtract each entity's OLS line of best fit.
linear = detrend(freq="1mo", method="linear")
X_new = X.pipe(linear).collect()
X_original = linear.invert(X_new)

# Mean: subtract each entity's mean level.
mean = detrend(freq="1mo", method="mean")
X_new = X.pipe(mean).collect()
```

## Power transforms

Both fit their parameter per entity and are invertible.

```python
from panelary.preprocessing import boxcox, yeojohnson

# Box-Cox (positive data only); `method` picks how lambda is estimated.
transformer = boxcox(method="mle")      # or "pearsonr"
X_new = X.pipe(transformer).collect()
X_original = transformer.invert(X_new)

# Yeo-Johnson handles zero and negative values.
transformer = yeojohnson()
X_new = X.pipe(transformer).collect()
X_original = transformer.invert(X_new)
```

## Local scaling

Standardise each entity's series by subtracting its mean and dividing by its
standard deviation. Invertible.

```python
from panelary.preprocessing import scale

transformer = scale(use_mean=True, use_std=True)
X_new = X.pipe(transformer).collect()
X_original = transformer.invert(X_new)
```

## Rolling statistics

Given a list of window sizes, `roll` computes rolling statistics for every
numeric column of every entity. **Not** invertible. Supported statistics:
`mean`, `min`, `max`, `mlm` (max less min), `sum`, `std`, `cv` (coefficient of
variation).

```python
from panelary.preprocessing import roll

# Moving averages (MA10, MA30, MA60) and moving sums for a daily panel.
transformer = roll(
    window_sizes=[10, 30, 60],
    stats=["mean", "sum"],
    freq="1d",
)
X_new = X.pipe(transformer).collect()
```

## Imputation

`impute` fills missing values in the numeric columns, per entity. The method is
the first argument: `"mean"`, `"median"`, `"fill"` (mean for floats, median for
integers), `"ffill"`, `"bfill"`, `"interpolate"`, `"cafe"`, or a constant.

```python
from panelary.preprocessing import impute

X_new = X.pipe(impute(method="ffill")).collect()
X_new = X.pipe(impute(method=0.0)).collect()        # constant fill
```

!!! warning "`bfill` and `interpolate` read the future"
    Backward fill copies a *later* observation back over a gap, and linear
    interpolation blends the observations on *both* sides of one. Both leak
    look-ahead information and pass silently through cross-validation, so
    Panelary emits a `LeakageWarning` when you use them. Prefer
    `impute(method="cafe")` — the strictly point-in-time, model-based imputer
    described in the [Imputation guide](imputation.md) — or `"ffill"`. Pass
    `allow_leaky=True` to acknowledge the leak and silence the warning (for
    whole-sample exploration outside a backtest).

## See also

- [Imputation](imputation.md) — the leak-safe CAFE imputer and its by-products.
- [Feature Engineering](features.md) — the `.panel` / `.xs` / `.ts` operator
  namespaces, which are the leak-safe path for *feature* construction.
- [Preprocessing API reference](../api-reference/preprocessing.md).
