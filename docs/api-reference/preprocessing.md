# Preprocessing

`panelary.preprocessing` is the transformer layer for reshaping and conditioning a panel
before modelling: reindexing and resampling, lagging and rolling, scaling and imputation,
differencing and detrending. Each entry point is a curried `@transformer` — call it with its
parameters to get a function of `frame -> frame`, so transforms compose and drop into a
`Pipeline` unchanged.

Leak-safety splits these into two groups, and the distinction is the whole story. **Pure
per-row transforms** (`lag`, `diff`, `roll`, `trim`, `resample`, `one_hot_encode`) are causal by
construction when applied `.over(entity_col)` in time order. **Fitted transforms** (`scale`,
`impute`, `boxcox`, `yeojohnson`, `detrend`, `deseasonalize_fourier`, `fractional_diff`) learn
state and must be fit on training rows only, then applied frozen; fitting them on a whole panel
is a look-ahead, and the module raises `LeakageWarning` where it can detect the mistake.

## What's here

| Task | Entry point |
| --- | --- |
| Regularise the time axis | `reindex`, `time_to_arange`, `resample`, `trim`, `coerce_dtypes` |
| Build causal regressors | `lag`, `roll`, `one_hot_encode` |
| Rescale | `scale`, `boxcox`, `yeojohnson` |
| Fill gaps | `impute`, `cafe_impute` |
| Remove trend and seasonality | `diff`, `detrend`, `deseasonalize_fourier`, `fractional_diff` |

## See also

- [Preprocessing guide](../user-guide/preprocessing.md) — the narrative walkthrough.
- [`imputation`](imputation.md) — the model-based, point-in-time `CafeImputer`.
- [Leak-safety](../concepts/leak-safety.md) — why fitted transforms must refit per fold.

## API

::: panelary.preprocessing
