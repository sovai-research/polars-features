# Econometric feature generators

`panelary.econ.features` is a self-contained subpackage of **cheap, causal**
econometric feature builders. Every generator is either a pure trailing-window
statistic — row `t` is a function of rows `≤ t` within the same entity — or a
`PanelTransformer` that learns its parameters on the training rows and applies
them frozen. Nothing here is two-sided, and nothing fits on the rows it later
transforms.

Everything is pure NumPy + Polars: no `scipy`, no `statsmodels`, no Rust.

## What's here

| Block | Module | Public surface |
| --- | --- | --- |
| Unit roots / persistence | `_unitroot` | `adf`, `kpss`, `phillips_perron`, `dfgls`, `ng_perron`, `zivot_andrews`, `unit_root_table`, `rolling_unit_root_features`, `StationarityDifferencer` |
| Long memory | `_longmemory` | `gph`, `local_whittle`, `estimate_fractional_order`, `long_memory_table`, `rolling_long_memory_features`, `AutoFracDiff` |
| Realized volatility | `_harrv` | `realized_variance`, `bipower_variation`, `jump_component`, `realized_measures`, `har_terms`, `har_features`, `daily_realized_measures`, `HARModel` |
| Yield curve | `_nelson_siegel` | `nelson_siegel_loadings`, `nelson_siegel_fit`, `nelson_siegel_factors`, `NelsonSiegel` |
| Liquidity | `_liquidity` | `amihud_illiquidity`, `roll_spread`, `amivest_liquidity`, `liquidity_features` |
| Tails | `_evt` | `hill_index`, `gpd_fit`, `pot_var_es`, `evt_features` |
| Seasonality | `_decompose` | `causal_seasonal_decompose`, `seasonal_strength`, `decompose_features`, `CausalSeasonalDecomposer` |
| Shared numerics | `_common` | `ols`, `norm_cdf`, `norm_ppf`, `interp_pvalue`, `rolling_apply`, `rolling_beta`, `per_entity_apply`, `per_entity_reduce` |

## Two ways to use a test statistic

Every unit-root and long-memory estimator has **two** panel surfaces, and the
distinction is the whole leak-safety story:

- **`*_table(...)`** runs the estimator on each entity's *full* series and
  returns one row per entity. This is a **fitted, whole-sample** quantity — run
  it on training rows only, or let `StationarityDifferencer` / `AutoFracDiff`
  own it for you.
- **`rolling_*_features(...)`** runs the estimator on each row's **trailing
  window** and returns a per-row column. This is causal: appending future rows
  cannot change an earlier value, so it can be computed once over the whole
  panel and used inside any split.

## p-values

The reported p-values are **approximations**: published critical-value tables
(MacKinnon 2010 response surfaces for ADF/PP, KPSS 1992, ERS 1996, Ng-Perron
2001, Zivot-Andrews 1992) are interpolated in the normal-quantile domain. They
are exact at the tabulated points and monotone in the statistic everywhere else.
Each result object also carries the `crit_values` actually used, so a strict
comparison against a critical value is always available.

## Leak-safety declarations

| Transformer | `panel_safe` | `leakage_safe` | What it learns on `fit` |
| --- | --- | --- | --- |
| `StationarityDifferencer` | ✅ | ✅ | A differencing order per entity (ADF + KPSS) |
| `AutoFracDiff` | ✅ | ✅ | A fractional-differencing order `d` per entity |
| `HARModel` | ✅ | ✅ | One HAR-RV OLS per entity, plus pooled fallback coefficients |
| `NelsonSiegel` | ✅ | ✅ | The curve decay `λ` (optional grid search over train dates) |
| `CausalSeasonalDecomposer` | ✅ | ✅ | Nothing (a pure trailing-window transform) |

## Deliberate omissions

- **No two-sided STL.** A classical STL trend at time `t` averages observations
  on *both* sides of `t`, which rewrites history every time new data arrives.
  `causal_seasonal_decompose` ships a trailing-window seasonal-trend
  decomposition instead; it lags a centred trend by roughly half a window, and
  that lag is the price of causality.
- **PWM, not MLE, for the GPD.** `gpd_fit` uses probability-weighted moments:
  closed form, no optimiser, no `scipy`, and stable at rolling-window sample
  sizes. Its shape estimate is bounded above by 1 by construction, so for
  infinite-mean tails read `hill_index` instead.

## See also

- [Econometric Features guide](../user-guide/econ-features.md) — the narrative walkthrough.
- [`econ`](econ.md) — panel estimators and inference.
- [`detect`](detect.md) — explosive-regime and changepoint detection.

## API

::: panelary.econ.features
