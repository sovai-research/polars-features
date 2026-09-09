# Feature extraction

`panelary.feature_extractors` is the tsfresh-style feature library: ~70 Polars-native
time-series characteristics, each usable as a bare `polars.Expr` inside `.over(entity_col)`
or through the batch entry points `extract_features` / `FeatureExtractor`.

Every extractor is a pure function of the series it is handed. Used the standard way — one
`group_by(entity)` or one `.over(entity)` per entity, in time order — no value can see another
entity's rows, and windowed variants only look backwards. Features that summarise a *whole*
series (`linear_trend`, `augmented_dickey_fuller`, `max_drawdown`, …) are whole-sample
statistics by definition: compute them on training rows only, or wrap them in a trailing
window, before using them as per-row regressors.

## What's here

| Family | Examples |
| --- | --- |
| Distribution shape | `variation_coefficient`, `symmetry_looking`, `return_skew`, `return_kurtosis`, `large_standard_deviation` |
| Energy & magnitude | `absolute_energy`, `absolute_maximum`, `root_mean_square`, `energy_ratios`, `realized_volatility` |
| Change & trend | `mean_change`, `mean_abs_change`, `absolute_sum_of_changes`, `linear_trend`, `cid_ce` |
| Complexity & entropy | `approximate_entropy`, `sample_entropy`, `binned_entropy`, `permutation_entropy`, `fourier_entropy`, `lempel_ziv_complexity` |
| Autocorrelation & dynamics | `autocorrelation`, `autoregressive_coefficients`, `c3`, `time_reversal_asymmetry_statistic`, `friedrich_coefficients` |
| Spectral | `fft_coefficients`, `cwt_coefficients`, `spkt_welch_density`, `number_cwt_peaks` |
| Counting & location | `count_above_mean`, `number_peaks`, `number_crossings`, `first_location_of_maximum`, `index_mass_quantile` |
| Streaks & drawdown | `longest_streak_above_mean`, `longest_winning_streak`, `max_drawdown`, `streak_length_stats` |
| Stationarity | `augmented_dickey_fuller`, `change_quantiles`, `ratio_beyond_r_sigma` |
| Batch API | `extract_features`, `FeatureExtractor` |

`extract_features` compiles every requested feature into a **single lazy Polars plan**, so a
bulk request costs one pass over the frame rather than one pass per feature.

## See also

- [catch22](catch22.md) — the 22-feature canonical subset, z-scored and redundancy-pruned.
- [Feature Extraction (legacy)](../user-guide/feature-extraction.md) — the narrative guide.

## API

::: panelary.feature_extractors
