# Honest validation & selection

`panelary.validation` is the layer that makes "leak-safe" **measurable**. Panelary's other
pillars produce features, factors and attributions; this one produces *evidence*. After a
search over many candidate models or signals, it answers the only question that matters: how
much of the measured performance is real, and how much is an artefact of the search?

Everything here operates on already-computed backtest output — returns, forecast errors, fold
scores — so nothing fits a model on your data. Its leak-safety contribution is upstream: the
splitters below are what make the backtest honest in the first place, by purging training
observations whose label spans overlap a test fold and embargoing the rows immediately after
it.

## What's here

| Question | Entry point |
| --- | --- |
| Split a panel so overlapping labels cannot leak across the boundary | `PurgedKFold`, `walk_forward_splits`, `purged_calibration_split` |
| Turn one backtest trajectory into a *distribution* of paths | `CombinatorialPurgedCV`, `cpcv_splits`, `cpcv_backtest_paths` |
| Is this Sharpe ratio real after N trials? | `deflated_sharpe_ratio`, `probabilistic_sharpe_ratio`, `expected_maximum_sharpe` |
| How likely is this backtest to be overfit? | `probability_of_backtest_overfitting` |
| Control error rates across many simultaneous tests | `romano_wolf`, `holm_bonferroni`, `benjamini_hochberg`, `benjamini_yekutieli` |
| Is model A significantly better than model B? | `diebold_mariano`, `superior_predictive_ability`, `model_confidence_set` |
| Score a *distributional* forecast properly | `crps_ensemble`, `pinball_loss`, `interval_score`, `pit_values` |
| Resample a dependent series without destroying its dependence | `moving_block_bootstrap`, `stationary_bootstrap`, `wild_bootstrap`, `sieve_bootstrap` |
| Prediction intervals with finite-sample coverage | `split_conformal_interval`, `adaptive_conformal_intervals`, `enbpi` |

## Why the corrections matter

A backtest is a *search*. Try enough configurations and the best in-sample Sharpe ratio is
large even when every candidate is worthless, because you are reading the maximum of a
sampling distribution rather than a mean. The Deflated Sharpe Ratio corrects the threshold for
the number of trials and the higher moments of the returns; the Probability of Backtest
Overfitting estimates, by combinatorial recombination of the folds, how often the in-sample
winner underperforms the median out of sample.

For a *panel*, the multiple-testing correction has to survive arbitrary cross-sectional
dependence — entities move together, so tests are not independent. That is why
`benjamini_yekutieli` is here alongside `benjamini_hochberg`: BY is the variant valid under
arbitrary dependence, at the cost of a more conservative threshold.

## See also

- [Honest Validation guide](../user-guide/honest-validation.md) — the narrative walkthrough.
- [Validation guide](../user-guide/validation.md) — choosing and configuring splits.
- [`cross_validation`](cross-validation.md) — the simple chronological splitters.

## API

### Package surface

::: panelary.validation
    options:
      members:
        - CVReport
        - CombinatorialPurgedCV
        - PurgedKFold
        - cross_validate
        - expanding_window_split
        - sliding_window_split

### Splitters & backtest paths

::: panelary.validation._cv
    options:
      members:
        - IndexSplit
        - cpcv_splits
        - walk_forward_splits
        - purged_calibration_split
        - fold_boundaries
        - cpcv_backtest_paths
        - walk_forward_backtest_path

### Selection statistics

::: panelary.validation._selection_stats
    options:
      members:
        - MultipleTestResult
        - probabilistic_sharpe_ratio
        - expected_maximum_sharpe
        - deflated_sharpe_ratio
        - minimum_track_record_length
        - probability_of_backtest_overfitting
        - romano_wolf
        - romano_wolf_mean_test
        - holm_bonferroni
        - benjamini_hochberg
        - benjamini_yekutieli

### Forecast comparison & scoring rules

::: panelary.validation._forecast_tests
    options:
      members:
        - DieboldMarianoResult
        - SPAResult
        - MCSResult
        - newey_west_variance
        - diebold_mariano
        - superior_predictive_ability
        - model_confidence_set
        - crps_ensemble
        - crps_gaussian
        - crps_from_quantiles
        - pinball_loss
        - pinball_loss_expr
        - interval_score
        - pit_values
        - pit_histogram
        - score_quantile_forecasts

### Bootstrap resampling

::: panelary.validation._bootstrap
    options:
      members:
        - resolve_segments
        - block_bootstrap_indices
        - moving_block_bootstrap
        - circular_block_bootstrap
        - stationary_bootstrap
        - wild_bootstrap
        - sieve_bootstrap

### Conformal prediction for time series

::: panelary.conformal
    options:
      members:
        - conformal_calibration_split
        - conformal_quantile
        - split_conformal_interval
        - AdaptiveConformalResult
        - adaptive_conformal_intervals
        - conformal_pid_intervals
        - nexcp_quantile
        - nexcp_intervals
        - cqr_scores
        - conformalized_quantile_regression
        - enbpi
        - conformalize
