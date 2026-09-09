# `validation` — honest validation & selection

::: panelary.validation

## Splitters & backtest paths

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

## Selection statistics

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

## Forecast comparison & scoring rules

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

## Bootstrap resampling

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

## Conformal prediction for time series

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
