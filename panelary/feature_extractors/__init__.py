"""tsfresh-style time-series feature extraction, Polars-native and panel-safe.

This package is the public :mod:`panelary.feature_extractors` module. It was a
single 3.5k-line file until it was split into layers; the import path, every
public symbol and both import side effects are unchanged.

Importing it has two deliberate side effects:

1. it registers the ``ts`` Polars expression namespace
   (``pl.col("x").ts.absolute_energy()``), and
2. it registers one :class:`~panelary.registry.FeatureSpec` per scalar
   aggregation with the process-wide :mod:`panelary.registry`.

Layout
------
``_kernels``
    Vendored numeric kernels (``ricker``, Lempel-Ziv, CUSUM) and the shared
    ``*_EXPR`` type aliases. Depends on nothing else here.
``_stats``
    The general-purpose feature functions.
``_finance``
    Finance-flavoured summaries (realized volatility, drawdown, MCI).
``_namespace``
    The ``@pl.api.register_expr_namespace("ts")`` class delegating to the above.
``_catalogue``
    The scalar-aggregation catalogue, its registry registrations, and the bulk
    :func:`extract_features` entry point.
"""

from __future__ import annotations

# --- side effect: registers every `ts` scalar aggregation with `registry` -
from ._catalogue import (
    _TS_SCALAR_AGG_SPECS,
    _TS_SCALAR_AGGS,
    _ts_scalar_agg_builder,
    extract_features,
)

# --- finance-flavoured feature functions ----------------------------------
from ._finance import (
    max_drawdown,
    num_direction_changes,
    realized_volatility,
    return_kurtosis,
    return_skew,
    signed_mci,
)

# --- vendored kernels + shared type aliases -------------------------------
from ._kernels import (
    BOOL_EXPR,
    FLOAT_EXPR,
    FLOAT_INT_EXPR,
    INT_EXPR,
    LIST_EXPR,
    MAP_EXPR,
    MAP_LIST_EXPR,
    TIME_SERIES_T,
    _cusum_events,
    _cusum_events_py,
    _get_cusum_numba,
    _lempel_ziv_complexity_batch,
    _lempel_ziv_complexity_count,
    ricker,
)

# --- side effect: registers the `ts` Polars expression namespace ----------
from ._namespace import FeatureExtractor

# --- general-statistics feature functions ---------------------------------
from ._stats import (
    _BENFORD_DIST_SERIES,
    ApEn,
    _chebyshev_counter,
    _into_sequential_chunks,
    absolute_energy,
    absolute_maximum,
    absolute_sum_of_changes,
    approximate_entropy,
    augmented_dickey_fuller,
    autocorrelation,
    autoregressive_coefficients,
    benford_correlation,
    binned_entropy,
    c3,
    change_quantiles,
    cid_ce,
    count_above,
    count_above_mean,
    count_below,
    count_below_mean,
    cwt_coefficients,
    energy_ratios,
    fft_coefficients,
    first_location_of_maximum,
    first_location_of_minimum,
    fourier_entropy,
    friedrich_coefficients,
    harmonic_mean,
    has_duplicate,
    has_duplicate_max,
    has_duplicate_min,
    index_mass_quantile,
    large_standard_deviation,
    last_location_of_maximum,
    last_location_of_minimum,
    lempel_ziv_complexity,
    linear_trend,
    longest_losing_streak,
    longest_streak_above,
    longest_streak_above_mean,
    longest_streak_below,
    longest_streak_below_mean,
    longest_winning_streak,
    max_abs_change,
    mean_abs_change,
    mean_change,
    mean_n_absolute_max,
    mean_second_derivative_central,
    number_crossings,
    number_cwt_peaks,
    number_peaks,
    percent_reoccurring_points,
    percent_reoccurring_values,
    permutation_entropy,
    range_change,
    range_count,
    range_over_mean,
    ratio_beyond_r_sigma,
    ratio_n_unique_to_length,
    root_mean_square,
    sample_entropy,
    spkt_welch_density,
    streak_length_stats,
    sum_reoccurring_points,
    sum_reoccurring_values,
    symmetry_looking,
    time_reversal_asymmetry_statistic,
    var_gt_std,
    variation_coefficient,
)

__all__ = [
    "ApEn",
    "BOOL_EXPR",
    "FLOAT_EXPR",
    "FLOAT_INT_EXPR",
    "FeatureExtractor",
    "INT_EXPR",
    "LIST_EXPR",
    "MAP_EXPR",
    "MAP_LIST_EXPR",
    "TIME_SERIES_T",
    "absolute_energy",
    "absolute_maximum",
    "absolute_sum_of_changes",
    "approximate_entropy",
    "augmented_dickey_fuller",
    "autocorrelation",
    "autoregressive_coefficients",
    "benford_correlation",
    "binned_entropy",
    "c3",
    "change_quantiles",
    "cid_ce",
    "count_above",
    "count_above_mean",
    "count_below",
    "count_below_mean",
    "cwt_coefficients",
    "energy_ratios",
    "extract_features",
    "fft_coefficients",
    "first_location_of_maximum",
    "first_location_of_minimum",
    "fourier_entropy",
    "friedrich_coefficients",
    "harmonic_mean",
    "has_duplicate",
    "has_duplicate_max",
    "has_duplicate_min",
    "index_mass_quantile",
    "large_standard_deviation",
    "last_location_of_maximum",
    "last_location_of_minimum",
    "lempel_ziv_complexity",
    "linear_trend",
    "longest_losing_streak",
    "longest_streak_above",
    "longest_streak_above_mean",
    "longest_streak_below",
    "longest_streak_below_mean",
    "longest_winning_streak",
    "max_abs_change",
    "max_drawdown",
    "mean_abs_change",
    "mean_change",
    "mean_n_absolute_max",
    "mean_second_derivative_central",
    "num_direction_changes",
    "number_crossings",
    "number_cwt_peaks",
    "number_peaks",
    "percent_reoccurring_points",
    "percent_reoccurring_values",
    "permutation_entropy",
    "range_change",
    "range_count",
    "range_over_mean",
    "ratio_beyond_r_sigma",
    "ratio_n_unique_to_length",
    "realized_volatility",
    "return_kurtosis",
    "return_skew",
    "ricker",
    "root_mean_square",
    "sample_entropy",
    "signed_mci",
    "spkt_welch_density",
    "streak_length_stats",
    "sum_reoccurring_points",
    "sum_reoccurring_values",
    "symmetry_looking",
    "time_reversal_asymmetry_statistic",
    "var_gt_std",
    "variation_coefficient",
]
