"""The ``ts`` Polars expression namespace.

Importing this module registers ``pl.col(...).ts`` via
:func:`polars.api.register_expr_namespace`. Every method is a thin delegation to
a feature function in :mod:`._stats` (or an inline Polars expression), so the
namespace stays a presentation layer over the catalogue.
"""

from __future__ import annotations

import math

import polars as pl

from panelary._internal._type_aliases import DetrendMethod

from ._kernels import ClosedInterval, _cusum_events, _lempel_ziv_complexity_batch
from ._stats import (
    absolute_energy,
    absolute_maximum,
    absolute_sum_of_changes,
    autocorrelation,
    benford_correlation,
    binned_entropy,
    c3,
    change_quantiles,
    cid_ce,
    count_above,
    count_above_mean,
    count_below,
    count_below_mean,
    energy_ratios,
    first_location_of_maximum,
    first_location_of_minimum,
    harmonic_mean,
    has_duplicate,
    has_duplicate_max,
    has_duplicate_min,
    index_mass_quantile,
    large_standard_deviation,
    last_location_of_maximum,
    last_location_of_minimum,
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
    streak_length_stats,
    sum_reoccurring_points,
    sum_reoccurring_values,
    symmetry_looking,
    time_reversal_asymmetry_statistic,
    var_gt_std,
    variation_coefficient,
)


@pl.api.register_expr_namespace("ts")
class FeatureExtractor:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def absolute_energy(self) -> pl.Expr:
        """
        Compute the absolute energy of a time series.

        Returns
        -------
        An expression of the output
        """
        return absolute_energy(self._expr)

    def absolute_maximum(self) -> pl.Expr:
        """
        Compute the absolute maximum of a time series.

        Returns
        -------
        An expression of the output
        """
        return absolute_maximum(self._expr)

    def absolute_sum_of_changes(self) -> pl.Expr:
        """
        Compute the absolute sum of changes of a time series.

        Returns
        -------
        An expression of the output
        """
        return absolute_sum_of_changes(self._expr)

    def autocorrelation(self, n_lags: int) -> pl.Expr:
        """
        Calculate the autocorrelation for a specified lag. The autocorrelation measures the linear
        dependence between a time-series and a lagged version of itself.

        Parameters
        ----------
        n_lags : int
            The lag at which to calculate the autocorrelation. Must be a non-negative integer.

        Returns
        -------
        An expression of the output
        """
        return autocorrelation(self._expr, n_lags)

    def root_mean_square(self) -> pl.Expr:
        """
        Calculate the root mean square.

        Returns
        -------
        An expression of the output
        """
        return root_mean_square(self._expr)

    def benford_correlation(self) -> pl.Expr:
        """
        Returns the correlation between the first digit distribution of the input time series and
        the Newcomb-Benford's Law distribution.

        Returns
        -------
        An expression of the output
        """
        return benford_correlation(self._expr)

    def binned_entropy(self, bin_count: int = 10) -> pl.Expr:
        """
        Calculates the entropy of a binned histogram for a given time series. It is highly recommended
        that you impute the time series before calling this.

        Parameters
        ----------
        bin_count : int, optional
            The number of bins to use in the histogram. Default is 10.

        Returns
        -------
        An expression of the output
        """
        return binned_entropy(self._expr, bin_count)

    def c3(self, n_lags: int) -> pl.Expr:
        """
        Measure of non-linearity in the time series using c3 statistics.

        Parameters
        ----------
        n_lags : int
            The lag that should be used in the calculation of the feature.

        Returns
        -------
        An expression of the output
        """
        return c3(self._expr, n_lags)

    def change_quantiles(
        self, q_low: float, q_high: float, is_abs: bool = True
    ) -> pl.Expr:
        """
        First fixes a corridor given by the quantiles ql and qh of the distribution of x.
        Then calculates the average, absolute value of consecutive changes of the series x inside this corridor.

        Parameters
        ----------
        q_low : float
            The lower quantile of the corridor. Must be less than `q_high`.
        q_high : float
            The upper quantile of the corridor. Must be greater than `q_low`.
        is_abs : bool
            If True, takes absolute difference.

        Returns
        -------
        An expression of the output
        """
        return change_quantiles(self._expr, q_low, q_high, is_abs)

    def cid_ce(self, normalize: bool = False) -> pl.Expr:
        """
        Computes estimate of time-series complexity[^1].

        A more complex time series has more peaks and valleys.
        This feature is calculated by:

        Parameters
        ----------
        normalize : bool, optional
            If True, z-normalizes the time-series before computing the feature.
            Default is False.

        Returns
        -------
        An expression of the output
        """
        return cid_ce(self._expr, normalize)

    def count_above(self, threshold: float = 0.0) -> pl.Expr:
        """
        Calculate the percentage of values above or equal to a threshold.

        Parameters
        ----------
        threshold : float
            The threshold value for comparison.

        Returns
        -------
        An expression of the output
        """
        return count_above(self._expr, threshold)

    def count_above_mean(self) -> pl.Expr:
        """
        Count the number of values that are above the mean.

        Returns
        -------
        An expression of the output
        """
        return count_above_mean(self._expr)

    def count_below(self, threshold: float = 0.0) -> pl.Expr:
        """
        Calculate the percentage of values below or equal to a threshold.

        Parameters
        ----------
        threshold : float
            The threshold value for comparison.

        Returns
        -------
        An expression of the output
        """
        return count_below(self._expr, threshold)

    def count_below_mean(self) -> pl.Expr:
        """
        Count the number of values that are below the mean.

        Returns
        -------
        An expression of the output
        """
        return count_below_mean(self._expr)

    def energy_ratios(self, n_chunks: int = 10) -> pl.Expr:
        """
        Calculates sum of squares over the whole series for `n_chunks` equally segmented parts of the time-series.
        All ratios for all chunks will be returned at once.

        Parameters
        ----------
        n_chunks : int, optional
            The number of equally segmented parts to divide the time-series into. Default is 10.

        Returns
        -------
        An expression of the output
        """
        return energy_ratios(self._expr, n_chunks)

    def first_location_of_maximum(self) -> pl.Expr:
        """
        Returns the first location of the maximum value of x.
        The position is calculated relatively to the length of x.

        Returns
        -------
        An expression of the output
        """
        return first_location_of_maximum(self._expr)

    def first_location_of_minimum(self) -> pl.Expr:
        """
        Returns the first location of the minimum value of x.
        The position is calculated relatively to the length of x.

        Returns
        -------
        An expression of the output
        """
        return first_location_of_minimum(self._expr)

    def has_duplicate(self) -> pl.Expr:
        """
        Check if the time-series contains any duplicate values.

        Returns
        -------
        An expression of the output
        """
        return has_duplicate(self._expr)

    def has_duplicate_max(self) -> pl.Expr:
        """
        Check if the time-series contains any duplicate values equal to its maximum value.

        Returns
        -------
        An expression of the output
        """
        return has_duplicate_max(self._expr)

    def has_duplicate_min(self) -> pl.Expr:
        """
        Check if the time-series contains duplicate values equal to its minimum value.

        Returns
        -------
        An expression of the output
        """
        return has_duplicate_min(self._expr)

    def index_mass_quantile(self, q: float) -> pl.Expr:
        """
        Calculates the relative index i of time series x where q% of the mass of x lies left of i.
        For example for q = 50% this feature calculator will return the mass center of the time series.

        Parameters
        ----------
        q : float
            The quantile.

        Returns
        -------
        An expression of the output
        """
        return index_mass_quantile(self._expr, q)

    def large_standard_deviation(self, ratio: float = 0.25) -> pl.Expr:
        """
        Checks if the time-series has a large standard deviation: `std(x) > r * (max(X)-min(X))`.

        As a heuristic, the standard deviation should be a forth of the range of the values.

        Parameters
        ----------
        ratio : float
            The ratio of the interval to compare with.

        Returns
        -------
        An expression of the output
        """
        return large_standard_deviation(self._expr, ratio)

    def last_location_of_maximum(self) -> pl.Expr:
        """
        Returns the last location of the maximum value of x.
        The position is calculated relatively to the length of x.

        Returns
        -------
        An expression of the output
        """
        return last_location_of_maximum(self._expr)

    def last_location_of_minimum(self) -> pl.Expr:
        """
        Returns the last location of the minimum value of x.
        The position is calculated relatively to the length of x.

        Returns
        -------
        An expression of the output
        """
        return last_location_of_minimum(self._expr)

    def lempel_ziv_complexity(
        self, threshold: float | pl.Expr, as_ratio: bool = True
    ) -> pl.Expr:
        """
        Calculate a complexity estimate based on the Lempel-Ziv compression algorithm. The
        implementation here is a pure-Python transcription of Lilian Besson's code. Instead of returning
        the complexity value, we return a ratio w.r.t the length of the input series. If null is
        encountered, it will be interpreted as 0 in the bit sequence.

        Parameters
        ----------
        threshold: float | pl.Expr
            Either a number, or an expression representing a comparable quantity. If x > threshold,
            then it will be binarized as 1 and 0 otherwise.
        as_ratio: bool
            If true, return the complexity divided by length of sequence

        Returns
        -------
        Expr

        Reference
        ---------
        https://github.com/Naereen/Lempel-Ziv_Complexity/tree/master
        https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv_complexity
        """
        out = (self._expr > threshold).map_batches(
            _lempel_ziv_complexity_batch,
            return_dtype=pl.UInt32,
            returns_scalar=True,
        )
        if as_ratio:
            return out / self._expr.len()
        return out

    def linear_trend(self) -> pl.Expr:
        """
        Compute the slope, intercept, and RSS of the linear trend.

        Returns
        -------
        An expression of the output
        """
        return linear_trend(self._expr)

    def detrend(self, method: DetrendMethod = "linear") -> pl.Expr:
        """
        Detrends the time series by either removing a fitted linear regression or by
        removing the mean. This assumes that data is in order.

        Parameters
        ----------
        method
            Either `linear` or `mean`

        Returns
        -------
        An expression representing detrend-ed column
        """

        if method == "linear":
            N = self._expr.count()
            x = pl.int_range(0, N, dtype=pl.Float64, eager=False)
            coeff = pl.cov(self._expr, x) / x.var()
            const = self._expr.mean() - coeff * (N - 1) / 2
            return self._expr - x * coeff - const
        elif method == "mean":
            return self._expr - self._expr.mean()
        else:
            raise ValueError(f"Unknown detrend method: {method}")

    def longest_streak_above_mean(self) -> pl.Expr:
        """
        Returns the length of the longest consecutive subsequence in x that is greater than the mean of x.

        Returns
        -------
        An expression of the output
        """
        return longest_streak_above_mean(self._expr)

    def longest_streak_below_mean(self) -> pl.Expr:
        """
        Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x.

        Returns
        -------
        An expression of the output
        """
        return longest_streak_below_mean(self._expr)

    def longest_streak_above(self, threshold: float) -> pl.Expr:
        """
        Returns the longest streak of changes >= threshold of the time series. A change
        is counted when (x_t+1 - x_t) >= threshold. Note that the streaks here
        are about the changes for consecutive values in the time series, not the individual values.

        Parameters
        ----------
        threshold : float
            The threshold value for comparison.

        Returns
        -------
        An expression of the output
        """
        return longest_streak_above(self._expr, threshold)

    def longest_streak_below(self, threshold: float) -> pl.Expr:
        """
        Returns the longest streak of changes <= threshold of the time series. A change
        is counted when (x_t+1 - x_t) <= threshold. Note that the streaks here
        are about the changes for consecutive values in the time series, not the individual values.

        Parameters
        ----------
        threshold : float
            The threshold value for comparison.

        Returns
        -------
        An expression of the output
        """
        return longest_streak_below(self._expr, threshold)

    def mean_abs_change(self) -> pl.Expr:
        """
        Compute mean absolute change.

        Returns
        -------
        An expression of the output
        """
        return mean_abs_change(self._expr)

    def max_abs_change(self) -> pl.Expr:
        """
        Compute the maximum absolute change from X_t to X_t+1.

        Returns
        -------
        An expression of the output
        """
        return max_abs_change(self._expr)

    def mean_change(self) -> pl.Expr:
        """
        Compute mean change.

        Returns
        -------
        An expression of the output
        """
        return mean_change(self._expr)

    def mean_n_absolute_max(self, n_maxima: int) -> pl.Expr:
        """
        Calculates the arithmetic mean of the n absolute maximum values of the time series.

        Parameters
        ----------
        n_maxima : int
            The number of maxima to consider.

        Returns
        -------
        An expression of the output
        """
        return mean_n_absolute_max(self._expr, n_maxima)

    def mean_second_derivative_central(self) -> pl.Expr:
        """
        Returns the mean value of a central approximation of the second derivative.

        Returns
        -------
        An expression of the output
        """
        return mean_second_derivative_central(self._expr)

    def number_crossings(self, crossing_value: float = 0.0) -> pl.Expr:
        """
        Calculates the number of crossings of x on m, where m is the crossing value.

        A crossing is defined as two sequential values where the first value is lower than m and the next is greater,
        or vice-versa. If you set m to zero, you will get the number of zero crossings.

        Parameters
        ----------
        crossing_value : float
            The crossing value. Defaults to 0.0.

        Returns
        -------
        An expression of the output
        """
        return number_crossings(self._expr, crossing_value)

    def percent_reoccurring_points(self) -> pl.Expr:
        """
        Returns the percentage of non-unique data points in the time series. Non-unique data points are those that occur
        more than once in the time series.

        The percentage is calculated as follows:

            # of data points occurring more than once / # of all data points

        This means the ratio is normalized to the number of data points in the time series, in contrast to the
        `percent_reoccuring_values` function.

        Returns
        -------
        An expression of the output
        """
        return percent_reoccurring_points(self._expr)

    def percent_reoccurring_values(self) -> pl.Expr:
        """
        Returns the percentage of values that are present in the time series more than once.

        The percentage is calculated as follows:

            len(different values occurring more than once) / len(different values)

        This means the percentage is normalized to the number of unique values in the time series, in contrast to the
        `percent_reoccurring_points` function.

        Returns
        -------
        An expression of the output
        """
        return percent_reoccurring_values(self._expr)

    def number_peaks(self, support: int) -> pl.Expr:
        """
        Calculates the number of peaks of at least support n in the time series x. A peak of support n is defined as a
        subsequence of x where a value occurs, which is bigger than its n neighbours to the left and to the right.

        Hence in the sequence

        ``x = [3, 0, 0, 4, 0, 0, 13]``

        4 is a peak of support 1 and 2 because in the subsequences

        ``[0, 4, 0]``
        ``[0, 0, 4, 0, 0]``

        4 is still the highest value. Here, 4 is not a peak of support 3 because 13 is the 3th neighbour to the right of 4
        and its bigger than 4.

        Parameters
        ----------
        support : int
            Support of the peak

        Returns
        -------
        An expression of the output
        """
        return number_peaks(self._expr, support)

    def permutation_entropy(
        self,
        tau: int = 1,
        n_dims: int = 3,
        base: float = math.e,
    ) -> pl.Expr:
        """
        Computes permutation entropy. It is recommended that users should impute the time series
        before calling this.

        Parameters
        ----------
        tau : int
            The embedding time delay which controls the number of time periods between elements
            of each of the new column vectors. The recommended value is 1.
        n_dims : int, > 1
            The embedding dimension which controls the length of each of the new column vectors. The
            recommended range is 3-7.
        base : float
            The base for log in the entropy computation

        Returns
        -------
        An expression of the output
        """
        return permutation_entropy(self._expr, tau, n_dims, base)

    def range_count(
        self, lower: float, upper: float, closed: ClosedInterval = "left"
    ) -> pl.Expr:
        """
        Computes values of input expression that is between lower (inclusive) and upper (exclusive).

        Parameters
        ----------
        lower : float
            The lower bound, inclusive
        upper : float
            The upper bound, exclusive
        closed : ClosedInterval
            Whether or not the boundaries should be included/excluded

        Returns
        -------
        An expression of the output
        """
        return range_count(self._expr, lower, upper, closed)

    def ratio_beyond_r_sigma(self, ratio: float = 0.25) -> pl.Expr:
        """
        Returns the ratio of values in the series that is beyond r*std from mean on both sides.

        Parameters
        ----------
        ratio : float
            The scaling factor for std

        Returns
        -------
        An expression of the output
        """
        return ratio_beyond_r_sigma(self._expr, ratio)

    # Originally named: `sum_of_reoccurring_data_points`
    def sum_reoccurring_points(self) -> pl.Expr:
        """
        Returns the sum of all data points that are present in the time series more than once.

        For example, `sum_reoccurring_points(pl.Series([2, 2, 2, 2, 1]))` returns 8, as 2 is a reoccurring value, so all 2's
        are summed up.

        This is in contrast to the `sum_reoccurring_values` function, where each reoccuring value is only counted once.

        Returns
        -------
        An expression of the output
        """
        return sum_reoccurring_points(self._expr)

    # Originally named: `sum_of_reoccurring_values`
    def sum_reoccurring_values(self) -> pl.Expr:
        """
        Returns the sum of all values that are present in the time series more than once.

        For example, `sum_reoccurring_values(pl.Series([2, 2, 2, 2, 1]))` returns 2, as 2 is a reoccurring value, so it is
        summed up with all other reoccuring values (there is none), so the result is 2.

        This is in contrast to the `sum_reoccurring_points` function, where each reoccuring value is only counted as often
        as it is present in the data.

        Returns
        -------
        An expression of the output
        """
        return sum_reoccurring_values(self._expr)

    def symmetry_looking(self, ratio: float = 0.25) -> pl.Expr:
        """
        Check if the distribution of x looks symmetric.

        A distribution is considered symmetric if: `| mean(X)-median(X) | < ratio * (max(X)-min(X))`

        Parameters
        ----------
        ratio : float
            Multiplier on distance between max and min.

        Returns
        -------
        An expression of the output
        """
        return symmetry_looking(self._expr, ratio)

    def time_reversal_asymmetry_statistic(self, n_lags: int) -> pl.Expr:
        """
        Returns the time reversal asymmetry statistic.

        Parameters
        ----------
        n_lags : int
            The lag that should be used in the calculation of the feature.

        Returns
        -------
        An expression of the output
        """
        return time_reversal_asymmetry_statistic(self._expr, n_lags)

    def variation_coefficient(self) -> pl.Expr:
        """
        Calculate the coefficient of variation (CV).

        Returns
        -------
        An expression of the output
        """
        return variation_coefficient(self._expr)

    def var_gt_std(self, ddof: int = 1) -> pl.Expr:
        """
        Is the variance >= std? In other words, is var >= 1?

        Parameters
        ----------
        ddof : int
            Delta Degrees of Freedom used when computing var/std.

        Returns
        -------
        An expression of the output
        """
        return var_gt_std(self._expr, ddof)

    def harmonic_mean(self) -> pl.Expr:
        """
        Returns the harmonic mean of the expression

        Returns
        -------
        An expression of the output
        """
        return harmonic_mean(self._expr)

    def range_over_mean(self) -> pl.Expr:
        """
        Returns the range (max - min) over mean of the time series.

        Returns
        -------
        An expression of the output
        """
        return range_over_mean(self._expr)

    def range_change(self, percentage: bool = True) -> pl.Expr:
        """
        Returns the range (max - min) over mean of the time series.

        Parameters
        ----------
        percentage : bool
            Compute the percentage if set to True

        Returns
        -------
        An expression of the output
        """
        return range_change(self._expr, percentage)

    def streak_length_stats(self, above: bool, threshold: float) -> pl.Expr:
        """
        Returns some statistics of the length of the streaks of the time series. Note that the streaks here
        are about the changes for consecutive values in the time series, not the individual values.

        The statistics include: min length, max length, average length, std of length,
        10-percentile length, median length, 90-percentile length, and mode of the length. If input is Series,
        a dictionary will be returned. If input is an expression, the expression will evaluate to a struct
        with the fields ordered by the statistics.

        Parameters
        ----------
        above: bool
            Above (>=) or below (<=) the given threshold
        threshold
            The threshold for the change (x_t+1 - x_t) to be counted

        Returns
        -------
        An expression of the output
        """
        return streak_length_stats(self._expr, above, threshold)

    def longest_winning_streak(self) -> pl.Expr:
        """
        Returns the longest winning streak of the time series. A win is counted when
        (x_t+1 - x_t) >= 0

        Returns
        -------
        An expression of the output
        """
        return longest_winning_streak(self._expr)

    def longest_losing_streak(self) -> pl.Expr:
        """
        Returns the longest losing streak of the time series. A loss is counted when
        (x_t+1 - x_t) <= 0

        Returns
        -------
        An expression of the output
        """
        return longest_losing_streak(self._expr)

    def ratio_n_unique_to_length(self) -> pl.Expr:
        """
        Calculate the ratio of the number of unique values to the length of the time-series.

        Returns
        -------
        An expression of the output
        """
        return ratio_n_unique_to_length(self._expr)

    def cusum(
        self,
        threshold: float,
        warmup_period: int,
        drift: float = 0.0,
    ) -> pl.Expr:
        """
        Cumulative sum (CUSUM) filter to detect abrupt changes in data.

        The CUSUM filter is a quality control method, designed to detect a shift in the
        mean value of the measured quantity away from a target value.

        The general formula for the CUSUM filter can be found here:
        https://en.wikipedia.org/wiki/CUSUM

        And the original paper that introduces it can be found here:
        https://www.tandfonline.com/doi/abs/10.1080/00401706.1961.10489922

        Parameters
        ----------
        threshold : float
            The threshold for the change (x_t+1 - x_t) to be counted
        warmup_period : int
            The number of observations which are used to estimate the mean and standard
            deviation of the data.
        drift : float
            The drift coefficient for the CUSUM filter. Default value is 0.

        Returns
        -------
        An expression of the output
        """

        def _batch(s: pl.Series) -> pl.Series:
            events = _cusum_events(
                s.cast(pl.Float64).to_numpy(),
                threshold,
                warmup_period,
                drift,
            )
            return pl.Series(events, dtype=pl.Int32)

        return self._expr.map_batches(_batch, return_dtype=pl.Int32)

    def frac_diff(
        self,
        d: float,
        min_weight: float | None = None,
        window_size: int | None = None,
    ) -> pl.Expr:
        """Compute the fractional differential of a time series.

        This particular functionality is referenced in Advances in Financial Machine
        Learning by Marcos Lopez de Prado (2018).

        For feature creation purposes, it is suggested that the minimum value of d
        is used that removes stationarity from the time series. This can be achieved
        by running the augmented dickey-fuller test on the time series for different
        values of d and selecting the minimum value that makes the time series
        stationary.

        Parameters
        ----------
        d : float
            The fractional order of the differencing operator.
        min_weight : float, optional
            The minimum weight to use for calculations (the weight-magnitude
            ``threshold``). If specified, the window size is computed from this
            value and ``window_size`` is not needed.
        window_size : int, optional
            The window size of the fractional differencing operator (a hard
            ``max_width`` cap on the kernel). If specified, ``min_weight`` is not
            needed and the canonical default threshold governs early truncation.

        Notes
        -----
        This is a thin, causal, entity-agnostic shim over the shared
        :func:`panelary._internal._ffd.frac_diff_expr` builder -- the single source
        of truth for fractional differencing across every Panelary surface
        (``.panel.frac_diff``, :class:`~panelary.transform.frac_diff.FracDiff`,
        and :func:`panelary.preprocessing.fractional_diff`). It returns a
        bare :class:`polars.Expr`; compose it with ``.over(entity)`` to apply per
        entity. The first ``width - 1`` rows are emitted as ``null`` (incomplete
        leading window) -- never zero-filled.
        """
        from panelary._internal._ffd import frac_diff_expr

        # Assert only one of min_weight or window_size is specified
        if min_weight is not None and window_size is not None:
            raise ValueError("Only one of min_weight or window_size can be specified.")

        # Assert either min_weight or window_size is specified
        if min_weight is None and window_size is None:
            raise ValueError("Either min_weight or window_size must be specified.")

        if min_weight is not None:
            return frac_diff_expr(self._expr, d=d, threshold=min_weight)
        return frac_diff_expr(self._expr, d=d, max_width=window_size)

    def max_drawdown(self) -> pl.Expr:
        """
        Compute the maximum drawdown of the time series.

        Returns
        -------
        An expression of the output
        """
        return (self._expr / self._expr.cum_max()).min() - 1

    def num_direction_changes(self) -> pl.Expr:
        """
        Calculate the number of direction changes in the time series.

        Returns
        -------
        An expression of the output
        """
        return (self._expr.diff().sign().diff().abs() > 0).sum()

    def return_kurtosis(self) -> pl.Expr:
        """
        Compute the kurtosis of the return series.

        Returns
        -------
        An expression of the output
        """
        return self._expr.pct_change().kurtosis()

    def return_skew(self) -> pl.Expr:
        """
        Compute the skewness of the return series.

        Returns
        -------
        An expression of the output
        """
        return self._expr.pct_change().skew()

    def marginal_cost_of_immediacy(self) -> pl.Expr:
        """
        Compute Marginal Cost of Immediacy (MCI) as half of the quoted spread:

        MCI = (ask_price - bid_price) / 2

        Assumes the input expression is a Struct with fields:
            - 'bid_price'
            - 'ask_price'

        Returns
        -------
        Expr : expression producing MCI values
        """
        return (
            self._expr.struct.field("ask_price") - self._expr.struct.field("bid_price")
        ) / 2
