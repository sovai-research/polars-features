"""The 22 canonical catch22 feature functions (clean-room re-implementation).

Each function takes a 1-D array-like, z-scores it (mean 0, ``ddof=1``
standard deviation) and returns a single ``float``.  Written directly from
the algorithmic descriptions in Lubba et al. (2019); no code is vendored
from the GPL-licensed ``pycatch22`` / ``hctsa`` sources.
"""

from __future__ import annotations

import numpy as np

from panelary._internal import _numpy_stats

from ._helpers import (
    _acf,
    _coarsegrain_quantile,
    _first_zero_ac,
    _histcounts,
    _line_sse,
    _local_simple_residuals,
    _longest_run,
    _lsq_spline_fit,
    _num_bins_auto,
    _pair_counts,
    _zscore,
)


# ---------------------------------------------------------------------------
# Distribution features
# ---------------------------------------------------------------------------
def DN_HistogramMode_5(x) -> float:
    """Mode of the z-scored distribution estimated from a 5-bin histogram.

    Returns the centre of the most-populated bin; ties are resolved by
    averaging the centres of all bins sharing the maximum count.
    """
    return _histogram_mode(_zscore(x), 5)


def DN_HistogramMode_10(x) -> float:
    """Mode of the z-scored distribution estimated from a 10-bin histogram.

    See :func:`DN_HistogramMode_5`.
    """
    return _histogram_mode(_zscore(x), 10)


def _histogram_mode(y: np.ndarray, n_bins: int) -> float:
    if y.size == 0 or y.max() == y.min():
        return np.nan
    counts, edges = _histcounts(y, n_bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    max_count = counts.max()
    return float(centers[counts == max_count].mean())


def CO_trev_1_num(x) -> float:
    """Numerator of the time-reversibility statistic *trev* at lag 1.

    ``trev_num = mean((y[t+1] - y[t]) ** 3)`` on the z-scored series -- a
    measure of temporal (a)symmetry.
    """
    y = _zscore(x)
    if y.size < 2:
        return np.nan
    d = np.diff(y)
    return float(np.mean(d**3))


def MD_hrv_classic_pnn40(x) -> float:
    """pNN40: fraction of successive differences with ``|diff| > 0.04``.

    Adapted from the classic heart-rate-variability pNNx statistic.  catch22
    scales successive differences by 1000 and thresholds at 40, i.e. an
    absolute threshold of ``0.04`` on the z-scored series.
    """
    y = _zscore(x)
    if y.size < 2:
        return np.nan
    d = np.abs(np.diff(y))
    return float(np.mean(d > 0.04))


# ---------------------------------------------------------------------------
# Autocorrelation-based features
# ---------------------------------------------------------------------------
def CO_f1ecac(x) -> float:
    """First ``1/e`` crossing of the autocorrelation function.

    The (linearly interpolated) lag at which the ACF first drops below
    ``1/e``.  Falls back to the series length if the ACF never crosses.
    """
    y = _zscore(x)
    acf = _acf(y)
    n = acf.size
    thresh = 1.0 / np.e
    below = np.flatnonzero(acf[1:] < thresh)
    if below.size == 0:
        return float(n)
    i = int(below[0])
    slope = acf[i + 1] - acf[i]
    if slope == 0:
        return float(i)
    return float(i + (thresh - acf[i]) / slope)


def CO_FirstMin_ac(x) -> float:
    """Lag of the first local minimum of the autocorrelation function."""
    y = _zscore(x)
    acf = _acf(y)
    n = acf.size
    if n < 3:
        return float(n)
    mid = acf[1:-1]
    minima = np.flatnonzero((mid < acf[:-2]) & (mid < acf[2:]))
    return float(int(minima[0]) + 1) if minima.size else float(n)


def CO_HistogramAMI_even_2_5(x) -> float:
    """Automutual information at lag 2 using 5 equal-width bins.

    The AMI ``sum_ij p_ij * log(p_ij / (p_i p_j))`` (natural log) between the
    series and its lag-2 copy, with the joint distribution estimated from a
    5x5 equal-width histogram spanning ``[min-0.1, max+0.1]``.
    """
    y = _zscore(x)
    tau, n_bins = 2, 5
    if y.size <= tau + 1:
        return np.nan
    y1, y2 = y[:-tau], y[tau:]
    fmin, fmax = y.min(), y.max()
    step = (fmax - fmin + 0.2) / n_bins
    if step <= 0:
        return np.nan
    edges = fmin - 0.1 + np.arange(n_bins + 1) * step
    joint, _, _ = np.histogram2d(y1, y2, bins=[edges, edges])
    p = joint / joint.sum()
    pi = p.sum(axis=1, keepdims=True)
    pj = p.sum(axis=0, keepdims=True)
    denom = pi * pj
    mask = (p > 0) & (denom > 0)
    return float(np.sum(p[mask] * np.log(p[mask] / denom[mask])))


def IN_AutoMutualInfoStats_40_gaussian_fmmi(x) -> float:
    """First minimum of the Gaussian automutual information over lags 1..40.

    Under a Gaussian assumption the AMI at lag ``k`` is
    ``-0.5 * log(1 - r_k**2)`` where ``r_k`` is the Pearson correlation between
    the series and its lag-``k`` copy.  Returns the (0-based) index into the
    lag array of the first local minimum, or the number of lags if none.
    """
    y = _zscore(x)
    n = y.size
    tau_max = min(40, int(np.ceil(n / 2)))
    if tau_max < 2:
        return np.nan
    ami = np.empty(tau_max)
    for k in range(1, tau_max + 1):
        a, b = y[:-k], y[k:]
        if a.size < 2 or a.std() == 0 or b.std() == 0:
            ami[k - 1] = 0.0
            continue
        r = np.corrcoef(a, b)[0, 1]
        r = min(max(r, -0.9999999), 0.9999999)
        ami[k - 1] = -0.5 * np.log(1.0 - r * r)
    if tau_max < 3:
        return float(tau_max)
    mid = ami[1:-1]
    minima = np.flatnonzero((mid < ami[:-2]) & (mid < ami[2:]))
    return float(int(minima[0]) + 1) if minima.size else float(tau_max)


def CO_Embed2_Dist_tau_d_expfit_meandiff(x) -> float:
    """Goodness of an exponential fit to successive distances in a 2-D embedding.

    The series is embedded in 2-D with delay ``tau`` (the first ACF zero-
    crossing, capped at ``n/10``).  The Euclidean distances between successive
    embedded points are histogrammed (Scott's rule) and compared, bin by bin,
    to an exponential density with rate ``1/mean(distance)``; the feature is
    the mean absolute density difference.
    """
    y = _zscore(x)
    n = y.size
    tau = _first_zero_ac(y)
    if tau > n // 10:
        tau = n // 10
    if tau < 1 or n - tau - 1 < 2:
        return np.nan
    dx = y[1 : n - tau] - y[0 : n - tau - 1]
    dy = y[1 + tau : n] - y[tau : n - 1]
    d = np.sqrt(dx * dx + dy * dy)
    if d.size < 2:
        return np.nan
    lam = d.mean()
    if lam <= 0:
        return np.nan
    n_bins = _num_bins_auto(d)
    if n_bins <= 0:
        return 0.0
    counts, edges = _histcounts(d, n_bins)
    norm = counts / d.size
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    exp_pdf = np.exp(-centers / lam) / lam
    exp_pdf[exp_pdf < 0] = 0.0
    diff = np.abs(norm / widths - exp_pdf)
    return float(diff.mean())


# ---------------------------------------------------------------------------
# Binary / symbolic features
# ---------------------------------------------------------------------------
def SB_BinaryStats_mean_longstretch1(x) -> float:
    """Longest run of consecutive values above the mean (binarise by mean)."""
    y = _zscore(x)
    if y.size == 0:
        return np.nan
    return float(_longest_run(y > y.mean()))


def SB_BinaryStats_diff_longstretch0(x) -> float:
    """Longest run of consecutive non-increases (binarise the diff by 0).

    The successive-difference series is binarised (``1`` if ``> 0`` else ``0``)
    and the length of the longest run of ``0`` s (consecutive decreases /
    flats) is returned.
    """
    y = _zscore(x)
    if y.size < 2:
        return np.nan
    d = np.diff(y)
    return float(_longest_run(d <= 0))


def SB_MotifThree_quantile_hh(x) -> float:
    """Entropy of the length-2 motif distribution over a 3-letter alphabet.

    The series is symbolised into 3 equiprobable (quantile) letters; the 3x3
    matrix of consecutive letter pairs is normalised into a probability
    distribution whose (natural-log) Shannon entropy ``hh`` is returned.
    """
    y = _zscore(x)
    if y.size < 3:
        return np.nan
    s = _coarsegrain_quantile(y, 3)
    counts = _pair_counts(s, 3)
    p = counts / counts.sum()
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)))


def SB_TransitionMatrix_3ac_sumdiagcov(x) -> float:
    """Trace of the covariance of a 3-state transition matrix (ac downsampling).

    The series is downsampled by ``tau`` (the first ACF zero-crossing),
    symbolised into 3 equiprobable states, and its 3x3 transition-probability
    matrix formed.  The feature is the sum of the diagonal (trace) of the
    covariance matrix of that transition matrix's columns.
    """
    y = _zscore(x)
    tau = _first_zero_ac(y)
    if tau < 1:
        tau = 1
    yd = y[::tau]
    if yd.size < 4:
        return np.nan
    s = _coarsegrain_quantile(yd, 3)
    T = _pair_counts(s, 3)
    T /= yd.size - 1
    cov = np.cov(T, rowvar=False, ddof=1)
    return float(np.trace(cov))


# ---------------------------------------------------------------------------
# Periodicity / forecasting features
# ---------------------------------------------------------------------------
def PD_PeriodicityWang_th0_01(x) -> float:
    """Periodicity measure of Wang et al. with threshold ``0.01``.

    The series is spline-detrended, its ACF computed, and the lag of the first
    ACF peak whose height exceeds the preceding trough by more than ``0.01``
    (and is positive) is returned.

    NEEDS REVIEW: the reference uses a bespoke piecewise-cubic ``splinefit``;
    here a least-squares cubic spline with 3 interior knots is used, so exact
    numerical parity with ``pycatch22`` is not guaranteed.

    The spline is fitted by :func:`_lsq_spline_fit` (pure NumPy), so this
    feature returns the same value with and without SciPy installed.
    """
    y = _zscore(x)
    n = y.size
    if n < 8:
        return np.nan
    th = 0.01
    t = np.arange(n, dtype=float)
    knots = np.linspace(0, n - 1, 5)[1:-1]  # 3 interior knots
    try:
        y_spline = _lsq_spline_fit(t, y, knots, k=3)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
        y_spline = np.zeros(n)
    y_sub = y - y_spline

    ac_max = int(np.ceil(n / 3))
    acf = _acf(y_sub)[:ac_max]

    if acf.size < 3:
        return 0.0
    # Turning points from the sign pattern of consecutive first differences:
    # position i is a trough when slope_in < 0 < slope_out, a peak when
    # slope_in > 0 > slope_out. Vectorised replacement for the scalar scan.
    diffs = np.diff(acf)
    slope_in, slope_out = diffs[:-1], diffs[1:]
    troughs = np.flatnonzero((slope_in < 0) & (slope_out > 0)) + 1
    peaks = np.flatnonzero((slope_in > 0) & (slope_out < 0)) + 1
    if peaks.size == 0 or troughs.size == 0:
        return 0.0

    # For each peak, the last trough strictly before it (both index arrays are
    # already sorted ascending, so one searchsorted replaces the inner scan).
    prior = np.searchsorted(troughs, peaks, side="left") - 1
    has_prior = prior >= 0
    trough_of = troughs[np.where(has_prior, prior, 0)]
    accepted = has_prior & (acf[peaks] - acf[trough_of] >= th) & (acf[peaks] >= 0.0)
    hits = np.flatnonzero(accepted)
    return float(int(peaks[hits[0]])) if hits.size else 0.0


def FC_LocalSimple_mean1_tauresrat(x) -> float:
    """Ratio of first ACF-zero of the residuals to that of the series.

    "Local simple" mean forecasting with a training window of 1 point; the
    feature is ``firstzero_ac(residuals) / firstzero_ac(series)``.
    """
    y = _zscore(x)
    res = _local_simple_residuals(y, 1)
    if res.size < 2:
        return np.nan
    denom = _first_zero_ac(y)
    if denom == 0:
        return np.nan
    return float(_first_zero_ac(res) / denom)


def FC_LocalSimple_mean3_stderr(x) -> float:
    """Standard deviation of "local simple" mean-forecast residuals (window 3)."""
    y = _zscore(x)
    res = _local_simple_residuals(y, 3)
    if res.size < 2:
        return np.nan
    return float(res.std(ddof=1))


# ---------------------------------------------------------------------------
# Outlier-timing features
# ---------------------------------------------------------------------------
def _outlier_include(y: np.ndarray, sign: int) -> float:
    """Shared core of the ``DN_OutlierInclude`` features.

    NEEDS REVIEW: the trimming rule (which thresholds contribute to the final
    median) is a faithful best-effort re-derivation; exact parity with the
    reference C implementation is not guaranteed.
    """
    inc = 0.01
    n = y.size
    yw = sign * y
    max_val = yw.max()
    if max_val < inc:
        return 0.0
    n_thresh = int(max_val / inc) + 1
    thresholds = np.arange(n_thresh, dtype=np.float64) * inc

    # `high = flatnonzero(yw >= thresh)` for every threshold, without the scan:
    # the retained set at threshold t is exactly the `count(t)` largest values,
    # and ties are all-or-nothing because the test is on the value itself. So
    # sorting once gives every threshold's index set as a prefix of `order`.
    ascending = np.sort(yw)
    counts = n - np.searchsorted(ascending, thresholds, side="left")
    pct = 100.0 * counts / n

    # keep the thresholds where at least 2% of points remain
    valid = pct > 2.0
    if not valid.any():
        return np.nan

    # Only the *valid* thresholds contribute to the final median, and thresholds
    # that select the same number of points select the same index set, so the
    # O(n) `flatnonzero` scan runs once per distinct retained count instead of
    # once per threshold. Both prunings are exact.
    kept_thresholds = thresholds[valid]
    _, first_of_count, inverse = np.unique(
        counts[valid], return_index=True, return_inverse=True
    )
    medians = np.fromiter(
        (np.median(np.flatnonzero(yw >= kept_thresholds[j])) for j in first_of_count),
        dtype=np.float64,
        count=first_of_count.size,
    )
    # Expand back so every retained threshold still contributes its own entry to
    # the outer median (deduplicating there would reweight the distribution).
    med_pos = medians[inverse.ravel()] / (n / 2.0) - 1.0
    return float(np.nanmedian(med_pos))


def DN_OutlierInclude_p_001_mdrmd(x) -> float:
    """Median outlier timing as positive outliers are progressively included.

    Thresholds increase in steps of ``0.01``; at each level the centred median
    time-index of points above the threshold is recorded, and the median of
    those values (over the well-populated threshold range) is returned.
    """
    return _outlier_include(_zscore(x), sign=1)


def DN_OutlierInclude_n_001_mdrmd(x) -> float:
    """Negative-outlier counterpart of :func:`DN_OutlierInclude_p_001_mdrmd`."""
    return _outlier_include(_zscore(x), sign=-1)


# ---------------------------------------------------------------------------
# Power-spectrum features
# ---------------------------------------------------------------------------
def _welch_spectrum(y: np.ndarray):
    """Welch power spectrum with a rectangular window over the whole series.

    Returns ``(w, S)`` where ``w`` is the angular frequency in ``[0, pi]`` and
    ``S`` the power spectral density.
    """
    n = y.size
    f, pxx = _numpy_stats.welch(
        y,
        window="boxcar",
        nperseg=n,
        noverlap=0,
        detrend=False,
        return_onesided=True,
        scaling="density",
    )
    return 2.0 * np.pi * f, pxx


def _welch_cumulative(y: np.ndarray):
    if y.size < 4:
        return None
    w, s = _welch_spectrum(y)
    if w.size < 2:
        return None
    dw = w[1] - w[0]
    area = np.sum(s) * dw
    if area <= 0:
        return None
    s_norm = s / area
    cs = np.cumsum(s_norm) * dw
    return w, cs


def SP_Summaries_welch_rect_area_5_1(x) -> float:
    """Normalised spectral power in the first fifth of the frequency range.

    The rectangular-window Welch spectrum is normalised to unit area; the
    feature is the cumulative power up to one fifth of the frequency axis.
    """
    res = _welch_cumulative(_zscore(x))
    if res is None:
        return np.nan
    w, cs = res
    idx = w.size // 5
    idx = min(idx, cs.size - 1)
    return float(cs[idx])


def SP_Summaries_welch_rect_centroid(x) -> float:
    """Spectral centroid: angular frequency where cumulative power reaches 0.5."""
    res = _welch_cumulative(_zscore(x))
    if res is None:
        return np.nan
    w, cs = res
    idx = int(np.argmax(cs >= 0.5))
    return float(w[idx])


# ---------------------------------------------------------------------------
# Fluctuation-scaling (DFA family) features
# ---------------------------------------------------------------------------
def _fluctuation_analysis(x, how: str) -> float:
    """Shared core of the ``SC_FluctAnal`` features.

    Computes the fluctuation ``F(tau)`` of the integrated (profile) series over
    log-spaced window sizes ``tau`` -- root-mean-square of linearly-detrended
    residuals for ``dfa`` (order-2), or of the detrended range for
    ``rsrangefit`` -- then fits two straight lines to ``log F`` vs ``log tau``
    and returns the fraction of scales in the first (small-scale) regime at the
    best breakpoint.

    NEEDS REVIEW: the exact scale grid and breakpoint convention differ subtly
    between implementations; this is a documented best-effort re-derivation and
    may not match ``pycatch22`` to the last digit.  It returns a value in
    ``[0, 1]``.
    """
    y = _zscore(x)
    n = y.size
    if n < 20:
        return np.nan
    profile = np.cumsum(y - y.mean())

    tau_min, tau_max = 5, n // 2
    if tau_max <= tau_min + 1:
        return np.nan
    taus = np.unique(
        np.round(np.exp(np.linspace(np.log(tau_min), np.log(tau_max), 50))).astype(int)
    )
    taus = taus[(taus >= tau_min) & (taus <= tau_max)]

    fluct, used = [], []
    for tau in taus:
        n_win = n // tau
        if n_win < 1:
            continue
        t = np.arange(tau, dtype=float)
        # Batched linear detrend: every window of length ``tau`` shares the same
        # design matrix ``[t, 1]``, so all ``n_win`` OLS line fits are solved in a
        # single ``lstsq`` instead of calling ``np.polyfit`` once per window.
        seg_mat = profile[: n_win * tau].reshape(n_win, tau).T  # (tau, n_win)
        amat = np.vstack([t, np.ones_like(t)]).T  # (tau, 2)
        coef, *_ = np.linalg.lstsq(amat, seg_mat, rcond=None)  # (2, n_win)
        resid = seg_mat - amat @ coef  # (tau, n_win)
        if how == "dfa":
            sq = np.mean(resid**2, axis=0)
        else:  # rsrangefit: squared range of the detrended profile
            sq = (resid.max(axis=0) - resid.min(axis=0)) ** 2
        f = np.sqrt(np.mean(sq))
        if f > 0:
            fluct.append(f)
            used.append(tau)

    if len(fluct) < 5:
        return np.nan
    log_tau = np.log(np.asarray(used, dtype=float))
    log_f = np.log(np.asarray(fluct))
    ntt = log_f.size

    best_err, best_br = np.inf, ntt // 2
    # breakpoint shared between the two segments; >= 2 points each side
    for br in range(2, ntt - 1):
        err = _line_sse(log_tau[:br], log_f[:br]) + _line_sse(
            log_tau[br - 1 :], log_f[br - 1 :]
        )
        if err < best_err:
            best_err, best_br = err, br
    return float(best_br / ntt)


def SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(x) -> float:
    """DFA fluctuation-scaling: proportion of scales in the first regime.

    Detrended fluctuation analysis (order-2 RMS) variant.  See
    :func:`_fluctuation_analysis`.
    """
    return _fluctuation_analysis(x, "dfa")


def SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(x) -> float:
    """Rescaled-range fluctuation-scaling: proportion of scales in first regime.

    Rescaled-range (range-of-detrended-profile) variant.  See
    :func:`_fluctuation_analysis`.
    """
    return _fluctuation_analysis(x, "rsrangefit")
