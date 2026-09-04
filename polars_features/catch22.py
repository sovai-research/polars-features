"""Clean-room, Polars-native implementation of the *catch22* feature set.

catch22 (Lubba et al., 2019, "catch22: CAnonical Time-series CHaracteristics",
*Data Mining and Knowledge Discovery* 33, 1821-1852) is a curated collection of
22 time-series features selected from the ~7700 features of the *hctsa* library
for high classification performance and low mutual redundancy.

This module is a **clean-room re-implementation** written directly from the
published algorithmic descriptions -- it does **not** vendor or copy any code
from the GPL-licensed ``pycatch22`` / ``hctsa`` sources.  Each of the 22
canonical features is implemented as a small, documented Python function that
operates on a 1-D :class:`numpy.ndarray`.

Following the standard "catch22" variant, every feature *z-scores* the input
series (mean 0, standard deviation 1, using the sample standard deviation with
``ddof=1``) before computing.  Z-scoring is idempotent, so the feature functions
are safe to call directly on raw data.

Two convenience entry points are provided:

* :func:`catch22_all` -- compute all 22 features (or 24 with ``catch24=True``,
  which appends the raw mean and standard deviation) for a single 1-D array,
  returning a ``dict[str, float]``.
* :func:`catch22_features` -- a Polars-friendly entry point that computes the
  features **per entity** on a (long-format) panel :class:`polars.DataFrame` /
  :class:`polars.LazyFrame`, returning one row per entity.  The per-entity
  ``group_by`` makes the computation leak-safe: every feature uses only the
  values of its own series.

Correctness notes
-----------------
A handful of the canonical features rely on fairly intricate sub-routines
(spline detrending, adaptive histogram binning, piecewise-linear fitting of
fluctuation scaling).  Those are implemented as faithful, documented
best-effort re-derivations and are flagged in their docstrings as
``NEEDS REVIEW`` for exact numerical parity with ``pycatch22``.  They are
algorithmically sound and return finite, correctly-signed values, but their
last-digit outputs may differ from the reference C implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from polars_features import _numpy_stats


def _get_lsq_spline():
    """Lazily import ``scipy.interpolate.LSQUnivariateSpline`` (optional extra).

    Kept out of module import so ``import polars_features.catch22`` does not pull
    in SciPy. Returns ``None`` when SciPy is not installed, and the single
    feature that uses it falls back to a zero spline (as before).
    """
    try:  # pragma: no cover - trivial import shim
        from scipy.interpolate import LSQUnivariateSpline
    except ImportError:  # pragma: no cover - scipy is an optional extra
        return None
    return LSQUnivariateSpline


__all__ = [
    "CATCH22_NAMES",
    "CATCH24_EXTRA_NAMES",
    "catch22_all",
    "catch22_features",
    "catch22_all_expr",
    # individual features
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "FC_LocalSimple_mean1_tauresrat",
    "FC_LocalSimple_mean3_stderr",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1",
    "SP_Summaries_welch_rect_centroid",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _as_1d(x) -> np.ndarray:
    """Coerce input to a contiguous 1-D float64 array with NaNs dropped."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size and np.isnan(arr).any():
        arr = arr[~np.isnan(arr)]
    return arr


def _zscore(x: np.ndarray) -> np.ndarray:
    """Z-score a series using the sample standard deviation (``ddof=1``).

    Returns an all-zero array if the input has (near-)zero spread, matching the
    degenerate-input behaviour used throughout catch22.  Z-scoring is
    idempotent, so applying it inside every feature keeps the feature functions
    self-contained.
    """
    x = _as_1d(x)
    if x.size == 0:
        return x
    mean = x.mean()
    std = x.std(ddof=1) if x.size > 1 else 0.0
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(x)
    return (x - mean) / std


def _acf(y: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation function via FFT (Wiener-Khinchin).

    Returns ``acf`` with ``acf[0] == 1`` and ``acf[k]`` the biased
    autocorrelation estimate at lag ``k`` for ``k = 0 .. n-1``.
    """
    n = y.size
    y = y - y.mean()
    if n == 0:
        return np.array([1.0])
    nfft = int(2 ** np.ceil(np.log2(2 * n - 1))) if n > 1 else 1
    f = np.fft.fft(y, n=nfft)
    acov = np.fft.ifft(f * np.conjugate(f))[:n].real
    if acov[0] == 0:
        out = np.zeros(n)
        out[0] = 1.0
        return out
    return acov / acov[0]


def _first_zero_ac(y: np.ndarray) -> int:
    """First lag ``tau >= 1`` at which the ACF crosses (<= 0). Falls back to n."""
    acf = _acf(y)
    n = acf.size
    for tau in range(1, n):
        if acf[tau] <= 0:
            return tau
    return n


def _histcounts(y: np.ndarray, n_bins: int):
    """Equal-width histogram over ``[min, max]``; the maximum falls in the last
    bin (numpy's default). Returns ``(counts, edges)``."""
    counts, edges = np.histogram(y, bins=n_bins)
    return counts.astype(np.float64), edges


def _coarsegrain_quantile(y: np.ndarray, n_groups: int) -> np.ndarray:
    """Symbolise a series into ``n_groups`` equiprobable (quantile) groups.

    Returns integer labels in ``0 .. n_groups - 1``.
    """
    qs = np.quantile(y, np.arange(1, n_groups) / n_groups, method="linear")
    # np.searchsorted maps values below the first threshold -> 0, etc.
    labels = np.searchsorted(qs, y, side="right")
    return np.clip(labels, 0, n_groups - 1).astype(int)


def _num_bins_auto(y: np.ndarray) -> int:
    """Scott's normal-reference rule for the number of equal-width bins."""
    n = y.size
    std = y.std(ddof=1) if n > 1 else 0.0
    if std < 1e-3 or n < 2:
        return 0
    span = y.max() - y.min()
    return int(np.ceil(span / (3.5 * std / n ** (1.0 / 3.0))))


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of ``True`` values in a boolean array."""
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _local_simple_residuals(y: np.ndarray, train_len: int) -> np.ndarray:
    """Residuals of a "local simple" mean forecaster.

    Point ``i`` (for ``i >= train_len``) is predicted by the mean of the
    preceding ``train_len`` points; the residual is ``y[i] - prediction``.
    """
    n = y.size
    if n <= train_len:
        return np.array([])
    # prediction for target i is the mean of the previous `train_len` values
    preds = np.empty(n - train_len)
    for i in range(train_len, n):
        preds[i - train_len] = y[i - train_len : i].mean()
    return y[train_len:] - preds


def _line_sse(x: np.ndarray, y: np.ndarray) -> float:
    """Sum of squared residuals of an ordinary least-squares line fit."""
    if x.size < 2:
        return 0.0
    coef = np.polyfit(x, y, 1)
    resid = y - np.polyval(coef, x)
    return float(np.dot(resid, resid))


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
    for i in range(n - 1):
        if acf[i + 1] < thresh:
            slope = acf[i + 1] - acf[i]
            if slope == 0:
                return float(i)
            return float(i + (thresh - acf[i]) / slope)
    return float(n)


def CO_FirstMin_ac(x) -> float:
    """Lag of the first local minimum of the autocorrelation function."""
    y = _zscore(x)
    acf = _acf(y)
    n = acf.size
    for i in range(1, n - 1):
        if acf[i] < acf[i - 1] and acf[i] < acf[i + 1]:
            return float(i)
    return float(n)


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
    for i in range(1, tau_max - 1):
        if ami[i] < ami[i - 1] and ami[i] < ami[i + 1]:
            return float(i)
    return float(tau_max)


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
    counts = np.zeros((3, 3))
    for a, b in zip(s[:-1], s[1:], strict=False):
        counts[a, b] += 1
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
    T = np.zeros((3, 3))
    for a, b in zip(s[:-1], s[1:], strict=False):
        T[a, b] += 1
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
    """
    y = _zscore(x)
    n = y.size
    if n < 8:
        return np.nan
    th = 0.01
    t = np.arange(n, dtype=float)
    y_spline = np.zeros(n)
    LSQUnivariateSpline = _get_lsq_spline()
    if LSQUnivariateSpline is not None:
        try:
            knots = np.linspace(0, n - 1, 5)[1:-1]  # 3 interior knots
            y_spline = LSQUnivariateSpline(t, y, knots, k=3)(t)
        except Exception:
            y_spline = np.zeros(n)
    y_sub = y - y_spline

    ac_max = int(np.ceil(n / 3))
    acf = _acf(y_sub)[:ac_max]

    troughs, peaks = [], []
    for i in range(1, acf.size - 1):
        slope_in = acf[i] - acf[i - 1]
        slope_out = acf[i + 1] - acf[i]
        if slope_in < 0 and slope_out > 0:
            troughs.append(i)
        elif slope_in > 0 and slope_out < 0:
            peaks.append(i)

    out = 0
    for ip in peaks:
        prior = [tr for tr in troughs if tr < ip]
        if not prior:
            continue
        it = prior[-1]
        if acf[ip] - acf[it] < th:
            continue
        if acf[ip] < 0:
            continue
        out = ip
        break
    return float(out)


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
    pct = np.full(n_thresh, np.nan)
    med_pos = np.full(n_thresh, np.nan)
    for j in range(n_thresh):
        thresh = j * inc
        high = np.flatnonzero(yw >= thresh)
        pct[j] = 100.0 * high.size / n
        if high.size > 0:
            med_pos[j] = np.median(high) / (n / 2.0) - 1.0
    # keep the (prefix of) thresholds where at least 2% of points remain
    valid = pct > 2.0
    if not valid.any():
        return np.nan
    return float(np.nanmedian(med_pos[valid]))


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


# ---------------------------------------------------------------------------
# Registry & aggregate entry points
# ---------------------------------------------------------------------------
CATCH22_FUNCS = {
    "DN_HistogramMode_5": DN_HistogramMode_5,
    "DN_HistogramMode_10": DN_HistogramMode_10,
    "CO_f1ecac": CO_f1ecac,
    "CO_FirstMin_ac": CO_FirstMin_ac,
    "CO_HistogramAMI_even_2_5": CO_HistogramAMI_even_2_5,
    "CO_trev_1_num": CO_trev_1_num,
    "CO_Embed2_Dist_tau_d_expfit_meandiff": CO_Embed2_Dist_tau_d_expfit_meandiff,
    "IN_AutoMutualInfoStats_40_gaussian_fmmi": IN_AutoMutualInfoStats_40_gaussian_fmmi,
    "MD_hrv_classic_pnn40": MD_hrv_classic_pnn40,
    "SB_BinaryStats_mean_longstretch1": SB_BinaryStats_mean_longstretch1,
    "SB_BinaryStats_diff_longstretch0": SB_BinaryStats_diff_longstretch0,
    "SB_MotifThree_quantile_hh": SB_MotifThree_quantile_hh,
    "SB_TransitionMatrix_3ac_sumdiagcov": SB_TransitionMatrix_3ac_sumdiagcov,
    "PD_PeriodicityWang_th0_01": PD_PeriodicityWang_th0_01,
    "FC_LocalSimple_mean1_tauresrat": FC_LocalSimple_mean1_tauresrat,
    "FC_LocalSimple_mean3_stderr": FC_LocalSimple_mean3_stderr,
    "DN_OutlierInclude_p_001_mdrmd": DN_OutlierInclude_p_001_mdrmd,
    "DN_OutlierInclude_n_001_mdrmd": DN_OutlierInclude_n_001_mdrmd,
    "SP_Summaries_welch_rect_area_5_1": SP_Summaries_welch_rect_area_5_1,
    "SP_Summaries_welch_rect_centroid": SP_Summaries_welch_rect_centroid,
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1": SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
}

#: Canonical ordering of the 22 catch22 feature names.
CATCH22_NAMES: tuple[str, ...] = tuple(CATCH22_FUNCS)

#: Extra features added by the "catch24" variant (raw mean and spread).
CATCH24_EXTRA_NAMES: tuple[str, ...] = ("DN_Mean", "DN_Spread_Std")


def _resolve_names(which, catch24: bool) -> list[str]:
    if which == "all":
        names = list(CATCH22_NAMES)
    elif isinstance(which, (list, tuple)):
        names = []
        for w in which:
            if w not in CATCH22_FUNCS:
                raise ValueError(
                    f"Unknown catch22 feature {w!r}. Valid names: {CATCH22_NAMES}"
                )
            names.append(w)
    else:
        raise TypeError("`which` must be 'all' or a list/tuple of feature names.")
    if catch24 and which == "all":
        names += list(CATCH24_EXTRA_NAMES)
    return names


def _compute(x: np.ndarray, names: Sequence[str]) -> dict[str, float]:
    x = _as_1d(x)
    out: dict[str, float] = {}
    for name in names:
        if name == "DN_Mean":
            out[name] = float(x.mean()) if x.size else np.nan
        elif name == "DN_Spread_Std":
            out[name] = float(x.std(ddof=1)) if x.size > 1 else np.nan
        else:
            try:
                out[name] = float(CATCH22_FUNCS[name](x))
            except Exception:
                out[name] = np.nan
    return out


def catch22_all(x, *, catch24: bool = False) -> dict[str, float]:
    """Compute the full catch22 (or catch24) feature set for one 1-D series.

    Parameters
    ----------
    x : array-like
        A single 1-D time series (list, :class:`numpy.ndarray` or
        :class:`polars.Series`).  The series is z-scored internally.
    catch24 : bool, default False
        If ``True``, additionally return the raw mean (``DN_Mean``) and sample
        standard deviation (``DN_Spread_Std``), yielding 24 features.

    Returns
    -------
    dict of str -> float
        Mapping from feature name to value, in canonical order.
    """
    names = _resolve_names("all", catch24)
    return _compute(x, names)


def catch22_features(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    column: str,
    which="all",
    catch24: bool = False,
) -> pl.DataFrame:
    """Compute catch22 features per entity for a long-format panel.

    The features are computed independently for each entity's sorted series,
    which makes the result leak-safe (no cross-entity or future information
    enters any feature).

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long-format panel.  A ``LazyFrame`` is collected internally.
    entity : str, optional
        Column identifying the entity/series.  If ``None``, the whole frame is
        treated as a single series and the result has one row.
    time : str, optional
        Column used to sort each entity's series in ascending order before
        feature extraction.  If ``None``, the existing row order is used.
    column : str
        Name of the value column to extract features from.
    which : {"all"} or list of str, default "all"
        Either ``"all"`` (the canonical 22 features) or an explicit list of
        feature names to compute.
    catch24 : bool, default False
        When ``which="all"``, also emit ``DN_Mean`` and ``DN_Spread_Std``.

    Returns
    -------
    polars.DataFrame
        One row per entity (or a single row when ``entity is None``), with the
        entity column (if any) followed by one column per requested feature.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    names = _resolve_names(which, catch24)

    if entity is None:
        sub = df.sort(time) if time is not None else df
        x = sub.get_column(column).to_numpy()
        return pl.DataFrame([_compute(x, names)])

    # Route through the lazy struct-expr path so Polars parallelises the
    # per-entity feature computation across threads (the former Python `for`
    # loop over groups was single-threaded). Sorting by [entity, time] yields
    # each group in time order (equivalent to the old per-group `sub.sort`);
    # the original first-appearance entity ordering is restored afterwards so
    # the output is row-for-row identical.
    sort_keys = [entity, time] if time is not None else [entity]
    result = (
        df.lazy()
        .sort(sort_keys, maintain_order=True)
        .group_by(entity, maintain_order=True)
        .agg(catch22_all_expr(column, which=which, catch24=catch24, alias="__c22"))
        .unnest("__c22")
        .collect()
    )

    order = (
        df.select(pl.col(entity)).unique(maintain_order=True).with_row_index("__ord")
    )
    return result.join(order, on=entity, how="left").sort("__ord").drop("__ord")


def catch22_all_expr(
    column: str,
    *,
    which="all",
    catch24: bool = False,
    alias: str = "catch22",
) -> pl.Expr:
    """Return a Polars expression producing a per-group catch22 struct.

    Designed for use inside ``group_by(entity).agg(...)`` so that features are
    computed per entity (leak-safe).  Unnest the resulting struct column to get
    one column per feature::

        (
            df.group_by("entity")
              .agg(catch22_all_expr("value"))
              .unnest("catch22")
        )

    Note: the caller is responsible for sorting each group by time beforehand
    (e.g. ``df.sort("entity", "time")``) if temporal ordering matters.
    """
    names = _resolve_names(which, catch24)
    struct_dtype = pl.Struct([pl.Field(n, pl.Float64) for n in names])

    def _fn(s: pl.Series) -> pl.Series:
        d = _compute(s.to_numpy(), names)
        return pl.Series([d], dtype=struct_dtype)

    return (
        pl.col(column)
        .map_batches(_fn, return_dtype=struct_dtype, returns_scalar=True)
        .alias(alias)
    )


# ---------------------------------------------------------------------------
# Optional, double-registration-guarded Polars namespace: ``expr.catch22``
# ---------------------------------------------------------------------------
if not hasattr(pl.Expr, "catch22"):

    @pl.api.register_expr_namespace("catch22")
    class _Catch22ExprNamespace:  # noqa: D101 - thin convenience wrapper
        def __init__(self, expr: pl.Expr) -> None:
            self._expr = expr

        def all(self, *, catch24: bool = False, alias: str = "catch22") -> pl.Expr:
            """Struct expression of all catch22 (or catch24) features.

            Use inside ``group_by(...).agg(...)``; see :func:`catch22_all_expr`.
            """
            names = _resolve_names("all", catch24)
            struct_dtype = pl.Struct([pl.Field(n, pl.Float64) for n in names])

            def _fn(s: pl.Series) -> pl.Series:
                return pl.Series([_compute(s.to_numpy(), names)], dtype=struct_dtype)

            return self._expr.map_batches(
                _fn, return_dtype=struct_dtype, returns_scalar=True
            ).alias(alias)
