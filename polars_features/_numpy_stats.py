"""Dependency-free numpy replacements for the handful of SciPy calls used by
the eager feature-extraction modules.

Importing this module must never pull in SciPy (or anything beyond numpy), so
that ``import polars_features.feature_extractors`` / ``.catch22`` stays scipy
free.  Every function here is a drop-in for the SciPy call it replaces, verified
to match SciPy to the tolerances measured in the D3 investigation (mostly
``<= 1e-14``); see ``tests/test_numpy_stats_parity.py``.

Covered:

* :func:`lstsq`        -> ``scipy.linalg.lstsq`` (exact; wraps ``np.linalg.lstsq``)
* :func:`norm_ppf`     -> ``scipy.stats.norm.ppf`` (Acklam + one Halley step)
* :func:`norm_cdf`     -> ``scipy.stats.norm.cdf`` (``math.erf``)
* :func:`skew`         -> ``scipy.stats.skew`` (bias=True)
* :func:`kurtosis`     -> ``scipy.stats.kurtosis`` (Fisher, bias=True)
* :func:`normaltest_stat` -> ``scipy.stats.normaltest(...)[0]`` (D'Agostino K^2)
* :func:`welch`        -> ``scipy.signal.welch`` (rfft + periodic window)
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "lstsq",
    "norm_cdf",
    "norm_ppf",
    "skew",
    "kurtosis",
    "normaltest_stat",
    "welch",
]


# ---------------------------------------------------------------------------
# Linear least squares
# ---------------------------------------------------------------------------
def lstsq(a, b, cond=None):
    """Drop-in for ``scipy.linalg.lstsq``.

    Returns ``(coeffs, residues, rank, singular_values)`` just like
    ``scipy.linalg.lstsq``; delegates to :func:`numpy.linalg.lstsq`.  The
    ``cond`` argument is accepted for signature compatibility and mapped to
    numpy's ``rcond``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    coeffs, residues, rank, sv = np.linalg.lstsq(a, b, rcond=cond)
    return coeffs, residues, rank, sv


# ---------------------------------------------------------------------------
# Normal distribution ppf / cdf
# ---------------------------------------------------------------------------
_ERF = np.vectorize(math.erf, otypes=[float])


def norm_cdf(x):
    """Standard normal CDF, matching ``scipy.stats.norm.cdf`` (<= 1e-16)."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + _ERF(x / math.sqrt(2.0)))


# Acklam rational-approximation coefficients.
_PPF_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_PPF_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_PPF_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_PPF_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)


def norm_ppf(p):
    """Standard normal inverse CDF, matching ``scipy.stats.norm.ppf`` (<= 1e-14).

    Acklam's rational approximation followed by one Halley refinement using the
    erf-based CDF.  Accepts scalars or arrays; returns an array-like matching the
    input shape (a 0-d array for scalar input).
    """
    a, b, c, d = _PPF_A, _PPF_B, _PPF_C, _PPF_D
    p = np.asarray(p, dtype=float)
    scalar = p.ndim == 0
    p = np.atleast_1d(p)
    x = np.empty_like(p)

    plow, phigh = 0.02425, 1 - 0.02425
    lo = p < plow
    hi = p > phigh
    mid = ~(lo | hi)

    q = np.sqrt(-2 * np.log(p[lo]))
    x[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    x[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )
    q = p[mid] - 0.5
    r = q * q
    x[mid] = (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )

    # One Halley refinement step.
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * np.exp(x * x / 2.0)
    x = x - u / (1 + x * u / 2.0)
    return x[0] if scalar else x


# ---------------------------------------------------------------------------
# Moments: skew / kurtosis (SciPy defaults: bias=True, kurtosis Fisher)
# ---------------------------------------------------------------------------
def skew(a):
    """Sample skewness, matching ``scipy.stats.skew`` defaults (bias=True)."""
    a = np.asarray(a, dtype=float)
    m = a - a.mean()
    m2 = np.mean(m**2)
    m3 = np.mean(m**3)
    return m3 / m2**1.5


def kurtosis(a):
    """Excess kurtosis, matching ``scipy.stats.kurtosis`` defaults
    (Fisher=True, bias=True)."""
    a = np.asarray(a, dtype=float)
    m = a - a.mean()
    m2 = np.mean(m**2)
    m4 = np.mean(m**4)
    return m4 / m2**2 - 3.0


# ---------------------------------------------------------------------------
# D'Agostino-Pearson normality test statistic
# ---------------------------------------------------------------------------
def normaltest_stat(a):
    """The ``K^2`` statistic of ``scipy.stats.normaltest`` (the ``[0]`` value).

    Matches SciPy's statistic to ~1e-14.  The two-degree-of-freedom p-value is
    the closed form ``exp(-K^2 / 2)`` if ever needed, but only the statistic is
    used in-tree.
    """
    a = np.asarray(a, dtype=float)
    n = a.size

    # skewtest
    b1 = skew(a)
    y = b1 * math.sqrt(((n + 1) * (n + 3)) / (6.0 * (n - 2)))
    beta2 = (
        3.0
        * (n * n + 27 * n - 70)
        * (n + 1)
        * (n + 3)
        / ((n - 2.0) * (n + 5) * (n + 7) * (n + 9))
    )
    w2 = -1 + math.sqrt(2 * (beta2 - 1))
    delta = 1 / math.sqrt(0.5 * math.log(w2))
    alpha = math.sqrt(2.0 / (w2 - 1))
    y = 1e-300 if y == 0 else y
    z_skew = delta * math.log(y / alpha + math.sqrt((y / alpha) ** 2 + 1))

    # kurtosistest
    b2 = kurtosis(a) + 3.0
    e = 3.0 * (n - 1) / (n + 1)
    varb2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))
    x = (b2 - e) / math.sqrt(varb2)
    sqrtbeta1 = (
        6.0
        * (n * n - 5 * n + 2)
        / ((n + 7) * (n + 9))
        * math.sqrt((6.0 * (n + 3) * (n + 5)) / (n * (n - 2) * (n - 3)))
    )
    aa = 6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + math.sqrt(1 + 4.0 / sqrtbeta1**2))
    term1 = 1 - 2 / (9.0 * aa)
    denom = 1 + x * math.sqrt(2 / (aa - 4.0))
    term2 = np.sign(denom) * ((1 - 2.0 / aa) / abs(denom)) ** (1 / 3.0)
    z_kurt = (term1 - term2) / math.sqrt(2 / (9.0 * aa))

    return float(z_skew**2 + z_kurt**2)


# ---------------------------------------------------------------------------
# Welch power spectral density
# ---------------------------------------------------------------------------
def _get_window(window: str, nperseg: int) -> np.ndarray:
    if window == "hann":
        # periodic Hann (SciPy get_window default sym=False)
        return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nperseg) / nperseg)
    if window == "boxcar":
        return np.ones(nperseg, dtype=float)
    raise ValueError(f"Unsupported window {window!r}")


def welch(
    x,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
    detrend: str | bool = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
):
    """Estimate the power spectral density using Welch's method.

    Drop-in for the ``scipy.signal.welch`` calls used in-tree: one-sided,
    density scaling, ``hann``/``boxcar`` windows, ``constant``/no detrend.
    Matches SciPy's default parameterization (``Pxx`` to ~3e-14).

    Returns ``(f, Pxx)``.
    """
    if not return_onesided:
        raise NotImplementedError("only one-sided welch is supported")
    if scaling != "density":
        raise NotImplementedError("only density scaling is supported")

    x = np.asarray(x, dtype=float)
    n = x.size
    nperseg = min(nperseg, n)
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap

    win = _get_window(window, nperseg)
    scale = 1.0 / (fs * (win * win).sum())

    segs = []
    for s in range(0, n - nperseg + 1, step):
        seg = x[s : s + nperseg]
        if detrend == "constant":
            seg = seg - seg.mean()
        seg = seg * win
        z = np.fft.rfft(seg, n=nperseg)
        p = (z.conj() * z).real * scale
        if nperseg % 2 == 0:
            p[1:-1] *= 2
        else:
            p[1:] *= 2
        segs.append(p)

    pxx = np.mean(segs, axis=0)
    f = np.fft.rfftfreq(nperseg, 1 / fs)
    return f, pxx
