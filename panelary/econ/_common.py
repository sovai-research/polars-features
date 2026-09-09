"""Shared, dependency-free numerical helpers for :mod:`panelary.econ`.

Everything here is **pure NumPy**. No
``scipy`` / ``statsmodels`` / ``linearmodels``: the econometric estimators in
this package must work with Panelary's ``{numpy, polars}`` core.

Contents
--------
* **Distributions** -- :func:`chi2_sf`, :func:`t_sf`, :func:`t_cdf`,
  :func:`f_sf`, re-exported :func:`norm_cdf` / :func:`norm_ppf`. Implemented
  from the standard series / continued-fraction expansions of the incomplete
  gamma and incomplete beta functions.
* **Linear algebra** -- :func:`ols`, :func:`pinv_sym`.
* **Grouping** -- :func:`factorize`, :func:`group_mean`, :func:`group_sum`:
  the numpy side of a Polars ``group_by``.
* **HAC** -- :func:`bartlett_weights`, :func:`newey_west_lrv`,
  :func:`newey_west_scalar`, :func:`auto_bandwidth`.
* **Misc** -- :func:`winsorize`.

Notes
-----
The distribution functions target ~1e-12 relative accuracy over the ranges the
tests exercise; they are used for p-values, never inside an optimisation loop.
"""

from __future__ import annotations

import math

import numpy as np

from panelary._numpy_stats import norm_cdf, norm_ppf

__all__ = [
    "chi2_sf",
    "t_sf",
    "t_cdf",
    "f_sf",
    "norm_cdf",
    "norm_ppf",
    "ols",
    "pinv_sym",
    "factorize",
    "group_mean",
    "group_sum",
    "bartlett_weights",
    "newey_west_lrv",
    "newey_west_scalar",
    "auto_bandwidth",
    "winsorize",
]

_EPS = np.finfo(float).eps


# --------------------------------------------------------------------------- #
# Incomplete gamma / beta -> chi2, t, F tails
# --------------------------------------------------------------------------- #
def _gamma_p_series(a: float, x: float) -> float:
    """Lower regularised incomplete gamma ``P(a, x)`` by its power series."""
    ap = a
    total = 1.0 / a
    term = total
    for _ in range(1000):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * 1e-16:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_cf(a: float, x: float) -> float:
    """Upper regularised incomplete gamma ``Q(a, x)`` by continued fraction."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q(a: float, x: float) -> float:
    """Upper regularised incomplete gamma function ``Q(a, x) = 1 - P(a, x)``."""
    if x < 0.0 or a <= 0.0:
        raise ValueError(f"invalid arguments to the incomplete gamma: a={a}, x={x}.")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gamma_p_series(a, x)
    return _gamma_q_cf(a, x)


def chi2_sf(x: float | np.ndarray, df: float) -> float | np.ndarray:
    """Survival function ``P(X > x)`` of a chi-squared with ``df`` degrees of freedom.

    Parameters
    ----------
    x : float or ndarray
        Quantile(s); negative values return 1.
    df : float
        Degrees of freedom, ``> 0``.

    Returns
    -------
    float or ndarray
        Upper-tail probability. Matches ``scipy.stats.chi2.sf`` to ~1e-12.
    """
    if df <= 0:
        raise ValueError(f"`df` must be positive, got {df}.")
    arr = np.asarray(x, dtype=float)
    flat = np.atleast_1d(arr).ravel()
    out = np.empty_like(flat)
    for i, v in enumerate(flat):
        if not np.isfinite(v):
            out[i] = np.nan if np.isnan(v) else (0.0 if v > 0 else 1.0)
        elif v <= 0.0:
            out[i] = 1.0
        else:
            out[i] = _gamma_q(0.5 * df, 0.5 * v)
    out = np.clip(out, 0.0, 1.0)
    if arr.ndim == 0:
        return float(out[0])
    return out.reshape(arr.shape)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularised incomplete beta (Lentz's method)."""
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return (
        1.0
        - math.exp(
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + b * math.log1p(-x)
            + a * math.log(x)
        )
        * _betacf(b, a, 1.0 - x)
        / b
    )


def t_sf(t: float | np.ndarray, df: float) -> float | np.ndarray:
    """Upper-tail ``P(T > t)`` of Student's t with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError(f"`df` must be positive, got {df}.")
    arr = np.asarray(t, dtype=float)
    flat = np.atleast_1d(arr).ravel()
    out = np.empty_like(flat)
    for i, v in enumerate(flat):
        if np.isnan(v):
            out[i] = np.nan
            continue
        xx = df / (df + v * v)
        half = 0.5 * _betainc(0.5 * df, 0.5, xx)
        out[i] = half if v > 0 else 1.0 - half
    if arr.ndim == 0:
        return float(out[0])
    return out.reshape(arr.shape)


def t_cdf(t: float | np.ndarray, df: float) -> float | np.ndarray:
    """CDF ``P(T <= t)`` of Student's t with ``df`` degrees of freedom."""
    return 1.0 - np.asarray(t_sf(t, df))


def f_sf(x: float, dfn: float, dfd: float) -> float:
    """Upper-tail ``P(F > x)`` of an F distribution with ``(dfn, dfd)`` d.o.f."""
    if x <= 0:
        return 1.0
    xx = dfd / (dfd + dfn * x)
    return float(_betainc(0.5 * dfd, 0.5 * dfn, xx))


# --------------------------------------------------------------------------- #
# Linear algebra
# --------------------------------------------------------------------------- #
def pinv_sym(a: np.ndarray, *, rcond: float = 1e-12) -> np.ndarray:
    """Pseudo-inverse of a symmetric matrix via its eigendecomposition."""
    a = 0.5 * (np.asarray(a, dtype=float) + np.asarray(a, dtype=float).T)
    w, v = np.linalg.eigh(a)
    keep = np.abs(w) > rcond * max(np.abs(w).max(), _EPS)
    winv = np.zeros_like(w)
    winv[keep] = 1.0 / w[keep]
    return (v * winv) @ v.T


def ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ordinary least squares.

    Parameters
    ----------
    x : ndarray, shape (n, k)
        Design matrix (include your own intercept column if you want one).
    y : ndarray, shape (n,) or (n, m)
        Response.

    Returns
    -------
    beta : ndarray, shape (k,) or (k, m)
    resid : ndarray
        ``y - x @ beta``.
    xtx_inv : ndarray, shape (k, k)
        ``(x'x)^-1`` (pseudo-inverse if ``x`` is rank-deficient).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xtx = x.T @ x
    xtx_inv = pinv_sym(xtx)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    return beta, resid, xtx_inv


# --------------------------------------------------------------------------- #
# Grouping (the numpy side of a Polars group_by)
# --------------------------------------------------------------------------- #
def factorize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map ``values`` to dense integer codes.

    Returns
    -------
    codes : ndarray of int64, shape (n,)
    levels : ndarray, shape (n_levels,)
        Sorted unique values; ``levels[codes] == values``.
    """
    arr = np.asarray(values)
    levels, codes = np.unique(arr, return_inverse=True)
    return codes.astype(np.int64, copy=False).ravel(), levels


def group_sum(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum ``values`` (``(n,)`` or ``(n, k)``) within each integer group code."""
    v = np.asarray(values, dtype=float)
    single = v.ndim == 1
    mat = v[:, None] if single else v
    out = np.empty((n_groups, mat.shape[1]), dtype=float)
    for j in range(mat.shape[1]):
        out[:, j] = np.bincount(codes, weights=mat[:, j], minlength=n_groups)
    return out[:, 0] if single else out


def group_mean(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Mean of ``values`` within each integer group code (empty groups -> 0)."""
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    counts[counts == 0.0] = 1.0
    sums = group_sum(values, codes, n_groups)
    if sums.ndim == 1:
        return sums / counts
    return sums / counts[:, None]


# --------------------------------------------------------------------------- #
# HAC / Newey-West
# --------------------------------------------------------------------------- #
def bartlett_weights(lags: int) -> np.ndarray:
    """Bartlett kernel weights ``1 - j / (lags + 1)`` for ``j = 1 .. lags``."""
    if lags < 0:
        raise ValueError(f"`lags` must be >= 0, got {lags}.")
    j = np.arange(1, lags + 1, dtype=float)
    return 1.0 - j / (lags + 1.0)


def auto_bandwidth(n: int) -> int:
    """Newey-West's rule-of-thumb bandwidth ``floor(4 (n/100)^(2/9))``."""
    if n <= 1:
        return 0
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_lrv(scores: np.ndarray, lags: int) -> np.ndarray:
    """HAC long-run covariance of a ``(T, k)`` score series (Bartlett kernel).

    Returns ``S = Gamma_0 + sum_j w_j (Gamma_j + Gamma_j')`` where
    ``Gamma_j = sum_t s_t s_{t-j}'``. Note this is a **sum**, not an average:
    callers divide by ``T`` where their formula requires it.
    """
    s = np.atleast_2d(np.asarray(scores, dtype=float))
    if s.shape[0] == 1 and s.shape[1] != 1:
        s = s.T
    t_obs = s.shape[0]
    lags = int(min(lags, max(t_obs - 1, 0)))
    out = s.T @ s
    for j, w in enumerate(bartlett_weights(lags), start=1):
        gamma = s[j:].T @ s[:-j]
        out = out + w * (gamma + gamma.T)
    return out


def newey_west_scalar(series: np.ndarray, lags: int) -> float:
    """Newey-West variance of the **mean** of a scalar series.

    ``Var(xbar) = (1/T^2) [ Gamma_0 + sum_j w_j (Gamma_j + Gamma_j') ]`` with
    the series demeaned first. This is the standard Fama-MacBeth standard error
    on the lambda time series.
    """
    x = np.asarray(series, dtype=float).ravel()
    x = x[np.isfinite(x)]
    t_obs = x.size
    if t_obs == 0:
        return float("nan")
    if t_obs == 1:
        return 0.0
    dev = (x - x.mean())[:, None]
    s = float(newey_west_lrv(dev, lags)[0, 0])
    return s / (t_obs * t_obs)


def winsorize(x: np.ndarray, limit: float) -> np.ndarray:
    """Symmetrically winsorize ``x`` at the ``limit`` / ``1 - limit`` quantiles."""
    if limit <= 0.0:
        return np.asarray(x, dtype=float)
    if not 0.0 < limit < 0.5:
        raise ValueError(f"`limit` must be in (0, 0.5), got {limit}.")
    arr = np.asarray(x, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    lo, hi = np.quantile(finite, [limit, 1.0 - limit])
    return np.clip(arr, lo, hi)
