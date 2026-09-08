"""Unit-root / stationarity battery: ADF, KPSS, PP, DF-GLS, Ng-Perron, Zivot-Andrews.

Two uses, both leak-safe:

1. **Preprocessing.** :class:`StationarityDifferencer` picks a differencing order
   per entity from the *training* rows only and freezes it, so the same order is
   applied to every later fold. Choosing the transform from the full sample is a
   classic, silent look-ahead leak; this class exists to make that impossible.
2. **Features.** The test statistic, its approximate p-value and (for
   Zivot-Andrews) the estimated break date are cheap persistence / regime
   descriptors. Computed on **trailing** windows via
   :func:`rolling_unit_root_features` they are invariant to future rows.

All kernels are pure NumPy: no ``scipy``, no ``statsmodels``.

Critical values
---------------
p-values are **approximations** obtained by interpolating published
critical-value tables in the normal-quantile domain (see
:func:`polars_features.econ.features._common.interp_pvalue`); they are exact at
the tabulated points. The embedded tables are:

* ADF / Phillips-Perron ``tau``: MacKinnon (2010) response surfaces
  ``crit = b_inf + b1/T + b2/T^2 + b3/T^3``, for the no-constant, constant and
  constant-plus-trend cases.
* KPSS: Kwiatkowski, Phillips, Schmidt & Shin (1992), Table 1.
* DF-GLS: the demeaned case is asymptotically the Dickey-Fuller no-constant
  distribution; the detrended case uses Elliott, Rothenberg & Stock (1996),
  Table 1, interpolated in sample size.
* Ng-Perron (2001), Table 1, for ``MZa`` / ``MZt`` / ``MSB`` / ``MPT``.
* Zivot & Andrews (1992), Tables 2-4, for the three break models.

References
----------
Dickey & Fuller (1979); Kwiatkowski et al. (1992); Phillips & Perron (1988);
Elliott, Rothenberg & Stock (1996); Ng & Perron (2001); Zivot & Andrews (1992);
MacKinnon (2010), "Critical Values for Cointegration Tests".
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import polars as pl

from polars_features.core.panel_frame import PanelFrame
from polars_features.core.protocol import PanelTransformer
from polars_features.econ.features._common import (
    add_deterministic,
    auto_hac_lags,
    bartlett_lrvar,
    interp_pvalue,
    lagmat,
    ols,
    per_entity_apply,
    per_entity_reduce,
    rolling_apply,
    sorted_panel,
)

__all__ = [
    "UnitRootResult",
    "ZivotAndrewsResult",
    "NgPerronResult",
    "adf",
    "kpss",
    "phillips_perron",
    "dfgls",
    "ng_perron",
    "zivot_andrews",
    "unit_root_table",
    "rolling_unit_root_features",
    "StationarityDifferencer",
]


# --------------------------------------------------------------------------- #
# Critical-value tables
# --------------------------------------------------------------------------- #
#: MacKinnon (2010) response-surface coefficients for the Dickey-Fuller ``tau``
#: distribution, N = 1 (no cointegration). Keyed by trend case, then by
#: significance level, giving ``(b_inf, b1, b2, b3)`` for
#: ``crit(T) = b_inf + b1/T + b2/T**2 + b3/T**3``.
_MACKINNON_TAU: dict[str, dict[float, tuple[float, float, float, float]]] = {
    "n": {
        0.01: (-2.56574, -2.2358, -3.627, 0.0),
        0.025: (-2.21778, -1.1163, -3.481, 12.600),
        0.05: (-1.94100, -0.2686, -3.365, 31.223),
        0.10: (-1.61682, 0.2656, -2.714, -5.807),
    },
    "c": {
        0.01: (-3.43035, -6.5393, -16.786, -79.433),
        0.025: (-3.12369, -4.2334, -5.999, -29.250),
        0.05: (-2.86154, -2.8903, -4.234, -40.040),
        0.10: (-2.56677, -1.5384, -2.809, 0.0),
    },
    "ct": {
        0.01: (-3.95877, -9.0531, -28.428, -134.155),
        0.025: (-3.66426, -6.0620, -13.883, -66.187),
        0.05: (-3.41049, -4.3904, -9.036, -45.374),
        0.10: (-3.12705, -2.5856, -3.925, -22.380),
    },
}

#: KPSS critical values (Kwiatkowski et al. 1992, Table 1). Upper tail.
_KPSS_CRIT: dict[str, dict[float, float]] = {
    "c": {0.10: 0.347, 0.05: 0.463, 0.025: 0.574, 0.01: 0.739},
    "ct": {0.10: 0.119, 0.05: 0.146, 0.025: 0.176, 0.01: 0.216},
}

#: DF-GLS detrended-case critical values (Elliott, Rothenberg & Stock 1996,
#: Table 1), by sample size; interpolated in ``T``. The demeaned case uses the
#: Dickey-Fuller no-constant response surface instead.
_ERS_CT_CRIT: dict[int, dict[float, float]] = {
    50: {0.01: -3.77, 0.05: -3.19, 0.10: -2.89},
    100: {0.01: -3.58, 0.05: -3.03, 0.10: -2.74},
    200: {0.01: -3.46, 0.05: -2.93, 0.10: -2.64},
    10_000: {0.01: -3.48, 0.05: -2.89, 0.10: -2.57},
}

#: Ng-Perron (2001) Table 1 critical values, by trend case and statistic.
_NG_PERRON_CRIT: dict[str, dict[str, dict[float, float]]] = {
    "c": {
        "mza": {0.01: -13.8, 0.05: -8.1, 0.10: -5.7},
        "mzt": {0.01: -2.58, 0.05: -1.98, 0.10: -1.62},
        "msb": {0.01: 0.174, 0.05: 0.233, 0.10: 0.275},
        "mpt": {0.01: 1.78, 0.05: 3.17, 0.10: 4.45},
    },
    "ct": {
        "mza": {0.01: -23.8, 0.05: -17.3, 0.10: -14.2},
        "mzt": {0.01: -3.42, 0.05: -2.91, 0.10: -2.62},
        "msb": {0.01: 0.143, 0.05: 0.168, 0.10: 0.185},
        "mpt": {0.01: 4.03, 0.05: 5.48, 0.10: 6.67},
    },
}

#: Zivot & Andrews (1992) critical values by break model.
_ZA_CRIT: dict[str, dict[float, float]] = {
    "intercept": {0.01: -5.34, 0.05: -4.80, 0.10: -4.58},
    "trend": {0.01: -4.93, 0.05: -4.42, 0.10: -4.11},
    "both": {0.01: -5.57, 0.05: -5.08, 0.10: -4.82},
}


def _mackinnon_crit(trend: str, nobs: int) -> dict[float, float]:
    table = _MACKINNON_TAU[trend]
    t = float(max(nobs, 2))
    return {
        lvl: b0 + b1 / t + b2 / t**2 + b3 / t**3
        for lvl, (b0, b1, b2, b3) in table.items()
    }


def _ers_ct_crit(nobs: int) -> dict[float, float]:
    sizes = sorted(_ERS_CT_CRIT)
    out: dict[float, float] = {}
    for lvl in (0.01, 0.05, 0.10):
        xs = np.array(sizes, dtype=float)
        ys = np.array([_ERS_CT_CRIT[s][lvl] for s in sizes], dtype=float)
        out[lvl] = float(np.interp(float(nobs), xs, ys))
    return out


def _pvalue(stat: float, crit: dict[float, float]) -> float:
    levels = sorted(crit)
    return interp_pvalue(stat, [crit[lv] for lv in levels], levels)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
class UnitRootResult(NamedTuple):
    """A single unit-root / stationarity test outcome.

    Attributes
    ----------
    stat : float
        The test statistic.
    pvalue : float
        Approximate p-value from the embedded critical-value table.
    lags : int
        Number of augmenting lags / HAC bandwidth actually used.
    nobs : int
        Effective number of observations in the test regression.
    crit_values : dict
        The critical values used, keyed by significance level.
    null_is_unit_root : bool
        ``True`` for ADF/PP/DF-GLS/ZA (small statistic rejects a unit root),
        ``False`` for KPSS (the null is stationarity).
    """

    stat: float
    pvalue: float
    lags: int
    nobs: int
    crit_values: dict[float, float]
    null_is_unit_root: bool


class ZivotAndrewsResult(NamedTuple):
    """Zivot-Andrews minimum-t outcome plus the estimated break location."""

    stat: float
    pvalue: float
    break_index: int
    break_fraction: float
    lags: int
    nobs: int
    crit_values: dict[float, float]
    regression: str


class NgPerronResult(NamedTuple):
    """The four Ng-Perron M statistics with their p-values."""

    mza: float
    mzt: float
    msb: float
    mpt: float
    pvalue_mzt: float
    lags: int
    nobs: int
    crit_values: dict[str, dict[float, float]]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clean(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _default_maxlag(nobs: int) -> int:
    return int(np.ceil(12.0 * (nobs / 100.0) ** 0.25))


def _ic(ssr: float, nobs: int, k: int, criterion: str) -> float:
    if ssr <= 0 or nobs <= 0:
        return np.inf
    llf_term = nobs * np.log(ssr / nobs)
    if criterion == "aic":
        return llf_term + 2.0 * k
    if criterion == "bic":
        return llf_term + k * np.log(nobs)
    raise ValueError(f"unknown information criterion {criterion!r}.")


def _adf_design(x: np.ndarray, lags: int, trend: str) -> tuple[np.ndarray, np.ndarray]:
    """Build the augmented Dickey-Fuller regression ``dy ~ y_{t-1} + dlags + det``."""
    dx = np.diff(x)
    n = dx.shape[0]
    rows = n - lags
    if rows <= 0:
        raise ValueError("series too short for the requested number of lags.")
    y = dx[lags:]
    level = x[lags : lags + rows]  # y_{t-1} aligned with dy_t
    cols = [level[:, None]]
    if lags > 0:
        cols.append(lagmat(dx, lags))
    det = add_deterministic(rows, trend)
    if det.shape[1]:
        cols.append(det)
    return np.column_stack(cols), y


# --------------------------------------------------------------------------- #
# ADF
# --------------------------------------------------------------------------- #
def adf(
    x: np.ndarray,
    *,
    trend: str = "c",
    lags: int | None = None,
    max_lags: int | None = None,
    criterion: str = "aic",
) -> UnitRootResult:
    """Augmented Dickey-Fuller test (null: a unit root).

    Fits ``dy_t = gamma * y_{t-1} + sum_i delta_i dy_{t-i} + deterministic + e_t``
    and reports the t-statistic on ``gamma``. Small (very negative) statistics
    reject the unit root, i.e. favour stationarity.

    Parameters
    ----------
    x : numpy.ndarray
        1-D series in time order. Non-finite values are dropped.
    trend : {"n", "c", "ct"}, default="c"
        Deterministic terms in the test regression.
    lags : int, optional
        Fixed number of augmenting lags. When ``None`` the lag order minimising
        ``criterion`` over ``0 .. max_lags`` is used.
    max_lags : int, optional
        Upper bound for lag selection; defaults to ``ceil(12 * (n/100)**0.25)``.
    criterion : {"aic", "bic"}, default="aic"
        Information criterion used when ``lags`` is ``None``.

    Returns
    -------
    UnitRootResult
    """
    arr = _clean(x)
    n = arr.shape[0]
    if n < 8:
        raise ValueError(f"ADF needs at least 8 finite observations, got {n}.")
    if lags is None:
        upper = _default_maxlag(n) if max_lags is None else int(max_lags)
        upper = int(min(upper, max(0, (n - 4) // 2)))
        best, best_ic = 0, np.inf
        for cand in range(upper + 1):
            try:
                X, y = _adf_design(arr, cand, trend)
            except ValueError:
                break
            # Compare every candidate on the SAME sample (the one the largest
            # lag order can support), otherwise the criteria are not comparable.
            drop = upper - cand
            X, y = X[drop:], y[drop:]
            if y.shape[0] <= X.shape[1]:
                break
            res = ols(X, y)
            val = _ic(float(res.resid @ res.resid), res.nobs, X.shape[1], criterion)
            if val < best_ic:
                best, best_ic = cand, val
        lags = best
    X, y = _adf_design(arr, int(lags), trend)
    res = ols(X, y)
    stat = float(res.tstat[0])
    crit = _mackinnon_crit(trend, res.nobs)
    return UnitRootResult(stat, _pvalue(stat, crit), int(lags), res.nobs, crit, True)


# --------------------------------------------------------------------------- #
# KPSS
# --------------------------------------------------------------------------- #
def kpss(x: np.ndarray, *, trend: str = "c", lags: int | None = None) -> UnitRootResult:
    """KPSS stationarity test (null: **stationarity** around a level or trend).

    ``eta = sum_t S_t^2 / (n^2 * s^2(l))`` where ``S_t`` is the partial sum of
    the residuals from regressing ``x`` on the deterministic terms and ``s^2(l)``
    is a Bartlett long-run variance. **Large** statistics reject stationarity, so
    KPSS is the natural confirmatory complement to ADF.

    Parameters
    ----------
    x : numpy.ndarray
        1-D series in time order.
    trend : {"c", "ct"}, default="c"
        ``"c"`` tests level stationarity, ``"ct"`` trend stationarity.
    lags : int, optional
        Bartlett bandwidth; defaults to ``ceil(4 * (n/100)**0.25)``.
    """
    if trend not in ("c", "ct"):
        raise ValueError(f"KPSS `trend` must be 'c' or 'ct', got {trend!r}.")
    arr = _clean(x)
    n = arr.shape[0]
    if n < 8:
        raise ValueError(f"KPSS needs at least 8 finite observations, got {n}.")
    det = add_deterministic(n, trend)
    resid = ols(det, arr).resid
    s = np.cumsum(resid)
    if lags is None:
        lags = int(np.ceil(4.0 * (n / 100.0) ** 0.25))
    lrv = bartlett_lrvar(resid, int(lags))
    stat = float(np.sum(s**2) / (n**2 * lrv)) if lrv > 0 else float("nan")
    crit = dict(_KPSS_CRIT[trend])
    return UnitRootResult(stat, _pvalue(stat, crit), int(lags), n, crit, False)


# --------------------------------------------------------------------------- #
# Phillips-Perron
# --------------------------------------------------------------------------- #
def phillips_perron(
    x: np.ndarray, *, trend: str = "c", lags: int | None = None
) -> UnitRootResult:
    """Phillips-Perron ``Z_tau`` test (null: a unit root).

    Runs the *un-augmented* regression ``y_t = rho * y_{t-1} + deterministic``
    and corrects the t-statistic non-parametrically for serial correlation with a
    Bartlett long-run variance, instead of adding lagged differences as ADF does.

    Parameters
    ----------
    x : numpy.ndarray
        1-D series in time order.
    trend : {"n", "c", "ct"}, default="c"
        Deterministic terms.
    lags : int, optional
        Bartlett bandwidth; defaults to the Newey-West rule
        ``floor(4 * (n/100)**(2/9))``.

    Notes
    -----
    ``Z_tau = sqrt(gamma0 / lambda2) * t_rho
    - (lambda2 - gamma0) * n * se(rho) / (2 * lambda * s)``, with ``gamma0`` the
    residual variance, ``lambda2`` its long-run counterpart and ``s`` the
    regression standard error (Phillips & Perron 1988).
    """
    arr = _clean(x)
    n = arr.shape[0]
    if n < 8:
        raise ValueError(f"Phillips-Perron needs at least 8 observations, got {n}.")
    y = arr[1:]
    lag = arr[:-1]
    rows = y.shape[0]
    det = add_deterministic(rows, trend)
    X = np.column_stack([lag[:, None], det]) if det.shape[1] else lag[:, None]
    res = ols(X, y)
    u = res.resid
    if lags is None:
        lags = auto_hac_lags(rows)
    gamma0 = float(u @ u) / rows
    lam2 = bartlett_lrvar(u, int(lags))
    s = float(np.sqrt(res.sigma2))
    se_rho = float(res.se[0])
    # t-statistic for H0: rho == 1 (NOT the default `rho == 0` t-stat).
    t_rho = float((res.beta[0] - 1.0) / se_rho) if se_rho > 0 else np.nan
    if not (lam2 > 0 and gamma0 > 0 and s > 0 and np.isfinite(t_rho)):
        stat = float("nan")
    else:
        lam = float(np.sqrt(lam2))
        stat = float(
            np.sqrt(gamma0 / lam2) * t_rho
            - (lam2 - gamma0) * rows * se_rho / (2.0 * lam * s)
        )
    crit = _mackinnon_crit(trend, rows)
    return UnitRootResult(stat, _pvalue(stat, crit), int(lags), rows, crit, True)


# --------------------------------------------------------------------------- #
# GLS detrending (shared by DF-GLS and Ng-Perron)
# --------------------------------------------------------------------------- #
def _gls_detrend(x: np.ndarray, trend: str) -> tuple[np.ndarray, float]:
    """Elliott-Rothenberg-Stock local-to-unity GLS detrending.

    Quasi-differences the series and the deterministic terms at
    ``alpha = 1 + cbar / n`` (``cbar = -7`` demeaned, ``-13.5`` detrended),
    regresses one on the other and returns the detrended series plus ``cbar``.
    """
    n = x.shape[0]
    cbar = -7.0 if trend == "c" else -13.5
    alpha = 1.0 + cbar / n
    z = add_deterministic(n, "c" if trend == "c" else "ct")
    xq = np.empty(n, dtype=float)
    zq = np.empty_like(z)
    xq[0] = x[0]
    zq[0] = z[0]
    xq[1:] = x[1:] - alpha * x[:-1]
    zq[1:] = z[1:] - alpha * z[:-1]
    psi = ols(zq, xq).beta
    return x - z @ psi, cbar


def dfgls(
    x: np.ndarray,
    *,
    trend: str = "c",
    lags: int | None = None,
    max_lags: int | None = None,
    criterion: str = "maic",
) -> UnitRootResult:
    """Elliott-Rothenberg-Stock DF-GLS test (null: a unit root).

    A GLS-detrended ADF: the deterministic terms are removed by local-to-unity
    quasi-differencing *before* the Dickey-Fuller regression, which raises power
    substantially against near-unit-root alternatives.

    Parameters
    ----------
    x : numpy.ndarray
        1-D series in time order.
    trend : {"c", "ct"}, default="c"
        ``"c"`` demeans, ``"ct"`` detrends.
    lags : int, optional
        Fixed lag order; when ``None`` it is chosen by ``criterion``.
    max_lags : int, optional
        Upper bound for lag selection.
    criterion : {"maic", "aic", "bic"}, default="maic"
        ``"maic"`` is the Ng-Perron modified AIC, which is the recommended
        selector for GLS-detrended tests.
    """
    if trend not in ("c", "ct"):
        raise ValueError(f"DF-GLS `trend` must be 'c' or 'ct', got {trend!r}.")
    arr = _clean(x)
    n = arr.shape[0]
    if n < 12:
        raise ValueError(f"DF-GLS needs at least 12 finite observations, got {n}.")
    detrended, _ = _gls_detrend(arr, trend)
    if lags is None:
        lags = _select_gls_lags(detrended, max_lags, criterion)
    X, y = _adf_design(detrended, int(lags), "n")
    res = ols(X, y)
    stat = float(res.tstat[0])
    crit = _mackinnon_crit("n", res.nobs) if trend == "c" else _ers_ct_crit(res.nobs)
    return UnitRootResult(stat, _pvalue(stat, crit), int(lags), res.nobs, crit, True)


def _select_gls_lags(
    detrended: np.ndarray, max_lags: int | None, criterion: str
) -> int:
    n = detrended.shape[0]
    upper = _default_maxlag(n) if max_lags is None else int(max_lags)
    upper = int(min(upper, max(0, (n - 6) // 2)))
    best, best_val = 0, np.inf
    for cand in range(upper + 1):
        try:
            X, y = _adf_design(detrended, cand, "n")
        except ValueError:
            break
        # Ng & Perron require every candidate to be evaluated on the common
        # sample implied by the maximum lag order, so trim the extra rows.
        drop = upper - cand
        X, y = X[drop:], y[drop:]
        if y.shape[0] <= X.shape[1]:
            break
        res = ols(X, y)
        ssr = float(res.resid @ res.resid)
        if criterion in ("aic", "bic"):
            val = _ic(ssr, res.nobs, X.shape[1], criterion)
        elif criterion == "maic":
            # Ng-Perron (2001) modified AIC: the extra `tau` term penalises the
            # bias in the sum of the autoregressive coefficients, which is what
            # makes it the right selector for a GLS-detrended test.
            sigma2 = ssr / res.nobs
            level = X[:, 0]
            tau = (
                (res.beta[0] ** 2) * float(level @ level) / sigma2
                if sigma2 > 0
                else np.inf
            )
            val = (
                np.log(sigma2) + 2.0 * (tau + cand) / res.nobs if sigma2 > 0 else np.inf
            )
        else:
            raise ValueError(f"unknown criterion {criterion!r}.")
        if val < best_val:
            best, best_val = cand, val
    return best


def ng_perron(
    x: np.ndarray, *, trend: str = "c", lags: int | None = None
) -> NgPerronResult:
    """Ng-Perron (2001) M unit-root tests on GLS-detrended data.

    Reports ``MZa``, ``MZt``, ``MSB`` and ``MPT``, which combine the modified
    Phillips-Perron statistics of Perron & Ng (1996) with ERS GLS detrending and
    an autoregressive long-run-variance estimate. They have far better size in
    the presence of a large negative MA root than ADF/PP.

    Parameters
    ----------
    x : numpy.ndarray
        1-D series in time order.
    trend : {"c", "ct"}, default="c"
    lags : int, optional
        Lag order for the autoregressive long-run variance; MAIC-selected when
        ``None``.
    """
    if trend not in ("c", "ct"):
        raise ValueError(f"Ng-Perron `trend` must be 'c' or 'ct', got {trend!r}.")
    arr = _clean(x)
    n = arr.shape[0]
    if n < 12:
        raise ValueError(f"Ng-Perron needs at least 12 finite observations, got {n}.")
    detrended, cbar = _gls_detrend(arr, trend)
    if lags is None:
        lags = _select_gls_lags(detrended, None, "maic")
    X, y = _adf_design(detrended, int(lags), "n")
    res = ols(X, y)
    sigma2 = float(res.resid @ res.resid) / res.nobs
    sum_delta = float(np.sum(res.beta[1:])) if int(lags) > 0 else 0.0
    denom = (1.0 - sum_delta) ** 2
    s2_ar = sigma2 / denom if denom > 0 else np.nan
    lagged = detrended[:-1]
    kappa = float(lagged @ lagged) / n**2
    last2 = float(detrended[-1] ** 2) / n
    if not np.isfinite(s2_ar) or s2_ar <= 0 or kappa <= 0:
        mza = mzt = msb = mpt = float("nan")
    else:
        mza = (last2 - s2_ar) / (2.0 * kappa)
        msb = float(np.sqrt(kappa / s2_ar))
        mzt = mza * msb
        if trend == "c":
            mpt = (cbar**2 * kappa - cbar * last2) / s2_ar
        else:
            mpt = (cbar**2 * kappa + (1.0 - cbar) * last2) / s2_ar
    crit = {k: dict(v) for k, v in _NG_PERRON_CRIT[trend].items()}
    return NgPerronResult(
        float(mza),
        float(mzt),
        float(msb),
        float(mpt),
        _pvalue(float(mzt), crit["mzt"]),
        int(lags),
        res.nobs,
        crit,
    )


# --------------------------------------------------------------------------- #
# Zivot-Andrews
# --------------------------------------------------------------------------- #
def zivot_andrews(
    x: np.ndarray,
    *,
    regression: str = "intercept",
    lags: int | None = None,
    max_lags: int | None = None,
    trim: float = 0.15,
) -> ZivotAndrewsResult:
    """Zivot-Andrews test for a unit root against a break-in-trend alternative.

    Sweeps every candidate break fraction in ``[trim, 1 - trim]``, runs the ADF
    regression augmented with the break dummies, and reports the **minimum**
    t-statistic together with the argmin break location. Failing to allow for a
    break is a common reason ADF spuriously fails to reject.

    Parameters
    ----------
    x : numpy.ndarray
        1-D series in time order.
    regression : {"intercept", "trend", "both"}, default="intercept"
        Model A (level shift), model B (trend slope shift), or model C (both).
    lags : int, optional
        Fixed lag order. When ``None`` it is chosen once by AIC on the no-break
        ADF regression, so the sweep stays ``O(n)`` regressions.
    max_lags : int, optional
        Upper bound for that lag selection.
    trim : float, default=0.15
        Fraction of the sample excluded at each end when searching for a break.

    Returns
    -------
    ZivotAndrewsResult
        Includes ``break_index`` (the position in the cleaned series) and
        ``break_fraction``.
    """
    if regression not in ("intercept", "trend", "both"):
        raise ValueError(
            f"`regression` must be 'intercept', 'trend' or 'both', got {regression!r}."
        )
    if not 0.0 < trim < 0.5:
        raise ValueError(f"`trim` must lie in (0, 0.5), got {trim!r}.")
    arr = _clean(x)
    n = arr.shape[0]
    if n < 20:
        raise ValueError(f"Zivot-Andrews needs at least 20 observations, got {n}.")
    if lags is None:
        lags = adf(arr, trend="ct", max_lags=max_lags, criterion="aic").lags
    lags = int(lags)

    base, y = _adf_design(arr, lags, "ct")
    rows = y.shape[0]
    offset = lags + 1  # index in `arr` of the first regression row
    t_index = np.arange(rows, dtype=float)
    lo = int(np.ceil(trim * n)) - offset
    hi = int(np.floor((1.0 - trim) * n)) - offset
    lo = max(lo, 1)
    hi = min(hi, rows - 2)
    if hi <= lo:
        raise ValueError("trimmed break window is empty; series too short.")

    best_stat, best_pos = np.inf, lo
    for pos in range(lo, hi + 1):
        du = (t_index >= pos).astype(float)
        dt = np.where(t_index >= pos, t_index - pos, 0.0)
        if regression == "intercept":
            extra = du[:, None]
        elif regression == "trend":
            extra = dt[:, None]
        else:
            extra = np.column_stack([du, dt])
        res = ols(np.column_stack([base, extra]), y)
        stat = float(res.tstat[0])
        if np.isfinite(stat) and stat < best_stat:
            best_stat, best_pos = stat, pos
    crit = dict(_ZA_CRIT[regression])
    break_index = int(best_pos + offset)
    return ZivotAndrewsResult(
        best_stat,
        _pvalue(best_stat, crit),
        break_index,
        float(break_index) / n,
        lags,
        rows,
        crit,
        regression,
    )


# --------------------------------------------------------------------------- #
# Panel surface
# --------------------------------------------------------------------------- #
_TESTS = ("adf", "kpss", "pp", "dfgls", "ng_perron", "zivot_andrews")


def _run_tests(x: np.ndarray, tests: Sequence[str], trend: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in tests:
        try:
            if name == "adf":
                r = adf(x, trend=trend)
                out["adf_stat"], out["adf_pvalue"] = r.stat, r.pvalue
                out["adf_lags"] = float(r.lags)
            elif name == "kpss":
                r = kpss(x, trend="c" if trend == "n" else trend)
                out["kpss_stat"], out["kpss_pvalue"] = r.stat, r.pvalue
            elif name == "pp":
                r = phillips_perron(x, trend=trend)
                out["pp_stat"], out["pp_pvalue"] = r.stat, r.pvalue
            elif name == "dfgls":
                r = dfgls(x, trend="c" if trend == "n" else trend)
                out["dfgls_stat"], out["dfgls_pvalue"] = r.stat, r.pvalue
            elif name == "ng_perron":
                r = ng_perron(x, trend="c" if trend == "n" else trend)
                out["ng_mza"], out["ng_mzt"] = r.mza, r.mzt
                out["ng_msb"], out["ng_mpt"] = r.msb, r.mpt
                out["ng_pvalue"] = r.pvalue_mzt
            elif name == "zivot_andrews":
                r = zivot_andrews(x)
                out["za_stat"], out["za_pvalue"] = r.stat, r.pvalue
                out["za_break_index"] = float(r.break_index)
                out["za_break_fraction"] = r.break_fraction
            else:
                raise ValueError(
                    f"unknown test {name!r}; choose from {sorted(_TESTS)}."
                )
        except ValueError as exc:
            if "unknown test" in str(exc):
                raise
            # Series too short / degenerate for this test: emit nulls rather
            # than failing the whole panel.
            for key in _keys_for(name):
                out.setdefault(key, float("nan"))
    return out


def _keys_for(name: str) -> tuple[str, ...]:
    return {
        "adf": ("adf_stat", "adf_pvalue", "adf_lags"),
        "kpss": ("kpss_stat", "kpss_pvalue"),
        "pp": ("pp_stat", "pp_pvalue"),
        "dfgls": ("dfgls_stat", "dfgls_pvalue"),
        "ng_perron": ("ng_mza", "ng_mzt", "ng_msb", "ng_mpt", "ng_pvalue"),
        "zivot_andrews": (
            "za_stat",
            "za_pvalue",
            "za_break_index",
            "za_break_fraction",
        ),
    }[name]


def unit_root_table(
    df: pl.DataFrame | pl.LazyFrame,
    column: str,
    *,
    entity: str,
    time: str,
    tests: Sequence[str] = ("adf", "kpss"),
    trend: str = "c",
) -> pl.DataFrame:
    """Run the unit-root battery on each entity's **full** series.

    One row per entity with the requested statistics. This is a *fitted*,
    whole-sample quantity: use it on training rows only (or inside
    :class:`StationarityDifferencer`), never on a panel that spans a fold
    boundary. For a leak-safe per-row feature use
    :func:`rolling_unit_root_features` instead.
    """
    return per_entity_reduce(
        df,
        entity=entity,
        time=time,
        columns=[column],
        func=lambda data: _run_tests(data[column], tests, trend),
    )


def rolling_unit_root_features(
    df: pl.DataFrame | pl.LazyFrame,
    column: str,
    *,
    entity: str,
    time: str,
    window: int = 252,
    tests: Sequence[str] = ("adf",),
    trend: str = "c",
    min_periods: int | None = None,
    prefix: str | None = None,
) -> pl.DataFrame:
    """Trailing-window unit-root statistics as per-row panel features.

    Each row ``t`` gets the statistics computed on ``x[t - window + 1 : t + 1]``
    of its own entity, so the features are causal and invariant to appended
    future rows -- the leak-safe way to use a unit-root test as a persistence /
    regime signal.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel.
    column : str
        Column to test.
    entity, time : str
        Panel keys.
    window : int, default=252
        Trailing window length.
    tests : sequence of str, default=("adf",)
        Any of ``"adf"``, ``"kpss"``, ``"pp"``, ``"dfgls"``, ``"ng_perron"``,
        ``"zivot_andrews"``. Each test costs one regression sweep per row, so
        keep the list short on large panels.
    min_periods : int, optional
        Minimum observations before a value is emitted (defaults to ``window``).
    prefix : str, optional
        Prefix for the emitted columns (defaults to ``f"{column}_"``).

    Returns
    -------
    polars.DataFrame
        ``df`` sorted by ``(entity, time)`` with the feature columns appended.
    """
    pfx = f"{column}_" if prefix is None else prefix
    keys: list[str] = []
    for name in tests:
        keys.extend(_keys_for(name))

    def kernel(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        x = data[column]
        out = {f"{pfx}{k}": np.full(x.shape[0], np.nan) for k in keys}
        for key in keys:
            col = f"{pfx}{key}"
            out[col] = rolling_apply(
                x,
                window,
                lambda chunk, _k=key: _run_tests(chunk, tests, trend).get(
                    _k, float("nan")
                ),
                min_periods=min_periods,
            )
        return out

    return per_entity_apply(df, entity=entity, time=time, columns=[column], func=kernel)


# --------------------------------------------------------------------------- #
# Train-only differencing choice
# --------------------------------------------------------------------------- #
class StationarityDifferencer(PanelTransformer):
    """Choose a differencing order per entity on the training rows, then freeze it.

    ``fit`` runs an ADF (and optionally a confirmatory KPSS) on each entity's
    training series and records the smallest differencing order ``0 .. max_order``
    that looks stationary. ``transform`` applies those *frozen* orders with
    ``diff().over(entity)``. Because the order is learned once on train and never
    re-estimated, no test-fold information can reach the transform -- which is
    exactly the leak the "difference until stationary" recipe usually introduces.

    Parameters
    ----------
    columns : sequence of str
        Columns to difference.
    max_order : int, default=1
        Largest differencing order considered.
    alpha : float, default=0.05
        Significance level for the ADF rejection (and the KPSS non-rejection).
    trend : {"n", "c", "ct"}, default="c"
        Deterministic terms in the test regressions.
    confirm_with_kpss : bool, default=True
        Require KPSS to *fail* to reject stationarity as well as ADF rejecting a
        unit root. More conservative, fewer spurious "already stationary" calls.
    suffix : str, optional
        When given, the differenced series is written to ``f"{col}{suffix}"``
        instead of replacing ``col``.
    entity, time : str, optional
        Default panel keys for bare polars frames.

    Attributes
    ----------
    orders_ : dict
        ``{(entity, column): order}`` learned at fit time.
    default_order_ : dict
        ``{column: order}`` fallback (the modal fitted order) applied to entities
        unseen during ``fit``.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_features.econ.features import StationarityDifferencer
    >>> tr = StationarityDifferencer(["px"], entity="id", time="t").fit(train)
    ... # doctest: +SKIP
    >>> tr.transform(test).collect()  # doctest: +SKIP
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        columns: Sequence[str],
        *,
        max_order: int = 1,
        alpha: float = 0.05,
        trend: str = "c",
        confirm_with_kpss: bool = True,
        suffix: str | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self.columns = list(columns)
        if not self.columns:
            raise ValueError("`columns` must name at least one column.")
        if max_order < 0:
            raise ValueError(f"`max_order` must be >= 0, got {max_order!r}.")
        self.max_order = int(max_order)
        self.alpha = float(alpha)
        self.trend = trend
        self.confirm_with_kpss = bool(confirm_with_kpss)
        self.suffix = suffix
        self.orders_: dict[tuple[object, str], int] = {}
        self.default_order_: dict[str, int] = {}

    def _is_stationary(self, x: np.ndarray) -> bool:
        try:
            adf_res = adf(x, trend=self.trend)
        except ValueError:
            return False
        rejects_unit_root = adf_res.pvalue < self.alpha
        if not self.confirm_with_kpss:
            return bool(rejects_unit_root)
        try:
            kpss_res = kpss(x, trend="c" if self.trend == "n" else self.trend)
        except ValueError:
            return bool(rejects_unit_root)
        return bool(rejects_unit_root and kpss_res.pvalue > self.alpha)

    def _choose_order(self, x: np.ndarray) -> int:
        series = np.asarray(x, dtype=float)
        for order in range(self.max_order + 1):
            candidate = np.diff(series, n=order) if order else series
            finite = candidate[np.isfinite(candidate)]
            if finite.shape[0] < 8:
                return order
            if self._is_stationary(finite):
                return order
        return self.max_order

    def _fit(self, panel: PanelFrame) -> None:
        frame = sorted_panel(panel.lazy(), panel.entity_col, panel.time_col)
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise ValueError(
                f"column(s) {missing} not found in panel; available: {frame.columns}."
            )
        self.orders_ = {}
        per_column: dict[str, list[int]] = {c: [] for c in self.columns}
        from polars_features.econ.features._common import entity_arrays

        for key, _idx, data in entity_arrays(frame, panel.entity_col, self.columns):
            for col in self.columns:
                order = self._choose_order(data[col])
                self.orders_[(key, col)] = order
                per_column[col].append(order)
        for col, orders in per_column.items():
            if orders:
                counts = np.bincount(np.asarray(orders, dtype=int))
                self.default_order_[col] = int(np.argmax(counts))
            else:
                self.default_order_[col] = 0

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        ent = panel.entity_col
        exprs = []
        for col in self.columns:
            name = f"{col}{self.suffix}" if self.suffix else col
            default = self.default_order_.get(col, 0)
            expr = pl.col(col).cast(pl.Float64)
            # Branch on the frozen per-entity order with a chained when/then so
            # the whole thing stays a single lazy expression.
            orders = {
                key[0]: order for key, order in self.orders_.items() if key[1] == col
            }
            distinct = sorted({*orders.values(), default})
            if len(distinct) == 1:
                order = distinct[0]
                exprs.append(_diff_expr(expr, order, ent).alias(name))
                continue
            branches = None
            for order in distinct:
                members = [k for k, v in orders.items() if v == order]
                if not members:
                    continue
                cond = pl.col(ent).is_in(members)
                value = _diff_expr(expr, order, ent)
                branches = (
                    pl.when(cond).then(value)
                    if branches is None
                    else branches.when(cond).then(value)
                )
            fallback = _diff_expr(expr, default, ent)
            exprs.append(
                (fallback if branches is None else branches.otherwise(fallback)).alias(
                    name
                )
            )
        return panel.with_columns(exprs)


def _diff_expr(expr: pl.Expr, order: int, entity: str) -> pl.Expr:
    out = expr
    for _ in range(int(order)):
        out = out.diff().over(entity)
    return out
