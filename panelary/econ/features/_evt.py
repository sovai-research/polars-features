"""Extreme-value tail features: Hill index, POT-GPD fit, EVT VaR and expected shortfall.

Tail risk is exactly where a Gaussian-moment feature is least informative, and
exactly where a rolling window has enough exceedances to say something. All the
estimators here are fitted on a **trailing** window of one entity's own history,
so a row's tail features never see a future observation.

* :func:`hill_index` -- Hill's (1975) estimator of the tail index ``alpha``; its
  reciprocal ``xi = 1 / alpha`` is the Pareto shape.
* :func:`gpd_fit` -- generalised-Pareto ``(xi, sigma)`` fitted to peaks over a
  threshold by **probability-weighted moments** (Hosking & Wallis 1987). PWM is
  closed-form, needs no optimiser and no ``scipy``, and is markedly more stable
  than MLE at the sample sizes a rolling window offers.
* :func:`pot_var_es` -- the POT tail-quantile formulas for value-at-risk and
  expected shortfall.
* :func:`evt_features` -- all of the above as trailing panel columns.

References
----------
Hill (1975); Hosking & Wallis (1987), *Technometrics*; Pickands (1975);
McNeil & Frey (2000) for the POT VaR/ES formulas.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import polars as pl

from panelary.econ.features._common import per_entity_apply

__all__ = [
    "GPDFit",
    "hill_index",
    "gpd_fit",
    "pot_var_es",
    "evt_features",
]


class GPDFit(NamedTuple):
    """Generalised-Pareto fit to peaks over a threshold.

    Attributes
    ----------
    xi : float
        Shape. ``xi > 0`` is a heavy (Pareto-type) tail, ``xi = 0`` exponential,
        ``xi < 0`` a bounded tail.
    sigma : float
        Scale.
    threshold : float
        The threshold ``u`` the exceedances were measured from.
    n_exceed : int
        Number of exceedances used.
    nobs : int
        Size of the sample the threshold was taken from.
    """

    xi: float
    sigma: float
    threshold: float
    n_exceed: int
    nobs: int


def _tail_sample(x: np.ndarray, tail: str) -> np.ndarray:
    """Orient the sample so the tail of interest is the **upper** tail."""
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if tail == "upper":
        return arr
    if tail == "lower":
        return -arr
    raise ValueError(f"`tail` must be 'upper' or 'lower', got {tail!r}.")


def hill_index(x: np.ndarray, *, k: int | None = None, tail: str = "lower") -> float:
    """Hill tail index ``alpha`` of the ``k`` largest observations.

    ``1 / alpha = (1 / k) * sum_{i=1..k} log X_(i) - log X_(k+1)`` on the order
    statistics sorted descending; the returned value is ``alpha``, the Pareto
    tail exponent (larger means a thinner tail). Only strictly positive
    exceedances contribute, so the sample is shifted to be positive if needed.

    Parameters
    ----------
    x : numpy.ndarray
        Sample (e.g. returns) in any order.
    k : int, optional
        Number of order statistics in the tail; defaults to
        ``max(10, floor(0.1 * n))``.
    tail : {"lower", "upper"}, default="lower"
        Which tail to measure. ``"lower"`` (losses) is the default for returns.

    Returns
    -------
    float
        The tail index ``alpha``, or ``nan`` when the tail is degenerate.
    """
    arr = _tail_sample(x, tail)
    n = arr.shape[0]
    if n < 12:
        return float("nan")
    kk = int(k) if k is not None else max(10, int(np.floor(0.1 * n)))
    kk = int(min(kk, n - 1))
    if kk < 2:
        return float("nan")
    order = np.sort(arr)[::-1]
    top = order[: kk + 1]
    # Shift into the positive orthant if the tail straddles zero, otherwise the
    # logs are undefined. The Hill estimator is not shift invariant, so this is a
    # documented fallback rather than the intended usage.
    if top[-1] <= 0:
        shift = -top[-1] + 1e-12 + np.abs(top[-1]) * 1e-6
        top = top + shift
    with np.errstate(invalid="ignore", divide="ignore"):
        xi = float(np.mean(np.log(top[:kk]) - np.log(top[kk])))
    if not np.isfinite(xi) or xi <= 0:
        return float("nan")
    return float(1.0 / xi)


def gpd_fit(
    x: np.ndarray,
    *,
    threshold: float | None = None,
    threshold_quantile: float = 0.90,
    tail: str = "lower",
) -> GPDFit:
    """Fit a generalised Pareto distribution to peaks over a threshold (PWM).

    Uses Hosking & Wallis's probability-weighted-moment estimator on the
    exceedances ``y = x - u``:

    ``xi = 2 - a0 / (a0 - 2 * a1)``, ``sigma = 2 * a0 * a1 / (a0 - 2 * a1)``,
    with ``a_r = mean_i (1 - p_i)**r * y_(i)`` and ``p_i = (i - 0.35) / n``.

    Validity range
    --------------
    PWM is consistent and efficient for the moderately-heavy tails that dominate
    financial data (``xi < 0.5``), which is the regime it is used in here. It is
    *not* a general-purpose fitter: its shape estimate is bounded above by 1 by
    construction, so a genuinely infinite-mean tail (``xi > 1``, e.g. a Pareto
    with ``alpha < 1``) saturates near 1 rather than being reported honestly. For
    such tails read :func:`hill_index`, which has no such bound.

    Parameters
    ----------
    x : numpy.ndarray
        Sample.
    threshold : float, optional
        Explicit threshold, expressed in the **tail-oriented** sample (for
        ``tail="lower"`` the sample is negated first). When ``None`` the
        ``threshold_quantile`` empirical quantile is used.
    threshold_quantile : float, default=0.90
        Quantile defining the threshold when ``threshold`` is ``None``.
    tail : {"lower", "upper"}, default="lower"

    Returns
    -------
    GPDFit
    """
    arr = _tail_sample(x, tail)
    n = arr.shape[0]
    if n < 20:
        return GPDFit(float("nan"), float("nan"), float("nan"), 0, n)
    u = (
        float(threshold)
        if threshold is not None
        else float(np.quantile(arr, threshold_quantile))
    )
    excess = np.sort(arr[arr > u] - u)
    ne = excess.shape[0]
    if ne < 8:
        return GPDFit(float("nan"), float("nan"), u, ne, n)
    p = (np.arange(1, ne + 1) - 0.35) / ne
    a0 = float(np.mean(excess))
    a1 = float(np.mean((1.0 - p) * excess))
    denom = a0 - 2.0 * a1
    if not np.isfinite(denom) or abs(denom) < 1e-15:
        return GPDFit(float("nan"), float("nan"), u, ne, n)
    xi = 2.0 - a0 / denom
    sigma = 2.0 * a0 * a1 / denom
    if not np.isfinite(sigma) or sigma <= 0:
        return GPDFit(float("nan"), float("nan"), u, ne, n)
    return GPDFit(float(xi), float(sigma), u, ne, n)


def pot_var_es(
    x: np.ndarray,
    *,
    q: float = 0.99,
    threshold_quantile: float = 0.90,
    tail: str = "lower",
) -> tuple[float, float, GPDFit]:
    """Peaks-over-threshold value-at-risk and expected shortfall.

    ``VaR_q = u + (sigma / xi) * ((n / N_u * (1 - q)) ** -xi - 1)`` and
    ``ES_q = (VaR_q + sigma - xi * u) / (1 - xi)`` for ``xi < 1``
    (McNeil & Frey 2000). Both are returned as **positive loss magnitudes** when
    ``tail="lower"``, which is the usual reporting convention for returns.

    Parameters
    ----------
    x : numpy.ndarray
        Sample of returns.
    q : float, default=0.99
        Confidence level.
    threshold_quantile : float, default=0.90
        Threshold quantile for the GPD fit.
    tail : {"lower", "upper"}, default="lower"

    Returns
    -------
    (float, float, GPDFit)
        ``(VaR, ES, fit)``. ``ES`` is ``nan`` when ``xi >= 1`` (infinite mean).
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"`q` must lie in (0, 1), got {q!r}.")
    fit = gpd_fit(x, threshold_quantile=threshold_quantile, tail=tail)
    if not np.isfinite(fit.xi) or fit.n_exceed == 0:
        return float("nan"), float("nan"), fit
    ratio = fit.nobs / fit.n_exceed * (1.0 - q)
    if ratio <= 0:
        return float("nan"), float("nan"), fit
    if abs(fit.xi) < 1e-8:
        var = fit.threshold - fit.sigma * np.log(ratio)
    else:
        var = fit.threshold + (fit.sigma / fit.xi) * (ratio ** (-fit.xi) - 1.0)
    es = (
        (var + fit.sigma - fit.xi * fit.threshold) / (1.0 - fit.xi)
        if fit.xi < 1.0
        else float("nan")
    )
    return float(var), float(es), fit


def evt_features(
    df: pl.DataFrame | pl.LazyFrame,
    column: str,
    *,
    entity: str,
    time: str,
    window: int = 252,
    q: float = 0.99,
    threshold_quantile: float = 0.90,
    hill_k: int | None = None,
    tail: str = "lower",
    min_periods: int | None = None,
    prefix: str | None = None,
) -> pl.DataFrame:
    """Trailing-window EVT tail features for a long panel.

    For every row, fits the tail on ``x[t - window + 1 : t + 1]`` of that entity
    and emits ``hill_alpha``, ``gpd_xi``, ``gpd_sigma``, ``evt_var`` and
    ``evt_es``. Strictly causal: appending future rows cannot change any earlier
    row.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel.
    column : str
        Returns column.
    entity, time : str
        Panel keys.
    window : int, default=252
        Trailing window length.
    q : float, default=0.99
        VaR/ES confidence level.
    threshold_quantile : float, default=0.90
        POT threshold quantile inside each window.
    hill_k : int, optional
        Number of order statistics for the Hill estimator.
    tail : {"lower", "upper"}, default="lower"
        Which tail to model (``"lower"`` = losses).
    min_periods : int, optional
        Minimum observations before a value is emitted (defaults to ``window``).
    prefix : str, optional
        Prefix for emitted columns (defaults to ``f"{column}_"``).

    Returns
    -------
    polars.DataFrame
        ``df`` sorted by ``(entity, time)`` with the feature columns appended.
    """
    pfx = f"{column}_" if prefix is None else prefix

    def kernel(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        x = data[column]
        n = x.shape[0]
        names = (
            "hill_alpha",
            "gpd_xi",
            "gpd_sigma",
            "evt_var",
            "evt_es",
            "evt_n_exceed",
        )
        out = {f"{pfx}{name}": np.full(n, np.nan) for name in names}
        mp = window if min_periods is None else int(min_periods)
        # One pass: each window is fitted ONCE and all five features read off it.
        for t in range(n):
            chunk = x[max(0, t - window + 1) : t + 1]
            finite = chunk[np.isfinite(chunk)]
            if finite.shape[0] < mp:
                continue
            out[f"{pfx}hill_alpha"][t] = hill_index(chunk, k=hill_k, tail=tail)
            var, es, fit = pot_var_es(
                chunk, q=q, threshold_quantile=threshold_quantile, tail=tail
            )
            out[f"{pfx}gpd_xi"][t] = fit.xi
            out[f"{pfx}gpd_sigma"][t] = fit.sigma
            out[f"{pfx}evt_var"][t] = var
            out[f"{pfx}evt_es"][t] = es
            out[f"{pfx}evt_n_exceed"][t] = float(fit.n_exceed)
        return out

    return per_entity_apply(df, entity=entity, time=time, columns=[column], func=kernel)
