"""**Causal** seasonal-trend decomposition and seasonal-strength features.

A textbook STL (or any classical decomposition using a centred moving average)
is **two-sided**: the trend at time ``t`` is an average of observations on both
sides of ``t``, so every "trend", "seasonal" and "remainder" value silently
embeds the future. Feeding that into a backtest is one of the most common
look-ahead leaks in time-series feature engineering, and it is why this module
does *not* wrap a classical STL.

What is implemented instead is a trailing-window seasonal-trend decomposition:

* **trend** ``T_t`` -- a trailing moving average over the last ``trend_window``
  observations (ending at ``t``). It lags a centred trend by roughly half a
  window; that lag is the price of causality and is deliberate.
* **seasonal** ``S_t`` -- the mean of the detrended series over the *same
  seasonal phase* at times ``t, t - period, t - 2 * period, ...``, restricted to
  the most recent ``seasonal_cycles`` cycles and then re-centred so the phases
  sum to zero across the trailing cycle.
* **remainder** ``R_t = x_t - T_t - S_t``.
* **strengths** -- Wang-Smith-Hyndman trend and seasonal strength,
  ``max(0, 1 - Var(R) / Var(T + R))`` and ``max(0, 1 - Var(R) / Var(S + R))``,
  measured over a trailing window.

Each of those touches only rows ``<= t``. The decomposition is refined by a
small number of inner iterations (trend, then seasonal, then trend again), the
same alternating idea STL uses, minus the two-sided smoothers.

This is a deliberate substitute for the ``augurs`` MSTL suggested in the plan:
that is a Rust crate, and PanelKit has been pure-Python since 0.4.0.

References
----------
Cleveland et al. (1990) for STL itself (as the *non*-causal reference);
Wang, Smith & Hyndman (2006) for the strength measures.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from polars_features.core.panel_frame import PanelFrame
from polars_features.core.protocol import PanelTransformer
from polars_features.econ.features._common import per_entity_apply

__all__ = [
    "causal_seasonal_decompose",
    "seasonal_strength",
    "decompose_features",
    "CausalSeasonalDecomposer",
]


def _trailing_mean(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Trailing mean over finite values, right-aligned at each row."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    w = int(window)
    filled = np.where(np.isfinite(x), x, 0.0)
    valid = np.isfinite(x).astype(float)
    csum = np.concatenate([[0.0], np.cumsum(filled)])
    cval = np.concatenate([[0.0], np.cumsum(valid)])
    idx = np.arange(n)
    lo = np.maximum(idx - w + 1, 0)
    sums = csum[idx + 1] - csum[lo]
    counts = cval[idx + 1] - cval[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(counts >= min_periods, sums / np.maximum(counts, 1), np.nan)
    return out


def _phase_trailing_mean(x: np.ndarray, period: int, cycles: int | None) -> np.ndarray:
    """Trailing mean of ``x`` over the same seasonal phase.

    ``out[t] = mean(x[t], x[t - period], ..., x[t - (cycles - 1) * period])``
    over whatever of those indices exist and are finite. Every contributing index
    is ``<= t``, so this is causal.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=float)
    p = int(period)
    for phase in range(p):
        idx = np.arange(phase, n, p)
        if idx.shape[0] == 0:
            continue
        vals = x[idx]
        finite = np.isfinite(vals)
        filled = np.where(finite, vals, 0.0)
        csum = np.concatenate([[0.0], np.cumsum(filled)])
        cval = np.concatenate([[0.0], np.cumsum(finite.astype(float))])
        k = np.arange(idx.shape[0])
        lo = np.maximum(k - cycles + 1, 0) if cycles else np.zeros_like(k)
        sums = csum[k + 1] - csum[lo]
        counts = cval[k + 1] - cval[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[idx] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return out


def causal_seasonal_decompose(
    x: np.ndarray,
    *,
    period: int,
    trend_window: int | None = None,
    seasonal_cycles: int | None = None,
    n_iter: int = 2,
) -> dict[str, np.ndarray]:
    """Trailing-window trend / seasonal / remainder decomposition.

    Parameters
    ----------
    x : numpy.ndarray
        1-D series in time order.
    period : int
        Seasonal period (e.g. 12 for monthly-in-year, 5 for weekday effects).
    trend_window : int, optional
        Length of the trailing trend average; defaults to ``2 * period + 1``.
    seasonal_cycles : int, optional
        How many past cycles the seasonal average may use. ``None`` uses every
        past cycle (an expanding phase mean).
    n_iter : int, default=2
        Inner alternating iterations (trend then seasonal). Two is usually
        enough; one is the plain "trailing MA then phase mean" recipe.

    Returns
    -------
    dict of numpy.ndarray
        ``{"trend", "seasonal", "remainder", "detrended"}``, all the same length
        as ``x``.
    """
    arr = np.asarray(x, dtype=float).ravel()
    n = arr.shape[0]
    p = int(period)
    if p < 2:
        raise ValueError(f"`period` must be >= 2, got {period!r}.")
    if n_iter < 1:
        raise ValueError(f"`n_iter` must be >= 1, got {n_iter!r}.")
    tw = int(trend_window) if trend_window is not None else 2 * p + 1
    min_trend = max(2, min(p, tw))

    seasonal = np.zeros(n, dtype=float)
    trend = np.full(n, np.nan, dtype=float)
    for _ in range(int(n_iter)):
        deseasonalised = arr - np.nan_to_num(seasonal, nan=0.0)
        trend = _trailing_mean(deseasonalised, tw, min_trend)
        detrended = arr - trend
        raw_seasonal = _phase_trailing_mean(detrended, p, seasonal_cycles)
        # Re-centre so the seasonal component has (trailing) zero mean and does
        # not absorb level -- the trailing mean keeps this causal.
        centre = _trailing_mean(raw_seasonal, p, 1)
        seasonal = raw_seasonal - np.nan_to_num(centre, nan=0.0)
    detrended = arr - trend
    remainder = detrended - seasonal
    return {
        "trend": trend,
        "seasonal": seasonal,
        "remainder": remainder,
        "detrended": detrended,
    }


def _trailing_var(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    mean = _trailing_mean(x, window, min_periods)
    mean_sq = _trailing_mean(np.asarray(x, dtype=float) ** 2, window, min_periods)
    return np.maximum(mean_sq - mean**2, 0.0)


def seasonal_strength(
    parts: dict[str, np.ndarray], *, window: int, min_periods: int | None = None
) -> dict[str, np.ndarray]:
    """Trailing Wang-Smith-Hyndman trend / seasonal strength from a decomposition.

    ``strength_trend = max(0, 1 - Var(R) / Var(T + R))`` and
    ``strength_seasonal = max(0, 1 - Var(R) / Var(S + R))``, with every variance
    taken over the trailing ``window``. Values near 1 mean the component
    dominates the remainder.
    """
    mp = window if min_periods is None else int(min_periods)
    trend = parts["trend"]
    seasonal = parts["seasonal"]
    remainder = parts["remainder"]
    var_r = _trailing_var(remainder, window, mp)
    var_tr = _trailing_var(trend + remainder, window, mp)
    var_sr = _trailing_var(seasonal + remainder, window, mp)
    with np.errstate(invalid="ignore", divide="ignore"):
        st = np.where(var_tr > 0, 1.0 - var_r / var_tr, np.nan)
        ss = np.where(var_sr > 0, 1.0 - var_r / var_sr, np.nan)
    return {
        "strength_trend": np.clip(st, 0.0, 1.0),
        "strength_seasonal": np.clip(ss, 0.0, 1.0),
    }


def decompose_features(
    df: pl.DataFrame | pl.LazyFrame,
    column: str,
    *,
    entity: str,
    time: str,
    period: int,
    trend_window: int | None = None,
    seasonal_cycles: int | None = None,
    n_iter: int = 2,
    strength_window: int | None = None,
    prefix: str | None = None,
) -> pl.DataFrame:
    """Causal decomposition columns for a long panel.

    Emits ``{prefix}trend``, ``{prefix}seasonal``, ``{prefix}remainder`` and,
    when ``strength_window`` is given, ``{prefix}strength_trend`` /
    ``{prefix}strength_seasonal``. Every value uses only rows ``<= t`` within the
    same entity.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel.
    column : str
        Column to decompose.
    entity, time : str
        Panel keys.
    period : int
        Seasonal period.
    trend_window, seasonal_cycles, n_iter
        Passed to :func:`causal_seasonal_decompose`.
    strength_window : int, optional
        Window for the strength features; defaults to ``3 * period``.
    prefix : str, optional
        Prefix for emitted columns (defaults to ``f"{column}_"``).

    Returns
    -------
    polars.DataFrame
    """
    pfx = f"{column}_" if prefix is None else prefix
    sw = int(strength_window) if strength_window is not None else 3 * int(period)

    def kernel(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        parts = causal_seasonal_decompose(
            data[column],
            period=period,
            trend_window=trend_window,
            seasonal_cycles=seasonal_cycles,
            n_iter=n_iter,
        )
        out = {
            f"{pfx}trend": parts["trend"],
            f"{pfx}seasonal": parts["seasonal"],
            f"{pfx}remainder": parts["remainder"],
        }
        if sw:
            for name, values in seasonal_strength(parts, window=sw).items():
                out[f"{pfx}{name}"] = values
        return out

    return per_entity_apply(df, entity=entity, time=time, columns=[column], func=kernel)


class CausalSeasonalDecomposer(PanelTransformer):
    """Pipeline step wrapping :func:`decompose_features`.

    The decomposition has no fitted parameters -- every value is a trailing
    statistic of the row's own entity history -- so ``fit`` only validates the
    requested columns and ``transform`` does the work. It is declared
    ``leakage_safe = True`` because the underlying smoother is one-sided; a
    classical two-sided STL would not be, which is precisely why it is not
    offered here.

    Parameters
    ----------
    columns : sequence of str
        Columns to decompose.
    period : int
        Seasonal period.
    trend_window, seasonal_cycles, n_iter, strength_window
        Passed through to :func:`decompose_features`.
    entity, time : str, optional
        Default panel keys for bare polars frames.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        columns: Sequence[str],
        *,
        period: int,
        trend_window: int | None = None,
        seasonal_cycles: int | None = None,
        n_iter: int = 2,
        strength_window: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self.columns = list(columns)
        if not self.columns:
            raise ValueError("`columns` must name at least one column.")
        self.period = int(period)
        self.trend_window = trend_window
        self.seasonal_cycles = seasonal_cycles
        self.n_iter = int(n_iter)
        self.strength_window = strength_window
        self.feature_names_out_: list[str] = []

    def _fit(self, panel: PanelFrame) -> None:
        missing = [c for c in self.columns if c not in panel]
        if missing:
            raise ValueError(
                f"column(s) {missing} not found in panel; available: {panel.columns}."
            )
        names: list[str] = []
        for col in self.columns:
            names.extend([f"{col}_trend", f"{col}_seasonal", f"{col}_remainder"])
            names.extend([f"{col}_strength_trend", f"{col}_strength_seasonal"])
        self.feature_names_out_ = names

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        frame = panel.collect()
        for col in self.columns:
            frame = decompose_features(
                frame,
                col,
                entity=panel.entity_col,
                time=panel.time_col,
                period=self.period,
                trend_window=self.trend_window,
                seasonal_cycles=self.seasonal_cycles,
                n_iter=self.n_iter,
                strength_window=self.strength_window,
            )
        return PanelFrame(frame.lazy(), entity=panel.entity_col, time=panel.time_col)
