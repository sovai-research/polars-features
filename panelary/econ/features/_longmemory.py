"""Long-memory estimation: GPH and Robinson local-Whittle, wired into frac-diff.

The fixed-width fractional-differencing kernel in :mod:`panelary._ffd`
needs an order ``d``. Hard-coding one ``d`` for a whole panel is both arbitrary
and, if it were tuned by looking at the whole sample, a leak. This module
estimates ``d`` *from the data*:

* :func:`gph` -- Geweke & Porter-Hudak (1983) log-periodogram regression.
* :func:`local_whittle` -- Robinson (1995) Gaussian semiparametric estimator
  (lower asymptotic variance than GPH; the default).
* :func:`estimate_fractional_order` -- picks the estimator, and handles the
  non-stationary ``d >= 0.5`` region by differencing once and adding one back.
* :class:`AutoFracDiff` -- a :class:`~panelary.core.protocol.PanelTransformer`
  that estimates ``d`` **per entity on the training rows only**, freezes the
  resulting weight kernel, and applies the causal frac-diff filter at transform
  time.

References
----------
Geweke & Porter-Hudak (1983), *JTSA*; Robinson (1995), *Annals of Statistics*;
Hurvich, Deo & Brodsky (1998) for the bandwidth rule; de Prado (2018) Ch. 5 for
the fixed-width frac-diff filter itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import polars as pl

from panelary._ffd import DEFAULT_THRESHOLD, ffd_weights, frac_diff_expr
from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer
from panelary.econ.features._common import (
    entity_arrays,
    norm_sf,
    ols,
    per_entity_apply,
    per_entity_reduce,
    rolling_apply,
    sorted_panel,
)

__all__ = [
    "LongMemoryResult",
    "gph",
    "local_whittle",
    "estimate_fractional_order",
    "long_memory_table",
    "rolling_long_memory_features",
    "AutoFracDiff",
]


class LongMemoryResult(NamedTuple):
    """Estimated fractional-integration order.

    Attributes
    ----------
    d : float
        The estimate of the fractional-integration order.
    se : float
        Asymptotic standard error.
    tstat : float
        ``d / se`` (test of ``H0: d = 0``, i.e. no long memory).
    pvalue : float
        Two-sided normal p-value for that test.
    bandwidth : int
        Number of periodogram ordinates used.
    nobs : int
        Length of the series the estimate was computed on.
    method : str
        ``"gph"`` or ``"local_whittle"``.
    differenced : bool
        ``True`` if the series was differenced once before estimation and ``1``
        added back to ``d``.
    """

    d: float
    se: float
    tstat: float
    pvalue: float
    bandwidth: int
    nobs: int
    method: str
    differenced: bool


def _periodogram(x: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Fourier frequencies ``lambda_j`` and periodogram ``I(lambda_j)``, ``j = 1..m``."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    resid = x - x.mean()
    fft = np.fft.rfft(resid)
    inten = (np.abs(fft) ** 2) / (2.0 * np.pi * n)
    upper = min(m, inten.shape[0] - 1)
    j = np.arange(1, upper + 1, dtype=float)
    lam = 2.0 * np.pi * j / n
    return lam, inten[1 : upper + 1]


def _bandwidth(nobs: int, m: int | None, exponent: float) -> int:
    if m is not None:
        return int(max(2, min(int(m), nobs // 2)))
    return int(max(2, min(int(np.floor(nobs**exponent)), nobs // 2)))


def _finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def gph(
    x: np.ndarray,
    *,
    bandwidth: int | None = None,
    bandwidth_exponent: float = 0.5,
) -> LongMemoryResult:
    """Geweke-Porter-Hudak log-periodogram estimator of ``d``.

    Regresses ``log I(lambda_j)`` on ``-log(4 * sin(lambda_j / 2) ** 2)`` over the
    ``m`` lowest Fourier frequencies; the slope estimates ``d``. The reported
    standard error is the asymptotic ``pi / sqrt(24 * m)``.

    Parameters
    ----------
    x : numpy.ndarray
        1-D series in time order.
    bandwidth : int, optional
        Number of periodogram ordinates ``m``. Defaults to
        ``floor(n ** bandwidth_exponent)``.
    bandwidth_exponent : float, default=0.5
        Exponent of the default bandwidth rule.

    Returns
    -------
    LongMemoryResult
    """
    arr = _finite(x)
    n = arr.shape[0]
    if n < 16:
        raise ValueError(f"GPH needs at least 16 finite observations, got {n}.")
    m = _bandwidth(n, bandwidth, bandwidth_exponent)
    lam, inten = _periodogram(arr, m)
    good = inten > 0
    lam, inten = lam[good], inten[good]
    if lam.shape[0] < 3:
        raise ValueError("GPH: too few positive periodogram ordinates.")
    regressor = -np.log(4.0 * np.sin(lam / 2.0) ** 2)
    X = np.column_stack([np.ones_like(regressor), regressor])
    res = ols(X, np.log(inten))
    d = float(res.beta[1])
    se = float(np.pi / np.sqrt(24.0 * lam.shape[0]))
    t = d / se if se > 0 else np.nan
    return LongMemoryResult(
        d,
        se,
        float(t),
        float(2.0 * norm_sf(abs(t))),
        int(lam.shape[0]),
        n,
        "gph",
        False,
    )


def _whittle_objective(d: float, lam: np.ndarray, inten: np.ndarray) -> float:
    scaled = lam ** (2.0 * d) * inten
    mean = float(np.mean(scaled))
    if not np.isfinite(mean) or mean <= 0:
        return np.inf
    return float(np.log(mean) - 2.0 * d * float(np.mean(np.log(lam))))


def local_whittle(
    x: np.ndarray,
    *,
    bandwidth: int | None = None,
    bandwidth_exponent: float = 0.5,
    bounds: tuple[float, float] = (-0.5, 1.5),
) -> LongMemoryResult:
    """Robinson's local-Whittle (Gaussian semiparametric) estimator of ``d``.

    Minimises ``R(d) = log(mean_j lambda_j**(2d) I_j) - 2d * mean_j log lambda_j``
    over the ``m`` lowest Fourier frequencies. More efficient than
    :func:`gph`; asymptotic standard error ``1 / (2 * sqrt(m))``.

    The 1-D objective is minimised by a coarse grid scan followed by golden-section
    refinement, which is robust and needs no optimiser dependency.

    Parameters
    ----------
    x : numpy.ndarray
        1-D series in time order.
    bandwidth : int, optional
        Number of ordinates ``m``; defaults to ``floor(n ** bandwidth_exponent)``.
    bandwidth_exponent : float, default=0.5
    bounds : (float, float), default=(-0.5, 1.5)
        Search interval for ``d``.

    Returns
    -------
    LongMemoryResult
    """
    arr = _finite(x)
    n = arr.shape[0]
    if n < 16:
        raise ValueError(f"local-Whittle needs at least 16 observations, got {n}.")
    m = _bandwidth(n, bandwidth, bandwidth_exponent)
    lam, inten = _periodogram(arr, m)
    good = inten > 0
    lam, inten = lam[good], inten[good]
    if lam.shape[0] < 3:
        raise ValueError("local-Whittle: too few positive periodogram ordinates.")

    lo, hi = float(bounds[0]), float(bounds[1])
    grid = np.linspace(lo, hi, 81)
    values = np.array([_whittle_objective(g, lam, inten) for g in grid])
    best = int(np.argmin(values))
    a = grid[max(best - 1, 0)]
    b = grid[min(best + 1, grid.shape[0] - 1)]
    # Golden-section refinement on the bracketing interval.
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c, dd = b - phi * (b - a), a + phi * (b - a)
    fc, fd = _whittle_objective(c, lam, inten), _whittle_objective(dd, lam, inten)
    for _ in range(60):
        if abs(b - a) < 1e-8:
            break
        if fc < fd:
            b, dd, fd = dd, c, fc
            c = b - phi * (b - a)
            fc = _whittle_objective(c, lam, inten)
        else:
            a, c, fc = c, dd, fd
            dd = a + phi * (b - a)
            fd = _whittle_objective(dd, lam, inten)
    d = float((a + b) / 2.0)
    se = float(1.0 / (2.0 * np.sqrt(lam.shape[0])))
    t = d / se if se > 0 else np.nan
    return LongMemoryResult(
        d,
        se,
        float(t),
        float(2.0 * norm_sf(abs(t))),
        int(lam.shape[0]),
        n,
        "local_whittle",
        False,
    )


def estimate_fractional_order(
    x: np.ndarray,
    *,
    method: str = "local_whittle",
    bandwidth: int | None = None,
    bandwidth_exponent: float = 0.5,
    difference_if_above: float | None = 0.5,
) -> LongMemoryResult:
    """Estimate ``d`` with the requested estimator, handling the ``d >= 0.5`` region.

    Both GPH and local-Whittle are derived for a stationary
    ``d in (-0.5, 0.5)``. When the first-pass estimate lands at or above
    ``difference_if_above`` the series is differenced once, the estimator is
    re-run, and ``1`` is added back -- the standard way to cover integrated
    series (``d`` near or above ``0.5``, including a random walk at ``d = 1``).

    Parameters
    ----------
    x : numpy.ndarray
        1-D series in time order (training rows only).
    method : {"local_whittle", "gph"}, default="local_whittle"
    bandwidth, bandwidth_exponent
        Passed through to the estimator.
    difference_if_above : float or None, default=0.5
        Threshold that triggers the difference-and-add-one path; ``None``
        disables it.

    Returns
    -------
    LongMemoryResult
        With ``differenced=True`` when the fallback path was taken.
    """
    estimator = {"gph": gph, "local_whittle": local_whittle}.get(method)
    if estimator is None:
        raise ValueError(f"unknown method {method!r}; choose 'local_whittle' or 'gph'.")
    first = estimator(x, bandwidth=bandwidth, bandwidth_exponent=bandwidth_exponent)
    if difference_if_above is None or first.d < difference_if_above:
        return first
    diffed = np.diff(_finite(x))
    if diffed.shape[0] < 16:
        return first
    second = estimator(
        diffed, bandwidth=bandwidth, bandwidth_exponent=bandwidth_exponent
    )
    d = 1.0 + second.d
    t = d / second.se if second.se > 0 else np.nan
    return LongMemoryResult(
        float(d),
        second.se,
        float(t),
        float(2.0 * norm_sf(abs(t))),
        second.bandwidth,
        second.nobs,
        method,
        True,
    )


# --------------------------------------------------------------------------- #
# Panel surface
# --------------------------------------------------------------------------- #
def long_memory_table(
    df: pl.DataFrame | pl.LazyFrame,
    column: str,
    *,
    entity: str,
    time: str,
    method: str = "local_whittle",
    bandwidth_exponent: float = 0.5,
) -> pl.DataFrame:
    """Estimate ``d`` on each entity's full series (one row per entity).

    A *fitted* whole-sample quantity -- run it on training rows only. For a
    per-row causal feature use :func:`rolling_long_memory_features`.
    """

    def kernel(data: dict[str, np.ndarray]) -> dict[str, float]:
        try:
            res = estimate_fractional_order(
                data[column],
                method=method,
                bandwidth_exponent=bandwidth_exponent,
            )
        except ValueError:
            return {
                "d": float("nan"),
                "d_se": float("nan"),
                "d_tstat": float("nan"),
                "d_pvalue": float("nan"),
                "d_bandwidth": float("nan"),
                "d_differenced": float("nan"),
            }
        return {
            "d": res.d,
            "d_se": res.se,
            "d_tstat": res.tstat,
            "d_pvalue": res.pvalue,
            "d_bandwidth": float(res.bandwidth),
            "d_differenced": float(res.differenced),
        }

    return per_entity_reduce(
        df, entity=entity, time=time, columns=[column], func=kernel
    )


def rolling_long_memory_features(
    df: pl.DataFrame | pl.LazyFrame,
    column: str,
    *,
    entity: str,
    time: str,
    window: int = 252,
    method: str = "local_whittle",
    bandwidth_exponent: float = 0.5,
    min_periods: int | None = None,
    prefix: str | None = None,
) -> pl.DataFrame:
    """Trailing-window estimate of ``d`` as a per-row persistence feature.

    Row ``t`` uses only ``x[t - window + 1 : t + 1]`` within its entity, so the
    feature is causal and invariant to appended future rows.
    """
    pfx = f"{column}_" if prefix is None else prefix

    def one(chunk: np.ndarray) -> float:
        return estimate_fractional_order(
            chunk, method=method, bandwidth_exponent=bandwidth_exponent
        ).d

    def kernel(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            f"{pfx}d": rolling_apply(data[column], window, one, min_periods=min_periods)
        }

    return per_entity_apply(df, entity=entity, time=time, columns=[column], func=kernel)


# --------------------------------------------------------------------------- #
# Data-driven frac-diff
# --------------------------------------------------------------------------- #
class AutoFracDiff(PanelTransformer):
    """Fractional differencing with a **per-entity, train-only** order ``d``.

    ``fit`` estimates ``d`` for each entity from that entity's training rows
    (local-Whittle by default), clips it into ``[d_min, d_max]``, and builds the
    corresponding fixed-width weight kernel. ``transform`` applies the frozen
    kernels with the shared causal filter
    (:func:`panelary._ffd.frac_diff_expr`) under ``.over(entity)``.

    This is the leak-safe version of "difference each series by however much it
    needs": the order is a fitted parameter, so it is chosen once on train and
    reused verbatim on every test fold.

    Parameters
    ----------
    columns : sequence of str
        Columns to filter.
    method : {"local_whittle", "gph"}, default="local_whittle"
        Estimator for ``d``.
    d_min, d_max : float
        Clipping bounds for the estimated order (default ``0.0`` / ``1.0``).
    threshold : float
        Weight-magnitude cutoff passed to
        :func:`panelary._ffd.ffd_weights`.
    max_width : int, optional
        Hard cap on the kernel width.
    round_to : float, optional
        Round the estimated ``d`` to this grid (e.g. ``0.05``) so that entities
        share kernels and the fit is less jittery. ``None`` disables rounding.
    suffix : str, default="_ffd"
        Suffix for the emitted columns; pass ``""`` to replace in place.
    entity, time : str, optional
        Default panel keys for bare polars frames.

    Attributes
    ----------
    orders_ : dict
        ``{(entity, column): d}`` learned at fit time.
    default_order_ : dict
        ``{column: d}`` median fallback for entities unseen during ``fit``.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        columns: Sequence[str],
        *,
        method: str = "local_whittle",
        d_min: float = 0.0,
        d_max: float = 1.0,
        threshold: float = DEFAULT_THRESHOLD,
        max_width: int | None = None,
        bandwidth_exponent: float = 0.5,
        round_to: float | None = 0.05,
        suffix: str = "_ffd",
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self.columns = list(columns)
        if not self.columns:
            raise ValueError("`columns` must name at least one column.")
        if d_min < 0:
            raise ValueError(
                f"`d_min` must be >= 0 (the fixed-width filter diverges for "
                f"negative orders), got {d_min!r}."
            )
        if d_max < d_min:
            raise ValueError("`d_max` must be >= `d_min`.")
        self.method = method
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.threshold = float(threshold)
        self.max_width = max_width
        self.bandwidth_exponent = float(bandwidth_exponent)
        self.round_to = round_to
        self.suffix = suffix
        self.orders_: dict[tuple[object, str], float] = {}
        self.default_order_: dict[str, float] = {}

    def _estimate(self, x: np.ndarray) -> float:
        try:
            res = estimate_fractional_order(
                x, method=self.method, bandwidth_exponent=self.bandwidth_exponent
            )
            d = res.d
        except ValueError:
            d = self.d_min
        if not np.isfinite(d):
            d = self.d_min
        d = float(min(max(d, self.d_min), self.d_max))
        if self.round_to:
            d = float(round(d / self.round_to) * self.round_to)
            d = float(min(max(d, self.d_min), self.d_max))
        return d

    def _fit(self, panel: PanelFrame) -> None:
        frame = sorted_panel(panel.lazy(), panel.entity_col, panel.time_col)
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise ValueError(
                f"column(s) {missing} not found in panel; available: {frame.columns}."
            )
        self.orders_ = {}
        per_column: dict[str, list[float]] = {c: [] for c in self.columns}
        for key, _idx, data in entity_arrays(frame, panel.entity_col, self.columns):
            for col in self.columns:
                d = self._estimate(data[col])
                self.orders_[(key, col)] = d
                per_column[col].append(d)
        for col, orders in per_column.items():
            self.default_order_[col] = (
                float(np.median(orders)) if orders else self.d_min
            )

    def _weights(self, d: float) -> np.ndarray:
        return ffd_weights(d, self.threshold, self.max_width)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        ent = panel.entity_col
        exprs = []
        for col in self.columns:
            name = f"{col}{self.suffix}" if self.suffix else col
            default = self.default_order_.get(col, self.d_min)
            orders = {k[0]: v for k, v in self.orders_.items() if k[1] == col}
            distinct = sorted({*orders.values(), default})
            base = pl.col(col).cast(pl.Float64)
            if len(distinct) == 1:
                exprs.append(
                    frac_diff_expr(base, weights=self._weights(distinct[0]))
                    .over(ent)
                    .alias(name)
                )
                continue
            branches = None
            for d in distinct:
                members = [k for k, v in orders.items() if v == d]
                if not members:
                    continue
                cond = pl.col(ent).is_in(members)
                value = frac_diff_expr(base, weights=self._weights(d)).over(ent)
                branches = (
                    pl.when(cond).then(value)
                    if branches is None
                    else branches.when(cond).then(value)
                )
            fallback = frac_diff_expr(base, weights=self._weights(default)).over(ent)
            exprs.append(
                (fallback if branches is None else branches.otherwise(fallback)).alias(
                    name
                )
            )
        return panel.with_columns(exprs)
