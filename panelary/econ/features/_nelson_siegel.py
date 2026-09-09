"""Nelson-Siegel(-Svensson) level / slope / curvature from the maturity cross-section.

For each date the yield curve is fitted **across maturities**, which is a purely
contemporaneous operation: the factors at date ``t`` use only yields observed at
date ``t``. Nothing rolls, nothing looks ahead. The only fitted parameter is the
decay ``lambda`` (and ``lambda2`` for Svensson), which
:class:`NelsonSiegel` learns on the training dates and then freezes -- selecting
it from the whole sample would quietly leak.

Given the decay(s), the loadings are fixed and the remaining coefficients are a
2-4 column OLS per date:

``y(tau) = b0 + b1 * (1 - exp(-l*tau)) / (l*tau)
         + b2 * ((1 - exp(-l*tau)) / (l*tau) - exp(-l*tau))``

with ``b0`` the **level** (the asymptotic long rate) and ``b2`` the
**curvature**. The reported **slope** is ``-b1``, which equals the *long minus
short* spread: it is the negative of Diebold & Li's slope factor ``b1``, signed
so that an upward-sloping curve gives a positive value. Svensson adds a second
hump term at ``lambda2``.

References
----------
Nelson & Siegel (1987); Svensson (1994); Diebold & Li (2006), whose fixed
``lambda = 0.0609`` (maturities in months) is the default here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer
from panelary.econ.features._common import ols

__all__ = [
    "NelsonSiegelFit",
    "nelson_siegel_loadings",
    "nelson_siegel_fit",
    "nelson_siegel_factors",
    "NelsonSiegel",
]

#: Diebold & Li (2006) decay parameter for maturities measured in **months**.
DEFAULT_LAMBDA: float = 0.0609


class NelsonSiegelFit(NamedTuple):
    """One date's curve fit.

    Attributes
    ----------
    level, slope, curvature : float
        ``b0`` (the long rate), ``-b1`` (the long-minus-short spread) and ``b2``.
    beta : numpy.ndarray
        Raw coefficients ``[b0, b1, b2]`` (plus ``b3`` under Svensson).
    rmse : float
        Root-mean-square fitting error across the maturity cross-section.
    nobs : int
        Number of maturities used.
    lam : float
        The decay parameter used.
    """

    level: float
    slope: float
    curvature: float
    beta: np.ndarray
    rmse: float
    nobs: int
    lam: float


def nelson_siegel_loadings(
    maturities: np.ndarray,
    lam: float,
    *,
    lam2: float | None = None,
) -> np.ndarray:
    """Factor-loading matrix for the given maturities.

    Columns are ``[1, slope loading, curvature loading]`` (plus a second
    curvature loading at ``lam2`` for Svensson). The loadings depend only on the
    maturity grid and the decay, never on the yields.
    """
    tau = np.asarray(maturities, dtype=float).ravel()
    if np.any(tau <= 0):
        raise ValueError("maturities must be strictly positive.")
    if lam <= 0:
        raise ValueError(f"`lam` must be positive, got {lam!r}.")
    lt = lam * tau
    decay = (1.0 - np.exp(-lt)) / lt
    cols = [np.ones_like(tau), decay, decay - np.exp(-lt)]
    if lam2 is not None:
        if lam2 <= 0:
            raise ValueError(f"`lam2` must be positive, got {lam2!r}.")
        lt2 = lam2 * tau
        decay2 = (1.0 - np.exp(-lt2)) / lt2
        cols.append(decay2 - np.exp(-lt2))
    return np.column_stack(cols)


def nelson_siegel_fit(
    maturities: np.ndarray,
    yields: np.ndarray,
    *,
    lam: float = DEFAULT_LAMBDA,
    lam2: float | None = None,
) -> NelsonSiegelFit:
    """Fit the Nelson-Siegel curve to one date's maturity cross-section.

    Parameters
    ----------
    maturities : numpy.ndarray
        Maturities, strictly positive, in the same unit ``lam`` is calibrated to
        (months for the default ``lambda``).
    yields : numpy.ndarray
        Yields at those maturities. Non-finite pairs are dropped.
    lam : float, default=0.0609
        Decay parameter.
    lam2 : float, optional
        Second decay for the Svensson extension.

    Returns
    -------
    NelsonSiegelFit
    """
    tau = np.asarray(maturities, dtype=float).ravel()
    y = np.asarray(yields, dtype=float).ravel()
    if tau.shape != y.shape:
        raise ValueError("`maturities` and `yields` must have the same length.")
    good = np.isfinite(tau) & np.isfinite(y) & (tau > 0)
    tau, y = tau[good], y[good]
    ncoef = 4 if lam2 is not None else 3
    if tau.shape[0] < ncoef:
        return NelsonSiegelFit(
            float("nan"),
            float("nan"),
            float("nan"),
            np.full(ncoef, np.nan),
            float("nan"),
            int(tau.shape[0]),
            float(lam),
        )
    X = nelson_siegel_loadings(tau, lam, lam2=lam2)
    res = ols(X, y)
    rmse = float(np.sqrt(np.mean(res.resid**2)))
    return NelsonSiegelFit(
        float(res.beta[0]),
        float(-res.beta[1]),
        float(res.beta[2]),
        res.beta,
        rmse,
        int(tau.shape[0]),
        float(lam),
    )


def nelson_siegel_factors(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    date: str,
    maturity: str,
    yield_col: str,
    lam: float = DEFAULT_LAMBDA,
    lam2: float | None = None,
) -> pl.DataFrame:
    """Fit the curve **per date** and return one row of factors per date.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long ``(date, maturity, yield)`` frame.
    date, maturity, yield_col : str
        Column names.
    lam : float, default=0.0609
        Decay parameter (fixed across dates -- the Diebold-Li convention).
    lam2 : float, optional
        Second decay for Svensson.

    Returns
    -------
    polars.DataFrame
        ``date``, ``ns_level``, ``ns_slope``, ``ns_curvature``, ``ns_rmse``,
        ``ns_nobs`` (plus ``ns_curvature2`` under Svensson), sorted by date.
    """
    frame = df.collect() if isinstance(df, pl.LazyFrame) else df
    for col in (date, maturity, yield_col):
        if col not in frame.columns:
            raise ValueError(
                f"column {col!r} not found in frame; available: {frame.columns}."
            )
    frame = frame.sort([date, maturity])
    dates: list[object] = []
    rows: list[dict[str, float]] = []
    for (key,), group in frame.group_by([date], maintain_order=True):
        fit = nelson_siegel_fit(
            group.get_column(maturity).cast(pl.Float64).to_numpy(),
            group.get_column(yield_col).cast(pl.Float64).to_numpy(),
            lam=lam,
            lam2=lam2,
        )
        record = {
            "ns_level": fit.level,
            "ns_slope": fit.slope,
            "ns_curvature": fit.curvature,
            "ns_rmse": fit.rmse,
            "ns_nobs": float(fit.nobs),
        }
        if lam2 is not None:
            record["ns_curvature2"] = float(fit.beta[3])
        dates.append(key)
        rows.append(record)
    if not rows:
        return pl.DataFrame({date: []})
    out: dict[str, object] = {date: dates}
    for name in rows[0]:
        out[name] = [r[name] for r in rows]
    return pl.DataFrame(out).sort(date)


class NelsonSiegel(PanelTransformer):
    """Yield-curve factors as panel features, with a **train-fitted** decay.

    The panel is the ``(maturity, date)`` grid: ``entity`` is the maturity and
    ``time`` the date. ``fit`` optionally grid-searches ``lambda`` to minimise
    pooled squared fitting error **over the training dates** and freezes it;
    ``transform`` runs the per-date cross-sectional OLS with that frozen decay
    and broadcasts ``level`` / ``slope`` / ``curvature`` back onto every row of
    the corresponding date.

    Each date's factors depend only on that date's yields, so the features are
    contemporaneous and never look ahead.

    Parameters
    ----------
    yield_col : str
        Column holding the yields.
    lam : float, default=0.0609
        Decay used when ``fit_lambda`` is ``False``, and the fallback if the grid
        search fails.
    fit_lambda : bool, default=False
        Grid-search ``lambda`` on the training dates.
    lambda_grid : sequence of float, optional
        Candidate decays; defaults to 40 log-spaced points in ``[0.01, 0.3]``.
    svensson : bool, default=False
        Add the second Svensson hump (``lam2`` is searched/fixed at
        ``2 * lambda``).
    prefix : str, default="ns_"
        Prefix for the emitted columns.
    entity, time : str, optional
        Default panel keys (``entity`` = maturity column, ``time`` = date).

    Attributes
    ----------
    lambda_ : float
        The frozen decay.
    lambda2_ : float or None
        The frozen second decay under Svensson.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        yield_col: str,
        *,
        lam: float = DEFAULT_LAMBDA,
        fit_lambda: bool = False,
        lambda_grid: Sequence[float] | None = None,
        svensson: bool = False,
        prefix: str = "ns_",
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self.yield_col = yield_col
        self.lam = float(lam)
        self.fit_lambda = bool(fit_lambda)
        self.lambda_grid = (
            list(lambda_grid)
            if lambda_grid is not None
            else list(np.geomspace(0.01, 0.3, 40))
        )
        self.svensson = bool(svensson)
        self.prefix = prefix
        self.lambda_: float = float(lam)
        self.lambda2_: float | None = 2.0 * float(lam) if svensson else None

    def _pooled_sse(self, frame: pl.DataFrame, ent: str, tme: str, lam: float) -> float:
        lam2 = 2.0 * lam if self.svensson else None
        total = 0.0
        for _key, group in frame.group_by([tme], maintain_order=True):
            fit = nelson_siegel_fit(
                group.get_column(ent).cast(pl.Float64).to_numpy(),
                group.get_column(self.yield_col).cast(pl.Float64).to_numpy(),
                lam=lam,
                lam2=lam2,
            )
            if np.isfinite(fit.rmse):
                total += fit.rmse**2 * fit.nobs
        return total if total > 0 else np.inf

    def _fit(self, panel: PanelFrame) -> None:
        frame = panel.collect()
        if self.yield_col not in frame.columns:
            raise ValueError(
                f"column {self.yield_col!r} not found in panel; "
                f"available: {frame.columns}."
            )
        if not self.fit_lambda:
            self.lambda_ = self.lam
        else:
            best, best_sse = self.lam, np.inf
            for cand in self.lambda_grid:
                sse = self._pooled_sse(
                    frame, panel.entity_col, panel.time_col, float(cand)
                )
                if sse < best_sse:
                    best, best_sse = float(cand), sse
            self.lambda_ = best
        self.lambda2_ = 2.0 * self.lambda_ if self.svensson else None

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        factors = nelson_siegel_factors(
            panel.lazy(),
            date=panel.time_col,
            maturity=panel.entity_col,
            yield_col=self.yield_col,
            lam=self.lambda_,
            lam2=self.lambda2_,
        )
        renames = {
            c: f"{self.prefix}{c[3:]}" if c.startswith("ns_") else c
            for c in factors.columns
            if c != panel.time_col
        }
        factors = factors.rename(renames)
        return panel.pipe(
            lambda p: PanelFrame(
                p.lazy().join(factors.lazy(), on=panel.time_col, how="left"),
                entity=panel.entity_col,
                time=panel.time_col,
            )
        )
