"""Fama-MacBeth two-pass cross-sectional regression.

The finance staple: for every date ``t`` run a *cross-sectional* OLS of the
return on the characteristics,

.. math::

    r_{it} = \\lambda_{0t} + \\lambda_t' x_{i,t} + \\varepsilon_{it},

then treat the resulting ``lambda`` series as a time series of risk-premium
estimates. The reported coefficient is its time-series mean and the standard
error is a **Newey-West HAC** standard error of that mean, which handles the
autocorrelation the second pass induces.

Pass 1 maps exactly onto a Polars ``group_by(time)``: one small OLS per date.

Leak-safety
-----------
Two rules are baked in (plan section 7.3):

1. **Winsorising and standardising are done PER DATE**, never pooled over the
   whole sample. A global z-score would let the full-sample mean and standard
   deviation of a characteristic leak backwards into every earlier row; a
   per-date cross-sectional z-score uses only contemporaneous information.
2. :class:`FamaMacBethTransformer` freezes the **train-sample mean lambda** at
   ``fit`` time and applies it at ``transform`` time. Nothing about the test
   dates enters the fitted premia.

References
----------
Fama, E. F. & MacBeth, J. D. (1973). *Risk, return, and equilibrium: empirical
tests*, JPE 81(3).
Newey, W. & West, K. (1987). *A simple, positive semi-definite,
heteroskedasticity and autocorrelation consistent covariance matrix*.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame, as_panel
from panelary.core.protocol import PanelTransformer
from panelary.econ._common import (
    auto_bandwidth,
    factorize,
    newey_west_scalar,
    norm_ppf,
    pinv_sym,
    t_sf,
    winsorize,
)

__all__ = ["fama_macbeth", "FamaMacBethResult", "FamaMacBethTransformer"]


def _prepare_cross_section(
    xs: np.ndarray,
    *,
    winsor: float,
    standardize: bool,
) -> np.ndarray:
    """Winsorize then z-score each characteristic **within one date**."""
    out = xs.astype(float, copy=True)
    for j in range(out.shape[1]):
        col = out[:, j]
        if winsor:
            col = winsorize(col, winsor)
        if standardize:
            mu = float(np.nanmean(col))
            sd = float(np.nanstd(col, ddof=1)) if col.size > 1 else 0.0
            col = (col - mu) / sd if sd > 0 else col - mu
        out[:, j] = col
    return out


@dataclass
class FamaMacBethResult:
    """Result of a Fama-MacBeth two-pass regression.

    Attributes
    ----------
    names : list of str
        ``["const", *x]`` when ``intercept=True``.
    params : ndarray
        Time-series mean of the per-date lambdas.
    std_errors : ndarray
        Newey-West standard errors of those means.
    tstats, pvalues : ndarray
    lambdas : polars.DataFrame
        The per-date lambda series: one ``time`` column plus one column per
        term. **Contemporaneous** by construction -- safe to join back onto the
        panel as a date-level feature.
    n_periods : int
        Number of dates that produced a usable cross-section.
    n_obs : ndarray
        Cross-section size per date.
    lags : int
        Newey-West bandwidth actually used.
    r2 : ndarray
        Per-date cross-sectional R-squared.
    """

    names: list[str]
    params: np.ndarray
    std_errors: np.ndarray
    tstats: np.ndarray
    pvalues: np.ndarray
    lambdas: pl.DataFrame
    n_periods: int
    n_obs: np.ndarray
    lags: int
    r2: np.ndarray

    def summary(self) -> pl.DataFrame:
        """Coefficient table (mean lambda, NW standard error, t, p)."""
        return pl.DataFrame(
            {
                "term": self.names,
                "estimate": np.asarray(self.params, dtype=float),
                "std_error": np.asarray(self.std_errors, dtype=float),
                "t_stat": np.asarray(self.tstats, dtype=float),
                "p_value": np.asarray(self.pvalues, dtype=float),
            }
        )

    def conf_int(self, alpha: float = 0.05) -> pl.DataFrame:
        """Two-sided normal confidence intervals for the mean lambdas."""
        z = float(norm_ppf(1.0 - alpha / 2.0))
        return pl.DataFrame(
            {
                "term": self.names,
                "lower": self.params - z * self.std_errors,
                "upper": self.params + z * self.std_errors,
            }
        )

    def lambda_features(self, prefix: str = "fm_lambda_") -> pl.DataFrame:
        """The lambda series renamed as joinable date-level feature columns."""
        renames = {c: f"{prefix}{c}" for c in self.lambdas.columns[1:]}
        return self.lambdas.rename(renames)


def fama_macbeth(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    y: str,
    x: Sequence[str],
    entity: str | None = None,
    time: str | None = None,
    intercept: bool = True,
    winsorize_limit: float | None = 0.01,
    standardize: bool = False,
    lags: int = -1,
    min_cross_section: int | None = None,
) -> FamaMacBethResult:
    """Run a Fama-MacBeth two-pass cross-sectional regression.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long panel. One row per ``(entity, time)``.
    y : str
        Dependent variable (typically the *forward* return; the caller is
        responsible for having lagged the characteristics).
    x : sequence of str
        Characteristics used in the cross-sectional regressions.
    entity, time : str, optional
        Panel keys when ``data`` is a bare frame.
    intercept : bool, default=True
        Include a per-date intercept (``lambda_0``, the zero-beta rate).
    winsorize_limit : float or None, default=0.01
        Per-date symmetric winsorising of each characteristic. ``None`` / ``0``
        disables it. **Applied per date, never globally.**
    standardize : bool, default=False
        Per-date cross-sectional z-scoring of each characteristic. **Per date,
        never globally.**
    lags : int, default=-1
        Newey-West bandwidth for the standard errors of the mean lambdas.
        ``-1`` uses ``floor(4 (T/100)^(2/9))``.
    min_cross_section : int, optional
        Skip dates with fewer usable observations than this. Defaults to
        ``len(x) + intercept + 1`` (the minimum for an identified regression).

    Returns
    -------
    FamaMacBethResult

    Raises
    ------
    ValueError
        If no date yields a usable cross-section.

    Examples
    --------
    >>> res = fama_macbeth(panel, y="fwd_ret", x=["beta", "size", "bm"])  # doctest: +SKIP
    >>> res.summary()  # doctest: +SKIP
    """
    panel = (
        data
        if isinstance(data, PanelFrame)
        else as_panel(data, entity=entity, time=time)
    )
    x = list(x)
    if not x:
        raise ValueError("`x` must name at least one characteristic.")
    k = len(x) + int(intercept)
    min_n = int(min_cross_section) if min_cross_section is not None else k + 1

    needed = [y, *x]
    frame = (
        panel.sort_panel()
        .lazy()
        .select(
            [
                pl.col(panel.time_col),
                *[pl.col(c).cast(pl.Float64).alias(c) for c in needed],
            ]
        )
        .collect()
        .filter(
            pl.all_horizontal(
                [pl.col(c).is_not_null() & pl.col(c).is_not_nan() for c in needed]
            )
        )
    )
    if frame.height == 0:
        raise ValueError("no complete observations for the Fama-MacBeth regression.")

    time_vals = frame[panel.time_col].to_numpy()
    codes, levels = factorize(time_vals)
    yv = frame[y].to_numpy()
    xv = np.column_stack([frame[c].to_numpy() for c in x])

    order = np.argsort(codes, kind="stable")
    codes_s, yv, xv = codes[order], yv[order], xv[order]
    bounds = np.searchsorted(codes_s, np.arange(levels.size + 1))

    lam_rows: list[np.ndarray] = []
    used_times: list = []
    n_obs: list[int] = []
    r2: list[float] = []
    winsor = float(winsorize_limit or 0.0)
    for g in range(levels.size):
        lo, hi = bounds[g], bounds[g + 1]
        if hi - lo < min_n:
            continue
        xg = _prepare_cross_section(xv[lo:hi], winsor=winsor, standardize=standardize)
        yg = yv[lo:hi]
        design = np.column_stack([np.ones(hi - lo), xg]) if intercept else xg
        if np.linalg.matrix_rank(design) < design.shape[1]:
            continue
        xtx_inv = pinv_sym(design.T @ design)
        beta = xtx_inv @ (design.T @ yg)
        resid = yg - design @ beta
        tss = float(((yg - yg.mean()) ** 2).sum())
        lam_rows.append(beta)
        used_times.append(levels[g])
        n_obs.append(int(hi - lo))
        r2.append(1.0 - float(resid @ resid) / tss if tss > 0 else float("nan"))

    if not lam_rows:
        raise ValueError(
            "no date produced a usable cross-section; lower `min_cross_section` "
            "or check for collinear characteristics."
        )

    lam = np.vstack(lam_rows)
    names = (["const"] if intercept else []) + list(x)
    t_periods = lam.shape[0]
    band = lags if lags >= 0 else auto_bandwidth(t_periods)
    params = lam.mean(axis=0)
    se = np.array(
        [np.sqrt(newey_west_scalar(lam[:, j], band)) for j in range(lam.shape[1])]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        tstats = np.where(se > 0, params / se, np.nan)
    pvalues = 2.0 * np.asarray(t_sf(np.abs(tstats), max(t_periods - 1, 1)))

    lambdas = pl.DataFrame(
        {
            panel.time_col: pl.Series(np.asarray(used_times)),
            **{name: lam[:, j] for j, name in enumerate(names)},
            "n_obs": np.asarray(n_obs, dtype=np.int64),
            "r2": np.asarray(r2, dtype=float),
        }
    )

    return FamaMacBethResult(
        names=names,
        params=params,
        std_errors=se,
        tstats=tstats,
        pvalues=pvalues,
        lambdas=lambdas,
        n_periods=t_periods,
        n_obs=np.asarray(n_obs, dtype=np.int64),
        lags=band,
        r2=np.asarray(r2, dtype=float),
    )


class FamaMacBethTransformer(PanelTransformer):
    """Freeze train-sample Fama-MacBeth premia and emit them as features.

    ``fit`` runs :func:`fama_macbeth` on the training rows and stores the mean
    lambda vector (plus, optionally, the per-date standardisation is recomputed
    contemporaneously at transform time -- which is legitimate because a
    cross-sectional z-score on date ``t`` uses only date-``t`` information).

    ``transform`` emits, per row:

    * ``<prefix>pred`` -- ``x_i' * lambda_bar`` , the expected return implied by
      the *train-estimated* risk premia;
    * ``<prefix>resid`` -- ``y_i - <prefix>pred`` , the characteristic-adjusted
      residual (a classic "alpha" feature).

    Both use frozen coefficients, so no test-fold information enters them.

    Parameters
    ----------
    y : str
        Dependent variable. May be absent at transform time, in which case only
        the prediction column is emitted.
    x : sequence of str
        Characteristics.
    prefix : str, default="fm_"
    winsorize_limit, standardize, intercept, lags :
        Forwarded to :func:`fama_macbeth`.

    Attributes
    ----------
    result_ : FamaMacBethResult
        The full train-sample fit (lambda series, standard errors, ...).
    params_ : ndarray
        The frozen mean lambdas.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        *,
        y: str,
        x: Sequence[str],
        prefix: str = "fm_",
        intercept: bool = True,
        winsorize_limit: float | None = 0.01,
        standardize: bool = False,
        lags: int = -1,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self.y = y
        self.x = list(x)
        self.prefix = prefix
        self.intercept = bool(intercept)
        self.winsorize_limit = winsorize_limit
        self.standardize = bool(standardize)
        self.lags = int(lags)
        self.result_: FamaMacBethResult | None = None
        self.params_: np.ndarray | None = None

    def _fit(self, panel: PanelFrame) -> None:
        self.result_ = fama_macbeth(
            panel,
            y=self.y,
            x=self.x,
            intercept=self.intercept,
            winsorize_limit=self.winsorize_limit,
            standardize=self.standardize,
            lags=self.lags,
        )
        self.params_ = np.asarray(self.result_.params, dtype=float)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        assert self.params_ is not None  # noqa: S101 - guarded by `_check_fitted`
        params = self.params_
        offset = 0
        pred = pl.lit(0.0)
        if self.intercept:
            pred = pl.lit(float(params[0]))
            offset = 1
        for j, col in enumerate(self.x):
            pred = pred + pl.col(col).cast(pl.Float64) * float(params[offset + j])
        exprs = [pred.alias(f"{self.prefix}pred")]
        if self.y in panel.columns:
            exprs.append(
                (pl.col(self.y).cast(pl.Float64) - pred).alias(f"{self.prefix}resid")
            )
        return panel.with_columns(exprs)

    @property
    def output_names(self) -> list[str]:
        """Names of the emitted feature columns."""
        return [f"{self.prefix}pred", f"{self.prefix}resid"]
