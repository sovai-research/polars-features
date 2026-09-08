"""IVX predictive regression -- valid inference under persistent regressors.

A **clean-room implementation from the published papers** (Magdalinos &
Phillips 2009; Kostakis, Magdalinos & Stamatogiannis 2015). The R reference
implementation is GPL and was not consulted.

The problem
-----------
The classic predictive regression

.. math::

    y_t = \\mu + \\beta' x_{t-1} + u_t, \\qquad
    x_t = R_n x_{t-1} + e_t

is the workhorse of return predictability. When the predictor is highly
persistent (``R_n`` near unity) **and** its innovation is correlated with the
return innovation -- true of essentially every valuation ratio -- the OLS
t-statistic is badly size-distorted: it rejects "no predictability" far too
often, which is a large part of why the predictability literature is so
contested. The distortion does not vanish with sample size, and it depends on a
nuisance parameter (the local-to-unity coefficient) that cannot be estimated
consistently.

The IVX fix
-----------
Instrument the persistent regressor with a **mildly integrated** transform of
its own increments:

.. math::

    z_t = \\sum_{j=1}^{t} \\left(1 - \\frac{c_z}{n^{\\delta}}\\right)^{t-j}
          \\Delta x_j , \\qquad \\delta \\in (0, 1).

``z_t`` is persistent enough to stay correlated with ``x_t`` but mildly
integrated rather than (locally) unit-root, so the resulting IV estimator has a
**mixed-normal** limit and its Wald statistic is asymptotically chi-squared
*regardless* of the regressor's degree of persistence -- stationary, local to
unity, or unit root. That uniformity is the whole point: no pre-test on the
persistence is required.

Implementation
--------------
With ``X`` the (already-lagged) regressor matrix and ``y`` the outcome:

* ``z`` is built by the recursion ``z_t = rho_z z_{t-1} + (x_t - x_{t-1})``
  with ``rho_z = 1 - c_z / n^delta``;
* ``A = sum_t z_t (x_t - xbar)'``, ``beta_ivx = A^{-1} sum_t z_t (y_t - ybar)``;
* the covariance uses the mean-corrected instrument second moment,
  ``M = sigma_uu * sum_t z_t z_t' - n * zbar zbar' * omega_FM``, giving
  ``V = A^{-1} M A^{-1'}``. The mean-correction term carries the **fully
  modified** variance ``omega_FM = sigma_uu - sigma_ue sigma_ee^{-1} sigma_eu``
  (the part of ``u`` orthogonal to the regressor innovation ``e``) rather than
  ``sigma_uu``; using ``sigma_uu`` there leaves a large size distortion when the
  root is close to unity;
* ``W = (R beta - r)' (R V R')^{-1} (R beta - r) -> chi2_q``.

Use as a screening score
------------------------
:func:`ivx_screen` runs a univariate IVX regression for each candidate predictor
and ranks them by the IVX-Wald statistic. Unlike an OLS t-statistic this ranking
is not inflated for the most persistent candidates, so it is a **leak-aware**
screen: it is computed on training rows only, and :class:`IVXSelector` freezes
the chosen subset at ``fit`` time.

References
----------
Magdalinos, T. & Phillips, P. C. B. (2009). *Limit theory for cointegrated
systems with moderately integrated and moderately explosive regressors*, ET 25.
Kostakis, A., Magdalinos, T. & Stamatogiannis, M. P. (2015). *Robust econometric
inference for stock return predictability*, RFS 28(5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from polars_features.core.panel_frame import PanelFrame, as_panel
from polars_features.core.protocol import PanelTransformer
from polars_features.econ._common import chi2_sf, ols, pinv_sym

__all__ = [
    "ivx_instrument",
    "ivx",
    "IVXResult",
    "ivx_screen",
    "IVXSelector",
]


def ivx_instrument(
    x: np.ndarray, *, delta: float = 0.95, c_z: float = 1.0
) -> np.ndarray:
    """Build the mildly-integrated IVX instrument from a regressor path.

    Parameters
    ----------
    x : ndarray, shape (n, K)
        The regressor levels, ordered in time.
    delta : float, default=0.95
        Mild-integration exponent, in ``(0, 1)``. Larger values make the
        instrument more persistent (more power, slightly more size distortion);
        ``0.95`` is the value used throughout Kostakis et al.
    c_z : float, default=1.0
        Positive scaling constant in ``rho_z = 1 - c_z / n^delta``.

    Returns
    -------
    ndarray, shape (n - 1, K)
        ``z_t`` for ``t = 1 .. n-1``, aligned with ``x[1:]``.

    Raises
    ------
    ValueError
        If ``delta`` is outside ``(0, 1)`` or ``c_z <= 0``.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"`delta` must lie in (0, 1), got {delta}.")
    if c_z <= 0.0:
        raise ValueError(f"`c_z` must be positive, got {c_z}.")
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[0] == 1 and x.shape[1] != 1:
        x = x.T
    n = x.shape[0]
    if n < 3:
        raise ValueError(
            f"need at least 3 observations to build the IVX instrument, got {n}."
        )
    dx = np.diff(x, axis=0)
    rho = 1.0 - c_z / (n**delta)
    z = np.empty_like(dx)
    acc = np.zeros(x.shape[1])
    for i in range(dx.shape[0]):
        acc = rho * acc + dx[i]
        z[i] = acc
    return z


@dataclass
class IVXResult:
    """Result of an IVX predictive regression.

    Attributes
    ----------
    names : list of str
    params : ndarray, shape (K,)
        IVX coefficient estimates.
    std_errors : ndarray, shape (K,)
    wald : ndarray, shape (K,)
        Individual IVX-Wald statistics ``beta_k^2 / V_kk``, each asymptotically
        chi-squared with one degree of freedom. **This is the screening score.**
    pvalues : ndarray, shape (K,)
    vcov : ndarray, shape (K, K)
    joint_wald, joint_pvalue : float
        Test that all coefficients are zero.
    ols_params, ols_tstats : ndarray
        The plain OLS comparison, so the size distortion IVX corrects is visible.
    nobs : int
    delta, c_z : float
    rho_z : float
        The instrument's autoregressive root actually used.
    """

    names: list[str]
    params: np.ndarray
    std_errors: np.ndarray
    wald: np.ndarray
    pvalues: np.ndarray
    vcov: np.ndarray
    joint_wald: float
    joint_pvalue: float
    ols_params: np.ndarray
    ols_tstats: np.ndarray
    nobs: int
    delta: float
    c_z: float
    rho_z: float

    def summary(self) -> pl.DataFrame:
        """Coefficient table with IVX-Wald statistics and p-values."""
        return pl.DataFrame(
            {
                "term": self.names,
                "estimate": np.asarray(self.params, dtype=float),
                "std_error": np.asarray(self.std_errors, dtype=float),
                "ivx_wald": np.asarray(self.wald, dtype=float),
                "p_value": np.asarray(self.pvalues, dtype=float),
                "ols_estimate": np.asarray(self.ols_params, dtype=float),
                "ols_t": np.asarray(self.ols_tstats, dtype=float),
            }
        )


def ivx(
    y: np.ndarray,
    x: np.ndarray,
    *,
    names: Sequence[str] | None = None,
    delta: float = 0.95,
    c_z: float = 1.0,
) -> IVXResult:
    """Fit an IVX predictive regression ``y_t = mu + beta' x_t + u_t``.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Outcome. The caller is responsible for the timing: ``x[t]`` must be
        observable **before** ``y[t]`` (i.e. already lagged).
    x : ndarray, shape (n,) or (n, K)
        Predictor levels (not differences).
    names : sequence of str, optional
    delta, c_z : float
        IVX tuning constants (see :func:`ivx_instrument`).

    Returns
    -------
    IVXResult

    Raises
    ------
    ValueError
        On mismatched lengths or too few observations.

    Examples
    --------
    >>> res = ivx(returns[1:], dividend_yield[:-1])  # doctest: +SKIP
    >>> res.summary()  # doctest: +SKIP
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[0] == 1 and x.shape[1] != 1:
        x = x.T
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"`y` has {y.shape[0]} observations but `x` has {x.shape[0]}.")
    k = x.shape[1]
    if y.shape[0] < k + 4:
        raise ValueError(
            f"need at least {k + 4} observations for a {k}-regressor IVX fit, "
            f"got {y.shape[0]}."
        )

    z = ivx_instrument(x, delta=delta, c_z=c_z)
    xe = x[1:]
    ye = y[1:]
    n = ye.shape[0]

    xd = xe - xe.mean(axis=0)
    yd = ye - ye.mean()
    zbar = z.mean(axis=0)

    a_mat = z.T @ xd
    if np.linalg.matrix_rank(a_mat) < k:
        raise ValueError(
            "the IVX instrument is not correlated enough with the regressors "
            "(singular moment matrix); check for a constant or duplicated column."
        )
    a_inv = np.linalg.inv(a_mat)
    beta = a_inv @ (z.T @ yd)

    design = np.column_stack([np.ones(n), xe])
    ols_beta, ols_resid, ols_xtx_inv = ols(design, ye)

    # Residual second moments. `e_resid[i]` is the innovation of the regressor
    # at the same calendar date as `ols_resid[i - 1]`, hence the one-step
    # realignment below: the endogeneity that distorts OLS is
    # ``corr(u_t, e_t)``, and `xe[i]` is dated one period before `ye[i]`.
    lag_design = np.column_stack([np.ones(x.shape[0] - 1), x[:-1]])
    _, e_resid, _ = ols(lag_design, x[1:])
    e_resid = np.atleast_2d(e_resid)
    if e_resid.shape[0] != n:
        e_resid = e_resid.T
    m_pairs = n - 1
    sigma_uu = float(ols_resid @ ols_resid) / n
    sigma_ue = (ols_resid[:m_pairs, None] * e_resid[1:]).sum(axis=0) / m_pairs
    sigma_ee = (e_resid.T @ e_resid) / n
    # Fully-modified (conditional) variance: the part of `u` orthogonal to `e`.
    # Using it in the mean-correction term is what makes the IVX Wald statistic
    # correctly sized when the regressor is (locally) a unit root.
    omega_fm = sigma_uu - float(sigma_ue @ np.linalg.solve(sigma_ee, sigma_ue))
    omega_fm = max(omega_fm, 0.0)

    m_mat = (z.T @ z) * sigma_uu - n * np.outer(zbar, zbar) * omega_fm
    vcov = a_inv @ m_mat @ a_inv.T
    vcov = 0.5 * (vcov + vcov.T)

    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        wald = np.where(np.diag(vcov) > 0, beta**2 / np.diag(vcov), np.nan)
    pvalues = np.asarray(chi2_sf(wald, 1.0))
    joint = float(beta @ pinv_sym(vcov) @ beta)

    ols_sigma2 = float(ols_resid @ ols_resid) / max(n - k - 1, 1)
    ols_se = np.sqrt(np.clip(np.diag(ols_sigma2 * ols_xtx_inv), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        ols_t = np.where(ols_se[1:] > 0, ols_beta[1:] / ols_se[1:], np.nan)

    labels = list(names) if names is not None else [f"x{i}" for i in range(k)]
    return IVXResult(
        names=labels,
        params=beta,
        std_errors=se,
        wald=wald,
        pvalues=pvalues,
        vcov=vcov,
        joint_wald=joint,
        joint_pvalue=float(chi2_sf(joint, k)),
        ols_params=ols_beta[1:],
        ols_tstats=ols_t,
        nobs=n,
        delta=delta,
        c_z=c_z,
        rho_z=1.0 - c_z / (x.shape[0] ** delta),
    )


# --------------------------------------------------------------------------- #
# Panel screening
# --------------------------------------------------------------------------- #
def ivx_screen(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    y: str,
    candidates: Sequence[str],
    entity: str | None = None,
    time: str | None = None,
    delta: float = 0.95,
    c_z: float = 1.0,
    pooled: bool = True,
    min_obs: int = 20,
) -> pl.DataFrame:
    """Rank candidate predictors by their IVX-Wald statistic.

    Each candidate is screened **univariately**: one IVX regression of ``y`` on
    that candidate. With ``pooled=True`` the per-entity Wald statistics are
    aggregated across entities by summing them (their sum is chi-squared with
    ``N`` degrees of freedom under independence across entities), which is the
    natural panel analogue of a Fisher-type combination.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
        The panel. ``y`` must already be aligned so that a candidate's value at
        row ``t`` predicts ``y`` at row ``t``.
    y : str
    candidates : sequence of str
    entity, time : str, optional
    delta, c_z : float
        IVX tuning constants.
    pooled : bool, default=True
        Aggregate over entities. When ``False`` the panel is treated as one long
        series (only sensible for a single-entity panel).
    min_obs : int, default=20
        Skip entities with fewer usable observations.

    Returns
    -------
    polars.DataFrame
        ``feature`` / ``ivx_wald`` / ``p_value`` / ``n_entities`` / ``mean_beta``,
        sorted by descending Wald statistic.
    """
    panel = (
        data
        if isinstance(data, PanelFrame)
        else as_panel(data, entity=entity, time=time)
    )
    cands = list(candidates)
    missing = [c for c in [y, *cands] if c not in panel.columns]
    if missing:
        raise ValueError(
            f"column(s) {missing} not found in panel. Available: {panel.columns}."
        )

    rows = []
    for cand in cands:
        frame = (
            panel.sort_panel()
            .lazy()
            .select(
                [
                    pl.col(panel.entity_col),
                    pl.col(y).cast(pl.Float64).alias("__y"),
                    pl.col(cand).cast(pl.Float64).alias("__x"),
                ]
            )
            .collect()
            .filter(
                pl.all_horizontal(
                    [
                        pl.col(c).is_not_null() & pl.col(c).is_not_nan()
                        for c in ("__y", "__x")
                    ]
                )
            )
        )
        if frame.height == 0:
            continue
        ent = frame[panel.entity_col].to_numpy()
        yy = frame["__y"].to_numpy()
        xx = frame["__x"].to_numpy()
        groups = (
            [np.flatnonzero(ent == g) for g in np.unique(ent)]
            if pooled
            else [np.arange(ent.size)]
        )
        stats, betas = [], []
        for idx in groups:
            if idx.size < max(min_obs, 6):
                continue
            try:
                res = ivx(yy[idx], xx[idx], names=[cand], delta=delta, c_z=c_z)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if np.isfinite(res.wald[0]):
                stats.append(float(res.wald[0]))
                betas.append(float(res.params[0]))
        if not stats:
            continue
        total = float(np.sum(stats))
        rows.append(
            {
                "feature": cand,
                "ivx_wald": total,
                "p_value": float(chi2_sf(total, len(stats))),
                "n_entities": len(stats),
                "mean_beta": float(np.mean(betas)),
            }
        )
    if not rows:
        raise ValueError(
            "no candidate produced a usable IVX regression; lower `min_obs` or "
            "check for constant / all-null candidate columns."
        )
    return pl.DataFrame(rows).sort("ivx_wald", descending=True)


class IVXSelector(PanelTransformer):
    """Select predictors by IVX-Wald screening, frozen at ``fit`` time.

    ``fit`` runs :func:`ivx_screen` on the **training rows only** and stores the
    surviving column names. ``transform`` keeps exactly that frozen set, so the
    selection can never absorb test-fold structure -- the same contract as the
    selectors in :mod:`polars_features.select`.

    Parameters
    ----------
    y : str
        Target used for screening.
    candidates : sequence of str, optional
        Columns to screen. Defaults to every numeric feature column except ``y``.
    k : int, optional
        Keep the ``k`` highest-Wald candidates.
    alpha : float, optional
        Keep candidates with ``p_value <= alpha``. Ignored when ``k`` is given.
        Defaults to ``0.05`` when neither is supplied.
    delta, c_z, min_obs :
        Forwarded to :func:`ivx_screen`.

    Attributes
    ----------
    ranking_ : polars.DataFrame
        The full screening table.
    selected_ : list of str
        The frozen selection.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        *,
        y: str,
        candidates: Sequence[str] | None = None,
        k: int | None = None,
        alpha: float | None = None,
        delta: float = 0.95,
        c_z: float = 1.0,
        min_obs: int = 20,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if k is not None and k < 1:
            raise ValueError(f"`k` must be >= 1, got {k}.")
        if alpha is not None and not 0.0 < alpha < 1.0:
            raise ValueError(f"`alpha` must lie in (0, 1), got {alpha}.")
        self.y = y
        self.candidates = list(candidates) if candidates is not None else None
        self.k = k
        self.alpha = alpha if alpha is not None else (None if k else 0.05)
        self.delta = float(delta)
        self.c_z = float(c_z)
        self.min_obs = int(min_obs)
        self.ranking_: pl.DataFrame | None = None
        self.selected_: list[str] = []

    def _fit(self, panel: PanelFrame) -> None:
        if self.candidates is not None:
            cands = list(self.candidates)
        else:
            schema = panel.schema
            cands = [
                c for c in panel.feature_cols if c != self.y and schema[c].is_numeric()
            ]
        if not cands:
            raise ValueError(
                "no numeric candidate columns available for IVX screening."
            )
        ranking = ivx_screen(
            panel,
            y=self.y,
            candidates=cands,
            delta=self.delta,
            c_z=self.c_z,
            min_obs=self.min_obs,
        )
        self.ranking_ = ranking
        if self.k is not None:
            self.selected_ = ranking.head(self.k)["feature"].to_list()
        else:
            self.selected_ = ranking.filter(pl.col("p_value") <= self.alpha)[
                "feature"
            ].to_list()

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        keep = [c for c in self.selected_ if c in panel.columns]
        return panel.select(*[pl.col(c) for c in keep])
