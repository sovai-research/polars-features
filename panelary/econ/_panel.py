"""Heterogeneous-panel estimators and panel unit-root / dependence tests.

Everything here is written **clean-room from the published papers** and uses
nothing beyond NumPy and Polars.

Estimators
----------
* :func:`mean_group` -- Pesaran & Smith (1995) Mean Group: run a separate
  time-series regression per entity, average the slopes, and use the *dispersion
  of the slopes* as the variance estimator. Consistent under arbitrary slope
  heterogeneity.
* :func:`cce_mg` / :func:`cce_pooled` -- Pesaran (2006) Common Correlated
  Effects. Unobserved common factors are proxied by the **cross-sectional
  averages** of the dependent variable and the regressors at each date; adding
  those averages to each entity's regression asymptotically annihilates the
  factor loadings. ``cce_mg`` averages the per-entity slopes (CCEMG);
  ``cce_pooled`` partials the augmentation terms out entity by entity
  (Frisch-Waugh) and then pools (CCEP).
* :func:`pmg` -- Pesaran, Shin & Smith (1999) Pooled Mean Group: an ARDL(1,1)
  error-correction model in which the **long-run** coefficients are common
  across entities while the speed of adjustment, short-run dynamics and
  intercepts are entity-specific. Estimated by concentrated (back-fitting) least
  squares, alternating between the per-entity short-run regressions and the
  pooled update of the long-run vector.

Tests
-----
* :func:`llc` -- Levin, Lin & Chu (2002) pooled panel unit root.
* :func:`ips` -- Im, Pesaran & Shin (2003) ``t-bar`` panel unit root.
* :func:`cips` -- Pesaran (2007) cross-sectionally augmented IPS, robust to a
  common factor.
* :func:`pesaran_cd` -- Pesaran (2004/2015) CD test for cross-sectional
  dependence.

**Critical values.** Rather than hard-coding the response-surface tables from
the papers, the null distributions of the LLC / IPS / CIPS statistics are
obtained by a cached, seeded **Monte-Carlo simulation of the identical estimator
pipeline** under an independent-random-walk null for the same ``(N, T, lags)``.
This is self-contained, exactly reproducible, and sidesteps interpolation error
in the published tables. Increase ``reps`` for tighter tail accuracy.

By-products as features
-----------------------
:class:`PanelSlopeFeatures` freezes the train-sample per-entity slopes and joins
them back as entity-level features; :class:`CrossSectionalAverages` emits the
CCE factor proxies (per-date cross-sectional means), which are contemporaneous
and therefore leak-safe.

References
----------
Pesaran, M. H. & Smith, R. (1995), JoE 68(1).
Pesaran, M. H., Shin, Y. & Smith, R. (1999), JASA 94(446).
Levin, A., Lin, C.-F. & Chu, C.-S. J. (2002), JoE 108(1).
Im, K. S., Pesaran, M. H. & Shin, Y. (2003), JoE 115(1).
Pesaran, M. H. (2006), Econometrica 74(4).
Pesaran, M. H. (2007), JAE 22(2).
Pesaran, M. H. (2004/2015), *General diagnostic tests for cross-sectional
dependence in panels*.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame, as_panel
from panelary.core.protocol import PanelTransformer
from panelary.econ._common import (
    factorize,
    norm_cdf,
    norm_ppf,
    ols,
    t_sf,
)

__all__ = [
    "PanelFitResult",
    "PanelTestResult",
    "mean_group",
    "cce_mg",
    "cce_pooled",
    "pmg",
    "llc",
    "ips",
    "cips",
    "pesaran_cd",
    "PanelSlopeFeatures",
    "CrossSectionalAverages",
]


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class PanelFitResult:
    """A fitted heterogeneous-panel estimator.

    Attributes
    ----------
    method : str
        ``"MG"``, ``"CCEMG"``, ``"CCEP"`` or ``"PMG"``.
    names : list of str
        Reported coefficient names.
    params, std_errors, tstats, pvalues : ndarray
    n_entities, n_obs : int
    per_entity : polars.DataFrame
        One row per entity carrying that entity's own estimates (slopes, and for
        PMG the speed of adjustment and short-run terms). These are the
        **entity-level features** the plan asks for.
    extra : dict
        Method-specific extras, e.g. ``{"ec_speed": ..., "n_iter": ...}``.
    """

    method: str
    names: list[str]
    params: np.ndarray
    std_errors: np.ndarray
    tstats: np.ndarray
    pvalues: np.ndarray
    n_entities: int
    n_obs: int
    per_entity: pl.DataFrame
    extra: dict = field(default_factory=dict)

    def summary(self) -> pl.DataFrame:
        """Coefficient table."""
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
        """Two-sided normal confidence intervals."""
        z = float(norm_ppf(1.0 - alpha / 2.0))
        return pl.DataFrame(
            {
                "term": self.names,
                "lower": self.params - z * self.std_errors,
                "upper": self.params + z * self.std_errors,
            }
        )


@dataclass
class PanelTestResult:
    """A panel hypothesis test.

    Attributes
    ----------
    name : str
    statistic : float
        The raw statistic (pooled t, t-bar, CIPS, CD).
    z_statistic : float
        Standardised statistic, asymptotically ``N(0, 1)`` under the null.
    p_value : float
    null : str
        Plain-English statement of the null hypothesis.
    n_entities, n_obs : int
    extra : dict
    """

    name: str
    statistic: float
    z_statistic: float
    p_value: float
    null: str
    n_entities: int
    n_obs: int
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Panel plumbing
# --------------------------------------------------------------------------- #
def _entity_blocks(
    panel: PanelFrame, columns: Sequence[str]
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """Return ``(levels, per-entity index arrays, time codes, stacked matrix)``.

    Rows are sorted by ``(entity, time)`` and rows with a null/NaN in any
    requested column are dropped.
    """
    cols = list(columns)
    frame = (
        panel.sort_panel()
        .lazy()
        .select(
            [
                pl.col(panel.entity_col),
                pl.col(panel.time_col),
                *[pl.col(c).cast(pl.Float64).alias(c) for c in cols],
            ]
        )
        .collect()
        .filter(
            pl.all_horizontal(
                [pl.col(c).is_not_null() & pl.col(c).is_not_nan() for c in cols]
            )
        )
    )
    if frame.height == 0:
        raise ValueError(f"no complete observations in the panel for columns {cols}.")
    ecodes, elevels = factorize(frame[panel.entity_col].to_numpy())
    tcodes, _ = factorize(frame[panel.time_col].to_numpy())
    mat = np.column_stack([frame[c].to_numpy() for c in cols]).astype(float)
    idx = [np.flatnonzero(ecodes == g) for g in range(elevels.size)]
    return elevels, idx, tcodes, mat


def _cross_sectional_means(mat: np.ndarray, tcodes: np.ndarray) -> np.ndarray:
    """Per-date cross-sectional mean of every column, broadcast back to rows."""
    n_times = int(tcodes.max()) + 1
    counts = np.bincount(tcodes, minlength=n_times).astype(float)
    counts[counts == 0.0] = 1.0
    out = np.empty_like(mat)
    for j in range(mat.shape[1]):
        sums = np.bincount(tcodes, weights=mat[:, j], minlength=n_times)
        out[:, j] = (sums / counts)[tcodes]
    return out


def _mg_variance(betas: np.ndarray) -> np.ndarray:
    """Pesaran-Smith non-parametric variance of the mean-group estimator."""
    n = betas.shape[0]
    if n < 2:
        return np.full((betas.shape[1], betas.shape[1]), np.nan)
    dev = betas - betas.mean(axis=0)
    return (dev.T @ dev) / (n * (n - 1.0))


def _finalize_mg(
    method: str,
    names: list[str],
    betas: np.ndarray,
    entity_levels: np.ndarray,
    keep: np.ndarray,
    n_obs: int,
    per_entity_extra: dict | None = None,
) -> PanelFitResult:
    """Average per-entity slopes and attach the MG variance."""
    params = betas.mean(axis=0)
    vcov = _mg_variance(betas)
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstats = np.where(se > 0, params / se, np.nan)
    df = max(betas.shape[0] - 1, 1)
    pvalues = 2.0 * np.asarray(t_sf(np.abs(tstats), df))
    per_entity = pl.DataFrame(
        {
            "entity": pl.Series(np.asarray(entity_levels)[keep]),
            **{f"beta_{n}": betas[:, j] for j, n in enumerate(names)},
            **(per_entity_extra or {}),
        }
    )
    return PanelFitResult(
        method=method,
        names=names,
        params=params,
        std_errors=se,
        tstats=tstats,
        pvalues=pvalues,
        n_entities=int(betas.shape[0]),
        n_obs=n_obs,
        per_entity=per_entity,
        extra={"vcov": vcov},
    )


# --------------------------------------------------------------------------- #
# Mean Group / CCE
# --------------------------------------------------------------------------- #
def _per_entity_ols(
    idx: list[np.ndarray],
    y: np.ndarray,
    design: np.ndarray,
    n_report: int,
    *,
    min_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an OLS per entity; return ``(betas[:, :n_report], keep_mask)``."""
    betas: list[np.ndarray] = []
    keep = np.zeros(len(idx), dtype=bool)
    k = design.shape[1]
    for g, rows in enumerate(idx):
        if rows.size < max(min_obs, k + 1):
            continue
        dg = design[rows]
        if np.linalg.matrix_rank(dg) < k:
            continue
        beta, _, _ = ols(dg, y[rows])
        betas.append(beta[:n_report])
        keep[g] = True
    if not betas:
        raise ValueError(
            "no entity had enough usable observations to fit; check `min_obs` "
            "and the panel's per-entity sample sizes."
        )
    return np.vstack(betas), keep


def mean_group(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    y: str,
    x: Sequence[str],
    entity: str | None = None,
    time: str | None = None,
    intercept: bool = True,
    min_obs: int = 5,
) -> PanelFitResult:
    """Pesaran-Smith (1995) Mean Group estimator.

    Fits ``y_it = a_i + b_i' x_it + e_it`` separately for every entity and
    reports ``b_MG = mean_i(b_i)`` with the non-parametric variance
    ``(1 / (N (N-1))) * sum_i (b_i - b_MG)(b_i - b_MG)'``.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
    y : str
    x : sequence of str
    entity, time : str, optional
    intercept : bool, default=True
    min_obs : int, default=5
        Minimum per-entity observations; entities below it are dropped.

    Returns
    -------
    PanelFitResult
    """
    panel = (
        data
        if isinstance(data, PanelFrame)
        else as_panel(data, entity=entity, time=time)
    )
    x = list(x)
    levels, idx, _, mat = _entity_blocks(panel, [y, *x])
    yv, xv = mat[:, 0], mat[:, 1:]
    design = np.column_stack([np.ones(xv.shape[0]), xv]) if intercept else xv
    off = 1 if intercept else 0
    betas_full, keep = _per_entity_ols(idx, yv, design, off + len(x), min_obs=min_obs)
    return _finalize_mg("MG", list(x), betas_full[:, off:], levels, keep, mat.shape[0])


def cce_mg(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    y: str,
    x: Sequence[str],
    entity: str | None = None,
    time: str | None = None,
    min_obs: int = 5,
) -> PanelFitResult:
    """Pesaran (2006) Common Correlated Effects Mean Group (CCEMG).

    Each entity's regression is augmented with the **cross-sectional averages**
    of the dependent variable and the regressors at the same date, which proxy
    the unobserved common factors:

    ``y_it = a_i + b_i' x_it + c_i * ybar_t + d_i' xbar_t + e_it``.

    The reported coefficient is ``mean_i(b_i)`` with the Mean-Group variance.

    Parameters
    ----------
    data, y, x, entity, time, min_obs :
        As for :func:`mean_group`.

    Returns
    -------
    PanelFitResult
        ``extra["csa_columns"]`` names the factor proxies that were added.
    """
    panel = (
        data
        if isinstance(data, PanelFrame)
        else as_panel(data, entity=entity, time=time)
    )
    x = list(x)
    levels, idx, tcodes, mat = _entity_blocks(panel, [y, *x])
    csa = _cross_sectional_means(mat, tcodes)
    yv, xv = mat[:, 0], mat[:, 1:]
    design = np.column_stack([np.ones(xv.shape[0]), xv, csa])
    betas_full, keep = _per_entity_ols(idx, yv, design, 1 + len(x), min_obs=min_obs)
    res = _finalize_mg("CCEMG", list(x), betas_full[:, 1:], levels, keep, mat.shape[0])
    res.extra["csa_columns"] = [f"{c}_csa" for c in [y, *x]]
    return res


def cce_pooled(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    y: str,
    x: Sequence[str],
    entity: str | None = None,
    time: str | None = None,
    min_obs: int = 5,
) -> PanelFitResult:
    """Pesaran (2006) Common Correlated Effects Pooled estimator (CCEP).

    The entity-specific intercept and factor proxies are partialled out of
    ``y`` and ``x`` **within each entity** (Frisch-Waugh), then a single pooled
    OLS is run on the residualised data. Standard errors are clustered by
    entity, which is the natural analogue of the Mean-Group dispersion estimator
    when the slopes are treated as homogeneous.

    Parameters
    ----------
    data, y, x, entity, time, min_obs :
        As for :func:`mean_group`.

    Returns
    -------
    PanelFitResult
    """
    panel = (
        data
        if isinstance(data, PanelFrame)
        else as_panel(data, entity=entity, time=time)
    )
    x = list(x)
    levels, idx, tcodes, mat = _entity_blocks(panel, [y, *x])
    csa = _cross_sectional_means(mat, tcodes)
    yv, xv = mat[:, 0], mat[:, 1:]
    aug = np.column_stack([np.ones(xv.shape[0]), csa])

    ry_parts, rx_parts, group_ids, kept = [], [], [], []
    for g, rows in enumerate(idx):
        if rows.size < max(min_obs, aug.shape[1] + len(x) + 1):
            continue
        a = aug[rows]
        _, ry, _ = ols(a, yv[rows])
        _, rx, _ = ols(a, xv[rows])
        ry_parts.append(ry)
        rx_parts.append(np.atleast_2d(rx))
        group_ids.append(np.full(rows.size, len(kept), dtype=np.int64))
        kept.append(g)
    if not ry_parts:
        raise ValueError("no entity had enough observations for CCEP.")

    ry = np.concatenate(ry_parts)
    rx = np.vstack(rx_parts)
    gid = np.concatenate(group_ids)
    n, k = rx.shape
    n_groups = len(kept)
    beta, resid, xtx_inv = ols(rx, ry)
    scores = rx * resid[:, None]
    agg = np.zeros((n_groups, k))
    for j in range(k):
        agg[:, j] = np.bincount(gid, weights=scores[:, j], minlength=n_groups)
    meat = agg.T @ agg
    corr = (
        (n_groups / (n_groups - 1.0)) * ((n - 1.0) / max(n - k, 1))
        if n_groups > 1
        else 1.0
    )
    vcov = corr * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstats = np.where(se > 0, beta / se, np.nan)
    pvalues = 2.0 * np.asarray(t_sf(np.abs(tstats), max(n_groups - 1, 1)))
    per_entity = pl.DataFrame({"entity": pl.Series(np.asarray(levels)[kept])})
    return PanelFitResult(
        method="CCEP",
        names=list(x),
        params=beta,
        std_errors=se,
        tstats=tstats,
        pvalues=pvalues,
        n_entities=n_groups,
        n_obs=n,
        per_entity=per_entity,
        extra={"vcov": vcov, "csa_columns": [f"{c}_csa" for c in [y, *x]]},
    )


# --------------------------------------------------------------------------- #
# Pooled Mean Group (ARDL error-correction)
# --------------------------------------------------------------------------- #
def _entity_ec_arrays(
    idx: list[np.ndarray], yv: np.ndarray, xv: np.ndarray
) -> list[dict[str, np.ndarray]]:
    """Build the ARDL(1,1) error-correction pieces for every entity."""
    blocks = []
    for rows in idx:
        if rows.size < 4:
            blocks.append({})
            continue
        y_i = yv[rows]
        x_i = xv[rows]
        blocks.append(
            {
                "dy": np.diff(y_i),
                "y_lag": y_i[:-1],
                "x_lag": x_i[:-1],
                "dx": np.diff(x_i, axis=0),
            }
        )
    return blocks


def pmg(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    y: str,
    x: Sequence[str],
    entity: str | None = None,
    time: str | None = None,
    max_iter: int = 200,
    tol: float = 1e-10,
    min_obs: int = 6,
) -> PanelFitResult:
    """Pesaran-Shin-Smith (1999) Pooled Mean Group estimator.

    Estimates the ARDL(1,1) error-correction representation

    .. math::

        \\Delta y_{it} = \\phi_i \\left( y_{i,t-1} - \\theta' x_{i,t-1} \\right)
                       + \\delta_i' \\Delta x_{it} + \\mu_i + \\varepsilon_{it},

    in which the **long-run vector** ``theta`` is common to all entities while
    the adjustment speed ``phi_i``, short-run coefficients ``delta_i`` and
    intercepts ``mu_i`` are entity-specific.

    The estimator alternates two concentrated least-squares steps until the
    long-run vector stops moving:

    1. **Entity step** -- given ``theta``, form the error-correction term
       ``xi_it = y_{i,t-1} - theta' x_{i,t-1}`` and regress ``dy`` on
       ``[xi, dx, 1]`` separately for each entity to get ``(phi_i, delta_i, mu_i)``.
    2. **Pooled step** -- given ``(phi_i, delta_i, mu_i)``, the model is linear in
       ``theta``: stack ``dy - phi_i y_{lag} - delta_i' dx - mu_i`` on
       ``-phi_i x_{lag}`` across all entities and solve one pooled OLS.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
    y : str
    x : sequence of str
    entity, time : str, optional
    max_iter : int, default=200
    tol : float, default=1e-10
        Convergence tolerance on the maximum change in ``theta``.
    min_obs : int, default=6
        Minimum per-entity observations.

    Returns
    -------
    PanelFitResult
        ``params`` are the long-run coefficients ``theta``; standard errors come
        from the pooled step's information matrix. ``per_entity`` carries
        ``phi_i`` (the entity's speed of adjustment -- a genuinely useful
        entity-level feature) and the short-run ``delta_i``.
        ``extra["ec_speed"]`` is ``mean_i(phi_i)``.

    Raises
    ------
    ValueError
        If no entity has enough observations.
    """
    panel = (
        data
        if isinstance(data, PanelFrame)
        else as_panel(data, entity=entity, time=time)
    )
    x = list(x)
    levels, idx, _, mat = _entity_blocks(panel, [y, *x])
    yv, xv = mat[:, 0], mat[:, 1:]
    k = len(x)
    blocks = _entity_ec_arrays(idx, yv, xv)
    usable = [
        g for g, b in enumerate(blocks) if b and b["dy"].size >= max(min_obs, 2 * k + 2)
    ]
    if not usable:
        raise ValueError("no entity had enough observations for the PMG ARDL model.")

    # Initialise theta from a pooled static long-run regression.
    pooled_x = np.column_stack([np.ones(xv.shape[0]), xv])
    theta = ols(pooled_x, yv)[0][1:]

    phis = np.zeros(len(usable))
    deltas = np.zeros((len(usable), k))
    mus = np.zeros(len(usable))
    n_iter = 0
    n_eff = 0
    for sweep in range(1, max_iter + 1):
        n_iter = sweep
        # --- entity step ---------------------------------------------------
        for i, g in enumerate(usable):
            b = blocks[g]
            xi = b["y_lag"] - b["x_lag"] @ theta
            design = np.column_stack([xi, b["dx"], np.ones(xi.size)])
            coef = ols(design, b["dy"])[0]
            phis[i] = coef[0]
            deltas[i] = coef[1 : 1 + k]
            mus[i] = coef[-1]
        # --- pooled step ---------------------------------------------------
        lhs_parts, rhs_parts = [], []
        for i, g in enumerate(usable):
            b = blocks[g]
            lhs = b["dy"] - phis[i] * b["y_lag"] - b["dx"] @ deltas[i] - mus[i]
            lhs_parts.append(lhs)
            rhs_parts.append(-phis[i] * b["x_lag"])
        lhs = np.concatenate(lhs_parts)
        rhs = np.vstack(rhs_parts)
        n_eff = lhs.size
        new_theta, resid, xtx_inv = ols(rhs, lhs)
        shift = float(np.max(np.abs(new_theta - theta)))
        theta = new_theta
        if shift < tol:
            break

    sigma2 = float(resid @ resid) / max(n_eff - k - 3 * len(usable), 1)
    vcov = sigma2 * xtx_inv
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstats = np.where(se > 0, theta / se, np.nan)
    pvalues = 2.0 * np.asarray(t_sf(np.abs(tstats), max(n_eff - k, 1)))

    per_entity = pl.DataFrame(
        {
            "entity": pl.Series(np.asarray(levels)[usable]),
            "ec_speed": phis,
            "intercept": mus,
            **{f"d_{c}": deltas[:, j] for j, c in enumerate(x)},
        }
    )
    return PanelFitResult(
        method="PMG",
        names=list(x),
        params=theta,
        std_errors=se,
        tstats=tstats,
        pvalues=pvalues,
        n_entities=len(usable),
        n_obs=n_eff,
        per_entity=per_entity,
        extra={
            "ec_speed": float(phis.mean()),
            "n_iter": n_iter,
            "converged": bool(shift < tol),
            "vcov": vcov,
        },
    )


# --------------------------------------------------------------------------- #
# Panel unit-root tests
# --------------------------------------------------------------------------- #
def _adf_design(
    series: np.ndarray, lags: int, trend: str
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(design, dy)`` for the augmented Dickey-Fuller regression."""
    y = np.asarray(series, dtype=float)
    dy_all = np.diff(y)
    n = dy_all.size - lags
    if n <= lags + 3:
        return np.empty((0, 0)), np.empty(0)
    dy = dy_all[lags:]
    cols = [y[lags : lags + n]]
    for j in range(1, lags + 1):
        cols.append(dy_all[lags - j : lags - j + n])
    if trend in ("c", "ct"):
        cols.append(np.ones(n))
    if trend == "ct":
        cols.append(np.arange(n, dtype=float))
    return np.column_stack(cols), dy


def _adf_tstat(series: np.ndarray, lags: int = 0, trend: str = "c") -> float:
    """t-statistic on the lagged level of an ADF regression (NaN if infeasible)."""
    design, dy = _adf_design(series, lags, trend)
    if design.size == 0:
        return float("nan")
    n, k = design.shape
    if n <= k:
        return float("nan")
    beta, resid, xtx_inv = ols(design, dy)
    sigma2 = float(resid @ resid) / (n - k)
    var = sigma2 * xtx_inv[0, 0]
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    return float(beta[0] / np.sqrt(var))


def _cadf_tstat(
    series: np.ndarray, ybar: np.ndarray, lags: int = 0, trend: str = "c"
) -> float:
    """Pesaran (2007) cross-sectionally augmented DF t-statistic."""
    y = np.asarray(series, dtype=float)
    yb = np.asarray(ybar, dtype=float)
    dy_all = np.diff(y)
    dyb_all = np.diff(yb)
    n = dy_all.size - lags
    if n <= lags + 4:
        return float("nan")
    dy = dy_all[lags:]
    cols = [y[lags : lags + n], yb[lags : lags + n], dyb_all[lags:]]
    for j in range(1, lags + 1):
        cols.append(dy_all[lags - j : lags - j + n])
        cols.append(dyb_all[lags - j : lags - j + n])
    if trend in ("c", "ct"):
        cols.append(np.ones(n))
    if trend == "ct":
        cols.append(np.arange(n, dtype=float))
    design = np.column_stack(cols)
    if design.shape[0] <= design.shape[1]:
        return float("nan")
    beta, resid, xtx_inv = ols(design, dy)
    sigma2 = float(resid @ resid) / (design.shape[0] - design.shape[1])
    var = sigma2 * xtx_inv[0, 0]
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    return float(beta[0] / np.sqrt(var))


def _balanced_matrix(
    panel: PanelFrame, value: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot a panel into a ``(T, N)`` matrix (NaN where an observation is missing)."""
    frame = (
        panel.sort_panel()
        .lazy()
        .select(
            [
                pl.col(panel.entity_col),
                pl.col(panel.time_col),
                pl.col(value).cast(pl.Float64).alias(value),
            ]
        )
        .collect()
    )
    ecodes, elevels = factorize(frame[panel.entity_col].to_numpy())
    tcodes, tlevels = factorize(frame[panel.time_col].to_numpy())
    mat = np.full((tlevels.size, elevels.size), np.nan)
    mat[tcodes, ecodes] = frame[value].to_numpy()
    return mat, elevels, tlevels


@lru_cache(maxsize=64)
def _adf_null_moments(
    t_obs: int, lags: int, trend: str, reps: int, seed: int
) -> tuple[float, float]:
    """Monte-Carlo mean/sd of a single ADF t-statistic under a random-walk null."""
    rng = np.random.default_rng(seed)
    walks = np.cumsum(rng.standard_normal((reps, t_obs)), axis=1)
    stats = np.array([_adf_tstat(walks[r], lags, trend) for r in range(reps)])
    stats = stats[np.isfinite(stats)]
    if stats.size < 10:
        return float("nan"), float("nan")
    return float(stats.mean()), float(stats.std(ddof=1))


@lru_cache(maxsize=64)
def _pooled_null_moments(
    kind: str, n_entities: int, t_obs: int, lags: int, trend: str, reps: int, seed: int
) -> tuple[float, float]:
    """Monte-Carlo mean/sd of a pooled panel statistic under an independent-RW null."""
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        walks = np.cumsum(rng.standard_normal((t_obs, n_entities)), axis=0)
        if kind == "llc":
            out[r] = _llc_statistic(walks, lags, trend)
        elif kind == "ips":
            out[r] = float(
                np.nanmean(
                    [_adf_tstat(walks[:, i], lags, trend) for i in range(n_entities)]
                )
            )
        elif kind == "cips":
            ybar = np.nanmean(walks, axis=1)
            out[r] = float(
                np.nanmean(
                    [
                        _cadf_tstat(walks[:, i], ybar, lags, trend)
                        for i in range(n_entities)
                    ]
                )
            )
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"unknown null-simulation kind {kind!r}.")
    out = out[np.isfinite(out)]
    if out.size < 10:
        return float("nan"), float("nan")
    return float(out.mean()), float(out.std(ddof=1))


def _llc_statistic(mat: np.ndarray, lags: int, trend: str) -> float:
    """Pooled Levin-Lin-Chu t-statistic on a ``(T, N)`` matrix (NaN-tolerant)."""
    e_parts, v_parts = [], []
    for i in range(mat.shape[1]):
        col = mat[:, i]
        col = col[np.isfinite(col)]
        design, dy = _adf_design(col, lags, trend)
        if design.size == 0 or design.shape[0] <= design.shape[1]:
            continue
        aux = design[:, 1:]
        y_lag = design[:, 0]
        if aux.shape[1] > 0:
            e = ols(aux, dy)[1]
            v = ols(aux, y_lag)[1]
        else:
            e, v = dy, y_lag
        sd = float(np.std(e, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            continue
        e_parts.append(e / sd)
        v_parts.append(v / sd)
    if not e_parts:
        return float("nan")
    e = np.concatenate(e_parts)
    v = np.concatenate(v_parts)
    denom = float(v @ v)
    if denom <= 0:
        return float("nan")
    rho = float(v @ e) / denom
    resid = e - rho * v
    sigma2 = float(resid @ resid) / max(e.size - 1, 1)
    se = float(np.sqrt(sigma2 / denom))
    if not np.isfinite(se) or se <= 0:
        return float("nan")
    return rho / se


def llc(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    value: str,
    entity: str | None = None,
    time: str | None = None,
    lags: int = 0,
    trend: str = "c",
    reps: int = 400,
    seed: int = 0,
) -> PanelTestResult:
    """Levin-Lin-Chu (2002) pooled panel unit-root test.

    Each entity's ADF regression is orthogonalised against its deterministic and
    lagged-difference terms and rescaled by its own residual standard deviation;
    the standardised residuals are then pooled into a single t-statistic on the
    common autoregressive parameter. The null is a **common unit root** for every
    entity; the alternative is a common stationary root.

    The p-value is one-sided (left tail) and is computed by standardising the
    pooled t-statistic with Monte-Carlo moments simulated from the *same*
    pipeline under an independent-random-walk null.

    Parameters
    ----------
    data, value, entity, time : ...
    lags : int, default=0
        Lagged differences included in each ADF regression.
    trend : {"n", "c", "ct"}, default="c"
    reps : int, default=400
        Monte-Carlo replications for the null moments (cached per configuration).
    seed : int, default=0

    Returns
    -------
    PanelTestResult
    """
    panel = (
        data
        if isinstance(data, PanelFrame)
        else as_panel(data, entity=entity, time=time)
    )
    mat, elevels, tlevels = _balanced_matrix(panel, value)
    stat = _llc_statistic(mat, lags, trend)
    mu, sd = _pooled_null_moments(
        "llc", int(mat.shape[1]), int(mat.shape[0]), lags, trend, reps, seed
    )
    z = (stat - mu) / sd if np.isfinite(sd) and sd > 0 else float("nan")
    p = float(norm_cdf(z)) if np.isfinite(z) else float("nan")
    return PanelTestResult(
        name="Levin-Lin-Chu",
        statistic=stat,
        z_statistic=float(z),
        p_value=p,
        null="all entities share a unit root",
        n_entities=int(elevels.size),
        n_obs=int(np.isfinite(mat).sum()),
        extra={"lags": lags, "trend": trend, "null_mean": mu, "null_sd": sd},
    )


def ips(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    value: str,
    entity: str | None = None,
    time: str | None = None,
    lags: int = 0,
    trend: str = "c",
    reps: int = 400,
    seed: int = 0,
) -> PanelTestResult:
    """Im-Pesaran-Shin (2003) ``t-bar`` panel unit-root test.

    Runs a separate ADF regression per entity and averages the t-statistics.
    Under the null every entity has a unit root; under the alternative a
    non-vanishing fraction of entities is stationary (heterogeneous roots are
    allowed, unlike LLC).

    ``t-bar`` is standardised with Monte-Carlo moments of the individual ADF
    statistic (which is what the published ``E[t]``/``Var[t]`` tables tabulate),
    giving ``Z = sqrt(N) (tbar - E[t]) / sqrt(Var[t]) -> N(0, 1)``.

    Parameters
    ----------
    data, value, entity, time, lags, trend, reps, seed : ...

    Returns
    -------
    PanelTestResult
        ``extra["per_entity"]`` holds each entity's own ADF t-statistic -- a
        ready-made persistence feature.
    """
    panel = (
        data
        if isinstance(data, PanelFrame)
        else as_panel(data, entity=entity, time=time)
    )
    mat, elevels, _ = _balanced_matrix(panel, value)
    stats, kept = [], []
    lengths = []
    for i in range(mat.shape[1]):
        col = mat[:, i][np.isfinite(mat[:, i])]
        s = _adf_tstat(col, lags, trend)
        if np.isfinite(s):
            stats.append(s)
            kept.append(elevels[i])
            lengths.append(col.size)
    if not stats:
        raise ValueError("no entity produced a usable ADF regression for IPS.")
    stats_arr = np.asarray(stats)
    tbar = float(stats_arr.mean())
    n_kept = stats_arr.size
    t_typ = int(np.median(lengths))
    mu, sd = _adf_null_moments(t_typ, lags, trend, reps, seed)
    z = (
        np.sqrt(n_kept) * (tbar - mu) / sd
        if np.isfinite(sd) and sd > 0
        else float("nan")
    )
    p = float(norm_cdf(z)) if np.isfinite(z) else float("nan")
    return PanelTestResult(
        name="Im-Pesaran-Shin t-bar",
        statistic=tbar,
        z_statistic=float(z),
        p_value=p,
        null="every entity has a unit root",
        n_entities=n_kept,
        n_obs=int(np.isfinite(mat).sum()),
        extra={
            "lags": lags,
            "trend": trend,
            "null_mean": mu,
            "null_sd": sd,
            "per_entity": pl.DataFrame(
                {"entity": pl.Series(np.asarray(kept)), "adf_t": stats_arr}
            ),
        },
    )


def cips(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    value: str,
    entity: str | None = None,
    time: str | None = None,
    lags: int = 0,
    trend: str = "c",
    reps: int = 400,
    seed: int = 0,
) -> PanelTestResult:
    """Pesaran (2007) cross-sectionally augmented IPS (CIPS) test.

    Each entity's DF regression is augmented with the cross-sectional average of
    the level and of the first difference, which removes a single common factor.
    ``CIPS = mean_i(CADF_i)``. Robust to the cross-sectional dependence that
    invalidates LLC and IPS.

    Parameters
    ----------
    data, value, entity, time, lags, trend, reps, seed : ...

    Returns
    -------
    PanelTestResult
        ``extra["per_entity"]`` holds the individual CADF statistics.
    """
    panel = (
        data
        if isinstance(data, PanelFrame)
        else as_panel(data, entity=entity, time=time)
    )
    mat, elevels, _ = _balanced_matrix(panel, value)
    with np.errstate(invalid="ignore"):
        ybar = np.nanmean(mat, axis=1)
    stats, kept = [], []
    for i in range(mat.shape[1]):
        col = mat[:, i]
        good = np.isfinite(col) & np.isfinite(ybar)
        s = _cadf_tstat(col[good], ybar[good], lags, trend)
        if np.isfinite(s):
            stats.append(s)
            kept.append(elevels[i])
    if not stats:
        raise ValueError("no entity produced a usable CADF regression for CIPS.")
    stats_arr = np.asarray(stats)
    cips_stat = float(stats_arr.mean())
    mu, sd = _pooled_null_moments(
        "cips", int(mat.shape[1]), int(mat.shape[0]), lags, trend, reps, seed
    )
    z = (cips_stat - mu) / sd if np.isfinite(sd) and sd > 0 else float("nan")
    p = float(norm_cdf(z)) if np.isfinite(z) else float("nan")
    return PanelTestResult(
        name="Pesaran CIPS",
        statistic=cips_stat,
        z_statistic=float(z),
        p_value=p,
        null="every entity has a unit root (common factor allowed)",
        n_entities=int(stats_arr.size),
        n_obs=int(np.isfinite(mat).sum()),
        extra={
            "lags": lags,
            "trend": trend,
            "null_mean": mu,
            "null_sd": sd,
            "per_entity": pl.DataFrame(
                {"entity": pl.Series(np.asarray(kept)), "cadf_t": stats_arr}
            ),
        },
    )


def pesaran_cd(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    value: str,
    entity: str | None = None,
    time: str | None = None,
    min_overlap: int = 3,
) -> PanelTestResult:
    """Pesaran CD test for cross-sectional dependence.

    ``CD = sqrt(2 / (N (N-1))) * sum_{i<j} sqrt(T_ij) * rho_ij``, where
    ``rho_ij`` is the pairwise correlation of the two entities' series over
    their ``T_ij`` overlapping dates. Under the null of cross-sectional
    independence, ``CD -> N(0, 1)``. The test is two-sided.

    Feed it regression residuals (e.g. ``HDFEResult.resid`` joined back onto the
    panel) to test whether fixed effects have removed the common factor -- a
    rejection is the signal to switch to a CCE estimator.

    Parameters
    ----------
    data, value, entity, time : ...
    min_overlap : int, default=3
        Minimum overlapping observations for a pair to contribute.

    Returns
    -------
    PanelTestResult
        ``extra["mean_abs_corr"]`` is the average absolute pairwise correlation.
    """
    panel = (
        data
        if isinstance(data, PanelFrame)
        else as_panel(data, entity=entity, time=time)
    )
    mat, elevels, _ = _balanced_matrix(panel, value)
    n = mat.shape[1]
    if n < 2:
        raise ValueError("the CD test needs at least two entities.")
    total = 0.0
    abs_total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            good = np.isfinite(mat[:, i]) & np.isfinite(mat[:, j])
            t_ij = int(good.sum())
            if t_ij < min_overlap:
                continue
            a = mat[good, i] - mat[good, i].mean()
            b = mat[good, j] - mat[good, j].mean()
            denom = np.sqrt(float(a @ a) * float(b @ b))
            if denom <= 0:
                continue
            rho = float(a @ b) / denom
            total += np.sqrt(t_ij) * rho
            abs_total += abs(rho)
            pairs += 1
    if pairs == 0:
        raise ValueError("no entity pair had enough overlapping observations.")
    cd = float(np.sqrt(2.0 / (n * (n - 1.0))) * total)
    p = 2.0 * (1.0 - float(norm_cdf(abs(cd))))
    return PanelTestResult(
        name="Pesaran CD",
        statistic=cd,
        z_statistic=cd,
        p_value=p,
        null="no cross-sectional dependence",
        n_entities=n,
        n_obs=int(np.isfinite(mat).sum()),
        extra={"pairs": pairs, "mean_abs_corr": abs_total / pairs},
    )


# --------------------------------------------------------------------------- #
# Feature transformers
# --------------------------------------------------------------------------- #
class PanelSlopeFeatures(PanelTransformer):
    """Freeze train-sample per-entity slopes and join them back as features.

    ``fit`` runs one of the heterogeneous-panel estimators on the training rows
    and stores the resulting per-entity coefficient table. ``transform`` joins
    those *frozen* coefficients onto the panel by entity. Because the
    coefficients never see a test row, the emitted columns are leak-safe.

    Parameters
    ----------
    y : str
    x : sequence of str
    method : {"mg", "ccemg", "pmg"}, default="mg"
    prefix : str, default="mg_"
    unseen : {"mean", "null"}, default="mean"
        Value used for an entity absent from the training sample: either the
        train-sample mean coefficient or ``null``.

    Attributes
    ----------
    result_ : PanelFitResult
    coef_table_ : polars.DataFrame
    """

    panel_safe = True
    leakage_safe = True

    _METHODS = {"mg": mean_group, "ccemg": cce_mg, "pmg": pmg}

    def __init__(
        self,
        *,
        y: str,
        x: Sequence[str],
        method: str = "mg",
        prefix: str = "mg_",
        unseen: str = "mean",
        entity: str | None = None,
        time: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if method not in self._METHODS:
            raise ValueError(
                f"unknown `method={method!r}`; expected one of {sorted(self._METHODS)}."
            )
        if unseen not in ("mean", "null"):
            raise ValueError(f"`unseen` must be 'mean' or 'null', got {unseen!r}.")
        self.y = y
        self.x = list(x)
        self.method = method
        self.prefix = prefix
        self.unseen = unseen
        self.kwargs = dict(kwargs)
        self.result_: PanelFitResult | None = None
        self.coef_table_: pl.DataFrame | None = None

    def _fit(self, panel: PanelFrame) -> None:
        fn = self._METHODS[self.method]
        res = fn(panel, y=self.y, x=self.x, **self.kwargs)  # type: ignore[operator]
        table = res.per_entity.rename({"entity": panel.entity_col})
        renames = {
            c: f"{self.prefix}{c}" for c in table.columns if c != panel.entity_col
        }
        self.result_ = res
        self.coef_table_ = table.rename(renames)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        assert self.coef_table_ is not None  # noqa: S101 - guarded by `_check_fitted`
        table = self.coef_table_
        lf = panel.lazy().join(
            table.lazy().cast({panel.entity_col: panel.schema[panel.entity_col]}),
            on=panel.entity_col,
            how="left",
        )
        if self.unseen == "mean":
            fills = [
                pl.col(c).fill_null(float(np.nanmean(table[c].to_numpy())))
                for c in table.columns
                if c != panel.entity_col
            ]
            lf = lf.with_columns(fills)
        return PanelFrame(
            lf, entity=panel.entity_col, time=panel.time_col, validate=False
        )

    @property
    def output_names(self) -> list[str]:
        """Names of the emitted per-entity coefficient columns."""
        if self.coef_table_ is None:
            return []
        return [c for c in self.coef_table_.columns if c.startswith(self.prefix)]


class CrossSectionalAverages(PanelTransformer):
    """Emit Pesaran-CCE factor proxies: per-date cross-sectional means.

    For every requested column this appends ``<col><suffix>``, the mean of that
    column across all entities observed on the same date. These are the
    augmentation terms of the CCE estimators and are useful features in their own
    right (they proxy the unobserved common factor driving the panel).

    They are **contemporaneous**, not forward-looking: date ``t``'s proxy uses
    only date-``t`` observations, so no future information enters. ``fit`` only
    records the column list; there is no fitted statistic that could leak.

    Parameters
    ----------
    columns : sequence of str
    suffix : str, default="_csa"
    demean : bool, default=False
        Also emit ``<col><suffix>_dev``, the entity's deviation from the
        cross-sectional average.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        columns: Sequence[str],
        *,
        suffix: str = "_csa",
        demean: bool = False,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self.columns = list(columns)
        if not self.columns:
            raise ValueError("`columns` must name at least one column.")
        self.suffix = suffix
        self.demean = bool(demean)

    def _fit(self, panel: PanelFrame) -> None:
        missing = [c for c in self.columns if c not in panel.columns]
        if missing:
            raise ValueError(
                f"column(s) {missing} not found in panel. Available: {panel.columns}."
            )

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        exprs = [
            pl.col(c)
            .cast(pl.Float64)
            .mean()
            .over(panel.time_col)
            .alias(f"{c}{self.suffix}")
            for c in self.columns
        ]
        out = panel.with_columns(exprs)
        if self.demean:
            out = out.with_columns(
                [
                    (pl.col(c).cast(pl.Float64) - pl.col(f"{c}{self.suffix}")).alias(
                        f"{c}{self.suffix}_dev"
                    )
                    for c in self.columns
                ]
            )
        return out

    @property
    def output_names(self) -> list[str]:
        """Names of the emitted proxy columns."""
        names = [f"{c}{self.suffix}" for c in self.columns]
        if self.demean:
            names += [f"{c}{self.suffix}_dev" for c in self.columns]
        return names
