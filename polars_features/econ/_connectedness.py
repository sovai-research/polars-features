"""Diebold-Yilmaz connectedness: generalised-FEVD spillover networks.

A **clean-room implementation from the published papers** (Diebold & Yilmaz
2009/2012/2014; the generalised forecast-error variance decomposition of Koop,
Pesaran & Potter 1996 and Pesaran & Shin 1998). No GPL R source was consulted.

The idea
--------
Fit a VAR(``p``) to a cross-section of ``K`` series, invert it to its infinite
moving-average representation, and decompose each series' ``H``-step forecast
error variance into the shares attributable to shocks in every series. The
resulting ``K x K`` table *is* a weighted, directed network:

* **row ``i``** -- where entity ``i``'s forecast uncertainty comes **FROM**;
* **column ``j``** -- how much entity ``j`` transmits **TO** everyone else;
* **total connectedness** -- the share of total forecast error variance that
  crosses entity boundaries;
* **net** -- ``TO_i - FROM_i``: is ``i`` a net transmitter or a net receiver?

Generalised (rather than Cholesky) decomposition is used, so the answer does not
depend on the ordering of the series -- the reason Diebold and Yilmaz adopted it
in 2012.

The equations
-------------
With ``y_t = sum_{i=1}^{p} A_i y_{t-i} + eps_t``, ``Var(eps) = Sigma``, the
moving-average coefficients follow the Wold recursion

``Psi_0 = I``,  ``Psi_h = sum_{i=1}^{min(h,p)} A_i Psi_{h-i}``.

The generalised variance decomposition is

.. math::

    \\theta_{ij}(H) = \\frac{\\sigma_{jj}^{-1}
        \\sum_{h=0}^{H-1} \\left( e_i' \\Psi_h \\Sigma e_j \\right)^2}
        {\\sum_{h=0}^{H-1} e_i' \\Psi_h \\Sigma \\Psi_h' e_i},

which does **not** sum to one across ``j`` (the generalised shocks are not
orthogonal), so it is row-normalised,
``theta~_ij = theta_ij / sum_k theta_ik``. Every reported quantity is built from
the normalised table, which guarantees the structural identities:
rows sum to 100%, and total connectedness lies in ``[0, 100]``.

Rolling estimation
------------------
:func:`rolling_connectedness` re-estimates the VAR on a **trailing** window
ending at each date and stamps the resulting network statistics on that date.
Nothing from the future enters, so the emitted columns
(``dy_to``, ``dy_from``, ``dy_net``, ``dy_total``) are leak-safe cross-entity
features -- exactly the kind no per-entity feature library can produce.

References
----------
Diebold, F. X. & Yilmaz, K. (2012). *Better to give than to receive: predictive
directional measurement of volatility spillovers*, IJF 28(1).
Diebold, F. X. & Yilmaz, K. (2014). *On the network topology of variance
decompositions*, JoE 182(1).
Koop, G., Pesaran, M. H. & Potter, S. (1996), JoE 74(1).
Pesaran, M. H. & Shin, Y. (1998). *Generalized impulse response analysis in
linear multivariate models*, Economics Letters 58(1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from polars_features.core.panel_frame import PanelFrame
from polars_features.core.protocol import PanelTransformer
from polars_features.econ._common import factorize, pinv_sym

__all__ = [
    "var_ols",
    "ma_coefficients",
    "generalized_fevd",
    "connectedness",
    "ConnectednessResult",
    "rolling_connectedness",
    "ConnectednessFeatures",
]


# --------------------------------------------------------------------------- #
# VAR backend
# --------------------------------------------------------------------------- #
def var_ols(
    y: np.ndarray, lags: int = 1, *, trend: str = "c"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a reduced-form VAR(``lags``) by equation-by-equation OLS.

    Parameters
    ----------
    y : ndarray, shape (T, K)
        Multivariate series, rows ordered in time.
    lags : int, default=1
        VAR order ``p``.
    trend : {"n", "c"}, default="c"
        Include an intercept.

    Returns
    -------
    coefs : ndarray, shape (lags, K, K)
        ``coefs[i]`` is ``A_{i+1}``, so ``y_t ~= sum_i A_i y_{t-i}``.
    sigma : ndarray, shape (K, K)
        Residual covariance, degrees-of-freedom corrected.
    resid : ndarray, shape (T - lags, K)

    Raises
    ------
    ValueError
        If there are too few observations for the requested order.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 2:
        raise ValueError(f"`y` must be a (T, K) matrix, got shape {y.shape}.")
    t_obs, k = y.shape
    if lags < 1:
        raise ValueError(f"`lags` must be >= 1, got {lags}.")
    n = t_obs - lags
    n_params = k * lags + (1 if trend == "c" else 0)
    if n <= n_params:
        raise ValueError(
            f"a VAR({lags}) on {k} series needs more than {n_params} usable "
            f"observations, but only {max(n, 0)} are available."
        )
    blocks = [y[lags - i - 1 : t_obs - i - 1] for i in range(lags)]
    x = np.hstack(blocks)
    if trend == "c":
        x = np.hstack([np.ones((n, 1)), x])
    yy = y[lags:]
    beta = pinv_sym(x.T @ x) @ (x.T @ yy)
    resid = yy - x @ beta
    sigma = (resid.T @ resid) / max(n - n_params, 1)
    off = 1 if trend == "c" else 0
    coefs = np.empty((lags, k, k))
    for i in range(lags):
        coefs[i] = beta[off + i * k : off + (i + 1) * k, :].T
    return coefs, sigma, resid


def ma_coefficients(coefs: np.ndarray, horizon: int) -> np.ndarray:
    """Wold moving-average coefficients ``Psi_0 .. Psi_{horizon-1}``.

    Parameters
    ----------
    coefs : ndarray, shape (p, K, K)
        VAR coefficient matrices ``A_1 .. A_p``.
    horizon : int
        Number of MA matrices to return (``>= 1``).

    Returns
    -------
    ndarray, shape (horizon, K, K)
    """
    coefs = np.asarray(coefs, dtype=float)
    p, k, _ = coefs.shape
    if horizon < 1:
        raise ValueError(f"`horizon` must be >= 1, got {horizon}.")
    psi = np.zeros((horizon, k, k))
    psi[0] = np.eye(k)
    for h in range(1, horizon):
        acc = np.zeros((k, k))
        for i in range(1, min(h, p) + 1):
            acc += coefs[i - 1] @ psi[h - i]
        psi[h] = acc
    return psi


def generalized_fevd(
    coefs: np.ndarray, sigma: np.ndarray, horizon: int = 10
) -> np.ndarray:
    """Row-normalised generalised forecast-error variance decomposition.

    Parameters
    ----------
    coefs : ndarray, shape (p, K, K)
    sigma : ndarray, shape (K, K)
        Residual covariance.
    horizon : int, default=10
        Forecast horizon ``H``.

    Returns
    -------
    ndarray, shape (K, K)
        ``theta~[i, j]`` is the share of entity ``i``'s ``H``-step forecast error
        variance explained by shocks to entity ``j``. **Every row sums to
        exactly 1.**
    """
    sigma = np.asarray(sigma, dtype=float)
    psi = ma_coefficients(coefs, horizon)
    k = sigma.shape[0]
    sigma_jj = np.diag(sigma).astype(float)
    num = np.zeros((k, k))
    den = np.zeros(k)
    for h in range(horizon):
        ps = psi[h] @ sigma  # (K, K); entry (i, j) = e_i' Psi_h Sigma e_j
        num += ps**2
        den += np.einsum("ij,ij->i", ps, psi[h])
    safe_jj = np.where(sigma_jj > 0, sigma_jj, np.inf)
    theta = num / safe_jj[None, :]
    den = np.where(den > 0, den, np.inf)
    theta = theta / den[:, None]
    row_sums = theta.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return theta / row_sums


@dataclass
class ConnectednessResult:
    """A Diebold-Yilmaz connectedness network.

    Attributes
    ----------
    names : list of str
        Entity names, in the row/column order of :attr:`table`.
    table : ndarray, shape (K, K)
        Row-normalised generalised FEVD **in percent**; every row sums to 100.
    total : float
        Total connectedness index (the "spillover index"), in ``[0, 100]``:
        the share of forecast error variance coming from *other* entities.
    to_others, from_others, net : ndarray, shape (K,)
        Directional connectedness in percent. ``net = to_others - from_others``.
    net_pairwise : ndarray, shape (K, K)
        ``100 * (theta~_ji - theta~_ij) / K``: the net bilateral flow from
        ``j`` to ``i``. Antisymmetric.
    horizon, lags : int
    """

    names: list[str]
    table: np.ndarray
    total: float
    to_others: np.ndarray
    from_others: np.ndarray
    net: np.ndarray
    net_pairwise: np.ndarray
    horizon: int
    lags: int

    def to_frame(self) -> pl.DataFrame:
        """Per-entity directional connectedness as a tidy table."""
        return pl.DataFrame(
            {
                "entity": self.names,
                "to_others": np.asarray(self.to_others, dtype=float),
                "from_others": np.asarray(self.from_others, dtype=float),
                "net": np.asarray(self.net, dtype=float),
            }
        )

    def spillover_table(self) -> pl.DataFrame:
        """The full ``K x K`` FEVD table with row/column labels (percent)."""
        data = {"from\\to": self.names}
        for j, name in enumerate(self.names):
            data[name] = self.table[:, j]
        return pl.DataFrame(data)


def _summarise(table: np.ndarray, names: list[str], horizon: int, lags: int):
    """Turn a normalised FEVD table into directional connectedness measures."""
    k = table.shape[0]
    pct = 100.0 * table
    off = pct - np.diag(np.diag(pct))
    from_others = off.sum(axis=1)
    to_others = off.sum(axis=0)
    total = float(off.sum() / k)
    net_pairwise = (pct.T - pct) / k
    return ConnectednessResult(
        names=names,
        table=pct,
        total=total,
        to_others=to_others,
        from_others=from_others,
        net=to_others - from_others,
        net_pairwise=net_pairwise,
        horizon=horizon,
        lags=lags,
    )


def connectedness(
    y: np.ndarray,
    *,
    names: Sequence[str] | None = None,
    lags: int = 1,
    horizon: int = 10,
    trend: str = "c",
) -> ConnectednessResult:
    """Estimate a Diebold-Yilmaz connectedness network from a ``(T, K)`` matrix.

    Parameters
    ----------
    y : ndarray, shape (T, K)
        The cross-section of series (returns, realised volatilities, ...).
    names : sequence of str, optional
        Entity labels; defaults to ``["0", "1", ...]``.
    lags : int, default=1
        VAR order.
    horizon : int, default=10
        Forecast horizon for the variance decomposition.
    trend : {"n", "c"}, default="c"

    Returns
    -------
    ConnectednessResult

    Examples
    --------
    >>> res = connectedness(vol_matrix, lags=2, horizon=12)  # doctest: +SKIP
    >>> res.total, res.net  # doctest: +SKIP
    """
    y = np.asarray(y, dtype=float)
    coefs, sigma, _ = var_ols(y, lags, trend=trend)
    table = generalized_fevd(coefs, sigma, horizon)
    labels = list(names) if names is not None else [str(i) for i in range(y.shape[1])]
    return _summarise(table, labels, horizon, lags)


def connectedness_from_var(
    coefs: np.ndarray,
    sigma: np.ndarray,
    *,
    names: Sequence[str] | None = None,
    horizon: int = 10,
) -> ConnectednessResult:
    """Build a connectedness network from *known* VAR parameters.

    Useful for simulation studies and for the structural-identity tests: pass a
    block-diagonal ``coefs`` and diagonal ``sigma`` and every off-diagonal
    spillover is exactly zero.
    """
    coefs = np.asarray(coefs, dtype=float)
    table = generalized_fevd(coefs, sigma, horizon)
    labels = (
        list(names) if names is not None else [str(i) for i in range(coefs.shape[1])]
    )
    return _summarise(table, labels, horizon, coefs.shape[0])


# --------------------------------------------------------------------------- #
# Rolling / panel interface
# --------------------------------------------------------------------------- #
def _pivot(panel: PanelFrame, value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot a long panel into a ``(T, K)`` matrix plus its entity/time labels."""
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


def rolling_connectedness(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    value: str,
    window: int = 100,
    lags: int = 1,
    horizon: int = 10,
    entity: str | None = None,
    time: str | None = None,
    min_periods: int | None = None,
    prefix: str = "dy_",
) -> pl.DataFrame:
    """Roll a VAR over a trailing window and emit connectedness features per date.

    For every date ``t`` with at least ``min_periods`` observations in the
    trailing window ``(t - window, t]``, a VAR is estimated on that window only
    and the resulting network statistics are stamped on date ``t``. This is
    **strictly backward-looking**: the value at date ``t`` never uses an
    observation from after ``t``.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
    value : str
        The series to build the network on (returns, realised vol, ...).
    window : int, default=100
        Trailing window length in dates.
    lags : int, default=1
        VAR order inside each window.
    horizon : int, default=10
        FEVD horizon.
    entity, time : str, optional
    min_periods : int, optional
        Minimum observations in the window. Defaults to ``window``.
    prefix : str, default="dy_"

    Returns
    -------
    polars.DataFrame
        Long format, keyed by ``(entity, time)``, with ``<prefix>to``,
        ``<prefix>from``, ``<prefix>net`` and ``<prefix>total`` columns.
        Ready to ``join`` back onto the panel.

    Notes
    -----
    Dates before the window fills produce no rows (join with ``how="left"`` and
    the early rows get nulls). Entities that are entirely missing inside a window
    are dropped from that window's network.
    """
    from polars_features.core.panel_frame import as_panel

    panel = (
        data
        if isinstance(data, PanelFrame)
        else as_panel(data, entity=entity, time=time)
    )
    mat, elevels, tlevels = _pivot(panel, value)
    t_total, k = mat.shape
    if window < lags + k + 2:
        raise ValueError(
            f"`window={window}` is too short for a VAR({lags}) on {k} series; "
            f"use at least {lags + k + 2} dates."
        )
    min_p = int(min_periods) if min_periods is not None else int(window)

    out_entity: list = []
    out_time: list = []
    out_to: list[float] = []
    out_from: list[float] = []
    out_net: list[float] = []
    out_total: list[float] = []

    for end in range(min_p - 1, t_total):
        start = max(0, end - window + 1)
        block = mat[start : end + 1]
        if block.shape[0] < min_p:
            continue
        good = np.isfinite(block).all(axis=0)
        if good.sum() < 2:
            continue
        sub = block[:, good]
        try:
            res = connectedness(
                sub,
                names=[str(v) for v in np.asarray(elevels)[good]],
                lags=lags,
                horizon=horizon,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        idxs = np.flatnonzero(good)
        for pos, ent_idx in enumerate(idxs):
            out_entity.append(elevels[ent_idx])
            out_time.append(tlevels[end])
            out_to.append(float(res.to_others[pos]))
            out_from.append(float(res.from_others[pos]))
            out_net.append(float(res.net[pos]))
            out_total.append(res.total)

    return pl.DataFrame(
        {
            panel.entity_col: pl.Series(np.asarray(out_entity)),
            panel.time_col: pl.Series(np.asarray(out_time)),
            f"{prefix}to": np.asarray(out_to, dtype=float),
            f"{prefix}from": np.asarray(out_from, dtype=float),
            f"{prefix}net": np.asarray(out_net, dtype=float),
            f"{prefix}total": np.asarray(out_total, dtype=float),
        }
    )


class ConnectednessFeatures(PanelTransformer):
    """Attach rolling Diebold-Yilmaz network statistics to a panel.

    The features are computed on **trailing** windows, so each row's value uses
    only that entity's (and its peers') past -- the same guarantee any rolling
    feature carries. ``fit`` records the entity universe and window settings;
    nothing is estimated at fit time that could leak, and ``transform``
    recomputes the trailing windows on whatever rows it is given.

    Parameters
    ----------
    value : str
        Column to build the network on.
    window : int, default=100
    lags : int, default=1
    horizon : int, default=10
    prefix : str, default="dy_"

    Attributes
    ----------
    entities_ : ndarray
        The entity universe seen at ``fit`` time (informational).

    Warnings
    --------
    A transform applied to a *test fold alone* only sees that fold's history, so
    the first ``window - 1`` dates of the fold get nulls. Inside cross-validation
    prefer transforming the concatenated ``train + test`` frame **after** the
    split has fixed which rows are scored, or accept the warm-up nulls.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        *,
        value: str,
        window: int = 100,
        lags: int = 1,
        horizon: int = 10,
        prefix: str = "dy_",
        min_periods: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self.value = value
        self.window = int(window)
        self.lags = int(lags)
        self.horizon = int(horizon)
        self.prefix = prefix
        self.min_periods = min_periods
        self.entities_: np.ndarray | None = None

    def _fit(self, panel: PanelFrame) -> None:
        if self.value not in panel.columns:
            raise ValueError(
                f"column {self.value!r} not found in panel. Available: {panel.columns}."
            )
        self.entities_ = (
            panel.lazy()
            .select(pl.col(panel.entity_col).unique())
            .collect()
            .to_series()
            .to_numpy()
        )

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        feats = rolling_connectedness(
            panel,
            value=self.value,
            window=self.window,
            lags=self.lags,
            horizon=self.horizon,
            min_periods=self.min_periods,
            prefix=self.prefix,
        )
        schema = panel.schema
        feats = feats.cast(
            {
                panel.entity_col: schema[panel.entity_col],
                panel.time_col: schema[panel.time_col],
            }
        )
        lf = panel.lazy().join(
            feats.lazy(), on=[panel.entity_col, panel.time_col], how="left"
        )
        return PanelFrame(
            lf, entity=panel.entity_col, time=panel.time_col, validate=False
        )

    @property
    def output_names(self) -> list[str]:
        """Names of the emitted network-feature columns."""
        return [f"{self.prefix}{s}" for s in ("to", "from", "net", "total")]
