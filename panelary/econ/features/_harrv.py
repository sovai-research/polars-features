"""Realized volatility, bipower variation, jumps and Corsi's HAR-RV terms.

The whole block is trailing-window arithmetic on a returns column, so every
feature at row ``t`` is a function of rows ``<= t`` within the same entity:

* :func:`realized_measures` -- realized variance ``RV``, bipower variation
  ``BV`` (jump-robust), the jump component ``J = max(RV - BV, 0)`` and the
  relative jump ``J / RV``.
* :func:`har_terms` -- Corsi's (2009) daily / weekly / monthly heterogeneous
  volatility cascade, i.e. trailing means of ``RV`` over 1, 5 and 22 periods.
* :func:`har_features` -- both of the above as panel columns in one call.
* :class:`HARModel` -- one OLS per entity of ``RV_{t+h}`` on the HAR terms,
  fitted on the **training rows only**; ``transform`` emits the frozen-coefficient
  forecast as a feature.
* :func:`daily_realized_measures` -- aggregate intraday returns into per-day
  ``RV`` / ``BV`` / ``J``. Aggregation is *within* a day, so it is contemporaneous
  rather than forward looking.

References
----------
Andersen & Bollerslev (1998); Barndorff-Nielsen & Shephard (2004) for bipower
variation; Corsi (2009), "A Simple Approximate Long-Memory Model of Realized
Volatility", *Journal of Financial Econometrics*.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer
from panelary.econ.features._common import (
    entity_arrays,
    ols,
    sorted_panel,
)

__all__ = [
    "realized_variance",
    "bipower_variation",
    "jump_component",
    "realized_measures",
    "har_terms",
    "har_features",
    "daily_realized_measures",
    "HARModel",
]

#: ``pi / 2``, the scaling constant that makes bipower variation a consistent
#: estimator of integrated variance under a continuous price path.
_MU1_SQ_INV = np.pi / 2.0

#: Corsi's canonical daily / weekly / monthly horizons (trading days).
DEFAULT_HAR_LAGS: tuple[int, int, int] = (1, 5, 22)


def realized_variance(returns: np.ndarray, window: int) -> np.ndarray:
    """Trailing realized variance: the rolling sum of squared returns.

    ``RV_t = sum_{i=0}^{window-1} r_{t-i}**2``. Emits ``nan`` until the window is
    full.
    """
    r = np.asarray(returns, dtype=float).ravel()
    return _rolling_sum(np.where(np.isfinite(r), r, np.nan) ** 2, window)


def bipower_variation(returns: np.ndarray, window: int) -> np.ndarray:
    """Trailing bipower variation, the jump-robust volatility estimator.

    ``BV_t = (pi / 2) * (n / (n - 1)) * sum |r_{t-i}| * |r_{t-i-1}|`` over the
    trailing window. Because consecutive absolute returns are multiplied, a
    single large jump contributes only linearly, whereas ``RV`` picks it up
    quadratically -- their difference isolates the jump component.
    """
    r = np.asarray(returns, dtype=float).ravel()
    absr = np.abs(np.where(np.isfinite(r), r, np.nan))
    prod = np.full_like(absr, np.nan)
    prod[1:] = absr[1:] * absr[:-1]
    n = int(window)
    scale = _MU1_SQ_INV * (n / (n - 1.0)) if n > 1 else _MU1_SQ_INV
    return scale * _rolling_sum(prod, window)


def jump_component(rv: np.ndarray, bv: np.ndarray) -> np.ndarray:
    """Non-negative jump component ``max(RV - BV, 0)``."""
    return np.maximum(np.asarray(rv, dtype=float) - np.asarray(bv, dtype=float), 0.0)


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling sum with a full-window requirement (``nan`` warm-up)."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    w = int(window)
    if w < 1:
        raise ValueError(f"`window` must be >= 1, got {window!r}.")
    out = np.full(n, np.nan, dtype=float)
    if n < w:
        return out
    filled = np.where(np.isfinite(x), x, 0.0)
    valid = np.isfinite(x).astype(float)
    csum = np.concatenate([[0.0], np.cumsum(filled)])
    cval = np.concatenate([[0.0], np.cumsum(valid)])
    idx = np.arange(w - 1, n)
    sums = csum[idx + 1] - csum[idx + 1 - w]
    counts = cval[idx + 1] - cval[idx + 1 - w]
    out[idx] = np.where(counts > 0, sums, np.nan)
    return out


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling mean over finite values (``nan`` until the window fills)."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    w = int(window)
    out = np.full(n, np.nan, dtype=float)
    if n < w:
        return out
    filled = np.where(np.isfinite(x), x, 0.0)
    valid = np.isfinite(x).astype(float)
    csum = np.concatenate([[0.0], np.cumsum(filled)])
    cval = np.concatenate([[0.0], np.cumsum(valid)])
    idx = np.arange(w - 1, n)
    counts = cval[idx + 1] - cval[idx + 1 - w]
    sums = csum[idx + 1] - csum[idx + 1 - w]
    out[idx] = np.where(counts == w, sums / w, np.nan)
    return out


def realized_measures(returns: np.ndarray, window: int = 22) -> dict[str, np.ndarray]:
    """``RV`` / ``BV`` / ``J`` / relative-jump arrays for one series."""
    rv = realized_variance(returns, window)
    bv = bipower_variation(returns, window)
    jump = jump_component(rv, bv)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(rv > 0, jump / rv, np.nan)
    return {"rv": rv, "bv": bv, "jump": jump, "rel_jump": rel}


def har_terms(
    rv: np.ndarray, lags: Sequence[int] = DEFAULT_HAR_LAGS
) -> dict[str, np.ndarray]:
    """Corsi's heterogeneous cascade: trailing means of ``RV`` over each horizon.

    Returns ``{"har_d": ..., "har_w": ..., "har_m": ...}`` for the canonical
    three horizons (any number of horizons is accepted; extra ones are named
    ``har_l{lag}``). All means are trailing and include the current row, so
    ``har_d`` is just ``RV_t``.
    """
    names = ["har_d", "har_w", "har_m"]
    out: dict[str, np.ndarray] = {}
    for i, lag in enumerate(lags):
        name = names[i] if i < len(names) else f"har_l{lag}"
        out[name] = _rolling_mean(rv, int(lag))
    return out


def har_features(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    returns: str | None = None,
    rv: str | None = None,
    window: int = 22,
    lags: Sequence[int] = DEFAULT_HAR_LAGS,
    prefix: str = "",
) -> pl.DataFrame:
    """Realized-measure and HAR columns for a long panel.

    Supply either ``returns`` (from which ``RV``/``BV``/``J`` are built on a
    trailing ``window``) or an already-computed ``rv`` column (e.g. from
    :func:`daily_realized_measures`), in which case only the HAR cascade terms
    are added.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel.
    entity, time : str
        Panel keys.
    returns : str, optional
        Returns column.
    rv : str, optional
        Pre-computed realized-variance column.
    window : int, default=22
        Trailing window for the realized measures when ``returns`` is given.
    lags : sequence of int, default=(1, 5, 22)
        HAR horizons.
    prefix : str, default=""
        Prefix for the emitted column names.

    Returns
    -------
    polars.DataFrame
        ``df`` sorted by ``(entity, time)`` with the new columns appended.
    """
    if (returns is None) == (rv is None):
        raise ValueError("pass exactly one of `returns` or `rv`.")
    source = returns if returns is not None else rv
    frame = sorted_panel(df, entity, time)
    if source not in frame.columns:
        raise ValueError(
            f"column {source!r} not found in frame; available: {frame.columns}."
        )
    n = frame.height
    buffers: dict[str, np.ndarray] = {}

    def _store(name: str, idx: np.ndarray, values: np.ndarray) -> None:
        key = f"{prefix}{name}"
        if key not in buffers:
            buffers[key] = np.full(n, np.nan, dtype=float)
        buffers[key][idx] = values

    for _key, idx, data in entity_arrays(frame, entity, [source]):
        series = data[source]
        if returns is not None:
            measures = realized_measures(series, window)
            for name, values in measures.items():
                _store(name, idx, values)
            rv_series = measures["rv"]
        else:
            rv_series = series
        for name, values in har_terms(rv_series, lags).items():
            _store(name, idx, values)
    # NaN is the in-kernel "not available" marker; emit real Polars nulls so
    # `is_null()` and `drop_nulls()` see the warm-up rows.
    return frame.with_columns(
        [pl.Series(name, values).fill_nan(None) for name, values in buffers.items()]
    )


def daily_realized_measures(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    date: str,
    returns: str,
) -> pl.DataFrame:
    """Aggregate intraday returns into per-``(entity, date)`` ``RV`` / ``BV`` / ``J``.

    ``RV`` is the within-day sum of squared returns and ``BV`` the within-day
    bipower variation; both use only observations from that same day, so the
    result is contemporaneous (never forward looking). Rows must already be in
    intraday time order within each ``(entity, date)`` group.

    Returns
    -------
    polars.DataFrame
        One row per ``(entity, date)`` with ``rv``, ``bv``, ``jump``,
        ``rel_jump`` and ``n_obs``.
    """
    frame = df.collect() if isinstance(df, pl.LazyFrame) else df
    for col in (entity, date, returns):
        if col not in frame.columns:
            raise ValueError(
                f"column {col!r} not found in frame; available: {frame.columns}."
            )
    absr = pl.col(returns).abs()
    grouped = (
        frame.group_by([entity, date], maintain_order=True)
        .agg(
            (pl.col(returns) ** 2).sum().alias("rv"),
            (absr * absr.shift(1)).sum().alias("_bp"),
            pl.len().alias("n_obs"),
        )
        .with_columns(
            pl.when(pl.col("n_obs") > 1)
            .then(
                _MU1_SQ_INV * (pl.col("n_obs") / (pl.col("n_obs") - 1)) * pl.col("_bp")
            )
            .otherwise(None)
            .alias("bv")
        )
        .drop("_bp")
    )
    return (
        grouped.with_columns(
            pl.max_horizontal(pl.col("rv") - pl.col("bv"), pl.lit(0.0)).alias("jump")
        )
        .with_columns(
            pl.when(pl.col("rv") > 0)
            .then(pl.col("jump") / pl.col("rv"))
            .otherwise(None)
            .alias("rel_jump")
        )
        .sort([entity, date])
    )


class HARModel(PanelTransformer):
    """Corsi HAR-RV regression: one OLS per entity, fitted on training rows only.

    ``fit`` regresses ``RV_{t+horizon}`` on ``[1, RV_d, RV_w, RV_m]`` within each
    entity of the training panel and stores the coefficients. ``transform``
    applies those frozen coefficients to each row's own trailing HAR terms and
    emits the forecast as a feature column -- so a test-fold row's feature uses
    only its own past and parameters learned on train.

    Entities with too few training rows (or unseen at transform time) fall back
    to the pooled coefficients estimated across all training entities.

    Parameters
    ----------
    returns : str, optional
        Returns column; realized measures are built from it on a trailing
        ``window``. Mutually exclusive with ``rv``.
    rv : str, optional
        Pre-computed realized-variance column.
    horizon : int, default=1
        Forecast horizon in rows.
    window : int, default=22
        Trailing window for the realized measures when ``returns`` is given.
    lags : sequence of int, default=(1, 5, 22)
        HAR horizons.
    log_target : bool, default=False
        Fit the regression in logs (``log RV``), the common specification for
        strictly positive variance; the emitted forecast is exponentiated back.
    min_train_rows : int, default=60
        Minimum usable rows before an entity gets its own coefficients.
    output : str, default="har_forecast"
        Name of the emitted forecast column.
    entity, time : str, optional
        Default panel keys for bare polars frames.

    Attributes
    ----------
    coef_ : dict
        ``{entity: numpy.ndarray}`` of fitted coefficients.
    pooled_coef_ : numpy.ndarray
        Fallback coefficients estimated on all training entities.
    feature_names_ : list of str
        The HAR term names used as regressors.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        *,
        returns: str | None = None,
        rv: str | None = None,
        horizon: int = 1,
        window: int = 22,
        lags: Sequence[int] = DEFAULT_HAR_LAGS,
        log_target: bool = False,
        min_train_rows: int = 60,
        output: str = "har_forecast",
        keep_terms: bool = True,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if (returns is None) == (rv is None):
            raise ValueError("pass exactly one of `returns` or `rv`.")
        if horizon < 1:
            raise ValueError(f"`horizon` must be >= 1, got {horizon!r}.")
        self.returns = returns
        self.rv = rv
        self.horizon = int(horizon)
        self.window = int(window)
        self.lags = tuple(int(x) for x in lags)
        self.log_target = bool(log_target)
        self.min_train_rows = int(min_train_rows)
        self.output = output
        self.keep_terms = bool(keep_terms)
        self.coef_: dict[object, np.ndarray] = {}
        self.pooled_coef_: np.ndarray | None = None
        self.feature_names_: list[str] = []

    # -- internals ------------------------------------------------------- #
    def _terms_frame(self, panel: PanelFrame) -> pl.DataFrame:
        return har_features(
            panel.lazy(),
            entity=panel.entity_col,
            time=panel.time_col,
            returns=self.returns,
            rv=self.rv,
            window=self.window,
            lags=self.lags,
        )

    def _term_names(self) -> list[str]:
        names = ["har_d", "har_w", "har_m"]
        return [
            names[i] if i < len(names) else f"har_l{lag}"
            for i, lag in enumerate(self.lags)
        ]

    def _design(self, data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        terms = self._term_names()
        rv_col = "rv" if self.returns is not None else self.rv
        target = np.asarray(data[rv_col], dtype=float)
        X = np.column_stack([np.ones(target.shape[0])] + [data[name] for name in terms])
        y = np.full(target.shape[0], np.nan)
        if self.horizon < target.shape[0]:
            y[: -self.horizon] = target[self.horizon :]
        if self.log_target:
            with np.errstate(invalid="ignore", divide="ignore"):
                y = np.where(y > 0, np.log(y), np.nan)
                X[:, 1:] = np.where(X[:, 1:] > 0, np.log(X[:, 1:]), np.nan)
        return X, y

    def _fit(self, panel: PanelFrame) -> None:
        frame = self._terms_frame(panel)
        rv_col = "rv" if self.returns is not None else self.rv
        cols = [rv_col, *self._term_names()]
        self.feature_names_ = self._term_names()
        self.coef_ = {}
        pooled_X: list[np.ndarray] = []
        pooled_y: list[np.ndarray] = []
        for key, _idx, data in entity_arrays(frame, panel.entity_col, cols):
            X, y = self._design(data)
            good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            if good.sum() >= max(self.min_train_rows, X.shape[1] + 2):
                self.coef_[key] = ols(X[good], y[good]).beta
            if good.any():
                pooled_X.append(X[good])
                pooled_y.append(y[good])
        if pooled_X:
            Xp = np.vstack(pooled_X)
            yp = np.concatenate(pooled_y)
            if yp.shape[0] > Xp.shape[1]:
                self.pooled_coef_ = ols(Xp, yp).beta
        if self.pooled_coef_ is None and not self.coef_:
            raise ValueError(
                "HARModel.fit: no entity had enough finite HAR rows to estimate "
                "coefficients. Lower `min_train_rows`/`window`, or supply longer "
                "series."
            )

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        frame = self._terms_frame(panel)
        rv_col = "rv" if self.returns is not None else self.rv
        cols = [rv_col, *self._term_names()]
        preds = np.full(frame.height, np.nan)
        fallback = self.pooled_coef_
        for key, idx, data in entity_arrays(frame, panel.entity_col, cols):
            X, _ = self._design(data)
            beta = self.coef_.get(key, fallback)
            if beta is None:
                continue
            yhat = X @ beta
            if self.log_target:
                yhat = np.exp(yhat)
            preds[idx] = yhat
        out = frame.with_columns(pl.Series(self.output, preds))
        if not self.keep_terms:
            drop = [c for c in cols if c not in panel.columns]
            out = out.drop([c for c in drop if c in out.columns])
        return PanelFrame(out.lazy(), entity=panel.entity_col, time=panel.time_col)
