"""Differencing and stationarity: ``diff`` and ``fractional_diff``.

Both are strictly backward-looking within an entity. ``fractional_diff`` routes
through the shared causal kernel in :mod:`panelary._internal._ffd`, so the four
public spellings of fractional differencing cannot drift apart.
"""

from __future__ import annotations

import polars as pl

from panelary.base.model import ModelState
from panelary.base.transformer import transformer
from panelary.preprocessing._base import PL_NUMERIC_COLS


@transformer
def diff(order: int, sp: int = 1, fill_strategy: str | None = None):
    """Difference time-series in panel data given order and seasonal period.

    Parameters
    ----------
    order : int
        The order to difference.
    sp : int
        Seasonal periodicity.
    fill_strategy : Optional[str]
        Strategy to fill nulls by. Nulls are not filled if None.
        Supported strategies include: ["backward", "forward", "mean", "zero"].
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        idx_cols = X.columns[:2]
        entity_col = idx_cols[0]
        time_col = idx_cols[1]

        X_first, X_last = pl.collect_all(
            [
                X.group_by(entity_col).head(1),
                X.group_by(entity_col).tail(1),
            ]
        )
        for _ in range(order):
            X = X.select(
                [
                    entity_col,
                    time_col,
                    PL_NUMERIC_COLS(entity_col, time_col).diff(n=sp).over(entity_col),
                ]
            )

        if fill_strategy:
            X = X.fill_null(strategy=fill_strategy)
        artifacts = {
            "X_new": X,
            "X_first": X_first.lazy(),
            "X_last": X_last.lazy(),
        }
        return artifacts

    def invert(
        state: ModelState, X: pl.LazyFrame, from_last: bool = False
    ) -> pl.LazyFrame:
        artifacts = state.artifacts
        entity_col = X.columns[0]
        time_col = X.columns[1]
        idx_cols = entity_col, time_col

        X_cutoff = artifacts["X_last"] if from_last else artifacts["X_first"]
        X_new = pl.concat(
            [
                X,
                X_cutoff.select(
                    pl.col(col).cast(dtype) for col, dtype in X.schema.items()
                ),
            ],
            how="diagonal",
        ).sort(idx_cols)
        for _ in range(order):
            X_new = X_new.select(
                [
                    entity_col,
                    time_col,
                    PL_NUMERIC_COLS(entity_col, time_col).cum_sum().over(entity_col),
                ]
            )
        X_new = (
            X.select(idx_cols)
            # Must drop duplicates to deal with case where
            # X to be inverted starts with timestamp == cutoff
            .join(
                X_new.unique(subset=[entity_col, time_col], keep="last"),
                on=idx_cols,
                how="left",
            )
        )
        return X_new

    return transform, invert


@transformer
def fractional_diff(
    d: float, min_weight: float | None = None, window_size: int | None = None
):
    """Compute the fractional differential of a time series.

    This particular functionality is referenced in Advances in Financial Machine
    Learning by Marcos Lopez de Prado (2018).

    For feature creation purposes, it is suggested that the minimum value of d
    is used that removes stationarity from the time series. This can be achieved
    by running the augmented dickey-fuller test on the time series for different
    values of d and selecting the minimum value that makes the time series
    stationary.

    Parameters
    ----------
    d : float
        The fractional order of the differencing operator.
    min_weight : float, optional
        The minimum weight to use for calculations (the weight-magnitude
        ``threshold``). If specified, the window size is computed from this value
        and ``window_size`` is not needed.
    window_size : int, optional
        The window size of the fractional differencing operator (a hard
        ``max_width`` cap on the kernel). If specified, ``min_weight`` is not
        needed and the canonical default threshold governs early truncation.

    Notes
    -----
    Routes through the shared, causal :func:`panelary._internal._ffd.frac_diff_expr`
    builder -- the single source of truth shared with ``.panel.frac_diff``,
    ``.ts.frac_diff`` and :class:`~panelary.transform.frac_diff.FracDiff`
    -- applied per entity via ``.over(entity)``. The incomplete leading window is
    emitted as ``null`` (never zero-filled).
    """
    from panelary._internal._ffd import frac_diff_expr

    if min_weight is None and window_size is None:
        raise ValueError("Either `min_weight` or `window_size` must be specified.")

    if min_weight is not None and window_size is not None:
        raise ValueError("Only one of `min_weight` or `window_size` must be specified.")

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        schema = X.collect_schema()
        names = schema.names()
        entity_col, time_col = names[:2]
        numeric = [
            c
            for c in names
            if c not in (entity_col, time_col) and schema[c].is_numeric()
        ]

        def _build(col: str) -> pl.Expr:
            # ``frac_diff_expr`` uses ``pl.sum_horizontal`` internally, so it must
            # be built per single column (a multi-column selector would be summed
            # across columns). Apply per numeric column, grouped ``.over(entity)``,
            # replacing the column in place -- matching the historical behaviour.
            if min_weight is not None:
                expr = frac_diff_expr(pl.col(col), d=d, threshold=min_weight)
            else:
                expr = frac_diff_expr(pl.col(col), d=d, max_width=window_size)
            return expr.over(entity_col).alias(col)

        X_new = X.with_columns([_build(c) for c in numeric])
        return {"X_new": X_new}

    return transform
