from __future__ import annotations

import polars as pl

from panelary.preprocessing import lag


def _join_X_y(y: pl.LazyFrame, X: pl.LazyFrame) -> pl.LazyFrame:
    on = set(y.columns[:2]) & set(X.columns[:2])
    if len(on) < 1:
        raise ValueError(
            "`X` must have at least one leading column identical to `y`'s"
            f" leading columns ({y.columns[0]}, {y.columns[1]})."
        )
    X_y = y.join(X, on=on, how="inner")
    return X_y


def make_reduction(
    lags: int, y: pl.LazyFrame, X: pl.LazyFrame | None = None
) -> pl.DataFrame:
    idx_cols = y.columns[:2]

    # Force LazyFrame
    y = y.lazy()
    if X is not None:
        X = X.lazy()

    # Generate lags
    y_lag = y.pipe(lag(lags=list(range(1, lags + 1))))
    if isinstance(y_lag, dict):
        y_lag = y_lag["X"]

    # Ensure columns are aligned before join
    y_lag = y_lag.select(
        [*idx_cols, *[col for col in y_lag.columns if col not in idx_cols]]
    )

    # Join lagged y with original y
    X_y = y_lag.join(y, on=idx_cols, how="inner")

    # Add exogenous X if available
    if X is not None:
        try:
            X_y = _join_X_y(X_y, X)
        except Exception as e:
            raise ValueError(f"Error joining X and y: {e}") from e

    # NOTE: Do not use streaming due to PyO3 backend instability
    try:
        return X_y.collect(streaming=False)
    except Exception as e:
        raise RuntimeError(f"Polars collect failed: {e}") from e


def make_direct_reduction(
    lags: int, max_horizons: int, y: pl.LazyFrame, X: pl.LazyFrame | None = None
) -> pl.DataFrame:
    idx_cols = y.columns[:2]
    # Defensive lazy
    y = y.lazy()
    X = X.lazy() if X is not None else X
    # Get lags
    y_lag = y.pipe(lag(lags=list(range(1, lags + max_horizons + 1))))
    X_y = y_lag.join(y, on=idx_cols, how="inner").select(
        [*y.columns, *y_lag.columns[2:]]
    )
    # Drop nulls in lagged columns
    if X is not None:
        X_y = _join_X_y(X_y, X)
    # NOTE: Cannot use streaming...
    # Raises error: pyo3_runtime.PanicException: internal error: entered unreachable code
    X_y_final = X_y.collect()
    return X_y_final


def make_y_lag(X_y: pl.DataFrame, target_col: str, lags: int):
    # NOTE: We should probably do a defensive sort on time before concat_list
    entity_col, time_col = X_y.columns[:2]
    y_lag = (
        X_y.lazy()
        .select([entity_col, time_col, pl.col(rf"^{target_col}__lag_(\d+)$")])
        .group_by(entity_col)
        .agg(pl.all().tail(lags))
        .collect(streaming=True)
        .lazy()
    )
    return y_lag
