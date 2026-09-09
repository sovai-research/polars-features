"""Per-entity standardisation (``scale``).

A **fitted** transform: the means and standard deviations are artifacts of the
fit, so it must be fit on training rows only and applied frozen to the test
fold via ``transform_new``. Fitting it on a whole panel is a look-ahead.
"""

from __future__ import annotations

import polars as pl

from panelary.base.model import ModelState
from panelary.base.transformer import transformer
from panelary.preprocessing._base import PL_NUMERIC_COLS


@transformer
def scale(use_mean: bool = True, use_std: bool = True, rescale_bool: bool = False):
    """
    Performs scaling and rescaling operations on the numeric columns of a DataFrame.

    Parameters
    ----------
    use_mean : bool
        Whether to subtract the mean from the numeric columns. Defaults to True.
    use_std : bool
        Whether to divide the numeric columns by the standard deviation. Defaults to True.
    rescale_bool : bool
        Whether to rescale boolean columns to the range [-1, 1]. Defaults to False.
    """

    if not (use_mean or use_std):
        raise ValueError("At least one of `use_mean` or `use_std` must be set to True")

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        idx_cols = X.columns[:2]
        entity_col, time_col = idx_cols
        numeric_cols = X.select(PL_NUMERIC_COLS(entity_col, time_col)).columns
        boolean_cols = None
        _mean = None
        _std = None
        if use_mean:
            _mean = X.group_by(entity_col).agg(
                PL_NUMERIC_COLS(entity_col, time_col).mean().name.suffix("_mean")
            )
            X = X.join(_mean, on=entity_col).select(
                idx_cols + [pl.col(col) - pl.col(f"{col}_mean") for col in numeric_cols]
            )
        if use_std:
            _std = X.group_by(entity_col).agg(
                PL_NUMERIC_COLS(entity_col, time_col).std().name.suffix("_std")
            )
            X = X.join(_std, on=entity_col).select(
                idx_cols + [pl.col(col) / pl.col(f"{col}_std") for col in numeric_cols]
            )
        if rescale_bool:
            boolean_cols = X.select(pl.col(pl.Boolean)).columns
            X = X.with_columns(pl.col(pl.Boolean).cast(pl.Int8) * 2 - 1)
        artifacts = {
            "X_new": X,
            "numeric_cols": numeric_cols,
            "boolean_cols": boolean_cols,
            "_mean": _mean,
            "_std": _std,
        }
        return artifacts

    def invert(state: ModelState, X: pl.LazyFrame) -> pl.LazyFrame:
        idx_cols = X.columns[:2]
        entity_col = idx_cols[0]
        artifacts = state.artifacts
        numeric_cols = artifacts["numeric_cols"]
        if use_std:
            _std = artifacts["_std"]
            X = X.join(_std, on=entity_col, how="left").select(
                idx_cols + [pl.col(col) * pl.col(f"{col}_std") for col in numeric_cols]
            )
        if use_mean:
            _mean = artifacts["_mean"]
            X = X.join(_mean, on=entity_col, how="left").select(
                idx_cols + [pl.col(col) + pl.col(f"{col}_mean") for col in numeric_cols]
            )
        if rescale_bool:
            X = X.with_columns(pl.col(artifacts["boolean_cols"]).cast(pl.Int8))
        return X

    def transform_new(state: ModelState, X: pl.LazyFrame) -> pl.LazyFrame:
        artifacts = state.artifacts
        idx_cols = X.columns[:2]
        numeric_cols = state.artifacts["numeric_cols"]
        _mean = artifacts["_mean"]
        _std = artifacts["_std"]
        if use_mean:
            X = X.join(_mean, on=idx_cols, how="left").select(
                idx_cols + [pl.col(col) - pl.col(f"{col}_mean") for col in numeric_cols]
            )
        if use_std:
            X = X.join(_std, on=idx_cols, how="left").select(
                idx_cols + [pl.col(col) / pl.col(f"{col}_std") for col in numeric_cols]
            )
        if rescale_bool:
            X = X.with_columns(pl.col(pl.Boolean).cast(pl.Int8) * 2 - 1)
        return X

    return transform, invert, transform_new
