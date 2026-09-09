"""Feature construction: ``lag``, ``one_hot_encode`` and ``roll``.

Pure per-row transforms. Every within-entity operation goes ``.over(entity_col)``
on a time-sorted panel, and ``roll`` shifts its windows so a row never sees its
own (or any later) observation.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from panelary.base.model import ModelState
from panelary.base.transformer import transformer
from panelary.offsets import _strip_freq_alias


@transformer
def lag(lags: list[int], is_sorted: bool = False):
    """Applies lag transformation to a LazyFrame. The time series is assumed to have no null values.

    Parameters
    ----------
    lags : List[int]
        A list of lag values to apply.
    is_sorted: bool
        If already sorted by entity and time columns already, this won't sort again and can save some
        time.
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col = X.columns[0]
        time_col = X.columns[1]
        max_lag = max(lags)
        lagged_series = (
            (
                pl.all()
                .exclude([entity_col, time_col])
                .shift(lag)
                .over(entity_col)
                .name.suffix(f"__lag_{lag}")
            )
            for lag in lags
        )
        # Pre-sorting seems to improve performance by ~20%
        X_new = X if is_sorted else X.sort(by=[entity_col, time_col])

        X_new = X_new.select(
            pl.col(entity_col).set_sorted(),
            pl.col(time_col).set_sorted(),
            *lagged_series,
        ).filter(pl.col(time_col).arg_sort().over(entity_col) >= max_lag)

        artifacts = {"X_new": X_new}
        return artifacts

    return transform


@transformer
def one_hot_encode(drop_first: bool = False):
    """Encode categorical features as a one-hot numeric array.

    Parameters
    ----------
    drop_first : bool
        Drop the first one hot feature.

    Raises
    ------
    ValueError
        if X passed into `transform_new` contains unknown categories.
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        # NOTE: You can't do lazy one hot encoding because
        # polars needs to know the unique values in the selected columns
        cat_cols = X.select(pl.col(pl.Categorical)).columns
        X_new = X.collect().to_dummies(
            columns=cat_cols, drop_first=drop_first, separator="__"
        )
        artifacts = {
            "X_new": X_new,
            "dummy_cols": X_new.columns,
        }
        return artifacts

    def invert(state: ModelState, X: pl.LazyFrame) -> pl.LazyFrame:
        return NotImplemented

    def transform_new(state: ModelState, X: pl.LazyFrame) -> pl.LazyFrame:
        cat_cols = X.select(pl.col(pl.Categorical)).columns
        dummy_cols = state.artifacts["dummy_cols"]
        X_new = X.collect().to_dummies(columns=cat_cols, separator="__")
        if len(set(dummy_cols) & set(X_new.columns)) < len(dummy_cols):
            raise ValueError(
                f"Missing categories: {set(dummy_cols) & set(X_new.columns)}"
            )
        return X_new

    return transform, invert, transform_new


@transformer
def roll(
    window_sizes: list[int],
    stats: list[Literal["mean", "min", "max", "mlm", "sum", "std", "cv"]],
    freq: str,
    fill_strategy: str | None = None,
):
    """
    Performs rolling window calculations on specified columns of a DataFrame.

    Parameters
    ----------
    window_sizes : List[int]
        A list of integers representing the window sizes for the rolling calculations.
    stats : List[Literal["mean", "min", "max", "mlm", "sum", "std", "cv"]]
        A list of statistical measures to calculate for each rolling window.\n
        Supported values are:\n
        - 'mean' for mean
        - 'min' for minimum
        - 'max' for maximum
        - 'mlm' for maximum minus minimum
        - 'sum' for sum
        - 'std' for standard deviation
        - 'cv' for coefficient of variation
    freq : str
        Offset alias supported by Polars.
    fill_strategy : Optional[str]
        Strategy to fill nulls by. Nulls are not filled if None.
        Supported strategies include: ["backward", "forward", "mean", "zero"].
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col = X.columns[:2]
        offset_n, offset_alias = _strip_freq_alias(freq)
        values = pl.all().exclude([entity_col, time_col])
        stat_exprs = {
            "mean": lambda w: values.mean().name.suffix(f"__rolling_mean_{w}"),
            "min": lambda w: values.min().name.suffix(f"__rolling_min_{w}"),
            "max": lambda w: values.max().name.suffix(f"__rolling_max_{w}"),
            "mlm": lambda w: (values.max() - values.min()).name.suffix(
                f"__rolling_mlm_{w}"
            ),
            "sum": lambda w: values.sum().name.suffix(f"__rolling_sum_{w}"),
            "std": lambda w: values.std().name.suffix(f"__rolling_std_{w}"),
            "cv": lambda w: (values.std() / values.mean()).name.suffix(
                f"__rolling_cv_{w}"
            ),
        }
        # Degrees of freedom
        X_all = [
            (
                X.sort([entity_col, time_col])
                .group_by_dynamic(
                    index_column=time_col,
                    by=entity_col,
                    offset=f"{w}{offset_alias}",
                    every=f"1{offset_alias}",
                    period=f"{offset_n * w}{offset_alias}",
                    start_by="datapoint",
                )
                .agg([stat_exprs[stat](w) for stat in stats])
                # NOTE: Must lag by 1 to avoid data leakage.
                # But given the current configuration, shift by w is what we want to do
                .select([entity_col, time_col, values.shift(w).over(entity_col)])
            )
            for w in window_sizes
        ]
        # Join all window lazy operations
        X_rolling = X_all[0]
        for X_window in X_all[1:]:
            X_rolling = X_rolling.join(X_window, on=[entity_col, time_col], how="inner")
        # Defensive join to match original X index
        # X_new = X.join(X_rolling, on=[entity_col, time_col], how="left").select(
        #     X_rolling.columns
        # )
        if fill_strategy:
            X_rolling = X_rolling.fill_null(strategy=fill_strategy)
        artifacts = {"X_new": X_rolling}
        return artifacts

    return transform
