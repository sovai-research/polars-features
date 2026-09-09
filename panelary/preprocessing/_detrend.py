"""Trend and seasonality removal: ``detrend`` and ``deseasonalize_fourier``.

Both are **fitted** transforms -- ``detrend`` keeps per-entity OLS coefficients,
``deseasonalize_fourier`` keeps a pickled per-entity regressor -- so both must be
fit on training rows only and inverted with the fitted state. scikit-learn is
optional and imported lazily inside the transform.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import polars as pl
import polars.selectors as cs

from panelary._internal._deps import require
from panelary.base.model import ModelState
from panelary.base.transformer import transformer
from panelary.offsets import _strip_freq_alias
from panelary.seasonality import add_fourier_terms


@transformer
def detrend(freq: str, method: Literal["linear", "mean"] = "linear"):
    """Removes mean or linear trend from numeric columns in a panel DataFrame.

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars.
    method : str
        If `mean`, subtracts mean from each time-series.
        If `linear`, subtracts line of best-fit (via OLS) from each time-series.
        Defaults to `linear`.
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col = X.columns[:2]
        if method == "linear":
            cols = X.columns
            X_new_temp = (
                X.with_columns(
                    pl.col(time_col).arg_sort().over(entity_col).alias("__x")
                )
                .with_columns(
                    *[
                        (pl.cov(pl.col(c), pl.col("__x")) / pl.col("__x").var())
                        .over(entity_col)
                        .name.suffix("__beta")
                        for c in X.columns[2:]
                    ],
                    *[
                        pl.col(c).mean().over(entity_col).name.suffix("__mean")
                        for c in X.columns[2:]
                    ],
                )
                .with_columns(
                    (pl.col(c + "__mean") - pl.col(c + "__beta") * (pl.len() - 1) / 2)
                    .over(entity_col)
                    .alias(c + "__alpha")
                    for c in X.columns[2:]
                )
                .with_columns(
                    (
                        pl.col(c)
                        - pl.col(c + "__beta") * pl.col("__x")
                        - pl.col(c + "__alpha")
                    ).alias(c)
                    for c in X.columns[2:]
                )
                .collect()
            )

            X_new = X_new_temp.select(cols)

            artifacts = {
                "_beta": X_new_temp.select(entity_col, cs.ends_with("__beta")).unique(
                    subset=[entity_col]
                ),
                "_alpha": X_new_temp.select(entity_col, cs.ends_with("__alpha")).unique(
                    subset=[entity_col]
                ),
                "X_new": X_new.lazy(),
                "_firsts": X_new_temp.group_by(entity_col).agg(
                    pl.col(time_col).min().alias("first_fitted")
                ),
            }
        elif method == "mean":
            _mean = X.group_by(entity_col).agg(
                pl.col(X.columns[2:]).mean().name.suffix("__mean")
            )
            X_new = X.with_columns(
                pl.col(X.columns[2:]) - pl.col(X.columns[2:]).mean().over(entity_col)
            )
            _mean, X_new = pl.collect_all([_mean, X_new])
            artifacts = {"_mean": _mean, "X_new": X_new.lazy()}
        else:
            raise ValueError(
                f"Method `{method}` not recognized. Expected `linear` or `mean`."
            )
        return artifacts

    def invert(state: ModelState, X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col = X.columns[:2]
        offset_n, offset_alias = _strip_freq_alias(freq)
        if method == "linear":
            _beta = state.artifacts["_beta"]
            _alpha = state.artifacts["_alpha"]
            firsts = state.artifacts["_firsts"]
            if offset_alias == "i":
                offset_expr = (
                    (pl.col("first") - pl.col("first_fitted")) // offset_n
                ).alias("offset")
            elif offset_alias == "d":
                offset_expr = (
                    (pl.col("first") - pl.col("first_fitted")).dt.total_days()
                    // offset_n
                ).alias("offset")
            elif offset_alias == "ms":
                offset_expr = (
                    (pl.col("first") - pl.col("first_fitted")).dt.total_milliseconds()
                    // offset_n
                ).alias("offset")
            elif offset_alias == "us":
                offset_expr = (
                    (pl.col("first") - pl.col("first_fitted")).dt.total_microseconds()
                    // offset_n
                ).alias("offset")
            elif offset_alias == "m":
                offset_expr = (
                    (pl.col("first") - pl.col("first_fitted")).dt.total_minutes()
                    // offset_n
                ).alias("offset")
            elif offset_alias == "s":
                offset_expr = (
                    (pl.col("first") - pl.col("first_fitted")).dt.total_seconds()
                    // offset_n
                ).alias("offset")
            elif offset_alias == "ns":
                offset_expr = (
                    (pl.col("first") - pl.col("first_fitted")).dt.total_nanoseconds()
                    // offset_n
                ).alias("offset")
            else:
                raise ValueError(
                    f"Currently, the offset alias {offset_alias} is not supported in .invert()."
                )

            grouped = (
                X.group_by(entity_col)
                .agg(pl.col(time_col).min().alias("first"))
                .collect()
            )
            offsets = grouped.join(firsts, on=entity_col).with_columns(offset_expr)
            # Note : pl.col(offset) here is generated above, then left joined to X.
            # So there is no need to do over.
            # In the code below, alpha, beta are all constant over entity because
            # of the left join
            x = pl.col(time_col).arg_sort().over(entity_col) + pl.col("offset")
            X_new = (
                X.join(offsets.lazy(), on=entity_col, how="left")
                .join(_beta.lazy(), on=entity_col, how="left")
                .join(_alpha.lazy(), on=entity_col, how="left")
                .with_columns(
                    [
                        (
                            pl.col(col)
                            + pl.col(f"{col}__alpha")
                            + pl.col(f"{col}__beta") * x
                        )
                        for col in X.columns[2:]
                    ]
                )
                .select(X.columns)
            )
        else:
            X_new = (
                X.join(state.artifacts["_mean"].lazy(), on=entity_col, how="left")
                .with_columns(
                    [pl.col(col) + pl.col(f"{col}__mean") for col in X.columns[2:]]
                )
                .select(X.columns)
            )
        return X_new

    return transform, invert


@transformer
def deseasonalize_fourier(sp: int, K: int, robust: bool = False):
    """Removes seasonality via residualized regression with Fourier terms.

    Parameters
    ----------
    sp: int
        Seasonal period.
    K : int
        Maximum order(s) of Fourier terms.
        Must be less than `sp`.

    Notes
    -----
    Part of this transformer uses sklearn under-the-hood: it is not pure Polars and lazy.
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        linear_model = require("sklearn.linear_model", feature="deseasonalize_fourier")
        regressor_cls = (
            linear_model.LinearRegression if robust else linear_model.TheilSenRegressor
        )
        X = X.collect()  # Not lazy
        if X.shape[1] > 3:
            raise ValueError(
                "Got `X` with more than 3 columns."
                " Expected `x` with maximum 3 columns: entity column,"
                " time column, target column."
            )

        def _deseasonalize(inputs: pl.Series):
            # Coerce inputs
            X_y = inputs.struct.unnest()
            y = X_y.get_column(target_col).to_numpy()
            X = X_y.select(pl.all().exclude(target_col)).to_numpy()
            # Fit-predict
            regressor = regressor_cls().fit(y=y, X=X)
            # Subtract prediction from y
            y_pred = regressor.predict(X=X)
            y_new = y - y_pred
            return {
                target_col: y_new.tolist(),
                "seasonal": y_pred.tolist(),
                "regressor": pickle.dumps(regressor),
            }

        entity_col, time_col, target_col = X.columns[:3]
        X_with_features = X.join(
            X.pipe(add_fourier_terms(sp=sp, K=K)).collect(),
            on=[entity_col, time_col],
            how="left",
        )
        fourier_cols = list(set(X_with_features.columns) - set(X.columns))
        return_dtype = pl.Struct(
            [
                pl.Field(name=target_col, dtype=pl.List(pl.Float64)),
                pl.Field(name="seasonal", dtype=pl.List(pl.Float64)),
                pl.Field(name="regressor", dtype=pl.Binary),
            ]
        )
        X_new = (
            X_with_features.group_by(entity_col)
            .agg(
                [
                    time_col,
                    pl.struct([target_col, *fourier_cols])
                    .map_elements(_deseasonalize, return_dtype=return_dtype)
                    .alias("result"),
                    pl.col(X.columns[3:]),
                ]
            )
            .select([time_col, entity_col, pl.col("result")])
            .unnest("result")
        )
        artifacts = {
            "X_new": X_new.select([entity_col, time_col, target_col])
            .explode([time_col, target_col])
            .lazy(),
            "X_seasonal": X_new.select(
                [entity_col, time_col, pl.col("seasonal").alias(target_col)]
            )
            .explode([time_col, target_col])
            .lazy(),
            "regressors": X_new.select([entity_col, "regressor"]),
            "fourier_cols": fourier_cols,
        }
        return artifacts

    def invert(state: ModelState, X: pl.LazyFrame) -> pl.LazyFrame:
        X = X.collect()
        entity_col, time_col, target_col = X.columns[:3]

        if X.shape[1] > 3:
            raise ValueError(
                "Got `X` with more than 3 columns."
                "Expected `x` with maximum 3 columns: entity column, time column, target column."
            )

        regressors = state.artifacts["regressors"]
        fourier_cols = state.artifacts["fourier_cols"]

        def _reseasonalize(inputs: Mapping[str, Any]):
            # Coerce inputs
            regressor = pickle.loads(inputs["regressor"])
            y = inputs[target_col]
            X = np.array(inputs["fourier"]).reshape((len(y), len(fourier_cols)))
            # Predict
            y_pred = regressor.predict(X=X)
            # Add prediction to y
            y_new = np.array(y) + y_pred
            return y_new.tolist()

        X_with_features = X.join(
            X.pipe(add_fourier_terms(sp=sp, K=K)).collect(),
            on=[entity_col, time_col],
            how="left",
        )
        y_new = (
            X_with_features.group_by(entity_col)
            .agg([time_col, target_col, *fourier_cols])
            .join(regressors, on=entity_col, how="left")
            .select(
                [
                    entity_col,
                    time_col,
                    pl.struct(
                        [
                            target_col,
                            pl.concat_list(fourier_cols).alias("fourier"),
                            "regressor",
                        ]
                    ).map_elements(_reseasonalize, return_dtype=pl.List(pl.Float64)),
                ]
            )
            .explode(pl.all().exclude(entity_col))
        )
        return y_new.lazy()

    return transform, invert
