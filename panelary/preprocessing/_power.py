"""Power transforms: ``boxcox`` and ``yeojohnson``.

Both estimate a per-entity lambda, so both are **fitted** transforms and must be
fit per fold. SciPy is an optional dependency: it is imported lazily inside the
transform, through :func:`panelary._internal._deps.require`, so importing this
module stays free.
"""

from __future__ import annotations

import polars as pl

from panelary._internal._deps import require
from panelary.base.model import ModelState
from panelary.base.transformer import transformer
from panelary.preprocessing._base import PL_NUMERIC_COLS


@transformer
def boxcox(method: str = "mle"):
    """Applies the Box-Cox transformation to numeric columns in a panel DataFrame.

    Parameters
    ----------
    method : str
        The method used to determine the lambda parameter of the Box-Cox transformation.

        Supported methods:\n
        - `mle`: maximum likelihood estimation
        - `pearsonr`: Pearson correlation coefficient
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        optimize = require("scipy.optimize", feature="boxcox")
        boxcox_normmax = require("scipy.stats", feature="boxcox").boxcox_normmax

        def optimizer(fun):
            return optimize.minimize_scalar(
                fun,
                bounds=(-2.0, 2.0),
                method="bounded",
                options={"maxiter": 200, "xatol": 1e-12},
            )

        idx_cols = X.columns[:2]
        entity_col, time_col = idx_cols
        gb = X.group_by(X.columns[0])
        # Step 1. Compute optimal lambdas
        lmbds = gb.agg(
            PL_NUMERIC_COLS(entity_col, time_col)
            .map_batches(
                lambda x: pl.Series(
                    [boxcox_normmax(x.to_numpy(), method=method, optimizer=optimizer)]
                ),
                returns_scalar=True,
                return_dtype=pl.Float64,
            )
            .name.suffix("__lmbd")
        )
        # Step 2. Transform
        cols = X.select(PL_NUMERIC_COLS(entity_col, time_col)).columns
        X_new = X.join(lmbds, on=entity_col, how="left").select(
            idx_cols
            + [
                pl.when(pl.col(f"{col}__lmbd") == 0)
                .then(pl.col(col).log())
                .otherwise(
                    (pl.col(col) ** pl.col(f"{col}__lmbd") - 1) / pl.col(f"{col}__lmbd")
                )
                for col in cols
            ]
        )
        artifacts = {"X_new": X_new, "lmbds": lmbds}
        return artifacts

    def invert(state: ModelState, X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col = X.columns[:2]
        lmbds = state.artifacts["lmbds"]
        cols = X.select(PL_NUMERIC_COLS(entity_col, time_col)).columns
        X_new = (
            X.join(lmbds, on=entity_col, how="left", suffix="__lmbd")
            .with_columns(
                [
                    pl.when(pl.col(f"{col}__lmbd") == 0)
                    .then(pl.col(col).exp())
                    .otherwise(
                        (pl.col(col) * pl.col(f"{col}__lmbd") + 1)
                        ** (1 / pl.col(f"{col}__lmbd"))
                    )
                    for col in cols
                ]
            )
            .select(X.columns)
        )
        return X_new

    return transform, invert


@transformer
def yeojohnson(brack: tuple = (-2, 2)):
    def transform(X: pl.LazyFrame) -> dict:
        yeojohnson_normmax = require(
            "scipy.stats", feature="yeojohnson"
        ).yeojohnson_normmax
        idx_cols = X.columns[:2]
        entity_col, time_col = idx_cols
        cols = X.select(PL_NUMERIC_COLS(entity_col, time_col)).columns

        df = X.collect()
        out = []
        for group, subdf in df.group_by(entity_col, maintain_order=True):
            row = {entity_col: group[0]}
            for col in cols:
                x = subdf[col].to_numpy()
                lmbd = yeojohnson_normmax(x, brack)
                row[f"{col}__lmbd"] = lmbd
            out.append(row)
        lmbds = pl.DataFrame(out)

        X_new = X.join(lmbds.lazy(), on=entity_col, how="left").select(
            idx_cols
            + [
                pl.when((pl.col(col) >= 0) & (pl.col(f"{col}__lmbd") == 0))
                .then(pl.col(col).log1p())
                .when(pl.col(col) >= 0)
                .then(
                    ((pl.col(col) + 1) ** pl.col(f"{col}__lmbd") - 1)
                    / pl.col(f"{col}__lmbd")
                )
                .when((pl.col(col) < 0) & (pl.col(f"{col}__lmbd") == 2))
                .then(-pl.col(col).log1p())
                .otherwise(
                    -((-pl.col(col) + 1) ** (2 - pl.col(f"{col}__lmbd")) - 1)
                    / (2 - pl.col(f"{col}__lmbd"))
                )
                .alias(col)
                for col in cols
            ]
        )

        return {"X_new": X_new, "lmbds": lmbds.lazy()}

    def invert(state: ModelState, X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col = X.columns[:2]
        lmbds = state.artifacts["lmbds"]
        cols = X.select(PL_NUMERIC_COLS(entity_col, time_col)).columns

        return (
            X.join(lmbds, on=entity_col, how="left")
            .with_columns(
                [
                    pl.when((pl.col(col) >= 0) & (pl.col(f"{col}__lmbd") == 0))
                    .then(pl.col(col).exp() - 1)
                    .when(pl.col(col) >= 0)
                    .then(
                        (
                            (pl.col(col) * pl.col(f"{col}__lmbd") + 1)
                            ** (1 / pl.col(f"{col}__lmbd"))
                        )
                        - 1
                    )
                    .when((pl.col(col) < 0) & (pl.col(f"{col}__lmbd") == 2))
                    .then(1 - (-(pl.col(col)).exp()))
                    .otherwise(
                        1
                        - (
                            (-(2 - pl.col(f"{col}__lmbd")) * pl.col(col) + 1)
                            ** (1 / (2 - pl.col(f"{col}__lmbd")))
                        )
                    )
                    .alias(col)
                    for col in cols
                ]
            )
            .select(X.columns)
        )

    return transform, invert
