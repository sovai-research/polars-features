from __future__ import annotations

import pickle
import warnings
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import polars as pl
import polars.selectors as cs

from panelary._deps import require
from panelary.base import transformer
from panelary.base.model import ModelState
from panelary.offsets import _strip_freq_alias
from panelary.seasonality import add_fourier_terms

# from panelary._panelary_rust import frac_diff


def PL_NUMERIC_COLS(*exclude):
    return cs.numeric() - cs.by_name(exclude)


class LeakageWarning(UserWarning):
    """Warning that an operation reads the future (look-ahead / target leakage).

    Emitted by leak-prone code paths that are easy to use unsafely inside a
    backtest or cross-validation split -- for example :func:`impute` with the
    ``"bfill"`` or ``"interpolate"`` methods, which fill a missing value using
    later (future) observations. Such fills pass silently through CV yet inflate
    out-of-sample performance. Prefer a strictly point-in-time, leak-safe imputer
    (e.g. ``impute("cafe")`` / :func:`cafe_impute`, or forward-fill) unless you
    have deliberately opted in via ``allow_leaky=True``.
    """


#: Imputation methods that read the future and therefore leak inside a CV split.
_LEAKY_IMPUTE_METHODS: frozenset[str] = frozenset({"bfill", "interpolate"})


@transformer
def reindex(drop_duplicates: bool = False):
    """Reindexes the entity and time columns to have every possible combination of (entity, time).

    Parameters
    ---------
    drop_duplicates : bool
        Defaults to False. If True, duplicates are dropped before reindexing.
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col = X.columns[:2]
        if drop_duplicates:
            entities = X.select(pl.col(entity_col).unique())
            timestamps = X.select(pl.col(time_col).unique())
        else:
            entities = X.select(entity_col)
            timestamps = X.select(time_col)
        idx = entities.join(timestamps, how="cross")
        X_new = idx.join(X, how="left", on=[entity_col, time_col])
        artifacts = {"X_new": X_new}
        return artifacts

    return transform


@transformer
def coerce_dtypes(schema: Mapping[str, pl.DataType]):
    """Coerces the column datatypes of a DataFrame using the provided schema.

    Parameters
    ----------
    schema : Mapping[str, pl.DataType]
        A dictionary-like object mapping column names to the desired data types.
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        X_new = X.with_columns(
            [pl.col(col).cast(dtype) for col, dtype in schema.items()]
        )
        artifacts = {"X_new": X_new}
        return artifacts

    return transform


@transformer
def time_to_arange(eager: bool = False):
    """Coerces time column into arange per entity.

    Assumes even-spaced time-series and homogeneous start dates.
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col = X.columns[:2]
        time_range_expr = (
            pl.int_ranges(0, pl.count(time_col), dtype=pl.UInt32)
            .over(entity_col)
            .alias(time_col)
        )
        other_cols = pl.all().exclude([entity_col, time_col])
        X_new = X.select([entity_col, time_range_expr, other_cols])
        if eager:
            X_new = X_new.collect(streaming=True)
        artifacts = {"X_new": X_new}
        return artifacts

    return transform


@transformer
def resample(freq: str, agg_method: str, impute_method: str | int | float):
    """
    Resamples and transforms a DataFrame using the specified frequency, aggregation method, and imputation method.

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars.
    agg_method : str
        The aggregation method to use for resampling. Supported values are 'sum', 'mean', and 'median'.
    impute_method : Union[str, int, float]
        The method used for imputing missing values. If a string, supported values are 'ffill' (forward fill)
        and 'bfill' (backward fill). If an int or float, missing values will be filled with the provided value.
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col, target_col = X.columns
        agg_exprs = {
            "sum": pl.sum(target_col),
            "mean": pl.mean(target_col),
            "median": pl.median(target_col),
        }
        X_new = (
            # Defensive resampling
            X.lazy()
            .group_by_dynamic(time_col, every=freq, by=entity_col)
            .agg(agg_exprs[agg_method])
            # Must defensive sort columns otherwise time_col and target_col
            # positions are incorrectly swapped in lazy
            .select([entity_col, time_col, target_col])
            # Impute gaps after reindex
            .pipe(impute(impute_method))
            # Defensive fill null with 0 for impute method `ffill`
            .fill_null(0)
        )
        artifacts = {"X_new": X_new}
        return artifacts

    return transform


@transformer
def trim(direction: Literal["both", "left", "right"] = "both"):
    """Trims time-series in panel to have the same start or end dates as the shortest time-series.

    Parameters
    ----------
    direction : Literal["both", "left", "right"]
        Defaults to "both". If "left" trims from start date of the shortest time series);
        if "right" trims up to the end date of the shortest time-series; or otherwise
        "both" trims between start and end dates of the shortest time-series
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col = X.columns[:2]

        start = pl.col(time_col).min().over(entity_col).max()
        end = pl.col(time_col).max().over(entity_col).min()

        if direction == "both":
            expr = (pl.col(time_col) >= start) & (pl.col(time_col) <= end)
        elif direction == "left":
            expr = pl.col(time_col) >= start
        else:
            expr = pl.col(time_col) <= start
        X_new = X.filter(expr)
        artifacts = {"X_new": X_new}
        return artifacts

    return transform


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


@transformer
def impute(
    method: Literal["mean", "median", "fill", "ffill", "bfill", "interpolate", "cafe"]
    | int
    | float,
    *,
    allow_leaky: bool = False,
):
    """
    Performs missing value imputation on numeric columns of a DataFrame grouped by entity.

    Parameters
    ----------
    method : Union[str, int, float]
        The imputation method to use.

        Supported methods are:\n
        - 'mean': Replace missing values with the mean of the corresponding column.
        - 'median': Replace missing values with the median of the corresponding column.
        - 'fill': Replace missing values with the mean for float columns and the median for integer columns.
        - 'ffill': Forward fill missing values.
        - 'bfill': Backward fill missing values. **Leaky** -- reads future rows.
        - 'interpolate': Interpolate missing values using linear interpolation.
          **Leaky** -- reads future rows.
        - 'cafe': Leak-safe, strictly point-in-time model-based imputation via
          :func:`cafe_impute` (requires the optional ``cafe`` dependency, i.e.
          ``pip install panelary[cafe]``).
        - int or float: Replace missing values with the specified constant.
    allow_leaky : bool, keyword-only, default False
        The ``'bfill'`` and ``'interpolate'`` methods fill a missing value using
        *later* (future) observations, so they leak look-ahead information and
        pass silently through cross-validation. By default using either emits a
        :class:`LeakageWarning` steering you toward a point-in-time imputer
        (``impute('cafe')`` / :func:`cafe_impute`, or ``'ffill'``). Set
        ``allow_leaky=True`` to acknowledge the leak and suppress the warning
        (e.g. for whole-sample EDA outside a backtest). Has no effect on the
        other, leak-safe methods.

    Warns
    -----
    LeakageWarning
        When ``method`` is ``'bfill'`` or ``'interpolate'`` and
        ``allow_leaky`` is ``False``.
    """

    def _warn_if_leaky() -> None:
        if method in _LEAKY_IMPUTE_METHODS and not allow_leaky:
            warnings.warn(
                f"impute(method={method!r}) reads future rows to fill missing "
                "values (backward fill / interpolation look at later "
                "observations), which leaks look-ahead information and passes "
                "silently through cross-validation. Prefer a strictly "
                "point-in-time imputer such as impute('cafe') / cafe_impute "
                "(leak-safe, model-based) or impute('ffill') (forward fill). "
                "Pass allow_leaky=True to acknowledge the leak and silence this "
                "warning.",
                LeakageWarning,
                stacklevel=2,
            )

    def method_to_expr(entity_col, time_col):
        """Fill-in methods."""
        return {
            "mean": PL_NUMERIC_COLS(entity_col, time_col).fill_null(
                PL_NUMERIC_COLS(entity_col, time_col).mean().over(entity_col)
            ),
            "median": PL_NUMERIC_COLS(entity_col, time_col).fill_null(
                PL_NUMERIC_COLS(entity_col, time_col).median().over(entity_col)
            ),
            "fill": [
                cs.float().fill_null(cs.float().mean().over(entity_col)),
                cs.integer().fill_null(cs.integer().median().over(entity_col)),
            ],
            "ffill": PL_NUMERIC_COLS(entity_col, time_col)
            .fill_null(strategy="forward")
            .over(entity_col),
            "bfill": PL_NUMERIC_COLS(entity_col, time_col)
            .fill_null(strategy="backward")
            .over(entity_col),
            "interpolate": PL_NUMERIC_COLS(entity_col, time_col)
            .interpolate()
            .over(entity_col),
        }

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        _warn_if_leaky()
        entity_col, time_col = X.collect_schema().names()[:2]
        # The 'cafe' alias routes to the model-based, point-in-time imputer while
        # leaving every existing method's behaviour untouched.
        if method == "cafe":
            return cafe_impute().func(X)
        if isinstance(method, (int, float)):
            expr = PL_NUMERIC_COLS(entity_col, time_col).fill_null(pl.lit(method))
        else:
            expr = method_to_expr(entity_col, time_col)[method]
        X_new = X.with_columns(expr)
        return {"X_new": X_new}

    return transform


def _require_cafe(feature: str = "cafe_impute"):
    """Lazily import the optional ``cafe`` dependency with an actionable error.

    The single entry point to CAFE for the whole package: both
    :func:`cafe_impute` and :class:`panelary.imputation.CafeImputer` reach the
    ``cafe`` package through here (and only through :func:`_cafe_impute_frame`),
    so the optional dependency is imported lazily, inside the function that
    needs it, exactly once in the codebase.

    Parameters
    ----------
    feature : str, default "cafe_impute"
        Human-readable name of the caller, used to make the missing-dependency
        error concrete.

    Returns
    -------
    ModuleType
        The imported ``cafe`` module.

    Raises
    ------
    ImportError
        If ``cafe`` is not installed, with a ``pip install`` hint.
    """
    return require("cafe", feature=feature)


def _cafe_impute_frame(
    df: pl.DataFrame,
    *,
    entity_col: str,
    time_col: str,
    engine: Literal["joint", "per_entity"] = "joint",
    columns: list[str] | None = None,
    add_uncertainty: bool = False,
    add_anomaly: bool = False,
    add_missingness: bool = False,
    feature: str = "cafe_impute",
) -> pl.DataFrame:
    """Fill a collected panel with CAFE, optionally appending by-products.

    The single CAFE imputation kernel. :func:`cafe_impute` (the functime-shaped
    transformer) and :class:`panelary.imputation.CafeImputer` (the
    :class:`~panelary.core.protocol.PanelTransformer`) are both thin adapters
    over this function, so the two public names cannot drift apart.

    CAFE is strictly point-in-time: every filled cell and every by-product uses
    only past + contemporaneous information within its entity, so this is safe
    inside walk-forward cross-validation and must still be applied per fold.

    Parameters
    ----------
    df : polars.DataFrame
        The collected long-format panel.
    entity_col, time_col : str
        The panel keys.
    engine : {"joint", "per_entity"}, default "joint"
        Panel imputation strategy passed through to CAFE.
    columns : list of str, optional
        Restrict imputation (and any by-products) to these numeric feature
        columns. Defaults to every numeric feature column.
    add_uncertainty : bool, default False
        Append ``<col>__cafe_sigma`` per imputed column.
    add_anomaly : bool, default False
        Append a single per-row ``cafe_anomaly`` score in ``[0, 1]``.
    add_missingness : bool, default False
        Append a boolean ``<col>__cafe_was_imputed`` indicator per imputed column.
    feature : str, default "cafe_impute"
        Caller name used in the missing-dependency error message.

    Returns
    -------
    polars.DataFrame
        The filled panel, with any requested by-product columns appended.

    Raises
    ------
    ImportError
        If the optional ``cafe`` dependency is not installed.
    """
    cafe = _require_cafe(feature)

    keys = {entity_col, time_col}
    num_cols = [c for c, dt in df.schema.items() if dt.is_numeric() and c not in keys]
    target_cols = list(columns) if columns is not None else num_cols

    # The fill itself (lean, cross-entity-aware path).
    filled = cafe.impute(df, panel=(time_col, entity_col), engine=engine)

    # Honour `columns`: restore the original values for numeric columns the
    # caller did not ask to impute.
    untouched = [c for c in num_cols if c not in set(target_cols)]
    if untouched:
        filled = filled.with_columns([df.get_column(c) for c in untouched])

    extra: list[pl.Series] = []

    if add_missingness:
        for col in target_cols:
            extra.append(df.get_column(col).is_null().alias(f"{col}__cafe_was_imputed"))

    if (add_uncertainty or add_anomaly) and target_cols:
        n_rows = df.height
        col_idx = {c: j for j, c in enumerate(target_cols)}
        sigma = np.full((n_rows, len(target_cols)), np.nan)
        anomaly = np.full(n_rows, np.nan)

        # Per-entity, strictly point-in-time traced pass. Rows are gathered in
        # time order within each entity and scattered back to their original
        # positions, so every by-product is causal per entity.
        df_idx = df.with_row_index("__cafe_row__")
        for _, sub in df_idx.group_by(entity_col, maintain_order=True):
            sub = sub.sort(time_col)
            rows = sub.get_column("__cafe_row__").to_numpy()
            mat = sub.select(target_cols).to_numpy().astype(float)
            res = cafe.CAFE().run(mat)
            if add_uncertainty:
                sigma[rows] = np.asarray(res.uncertainty)
            if add_anomaly:
                anomaly[rows] = np.asarray(res.anomaly_scores())

        for col in target_cols:
            j = col_idx[col]
            if add_uncertainty:
                extra.append(pl.Series(f"{col}__cafe_sigma", sigma[:, j]))
        if add_anomaly:
            extra.append(pl.Series("cafe_anomaly", anomaly))

    if extra:
        filled = filled.with_columns(extra)

    return filled


@transformer
def cafe_impute(
    engine: Literal["joint", "per_entity"] = "joint",
    add_uncertainty: bool = False,
    add_recoverability: bool = False,
    add_anomaly: bool = False,
    add_missingness: bool = False,
    columns: list[str] | None = None,
):
    """Leak-safe, model-based imputation of a panel via CAFE.

    CAFE (Causal Adaptive Factor Estimation) is a strictly point-in-time
    (no look-ahead) imputer: each filled cell uses only past + contemporaneous
    information within its entity, so the transform is safe to use inside
    walk-forward cross-validation. Numeric feature columns are filled; the
    entity/time keys and any non-numeric columns pass through untouched and the
    original column order is preserved.

    Requires the optional ``cafe`` dependency (``pip install panelary[cafe]``).

    Parameters
    ----------
    engine : {"joint", "per_entity"}, default "joint"
        Panel imputation strategy. ``"joint"`` pools the contemporaneous
        cross-section (the validated default); ``"per_entity"`` imputes each
        entity's series independently.
    add_uncertainty : bool, default False
        Append a per-cell posterior standard deviation column ``<col>__cafe_sigma``
        for every imputed numeric column (NaN where the value was observed).
    add_recoverability : bool, default False
        **Not supported.** Kept only so the keyword stays a stable part of the
        signature; passing ``True`` raises :class:`NotImplementedError`. See the
        Notes below.
    add_anomaly : bool, default False
        Append a single per-row outlier score column ``cafe_anomaly`` in
        ``[0, 1]`` (0 = perfect fit, 1 = strong outlier), causal per entity.
    add_missingness : bool, default False
        Append a boolean ``<col>__cafe_was_imputed`` indicator per imputed column
        that preserves the original missing pattern (which imputation erases).
    columns : list of str, optional
        Restrict imputation (and any by-products) to these numeric feature
        columns. Other numeric columns pass through with their original values.
        Defaults to every numeric feature column.

    Raises
    ------
    NotImplementedError
        If ``add_recoverability=True``. ``cafe-impute`` (the only published
        version is 0.1.0) exposes no recoverability certificate: ``CafeResult``
        offers ``uncertainty`` / ``confidence_interval`` (posterior predictive
        sd), ``anomaly_scores``, ``missingness_features``, ``decompose``,
        ``factors``, ``effective_rank`` and ``dependency_network`` -- and
        nothing else. A ``[0, 1]`` certificate would need the *prior* (pre-
        conditioning) variance of each cell to normalise the posterior variance
        against, and that quantity is not recorded in CAFE's trace. Normalising
        by a sample variance instead would either be an invented metric or,
        worse, use the whole sample and break the point-in-time contract, so
        this raises instead of guessing.

    Notes
    -----
    The by-products (uncertainty / anomaly) are produced by the strictly-causal
    per-entity traced pass, so appending future rows never changes an earlier
    ``(entity, time)`` cell's value or by-product.

    The work is done by :func:`_cafe_impute_frame`, the one CAFE kernel;
    :class:`panelary.imputation.CafeImputer` is the ``PanelTransformer``-shaped
    adapter over the same kernel.
    """
    if add_recoverability:
        raise NotImplementedError(
            "cafe_impute(add_recoverability=True) is not supported: the `cafe` "
            "package (0.1.0) exposes no recoverability certificate, and any "
            "[0, 1] score derived from what it does expose would either be "
            "invented or would need whole-sample normalisation, which would "
            "break the point-in-time contract. Use `add_uncertainty=True` for "
            "the per-cell posterior standard deviation instead."
        )

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col = X.collect_schema().names()[:2]
        filled = _cafe_impute_frame(
            X.collect(),
            entity_col=entity_col,
            time_col=time_col,
            engine=engine,
            columns=columns,
            add_uncertainty=add_uncertainty,
            add_anomaly=add_anomaly,
            add_missingness=add_missingness,
            feature="cafe_impute",
        )
        return {"X_new": filled.lazy()}

    return transform


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
    Routes through the shared, causal :func:`panelary._ffd.frac_diff_expr`
    builder -- the single source of truth shared with ``.panel.frac_diff``,
    ``.ts.frac_diff`` and :class:`~panelary.transform.frac_diff.FracDiff`
    -- applied per entity via ``.over(entity)``. The incomplete leading window is
    emitted as ``null`` (never zero-filled).
    """
    from panelary._ffd import frac_diff_expr

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
