"""Missing-value imputation: ``impute`` and the CAFE kernel.

``impute`` covers the expression-based fills; ``'bfill'`` and ``'interpolate'``
read future rows and therefore emit a :class:`~panelary.preprocessing.LeakageWarning`
unless ``allow_leaky=True``. :func:`_cafe_impute_frame` is the single CAFE
kernel, shared with :class:`panelary.imputation.CafeImputer`; it is re-exported
from ``panelary.preprocessing`` because that is where the imputer imports it
from, at call time.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import polars as pl
import polars.selectors as cs

from panelary._internal._deps import require
from panelary.base.transformer import transformer
from panelary.preprocessing._base import (
    _LEAKY_IMPUTE_METHODS,
    PL_NUMERIC_COLS,
    LeakageWarning,
)


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
