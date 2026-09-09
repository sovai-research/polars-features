"""Leak-safe scaling transformers for panels.

Two complementary scalers live here:

* :class:`TimeSeriesScaler` rescales each entity's series using statistics
  learned **on the training rows only** (fit-on-train, apply-at-transform), so
  the transform of a test fold never sees test-fold statistics. This is the
  panel analogue of :class:`sklearn.preprocessing.StandardScaler`, but the
  parameters are learned per entity (or globally) rather than per column over a
  flat table.
* :class:`CrossSectionalScaler` standardizes within each timestamp across
  entities (``.over(time_col)``). Because every row is rescaled using only
  values observed at *the same date*, it is leak-safe by construction without
  any fitted state.

Both transformers honour the :class:`~panelary.core.protocol.PanelTransformer`
contract and declare ``panel_safe = True`` and ``leakage_safe = True``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer

if TYPE_CHECKING:
    pass

__all__ = ["TimeSeriesScaler", "CrossSectionalScaler"]

_TS_MODES = ("zscore", "minmax", "robust")
ScalerMode = Literal["zscore", "minmax", "robust"]

# A small floor for denominators so constant series do not produce inf/nan.
_EPS = 1e-12


def _as_column_list(columns: str | Sequence[str] | None) -> list[str] | None:
    """Normalise a column argument to ``None`` or a list of names."""
    if columns is None:
        return None
    if isinstance(columns, str):
        return [columns]
    cols = list(columns)
    if not cols:
        raise ValueError(
            "`columns` was an empty sequence; pass None to scale all features."
        )
    return cols


def _check_columns_exist(
    panel: PanelFrame, columns: Sequence[str], *, where: str
) -> None:
    """Raise a helpful error if any requested column is missing from ``panel``."""
    available = set(panel.columns)
    missing = [c for c in columns if c not in available]
    if missing:
        raise ValueError(
            f"{where}: column(s) {missing} not found in panel. "
            f"Available columns: {panel.columns}."
        )


def _check_columns_numeric(
    panel: PanelFrame, columns: Sequence[str], *, where: str
) -> None:
    """Raise if any requested column is not numeric (scaling needs arithmetic)."""
    schema = panel.schema
    bad = [c for c in columns if not schema[c].is_numeric()]
    if bad:
        raise ValueError(
            f"{where}: column(s) {bad} are not numeric and cannot be scaled "
            f"(dtypes: {[(c, schema[c]) for c in bad]}). Pass only numeric "
            "columns via `columns=`."
        )


def _numeric_feature_cols(panel: PanelFrame) -> list[str]:
    """Return feature columns with a numeric dtype, in schema order."""
    schema = panel.schema
    return [c for c in panel.feature_cols if schema[c].is_numeric()]


class TimeSeriesScaler(PanelTransformer):
    """Per-entity (or global) rescaling using train-only statistics.

    Statistics are learned in :meth:`fit` from the rows handed to it (the
    training panel) and re-applied verbatim in :meth:`transform`. Nothing is
    re-estimated at transform time, so applying a fitted scaler to a future
    test fold cannot leak test-fold information into the features.

    Parameters
    ----------
    columns : str or sequence of str, optional
        Feature columns to scale. ``None`` (default) scales every feature
        column (everything that is neither the entity nor the time key).
    mode : {"zscore", "minmax", "robust"}, default="zscore"
        Scaling rule.

        * ``"zscore"`` : ``(x - mean) / std``.
        * ``"minmax"`` : ``(x - min) / (max - min)`` mapping to ``[0, 1]``.
        * ``"robust"`` : ``(x - median) / IQR`` with ``IQR = q75 - q25``.
    by_entity : bool, default=True
        If True, learn one set of statistics per entity. If False, learn a
        single global statistic per column (shared across all entities).
    suffix : str, optional
        If given, scaled columns are written to ``f"{col}{suffix}"`` and the
        originals are kept. If ``None`` (default) the scaled values overwrite
        the original columns in place.

    Attributes
    ----------
    panel_safe : bool
        Always ``True`` -- statistics are computed within entity boundaries
        (or globally) and never mix one entity's transform into another.
    leakage_safe : bool
        Always ``True`` -- parameters come solely from the ``fit`` panel.
    stats_ : polars.DataFrame
        The learned per-entity (or single-row global) statistics. Available
        after :meth:`fit`.

    Notes
    -----
    **Leakage guarantee.** ``fit`` reads only its input rows; ``transform``
    only joins/broadcasts those learned constants. There is no expanding or
    rolling recomputation at transform time, so for any walk-forward split
    ``scaler.fit(train).transform(test)`` uses train statistics only.

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.core import PanelFrame
    >>> df = pl.DataFrame(
    ...     {"id": ["a", "a", "b", "b"], "t": [1, 2, 1, 2], "x": [1.0, 3.0, 10.0, 20.0]}
    ... )
    >>> panel = PanelFrame(df, entity="id", time="t")
    >>> out = TimeSeriesScaler(mode="zscore").fit_transform(panel)
    >>> out.collect().shape
    (4, 3)
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        columns: str | Sequence[str] | None = None,
        *,
        mode: ScalerMode = "zscore",
        by_entity: bool = True,
        suffix: str | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if mode not in _TS_MODES:
            raise ValueError(f"`mode` must be one of {_TS_MODES}, got {mode!r}.")
        self.columns = _as_column_list(columns)
        self.mode: ScalerMode = mode
        self.by_entity = bool(by_entity)
        self.suffix = suffix
        # learned state
        self.stats_: pl.DataFrame | None = None
        self._scaled_cols_: list[str] = []

    # ------------------------------------------------------------------ #
    def _resolve_columns(self, panel: PanelFrame) -> list[str]:
        if self.columns is None:
            cols = _numeric_feature_cols(panel)
            if not cols:
                raise ValueError(
                    "TimeSeriesScaler: panel has no numeric feature columns to "
                    f"scale (columns={panel.columns}). Pass `columns=` explicitly."
                )
            return cols
        _check_columns_exist(panel, self.columns, where="TimeSeriesScaler.fit")
        _check_columns_numeric(panel, self.columns, where="TimeSeriesScaler.fit")
        return self.columns

    def _stat_exprs(self, col: str) -> list[pl.Expr]:
        """Return the aggregation expressions needed for ``col`` under ``mode``."""
        c = pl.col(col)
        if self.mode == "zscore":
            return [
                c.mean().alias(f"{col}__center"),
                c.std(ddof=0).alias(f"{col}__scale"),
            ]
        if self.mode == "minmax":
            return [
                c.min().alias(f"{col}__center"),
                (c.max() - c.min()).alias(f"{col}__scale"),
            ]
        # robust
        return [
            c.median().alias(f"{col}__center"),
            (c.quantile(0.75) - c.quantile(0.25)).alias(f"{col}__scale"),
        ]

    def _fit(self, panel: PanelFrame) -> None:
        cols = self._resolve_columns(panel)
        self._scaled_cols_ = list(cols)
        agg_exprs: list[pl.Expr] = []
        for col in cols:
            agg_exprs.extend(self._stat_exprs(col))

        lf = panel.lazy()
        if self.by_entity:
            stats = lf.group_by(panel.entity_col).agg(agg_exprs)
        else:
            stats = lf.select(agg_exprs)
        self.stats_ = stats.collect()

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        if self.stats_ is None:  # pragma: no cover - guarded by _check_fitted
            raise RuntimeError("TimeSeriesScaler is not fitted.")
        _check_columns_exist(
            panel, self._scaled_cols_, where="TimeSeriesScaler.transform"
        )

        lf = panel.lazy()
        if self.by_entity:
            lf = lf.join(self.stats_.lazy(), on=panel.entity_col, how="left")
        else:
            # broadcast the single-row global stats onto every row
            lf = lf.with_columns(
                [pl.lit(self.stats_[c][0]).alias(c) for c in self.stats_.columns]
            )

        out_exprs: list[pl.Expr] = []
        for col in self._scaled_cols_:
            center = pl.col(f"{col}__center")
            scale = pl.col(f"{col}__scale")
            # protect against zero/near-zero scale (constant series)
            safe_scale = pl.when(scale.abs() < _EPS).then(pl.lit(1.0)).otherwise(scale)
            scaled = (pl.col(col) - center) / safe_scale
            target = col if self.suffix is None else f"{col}{self.suffix}"
            out_exprs.append(scaled.alias(target))

        lf = lf.with_columns(out_exprs)
        # drop the temporary stat columns we joined/broadcast in
        stat_cols = [
            c for c in self.stats_.columns if c.endswith(("__center", "__scale"))
        ]
        lf = lf.drop(stat_cols)
        return PanelFrame(
            lf, entity=panel.entity_col, time=panel.time_col, validate=False
        )


class CrossSectionalScaler(PanelTransformer):
    """Standardize within each timestamp across entities (cross-section).

    For each date, the chosen column(s) are rescaled using statistics computed
    over *all entities observed at that same date* via ``expr.over(time_col)``.
    Because only same-date information is used, the transform is leak-safe by
    construction and carries no fitted state across the train/test boundary.

    Parameters
    ----------
    columns : str or sequence of str, optional
        Columns to standardize. ``None`` (default) standardizes every feature
        column.
    mode : {"zscore", "minmax", "robust"}, default="zscore"
        Same rules as :class:`TimeSeriesScaler`, but computed per timestamp.
    suffix : str, optional
        If given, results go to ``f"{col}{suffix}"`` and originals are kept;
        otherwise the columns are overwritten in place.

    Attributes
    ----------
    panel_safe : bool
        Always ``True``.
    leakage_safe : bool
        Always ``True`` -- each row only ever sees its own cross-section.

    Notes
    -----
    **Leakage guarantee.** The operation is purely a same-date cross-sectional
    z-score/min-max/robust scale. No past or future timestamps participate, so
    there is nothing to learn in :meth:`fit` (it is a no-op kept for API
    symmetry) and nothing can leak.

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.core import PanelFrame
    >>> df = pl.DataFrame(
    ...     {"id": ["a", "b", "a", "b"], "t": [1, 1, 2, 2], "x": [1.0, 3.0, 2.0, 8.0]}
    ... )
    >>> panel = PanelFrame(df, entity="id", time="t")
    >>> out = CrossSectionalScaler(mode="zscore").fit_transform(panel)
    >>> out.collect().shape
    (4, 3)
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        columns: str | Sequence[str] | None = None,
        *,
        mode: ScalerMode = "zscore",
        suffix: str | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if mode not in _TS_MODES:
            raise ValueError(f"`mode` must be one of {_TS_MODES}, got {mode!r}.")
        self.columns = _as_column_list(columns)
        self.mode: ScalerMode = mode
        self.suffix = suffix
        self._scaled_cols_: list[str] = []

    def _resolve_columns(self, panel: PanelFrame) -> list[str]:
        if self.columns is None:
            cols = _numeric_feature_cols(panel)
            if not cols:
                raise ValueError(
                    "CrossSectionalScaler: panel has no numeric feature columns "
                    f"to scale (columns={panel.columns}). Pass `columns=` explicitly."
                )
            return cols
        _check_columns_exist(panel, self.columns, where="CrossSectionalScaler.fit")
        _check_columns_numeric(panel, self.columns, where="CrossSectionalScaler.fit")
        return self.columns

    def _fit(self, panel: PanelFrame) -> None:
        # Stateless: only resolves/validates the target columns.
        self._scaled_cols_ = self._resolve_columns(panel)

    def _scaled_expr(self, col: str, time_col: str) -> pl.Expr:
        c = pl.col(col)
        if self.mode == "zscore":
            center = c.mean().over(time_col)
            scale = c.std(ddof=0).over(time_col)
        elif self.mode == "minmax":
            center = c.min().over(time_col)
            scale = (c.max() - c.min()).over(time_col)
        else:  # robust
            center = c.median().over(time_col)
            scale = (c.quantile(0.75) - c.quantile(0.25)).over(time_col)
        safe_scale = pl.when(scale.abs() < _EPS).then(pl.lit(1.0)).otherwise(scale)
        return (c - center) / safe_scale

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        _check_columns_exist(
            panel, self._scaled_cols_, where="CrossSectionalScaler.transform"
        )
        time_col = panel.time_col
        out_exprs = [
            self._scaled_expr(col, time_col).alias(
                col if self.suffix is None else f"{col}{self.suffix}"
            )
            for col in self._scaled_cols_
        ]
        lf = panel.lazy().with_columns(out_exprs)
        return PanelFrame(
            lf, entity=panel.entity_col, time=panel.time_col, validate=False
        )
