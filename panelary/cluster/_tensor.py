"""Polars long -> (entity, time, feature) tensor builder for k-Shape clustering.

This replaces SovAI's ``pandas_to_array`` (``clustering.py`` L13-70), whose two
leaks are deliberately *not* reproduced:

* it back-filled missing values (``bfill`` -- pulls future observations
  backward), and
* it fit a ``StandardScaler`` over the **entire** sample (mean/std computed
  across all dates, including the future).

The causal replacement here:

* pivots the long panel onto a dense ``(entity x time)`` grid (a cross join, so
  every entity has an equal-length series),
* **forward-fills within each entity only** (``forward_fill().over(entity)``),
  never backward, and
* optionally applies a per-entity **expanding (causal) z-normalisation**: the
  value at time ``t`` is standardised using only that entity's observations up to
  and including ``t``. Rows without enough history to estimate a variance are
  emitted as nulls (NaN in the tensor), never fabricated.

The result is a NaN-containing ``(n_entities, n_times, n_values)`` float tensor
plus the sorted entity / time indices and the matching dense key frame, so
callers can align computed features straight back onto ``(entity, time)`` rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame

__all__ = ["PanelTensor", "build_tensor"]


class PanelTensor(NamedTuple):
    """A dense panel tensor plus the indices needed to realign it.

    Attributes
    ----------
    tensor : numpy.ndarray
        ``(n_entities, n_times, n_values)`` float array, NaN where a value is
        missing / has insufficient history. Entity-major, time-ordered.
    entities : polars.Series
        Sorted unique entity ids (row order of ``tensor``).
    times : polars.Series
        Sorted unique time values (column order of ``tensor``).
    keys : polars.DataFrame
        The dense ``(entity, time)`` grid, sorted entity-major then by time, so
        ``keys`` rows line up with ``tensor.reshape(n_entities * n_times, ...)``.
    value_cols : list of str
        The value column names, in the tensor's last-axis order.
    """

    tensor: np.ndarray
    entities: pl.Series
    times: pl.Series
    keys: pl.DataFrame
    value_cols: list[str]


def build_tensor(
    panel: PanelFrame,
    value_cols: Sequence[str],
    *,
    forward_fill: bool = True,
    z_normalize: bool = True,
    ddof: int = 1,
    min_history: int = 2,
) -> PanelTensor:
    """Build a causal, dense ``(entity, time, value)`` tensor from ``panel``.

    Parameters
    ----------
    panel : PanelFrame
        The source panel.
    value_cols : sequence of str
        Feature columns whose per-entity series form the tensor's last axis.
    forward_fill : bool, default=True
        Forward-fill missing values within each entity (never backward).
    z_normalize : bool, default=True
        Apply per-entity expanding (causal) z-normalisation.
    ddof : int, default=1
        Delta degrees of freedom for the causal standard deviation.
    min_history : int, default=2
        Minimum number of observations required before a z-normalised value is
        emitted; earlier rows become NaN.

    Returns
    -------
    PanelTensor
    """
    value_cols = list(value_cols)
    if not value_cols:
        raise ValueError("`value_cols` must name at least one feature column.")
    ent, tim = panel.entity_col, panel.time_col
    missing = [c for c in value_cols if c not in panel]
    if missing:
        raise ValueError(
            f"value column(s) {missing} not found in panel. "
            f"Available columns: {panel.columns}."
        )

    base = panel.lazy().select([ent, tim, *value_cols]).collect()
    entities = base.select(pl.col(ent).unique().sort()).to_series()
    times = base.select(pl.col(tim).unique().sort()).to_series()
    n_ent, n_times = entities.len(), times.len()

    # Dense (entity x time) grid via cross join, so every entity is equal-length.
    grid = entities.to_frame().join(times.to_frame(), how="cross")
    merged = (
        grid.join(base, on=[ent, tim], how="left")
        .sort([ent, tim])
        .with_columns([pl.col(v).cast(pl.Float64) for v in value_cols])
    )

    if forward_fill:
        merged = merged.with_columns(
            [pl.col(v).forward_fill().over(ent).alias(v) for v in value_cols]
        )

    if z_normalize:
        znorm_exprs = []
        for v in value_cols:
            x = pl.col(v)
            cnt = x.is_not_null().cum_sum().over(ent)
            s1 = x.fill_null(0.0).cum_sum().over(ent)
            s2 = (x.fill_null(0.0) ** 2).cum_sum().over(ent)
            mean = s1 / cnt
            var = (s2 - cnt * mean**2) / (cnt - ddof)
            std = var.sqrt()
            z = (x - mean) / std
            znorm_exprs.append(
                pl.when((cnt >= min_history) & (std > 0))
                .then(z)
                .otherwise(None)
                .alias(v)
            )
        merged = merged.with_columns(znorm_exprs)

    keys = merged.select([ent, tim])
    tensor = np.empty((n_ent, n_times, len(value_cols)), dtype=float)
    for vi, v in enumerate(value_cols):
        tensor[:, :, vi] = merged.get_column(v).to_numpy().reshape(n_ent, n_times)

    return PanelTensor(
        tensor=tensor,
        entities=entities,
        times=times,
        keys=keys,
        value_cols=value_cols,
    )
