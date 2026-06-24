"""Fractional differencing (fixed-width window FFD).

:class:`FracDiff` applies fractional differencing of order ``d`` to each
entity's series using the **fixed-width window** ("FFD") formulation of
Marcos Lopez de Prado, *Advances in Financial Machine Learning* (Wiley, 2018),
Chapter 5. Fractional differencing makes a series stationary while preserving
as much memory (long-range dependence) as possible -- a middle ground between
the raw level (maximum memory, often non-stationary) and the first difference
(stationary, but memory destroyed).

The transform is **causal** (each output uses only the current and past values
within the same entity) and respects entity boundaries, so it is both
panel-safe and leakage-safe.

Implementation note
-------------------
A Rust ``frac_diff`` plugin exists in this project, but it is **not compiled**
in every environment. This module is the portable, pure-Python/Polars
reference: the binomial weights are computed in NumPy and applied as a causal
weighted rolling dot-product per entity via ``expr.over(entity_col)``. Results
should match the compiled plugin up to floating point.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from polars_features.core.panel_frame import PanelFrame
from polars_features.core.protocol import PanelTransformer

if TYPE_CHECKING:
    pass

__all__ = ["FracDiff", "ffd_weights"]


def ffd_weights(
    d: float, threshold: float = 1e-5, max_width: int | None = None
) -> np.ndarray:
    """Compute fixed-width-window fractional-differencing weights.

    The weights follow the recurrence (de Prado 2018, eq. 5.x)::

        w_0 = 1
        w_k = -w_{k-1} * (d - k + 1) / k

    Generation stops once ``|w_k| < threshold`` (or ``max_width`` is reached).
    The returned array is ordered from the *oldest* lag to the *current*
    observation, i.e. ``weights[-1]`` multiplies ``x_t`` and ``weights[0]``
    multiplies ``x_{t-(width-1)}``. This ordering makes it a drop-in kernel for
    a trailing weighted window.

    Parameters
    ----------
    d : float
        Differencing order. ``0`` is the identity, ``1`` is the first
        difference; non-integer values give fractional differencing.
    threshold : float, default=1e-5
        Magnitude below which trailing weights are dropped (controls window
        width). Must be positive.
    max_width : int, optional
        Hard cap on the window width, regardless of ``threshold``.

    Returns
    -------
    numpy.ndarray
        1-D array of weights, oldest-to-newest.
    """
    if threshold <= 0:
        raise ValueError(f"`threshold` must be positive, got {threshold!r}.")
    if max_width is not None and max_width < 1:
        raise ValueError(f"`max_width` must be >= 1, got {max_width!r}.")

    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
        if max_width is not None and len(weights) >= max_width:
            break
    # newest-first by construction (w_0 multiplies x_t); reverse to oldest-first
    return np.asarray(weights[::-1], dtype=np.float64)


class FracDiff(PanelTransformer):
    """Fixed-width fractional differencing per entity (causal).

    Applies a trailing weighted window (the FFD kernel) to each entity's
    series. The first ``width - 1`` observations of each entity have an
    incomplete window and are emitted as null.

    Parameters
    ----------
    columns : str or sequence of str, optional
        Columns to fractionally difference. ``None`` (default) processes every
        feature column.
    d : float, default=0.5
        Differencing order (see :func:`ffd_weights`). Must be in ``[0, 2]``.
    threshold : float, default=1e-5
        Weight-magnitude cutoff controlling the fixed window width.
    max_width : int, optional
        Hard cap on the window width.
    suffix : str, default="_fracdiff"
        Suffix for the output columns. Set to ``""`` to overwrite in place.

    Attributes
    ----------
    panel_safe : bool
        Always ``True`` -- windows never cross entity boundaries
        (``.over(entity_col)``).
    leakage_safe : bool
        Always ``True`` -- the kernel is strictly trailing (causal): each output
        uses only the current and past observations of its own entity.
    weights_ : numpy.ndarray
        The fitted FFD weights (oldest-to-newest), available after :meth:`fit`.
    width_ : int
        The window width (``len(weights_)``).

    References
    ----------
    Marcos Lopez de Prado, *Advances in Financial Machine Learning*, Wiley,
    2018, Chapter 5 ("Fractionally Differentiated Features").

    Examples
    --------
    >>> import polars as pl
    >>> from polars_features.core import PanelFrame
    >>> df = pl.DataFrame(
    ...     {"id": ["a"] * 6 + ["b"] * 6,
    ...      "t": list(range(6)) * 2,
    ...      "px": [1.0, 1.1, 1.2, 1.15, 1.3, 1.25] * 2}
    ... )
    >>> panel = PanelFrame(df, entity="id", time="t").sort_panel()
    >>> out = FracDiff(d=0.4, threshold=1e-3).fit_transform(panel)
    >>> "px_fracdiff" in out.columns
    True
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        columns: str | Sequence[str] | None = None,
        *,
        d: float = 0.5,
        threshold: float = 1e-5,
        max_width: int | None = None,
        suffix: str = "_fracdiff",
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if not (0.0 <= d <= 2.0):
            raise ValueError(f"`d` must be in [0, 2], got {d!r}.")
        if threshold <= 0:
            raise ValueError(f"`threshold` must be positive, got {threshold!r}.")
        if max_width is not None and max_width < 1:
            raise ValueError(f"`max_width` must be >= 1, got {max_width!r}.")
        if columns is None:
            self.columns: list[str] | None = None
        elif isinstance(columns, str):
            self.columns = [columns]
        else:
            cols = list(columns)
            if not cols:
                raise ValueError(
                    "`columns` was an empty sequence; pass None to process all features."
                )
            self.columns = cols
        self.d = float(d)
        self.threshold = float(threshold)
        self.max_width = max_width
        self.suffix = suffix
        # learned state
        self.weights_: np.ndarray | None = None
        self.width_: int = 0
        self._cols_: list[str] = []

    def _resolve_columns(self, panel: PanelFrame) -> list[str]:
        schema = panel.schema
        if self.columns is None:
            cols = [c for c in panel.feature_cols if schema[c].is_numeric()]
            if not cols:
                raise ValueError(
                    "FracDiff: panel has no numeric feature columns to difference "
                    f"(columns={panel.columns}). Pass `columns=` explicitly."
                )
            return cols
        available = set(panel.columns)
        missing = [c for c in self.columns if c not in available]
        if missing:
            raise ValueError(
                f"FracDiff.fit: column(s) {missing} not found in panel. "
                f"Available columns: {panel.columns}."
            )
        bad = [c for c in self.columns if not schema[c].is_numeric()]
        if bad:
            raise ValueError(
                f"FracDiff.fit: column(s) {bad} are not numeric and cannot be "
                "fractionally differenced. Pass only numeric columns via `columns=`."
            )
        return self.columns

    def _fit(self, panel: PanelFrame) -> None:
        self._cols_ = self._resolve_columns(panel)
        self.weights_ = ffd_weights(self.d, self.threshold, self.max_width)
        self.width_ = int(self.weights_.shape[0])

    def _fracdiff_expr(self, col: str, entity_col: str) -> pl.Expr:
        """Build a causal weighted-window dot product for ``col`` per entity.

        Sum over j of weights[j] * shift(width-1-j), i.e. weights[-1]*x_t plus
        older lags, computed within each entity. Windows that include a null
        (the leading edge of each entity) collapse to null.
        """
        assert self.weights_ is not None
        width = self.width_
        terms: list[pl.Expr] = []
        for j, w in enumerate(self.weights_):
            lag = (width - 1) - j  # j is oldest-first; lag is shift amount
            terms.append(pl.lit(float(w)) * pl.col(col).shift(lag).over(entity_col))
        return sum(terms[1:], terms[0])

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        if self.weights_ is None:  # pragma: no cover - guarded by _check_fitted
            raise RuntimeError("FracDiff is not fitted.")
        available = set(panel.columns)
        missing = [c for c in self._cols_ if c not in available]
        if missing:
            raise ValueError(
                f"FracDiff.transform: column(s) {missing} not found in panel. "
                f"Available columns: {panel.columns}."
            )
        entity_col = panel.entity_col
        out_exprs = [
            self._fracdiff_expr(col, entity_col).alias(f"{col}{self.suffix}")
            for col in self._cols_
        ]
        lf = panel.lazy().with_columns(out_exprs)
        return PanelFrame(lf, entity=entity_col, time=panel.time_col, validate=False)
