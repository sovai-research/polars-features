"""Cross-sectional factor neutralization.

:class:`Neutralize` removes, within each date, the component of a target column
that is linearly explained by a set of factor columns (e.g. sector dummies,
market beta, size). For each timestamp it runs an ordinary least-squares
regression of the target on the factors *across entities* and keeps the
**residual**. The residualised target is "neutral" to those factors at that
date.

This is leak-safe by construction: the regression for date ``t`` uses only the
cross-section at date ``t`` (same-date entities), so no past or future
information enters any row, and there is no fitted state carried across a
train/test boundary.

The per-cross-section OLS itself lives in the dependency-free leaf
:mod:`panelary.namespaces._neutralize_kernel`, which is also what backs
``pl.col(...).xs.neutralize(...)``. The transformer and the expression namespace
therefore share **one** implementation rather than two that can silently
diverge. (The kernel sits under ``namespaces/`` because that package must not
import ``transform``; see its module docstring.)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer
from panelary.namespaces._neutralize_kernel import (
    build_design,
    cross_section_residuals,
)

if TYPE_CHECKING:
    pass

__all__ = ["Neutralize"]


def _as_list(x: str | Sequence[str]) -> list[str]:
    return [x] if isinstance(x, str) else list(x)


class Neutralize(PanelTransformer):
    """Cross-sectional OLS factor neutralization (keep residuals).

    Within every date, regress ``target`` on ``factors`` across entities and
    replace the target with the regression residual. Numeric factors are used
    as-is; categorical / string factors are one-hot encoded (per date) into
    dummy columns before the regression.

    Parameters
    ----------
    target : str
        The column to neutralize.
    factors : str or sequence of str
        Factor column(s) to neutralize against. Numeric columns enter directly;
        String / Categorical / Boolean columns are one-hot encoded within each
        date's cross-section.
    add_intercept : bool, default=True
        If True, an intercept term is included in the regression so the
        residual is also de-meaned per date. Set False to neutralize only
        against the supplied factors without de-meaning.
    suffix : str, default="_neutral"
        Suffix for the residual column. Set to ``""`` to overwrite ``target``.

    Attributes
    ----------
    panel_safe : bool
        Always ``True``.
    leakage_safe : bool
        Always ``True`` -- each date is regressed independently on its own
        cross-section.

    Notes
    -----
    **Leakage guarantee.** The OLS fit is performed per ``time_col`` group on
    same-date rows only (via :meth:`polars.LazyFrame.group_by` +
    :func:`numpy.linalg.lstsq`). No information crosses dates, so for any split
    ``Neutralize().fit(train).transform(test)`` yields test residuals that
    depend only on each test date's own cross-section. :meth:`fit` is stateless
    (kept for protocol symmetry); the regression happens at transform time.

    Rows whose target or any factor is null are excluded from that date's fit
    and receive a null residual (the regression is undefined for them).

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.core import PanelFrame
    >>> df = pl.DataFrame(
    ...     {"id": ["a", "b", "c", "a", "b", "c"],
    ...      "t": [1, 1, 1, 2, 2, 2],
    ...      "ret": [0.10, 0.05, 0.20, -0.10, 0.00, 0.30],
    ...      "beta": [1.0, 0.5, 1.5, 1.0, 0.5, 1.5],
    ...      "sector": ["x", "y", "x", "x", "y", "x"]}
    ... )
    >>> panel = PanelFrame(df, entity="id", time="t")
    >>> out = Neutralize("ret", factors=["beta", "sector"]).fit_transform(panel)
    >>> out.collect().columns
    ['id', 't', 'ret', 'beta', 'sector', 'ret_neutral']
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        target: str,
        factors: str | Sequence[str],
        *,
        add_intercept: bool = True,
        suffix: str = "_neutral",
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if not isinstance(target, str):
            raise TypeError(
                f"`target` must be a column name (str), got {type(target).__name__!r}."
            )
        self.target = target
        self.factors = _as_list(factors)
        if not self.factors:
            raise ValueError("`factors` must contain at least one column.")
        if target in self.factors:
            raise ValueError(
                f"`target` {target!r} must not also appear in `factors` {self.factors}."
            )
        self.add_intercept = bool(add_intercept)
        self.suffix = suffix

    def _fit(self, panel: PanelFrame) -> None:
        # Stateless: validate that target & factors exist on the fit panel.
        available = set(panel.columns)
        missing = [c for c in [self.target, *self.factors] if c not in available]
        if missing:
            raise ValueError(
                f"Neutralize.fit: column(s) {missing} not found in panel. "
                f"Available columns: {panel.columns}."
            )

    def _build_design(self, df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Build the (one-hot expanded) numeric design matrix for one date.

        Thin wrapper over
        :func:`panelary.namespaces._neutralize_kernel.build_design`; returns the
        design matrix (without intercept) and the column names used.
        """
        return build_design(df, self.factors)

    def _residual_for_group(self, df: pl.DataFrame) -> pl.Series:
        """Compute per-row residuals for one date's cross-section.

        Delegates to
        :func:`panelary.namespaces._neutralize_kernel.cross_section_residuals`,
        the single implementation shared with ``pl.col(...).xs.neutralize(...)``
        (:mod:`panelary.namespaces.xs`), so the transformer and the expression
        namespace can never drift apart.
        """
        residual = cross_section_residuals(
            df, self.target, self.factors, add_intercept=self.add_intercept
        )
        return pl.Series(self.target, residual)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        available = set(panel.columns)
        missing = [c for c in [self.target, *self.factors] if c not in available]
        if missing:
            raise ValueError(
                f"Neutralize.transform: column(s) {missing} not found in panel. "
                f"Available columns: {panel.columns}."
            )

        entity_col = panel.entity_col
        time_col = panel.time_col
        df = panel.collect()

        overwrite = self.suffix == ""
        resid_name = (
            f"{self.target}{self.suffix}"
            if not overwrite
            else f"{self.target}__neutral_tmp"
        )

        # Stable per-date residuals, then re-attach by (entity, time) key.
        residuals = (
            df.group_by(time_col, maintain_order=True)
            .agg(
                pl.col(entity_col),
                pl.struct([self.target, *self.factors])
                .map_batches(
                    lambda s, _self=self: _self._residual_for_group(s.struct.unnest()),
                    return_dtype=pl.Float64,
                )
                .alias(resid_name),
            )
            .explode([entity_col, resid_name])
            .select(time_col, entity_col, resid_name)
        )

        out = df.join(residuals, on=[time_col, entity_col], how="left")
        if overwrite:
            out = out.drop(self.target).rename({resid_name: self.target})
        return PanelFrame(out.lazy(), entity=entity_col, time=time_col, validate=False)
