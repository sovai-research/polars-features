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
    >>> from polars_features.core import PanelFrame
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

        Returns the design matrix (without intercept) and the column names used.
        """
        schema = df.schema
        numeric_cols: list[str] = []
        cat_cols: list[str] = []
        for f in self.factors:
            dtype = schema[f]
            if dtype.is_numeric():
                numeric_cols.append(f)
            else:
                cat_cols.append(f)

        parts: list[np.ndarray] = []
        if numeric_cols:
            parts.append(df.select(numeric_cols).to_numpy().astype(np.float64))
        if cat_cols:
            dummies = df.select(cat_cols).to_dummies(columns=cat_cols)
            parts.append(dummies.to_numpy().astype(np.float64))

        if parts:
            design = np.hstack(parts)
        else:  # pragma: no cover - guarded by __init__
            design = np.empty((df.height, 0), dtype=np.float64)
        return design, numeric_cols + cat_cols

    def _residual_for_group(self, df: pl.DataFrame) -> pl.Series:
        """Compute per-row residuals for one date's cross-section."""
        n = df.height
        y_full = df.get_column(self.target).to_numpy().astype(np.float64)
        design, _ = self._build_design(df)

        if self.add_intercept:
            design = np.hstack([np.ones((n, 1), dtype=np.float64), design])

        residual = np.full(n, np.nan, dtype=np.float64)

        # rows usable for the fit: finite y and finite design row
        valid = np.isfinite(y_full)
        if design.shape[1] > 0:
            valid &= np.isfinite(design).all(axis=1)

        n_valid = int(valid.sum())
        # Need at least as many observations as parameters; otherwise leave nulls.
        if n_valid == 0 or (design.shape[1] > 0 and n_valid < design.shape[1]):
            # Not enough to fit; if intercept-only with >=1 obs, just de-mean.
            if design.shape[1] == 0 and n_valid >= 1:
                pass  # handled below by the lstsq path (design has the ones col)
            else:
                return pl.Series(self.target, residual)

        X = design[valid]
        y = y_full[valid]
        if X.shape[1] == 0:
            # No regressors at all (e.g. add_intercept=False, factors all-null):
            # residual equals the original value.
            residual[valid] = y
            return pl.Series(self.target, residual)

        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        residual[valid] = y - X @ coef
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
