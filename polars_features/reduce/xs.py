"""Per-date cross-sectional dimensionality reduction.

:class:`CrossSectionalPCA` fits a **fresh** PCA on each date's contemporaneous
cross-section (all entities observed on that date) and keeps that date's scores.
Because every fit uses only same-date data it is *leak-safe in time by
construction* -- it never touches the future -- but it deliberately mixes
information across entities within a date, exactly like the ``.xs`` operator
family. Hence ``panel_safe = False`` (entities are combined) and
``leakage_safe = True`` (no look-ahead).

Sign-fixing is applied per date so scores are comparable across dates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from polars_features.core.panel_frame import PanelFrame
from polars_features.core.protocol import PanelTransformer
from polars_features.reduce._base import _sign_of_max_abs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = ["CrossSectionalPCA"]


class CrossSectionalPCA(PanelTransformer):
    """Per-date cross-sectional PCA (xs-family; leak-safe in time).

    For each date, the entity-by-feature cross-section is (optionally)
    standardised, reduced with :class:`sklearn.decomposition.PCA`, sign-fixed and
    the resulting scores are attached to that date's rows. Dates with fewer than
    two entities are left null (a cross-section of one has no PCA).

    Because the reduction is contemporaneous, no train-fold parameters are
    learned: :meth:`fit` only records the feature columns, and the per-date fits
    happen at :meth:`transform` time on same-date data only (the documented
    xs-family exception to the "no fit at transform" rule).

    Parameters
    ----------
    n_components : int, default 2
        Number of cross-sectional components to emit (``cspc_1 .. cspc_k``). Per
        date the effective count is capped at ``min(n_entities, n_features)``;
        any shortfall is filled with nulls.
    columns : sequence of str, optional
        Feature columns to reduce. ``None`` uses every numeric feature column.
    standardize : bool, default True
        Standardise each date's cross-section before reducing.
    sign_fix : bool, default True
        Deterministically fix each component's sign per date.
    prefix : str, default "cspc"
        Prefix for the emitted component columns.
    keep_features : bool, default False
        If False (default) the output is the ``(entity, time)`` keys plus the
        component columns; if True they are appended to the panel.
    random_state : int, default 42
        Seed forwarded to :class:`sklearn.decomposition.PCA`.
    entity, time : str, optional
        Default panel keys for bare polars frames.

    Attributes
    ----------
    panel_safe : bool
        Always ``False`` -- entities are mixed within each date on purpose.
    leakage_safe : bool
        Always ``True`` -- each date uses only its own contemporaneous rows.
    feature_names_in_ : list of str
        The resolved input feature columns.
    component_names_ : list of str
        The emitted component column names.
    """

    panel_safe = False
    leakage_safe = True

    def __init__(
        self,
        *,
        n_components: int = 2,
        columns: Sequence[str] | None = None,
        standardize: bool = True,
        sign_fix: bool = True,
        prefix: str = "cspc",
        keep_features: bool = False,
        random_state: int = 42,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError(
                f"`n_components` must be a positive integer, got {n_components!r}."
            )
        self.n_components = n_components
        self.columns = list(columns) if columns is not None else None
        self.standardize = bool(standardize)
        self.sign_fix = bool(sign_fix)
        self.prefix = prefix
        self.keep_features = bool(keep_features)
        self.random_state = random_state
        self.feature_names_in_: list[str] = []
        self.component_names_: list[str] = []

    def _resolve_columns(self, panel: PanelFrame) -> list[str]:
        schema = panel.schema
        cols = (
            list(self.columns) if self.columns is not None else list(panel.feature_cols)
        )
        if self.columns is not None:
            missing = [c for c in cols if c not in panel]
            if missing:
                raise ValueError(
                    f"{type(self).__name__}: column(s) {missing} not found in "
                    f"panel. Available columns: {panel.columns}."
                )
        if not cols:
            raise ValueError(
                f"{type(self).__name__}: no feature columns to reduce "
                f"(columns={panel.columns}). Pass `columns=` explicitly."
            )
        non_numeric = [c for c in cols if not schema[c].is_numeric()]
        if non_numeric:
            raise ValueError(
                f"{type(self).__name__}: column(s) {non_numeric} are not numeric "
                "and cannot be reduced."
            )
        return cols

    def _fit(self, panel: PanelFrame) -> None:
        self.feature_names_in_ = self._resolve_columns(panel)
        self.component_names_ = [
            f"{self.prefix}_{i}" for i in range(1, self.n_components + 1)
        ]

    def _reduce_one(self, X: NDArray[Any]) -> NDArray[Any]:
        """Reduce a single date's ``(n_entities, n_features)`` matrix.

        Returns an ``(n_entities, n_components)`` score matrix padded with NaN
        when the date supports fewer components than requested.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        n_ent, n_feat = X.shape
        out = np.full((n_ent, self.n_components), np.nan)
        if n_ent < 2:
            return out
        # Same-date mean-fill of missing cells, then optional standardisation.
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        Xf = np.where(np.isnan(X), col_mean, X)
        if self.standardize:
            Xf = StandardScaler().fit_transform(Xf)
        k = min(self.n_components, n_ent, n_feat)
        if k < 1:
            return out
        pca = PCA(n_components=k, random_state=self.random_state)
        scores = np.asarray(pca.fit_transform(Xf))
        if self.sign_fix:
            comps = np.asarray(pca.components_)
            sign = np.array(
                [_sign_of_max_abs(comps[i]) for i in range(k)], dtype=np.float64
            )
            scores = scores * sign
        out[:, :k] = scores
        return out

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        missing = [c for c in self.feature_names_in_ if c not in panel]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.transform: feature column(s) {missing} not "
                f"found in panel. Available columns: {panel.columns}."
            )
        time_col = panel.time_col
        full = panel.collect()
        n_rows = full.height
        out_mat = np.full((n_rows, self.n_components), np.nan)

        # One conversion for the whole feature block, then per-date views by row
        # index. The previous loop paid a polars `select(...).to_numpy()` (and a
        # per-group DataFrame materialisation) for every date, which was ~30% of
        # `transform` on a 2500-date panel. The arithmetic per date is unchanged.
        X_all = full.select(self.feature_names_in_).to_numpy().astype(np.float64)
        groups = (
            full.lazy()
            .with_row_index("__row")
            .group_by(pl.col(time_col))
            .agg(pl.col("__row"))
            .collect()
            .get_column("__row")
        )
        for rows_series in groups:
            rows = rows_series.to_numpy()
            out_mat[rows, :] = self._reduce_one(X_all[rows])

        comp_cols = [
            pl.Series(name=name, values=out_mat[:, i])
            for i, name in enumerate(self.component_names_)
        ]
        base = full
        keys = [panel.entity_col, time_col]
        if self.keep_features:
            out = base.with_columns(comp_cols)
        else:
            out = base.select(keys).with_columns(comp_cols)
        return PanelFrame(out, entity=panel.entity_col, time=time_col, validate=False)
