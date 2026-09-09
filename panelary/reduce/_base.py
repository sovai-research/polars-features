"""Generic, leak-safe base for panel dimensionality reducers.

This module hosts :class:`_PanelReducer`, the keystone shared by every linear
reducer in :mod:`panelary.reduce`. It mirrors
:class:`panelary.models._PanelSupervised`: a thin, panel-aware wrapper
around an ordinary scikit-learn reducer (anything exposing ``fit`` / ``transform``
and, where relevant, ``components_`` / ``explained_variance_ratio_``) that honours
the :class:`~panelary.core.protocol.PanelTransformer` contract.

The single job of the base is to make dimensionality reduction *leak-safe*:

* the ``StandardScaler`` (when ``standardize=True``), the rotation, **and** the
  automatic ``n_components`` variance threshold are all learned from the rows
  handed to :meth:`~PanelTransformer.fit` (the training fold) and **only** those
  rows -- never the full sample. This is the fix for the upstream SovAI
  ``dimensionality_reduction`` which fits the scaler + rotation + variance
  threshold on the whole panel (train+test+future);
* components are **sign-fixed deterministically** (the largest-magnitude loading
  of each component is forced positive, falling back to the largest-magnitude
  score when the estimator exposes no loadings), so component signs are stable
  across refits and across row shuffles;
* imputation is *not* done here by look-ahead ``bfill`` -- callers compose a
  leak-safe :class:`~panelary.imputation.CafeImputer` upstream instead.

Concrete reducers subclass this and implement only :meth:`_make_reducer` (and,
optionally, :meth:`_cap_components` / :meth:`_validate_matrix`), exactly as the
model wrappers only override ``_default_estimator``.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = ["_PanelReducer"]


def _sign_of_max_abs(vec: NDArray[Any]) -> float:
    """Return the sign (+1.0/-1.0) of the largest-magnitude entry of ``vec``.

    Ties are broken by :func:`numpy.argmax` (first occurrence) and an exact-zero
    winner maps to ``+1.0``, so the result is fully deterministic for a given
    vector and, because it depends only on the *set* of values, invariant to row
    order.
    """
    if vec.size == 0:  # pragma: no cover - defensive
        return 1.0
    idx = int(np.argmax(np.abs(vec)))
    s = float(np.sign(vec[idx]))
    return s if s != 0.0 else 1.0


class _PanelReducer(PanelTransformer):
    """Shared machinery for the leak-safe panel dimensionality reducers.

    Concrete subclasses differ only in :meth:`_make_reducer` (the sklearn
    estimator they build for a resolved component count) and, occasionally, in
    :meth:`_cap_components` (the maximum admissible component count) or
    :meth:`_validate_matrix` (an input check, e.g. non-negativity for NMF).

    Parameters
    ----------
    n_components : int, optional
        Number of components to keep. When ``None`` (default) the count is
        resolved on the **training** matrix from ``explained_variance`` (for
        reducers that expose ``explained_variance_ratio_``) or as
        ``int(n_features * explained_variance)`` otherwise.
    explained_variance : float, default 0.95
        Variance target used to auto-resolve ``n_components`` (ignored when
        ``n_components`` is given).
    columns : sequence of str, optional
        Feature columns to reduce. ``None`` (default) uses every numeric feature
        column (all columns that are neither entity nor time).
    standardize : bool, default True
        If True, fit a :class:`sklearn.preprocessing.StandardScaler` on the
        training rows and apply it before projecting. The scaler statistics come
        solely from the ``fit`` panel.
    sign_fix : bool, default True
        If True, deterministically fix each component's sign (largest-magnitude
        loading forced positive) so signs are stable across refits/shuffles.
    prefix : str, default "pc"
        Prefix for the emitted component columns (``pc_1 .. pc_k``).
    keep_features : bool, default False
        If False (default) the output is the ``(entity, time)`` keys plus the
        component columns. If True, the component columns are appended to the
        original panel.
    random_state : int, default 42
        Seed forwarded to the underlying estimator for reproducibility.
    entity, time : str, optional
        Default panel keys, used when a bare polars frame is passed instead of a
        :class:`PanelFrame`.

    Attributes
    ----------
    panel_safe : bool
        Always ``True`` -- projection is per row, never mixing entities.
    leakage_safe : bool
        Always ``True`` -- scaler, rotation and component count come solely from
        the ``fit`` panel.
    reducer_ : object
        The fitted sklearn reducer. Available after :meth:`fit`.
    scaler_ : object or None
        The fitted ``StandardScaler`` (or ``None`` when ``standardize=False``).
    feature_names_in_ : list of str
        The resolved input feature columns, in the order fed to the estimator.
    n_components_ : int
        The resolved number of components.
    component_names_ : list of str
        The emitted component column names.
    components_ : numpy.ndarray or None
        Sign-fixed loadings ``(n_components, n_features)`` when the estimator
        exposes them, else ``None``.
    explained_variance_ratio_ : numpy.ndarray or None
        Per-component explained-variance ratio when available, else ``None``.
    sign_flip_ : numpy.ndarray
        The ``(n_components,)`` vector of applied component signs.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        *,
        n_components: int | None = None,
        explained_variance: float = 0.95,
        columns: Sequence[str] | None = None,
        standardize: bool = True,
        sign_fix: bool = True,
        prefix: str = "pc",
        keep_features: bool = False,
        random_state: int = 42,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if n_components is not None and (
            not isinstance(n_components, int) or n_components < 1
        ):
            raise ValueError(
                f"{type(self).__name__}: `n_components` must be a positive integer "
                f"or None, got {n_components!r}."
            )
        if not 0.0 < explained_variance <= 1.0:
            raise ValueError(
                f"{type(self).__name__}: `explained_variance` must be in (0, 1], "
                f"got {explained_variance!r}."
            )
        self.n_components = n_components
        self.explained_variance = explained_variance
        self.columns = list(columns) if columns is not None else None
        self.standardize = bool(standardize)
        self.sign_fix = bool(sign_fix)
        self.prefix = prefix
        self.keep_features = bool(keep_features)
        self.random_state = random_state
        # learned state (sklearn trailing-underscore convention)
        self.reducer_: Any | None = None
        self.scaler_: Any | None = None
        self.feature_names_in_: list[str] = []
        self.n_components_: int = 0
        self.component_names_: list[str] = []
        self.components_: NDArray[Any] | None = None
        self.explained_variance_ratio_: NDArray[Any] | None = None
        self.sign_flip_: NDArray[Any] | None = None

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _make_reducer(self, n_components: int) -> Any:
        """Return an unfitted sklearn reducer configured for ``n_components``."""
        raise NotImplementedError

    #: Whether the reducer exposes ``explained_variance_ratio_`` so that the
    #: component count can be auto-resolved from ``explained_variance``.
    _supports_explained_variance: bool = False

    def _cap_components(self, n_samples: int, n_features: int) -> int:
        """Maximum admissible component count for this reducer."""
        return max(1, min(n_samples, n_features))

    def _validate_matrix(self, X: NDArray[Any]) -> None:
        """Optional input-matrix check (e.g. non-negativity). Default: no-op."""

    # ------------------------------------------------------------------ #
    # Column resolution
    # ------------------------------------------------------------------ #
    def _resolve_columns(self, panel: PanelFrame) -> list[str]:
        schema = panel.schema
        if self.columns is not None:
            missing = [c for c in self.columns if c not in panel]
            if missing:
                raise ValueError(
                    f"{type(self).__name__}: column(s) {missing} not found in "
                    f"panel. Available columns: {panel.columns}."
                )
            cols = list(self.columns)
        else:
            cols = list(panel.feature_cols)
        if not cols:
            raise ValueError(
                f"{type(self).__name__}: no feature columns to reduce "
                f"(columns={panel.columns}). Pass `columns=` explicitly."
            )
        non_numeric = [c for c in cols if not schema[c].is_numeric()]
        if non_numeric:
            raise ValueError(
                f"{type(self).__name__}: column(s) {non_numeric} are not numeric "
                "and cannot be reduced. Encode them first or pass a numeric "
                "`columns=` list."
            )
        return cols

    # ------------------------------------------------------------------ #
    # PanelTransformer hooks
    # ------------------------------------------------------------------ #
    def _resolve_n_components(self, X_scaled: NDArray[Any]) -> int:
        n_samples, n_features = X_scaled.shape
        cap = self._cap_components(n_samples, n_features)
        if self.n_components is not None:
            return min(int(self.n_components), cap)
        if self._supports_explained_variance:
            temp = self._make_reducer(cap)
            temp.fit(X_scaled)
            ratio = np.cumsum(np.asarray(temp.explained_variance_ratio_))
            k = int(np.argmax(ratio >= self.explained_variance)) + 1
            return max(1, min(k, cap))
        return max(1, min(int(n_features * self.explained_variance), cap))

    def _fit(self, panel: PanelFrame) -> None:
        self.feature_names_in_ = self._resolve_columns(panel)
        X = (
            panel.lazy()
            .select(self.feature_names_in_)
            .collect()
            .to_numpy()
            .astype(np.float64)
        )
        self._validate_matrix(X)

        if self.standardize:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scaler_ = scaler
        else:
            X_scaled = X
            self.scaler_ = None

        k = self._resolve_n_components(X_scaled)
        reducer = self._make_reducer(k)
        scores = reducer.fit_transform(X_scaled)
        scores = np.asarray(scores)

        comps = getattr(reducer, "components_", None)
        if self.sign_fix:
            if comps is not None:
                sign = np.array(
                    [_sign_of_max_abs(np.asarray(comps)[i]) for i in range(k)],
                    dtype=np.float64,
                )
            else:
                sign = np.array(
                    [_sign_of_max_abs(scores[:, j]) for j in range(k)],
                    dtype=np.float64,
                )
        else:
            sign = np.ones(k, dtype=np.float64)

        self.reducer_ = reducer
        self.n_components_ = k
        self.component_names_ = [f"{self.prefix}_{i}" for i in range(1, k + 1)]
        self.sign_flip_ = sign
        self.components_ = (
            np.asarray(comps) * sign[:, None] if comps is not None else None
        )
        evr = getattr(reducer, "explained_variance_ratio_", None)
        self.explained_variance_ratio_ = None if evr is None else np.asarray(evr)

    def _project(self, X: NDArray[Any]) -> NDArray[Any]:
        """Scale (if fitted) then project ``X`` and apply the stored sign fix."""
        X_scaled = self.scaler_.transform(X) if self.scaler_ is not None else X
        scores = np.asarray(self.reducer_.transform(X_scaled))
        return scores * self.sign_flip_

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        if self.reducer_ is None:  # pragma: no cover - guarded by _check_fitted
            raise RuntimeError(f"{type(self).__name__} is not fitted.")
        missing = [c for c in self.feature_names_in_ if c not in panel]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.transform: feature column(s) {missing} not "
                f"found in panel. Available columns: {panel.columns}."
            )
        full = panel.collect()
        X = full.select(self.feature_names_in_).to_numpy().astype(np.float64)
        scores = self._project(X)
        comp_cols = [
            pl.Series(name=name, values=scores[:, i])
            for i, name in enumerate(self.component_names_)
        ]
        keys = [panel.entity_col, panel.time_col]
        if self.keep_features:
            out = full.with_columns(comp_cols)
        else:
            out = full.select(keys).with_columns(comp_cols)
        return PanelFrame(
            out, entity=panel.entity_col, time=panel.time_col, validate=False
        )
