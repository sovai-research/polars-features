"""Concrete, leak-safe panel estimators that terminate a :class:`Pipeline`.

This module provides the estimator layer of PanelKit: thin, panel-aware wrappers
around ordinary sklearn-style estimators that honour the
:class:`~polars_features.core.protocol.PanelEstimator` contract. They

* learn parameters in :meth:`~PanelEstimator.fit` from the rows they are handed
  (the training fold) and **only** those rows, and
* produce predictions in :meth:`~PanelEstimator.predict` that are re-aligned to
  the panel's ``(entity, time)`` keys.

Because a wrapper fits solely on the data passed to ``fit`` and never re-fits at
transform/predict time, it is both ``panel_safe`` and ``leakage_safe``: dropping
one at the end of a :class:`~polars_features.core.pipeline.Pipeline` keeps the
whole chain leak-safe under any purged / walk-forward split.

Two families live here:

* :class:`PanelSklearnRegressor` / :class:`PanelSklearnClassifier` wrap **any**
  estimator exposing the sklearn ``fit`` / ``predict`` API. With no estimator
  supplied they default to sklearn's dependency-free
  ``HistGradientBoosting{Regressor,Classifier}``, so the primary path needs
  nothing beyond scikit-learn.
* :class:`PanelLGBMRegressor` / :class:`PanelLGBMClassifier` are convenience
  constructors that lazily import LightGBM and, if it is not installed, fall back
  to the sklearn gradient-boosting default with a clear warning.

Notes
-----
The panel carries features *and* the target in its columns; the estimator is
told which column is the target (``target=``) and treats every other numeric
feature column as a predictor unless an explicit ``features=`` list is given.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import polars as pl

from polars_features.core.panel_frame import PanelFrame
from polars_features.core.protocol import PanelEstimator

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "PanelSklearnRegressor",
    "PanelSklearnClassifier",
    "PanelLGBMRegressor",
    "PanelLGBMClassifier",
]


def _as_feature_list(features: str | Sequence[str] | None) -> list[str] | None:
    """Normalise a ``features`` argument to ``None`` or a list of names."""
    if features is None:
        return None
    if isinstance(features, str):
        return [features]
    cols = list(features)
    if not cols:
        raise ValueError(
            "`features` was an empty sequence; pass None to use every feature "
            "column (all columns that are neither entity, time, nor target)."
        )
    return cols


def _clone_estimator(estimator: Any) -> Any:
    """Return an unfitted clone of ``estimator`` (falls back to the original).

    A fresh clone per :meth:`fit` prevents state from a previous fold leaking
    into the next when the same wrapper instance is re-used inside
    cross-validation.
    """
    try:
        from sklearn.base import clone

        return clone(estimator)
    except Exception:  # pragma: no cover - non-sklearn estimator without get_params
        return estimator


def _is_classifier(estimator: Any) -> bool:
    """Best-effort check whether ``estimator`` is a classifier."""
    try:
        from sklearn.base import is_classifier

        if is_classifier(estimator):
            return True
    except Exception:  # pragma: no cover - defensive
        pass
    return getattr(estimator, "_estimator_type", None) == "classifier"


class _PanelSupervised(PanelEstimator):
    """Shared machinery for the panel regressor/classifier wrappers.

    Concrete subclasses only differ in their default estimator and in whether
    they expose :meth:`predict_proba`. The flattening of a :class:`PanelFrame`
    into ``(X, y)`` design matrices, the leak-safe fit-on-train contract, and the
    re-alignment of predictions to ``(entity, time)`` all live here.

    Parameters
    ----------
    estimator : object, optional
        Any estimator implementing the sklearn ``fit(X, y[, sample_weight])`` /
        ``predict(X)`` API. If ``None`` the subclass' dependency-free default is
        used.
    target : str, optional
        Name of the target column in the panel. If omitted, the **last** feature
        column (in schema order) is treated as the target.
    features : str or sequence of str, optional
        Predictor columns. ``None`` (default) uses every feature column that is
        neither the target nor the sample-weight column.
    sample_weight : str, optional
        Name of a column holding per-row sample weights, forwarded to the
        underlying estimator's ``fit`` when supported.
    prediction_col : str, default="prediction"
        Name of the emitted prediction column.
    entity, time : str, optional
        Default panel keys used when a bare polars frame is passed instead of a
        :class:`PanelFrame`.

    Attributes
    ----------
    panel_safe : bool
        Always ``True`` -- rows are never mixed across entities during fitting.
    leakage_safe : bool
        Always ``True`` -- parameters come solely from the ``fit`` panel.
    estimator_ : object
        The fitted clone of ``estimator``. Available after :meth:`fit`.
    features_ : list of str
        The resolved predictor columns, in the order fed to the estimator.
    target_ : str
        The resolved target column name.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        estimator: Any | None = None,
        *,
        target: str | None = None,
        features: str | Sequence[str] | None = None,
        sample_weight: str | None = None,
        prediction_col: str = "prediction",
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self.estimator = (
            estimator if estimator is not None else self._default_estimator()
        )
        self.target = target
        self.features = _as_feature_list(features)
        self.sample_weight = sample_weight
        self.prediction_col = prediction_col
        # learned state
        self.estimator_: Any | None = None
        self.features_: list[str] = []
        self.target_: str | None = None
        self.classes_: NDArray[Any] | None = None

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    def _default_estimator(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Column resolution
    # ------------------------------------------------------------------ #
    def _resolve_target(self, panel: PanelFrame) -> str:
        feat_cols = panel.feature_cols
        if self.target is not None:
            if self.target not in panel:
                raise ValueError(
                    f"{type(self).__name__}: target column {self.target!r} not "
                    f"found in panel. Available columns: {panel.columns}."
                )
            return self.target
        if not feat_cols:
            raise ValueError(
                f"{type(self).__name__}: panel has no feature columns to use as a "
                f"target (columns={panel.columns}). Pass `target=` explicitly."
            )
        return feat_cols[-1]

    def _resolve_features(self, panel: PanelFrame, target: str) -> list[str]:
        schema = panel.schema
        if self.features is not None:
            missing = [c for c in self.features if c not in panel]
            if missing:
                raise ValueError(
                    f"{type(self).__name__}: feature column(s) {missing} not found "
                    f"in panel. Available columns: {panel.columns}."
                )
            feats = list(self.features)
        else:
            exclude = {target}
            if self.sample_weight is not None:
                exclude.add(self.sample_weight)
            feats = [c for c in panel.feature_cols if c not in exclude]
        if not feats:
            raise ValueError(
                f"{type(self).__name__}: no predictor columns left after removing "
                f"the target {target!r}. Pass `features=` explicitly."
            )
        non_numeric = [c for c in feats if not schema[c].is_numeric()]
        if non_numeric:
            raise ValueError(
                f"{type(self).__name__}: feature column(s) {non_numeric} are not "
                "numeric and cannot be fed to the estimator. Encode them first or "
                "pass a numeric `features=` list."
            )
        return feats

    # ------------------------------------------------------------------ #
    # PanelEstimator hooks
    # ------------------------------------------------------------------ #
    def _fit(self, panel: PanelFrame) -> None:
        target = self._resolve_target(panel)
        feats = self._resolve_features(panel, target)
        self.target_ = target
        self.features_ = feats

        select_cols = [*feats, target]
        if self.sample_weight is not None:
            if self.sample_weight not in panel:
                raise ValueError(
                    f"{type(self).__name__}: sample_weight column "
                    f"{self.sample_weight!r} not found in panel. "
                    f"Available columns: {panel.columns}."
                )
            select_cols.append(self.sample_weight)

        frame = panel.lazy().select(select_cols).collect()
        X = frame.select(feats).to_numpy()
        y = frame.get_column(target).to_numpy()

        fit_kwargs: dict[str, Any] = {}
        if self.sample_weight is not None:
            fit_kwargs["sample_weight"] = frame.get_column(
                self.sample_weight
            ).to_numpy()

        est = _clone_estimator(self.estimator)
        try:
            est.fit(X, y, **fit_kwargs)
        except TypeError:
            if fit_kwargs:
                # Estimator does not accept sample_weight; retry without it.
                warnings.warn(
                    f"{type(est).__name__}.fit does not accept `sample_weight`; "
                    "fitting without sample weights.",
                    stacklevel=2,
                )
                est.fit(X, y)
            else:  # pragma: no cover - re-raise genuine signature errors
                raise
        self.estimator_ = est
        self.classes_ = getattr(est, "classes_", None)

    def _predict(self, panel: PanelFrame) -> PanelFrame:
        if self.estimator_ is None:  # pragma: no cover - guarded by _check_fitted
            raise RuntimeError(f"{type(self).__name__} is not fitted.")
        missing = [c for c in self.features_ if c not in panel]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.predict: feature column(s) {missing} not "
                f"found in panel. Available columns: {panel.columns}."
            )
        keys = [panel.entity_col, panel.time_col]
        frame = panel.lazy().select([*keys, *self.features_]).collect()
        X = frame.select(self.features_).to_numpy()
        preds = self.estimator_.predict(X)
        out = frame.select(keys).with_columns(
            pl.Series(name=self.prediction_col, values=preds)
        )
        return PanelFrame(
            out, entity=panel.entity_col, time=panel.time_col, validate=False
        )

    def predict_proba(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> PanelFrame:
        """Predict class probabilities, aligned to ``(entity, time)``.

        Only available on classifier wrappers whose underlying estimator exposes
        ``predict_proba``. One probability column per class is emitted, named
        ``f"{prediction_col}_proba_{class}"``.

        Parameters
        ----------
        X : PanelFrame | polars.DataFrame | polars.LazyFrame
            Data to score.
        entity, time : str, optional
            Panel keys for a bare polars frame.

        Returns
        -------
        PanelFrame
            Probabilities keyed by ``(entity, time)``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        AttributeError
            If the underlying estimator has no ``predict_proba``.
        """
        panel = self._as_panel(X, method="predict_proba", entity=entity, time=time)
        self._check_fitted("predict_proba")
        if self.estimator_ is None or not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError(
                f"{type(self).__name__}: underlying estimator "
                f"{type(self.estimator_).__name__} has no `predict_proba`."
            )
        keys = [panel.entity_col, panel.time_col]
        frame = panel.lazy().select([*keys, *self.features_]).collect()
        proba = self.estimator_.predict_proba(frame.select(self.features_).to_numpy())
        classes = (
            self.classes_ if self.classes_ is not None else list(range(proba.shape[1]))
        )
        cols = [
            pl.Series(name=f"{self.prediction_col}_proba_{cls}", values=proba[:, i])
            for i, cls in enumerate(classes)
        ]
        out = frame.select(keys).with_columns(cols)
        return PanelFrame(
            out, entity=panel.entity_col, time=panel.time_col, validate=False
        )


class PanelSklearnRegressor(_PanelSupervised):
    """Leak-safe panel wrapper around any sklearn-style **regressor**.

    Flattens the panel's feature columns (dropping the entity/time keys and the
    target) on the training fold, fits the wrapped estimator, and re-aligns the
    continuous predictions back to ``(entity, time)``.

    Parameters
    ----------
    estimator : object, optional
        Any regressor with the sklearn ``fit`` / ``predict`` API. Defaults to
        :class:`sklearn.ensemble.HistGradientBoostingRegressor` (no extra
        dependency).
    target, features, sample_weight, prediction_col, entity, time
        See :class:`_PanelSupervised`.

    Examples
    --------
    >>> import polars as pl
    >>> from sklearn.linear_model import LinearRegression
    >>> from polars_features.core import PanelFrame
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["a", "a", "b", "b"],
    ...         "t": [1, 2, 1, 2],
    ...         "x": [0.0, 1.0, 2.0, 3.0],
    ...         "y": [0.0, 2.0, 4.0, 6.0],
    ...     }
    ... )
    >>> panel = PanelFrame(df, entity="id", time="t")
    >>> model = PanelSklearnRegressor(LinearRegression(), target="y").fit(panel)
    >>> model.predict(panel).collect().shape
    (4, 3)
    """

    def _default_estimator(self) -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor()


class PanelSklearnClassifier(_PanelSupervised):
    """Leak-safe panel wrapper around any sklearn-style **classifier**.

    Behaves like :class:`PanelSklearnRegressor` but predicts discrete labels and
    additionally exposes :meth:`predict_proba` when the underlying estimator
    supports it.

    Parameters
    ----------
    estimator : object, optional
        Any classifier with the sklearn ``fit`` / ``predict`` API. Defaults to
        :class:`sklearn.ensemble.HistGradientBoostingClassifier`.
    target, features, sample_weight, prediction_col, entity, time
        See :class:`_PanelSupervised`.
    """

    def _default_estimator(self) -> Any:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier()


def _make_lgbm(kind: str, params: dict[str, Any]) -> Any:
    """Build a LightGBM estimator, falling back to sklearn if it is unavailable.

    Parameters
    ----------
    kind : {"regressor", "classifier"}
        Which estimator family to build.
    params : dict
        Hyper-parameters forwarded to the LightGBM (or fallback) constructor.

    Returns
    -------
    object
        An unfitted estimator. LightGBM's sklearn-API estimator when the library
        is importable; otherwise the sklearn ``HistGradientBoosting*`` default
        (with a warning), so the wrapper still works dependency-free.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        warnings.warn(
            "lightgbm is not installed; falling back to sklearn "
            f"HistGradientBoosting{'Regressor' if kind == 'regressor' else 'Classifier'}"
            ". Install lightgbm (`pip install lightgbm`) to use the gradient "
            "booster, or use PanelSklearn{Regressor,Classifier} directly.",
            stacklevel=3,
        )
        if kind == "regressor":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(**params)
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(**params)
    if kind == "regressor":
        return lgb.LGBMRegressor(**params)
    return lgb.LGBMClassifier(**params)


class PanelLGBMRegressor(PanelSklearnRegressor):
    """Convenience :class:`PanelSklearnRegressor` backed by LightGBM.

    Lazily imports LightGBM and wraps :class:`lightgbm.LGBMRegressor`. If
    LightGBM is not installed, it transparently falls back to the sklearn
    gradient-boosting default (with a warning), so code depending on this class
    keeps working without the optional dependency.

    Parameters
    ----------
    target, features, sample_weight, prediction_col, entity, time
        See :class:`_PanelSupervised`.
    **lgbm_params
        Hyper-parameters forwarded to :class:`lightgbm.LGBMRegressor` (or the
        sklearn fallback).
    """

    def __init__(
        self,
        *,
        target: str | None = None,
        features: str | Sequence[str] | None = None,
        sample_weight: str | None = None,
        prediction_col: str = "prediction",
        entity: str | None = None,
        time: str | None = None,
        **lgbm_params: Any,
    ) -> None:
        super().__init__(
            _make_lgbm("regressor", lgbm_params),
            target=target,
            features=features,
            sample_weight=sample_weight,
            prediction_col=prediction_col,
            entity=entity,
            time=time,
        )


class PanelLGBMClassifier(PanelSklearnClassifier):
    """Convenience :class:`PanelSklearnClassifier` backed by LightGBM.

    Lazily imports LightGBM and wraps :class:`lightgbm.LGBMClassifier`, falling
    back to the sklearn gradient-boosting default (with a warning) when LightGBM
    is unavailable.

    Parameters
    ----------
    target, features, sample_weight, prediction_col, entity, time
        See :class:`_PanelSupervised`.
    **lgbm_params
        Hyper-parameters forwarded to :class:`lightgbm.LGBMClassifier` (or the
        sklearn fallback).
    """

    def __init__(
        self,
        *,
        target: str | None = None,
        features: str | Sequence[str] | None = None,
        sample_weight: str | None = None,
        prediction_col: str = "prediction",
        entity: str | None = None,
        time: str | None = None,
        **lgbm_params: Any,
    ) -> None:
        super().__init__(
            _make_lgbm("classifier", lgbm_params),
            target=target,
            features=features,
            sample_weight=sample_weight,
            prediction_col=prediction_col,
            entity=entity,
            time=time,
        )
