"""sklearn-shaped base classes for panel transformers and estimators.

These abstract base classes define the *contract* that every concrete
transformer / estimator in PanelKit must honour. The contract is the moat:

* **fit learns parameters on the given (training) data only.** A transformer's
  ``fit`` may compute statistics (means, quantiles, encoders, scalers, fitted
  coefficients) but only from the rows it is handed. It must never peek at data
  it will later transform.
* **transform applies learned parameters.** Re-fitting inside ``transform`` is
  forbidden; that would let test-fold statistics leak into the features.
* **forward-looking computation must be walk-forward.** Any feature that depends
  on a window of observations must be expressed per entity and looking
  *backwards only*, i.e. via ``expr.over(panel.entity_col)`` with non-negative
  shifts / trailing rolling windows. If a transform cannot express its
  computation this way it must declare ``leakage_safe = False`` so callers can
  refuse it inside cross-validation.

Two class-level booleans make the guarantees explicit and machine-checkable:

``panel_safe``
    The transform respects entity boundaries (never mixes rows across
    entities). Required for any operation involving shifts/rolling/cumulative
    statistics.
``leakage_safe``
    The transform never uses information from the future relative to each row
    (no look-ahead, no fitting on transform-time data). Required to be used
    inside leak-safe cross-validation.

Concrete subclasses **must** set both attributes; the metaclass-free check in
:meth:`PanelTransformer.__init_subclass__` enforces this at class-definition
time with a clear error.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, TypeVar

import polars as pl

from polars_features.core.panel_frame import PanelFrame, as_panel

if TYPE_CHECKING:
    # `Self` is 3.11+. We import it only for type-checkers; at runtime on 3.10
    # we fall back to a TypeVar bound to the base class (see below).
    from typing import Self

__all__ = ["PanelTransformer", "PanelEstimator"]

# Runtime-safe stand-in for `typing.Self` on Python 3.10. Methods annotated
# with the string "Self" resolve to this under `from __future__ import
# annotations`; type-checkers use the real `Self` imported above.
_TTransformer = TypeVar("_TTransformer", bound="PanelTransformer")


class _NotFitted:
    """Sentinel marking an estimator that has not been fitted yet."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<not fitted>"


_NOT_FITTED = _NotFitted()


class PanelTransformer(abc.ABC):
    """Abstract base class for leak-aware panel transformers.

    Subclasses implement :meth:`_fit` and :meth:`_transform` and **must** set
    the class attributes :attr:`panel_safe` and :attr:`leakage_safe`.

    The public :meth:`fit` / :meth:`transform` / :meth:`fit_transform` wrap the
    subclass hooks with contract checks (input typing, fitted-state tracking,
    and an optional leakage assertion).

    Attributes
    ----------
    panel_safe : bool
        Whether the transform respects entity boundaries. Class-level; must be
        declared by every concrete subclass.
    leakage_safe : bool
        Whether the transform is free of look-ahead and fits only on the data
        passed to :meth:`fit`. Class-level; must be declared by every concrete
        subclass.

    Notes
    -----
    **The guarantee.** If ``leakage_safe is True``, then for any walk-forward or
    purged split, ``transformer.fit(train).transform(test)`` produces test-fold
    features that depend only on (a) parameters learned from ``train`` and (b)
    each test row's own past within its entity. This is what makes the
    transformer composable inside :class:`~polars_features.core.pipeline.Pipeline`
    and the cross-validators without introducing leakage.
    """

    # Subclasses MUST override both. The sentinel `None` triggers the
    # __init_subclass__ guard so a forgotten declaration fails loudly.
    panel_safe: bool = None  # type: ignore[assignment]
    leakage_safe: bool = None  # type: ignore[assignment]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Allow intermediate abstract subclasses to defer declaration. Note:
        # `__abstractmethods__` is not yet populated by ABCMeta when
        # __init_subclass__ runs, so we detect abstractness by scanning for any
        # method still flagged `__isabstractmethod__` on this class' MRO view.
        is_abstract = any(
            getattr(getattr(cls, name, None), "__isabstractmethod__", False)
            for name in dir(cls)
        )
        if is_abstract:
            return
        for attr in ("panel_safe", "leakage_safe"):
            value = getattr(cls, attr, None)
            if not isinstance(value, bool):
                raise TypeError(
                    f"{cls.__name__} must declare a class-level boolean "
                    f"`{attr}` (got {value!r}). Every concrete PanelTransformer "
                    "must state whether it is panel-safe (respects entity "
                    "boundaries) and leakage-safe (no look-ahead / no fitting "
                    "on transform-time data). Example:\n\n"
                    f"    class {cls.__name__}(PanelTransformer):\n"
                    "        panel_safe = True\n"
                    "        leakage_safe = True\n"
                )

    def __init__(self, *, entity: str | None = None, time: str | None = None) -> None:
        # `_fitted` flips to True after a successful `fit`. Subclasses store
        # learned parameters as instance attributes (sklearn convention:
        # trailing-underscore names) during `_fit`.
        self._fitted: bool = False
        # The panel the transform was last fitted on. Recorded in `fit` so the
        # leakage-check flow can pass a meaningful ``train`` into
        # :meth:`_check_leakage` when a fitted transform is later applied across
        # a train/test boundary (e.g. by :class:`Pipeline`).
        self._fit_panel: PanelFrame | None = None
        # Optional default panel keys. When set, the public methods accept a
        # bare ``pl.DataFrame``/``pl.LazyFrame`` and wrap it into a PanelFrame
        # using these keys, so users can configure the keys once on the
        # transformer instead of at every call site.
        self._entity: str | None = entity
        self._time: str | None = time

    # ------------------------------------------------------------------ #
    # Hooks for subclasses
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _fit(self, panel: PanelFrame) -> None:
        """Learn parameters from ``panel`` (training data only).

        Store learned state on ``self`` (use trailing-underscore attribute names
        by convention). Must not return anything.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _transform(self, panel: PanelFrame) -> PanelFrame:
        """Apply learned parameters to ``panel`` and return a new PanelFrame.

        Must not mutate ``self`` (no re-fitting) and must stay lazy where
        possible.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Contract enforcement
    # ------------------------------------------------------------------ #
    def _as_panel(
        self,
        X: object,
        *,
        method: str,
        entity: str | None = None,
        time: str | None = None,
    ) -> PanelFrame:
        """Normalise ``X`` into a :class:`PanelFrame`.

        ``X`` may be a :class:`PanelFrame` (returned as-is, the "old way") or a
        bare :class:`polars.DataFrame` / :class:`polars.LazyFrame`, which is
        wrapped using the supplied ``entity`` / ``time`` (falling back to the
        instance defaults set on the constructor). This is what makes
        ``PanelFrame`` *optional sugar* rather than a mandatory wrapper.

        Parameters
        ----------
        X : PanelFrame | polars.DataFrame | polars.LazyFrame
            The input frame.
        method : str
            Calling method name, used only for error messages.
        entity, time : str, optional
            Per-call key overrides. They are only meaningful for bare frames.

        Raises
        ------
        ValueError
            If ``X`` is already a :class:`PanelFrame` *and* conflicting
            ``entity`` / ``time`` keys are passed, or if a bare frame is given
            with no resolvable keys.
        TypeError
            If ``X`` is neither a :class:`PanelFrame` nor a polars frame.
        """
        if isinstance(X, PanelFrame):
            # The old, mandatory-wrapper path. Honour the PanelFrame's own keys.
            # If the caller *also* passed keys, they must match (no silent
            # override of a validated PanelFrame's contract).
            if entity is not None and entity != X.entity_col:
                raise ValueError(
                    f"{method}: `entity={entity!r}` conflicts with the supplied "
                    f"PanelFrame's entity column {X.entity_col!r}. Omit `entity=` "
                    "when passing a PanelFrame (its keys win), or pass a bare "
                    "DataFrame/LazyFrame instead."
                )
            if time is not None and time != X.time_col:
                raise ValueError(
                    f"{method}: `time={time!r}` conflicts with the supplied "
                    f"PanelFrame's time column {X.time_col!r}. Omit `time=` when "
                    "passing a PanelFrame (its keys win), or pass a bare "
                    "DataFrame/LazyFrame instead."
                )
            return X

        if not isinstance(X, (pl.DataFrame, pl.LazyFrame)):
            raise TypeError(
                f"{method} expects a PanelFrame, polars.DataFrame, or "
                f"polars.LazyFrame, got {type(X).__name__!r}. Either wrap your "
                "data, e.g. `PanelFrame(df, entity='id', time='date')`, or pass a "
                "polars frame with `entity=`/`time=` keys."
            )

        ent = entity if entity is not None else self._entity
        tme = time if time is not None else self._time
        try:
            return as_panel(X, entity=ent, time=tme)
        except ValueError as exc:
            # `as_panel` falls back to col0/col1 when keys are omitted; it only
            # raises here when the frame has fewer than two columns. Re-raise
            # with guidance that points at the estimator entry point.
            raise ValueError(
                f"{method}: could not build a PanelFrame from the given frame. "
                "Pass `entity=`/`time=` (or set them on the transformer's "
                "constructor), or wrap the data in a PanelFrame explicitly. "
                f"Original error: {exc}"
            ) from exc

    @staticmethod
    def _check_input(panel: object, *, method: str) -> PanelFrame:
        """Validate that ``panel`` is a :class:`PanelFrame` (internal use).

        Retained for internal step-to-step threading inside
        :class:`~polars_features.core.pipeline.Pipeline`, where the value passed
        between steps is always already a :class:`PanelFrame`.
        """
        if not isinstance(panel, PanelFrame):
            raise TypeError(
                f"{method} expects a PanelFrame, got {type(panel).__name__!r}. "
                "Wrap your data first, e.g. "
                "`PanelFrame(df, entity='id', time='date')`."
            )
        return panel

    def _check_leakage(self, train: PanelFrame, test: PanelFrame | None = None) -> None:
        """Contract hook called before applying a fitted transform.

        The default implementation enforces the *declared* guarantee: if the
        transform is marked ``leakage_safe = False`` it raises when asked to
        operate across a train/test boundary (i.e. inside cross-validation),
        forcing the caller to make an explicit, leak-aware choice.

        Subclasses with richer invariants (e.g. "test times must all be > max
        train time") should override this and call ``super()._check_leakage``.

        Parameters
        ----------
        train : PanelFrame
            The data the transform was fitted on.
        test : PanelFrame, optional
            The data about to be transformed, if different from ``train``.

        Raises
        ------
        RuntimeError
            If the transform is not leakage-safe but is being applied across a
            train/test boundary.
        """
        if test is None:
            return
        if not self.leakage_safe:
            raise RuntimeError(
                f"{type(self).__name__} declares `leakage_safe = False` but is "
                "being applied across a train/test boundary. This transform may "
                "leak future information into test-fold features. Either make it "
                "leakage-safe (express features walk-forward via "
                "`.over(entity_col)`), or refit it per fold and acknowledge the "
                "risk explicitly."
            )

    def _check_fitted(self, method: str) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"this {type(self).__name__} instance is not fitted yet; call "
                f"`fit` before `{method}`."
            )

    # ------------------------------------------------------------------ #
    # Public API (sklearn-shaped)
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> Self:
        """Learn parameters from ``X`` (training data only).

        Parameters
        ----------
        X : PanelFrame | polars.DataFrame | polars.LazyFrame
            Training data. A :class:`PanelFrame` is used as-is (its keys win); a
            bare polars frame is wrapped on the fly using ``entity`` / ``time``
            (or the keys configured on the constructor). Parameters are learned
            **only** from these rows.
        entity, time : str, optional
            Panel keys to use when ``X`` is a bare polars frame. Ignored (and
            forbidden if conflicting) when ``X`` is already a :class:`PanelFrame`.

        Returns
        -------
        Self
            The fitted transformer, for chaining.
        """
        panel = self._as_panel(X, method="fit", entity=entity, time=time)
        self._fit(panel)
        self._fit_panel = panel
        self._fitted = True
        return self

    def transform(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> PanelFrame:
        """Apply the learned transform to ``X``.

        Parameters
        ----------
        X : PanelFrame | polars.DataFrame | polars.LazyFrame
            Data to transform (train or test). No parameters are learned here.
            Accepts a :class:`PanelFrame` or a bare polars frame (see
            :meth:`fit`).
        entity, time : str, optional
            Panel keys for a bare polars frame (see :meth:`fit`).

        Returns
        -------
        PanelFrame
            The transformed panel (lazy where possible). Call ``.collect()`` /
            ``.to_native()`` on it to get a plain polars frame back.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        panel = self._as_panel(X, method="transform", entity=entity, time=time)
        self._check_fitted("transform")
        return self._transform(panel)

    def fit_transform(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> PanelFrame:
        """Fit on ``X`` and immediately transform it.

        Equivalent to ``self.fit(X).transform(X)``. Safe to use on training
        data; do **not** use it on a test fold, as that would fit on test data.

        Parameters
        ----------
        X : PanelFrame | polars.DataFrame | polars.LazyFrame
            See :meth:`fit`.
        entity, time : str, optional
            Panel keys for a bare polars frame (see :meth:`fit`).

        Returns
        -------
        PanelFrame
        """
        panel = self._as_panel(X, method="fit_transform", entity=entity, time=time)
        return self.fit(panel).transform(panel)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed successfully."""
        return self._fitted

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if getattr(self, "_fitted", False) else "not fitted"
        return (
            f"{type(self).__name__}(panel_safe={self.panel_safe}, "
            f"leakage_safe={self.leakage_safe}, {state})"
        )


class PanelEstimator(PanelTransformer):
    """Abstract base class for panel estimators (transformers that predict).

    Extends :class:`PanelTransformer` with :meth:`predict`. The same leakage
    contract applies: :meth:`fit` learns from training rows only, and
    :meth:`predict` must produce predictions for each row using only its own
    past (within its entity) and the fitted parameters.

    By default an estimator is a transformer whose :meth:`_transform` appends
    its predictions; subclasses may override :meth:`_transform` if they emit
    additional engineered columns, but the canonical entry point for downstream
    consumers is :meth:`predict`.
    """

    @abc.abstractmethod
    def _predict(self, panel: PanelFrame) -> PanelFrame:
        """Produce predictions for ``panel`` and return them as a PanelFrame.

        The returned PanelFrame should retain the ``(entity, time)`` keys and
        carry one or more prediction columns.
        """
        raise NotImplementedError

    def predict(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> PanelFrame:
        """Predict on ``X`` using the fitted parameters.

        Parameters
        ----------
        X : PanelFrame | polars.DataFrame | polars.LazyFrame
            Data to predict on (train or test). Accepts a :class:`PanelFrame` or
            a bare polars frame (see :meth:`fit`).
        entity, time : str, optional
            Panel keys for a bare polars frame (see :meth:`fit`).

        Returns
        -------
        PanelFrame
            Predictions keyed by ``(entity, time)``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        panel = self._as_panel(X, method="predict", entity=entity, time=time)
        self._check_fitted("predict")
        return self._predict(panel)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        """Default: transforming an estimator means predicting on it."""
        return self._predict(panel)
