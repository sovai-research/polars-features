"""Leak-safe pipeline of panel transformers and estimators.

A :class:`Pipeline` chains named steps, each a
:class:`~polars_features.core.protocol.PanelTransformer` (and optionally a final
:class:`~polars_features.core.protocol.PanelEstimator`). It mirrors sklearn's
``Pipeline`` ergonomics — named steps, ``named_steps``, integer/string
``__getitem__``, slicing — while guaranteeing panel and leakage safety:

* :meth:`Pipeline.fit` fits each step **only on training-fold data**, threading
  the transformed :class:`~polars_features.core.panel_frame.PanelFrame` forward.
  Step *k* is fitted on the output of steps *0..k-1* applied to the train fold,
  so no step ever sees a future row or a test-fold statistic.
* :meth:`Pipeline.transform` / :meth:`Pipeline.predict` apply the **already
  fitted** steps in order; nothing is re-fitted, so applying the pipeline to a
  test fold cannot leak.

All steps except the last must be transformers. The last may be a transformer
(pure feature pipeline) or an estimator (then :meth:`predict` is available).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import polars as pl

from polars_features.core.panel_frame import PanelFrame
from polars_features.core.protocol import PanelEstimator, PanelTransformer

if TYPE_CHECKING:
    from typing import Self

__all__ = ["Pipeline"]

Step = tuple[str, PanelTransformer]


class Pipeline(PanelTransformer):
    """A sequential chain of panel transformers with a leak-safe ``fit``.

    Parameters
    ----------
    steps : list of (str, PanelTransformer)
        Named steps, applied in order. Names must be unique and must not contain
        ``"__"`` (reserved, sklearn-style, for nested parameter access). Every
        step except the last must be a
        :class:`~polars_features.core.protocol.PanelTransformer`; the last may be
        a :class:`~polars_features.core.protocol.PanelEstimator`.

    Attributes
    ----------
    steps : list of (str, PanelTransformer)
        The configured steps.
    named_steps : dict[str, PanelTransformer]
        Mapping from step name to the step object (sklearn-compatible).
    panel_safe : bool
        True iff every step is panel-safe.
    leakage_safe : bool
        True iff every step is leakage-safe.

    Raises
    ------
    TypeError
        If ``steps`` is malformed, a non-final step is not a transformer, or a
        step object is not a PanelTransformer.
    ValueError
        If step names are non-unique or contain ``"__"``.

    Examples
    --------
    >>> pipe = Pipeline([("scale", scaler), ("model", forecaster)])  # doctest: +SKIP
    >>> pipe.fit(train_panel)                                        # doctest: +SKIP
    >>> preds = pipe.predict(test_panel)                            # doctest: +SKIP

    Notes
    -----
    **Leakage guarantee.** Because every step is fitted strictly on the
    train-fold output of its predecessors and never re-fitted at transform time,
    the pipeline preserves the per-step leakage contract end to end. A pipeline
    is leakage-safe iff all of its steps are.
    """

    # Computed in __init__ from the constituent steps; declared here so the
    # PanelTransformer subclass-check passes (these are overwritten per-instance).
    panel_safe: bool = True
    leakage_safe: bool = True

    def __init__(
        self,
        steps: list[Step],
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self._validate_steps(steps)
        self.steps: list[Step] = list(steps)
        # Instance-level safety flags derived from the steps.
        self.panel_safe = all(t.panel_safe for _, t in self.steps)
        self.leakage_safe = all(t.leakage_safe for _, t in self.steps)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_steps(steps: object) -> None:
        if not isinstance(steps, (list, tuple)) or len(steps) == 0:
            raise TypeError(
                "Pipeline `steps` must be a non-empty list of (name, transformer) "
                f"tuples, got {steps!r}."
            )
        names: list[str] = []
        for i, step in enumerate(steps):
            if not (isinstance(step, tuple) and len(step) == 2):
                raise TypeError(
                    f"step {i} must be a (name, transformer) tuple, got {step!r}."
                )
            name, obj = step
            if not isinstance(name, str):
                raise TypeError(
                    f"step {i} name must be a string, got {type(name).__name__!r}."
                )
            if "__" in name:
                raise ValueError(
                    f"step name {name!r} must not contain '__' (reserved for "
                    "nested parameter access)."
                )
            if not isinstance(obj, PanelTransformer):
                raise TypeError(
                    f"step {name!r} must be a PanelTransformer, got "
                    f"{type(obj).__name__!r}."
                )
            names.append(name)
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"Pipeline step names must be unique; duplicates: {dupes}."
            )
        # All but last must be transformers (not necessarily estimators). Every
        # PanelTransformer can transform, so the only thing we forbid is a
        # *non-final* step that cannot transform — which is impossible given the
        # type check above. We additionally warn-by-raising if a non-final step
        # is an estimator used purely as a transformer is fine, so nothing else
        # to enforce here.

    # ------------------------------------------------------------------ #
    # PanelTransformer hooks
    # ------------------------------------------------------------------ #
    def _fit(self, panel: PanelFrame) -> None:
        """Fit each step on the train-fold output of its predecessors."""
        out = panel
        # Fit-transform every step except the last; fit the last appropriately.
        for name, transformer in self.steps[:-1]:
            out = transformer.fit_transform(out)
            if not isinstance(out, PanelFrame):
                raise TypeError(
                    f"step {name!r} ({type(transformer).__name__}) returned "
                    f"{type(out).__name__!r} from transform; every step must "
                    "return a PanelFrame so the next step can consume it."
                )
        # Final step: fit on the threaded output. Do not transform here; the
        # threaded value is only needed for downstream fitting which is done.
        last_name, last = self.steps[-1]
        last.fit(out)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        """Apply every fitted step's ``transform`` in order."""
        out = panel
        for name, transformer in self.steps:
            out = transformer.transform(out)
            if not isinstance(out, PanelFrame):
                raise TypeError(
                    f"step {name!r} ({type(transformer).__name__}) returned "
                    f"{type(out).__name__!r} from transform; expected PanelFrame."
                )
        return out

    def predict(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> PanelFrame:
        """Apply all transformers, then ``predict`` with the final estimator.

        Parameters
        ----------
        X : PanelFrame | polars.DataFrame | polars.LazyFrame
            Data to predict on (train or test). Accepts a :class:`PanelFrame` or
            a bare polars frame wrapped on the fly using ``entity`` / ``time``
            (or the keys configured on the constructor).
        entity, time : str, optional
            Panel keys for a bare polars frame.

        Returns
        -------
        PanelFrame
            Predictions from the final estimator, keyed by ``(entity, time)``.

        Raises
        ------
        RuntimeError
            If the pipeline is not fitted.
        TypeError
            If the final step is not a
            :class:`~polars_features.core.protocol.PanelEstimator`.
        """
        panel = self._as_panel(X, method="predict", entity=entity, time=time)
        self._check_fitted("predict")
        last_name, last = self.steps[-1]
        if not isinstance(last, PanelEstimator):
            raise TypeError(
                f"final step {last_name!r} is a {type(last).__name__}, which is "
                "not a PanelEstimator; `predict` requires the last step to be an "
                "estimator. Use `transform` for a feature-only pipeline."
            )
        out = panel
        for _name, transformer in self.steps[:-1]:
            out = transformer.transform(out)
        return last.predict(out)

    # ------------------------------------------------------------------ #
    # sklearn-shaped ergonomics
    # ------------------------------------------------------------------ #
    @property
    def named_steps(self) -> dict[str, PanelTransformer]:
        """Mapping from step name to step object."""
        return dict(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __getitem__(self, key: int | str | slice) -> PanelTransformer | Self:
        """Index a step by position or name, or slice into a sub-pipeline.

        Parameters
        ----------
        key : int | str | slice
            * ``int`` -> the transformer at that position.
            * ``str`` -> the transformer with that name.
            * ``slice`` -> a new :class:`Pipeline` of the selected steps.

        Returns
        -------
        PanelTransformer or Pipeline
        """
        if isinstance(key, slice):
            return type(self)(self.steps[key])
        if isinstance(key, str):
            try:
                return self.named_steps[key]
            except KeyError:
                raise KeyError(
                    f"no step named {key!r}; available steps: "
                    f"{[n for n, _ in self.steps]}."
                ) from None
        if isinstance(key, int):
            return self.steps[key][1]
        raise TypeError(
            f"Pipeline indices must be int, str, or slice, not {type(key).__name__!r}."
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        body = ", ".join(f"({n!r}, {type(o).__name__})" for n, o in self.steps)
        state = "fitted" if getattr(self, "_fitted", False) else "not fitted"
        return f"Pipeline(steps=[{body}], {state})"
