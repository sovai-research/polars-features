"""Model-based, leak-safe imputation transformers.

This module hosts :class:`CafeImputer`, a sklearn-shaped
:class:`~polars_features.core.protocol.PanelTransformer` wrapping CAFE (Causal
Adaptive Factor Estimation). CAFE is a strictly point-in-time (no look-ahead)
imputer: each filled cell uses only past + contemporaneous information within
its entity, so the transform is honestly ``leakage_safe = True`` and safe to use
inside walk-forward / purged cross-validation.

The heavy dependency (``cafe``) is optional and imported lazily, so importing
this module never requires it.
"""

from __future__ import annotations

from typing import Literal

from polars_features.core.panel_frame import PanelFrame
from polars_features.core.protocol import PanelTransformer

__all__ = ["CafeImputer"]


def _require_cafe():
    """Lazily import the optional ``cafe`` dependency with an actionable error."""
    try:
        import cafe
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(
            "CafeImputer requires the optional `cafe` dependency, which is not "
            "installed. Install it with `pip install polars_features[cafe]`."
        ) from exc
    return cafe


class CafeImputer(PanelTransformer):
    """Leak-safe, point-in-time panel imputation via CAFE.

    A :class:`~polars_features.core.protocol.PanelTransformer` that fills the
    numeric feature columns of a panel using CAFE's strictly causal (no
    look-ahead) model. Non-numeric columns and the ``(entity, time)`` keys pass
    through untouched and the original column order is preserved.

    Because CAFE is point-in-time, both guarantees are honestly ``True``:

    * ``panel_safe`` -- imputation respects entity boundaries.
    * ``leakage_safe`` -- no cell ever uses information from its own future.

    Parameters
    ----------
    engine : {"joint", "per_entity"}, default "joint"
        Panel imputation strategy passed through to CAFE. ``"joint"`` pools the
        contemporaneous cross-section (the validated default); ``"per_entity"``
        imputes each entity's series independently.
    entity, time : str, optional
        Default panel keys, used when :meth:`fit` / :meth:`transform` receive a
        bare polars frame instead of a :class:`PanelFrame`.

    Notes
    -----
    ``fit`` only records the feature columns present at training time (CAFE fits
    no cross-fold state); the actual, strictly-causal fill happens in
    ``transform``.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        *,
        engine: Literal["joint", "per_entity"] = "joint",
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self.engine = engine

    def _fit(self, panel: PanelFrame) -> None:
        # Near no-op: CAFE learns no train-fold parameters (it is fully online /
        # point-in-time at transform time). We only record the feature columns so
        # downstream introspection knows what will be imputed.
        self.feature_cols_ = list(panel.feature_cols)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        cafe = _require_cafe()
        df = panel.collect()
        filled = cafe.impute(
            df,
            panel=(panel.time_col, panel.entity_col),
            engine=self.engine,
        )
        return PanelFrame(
            filled,
            entity=panel.entity_col,
            time=panel.time_col,
            validate=False,
        )
