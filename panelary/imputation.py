"""Model-based, leak-safe imputation transformers.

This module hosts :class:`CafeImputer`, a sklearn-shaped
:class:`~panelary.core.protocol.PanelTransformer` wrapping CAFE (Causal
Adaptive Factor Estimation). CAFE is a strictly point-in-time (no look-ahead)
imputer: each filled cell uses only past + contemporaneous information within
its entity, so the transform is honestly ``leakage_safe = True`` and safe to use
inside walk-forward / purged cross-validation.

The heavy dependency (``cafe``) is optional and imported lazily, so importing
this module never requires it.

There is exactly one CAFE imputation kernel in the package,
:func:`panelary.preprocessing._cafe_impute_frame` (which owns the single
``_require_cafe`` lazy import). :class:`CafeImputer` and
:func:`panelary.preprocessing.cafe_impute` are two adapter shapes over it, so
they cannot drift apart. The kernel is imported *inside* :meth:`CafeImputer._transform`
rather than at module scope, which keeps this module's import graph to
``panelary.core`` alone and leaves the light-core import budget untouched.
"""

from __future__ import annotations

from typing import Literal

from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer

__all__ = ["CafeImputer"]


class CafeImputer(PanelTransformer):
    """Leak-safe, point-in-time panel imputation via CAFE.

    A :class:`~panelary.core.protocol.PanelTransformer` that fills the
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
    ``transform``. Being a fit step, it is still applied per fold, on training
    data only.

    The fill is delegated to the single CAFE kernel,
    :func:`panelary.preprocessing._cafe_impute_frame`, shared with
    :func:`panelary.preprocessing.cafe_impute`.

    See Also
    --------
    panelary.preprocessing.cafe_impute : the same kernel, as a pipeline
        transformer, with opt-in uncertainty / anomaly / missingness
        by-products.
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
        # Imported here, not at module scope: the kernel lives in the heavier
        # `preprocessing` module, and this keeps `panelary.imputation` importable
        # (and cheap) without it. `_cafe_impute_frame` performs the lazy,
        # `_deps.require`-routed import of the optional `cafe` package itself.
        from panelary.preprocessing import _cafe_impute_frame

        filled = _cafe_impute_frame(
            panel.collect(),
            entity_col=panel.entity_col,
            time_col=panel.time_col,
            engine=self.engine,
            feature="CafeImputer",
        )
        return PanelFrame(
            filled,
            entity=panel.entity_col,
            time=panel.time_col,
            validate=False,
        )
