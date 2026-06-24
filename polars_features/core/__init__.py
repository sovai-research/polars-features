"""PanelKit core: the correctness-by-construction panel ML kernel.

This subpackage is the *moat* of PanelKit. It provides a small, stable, fully
typed, lazy-first (pure Polars) foundation that every higher-level component
(concrete transformers, the feature registry, models) builds on:

* :class:`~polars_features.core.panel_frame.PanelFrame` — a typed, lazy view over
  a long-format ``(entity, time, *features)`` panel that validates the keys once
  and stays lazy.
* :class:`~polars_features.core.protocol.PanelTransformer` /
  :class:`~polars_features.core.protocol.PanelEstimator` — sklearn-shaped base
  classes carrying an explicit, machine-checkable leakage contract
  (``panel_safe`` / ``leakage_safe``).
* :class:`~polars_features.core.pipeline.Pipeline` — a leak-safe chain of steps
  with sklearn ergonomics.
* :mod:`~polars_features.core.model_selection` — finance-grade, leak-safe
  cross-validation (:class:`PurgedKFold`, :class:`CombinatorialPurgedCV`,
  walk-forward splitters) and de Prado overfitting metrics
  (:func:`deflated_sharpe_ratio`, :func:`probability_of_backtest_overfitting`).

All public names are re-exported here and (additively) from the top-level
``polars_features`` package.
"""

from __future__ import annotations

from polars_features.core.model_selection import (
    CombinatorialPurgedCV,
    PurgedKFold,
    deflated_sharpe_ratio,
    expanding_window_split,
    probability_of_backtest_overfitting,
    sliding_window_split,
)
from polars_features.core.panel_frame import PanelFrame, as_panel
from polars_features.core.pipeline import Pipeline
from polars_features.core.protocol import PanelEstimator, PanelTransformer

__all__ = [
    # Data view
    "PanelFrame",
    "as_panel",
    # Protocols
    "PanelTransformer",
    "PanelEstimator",
    # Pipeline
    "Pipeline",
    # Cross-validation
    "PurgedKFold",
    "CombinatorialPurgedCV",
    "expanding_window_split",
    "sliding_window_split",
    # Metrics
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
]
