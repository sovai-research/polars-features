"""Leak-safe, Polars-native labeling (López de Prado, AFML Ch. 3).

Public API
----------
triple_barrier
    Triple-barrier method with a trailing-volatility-scaled profit-take /
    stop-loss / vertical barrier.
fixed_horizon
    Fixed-horizon forward-return label (sign or continuous).
meta_label
    Meta-labeling: primary side signal + realized label -> act/pass.

Every labeler emits a ``t1`` column (the event-end timestamp) in the same
dtype as the input time column, forming the shared span contract consumed by
purged cross-validation.
"""

from __future__ import annotations

from polars_features.label._barriers import (
    fixed_horizon,
    meta_label,
    triple_barrier,
)

__all__ = [
    "triple_barrier",
    "fixed_horizon",
    "meta_label",
]
