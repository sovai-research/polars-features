"""Panel-native, leak-safe preprocessing transformers.

Every transformer here is a :class:`~panelary.core.protocol.PanelTransformer`
subclass that honours the leakage contract: parameters are learned in ``fit``
from training rows only, and ``transform`` applies them walk-forward (per
entity, looking backwards) or strictly within a same-date cross-section.

Exports
-------
TimeSeriesScaler
    Per-entity (or global) standardization/min-max/robust scaling using
    train-only statistics.
CrossSectionalScaler
    Same-date cross-sectional standardization (``.over(time_col)``).
CrossSectionalRank
    Same-date cross-sectional rank, optionally normalized to uniform or
    Gaussian.
Neutralize
    Same-date cross-sectional OLS factor neutralization (keep residuals).
FracDiff
    Per-entity fixed-width fractional differencing (de Prado 2018), causal.
ffd_weights
    Helper that computes the fixed-width fractional-differencing weights.
"""

from __future__ import annotations

from panelary.transform.frac_diff import FracDiff, ffd_weights
from panelary.transform.neutralize import Neutralize
from panelary.transform.rank import CrossSectionalRank
from panelary.transform.scaling import CrossSectionalScaler, TimeSeriesScaler

__all__ = [
    "TimeSeriesScaler",
    "CrossSectionalScaler",
    "CrossSectionalRank",
    "Neutralize",
    "FracDiff",
    "ffd_weights",
]
