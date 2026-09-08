"""Tier-2 cross-sectional / factor evaluation toolkit.

The ``factor`` package holds the *evaluation and estimation* surface for
cross-sectional factor research: leak-safe forward-return alignment, the
information coefficient (IC/ICIR), portfolio sorts, and per-date feature-set
orthogonalization. It is **Tier 2** — it may import numpy and freely imports the
Tier-1 ``namespaces`` layer, but is never imported by it, preserving the
documented import boundary.

Leak-safety is the moat and it is centralized, not hoped for:

* forward returns come from one audited backward-shift site
  (:func:`forward_return`) with a gap guard;
* every cross-sectional statistic is computed **per date**, never pooled/global
  (IC ``.over(time)``; sorts bucket per date; orthogonalization decomposes each
  date's cross-section on its own).

Public API
----------
:func:`forward_return`
    Leak-safe forward-return alignment (backward per-entity shift + gap guard).
:func:`ic`, :func:`ic_summary`
    Per-date information coefficient and its ICIR / t-stat / hit-rate summary.
:func:`portfolio_sort`
    Per-date quantile sort, long-short spread, and monotonicity.
:func:`orthogonalize`
    Per-date feature-set de-correlation (Gram-Schmidt / QR).
"""

from __future__ import annotations

from polars_features.registry import FeatureSpec, registry

from ._align import forward_return
from .ic import ic, ic_summary
from .neutralize import orthogonalize
from .portfolio import SortResult, portfolio_sort

__all__ = [
    "forward_return",
    "ic",
    "ic_summary",
    "portfolio_sort",
    "SortResult",
    "orthogonalize",
]

_LICENSE = "Apache-2.0"
_SOURCE = "PanelKit"

_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="forward_return",
        namespace="factor",
        input_shape="frame",
        output_shape="series",
        params={"horizon": int, "price": str, "ret": str},
        tier="B",
        panel_safe=True,  # backward per-entity shift; respects entity boundaries
        leakage_safe=True,  # the single audited negative-shift site + gap guard
        source=_SOURCE,
        license=_LICENSE,
    ),
    FeatureSpec(
        name="ic",
        namespace="factor",
        input_shape="frame",
        output_shape="frame",
        params={"method": str},
        tier="B",
        panel_safe=False,  # cross-sectional: correlates across entities per date
        leakage_safe=True,
        source=_SOURCE,
        license=_LICENSE,
    ),
    FeatureSpec(
        name="portfolio_sort",
        namespace="factor",
        input_shape="frame",
        output_shape="frame",
        params={"q": int},
        tier="B",
        panel_safe=False,
        leakage_safe=True,
        source=_SOURCE,
        license=_LICENSE,
    ),
    FeatureSpec(
        name="orthogonalize",
        namespace="factor",
        input_shape="frame",
        output_shape="frame",
        params={"method": str, "by": object},
        tier="B",
        panel_safe=False,
        leakage_safe=True,
        source=_SOURCE,
        license=_LICENSE,
    ),
)

for _spec in _SPECS:
    registry.register(_spec)
