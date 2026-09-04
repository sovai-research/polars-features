"""Leak-safe, Polars-native dimensionality reduction for panels.

Every reducer here honours the fit-on-train contract of
:class:`~polars_features.core.protocol.PanelTransformer`: the scaler, the
rotation and the automatic ``n_components`` count are learned from the training
rows only, and components are sign-fixed deterministically so signs are stable
across refits. This is the leak-safe port of SovAI's ``dimensionality_reduction``
/ ``regime_change_pca`` (which fit on the full sample and ``bfill`` the future).

Public API
----------
Row-wise feature reducers (``panel_safe=True, leakage_safe=True``):

* :class:`PanelPCA`, :class:`PanelSVD`, :class:`PanelFactorAnalysis`,
  :class:`PanelRandomProjection`, :class:`PanelKernelPCA`, :class:`PanelNMF`
* :class:`PanelUMAP` -- optional (needs ``umap-learn``); raises an informative
  error at fit time if the dependency is missing.
* :func:`reduce_features` -- a fit-and-transform convenience.

Returns-panel factors:

* :class:`StatisticalFactors` -- ``k`` PCA factors with train-fit, sign-fixed
  loadings (exposes ``.loadings_``); a ``scope="global"`` variant is
  ``leakage_safe=False``.

Cross-sectional (``panel_safe=False, leakage_safe=True``):

* :class:`CrossSectionalPCA` -- per-date cross-sectional reduction.
"""

from __future__ import annotations

import warnings

from polars_features.reduce.factors import StatisticalFactors
from polars_features.reduce.pca import (
    PanelFactorAnalysis,
    PanelKernelPCA,
    PanelNMF,
    PanelPCA,
    PanelRandomProjection,
    PanelSVD,
    reduce_features,
)
from polars_features.reduce.xs import CrossSectionalPCA

__all__ = [
    "PanelPCA",
    "PanelSVD",
    "PanelFactorAnalysis",
    "PanelRandomProjection",
    "PanelKernelPCA",
    "PanelNMF",
    "StatisticalFactors",
    "CrossSectionalPCA",
    "reduce_features",
]

# PanelUMAP is defined unconditionally (its `umap-learn` import is deferred to
# fit time), but we still guard the export so the subpackage stays importable no
# matter what, mirroring the top-level optional-component pattern.
try:
    from polars_features.reduce.pca import PanelUMAP
except ImportError as exc:  # pragma: no cover - defensive
    warnings.warn(
        f"PanelKit: optional component 'PanelUMAP' is unavailable "
        f"({exc.__class__.__name__}: {exc}).",
        stacklevel=2,
    )
else:
    __all__.append("PanelUMAP")
