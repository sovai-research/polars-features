"""Leak-safe, Polars-native dimensionality reduction for panels.

Every reducer here honours the fit-on-train contract of
:class:`~panelary.core.protocol.PanelTransformer`: the scaler, the
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

Latent-**factor extraction** (emits new ``factor_1 .. factor_r`` columns rather
than projecting onto a subset of existing ones). Part of Panelary's
*interactions* theme -- higher-order structure in the **data**, with ``order=k``
mirroring ``max_order=k`` on the model-attribution side:

======================================  ===========================
your problem                            the extractor
======================================  ===========================
"just give me the baseline"             :class:`PCAFactors`
weak / masked **non-Gaussian** factors  :class:`HFAFactors` (``order=3|4``)
maximally **independent** components    :class:`ICAFactors`
heavy tails / contaminated rows         :class:`RobustPCAFactors`
======================================  ===========================

* :func:`pca_factors`, :func:`hfa_factors`, :func:`ica_factors`,
  :func:`robust_pca_factors` -- the matrix-in, ``(factors, loadings, extra)``-out
  functional cores.
* :func:`n_factors`, :func:`bai_ng`, :func:`eigenvalue_ratio` -- the shared
  factor-count selectors (Bai--Ng information criteria and the eigenvalue-ratio
  rule), pure NumPy.
* :func:`hfa_cumulant_matrix` -- the higher-order multi-cumulant matrix itself,
  with a blocked accumulation path for large panels.

HFA and the factor-count selectors need nothing beyond ``numpy`` + ``polars``;
:class:`ICAFactors` lazily requires ``scikit-learn``
(``pip install 'panelary[ml]'``).
"""

from __future__ import annotations

import warnings

from panelary.reduce._estimators import (
    HFAFactors,
    ICAFactors,
    PCAFactors,
    RobustPCAFactors,
    pca_factors,
)
from panelary.reduce._hfa import hfa_cumulant_matrix, hfa_factors
from panelary.reduce._ica import ica_factors
from panelary.reduce._n_factors import bai_ng, eigenvalue_ratio, n_factors
from panelary.reduce._robust import robust_pca_factors
from panelary.reduce.factors import StatisticalFactors
from panelary.reduce.pca import (
    PanelFactorAnalysis,
    PanelKernelPCA,
    PanelNMF,
    PanelPCA,
    PanelRandomProjection,
    PanelSVD,
    reduce_features,
)
from panelary.reduce.xs import CrossSectionalPCA

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
    # --- latent-factor extraction (the "interactions" theme, input side) ---
    "PCAFactors",
    "HFAFactors",
    "ICAFactors",
    "RobustPCAFactors",
    "pca_factors",
    "hfa_factors",
    "ica_factors",
    "robust_pca_factors",
    "hfa_cumulant_matrix",
    "n_factors",
    "bai_ng",
    "eigenvalue_ratio",
]

# PanelUMAP is defined unconditionally (its `umap-learn` import is deferred to
# fit time), but we still guard the export so the subpackage stays importable no
# matter what, mirroring the top-level optional-component pattern.
try:
    from panelary.reduce.pca import PanelUMAP
except ImportError as exc:  # pragma: no cover - defensive
    warnings.warn(
        f"Panelary: optional component 'PanelUMAP' is unavailable "
        f"({exc.__class__.__name__}: {exc}).",
        stacklevel=2,
    )
else:
    __all__.append("PanelUMAP")
