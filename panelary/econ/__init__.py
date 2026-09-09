"""Panel & time-series econometric estimators.

The fourth pillar of Panelary: **features -> factors -> attribution -> honest
inference**. Everything in this package is written clean-room from the published
papers and depends on nothing beyond ``numpy`` and ``polars``.

What's here
-----------
:mod:`~panelary.econ._hdfe` -- **high-dimensional fixed effects**
    :func:`hdfe` absorbs N-way fixed effects by alternating projections (Gaure /
    ``reghdfe``), never materialising a dummy matrix, and reports classical,
    robust, one-way and two-way clustered (Cameron-Gelbach-Miller) and
    Driscoll-Kraay standard errors. :class:`HDFETransformer` turns the
    partialled-out columns into leak-controlled features.

:mod:`~panelary.econ._panel` -- **heterogeneous panels**
    Mean Group, Pesaran CCE (CCEMG and CCEP) and Pooled Mean Group, plus the
    LLC / IPS / CIPS panel unit-root tests and the Pesaran CD test for
    cross-sectional dependence. :class:`PanelSlopeFeatures` and
    :class:`CrossSectionalAverages` expose the by-products as features.

:mod:`~panelary.econ._famamacbeth` -- **Fama-MacBeth**
    Per-date cross-sectional regressions with Newey-West standard errors on the
    lambda series. Winsorising and standardising happen **per date**, never
    globally.

:mod:`~panelary.econ._connectedness` -- **Diebold-Yilmaz connectedness**
    Generalised-FEVD spillover networks over a rolling VAR: total, directional
    and net connectedness plus per-entity centrality as cross-entity features.

:mod:`~panelary.econ._ivx` -- **IVX predictive regression**
    Inference on predictability that stays correctly sized when the predictor is
    (locally) a unit root. The IVX-Wald statistic doubles as a leak-aware
    screening score.

:mod:`~panelary.econ._dml` -- **double machine learning**
    Cross-fitted DML for the partially linear model, with folds taken from
    Panelary's purged/embargoed splitters, plus post-double-selection LASSO with
    the rigorous (plug-in) penalty.

Leak-safety
-----------
Every class that learns parameters subclasses
:class:`~panelary.core.protocol.PanelTransformer`, declares
``panel_safe`` / ``leakage_safe``, learns everything in ``_fit`` from the rows it
is handed, and applies frozen state in ``_transform``.

Examples
--------
>>> from panelary.econ import hdfe, cce_mg, connectedness  # doctest: +SKIP
>>> res = hdfe(panel, y="ret", x=["size"], absorb=["firm", "date"],
...            cluster=["firm", "date"])  # doctest: +SKIP
"""

from __future__ import annotations

from panelary.econ._common import chi2_sf, newey_west_lrv, t_sf
from panelary.econ._connectedness import (
    ConnectednessFeatures,
    ConnectednessResult,
    connectedness,
    generalized_fevd,
    ma_coefficients,
    rolling_connectedness,
    var_ols,
)
from panelary.econ._dml import (
    DMLResult,
    DoubleMLTransformer,
    PDSResult,
    dml_partial_linear,
    lasso_penalty,
    post_double_selection,
    post_lasso,
    rigorous_lasso,
)
from panelary.econ._famamacbeth import (
    FamaMacBethResult,
    FamaMacBethTransformer,
    fama_macbeth,
)
from panelary.econ._hdfe import (
    HDFEResult,
    HDFETransformer,
    demean,
    hdfe,
)
from panelary.econ._ivx import (
    IVXResult,
    IVXSelector,
    ivx,
    ivx_instrument,
    ivx_screen,
)
from panelary.econ._panel import (
    CrossSectionalAverages,
    PanelFitResult,
    PanelSlopeFeatures,
    PanelTestResult,
    cce_mg,
    cce_pooled,
    cips,
    ips,
    llc,
    mean_group,
    pesaran_cd,
    pmg,
)

__all__ = [
    # HDFE
    "hdfe",
    "demean",
    "HDFEResult",
    "HDFETransformer",
    # heterogeneous panels
    "mean_group",
    "cce_mg",
    "cce_pooled",
    "pmg",
    "llc",
    "ips",
    "cips",
    "pesaran_cd",
    "PanelFitResult",
    "PanelTestResult",
    "PanelSlopeFeatures",
    "CrossSectionalAverages",
    # Fama-MacBeth
    "fama_macbeth",
    "FamaMacBethResult",
    "FamaMacBethTransformer",
    # connectedness
    "connectedness",
    "rolling_connectedness",
    "generalized_fevd",
    "ma_coefficients",
    "var_ols",
    "ConnectednessResult",
    "ConnectednessFeatures",
    # IVX
    "ivx",
    "ivx_instrument",
    "ivx_screen",
    "IVXResult",
    "IVXSelector",
    # double ML
    "dml_partial_linear",
    "post_double_selection",
    "rigorous_lasso",
    "post_lasso",
    "lasso_penalty",
    "DMLResult",
    "PDSResult",
    "DoubleMLTransformer",
    # shared numerics
    "chi2_sf",
    "t_sf",
    "newey_west_lrv",
]

# --- Causal econometric feature generators ----------------------------------
# `econ.features` is a self-contained sibling subpackage (unit-root battery,
# long-memory `d`, HAR-RV, Nelson-Siegel, liquidity, EVT, causal decomposition).
# Exposed as an attribute rather than star-imported so the two namespaces stay
# legible: estimators live here, feature generators live in `econ.features`.
from panelary.econ import features as features  # noqa: E402

__all__ += ["features"]
