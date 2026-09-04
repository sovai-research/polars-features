"""PanelKit: leak-safe, fast feature engineering and ML for panel data, built on Polars.

PanelKit turns a long-format ``(entity, time, *features)`` panel into a first-class,
lazy, leak-safe object. It provides the :class:`PanelFrame` view, a leak-safe
sklearn-shaped :class:`Pipeline`, the ``.panel`` / ``.xs`` Polars expression and
frame namespaces, finance-grade cross-validation (:class:`PurgedKFold`,
:class:`CombinatorialPurgedCV`) and backtest-overfitting metrics
(:func:`deflated_sharpe_ratio`, :func:`probability_of_backtest_overfitting`).

PanelKit is built on the foundations of `functime <https://github.com/functime-org/functime>`_
(Apache-2.0, actively maintained) and interoperates with functime and Nixtla rather
than replacing them. The import/PyPI package is currently ``polars_features``; the
public rename to ``panelkit`` is planned but not yet effective.

Public symbols are imported lazily and guarded: an optional component that fails to
import emits a warning but never breaks the rest of the package. Whatever is actually
available is listed in :data:`__all__`.
"""

from __future__ import annotations

import warnings

__version__ = "0.4.0"

__all__ = ["__version__"]

# from polars_features.feature_extractors import FeatureExtractor  # noqa: F401


def _warn_unavailable(component: str, exc: Exception) -> None:
    """Warn that an optional PanelKit component could not be imported."""
    warnings.warn(
        f"PanelKit: optional component {component!r} is unavailable "
        f"({exc.__class__.__name__}: {exc}). The rest of the package is unaffected.",
        stacklevel=3,
    )


# --- PanelKit core (leak-safe panel ML kernel) ------------------------------
try:
    from polars_features.core import (
        CombinatorialPurgedCV,
        PanelEstimator,
        PanelFrame,
        PanelTransformer,
        Pipeline,
        PurgedKFold,
        as_panel,
        deflated_sharpe_ratio,
        expanding_window_split,
        probability_of_backtest_overfitting,
        sliding_window_split,
    )
except ImportError as exc:
    _warn_unavailable("polars_features.core", exc)
else:
    __all__ += [
        "CombinatorialPurgedCV",
        "PanelEstimator",
        "PanelFrame",
        "PanelTransformer",
        "Pipeline",
        "PurgedKFold",
        "as_panel",
        "deflated_sharpe_ratio",
        "expanding_window_split",
        "probability_of_backtest_overfitting",
        "sliding_window_split",
    ]

# --- CPCV runner (cross_validate / validate / CVReport) ---------------------
try:
    from polars_features.core.model_selection import (
        CVReport,
        cross_validate,
        validate,
    )
except ImportError as exc:
    _warn_unavailable("polars_features.core.model_selection runner", exc)
else:
    __all__ += ["CVReport", "cross_validate", "validate"]

# --- Feature registry + expression namespaces -------------------------------
# Importing `namespaces` registers the `.panel` / `.xs` Polars expression and
# frame namespaces as a side effect (idempotent).
try:
    from polars_features import namespaces as namespaces
    from polars_features.registry import (
        FeatureRegistry,
        FeatureSpec,
        register_feature,
        registry,
    )
except ImportError as exc:
    _warn_unavailable("polars_features.namespaces/registry", exc)
else:
    __all__ += [
        "namespaces",
        "FeatureRegistry",
        "FeatureSpec",
        "register_feature",
        "registry",
    ]

# --- Panel-native transforms ------------------------------------------------
try:
    from polars_features import transform as transform
except ImportError as exc:
    _warn_unavailable("polars_features.transform", exc)
else:
    __all__ += ["transform"]

# --- Feature extraction -----------------------------------------------------
# Importing `feature_extractors` registers the `ts` Polars namespace as a side
# effect and exposes the bulk `extract_features` entry point.
try:
    from polars_features import feature_extractors as feature_extractors
except ImportError as exc:
    _warn_unavailable("polars_features.feature_extractors", exc)
else:
    __all__ += ["feature_extractors"]
    try:
        from polars_features.feature_extractors import extract_features
    except ImportError as exc:
        _warn_unavailable("polars_features.feature_extractors.extract_features", exc)
    else:
        __all__ += ["extract_features"]

# --- CAFE imputation --------------------------------------------------------
try:
    from polars_features.preprocessing import cafe_impute
except ImportError as exc:
    _warn_unavailable("polars_features.preprocessing.cafe_impute", exc)
else:
    __all__ += ["cafe_impute"]

try:
    from polars_features.imputation import CafeImputer
except ImportError as exc:
    _warn_unavailable("polars_features.imputation.CafeImputer", exc)
else:
    __all__ += ["CafeImputer"]

# --- Leakage verifier -------------------------------------------------------
try:
    from polars_features.testing import assert_no_lookahead
except ImportError as exc:
    _warn_unavailable("polars_features.testing.assert_no_lookahead", exc)
else:
    __all__ += ["assert_no_lookahead"]

# --- catch22 feature set (clean-room) ---------------------------------------
try:
    from polars_features import catch22 as catch22
except ImportError as exc:
    _warn_unavailable("polars_features.catch22", exc)
else:
    __all__ += ["catch22"]

# --- Labeling (triple-barrier) ----------------------------------------------
try:
    from polars_features import label as label
    from polars_features.label import triple_barrier
except ImportError as exc:
    _warn_unavailable("polars_features.label", exc)
else:
    __all__ += ["label", "triple_barrier"]

# --- Models & feature selection subpackages ---------------------------------
try:
    from polars_features import models as models
except ImportError as exc:
    _warn_unavailable("polars_features.models", exc)
else:
    __all__ += ["models"]

try:
    from polars_features import select as select
except ImportError as exc:
    _warn_unavailable("polars_features.select", exc)
else:
    __all__ += ["select"]

# --- Cross-sectional / factor evaluation ------------------------------------
try:
    from polars_features import factor as factor
except ImportError as exc:
    _warn_unavailable("polars_features.factor", exc)
else:
    __all__ += ["factor"]

# --- Dimensionality reduction -----------------------------------------------
try:
    from polars_features import reduce as reduce
except ImportError as exc:
    _warn_unavailable("polars_features.reduce", exc)
else:
    __all__ += ["reduce"]

# --- Clustering -------------------------------------------------------------
try:
    from polars_features import cluster as cluster
except ImportError as exc:
    _warn_unavailable("polars_features.cluster", exc)
else:
    __all__ += ["cluster"]
