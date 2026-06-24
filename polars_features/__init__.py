"functime: Time-series machine learning at scale."

__version__ = "0.9.5"

# from polars_features.feature_extractors import FeatureExtractor  # noqa: F401

# --- PanelKit core (additive, leak-safe panel ML kernel) --------------------
# Exposed lazily-imported so a failure here never breaks the rest of the
# package. See `polars_features.core` for the full API.
try:
    from polars_features.core import (  # noqa: F401
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
except Exception:  # pragma: no cover - never let core import break the package
    pass

# --- PanelKit feature registry + expression namespaces ----------------------
# Importing `namespaces` registers the `.panel` / `.xs` Polars expression
# namespaces as a side effect (idempotent).
try:
    from polars_features import namespaces as namespaces  # noqa: F401
    from polars_features.registry import (  # noqa: F401
        FeatureRegistry,
        FeatureSpec,
        register_feature,
        registry,
    )
except Exception:  # pragma: no cover
    pass

# --- PanelKit panel-native transforms ---------------------------------------
try:
    from polars_features import transform as transform  # noqa: F401
except Exception:  # pragma: no cover
    pass