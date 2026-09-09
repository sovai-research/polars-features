# Imputation

Model-based, leak-safe imputation transformers. This module hosts `CafeImputer`,
a sklearn-shaped `PanelTransformer` wrapping CAFE (Causal Adaptive Factor
Estimation). CAFE is a strictly point-in-time (no look-ahead) imputer: each
filled cell uses only past and contemporaneous information within its entity, so
the transform is honestly `leakage_safe = True` and safe to use inside
walk-forward / purged cross-validation.

The heavy dependency (`cafe`) is optional and imported lazily, so importing this
module never requires it.

::: panelary.imputation
