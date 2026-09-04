# Dimensionality reduction

Leak-safe dimensionality reduction for panels. Every reducer fits its scaler,
rotation, and automatic `n_components` selection on the **training rows only**
and applies the frozen transform at predict time, so it is `leakage_safe` under
any purged / walk-forward split — and it sign-fixes components deterministically
so principal-component signs are stable across refits.

`PanelPCA` and the thin `PanelSVD` / `PanelFactorAnalysis` /
`PanelRandomProjection` / `PanelKernelPCA` / `PanelNMF` subclasses reduce a row's
feature vector. `StatisticalFactors` extracts K statistical factors from a
returns panel (train-fit loadings; a `scope="global"` variant is gated
`leakage_safe=False`). `CrossSectionalPCA` reduces each date's cross-section.
`PanelUMAP` is available via the optional `dimreduce` extra.

::: polars_features.reduce
