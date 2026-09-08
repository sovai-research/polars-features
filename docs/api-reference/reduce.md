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

## Latent-factor extraction

The reducers above project a row onto a lower-dimensional space. The **factor
extractors** below instead emit new `factor_1 … factor_r` columns from a learned
loading matrix, and share one `fit`/`transform` contract plus one factor-count
selector. They are the input-side half of PanelKit's *interactions* theme:
higher-order structure in the **data** (`order=k`) alongside higher-order
structure in the **model** (`max_order=k`).

| Your problem | Extractor |
| --- | --- |
| Baseline; explained variance | `PCAFactors` |
| A weak or masked **non-Gaussian** factor | `HFAFactors(order=3)` (skewed) / `order=4` (heavy-tailed) |
| **Independent** rather than merely uncorrelated components | `ICAFactors` |
| Outliers rotating the subspace | `RobustPCAFactors` |

`HFAFactors` eigendecomposes a higher-order **multi-cumulant** matrix
(`M3M = Xᵀ((XXᵀ)∘(XXᵀ))X`) instead of the covariance matrix. Gaussian variables
have zero cumulants above order two, so everything Gaussian drops out and weak or
masked non-Gaussian factors become visible where PCA sees only noise. A blocked
accumulation path (`block_rows=`) keeps peak memory at `O(block_rows · n_rows)`
rather than `O(n_rows²)`. Implemented clean-room from the published equations.

`n_factors=None` resolves the count on the training rows: Bai–Ng `IC_p2` for the
covariance-based methods, the eigenvalue ratio of the cumulant spectrum for HFA.
See the [Latent Factors guide](../user-guide/factors.md) for the full narrative.

HFA, PCA, robust PCA and the selectors are pure NumPy; `ICAFactors` lazily
requires `scikit-learn` (`pip install 'polars-features[ml]'`).

::: polars_features.reduce
