# Latent Factors: when PCA is the wrong tool

PCA eigendecomposes the **covariance** matrix. That single fact is both why it is
the right default and why it fails in two situations panel data serves up
constantly:

- **The weak factor.** A real factor whose variance contribution is buried inside
  the noise bulk. PCA cannot separate it from noise, because to PCA it *is* the
  same size as noise.
- **The masked factor.** A larger, uninteresting source — a market level, a
  common measurement artefact, correlated idiosyncratic noise — owns the leading
  eigenvector. The factor you care about is fourth or fifth in line, or gone.

Gaussian variables have **zero cumulants above order two**. So if you swap the
covariance matrix for a higher-order *cumulant* matrix, everything Gaussian drops
out of the population object — bulk and mask alike — and what is left is exactly
the non-Gaussian structure, regardless of how small its variance share is. That
is HFA, and it is why `panelary.reduce` ships it.

!!! info "Part of the interactions theme"
    HFA is the **input-side** half of Panelary's interactions theme: higher-order
    structure in the *data* (`order=3|4` here) alongside higher-order structure
    in the *model* (Shapley interactions, `max_order=k`). Same word, same
    meaning, two places it shows up.

## Choosing an extractor

| Your problem | Extractor | Why |
| --- | --- | --- |
| Baseline; explained variance is what you want | `PCAFactors` | Fastest, best understood, gives `explained_variance_ratio_`. |
| A weak or masked **non-Gaussian** factor | `HFAFactors(order=3)` | Third-cumulant eigenanalysis: blind to anything Gaussian. Use for **skewed** factors. |
| A symmetric **heavy-tailed** factor | `HFAFactors(order=4)` | The third cumulant of a symmetric factor is zero; order 4 reads excess kurtosis instead. |
| You want **independent**, not merely uncorrelated, components | `ICAFactors` | FastICA. Needs the `ml` extra. |
| Outliers / contaminated rows are rotating your subspace | `RobustPCAFactors` | Huber-reweighted covariance; bounded influence per row. |

All four share one constructor shape, one `fit`/`transform` contract, one factor
count selector, and one output: new `factor_1 … factor_r` columns.

## The 60-second version

```python
import numpy as np
import polars as pl
from panelary.reduce import HFAFactors

# A weak skewed factor hiding behind a larger Gaussian one.
rng = np.random.default_rng(0)
n, p = 2_000, 20
f = rng.standard_exponential(n) - 1.0          # the factor you want (skewed)
g = rng.standard_normal(n)                     # the Gaussian mask
lam = rng.standard_normal((p, 1)) * 0.6        # weakly loaded
mu = rng.standard_normal((p, 1)) * 1.5         # strongly loaded
X = f[:, None] @ lam.T + g[:, None] @ mu.T + rng.standard_normal((n, p))

df = pl.DataFrame(
    {"id": np.repeat([f"e{i}" for i in range(10)], n // 10),
     "t": np.tile(np.arange(n // 10), 10),
     **{f"x{j}": X[:, j] for j in range(p)}}
)

train = df.filter(pl.col("t") < 150)
test = df.filter(pl.col("t") >= 150)

hfa = HFAFactors(n_factors=1, entity="id", time="t").fit(train)
out = hfa.transform(test).collect()
out.select("id", "t", "factor_1").head()
```

`fit` learned the standardisation statistics, the NaN fill values and the
loadings from `train` and nothing else; `transform` is a frozen linear map. On
this simulation the HFA loading vector tracks `lam` closely while an ordinary PCA
first component tracks `mu` — it finds the mask, not the factor.

## Choosing `r`

Leave `n_factors=None` and the count is resolved **on the training rows**, so the
choice of `r` is as leak-safe as the loadings:

```python
from panelary.reduce import n_factors, bai_ng, eigenvalue_ratio

n_factors(X)                          # Bai-Ng IC_p2 (the default)
n_factors(X, method="eigenratio")     # Ahn-Horenstein eigenvalue ratio
bai_ng(X, criterion="IC_p1", max_r=5)
```

- **`PCAFactors` / `ICAFactors` / `RobustPCAFactors`** default to **Bai–Ng**
  `IC_p2`: minimise `ln(V_k) + k·g(N, T)` over `k`, where `V_k` is the mean
  squared residual of the rank-`k` reconstruction.
- **`HFAFactors`** defaults to the **eigenvalue ratio** applied to its *cumulant*
  spectrum. A cumulant matrix has no explained-variance interpretation, so an
  information criterion built on reconstruction error does not apply to it; a
  ratio rule needs only a spectrum.

!!! warning "Bai–Ng needs a wide panel"
    The criterion is consistent as both `N` and `T` grow, and the search is
    capped at `max_r = 8` by default for exactly that reason. With fewer than
    ~25 feature columns the penalty is too weak to be trusted — pass an explicit
    `n_factors=` or use `method="eigenratio"`.

## HFA in detail

For a centred/standardised `X` of shape `(n, p)`:

```text
G    = X @ X.T                  # (n, n) Gram
M3M  = X.T @ ((G * G) @ X)      # (p, p) third-order multi-cumulant matrix
U    = top-r eigenvectors(M3M)  # (p, r) loadings, via eigh
F    = X @ U                    # (n, r) factor scores
```

`M3M` is a Hadamard square of a Gram matrix sandwiched by `X`, so it is symmetric
PSD (Schur product theorem) and `numpy.linalg.eigh` is the correct solver.

Written as `M3M = Σ_ij (xᵢ·xⱼ)² xᵢ xⱼᵀ`, the general case is obvious: **order `m`
uses the `(m-1)`-th Hadamard power of the Gram matrix.** At `m = 3` the Gaussian
population term is an odd moment and vanishes exactly. At `m = 4` it does not —
Isserlis' theorem gives `3·tr(S²)S² + 6·S⁴` — so `order=4` subtracts that
estimated Gaussian part (`gaussian_correction=True`, the default) and measures
excess kurtosis rather than re-deriving the covariance.

```python
from panelary.reduce import hfa_factors, hfa_cumulant_matrix

F, U, extra = hfa_factors(X, r=2, order=3)   # functional core
M = hfa_cumulant_matrix(X_centred, order=4)  # the matrix itself
extra["spectrum"]                            # full cumulant spectrum, descending
```

### Memory: the `(n, n)` Gram

A pooled panel is `entities × dates`, so `n` gets large and a dense `G` is
`O(n²)`. The accumulation decomposes over **row blocks** of `G`, so
`block_rows=` computes the identical matrix with peak memory
`O(block_rows · n)`:

```python
HFAFactors(3, block_rows=2048)     # bounded memory, same answer
```

- below **5,000 rows** the dense path runs (faithful and simple);
- between 5,000 and **50,000** rows blocking switches on automatically;
- above 50,000 rows an explicit `block_rows=` is **required** — the contraction
  is `O(n²p)` in time regardless of blocking, so the cap is a runtime guard too.
  The error message names the alternatives (subsample, aggregate, or fit
  cross-sectionally per date).

### Clean-room provenance

HFA is implemented from the published equations above. The authors' R package
`hofa` ships no LICENSE file and is therefore all-rights-reserved; none of its
source was consulted or translated.

## The leak-safety contract

Every extractor sets `panel_safe = True` and `leakage_safe = True`, which cashes
out to four guarantees:

1. **Fit on train only.** Loadings, `mean_` / `std_`, and the NaN fill values
   `column_mean_` come from the `fit` panel. Fitting on train+test gives
   *different* loadings.
2. **Row-local transform.** `F = ((X - mean_) / std_) @ loadings_` and nothing
   else — a row's factors never depend on which other rows share its fold.
3. **Sign stability.** Eigenvectors are only defined up to sign, so every loading
   column is flipped so its largest-magnitude entry is positive, and that sign is
   frozen at fit time. Without this, a refit can silently flip a factor and every
   downstream coefficient with it.
4. **Determinism.** Same rows in, bit-identical parameters out; the randomised
   paths (FastICA) are seeded by default.

They compose:

```python
from panelary.core.pipeline import Pipeline

pipe = Pipeline([("factors", HFAFactors(3, keep="all"))], entity="id", time="t")
pipe.fit(train)
pipe.transform(test)   # no leakage-gate complaint
```

`keep="all"` (the default) appends the factor columns to the panel so the
extractor drops in front of a model; `keep="factors"` returns just the
`(entity, time)` keys plus `factor_1 … factor_r`.

## Robust PCA and the outlier diagnostic

`RobustPCAFactors` runs IRLS on the projection residual: rows further than
`c` robust scales from the current subspace are downweighted in proportion to how
far out they are. The final weights are worth looking at on their own:

```python
from panelary.reduce import RobustPCAFactors

ext = RobustPCAFactors(2, entity="id", time="t").fit(train)
ext.row_weights_          # 1.0 == untouched, < 1 == downweighted outlier
```

With the default `c = 1.345` (95% Gaussian efficiency) the clean-data answer is
within a hair of ordinary PCA, so the robustness is close to free.

## Dependencies

HFA, PCA, robust PCA and both factor-count selectors are **pure NumPy** — they
run on a bare `numpy + polars` install. `ICAFactors` (and
`pca_factors(svd_solver="randomized")`) lazily import `scikit-learn`:

```bash
pip install 'panelary[ml]'
```

Importing `panelary.reduce` never pulls `scikit-learn` or `scipy` into
`sys.modules`.
