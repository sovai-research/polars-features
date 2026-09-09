# Implementation Plan — `reduce`: a leak-safe latent-factor family (HFA + friends)

**Status:** ready to implement · **Owner:** TBD · **Target branch:** `panelkit-roadmap-impl` (or a fresh `feat/reduce-factors`) — that branch name predates the PanelKit → Panelary rename and is deliberately left as-is
**Author of plan:** research + design session, 2026-09-04

---

> **Part of the "interactions" theme.** HFA is the **order ≥ 2, input-side** half of a
> library-wide throughline: higher-order structure in the *data* (HFA cumulant factors) and
> in the *model* (Shapley interactions, [shap-attribution-plan.md](shap-attribution-plan.md)).
> See [interactions-theme-plan.md](interactions-theme-plan.md) for the shared vocabulary,
> the functional-decomposition backbone, and the flagship **factor-interactions** workflow
> (Shapley interactions computed on HFA-derived factors).

## 1. Objective

Add a first-class **factor-extraction** capability to Panelary: a small, cohesive family of leak-safe latent-factor estimators that *emit new factor columns* (`F = X · U`), sharing one `Fit/Transform` contract and one factor-count selector.

The centrepiece is **HFA — Higher-order multi-cumulant Factor Analysis** (Huang, Lu et al., *JBES*), which does eigenanalysis on a higher-order **cumulant** matrix instead of PCA on the covariance matrix. It recovers **weak / non-Gaussian factors** where ordinary PCA fails. No implementation of HFA exists in Python or Rust today (only the authors' R package `hofa`), so this is a genuine differentiator.

HFA ships alongside three siblings so the module reads as a designed system, not a bolt-on:

> **weak / non-Gaussian factors → HFA · independent components → ICA · heavy tails / outliers → robust PCA · the baseline → PCA.**
> One namespace, one `Fit/Transform`, one shared factor-count selector, all leak-safe.

### Non-goals (explicitly out of scope — keep the module beautiful)
- **No tensor-decomposition engine** (TensorLy / Anandkumar tensor-power / Sandia Toolbox). HFA's matrix eigen-form exists *precisely to avoid* tensors. Do not add a tensor dependency.
- **No dynamic factor models** (state-space / EM / Kalman — dfms, statsmodels DFM). Different product; overlaps `forecasting/`. Defer indefinitely.
- **No POET / matrix factor models / SOFAR / reduced-rank regression.** Scope creep.
- **No Rust plugin work in v1.** The eigen step stays in NumPy. A Rust port of the cumulant contraction is a *later* performance task (see §12), not a capability decision.

---

## 2. Design principles (inherit the existing moat)

1. **Leak-safety is the contract.** Every estimator subclasses
   [`PanelTransformer`](../../panelary/core/protocol.py): `_fit` learns
   parameters on the training panel **only** (standardization stats + loadings);
   `_transform` applies them. `transformer.fit(train).transform(test)` must never
   let test-fold statistics leak in. Declare `panel_safe` / `leakage_safe`
   class attributes.
2. **Functional core + estimator wrapper**, mirroring `select/_unsupervised.py`
   (`pfa()` function + `PFASelector` class). Each method gets a plain function
   (`hfa_factors(...)`) and a `PanelTransformer` (`HFAFactors`).
3. **Minimal dependencies.** NumPy only for the core linear algebra.
   **No scipy** (follow the `_percentileofscore` precedent — reimplement any
   small stat). `sklearn` is **lazily imported** and only for the PCA/ICA/robust
   paths that genuinely reuse it (`PCA`, `FastICA`). HFA is pure NumPy.
4. **Reuse existing helpers.** `_feature_matrix()` and `_impute_column_mean()`
   from `select/_unsupervised.py` already do "collect features → float matrix →
   train-mean-impute NaNs". Promote them to a shared internal util rather than
   duplicating (see §4.1).
5. **Determinism.** Eigenvector signs are indeterminate; a fixed sign convention
   is **mandatory** so train and test factors are consistent (see §6.4). Seed
   any randomized SVD / FastICA.

---

## 3. Module layout

New subpackage `panelary/reduce/`, structured exactly like `select/`:

```
panelary/reduce/
    __init__.py          # public API + docstring (mirror select/__init__.py)
    _factors.py          # functional cores: pca_factors, hfa_factors, ica_factors, robust_pca_factors
    _n_factors.py        # factor-count selectors: n_factors(..., method=...)
    _estimators.py       # PanelTransformer wrappers + shared _FactorExtractor base
    _common.py           # shared helpers moved/promoted from select (_feature_matrix, _impute_column_mean, standardization)
```

Rationale for a new package (not extending `select/`): selectors **project onto a
subset of existing columns**; factor extractors **emit new columns** (`factor_1
… factor_r`). Different output contract → different base class → its own home.
The name `reduce` matches the SovAI unsupervised port plan ("cluster / reduce /
PFA / anomaly / pairwise").

Register in top-level `panelary/__init__.py` the same way `select` is
surfaced, and add to docs nav (see §11).

---

## 4. Public API

### 4.1 Shared helpers (`_common.py`)
Move (or re-export) from `select/_unsupervised.py` so both packages share one
implementation:
- `feature_matrix(panel, feats) -> np.ndarray` — collect to `(n_rows, n_features)` float.
- `impute_column_mean(mat) -> np.ndarray` — train-only per-column mean fill.
- **New:** `standardize_fit(mat) -> (mean, std)` and `standardize_apply(mat, mean, std)` — center/scale with train stats; guard `std == 0` (→ 1.0). HFA and PCA both need centered (usually standardized) inputs.

> Keep the import boundary intact: `reduce/` is **Tier-2 (estimator layer)**, so
> it may import from `core/` and NumPy/sklearn, but the Tier-1 `namespaces/`
> must not import from `reduce/`.

### 4.2 Functional cores (`_factors.py`)

All return a tuple of `(factors: np.ndarray (n_rows, r), loadings: np.ndarray (n_features, r), extra: dict)` so the estimator can persist `loadings` + standardization for transform-time reuse. **`fit` learns loadings; a separate apply step projects new rows** — do not recompute loadings in transform.

```python
def pca_factors(X, r=None, *, standardize=True, svd_solver="auto", random_state=0): ...
def hfa_factors(X, r=None, *, order=3, standardize=True, block_rows=None): ...
def ica_factors(X, r=None, *, standardize=True, random_state=0, max_iter=500): ...  # wraps sklearn FastICA
def robust_pca_factors(X, r=None, *, standardize=True, c=1.345, max_iter=50): ...    # Huber-PCA
```

`r=None` → resolve via `n_factors()` (§5).

### 4.3 Estimators (`_estimators.py`)

Shared base `_FactorExtractor(PanelTransformer)` handling the common shape:
- constructor: `n_factors: int | None`, `features`, `standardize`, `keep`
  (`"factors"` | `"all"` — whether to append factors to the input panel or
  return only keys + factors), `factor_prefix="factor"`, `entity`, `time`,
  plus method-specific kwargs.
- `_fit(panel)`: resolve features → `feature_matrix` → `impute_column_mean` →
  standardize (store `mean_`, `std_`) → call the functional core → store
  `loadings_`, `n_factors_`, `feature_names_in_`, apply sign convention.
- `_transform(panel)`: standardize new rows with stored stats → `F = Xc @ loadings_`
  → attach `factor_1..factor_r` columns to the panel, preserving `(entity, time)`.
- `get_feature_names_out()` → `["factor_1", …, "factor_r"]`.
- `explained_variance_ratio_` where meaningful (PCA / robust-PCA).

Concrete classes: `PCAFactors`, `HFAFactors`, `ICAFactors`, `RobustPCAFactors`.
All set `panel_safe = True`, `leakage_safe = True`.

### 4.4 `__init__.py` exports
`pca_factors, hfa_factors, ica_factors, robust_pca_factors, n_factors,
PCAFactors, HFAFactors, ICAFactors, RobustPCAFactors` + `__all__`.

### 4.5 (Optional, phase 4) frame-level sugar
A `.reduce` frame-level namespace convenience, e.g. `df.reduce.hfa(features=..., n_factors=3, over="date")`, mirroring how `xs`/`panel` frame namespaces wrap the estimator layer. Defer until the estimators are solid.

---

## 5. Factor-count selector (`_n_factors.py`)

`n_factors(X, *, method="bai_ng", max_r=None, standardize=True) -> int`

- `method="bai_ng"` — **Bai & Ng (2002)** information criteria `IC_p1` / `IC_p2`
  over `k = 1..max_r`, minimizing `ln(V_k) + k·g(N,T)` where `V_k` is the mean
  squared residual from the rank-`k` PCA reconstruction. ~30 lines of NumPy;
  reference: the `fbi` R package. This is the connective tissue that makes HFA
  *legible* (you need the PCA baseline + a principled `r`).
- `method="eigenratio"` — **eigenvalue-ratio** test (Ahn–Horenstein style):
  `r = argmax_k λ_k / λ_{k+1}`. Cheap, works on either the covariance spectrum
  (PCA) or the **cumulant-matrix** spectrum (HFA's `GER3/GER4` analogue).
- `max_r` default: `min(n_features, some cap, n_rows-1)`.

Both HFA and PCA estimators call this when `n_factors=None`. HFA defaults to
`method="eigenratio"` on its cumulant spectrum; PCA defaults to `"bai_ng"`.

---

## 6. HFA — algorithm specification

### 6.1 Reference math (order 3, the given formula)
For a centered/standardized observation matrix `X` of shape `(n_rows, p_features)`:

```
G    = X @ X.T                 # (n, n) Gram
M3M  = X.T @ ((G * G) @ X)     # (p, p)  == t(X) %*% ((X X') ∘ (X X')) %*% X
U    = eigenvectors(M3M)[:, :r]   # (p, r), top-r by eigenvalue
F    = X @ U                   # (n, r) factor scores
```

This is the third-order-multi-cumulant estimator (`HFA3`). `M3M` is symmetric
PSD → use `numpy.linalg.eigh` (not `eig`), take the top-`r` eigenvectors by
descending eigenvalue. Loadings `U` are `(p, r)`; **persist `U` + standardization
for transform** (transform does `F_new = standardize(X_new) @ U`).

### 6.2 Scaling / memory (important)
`G = X @ X.T` is `(n_rows, n_rows)` → **O(n²) memory**. Panels can have large
`n_rows` (entities × dates). Mitigations, in order:
1. **Default dense path** for `n_rows ≤ block guard` (e.g. 5–10k). Faithful and simple.
2. **Blocked accumulation**: `M3M = Σ_blocks X_b.T @ ((G_b * G_b) @ X)` computed
   over row-blocks of `G` so peak memory is `O(block · n_rows)` not `O(n²)`.
   Expose `block_rows` kwarg; auto-enable above the guard.
3. **Guard rail**: if `n_rows` exceeds a hard cap and `block_rows` is unset,
   raise a clear error suggesting `block_rows=` or a per-date cross-sectional
   mode. Do **not** silently allocate a huge matrix.
4. *(Future)* push the contraction into the Rust plugin (§12).

### 6.3 Variants
- **v1 MVP:** `order=3` only (the formula above).
- **Extension:** `order=4` (fourth-order cumulant, `HFA4`) — an additional
  Hadamard power / cumulant construction from `hofa`. Add behind the same
  `order=` kwarg once v1 lands; do not block v1 on it.

### 6.4 Sign & scale convention (mandatory for leak-safety)
Eigenvectors have arbitrary sign; without a convention, test-fold factors can
flip relative to train. Fix at fit time and store with the loadings:
- For each column `u_j` of `U`, flip sign so that the entry of largest absolute
  value is positive (`if u[argmax(abs(u)), j] < 0: u[:, j] *= -1`).
- Optionally L2-normalize factor columns; document whichever is chosen.

### 6.5 Clean-room note
`hofa` has **no LICENSE file** → treat as all-rights-reserved. Implement HFA
**from the paper's equations** (documented above), *not* by translating `hofa`'s
R source. Cite the paper in the docstring; do not copy code.

---

## 7. PCA factors + Bai–Ng (baseline)
Thin wrapper over `sklearn.decomposition.PCA` (lazy import, as `pfa` already
does at `select/_unsupervised.py:242`). Store `components_.T` as loadings,
`explained_variance_ratio_`. `n_factors=None` → Bai–Ng `IC_p2`. This is a
**must-add** alongside HFA: it is the reference every factor method is judged
against and shares the selector.

## 8. ICA (promote existing FastICA)
`projection_importance(..., "ica")` at `select/_unsupervised.py:474` already
imports `FastICA`. Promote to a first-class `ICAFactors` factor estimator
(`ica_factors` core): fit `FastICA(n_components=r, random_state=...)`, store the
unmixing matrix as loadings, transform new rows through it. Near-free; completes
the non-Gaussian angle.

## 9. Robust / Huber-PCA (optional, rounds out the quadrant)
Idea from R's `HDRFA` (Huber PCA). Iteratively-reweighted PCA: weight each row's
contribution to the covariance by a Huber function of its Mahalanobis-ish
residual, re-estimate the leading subspace, iterate to convergence. Pure NumPy,
no new deps. Ship last; the module is already complete and cohesive without it.

---

## 10. Leak-safety & determinism requirements (test these explicitly)
1. **Fit-on-train only:** loadings + standardization computed in `_fit`; never
   recomputed in `_transform`.
2. **Idempotent transform:** `transform(test)` yields identical factors
   regardless of test-set composition (row `i`'s factors depend only on row `i`
   + frozen loadings).
3. **Sign stability:** factors on overlapping rows are identical whether those
   rows are in the fit set or a later transform set (validates §6.4).
4. **Determinism:** fixed `random_state` → bit-identical repeated runs.

---

## 11. Testing & docs

### Tests (`tests/`, mirror existing `test_*.py` style, pytest)
- `tests/test_reduce_hfa.py`
  - **Recovery:** simulate `k` non-Gaussian latent factors + Gaussian noise with
    a **weak** loading regime; assert HFA recovers the subspace (canonical
    correlation / subspace angle to truth) **better than PCA** — the paper's core
    claim and the module's reason to exist.
  - **Shape / API / `get_feature_names_out`.**
  - **Sign convention** stability across fit/transform (§10.3).
  - **Scaling:** blocked path returns same `M3M` as dense within tolerance.
  - `eigh` used on a symmetric matrix; eigenvalues sorted descending.
- `tests/test_reduce_pca.py` — PCA factors + Bai–Ng picks the right `k` on
  simulated rank-`k` data.
- `tests/test_reduce_ica.py` — separates independent non-Gaussian sources.
- `tests/test_reduce_n_factors.py` — Bai–Ng & eigenratio on known-rank data.
- `tests/test_reduce_leakage.py` — the §10 leak-safety contract, ideally driven
  through the existing purged-CV splitter (see `tests/test_leakage_verifier.py`
  / `test_cv_runner.py` for the pattern) so it proves composability inside a
  `Pipeline`.

### Docs (`docs/`, mkdocs — the repo already builds strict)
- `docs/api-reference/reduce.md` (or `factors.md`) — API autodoc, matching the
  existing `docs/api-reference/*.md` layout; add to `mkdocs.yml` nav.
- `docs/user-guide/factors.md` — the "when PCA fails" narrative + the
  four-method decision table + a FRED-MD-style worked example.
- Update `docs/index.md` feature list and `CHANGELOG.md`.

---

## 12. Dependencies & performance
- **v1 runtime deps:** NumPy (already), sklearn (already, lazy — PCA/ICA/robust
  only). **No new hard dependency.** HFA core is pure NumPy.
- **Later (separate task, not v1):** move the HFA cumulant contraction + `eigh`
  into the Rust plugin. The eigen step maps to a single symmetric-EVD call
  (`faer`/`nalgebra`); the `Xᵀ((XXᵀ)∘(XXᵀ))X` contraction is a handful of ops
  and is where a Rust/Polars-native path would most help large panels. Expose
  via the existing plugin the way other Rust ops are surfaced. **Do not attempt
  in v1.**

---

## 13. Milestones & acceptance criteria

| # | Milestone | Deliverable | Done when |
|---|-----------|-------------|-----------|
| M1 | Scaffold + shared helpers | `reduce/` package, `_common.py`, base `_FactorExtractor`, PCA factors + Bai–Ng | `PCAFactors` fits/transforms leak-safely; `n_factors` picks correct `k` on simulated data; tests green |
| M2 | **HFA (the differentiator)** | `hfa_factors` + `HFAFactors` (order 3), sign convention, blocked path | Recovers weak non-Gaussian factors **better than PCA** in sim; leak-safety tests pass |
| M3 | ICA sibling | promote FastICA → `ICAFactors` | Separates independent sources; shares API |
| M4 | Docs + polish | API + user-guide pages, CHANGELOG, nav | `mkdocs build --strict` clean; decision table published |
| M5 (opt) | Robust-PCA | `RobustPCAFactors` | Beats PCA under heavy-tailed contamination in sim |
| M6 (opt) | HFA order 4 + Rust contraction | `order=4`; plugin path | perf win on large panels; parity with NumPy |

**M1 + M2 are the must-ship core.** M3 is near-free. M4 gates the release. M5/M6 are follow-ons.

---

## 14. Open decisions (resolve before/at M1)
1. **Module name:** `reduce/` (matches SovAI plan) vs `factor/`. → Recommend `reduce/`.
2. **Panel factor semantics:** pooled across all rows (matches existing `pfa`
   convention — `feature_matrix` is `(n_rows, n_features)`) vs per-date
   cross-sectional (`.over(date)`). → Recommend **pooled** for v1 (consistent
   with `pfa`), add cross-sectional mode later if demanded.
3. **`keep` default:** append factors to the input panel (`"all"`) vs return
   keys + factors only (`"factors"`). → Recommend append (`"all"`) so it drops
   into a `Pipeline` before an estimator.
4. **Standardize by default?** → Recommend **yes** (HFA/PCA both assume centered;
   standardizing also stabilizes the cumulant scale).

---

## 15. Reference map (prior art to consult, NOT to copy)
- **HFA method:** Huang, Lu et al., "Estimation of Factors Using Higher-Order
  Multi-Cumulants in Weak Factor Models," *JBES* (forthcoming); R pkg `hofa`
  (no license — clean-room only).
- **Closest cousins (4th-order cumulant eigenanalysis):** R `JADE`, `ICtest`
  (FOBI factor-count tests), `tensorBSS`.
- **Antecedent:** Boudt–Cornilly–Verdonck nearest-comoment estimator.
- **Baselines / selector:** `fbi` (Bai–Ng + FRED-MD), `HDRFA` (Huber PCA).
- **Existing in-repo patterns to mirror:** `select/_unsupervised.py` (`pfa`,
  `_UnsupervisedSelector`, `_feature_matrix`), `core/protocol.py`
  (`PanelTransformer`), `tests/test_leakage_verifier.py`.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
