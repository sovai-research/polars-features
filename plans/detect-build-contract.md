# `polars_features/detect/` — build contract (as issued to the implementation agents, 2026-09-08)

Pure **numpy + polars**. No scipy, sklearn, statsmodels, numba, Rust. Use
`polars_features.econ._common` for `ols`, `pinv_sym`, `norm_ppf`, `norm_cdf`,
`t_sf`, `factorize`, `group_mean`, `winsorize`.

Every public transformer subclasses `polars_features.core.protocol.PanelTransformer`
and MUST set `panel_safe` / `leakage_safe` class attributes.

## Hard invariants (every function)
1. **Prefix invariance.** f(x[:T])[t] == f(x[:T+k])[t] bitwise for all t <= T.
   No quantity may depend on `len(x)` — not window sizes, not thresholds,
   not lag orders, not normalisation constants.
2. **Determinism.** No unseeded RNG. Any RNG takes an explicit `seed: int`.
3. **float64 everywhere.** Upcast Float32 polars columns before accumulating.
4. `np.linalg.solve(A, b[..., None])[..., 0]` — never `solve(A, b)` (NumPy 2.0
   silently mis-solves when p == batch size). Add a test with p == K.
5. Use `@` / `matmul` for batched Gram matrices, never bare `einsum`.
6. Never `rolling_map` (measured 249x penalty). `map_batches` per group is the
   escape hatch; prefer native rolling expressions.

## File ownership — touch ONLY your file(s)
| File | Owner |
|---|---|
| `_moments.py`, `_bsadf.py` | Agent A |
| `_critvals.py` | Agent B |
| `_monitors.py` | Agent C |
| `_panel.py` | Agent D |
| `tests/test_detect_*.py` | Agent E |
| `__init__.py` | orchestrator (do not create) |

## Interfaces you may assume exist (code against these signatures)

### `_moments.py` (Agent A)
```python
def cumulative_moments(y: np.ndarray, *, lag: int = 0, block: int = 1000
                       ) -> "Moments": ...
    # anchors y -> y - y[0]; trend column normalised t/T_window; block-resets
    # the accumulator every `block` rows. Returns an opaque struct.

def window_adf(m: "Moments", starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    # vectorised ADF t-stat for every (start, end) pair. O(1) per window.
```

### `_bsadf.py` (Agent A)
```python
def bsadf_sequence(y: np.ndarray, *, min_window: int, lag: int = 0,
                   max_window: int | None = None, grid: int | None = 32,
                   refine_top_k: int = 0) -> np.ndarray:
    # length-len(y) float64, NaN before min_window. grid=None -> exhaustive
    # sup over all starts; grid=k -> geometric ladder of k window lengths.
    # refine_top_k>0 -> recompute the top-k candidates per endpoint by QR.

def bsadf_panel(y: np.ndarray, *, min_window: int, lag: int = 0, **kw) -> np.ndarray:
    # y is (N, T); returns (N, T). Vectorised over entities.
```

### `_critvals.py` (Agent B)
```python
def mc_table(*, min_window: int, lag: int, t_max: int, nrep: int = 2000,
             seed: int = 0, levels=(0.90, 0.95, 0.99), grid: int | None = 32
             ) -> np.ndarray:
    # ONE pass: nrep driftless RW paths of length t_max, bsadf_sequence on each,
    # then per-endpoint quantiles. Returns (len(levels), t_max). cv[:, t] is the
    # length-t critical value -- NEVER indexed by the caller's own sample size.

def kurozumi_boundary(k_over_m: np.ndarray, q: float) -> np.ndarray:
    # g(k/m) = q * (0.73 + 0.93 * log(0.90 + k/m))

def training_max_cv(train_stat: np.ndarray, *, alpha: float | None = None) -> float:
    # Astill-style: the critical value IS a training-sample order statistic.
```

### `_monitors.py` (Agent C) — all O(1) or O(m), no calibration needed
```python
def page_cusum_expr(col: pl.Expr) -> pl.Expr        # S - min_horizontal(S.cum_min(), 0)
def shiryaev_roberts(z, *, llr, r0=0.0) -> np.ndarray   # log domain, logaddexp
def focus(z: np.ndarray) -> np.ndarray              # monotone stack, O(log n) amortised
def hb_cusum(y: np.ndarray, *, r0: int, kappa=4.6) -> tuple[np.ndarray, np.ndarray]
def end_of_sample_S(y: np.ndarray, *, m: int = 10, studentise="white") -> np.ndarray
```

### `_panel.py` (Agent D)
```python
def breadth(stat: np.ndarray, cv: np.ndarray) -> np.ndarray      # (N,T),(T,)->(T,)
def cross_sectional_rank(stat: np.ndarray) -> np.ndarray          # (N,T)->(N,T) in [0,1]
def residualise(returns: np.ndarray, *, n_factors: int, min_periods: int
                ) -> np.ndarray   # backward-looking PCA (numpy SVD), betas frozen at t
def sieve_bootstrap_cv(y: np.ndarray, *, min_window: int, lag: int, nboot: int,
                       seed: int, levels=(0.9,0.95,0.99)) -> np.ndarray
    # resample WHOLE ROWS of the (T,N) restricted-ADF residual matrix to preserve
    # the contemporaneous cross-sectional covariance.
```

## Reference numbers to hit (measured this session)
- BSADF, T=1000, exhaustive, lag=0: target < 0.05 s pure numpy (C++ reference: 0.78 s).
- Agreement with per-window OLS: < 1e-11.
- Prefix invariance with min_window frozen: exactly 0.0.
- MC table nrep=1500, t_max=800: ~7 s single core.
- Geometric grid, 32 lengths: mean shortfall vs exhaustive ~0.063 t-units, corr 0.994.
- PSY finite-sample 95% GSADF cv at T=400, r0=0.10: ~2.20-2.23.
