# Honest validation

Leak-safe splitting stops information flowing backwards in time. It does
**not** stop you fooling yourself. If you try two hundred feature sets and
report the best one's Sharpe ratio, that number is a maximum of two hundred
draws — it says almost nothing about the strategy and almost everything about
the search.

`polars_features.validation` is the layer that closes the gap: it takes
out-of-sample results and answers *how much of this is real?*

```python
from polars_features.validation import (
    cpcv_splits, cpcv_backtest_paths, walk_forward_splits,
    deflated_sharpe_ratio, expected_maximum_sharpe,
    probability_of_backtest_overfitting,
    romano_wolf_mean_test, benjamini_yekutieli,
    diebold_mariano, superior_predictive_ability, model_confidence_set,
    stationary_bootstrap, crps_ensemble, score_quantile_forecasts,
)
```

Everything here is pure NumPy + Polars. No SciPy, no `arch`, no `statsmodels`.

---

## 1. One backtest is one draw. Take the distribution instead.

A walk-forward backtest gives you a single trajectory. Combinatorial Purged CV
(López de Prado, AFML Ch. 12) partitions the time axis into `N` groups, tests
every combination of `k` of them — purged and embargoed exactly as
[`PurgedKFold`](validation.md) does — and recombines the folds into
`C(N,k)·k/N` **full-length out-of-sample paths**.

```python
import numpy as np
from polars_features.validation import cpcv_backtest_paths

def fit_predict(train_pos, test_pos):
    # Fit on train_pos only; return one per-period value per test position.
    ...

paths = cpcv_backtest_paths(
    n_times=len(times), fit_predict=fit_predict,
    n_groups=6, n_test_groups=2, horizon=5, embargo=5,
)            # -> (n_times, 5) : five complete backtest paths
```

Now you have a *distribution* of Sharpe ratios, drawdowns and hit rates rather
than a point estimate — and the matrix drops straight into the overfitting
diagnostics below.

`walk_forward_splits` and `walk_forward_backtest_path` provide the purged
walk-forward comparison. (The splitters in `polars_features.cross_validation`
are *not* purged; use these when labels overlap.)

!!! note "Why CPCV lowers measured PBO"
    Walk-forward leaves most of the sample untested, so the in-sample winner is
    picked from a short out-of-sample record and frequently loses to a lucky
    rival. CPCV tests every observation on some path. The test suite pins this:
    on a fixture with one genuinely skilled strategy among seven look-alikes,
    CPCV's measured PBO is lower on every seed.

---

## 2. Deflate the Sharpe for the search that produced it

The Deflated Sharpe Ratio (Bailey & López de Prado, 2014) is the Probabilistic
Sharpe Ratio evaluated at a **selection-adjusted threshold**:

$$
SR_0 = \sqrt{V}\Big[(1-\gamma)\,\Phi^{-1}\!\big(1-\tfrac{1}{N}\big)
      + \gamma\,\Phi^{-1}\!\big(1-\tfrac{1}{Ne}\big)\Big],
\qquad
DSR = \Phi\!\left(\frac{(\widehat{SR}-SR_0)\sqrt{T-1}}
{\sqrt{1-\gamma_3\widehat{SR}+\frac{\gamma_4-1}{4}\widehat{SR}^2}}\right)
$$

```python
from polars_features.validation import deflated_sharpe_ratio, expected_maximum_sharpe

expected_maximum_sharpe(n_trials=100, sharpe_variance_across_trials=0.01)
# 0.2530602894  <- a zero-skill search over 100 configurations produces this

deflated_sharpe_ratio(
    0.10,                                # observed Sharpe, PER OBSERVATION
    n_trials=100,                        # N: configurations you actually tried
    n_observations=1001,                 # T
    sharpe_variance_across_trials=0.01,  # V: variance of the N trial Sharpes
    skewness=-0.5, kurtosis=6.0,         # non-normality of the returns
)
```

!!! danger "`N` and `V` are a matched pair"
    `N` is the number of configurations you searched and `V` is the variance of
    **those configurations'** Sharpes. Passing the number of CV paths as `N`, or
    the sampling variance of one strategy's Sharpe estimator as `V`, silently
    deflates against the wrong null. `cross_validate(..., trial_sharpes=[...])`
    derives both from the same sample, which is the only fully honest wiring;
    without it, `V` falls back to the dispersion of this strategy's CPCV paths
    (a proxy) and `n_trials` must be supplied by you.

**Do not annualise** the Sharpe you pass in: the formula needs the
per-observation value. Related: `probabilistic_sharpe_ratio` (against any
benchmark) and `minimum_track_record_length` (how much more data you need
before the claim becomes credible).

---

## 3. Probability of Backtest Overfitting

PBO (Bailey, Borwein, López de Prado & Zhu, 2017) asks: how often does the
in-sample-best configuration land in the bottom half out-of-sample?

```python
pbo = probability_of_backtest_overfitting(
    paths,                # (T, S): one column per strategy or backtest path
    n_partitions=8,
    statistic="sharpe",   # rank by Sharpe, not mean — vol matters
)
```

`pbo` near 0 means selection generalises; near or above 0.5 means your search
was no better than a coin flip.

---

## 4. Selecting features without lying about it

Screening a wide panel of candidate features produces hundreds of correlated
tests. Two different error rates are worth controlling:

| Goal | Tool | Valid when |
| --- | --- | --- |
| No false positives at all (FWER) | `romano_wolf_mean_test` | any dependence — learned from a bootstrap |
| Bounded *share* of false positives (FDR) | `benjamini_hochberg` | independence or positive dependence |
| Bounded FDR, worst case | `benjamini_yekutieli` | **arbitrary** dependence — the panel default |
| FWER, no bootstrap available | `holm_bonferroni` | any dependence, conservative |

```python
# X is (T, S): per-period information coefficients, one column per feature.
result = romano_wolf_mean_test(X, alpha=0.05, n_boot=999, seed=0)
result.rejected            # bool array: which features survive
result.adjusted_pvalues    # thresholdable at any level
```

Romano-Wolf's stepdown takes each critical value from the bootstrap
distribution of the **maximum** statistic over the not-yet-rejected hypotheses.
Because the maximum is drawn from the resampled *joint* distribution, it learns
the cross-sectional correlation instead of assuming the worst case — on a
panel of highly correlated features it rejects far more than Holm while still
controlling the family-wise error rate.

For FDR on a panel, prefer `benjamini_yekutieli`: the harmonic penalty is the
price of validity under the arbitrary cross-sectional dependence panels have.

---

## 5. Is model A actually better than model B?

The escalation ladder — pairwise, then against a benchmark, then the whole set:

```python
# 1. Pairwise, serial-correlation robust, small-sample corrected.
diebold_mariano(loss_a, loss_b, horizon=5, alternative="greater")

# 2. Does ANY of M models beat the benchmark, after the search over M?
superior_predictive_ability(benchmark_loss, model_losses, n_boot=1000, seed=0)

# 3. Which models can I not tell apart at all?
model_confidence_set(losses, alpha=0.10, n_boot=1000, seed=0).included
```

`diebold_mariano` uses a Newey-West/Bartlett HAC variance (default truncation
`horizon - 1`) plus the Harvey-Leybourne-Newbold small-sample correction, and
refers the statistic to `t_{T-1}`. SPA and the MCS run on PanelKit's own
stationary bootstrap, so they inherit the fold-boundary guarantee below.

For probabilistic forecasts, use the proper scoring rules: `crps_ensemble`,
`crps_gaussian`, `crps_from_quantiles`, `pinball_loss`, `interval_score`, and
`pit_values` / `pit_histogram` for calibration diagnostics. To score a whole
panel in one Polars pass:

```python
score_quantile_forecasts(
    df, y_true="y", quantile_cols=["q10", "q50", "q90"],
    levels=[0.1, 0.5, 0.9], by="entity",
)   # per entity: pinball_<level>, coverage_<level>, crps
```

---

## 6. Bootstrap blocks must never straddle a fold boundary

Every block scheme (`moving_block_bootstrap`, `circular_block_bootstrap`,
`stationary_bootstrap`) accepts `boundaries=`: the segment cut points a block
may not cross. Without it, a resample splices one fold's observations into
another's — leakage reintroduced through the back door.

```python
from polars_features.validation import fold_boundaries, stationary_bootstrap

splits = walk_forward_splits(n_times, n_splits=5, horizon=5, embargo=5)
cuts = fold_boundaries(splits)
boot = stationary_bootstrap(returns, block_length=20, n_boot=1000,
                            boundaries=cuts, seed=0)
```

Also available: `wild_bootstrap` (heteroskedastic residuals; Rademacher,
Mammen or Gaussian multipliers) and `sieve_bootstrap` (AR fit, IID residual
resampling, regeneration — smooth paths instead of spliced ones). All are
seeded and bit-reproducible.

Choose `block_length` on the order of the autocorrelation length; `T^(1/3)` is
the usual starting point.

---

## 7. Prediction intervals that survive drift

Classical split conformal assumes **exchangeability**, which a time series does
not have. Under a volatility ramp its coverage collapses. Three fixes live in
`polars_features.conformal`:

| Method | Idea | Use when |
| --- | --- | --- |
| `adaptive_conformal_intervals` (ACI) | adapt the *level* `α_t` online | arbitrary, unmodelled drift |
| `conformal_pid_intervals` | PID-control the *radius*, optional scorecaster | drift with structure you can forecast |
| `nexcp_intervals` (NexCP) | geometric weights on stale calibration points | slow drift, want a fixed level |
| `conformalized_quantile_regression` (CQR) | conformalise a quantile model | heteroskedasticity — width should vary |

```python
from polars_features.conformal import (
    conformal_calibration_split, adaptive_conformal_intervals,
)

train_pos, calib_pos = conformal_calibration_split(
    n_times, calibration_size=0.25, horizon=5, embargo=5,
)
# ... fit on train_pos, score on calib_pos ...
res = adaptive_conformal_intervals(scores, alpha=0.1, gamma=0.03)
res.coverage        # ~0.9 even through a regime change
res.radius          # the half-width used at each step
```

!!! warning "The calibration split must be purged and embargoed"
    Conformal coverage only holds if the calibration scores come from a model
    that never saw those rows. A random split silently destroys the guarantee on
    a time series. Always go through `conformal_calibration_split` (or
    `validation.purged_calibration_split`).

---

## Reference

- López de Prado, *Advances in Financial Machine Learning* (2018), Ch. 7 & 12.
- Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014); "The Probability
  of Backtest Overfitting" (2017).
- Romano & Wolf, "Stepwise Multiple Testing as Formalized Data Snooping",
  *Econometrica* 73(4) (2005). Benjamini & Hochberg (1995);
  Benjamini & Yekutieli (2001).
- Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997);
  Hansen, "A Test for Superior Predictive Ability" (2005);
  Hansen, Lunde & Nason, "The Model Confidence Set", *Econometrica* 79(2) (2011).
- Künsch (1989); Politis & Romano (1992, 1994); Bühlmann (1997).
- Gibbs & Candès (2021); Angelopoulos, Candès & Tibshirani (2023);
  Barber, Candès, Ramdas & Tibshirani (2023); Romano, Patterson & Candès (2019).
