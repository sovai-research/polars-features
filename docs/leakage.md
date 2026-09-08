# Leakage & correctness-by-construction

> **Your backtest is lying to you.**

The single most common reason a model that looks brilliant in research dies in
production is **leakage** — information from the future, or from the test set,
sneaking into features or fitting decisions that are supposed to know only the
past. This is the practical guide: how PanelKit's shipped tools purge and embargo
folds, run finance-grade cross-validation, drive purging from label spans, and
tell you whether a good-looking backtest is real.

For the *why* — the two leak axes, the `panel_safe` / `leakage_safe` contract, and
the future-perturbation verifier — start with
[Leak-safety: the conceptual foundation](concepts/leak-safety.md).

!!! tip "Everything on this page ships today"
    `PurgedKFold`, `CombinatorialPurgedCV`, the `t1` label-driven purge,
    `deflated_sharpe_ratio`, `probability_of_backtest_overfitting`, and the
    `validate` runner are all importable from `polars_features` right now. Every
    snippet below is executed as-is against the current build.

## Why ordinary k-fold is wrong for panels

Standard k-fold cross-validation shuffles rows. On a panel that destroys two
things at once: it scrambles time order (so "train" rows sit after "test" rows),
and it places training observations whose forward-looking labels **overlap** the
test window right next to the test set. Both hand the model information it would
never have at prediction time. The fix has two parts, both from López de Prado's
*Advances in Financial Machine Learning* (2018): **purging** and **embargo**.

## Purge & embargo, in prose

Picture the sorted, unique time axis of a panel as a row of cells, one per date.
A cross-validator carves out a contiguous block as the **test** fold and wants to
use everything else for training:

```text
time ->   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
train     T  T  T  T  T  T  .  .  .  .  T  T  T  T  T  T
test      .  .  .  .  .  .  X  X  X  X  .  .  .  .  .  .
```

Two boundary effects still leak:

- **Purge.** Each observation carries a forward-looking label that resolves some
  steps later — its label spans `[t, t1]`. A training row just *before* the test
  block whose label span reaches *into* the test block shares outcome information
  with the test set. Purging removes any training row whose label window overlaps
  any test label window. With a scalar `horizon`, that means dropping training
  positions within `horizon` steps on **either side** of the test block:

```text
time ->   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
test      .  .  .  .  .  .  X  X  X  X  .  .  .  .  .  .
purge     .  .  .  .  .  p  X  X  X  X  p  .  .  .  .  .   (horizon = 1)
train     T  T  T  T  T  .  .  .  .  .  .  T  T  T  T  T
```

- **Embargo.** Serial correlation means the rows *immediately after* a test block
  are still statistically entangled with it even once labels no longer overlap.
  The embargo removes a further `embargo` steps **after** each test block from
  training:

```text
time ->   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
test      .  .  .  .  .  .  X  X  X  X  .  .  .  .  .  .
purge     .  .  .  .  .  p  X  X  X  X  p  .  .  .  .  .
embargo   .  .  .  .  .  .  .  .  .  .  .  e  .  .  .  .   (embargo = 1, after purge)
train     T  T  T  T  T  .  .  .  .  .  .  .  T  T  T  T
```

Splitting happens on the **shared unique-time axis** and the resulting train/test
time sets are mapped back to *all* entities, so per-entity grouping is respected
automatically: an entity contributes its rows for the selected times only.

## `PurgedKFold`

`PurgedKFold` partitions the time axis into `n_splits` contiguous test folds and,
for each, returns the purged-and-embargoed training panel. It yields
`(train, test)` as [`PanelFrame`](api-reference/cross-validation.md) pairs (or
integer position arrays with `return_indices=True`).

```python
import numpy as np
import polars as pl
from polars_features import PanelFrame, PurgedKFold

rng = np.random.default_rng(0)
prices = pl.DataFrame({
    "ticker": ["A"] * 40 + ["B"] * 40,
    "date": list(range(40)) * 2,
    "close": np.r_[100 * np.cumprod(1 + rng.normal(0, 0.02, 40)),
                   50 * np.cumprod(1 + rng.normal(0, 0.02, 40))],
})
panel = PanelFrame(prices, entity="ticker", time="date")

cv = PurgedKFold(n_splits=4, horizon=1, embargo=1)
for i, (train, test) in enumerate(cv.split(panel)):
    print(f"fold {i}: n_train_times={len(train.time_index())} "
          f"test={test.time_index().to_list()}")
```

```text
fold 0: n_train_times=29 test=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
fold 1: n_train_times=28 test=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
fold 2: n_train_times=28 test=[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
fold 3: n_train_times=28 test=[30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
```

Note the interior folds keep fewer training times: they lose purged/embargoed
steps on both sides, while the edge folds only lose them on one side.

- `horizon` — label span in time-steps for the scalar-purge path. `0` means each
  label is point-in-time (only the exact test times are purged).
- `embargo` — steps removed after each test block.
- `n_splits` — number of folds (`>= 2`).

## `t1`: label-driven purge

A fixed scalar `horizon` is a blunt instrument — real labels resolve at *different*
times per observation. PanelKit's labelers emit an explicit **`t1`** column (the
event-end timestamp, same dtype as the time axis), and every splitter accepts
`t1=` to purge on the true, per-observation `[t, t1]` spans instead of a scalar.

Generate labels with the triple-barrier method, then hand the `t1` column straight
to the splitter:

```python
from polars_features import triple_barrier, PurgedKFold

labeled = triple_barrier(
    prices, entity="ticker", time="date", price="close",
    pt=2.0, sl=1.0, max_holding=5, vol_lookback=10,
)
print(labeled.select("ticker", "date", "label", "t1").head(3))

# Purge is now driven by each row's real label span [t, t1].
cv_t1 = PurgedKFold(n_splits=4, embargo=0, t1="t1")
train0, test0 = next(iter(cv_t1.split(labeled)))
print("t1-purged fold 0 train times:", train0.time_index().to_list()[:8], "...")
```

```text
shape: (3, 4)
┌────────┬──────┬───────┬─────┐
│ ticker ┆ date ┆ label ┆ t1  │
│ ---    ┆ ---  ┆ ---   ┆ --- │
│ str    ┆ i64  ┆ i64   ┆ i64 │
╞════════╪══════╪═══════╪═════╡
│ A      ┆ 0    ┆ 0     ┆ 5   │
│ A      ┆ 1    ┆ 0     ┆ 6   │
│ A      ┆ 2    ┆ 1     ┆ 6   │
└────────┴──────┴───────┴─────┘
t1-purged fold 0 train times: [13, 14, 15, 16, 17, 18, 19, 20] ...
```

`t1` may be a **column name** (aggregated to the max end-time per unique time —
the most conservative purge for a shared-time-axis panel) or an **array/Series**
aligned to the sorted unique-time index. When `t1` is given it **supersedes** the
scalar `horizon`. A training row is purged when its interval `[t_j, t1_j]`
overlaps *any* test interval `[t_i, t1_i]` (standard closed-interval
intersection). `triple_barrier`, `fixed_horizon`, and `meta_label` all emit `t1`.

## `CombinatorialPurgedCV` (CPCV)

A single walk-forward path gives you one out-of-sample track record — one draw from
a noisy distribution. **CPCV** partitions the time axis into `n_groups` contiguous
groups and tests *every* combination of `n_test_groups` of them, purging and
embargoing around each test block exactly as `PurgedKFold` does. Because each group
is tested in many combinations, the per-split predictions recombine into many
distinct **backtest paths**, giving a *distribution* of out-of-sample performance
instead of a point estimate.

```python
from polars_features import CombinatorialPurgedCV

cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=1)
print("n_splits:", cpcv.n_splits, "n_paths:", cpcv.n_paths)
```

```text
n_splits: 15 n_paths: 5
```

`n_splits` is `C(n_groups, n_test_groups)`; `n_paths` is
`C(N, k) * k / N`. It also supports `horizon=`, `embargo=`, and `t1=` like
`PurgedKFold`, plus `split_with_groups()` (yields the test-group tuple per fold)
and `backtest_paths()` (the split/group assignment for each reconstructed path).

## The `validate` runner: CV + overfitting diagnostics in one call

`validate.cpcv` (and `validate.purged_kfold`) fit an estimator across every fold
and, for CPCV, reconstruct the backtest paths and compute the overfitting
statistics for you, returning a [`CVReport`](api-reference/cross-validation.md).
The estimator can be a PanelKit `Pipeline`/estimator or any sklearn-shaped object
(`fit(X, y)` / `predict(X)` over numpy arrays); it is deep-copied per fold.

```python
from sklearn.linear_model import Ridge
from polars_features import validate

feat = labeled.with_columns(
    mom=pl.col("close").pct_change().over("ticker").fill_null(0.0)
).select("ticker", "date", "mom", "ret")

report = validate.cpcv(
    Ridge(), feat, y="ret",
    n_groups=6, n_test_groups=2, embargo=1,
    entity="ticker", time="date",
)
print(report.summary())
```

```text
{'metric': 'neg_mean_squared_error', 'n_splits': 15, 'mean_score': -0.0013,
 'std_score': 0.0002, 'min_score': -0.0017, 'max_score': -0.0009,
 'n_paths': 5, 'mean_path_sharpe': 0.1572, 'deflated_sharpe': 0.6303,
 'pbo': 0.3056}
```

`CVReport` also exposes `fold_scores`, `path_sharpes`, and the
`performance_matrix` (one column per path) that feeds the statistics below. Here
the 6-group / 2-test-group scheme yields 15 splits recombined into 5 backtest
paths, and the overfitting diagnostics are filled in automatically.

## Reading the backtest-overfitting statistics

Cross-validation tells you how a *fixed* model generalises. It does **not** correct
for the fact that you tried many models and kept the best — that selection is where
most fake alpha comes from. Two statistics from de Prado close that gap.

### Deflated Sharpe Ratio (DSR)

`deflated_sharpe_ratio` returns the **probability that the strategy is genuinely
skilled** once you account for (a) how many configurations you tried, (b) the
length of the return series, and (c) non-normality (skewness, fat tails). It
deflates the observed Sharpe by the *expected maximum* Sharpe you'd see from that
many trials under the null.

```python
from polars_features import deflated_sharpe_ratio

dsr = deflated_sharpe_ratio(
    observed_sharpe=0.08,   # per-observation, NOT annualised
    n_trials=50,            # how many configs you tried
    n_observations=250,     # length of the return series
    skewness=-0.5, kurtosis=4.0,
)
print(round(dsr, 4))
```

```text
0.1489
```

**How to read it.** DSR lives in `[0, 1]`. Near **1** the strategy is unlikely to
be a false positive of your selection process; near **0** the observed Sharpe is
easily explained by having tried many configs on a short, fat-tailed series.
`0.1489` here is a warning: a per-period Sharpe of 0.08 across only 250
observations, after 50 trials, is not convincing. Pass the observed Sharpe **per
observation** (do not annualise), and set `n_trials` to the true number of
configurations you searched.

### Probability of Backtest Overfitting (PBO)

`probability_of_backtest_overfitting` takes a `(T periods x S strategies)`
performance matrix and, via Combinatorially Symmetric Cross-Validation, estimates
**the probability that the strategy you'd pick as best in-sample lands in the
bottom half out-of-sample**.

```python
import numpy as np
from polars_features import probability_of_backtest_overfitting

rng = np.random.default_rng(0)
M = rng.normal(0, 1, size=(120, 8))   # 120 periods, 8 candidate strategies
M[:, 0] += 0.3                         # strategy 0 is genuinely better
pbo = probability_of_backtest_overfitting(M, n_partitions=10)
print(round(pbo, 4))
```

```text
0.0714
```

**How to read it.** PBO is in `[0, 1]`. Near **0**, the in-sample-best strategy
tends to *stay* good out-of-sample — your selection is meaningful (here strategy 0
has a real edge, so PBO is low). Near or above **0.5**, selecting the in-sample
winner is no better than a coin flip — a textbook overfit. `n_partitions` must be
even; reduce it when `T` is small. When you run CV through `validate.cpcv`, PBO is
computed for you from the reconstructed paths and reported as `report.pbo`.

### Using them together

- **DSR** guards the *single strategy you're about to ship*: is this Sharpe real
  given how hard I searched?
- **PBO** guards the *selection procedure itself*: does picking the in-sample best
  actually work here?

A strategy worth trusting has a **high DSR** and a **low PBO**. Either alarm on its
own is enough to send you back to the drawing board.

## Verifying your own transforms

None of the above helps if a custom feature leaks before it ever reaches the
splitter. Run the future-perturbation verifier on any transform you write — it is
a pure correctness check, no model required:

```python
import polars as pl
from polars_features import assert_no_lookahead

# raises AssertionError naming the first (column, entity, time) if the op leaks
assert_no_lookahead(
    pl.col("close").shift(1).over("ticker").alias("lag"),
    prices, entity="ticker", time="date",
)
```

See [Leak-safety: the conceptual foundation](concepts/leak-safety.md#how-assert_no_lookahead-works)
for how it works and for the CV-boundary form, `assert_no_train_test_leak`.

## See also

- [Leak-safety: the conceptual foundation](concepts/leak-safety.md)
- [Quickstart](quickstart.md)
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
  (triple-barrier — Ch. 3; purging, embargo — Ch. 7; CPCV — Ch. 12)
- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio."
  *Journal of Portfolio Management*, 40(5).
- Bailey, Borwein, López de Prado & Zhu (2017). "The Probability of Backtest
  Overfitting." *Journal of Computational Finance*, 20(4).
