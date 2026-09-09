# Validation

Backtesting a model on panel data is where most leakage creeps in: overlapping
labels, serial correlation across the time axis, and selection bias from trying
many configurations. Panelary implements the finance-grade cross-validation
machinery from López de Prado, *Advances in Financial Machine Learning* (AFML,
Ch. 7 & 12), adapted to long-format Polars panels.

| Tool | Purpose |
| --- | --- |
| [`PurgedKFold`](#purged-k-fold) | K-fold CV with **purge** + **embargo** on the shared time axis. |
| [`CombinatorialPurgedCV`](#combinatorial-purged-cv-cpcv) | CPCV — many backtest paths from combinations of test groups. |
| [`cross_validate`](#the-end-to-end-pipeline) / [`validate`](#convenience-recipes) | Run an estimator over a splitter → a [`CVReport`](#reading-the-report). |
| [`deflated_sharpe_ratio`](#backtest-overfitting-diagnostics) / [`probability_of_backtest_overfitting`](#backtest-overfitting-diagnostics) | Correct for multiple testing and selection bias. |

```python
from panelary import (
    cross_validate, validate, PurgedKFold, CombinatorialPurgedCV,
    deflated_sharpe_ratio, probability_of_backtest_overfitting,
)
```

## The leakage model

Time alignment runs along the **time axis**, shared across entities (a panel
backtest rebalances on common dates). Splitters therefore operate on the sorted
**unique time index**; purging and embargo are applied in time units, and the
resulting train/test time sets are mapped back to *all* entities, so per-entity
grouping is respected automatically.

Two knobs control leakage:

- **Purge** — a training observation is removed when its label window overlaps
  any test label window. Supply either a scalar `horizon` (fixed-width label
  span) or a per-row `t1=` column (the explicit label endpoint from a
  [labeler](labeling.md#the-t1-contract)). When `t1=` is given it supersedes
  `horizon`.
- **Embargo** — an extra `embargo` time-steps immediately *after* each test
  block are removed from training, to handle serial correlation.

## The end-to-end pipeline

The canonical flow is **label → features → estimator → `cross_validate` →
report**. Reusing the price panel from the [Labeling guide](labeling.md):

```python
import numpy as np
import polars as pl
from panelary import cross_validate, PurgedKFold
from panelary.label import fixed_horizon
from panelary.models import PanelSklearnRegressor
from sklearn.linear_model import Ridge

def make_prices(n_per=120, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for eid in ["AAA", "BBB", "CCC"]:
        steps = rng.normal(0.0005, 0.02, size=n_per)
        price = 100.0 * np.exp(np.cumsum(steps))
        frames.append(pl.DataFrame(
            {"ticker": [eid] * n_per, "date": list(range(n_per)), "close": price}
        ))
    return pl.concat(frames)

prices = make_prices()

# 1. Features (all trailing / walk-forward safe via .over(entity))
feat = (
    prices.lazy()
    .sort("ticker", "date")
    .with_columns(
        ret1=pl.col("close").pct_change().over("ticker"),
        mom5=(pl.col("close") / pl.col("close").shift(5).over("ticker") - 1.0),
        vol10=pl.col("close").pct_change().rolling_std(10).over("ticker"),
    )
    .collect()
)

# 2. Label (carries a `t1` label-endpoint column)
lab = fixed_horizon(feat, entity="ticker", time="date", price="close", horizon=5)
data = lab.drop_nulls(["ret1", "mom5", "vol10", "label"])

# 3. Estimator — a leak-safe panel wrapper around any sklearn model
model = PanelSklearnRegressor(
    Ridge(),
    target="label",
    features=["ret1", "mom5", "vol10"],
    entity="ticker",
    time="date",
)

# 4. Cross-validate with purge + embargo
cv = PurgedKFold(n_splits=5, horizon=5, embargo=2)
report = cross_validate(model, data, y="label", cv=cv, entity="ticker", time="date")

report.summary()
```

```python
{'metric': 'neg_mean_squared_error', 'n_splits': 5,
 'mean_score': -0.00187, 'std_score': 0.00097,
 'min_score': -0.00368, 'max_score': -0.00104}
```

`cross_validate` fits the estimator on each training fold and predicts the test
fold; the estimator is deep-copied per fold so folds are independent. The
`estimator` may be either a Panelary estimator (`fit(panel)` / `predict(panel)`)
or any sklearn-shaped object over numpy arrays. `y=` is a target **column name**
in `X`, or an array aligned to the panel rows.

!!! tip "Choose the metric"
    The default fold metric is negative mean squared error (higher is better).
    Pass `metric=fn` where `fn(y_true, y_pred) -> float` for anything else, e.g.
    accuracy for a classifier target.

## Purged K-Fold

`PurgedKFold` splits the sorted unique time index into `n_splits` contiguous
folds. Each fold is the test set once; the training set is the remaining times
with overlapping observations purged and the embargo removed. You can drive it
directly:

```python
cv = PurgedKFold(n_splits=4, horizon=3, embargo=1)
for i, (train, test) in enumerate(cv.split(data)):
    tr, te = train.collect(), test.collect()
    print(f"fold {i}: train_rows={tr.height} test_rows={te.height}")
```

```
fold 0: train_rows=225 test_rows=81
fold 1: train_rows=219 test_rows=78
fold 2: train_rows=219 test_rows=78
fold 3: train_rows=228 test_rows=78
```

`split` yields `(train_panel, test_panel)` as `PanelFrame` pairs. Pass
`return_indices=True` to instead get integer positions into the sorted unique
time index (handy for custom loops):

```python
cv_idx = PurgedKFold(n_splits=4, horizon=3, embargo=1, return_indices=True)
train_pos, test_pos = next(iter(cv_idx.split(data)))
train_pos[:5], test_pos[:5]
# (array([30, 31, 32, 33, 34]), array([0, 1, 2, 3, 4]))
```

Note the purge: with `horizon=3` the first three train positions after the test
block are removed, so training starts at position 30 rather than adjacent to the
test window.

!!! warning
    `cross_validate` needs a splitter that yields `PanelFrame`s, so construct
    `cv` with `return_indices=False` (the default) when passing it to the runner.

### Label-driven purge with `t1`

Instead of a scalar `horizon`, hand the labeler's `t1` column name to purge
against the *explicit* per-row label span. This is the recommended pairing with
[`triple_barrier`](labeling.md#triple-barrier), whose spans vary row-by-row:

```python
cv_t1 = PurgedKFold(n_splits=5, embargo=2, t1="t1")
report = cross_validate(model, data, y="label", cv=cv_t1, entity="ticker", time="date")
report.summary()["n_splits"]   # 5
```

When `t1=` is set it supersedes `horizon`. For a shared-time-axis panel, `t1` is
aggregated to the **max** end time per unique time (the most conservative purge).

## Combinatorial Purged CV (CPCV)

CPCV partitions the time index into `n_groups` contiguous groups and tests every
combination of `n_test_groups` of them — so there are `C(n_groups,
n_test_groups)` splits. Because each group appears in many test combinations,
CPCV reconstructs a number of distinct **backtest paths**, giving a
*distribution* of out-of-sample performance rather than a point estimate.

```python
cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, horizon=5, embargo=2)
cpcv.n_splits, cpcv.n_paths
# (15, 5)

report = cross_validate(model, data, y="label", cv=cpcv, entity="ticker", time="date")
report.summary()
```

```python
{'metric': 'neg_mean_squared_error', 'n_splits': 15,
 'mean_score': -0.00185, 'std_score': 0.00056,
 'min_score': -0.00291, 'max_score': -0.00098,
 'n_paths': 5, 'mean_path_sharpe': -0.205,
 'deflated_sharpe': 0.0042, 'pbo': 0.821}
```

With a target present, `cross_validate` stitches the per-split test predictions
into the `n_paths` full-coverage out-of-sample series, forms a per-period
strategy return, and fills the overfitting diagnostics automatically (see
below). CPCV is detected from the splitter, so no extra flags are needed.

## Convenience recipes

The `validate` namespace wraps the common cases so you skip constructing the
splitter by hand. Both accept the same purge/embargo/`t1` knobs and return a
`CVReport`:

```python
from panelary import validate

# Purged K-Fold in one call
rep = validate.purged_kfold(
    model, data, y="label",
    n_splits=5, horizon=5, embargo=2,
    entity="ticker", time="date",
)

# CPCV with label-driven purge
rep_cpcv = validate.cpcv(
    model, data, y="label",
    n_groups=6, n_test_groups=2, embargo=2, t1="t1",
    entity="ticker", time="date",
)
rep_cpcv.summary()["pbo"], rep_cpcv.summary()["deflated_sharpe"]
```

## Reading the report

`cross_validate` returns a `CVReport` dataclass. Beyond `summary()`, the useful
fields are:

- `fold_scores` — one score per split, in split order.
- `n_splits`, and for CPCV `n_paths`.
- `path_scores` / `path_sharpes` — per-path aggregate performance and Sharpe of
  each reconstructed backtest path (CPCV only).
- `performance_matrix` — the `(n_periods, n_paths)` matrix fed to the PBO
  estimator (CPCV only).
- `deflated_sharpe`, `pbo` — the overfitting diagnostics (CPCV only).

```python
report.fold_scores            # list[float], one per split
len(report.path_sharpes)      # 5  (CPCV n_paths)
```

## Backtest-overfitting diagnostics

CPCV fills these automatically, but both estimators are available standalone for
your own selection experiments.

**Deflated Sharpe Ratio** (Bailey & de Prado, 2014) — the probability the
strategy is truly skilled, correcting for the number of trials, non-normality,
and sample length. Pass the **per-observation** (non-annualised) Sharpe:

```python
deflated_sharpe_ratio(
    0.12,               # observed per-observation Sharpe
    n_trials=15,        # number of configurations tried
    n_observations=120, # length of the return series
    skewness=-0.3,
    kurtosis=4.0,
)
# 0.3116
```

A value near 1 means the result is unlikely to be a false positive of the
selection process.

**Probability of Backtest Overfitting** (PBO, via CSCV; Bailey et al., 2017) —
the probability that the in-sample-best strategy underperforms the median
out-of-sample. Feed a `(T, S)` matrix of per-period performance for `S`
strategy configurations:

```python
import numpy as np
rng = np.random.default_rng(1)
M = rng.normal(0, 1, size=(240, 8))   # T=240 periods, S=8 strategies
probability_of_backtest_overfitting(M, n_partitions=10)
# 0.5238
```

`n_partitions` must be an even integer `>= 2` and `<= T`. Values near 0 indicate
low overfitting; values near or above 0.5 mean selection is no better than
chance (as expected here for pure noise).

## API reference

::: panelary.core.model_selection.cross_validate
::: panelary.core.model_selection.CVReport
::: panelary.core.model_selection.PurgedKFold
::: panelary.core.model_selection.CombinatorialPurgedCV
::: panelary.core.model_selection.deflated_sharpe_ratio
::: panelary.core.model_selection.probability_of_backtest_overfitting
