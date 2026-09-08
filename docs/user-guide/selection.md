# Leak-Safe Feature Selection

Feature selection is a place leakage sneaks in quietly: rank your features on the
whole dataset, keep the top-k, and every fold's "test" rows have already voted on
which columns survive. PanelKit's selectors in `polars_features.select` are built
to be run **inside a fold / on a training set only**, so the choice of features
never benefits from data you are about to score on.

!!! warning "Run selection inside the fold"
    To stay leak-safe, call these selectors on the **training block of each CV
    fold** (or on your single train split), then apply the resulting feature set
    to the matching test block. Selecting once on the full panel and re-using
    that set across folds leaks test-fold structure into training. The
    `MRMRSelector` transformer and `mda` are designed around this: they compute
    only from the rows handed in.

Three complementary functions plus one pipeline transformer:

| Tool | What it does | When to use |
| --- | --- | --- |
| `mrmr(X, y, k, ...)` | Greedy minimum-Redundancy-Maximum-Relevance: features that track the target but not each other. | Fast, model-free shortlist; drops redundant/collinear columns. |
| `mda(estimator, X, y, cv, ...)` | Mean-Decrease-Accuracy (permutation importance) evaluated **through a purged CV splitter**. | Leak-free, model-based importance you can trust for ranking. |
| `mdi(estimator, features)` | Mean-Decrease-Impurity read from a fitted tree ensemble. | Cheap in-sample importance; not leak-free — use for speed/exploration. |
| `MRMRSelector(k, ...)` | `mrmr` wrapped as a pipeline `"select"` step. | Drop selection into a `Pipeline` so it re-fits per fold automatically. |

All of them accept a `PanelFrame` or a bare polars `DataFrame` / `LazyFrame`; for
bare frames pass `entity=` / `time=`.

## Example data

```python
import numpy as np
import polars as pl

rng = np.random.default_rng(0)
rows = []
for e in range(6):
    level = rng.normal()
    for t in range(40):
        mom = rng.normal()
        vol = rng.normal()
        mom_lag = mom * 0.9 + rng.normal() * 0.1   # redundant with `mom`
        noise = rng.normal()                        # irrelevant
        ret_fwd = 2.0 * mom - 1.0 * vol + level + rng.normal() * 0.1
        rows.append(
            {"ticker": f"stock_{e}", "date": t,
             "mom": mom, "vol": vol, "mom_lag": mom_lag,
             "noise": noise, "ret_fwd": ret_fwd}
        )
df = pl.DataFrame(rows)
train = df.filter(pl.col("date") < 30)   # select on the training block only
```

`mom` and `vol` genuinely drive `ret_fwd`; `mom_lag` is a near-duplicate of
`mom`; `noise` is irrelevant. A good selector keeps `mom` and `vol` and discards
the other two.

## mRMR — relevance without redundancy

`mrmr` greedily builds a feature set that is maximally correlated with the target
yet minimally redundant with the features already chosen (Ding & Peng, 2005).
Relevance is the absolute Pearson correlation with the target, computed
Polars-natively; redundancy is the mean absolute correlation with the picks so
far. It returns the selected names **in selection order** (most informative
first). Everything is computed only from the rows in `X`, so there is no global
fit to leak.

```python
from polars_features.select import mrmr

selected = mrmr(
    train, "ret_fwd", k=2,
    features=["mom", "vol", "mom_lag", "noise"],
    entity="ticker", time="date",
)
print(selected)   # ['mom', 'vol']
```

`mom_lag` is dropped even though it correlates strongly with the target, because
it is redundant with the already-selected `mom` — exactly the behaviour that
makes mRMR a good de-correlating shortlist. Pass `k` in `1..n_features`; omit
`features=` to consider every numeric non-key, non-target column.

## MDA — leak-free permutation importance

`mda` gives a model-based importance you can trust. For each `(train, test)` fold
from a **purged** splitter, it fits a fresh clone of your estimator on the train
block, scores the held-out block (accuracy for classifiers, R² for regressors),
then permutes each feature in the test block and records the drop in score
(Lopez de Prado, 2018, Ch. 8). Because the splitter purges overlapping label
windows, training and test never share information, so the importances are
leak-free.

```python
from sklearn.ensemble import RandomForestRegressor
from polars_features.core.model_selection import PurgedKFold
from polars_features.select import mda

cv = PurgedKFold(n_splits=4, embargo=1)
importance = mda(
    RandomForestRegressor(n_estimators=100, random_state=0),
    train, "ret_fwd", cv,
    features=["mom", "vol", "mom_lag", "noise"],
    entity="ticker", time="date",
)
print(importance)
```

```text
shape: (4, 3)
┌─────────┬────────────┬────────────────┐
│ feature ┆ importance ┆ importance_std │
│ ---     ┆ ---        ┆ ---            │
│ str     ┆ f64        ┆ f64            │
╞═════════╪════════════╪════════════════╡
│ mom_lag ┆ …          ┆ …              │
│ vol     ┆ …          ┆ …              │
│ mom     ┆ …          ┆ …              │
│ noise   ┆ …          ┆ …              │
└─────────┴────────────┴────────────────┘
```

The result is a polars `DataFrame` of `feature`, `importance` (mean decrease in
score across folds), and `importance_std` (per-fold stability), sorted by
importance. `noise` lands near zero. The estimator is cloned per fold, so the
instance you pass is left unfitted. Tune with `n_repeats=` (permutation repeats)
and `random_state=`.

!!! note "Redundant features and permutation importance"
    When two features are near-duplicates (`mom` and `mom_lag` here), permutation
    importance can split or inflate their scores — permuting one still leaves its
    twin to carry the signal. This is a known property of MDA on collinear
    inputs, and a reason to run a de-correlating step like `mrmr` first.

## MDI — cheap in-sample importance

`mdi` reads `feature_importances_` off an already-fitted tree ensemble (sklearn
forests / gradient boosters, LightGBM) and pairs each value with its feature
name. It is fast but **in-sample** — not leak-free the way `mda` is. Use it for
quick exploration; prefer `mda` when leak-safety matters.

```python
from polars_features.select import mdi

features = ["mom", "vol", "mom_lag", "noise"]
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(train.select(features).to_numpy(), train.get_column("ret_fwd").to_numpy())

print(mdi(rf, features))
```

```text
shape: (4, 2)
┌─────────┬────────────┐
│ feature ┆ importance │
│ ---     ┆ ---        │
│ str     ┆ f64        │
╞═════════╪════════════╡
│ mom_lag ┆ …          │
│ mom     ┆ …          │
│ vol     ┆ …          │
│ noise   ┆ …          │
└─────────┴────────────┘
```

Pass the `features` list in the **same order** it was fed to `estimator.fit`. It
raises `AttributeError` if the estimator has no `feature_importances_`, and
`ValueError` on a length mismatch.

## MRMRSelector — selection as a pipeline step

`MRMRSelector` wraps `mrmr` as a `PanelTransformer` so selection lives inside a
`Pipeline` and re-fits on each fold's training panel automatically — the
leak-safe way to select. It fits by running `mrmr` on the training panel and
remembering the chosen names; it transforms by projecting any panel onto
`(entity, time)` + the selected features (+ the target, when `keep_target=True`,
so a downstream estimator can still see it).

```python
from polars_features.select import MRMRSelector

selector = MRMRSelector(
    k=2,
    target="ret_fwd",
    features=["mom", "vol", "mom_lag", "noise"],
    entity="ticker", time="date",
).fit(train)

print(selector.selected_)                 # ['mom', 'vol']

reduced = selector.transform(train).collect()
print(reduced.columns)   # ['ticker', 'date', 'mom', 'vol', 'ret_fwd']
```

Because the feature set is fixed at `fit` time and `transform` never re-selects,
`MRMRSelector` is `panel_safe` and `leakage_safe`. Set `keep_target=False` to
drop the target from the transformed output. Feeding the reduced panel into a
`PanelSklearnRegressor` (see [Panel ML Models](models.md)) gives a compact,
de-correlated model that fits and predicts on the same `(entity, time)` keys.

## References

- Ding, C., & Peng, H. (2005). *Minimum redundancy feature selection from
  microarray gene expression data.* Journal of Bioinformatics and Computational
  Biology, 3(2).
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 8
  ("Feature Importance").
