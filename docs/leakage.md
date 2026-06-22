# Leakage & correctness-by-construction

> **Your backtest is lying to you.**

The single most common reason a model that looks brilliant in research dies in production is
**leakage** — information from the future, or from the test set, sneaking into features or
fitting decisions that are supposed to only know the past. PanelKit's central design goal is
to make leakage-proof feature engineering the *default*, not something you remember to do.

!!! note "Status"
    The leak-safe `Pipeline`, `PanelFrame`, labeling, and CPCV described here are **roadmap
    (coming soon)**. This page documents the correctness model PanelKit is being built around
    so you can evaluate the approach and design panel-safe code today. See the
    [Quickstart](quickstart.md) for what ships now.

## What is leakage in panel data?

Panel data is **many entities observed over time** (entity × time): stocks by day, customers
by month, sensors by minute. Two flavors of leakage dominate:

### 1. Temporal (lookahead) leakage

Information from time `t+k` influences a feature or label at time `t`. Classic culprits:

- A misaligned `.shift` that pulls the future backward.
- Rolling/aggregate windows that include the current or future row.
- A label built from a horizon that overlaps the features without purging.

### 2. Cross-sectional / fit leakage

Statistics computed over data the model shouldn't have seen yet:

- A **global** `StandardScaler` / winsorizer fit on the *entire* sample (including the test
  period), then applied to training folds.
- A cross-sectional **rank or z-score computed over the whole sample** instead of *per date*.
- **Feature selection** (e.g. mRMR, correlation filters) run once on all the data before
  splitting, so the test set leaks into which features you kept.

Even subtle versions of these inflate backtest performance dramatically and are invisible
unless your tooling prevents them.

## How PanelKit prevents leakage by construction

PanelKit treats the **panel as a first-class object** (`PanelFrame`, with an `entity` and a
`time` key) and enforces two contracts on every operation.

### The `panel_safe` contract

Within-entity operations stay **inside their entity** and run in **time order**. There is no
way to accidentally let one entity's history bleed into another's, and time-series operations
are always ordered. In Polars terms, per-entity transforms are expressed with `.over(entity)`
on a time-sorted panel:

```python
# within each ticker, in date order — cannot cross entities, cannot see the future
pk.col("close").panel.frac_diff(d=0.4).over("ticker")
```

### The `leakage_safe` contract

Cross-sectional and fit-based operations **never see the future or the test fold**.

- **Cross-sectional ops are per-date.** A rank or z-score is computed *within a single
  timestamp's slice* across entities — `.over(time)` — so it can only use information already
  available at that instant.

  ```python
  pk.col("close").xs.rank().over("date")  # rank across tickers, one date at a time
  ```

- **Anything with `fit` is fit per-fold, on training data only.** Scalers, winsorizers, and
  feature selectors live inside the `Pipeline`; during validation they are fit on each fold's
  training slice and applied to that fold's test slice. A global fit is not expressible in the
  normal path.

  ```python
  pipe = pk.Pipeline([
      pk.transform.winsorize(limits=0.01),  # fit on train slice of each fold only
      pk.select.mrmr(k=20),                  # selection inside the fold
      pk.models.lgbm_classifier(),
  ])
  ```

### Leak-safe labeling

Labels (e.g. `triple_barrier`) define an explicit **horizon** per observation. PanelKit tracks
each label's `[t0, t1]` span so validation can **purge** overlapping samples and apply an
**embargo** — preventing the train and test sets from sharing information across fold
boundaries.

### Validation: Combinatorial Purged Cross-Validation (CPCV)

Standard k-fold cross-validation is wrong for panels: shuffling rows destroys time order, and
adjacent train/test rows share label horizons. **CPCV** (López de Prado) splits the timeline
into groups, tests on combinations of held-out groups, and **purges + embargoes** around each
test block so leakage across the boundary is removed.

```python
report = pk.model_selection.validate.cpcv(
    pipe, feats, labels,
    n_splits=6, n_test_groups=2,
    embargo="5d",   # purge overlapping labels + embargo a buffer around test blocks
)
```

Because purging/embargo are driven by the label horizons recorded above, the whole
fit → transform → select → score loop is leak-safe **by construction**, not by convention.

## The mental model

| Operation kind | Safe scope | PanelKit expression |
| --- | --- | --- |
| Within-entity time-series transform | one entity, time-ordered | `.panel.*().over(entity)` |
| Cross-sectional comparison | one timestamp, across entities | `.xs.*().over(time)` |
| Stateful fit (scale, winsorize, select) | training slice of the current fold | inside `Pipeline` |
| Labeling | explicit `[t0, t1]` horizon | `label.triple_barrier(...)` |
| Validation | purged + embargoed folds | `model_selection.validate.cpcv(...)` |

If an operation can't be expressed within one of these scopes, that's the tooling telling you
it would leak.

## See also

- [Quickstart](quickstart.md)
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* (triple-barrier, purging, CPCV)
