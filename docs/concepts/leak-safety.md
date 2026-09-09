# Leak-safety: the conceptual foundation

> **Leak-safety is Panelary's moat.** Everything else is a convenience; this is the
> reason the library exists.

A model that scores brilliantly in research and dies in production has almost
always been *fed the future*. Panelary is designed so that the future cannot get
in — not by convention or code review, but by the shape of the API and by a
mechanical verifier you can run on any transform. This page explains what leakage
is in panel data, the two independent axes it travels along, and exactly how
Panelary closes each one.

## What is a panel, and what is leakage?

A **panel** is many entities observed over time: stocks by day, customers by
month, sensors by minute. Every row is one observation of one **entity** at one
**time**, and the features live in the remaining columns. Panelary models this
directly as a [`PanelFrame`][panelframe] — a lazy view that remembers which
column is the entity key and which is the time key.

**Leakage** is any path by which information the model would not have had at
prediction time flows into a feature, a label, or a fitting decision. The output
at time `t` is supposed to depend only on data available *at or before* `t`. When
it depends on anything later — even subtly — the backtest is measuring
information the strategy will never actually have.

## The two leak axes

Panel data leaks along **two independent axes**, and a correct library must close
both.

### Axis 1 — Temporal (per-entity look-ahead)

Information from time `t+k` influences a value at time `t`, *within the same
entity*. Classic culprits:

- A misaligned shift that pulls the future backwards (`shift(-1)` instead of
  `shift(1)`).
- A rolling or aggregate window that includes the current or a future row.
- A label whose horizon overlaps the features without being purged.

### Axis 2 — Cross-sectional (cross-entity)

Information from *other entities*, or from the whole sample, contaminates a value:

- A cross-sectional rank or z-score computed **over the entire sample** instead of
  *per date* — so an entity's score at one date depends on entities' values at
  other dates.
- A per-entity window silently computed **across the whole frame**, so one
  entity's history bleeds into the next.
- A global scaler, winsorizer, or feature selector fit on data that includes the
  test period.

The two axes are orthogonal. A `shift(1)` is temporally safe but says nothing
about cross-entity bleed; a per-date rank is cross-sectionally scoped but says
nothing about the future. Panelary tracks both with two explicit flags on every
operator.

## The `panel_safe` / `leakage_safe` contract

Every operator Panelary ships is registered as a
[`FeatureSpec`][featurespec] in a process-wide registry, and each spec carries two
booleans that map exactly onto the two axes:

| Flag | Meaning | Axis |
| --- | --- | --- |
| `leakage_safe` | The operator is **causal**: its value at time `t` never uses data after `t`. | Temporal |
| `panel_safe` | The operator does not bleed information **across entities** when applied per entity. | Cross-sectional |

These are not aspirational labels. They are backed by construction (see below),
audited (`registry.audit()` flags any operator missing either flag or carrying a
non-permissive license), and — for the temporal axis — independently checkable
with [`assert_no_lookahead`](#how-assert_no_lookahead-works).

```python
import panelary  # registers the .panel / .xs namespaces + specs
from panelary import registry

for name in ["frac_diff", "zscore", "rank", "demean"]:
    sp = registry.get(name)
    print(f"{sp.qualified_name}: panel_safe={sp.panel_safe} leakage_safe={sp.leakage_safe}")
```

```text
panel.frac_diff: panel_safe=True leakage_safe=True
panel.zscore: panel_safe=True leakage_safe=True
xs.rank: panel_safe=False leakage_safe=True
xs.demean: panel_safe=False leakage_safe=True
```

Read `xs.rank` carefully: it is `leakage_safe=True` (it only ever looks at one
date) but `panel_safe=False` — *by design*. A cross-sectional rank **must** mix
entities within a date; that is the whole point. The flag is honest about it
rather than pretending the op is something it isn't.

## How Panelary prevents each leak by construction

### Mandatory `over` keys close the cross-sectional axis

The `.panel` and `.xs` namespaces are the two Polars-native operator families,
and they enforce the axis discipline at the frame level by **requiring an `over`
key**.

- `.panel.*` operators are per-entity time-series transforms. On a bare frame they
  demand `over=<entity>`. Omit it and you get a hard error — because a per-entity
  op computed across the whole frame would bleed one entity's history into the
  next (an Axis-2 leak).
- `.xs.*` operators are cross-sectional. They demand `over=<time>` — a
  cross-sectional operation is undefined without a cross-section.

```python
import polars as pl
import panelary  # noqa: F401

df = pl.DataFrame({
    "ticker": ["A", "A", "A", "B", "B", "B"],
    "date":   [1, 2, 3, 1, 2, 3],
    "close":  [10.0, 11.0, 12.0, 20.0, 19.0, 21.0],
})

# SAFE: trailing z-score within each ticker, in date order.
safe = df.panel.zscore("close", window=2, over="ticker", suffix="_z")

# LEAKY (and refused): no entity key -> would compute across every ticker at once.
try:
    df.panel.zscore("close", window=2)
except ValueError as e:
    print(str(e)[:70])
```

```text
panel.zscore requires an `over` entity key (e.g. over='ticker'). Witho
```

The [expression form][panel-ns] (`pl.col("close").panel.zscore(2).over("ticker")`)
leaves grouping to you, so you can compose it however you need; the frame-level
form is the guard-railed default that cannot be called wrong.

### Causal windows close the temporal axis

Every `.panel` operator is a **causal kernel**: the value at row `t` is a function
of `x[t], x[t-1], …` only. `zscore` uses a trailing rolling mean/std; `frac_diff`
is a fixed-width convolution of lagged terms with leading rows set to `null` until
the window fills; `rs_vol` is a trailing standard deviation. None of them can see
a future row, so `leakage_safe=True` holds regardless of how you group.

Cross-sectional `.xs` operators are causal for free: they only ever touch one
timestamp's slice, so no future information is even in scope. That is why they are
uniformly `leakage_safe=True`.

### `t1` label spans + purge/embargo close the label axis

A label almost always looks *forward* — a triple-barrier or fixed-horizon label
resolves somewhere in `[t, t1]`. That forward span is itself a leakage vector: if
a training row's `[t, t1]` overlaps a test row's, the two share outcome
information across the fold boundary. Panelary's labelers emit an explicit **`t1`
column** (the event-end timestamp, in the same dtype as the time axis), and the
cross-validators consume it to **purge** overlapping training rows and **embargo**
a buffer after each test block. The mechanics live in the practical guide,
[Leakage & correctness-by-construction](../leakage.md).

## How `assert_no_lookahead` works

Flags and construction arguments are only as trustworthy as your ability to
*check* them. Panelary's keystone check is a **future-perturbation experiment** —
a model-agnostic test that any temporal leak must fail:

1. Run the operation on a panel and record its output.
2. Corrupt **every value strictly in the future** (past a cut time), leaving the
   past untouched.
3. Run the operation again.
4. If the operation is leak-free, every output cell **in the past** must be
   bit-identical (within `tol`) across the two runs. Any difference means
   information from a perturbed future row flowed backwards — a look-ahead.

The perturbation is deliberately violent (a ~1e6 random offset), so any leak
shows up far above tolerance. On failure the assertion names the first offending
`(column, entity, time)`.

```python
import polars as pl
from panelary import PanelFrame, assert_no_lookahead

df = pl.DataFrame({
    "ticker": ["A", "A", "A", "B", "B", "B"],
    "date":   [1, 2, 3, 1, 2, 3],
    "close":  [10.0, 11.0, 12.0, 20.0, 19.0, 21.0],
})
panel = PanelFrame(df, entity="ticker", time="date")

# A trailing lag is causal -> passes silently.
assert_no_lookahead(pl.col("close").shift(1).over("ticker").alias("lag"), panel)

# A forward lag reads t+1 -> caught.
try:
    assert_no_lookahead(pl.col("close").shift(-1).over("ticker").alias("lead"), panel)
except AssertionError as e:
    print(str(e)[:52])
```

```text
LOOK-AHEAD LEAK DETECTED: perturbing the future (dat
```

`op` may be a `polars.Expr` (applied via `with_columns`) or a callable
`frame -> frame` (a `PanelFrame`, `DataFrame`, or `LazyFrame` in and out — the
calling convention is auto-detected), so you can wrap an entire feature function,
not just a single expression.

The cross-validation-boundary form,
[`assert_no_train_test_leak`][testing], applies the same mechanism to a
`(train, test)` split: it perturbs the *test* fold and asserts the *train*-fold
outputs are unchanged. Use it with walk-forward splits (train entirely before
test) or against a purged train set — for an interior test block even a correct
backward-looking feature on a later train row legitimately depends on test-period
values, which is precisely why purging exists.

## The mental model

| Operation kind | Safe scope | Panelary expression | Flags |
| --- | --- | --- | --- |
| Within-entity time-series transform | one entity, time-ordered | `df.panel.*(..., over=entity)` | `panel_safe`, `leakage_safe` |
| Cross-sectional comparison | one timestamp, across entities | `df.xs.*(..., over=time)` | `leakage_safe` (mixes entities by design) |
| Forward-looking label | explicit `[t, t1]` span | `label.triple_barrier(...)` → `t1` column | n/a — the span is the contract |
| Validation | purged + embargoed folds | `PurgedKFold` / `CombinatorialPurgedCV` | n/a — consumes `t1` |

If an operation can't be expressed within one of these scopes, that's the tooling
telling you it would leak.

## See also

- [Leakage & correctness-by-construction](../leakage.md) — the practical guide:
  purge, embargo, CPCV, and the backtest-overfitting statistics.
- [Quickstart](../quickstart.md)
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.

[panelframe]: ../api-reference/panel-frame.md
[featurespec]: ../api-reference/registry.md
[panel-ns]: two-tier-api.md#tier-1-bare-frame-namespaces-panel-xs-ts
[testing]: ../api-reference/testing.md
