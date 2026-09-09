# The background-set leak

> Panelary's [leak-safety](leak-safety.md) page covers the two classic axes:
> feature leakage (a row seeing its own future) and label leakage (a fitting
> decision seeing the test fold). Attribution adds a **third, subtler surface**
> that every SHAP library leaves wide open: the *background set*.

## What a background set is, and why it leaks

A Shapley value is not a property of a model alone. It is a property of a model
**and a reference distribution**. The value of feature `j` for row `x` answers:

> How much of `f(x) - E[f]` is attributable to `x_j`?

That `E[f]` — the expectation — is taken over a *background* (reference,
baseline) set. Change the background and every attribution changes. It is a
first-class input to the computation, and every mainstream API asks you for it
casually:

```python
# The canonical leak. `X` is the whole dataset.
explainer = shap.TreeExplainer(model, data=X)
```

In a panel, that one line commits at least two errors:

1. **Cross-fold leakage.** `X` contains the test fold. The explanation of a
   training-fold row is computed against a distribution that includes rows the
   model is about to be scored on. Any narrative you build ("the model leans on
   momentum") is partly a description of the test set.
2. **Look-ahead within the fold.** Even restricted to training rows, pooling the
   whole fold means a row at time `t` is explained against rows from `t' > t`.
   The attribution for January 2007 is computed against a distribution that
   already knows about the crisis. This is the same look-ahead Panelary forbids
   everywhere else in the library, arriving through a side door.

## The third form: the *implicit* background

The tree-native "path-dependent" mode looks background-free — you pass no `data=`
at all — and is therefore the most dangerous of the three, because there is
nothing on the call site to review:

```python
booster.predict(dmatrix, pred_contribs=True)   # no background... right?
```

There is a background. It is the **training distribution frozen into the trees'
leaf cover counts**. It is invisible, unauditable, and it leaks exactly when the
model itself was trained on rows from the future of the observation being
explained. If the model is leak-free, so is this reference; if it is not, the
explanation inherits the leak and no amount of care at the attribution call site
will reveal it.

Panelary will not select this mode silently. It is available, but only with an
explicit acknowledgement:

```python
TimeAwareBackground(mode="path_dependent")
# ValueError: ... Panelary will not select it silently.

TimeAwareBackground(
    mode="path_dependent", i_accept_path_dependent_background=True
)  # allowed, and warns
```

## How Panelary closes it

[`TimeAwareBackground`](../api-reference/explain.md) makes the
reference an object with an auditable contract, not an argument you forget:

| Property | Guarantee |
| --- | --- |
| **Fold-bound** | Built from the panel passed to `fit()` — the training fold — and carries no rows from anywhere else. |
| **Past-only** | For an observation at `(e, t)`, admissible references are `{(e', t') : t' < t}`. |
| **Embargoed** | Optionally tightened to `t' <= t - embargo`, using the same purge/embargo convention as `PurgedKFold`, counted in positions of the *training* calendar (so it works for integer, date and datetime axes alike). |
| **Deterministic** | Sampling is seeded from `(seed, n_admissible)` — never from the frame being explained. |
| **Auditable** | `background.describe(times)` / `attributor.background_report(X)` return the reference-set size for every explained time. That table is the evidence. |

Because `TreeAttributor` is a
`PanelTransformer`, the binding
is automatic: `fit(train)` builds and freezes the reference, `transform(test)`
applies it and learns nothing. Drop it behind a purged splitter and attribution
inherits the split.

## The invariants, stated precisely

These are the contract, and they are the subject of
`tests/test_explain_leakage.py`:

1. The attribution at `(e, t)` is **identical** whether or not rows from
   `t' > t` are present in the frame being explained.
2. The attribution at `(e, t)` is **identical** whether or not other entities are
   present in the frame being explained.
3. Adding rows from `t' > t` to the **training** panel does not change the
   attribution at `(e, t)`.
4. Under `policy="fold"` (the behaviour of every other library), invariant 3
   **fails** — the test suite asserts that too, so the guarantee is a measured
   difference and not a slogan.

## When there is no past

The earliest rows of a panel have no admissible reference. Panelary does not
quietly widen the window; it emits **null** attributions and warns, or raises if
you pass `on_empty="error"`. A null is a true statement about what could be
computed without looking forward. Silently substituting a pooled background is
not.

## Cross-sectional attribution

For per-date, cross-sectional models the natural reference is the *same date's*
cross-section, not a time series of the entity's own past. Use
`policy="cross_sectional"`: the reference is that date's rows **from the training
fold**, falling back to the most recent strictly-past training date when the date
is not in the fold. A future cross-section is never pooled into a past date's
background.

## What this does *not* protect against

Leak-safe attribution explains the model you actually have. If the model was
trained on leaked features, the attribution will faithfully explain a leaky
model — correctly, and misleadingly. Run
[`assert_no_lookahead` / `assert_no_train_test_leak`](../api-reference/testing.md) on the features first; attribution is the
last step, not the first.
