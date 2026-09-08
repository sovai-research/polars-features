# Feature Attribution

`polars_features.explain` answers "why did the model predict *this*, for *this
entity*, on *this date*?" — without letting the future into the answer.

PanelKit reimplements none of the SHAP math. Exact TreeSHAP already ships inside
the boosters you train; `shap` ships the reference interventional engine;
`shapiq` ships any-order interactions. What PanelKit adds is the argument all of
them leave to you and nobody gets right in a panel: **the background set**. See
[The background-set leak](../concepts/attribution-leakage.md) for why that is the
dominant leakage surface in attribution.

!!! warning "Fit the attributor on the training fold"
    `TreeAttributor` is a `PanelTransformer`. `fit(train)` builds and **freezes**
    the reference set; `transform(test)` applies it and learns nothing. Calling
    `fit_transform` on the whole panel explains training rows against a
    background drawn from their own past — fine — but explaining a *test* fold
    means `fit(train).transform(test)`, exactly as with any other PanelKit step.

## Quick start

```python
import numpy as np
import polars as pl

from polars_features.explain import TimeAwareBackground, TreeAttributor
from polars_features.models import PanelLGBMRegressor

rng = np.random.default_rng(0)
n_e, n_t = 8, 60
panel = pl.DataFrame(
    {
        "ticker": [f"s{e}" for e in range(n_e) for _ in range(n_t)],
        "date": [t for _ in range(n_e) for t in range(n_t)],
        "momentum": rng.normal(size=n_e * n_t),
        "value": rng.normal(size=n_e * n_t),
        "size": rng.normal(size=n_e * n_t),
    }
).with_columns(
    (2.0 * pl.col("momentum") - pl.col("value") + 0.3 * pl.col("size")).alias("fwd_ret")
)

train = panel.filter(pl.col("date") < 45)
test = panel.filter(pl.col("date") >= 45)

model = PanelLGBMRegressor(
    target="fwd_ret", entity="ticker", time="date", n_estimators=100, verbose=-1
).fit(train)

attr = TreeAttributor(model, entity="ticker", time="date", embargo=1).fit(train)
shap = attr.attributions(test)
```

`shap` is a plain Polars frame: `ticker`, `date`, one `shap_<feature>` column per
feature, and `shap_base_value`. It is ready for `group_by` without any
reshaping. Pass `output="long"` for a tidy `(entity, time, feature, shap_value)`
frame instead.

Prove the plumbing with the Shapley efficiency (local-accuracy) axiom:

```python
attr.check_efficiency(test, raise_on_fail=True)
# n_rows │ n_checked │ max_abs_error │ mean_abs_error │ tol │ ok
```

`base + Σ φ_j == f(x)` on the model's **raw margin scale** (log-odds for
classifiers). If that identity holds, the engine, the background and the model
agree.

## Audit the reference set

The point of the object is that you can inspect it:

```python
attr.background_report(test)
# date │ n_admissible │ n_reference
```

`n_admissible` is how many training rows were strictly in the past of that date
(after the embargo); `n_reference` is how many were sampled. This table is the
evidence that no explanation saw the future.

## Choosing a value function

The single most consequential knob, and the one every library hides:

| `mode` | Semantics | Needs a background? | Use it when |
| --- | --- | --- | --- |
| `interventional` **(default)** | Marginal — "true to the model". Breaks feature correlations, so credit goes only to features the model actually *uses*. | **Yes** — fold-bound, past-only. | You want **actionable** attributions ("if I change this input, what moves?"), or the model is sparse and you want zero credit for unused features. |
| `conditional` | Observational — "true to the data". Spreads credit across correlated features. | Yes (reported as `shap_reference_expectation`). | You want to know **what the model's inputs indicate**, and you accept that two near-duplicate features share credit. Common in factor work, where collinearity is the norm. |
| `path_dependent` | The raw native call, no external reference. | **No** — and that is the problem. | Only with an explicit acknowledgement; see below. |

Two honest notes:

* **`conditional` and `path_dependent` run the same tree kernel.** For a tree
  ensemble, the standard exact estimator of the conditional expectation *is* the
  trees' own path-dependent traversal. PanelKit does not pretend otherwise. The
  difference is governance: `conditional` binds a fold-bound, past-only
  `TimeAwareBackground` and reports `E[f]` over it as an auditable column, while
  `path_dependent` binds nothing and must be acknowledged:

    ```python
    TimeAwareBackground(mode="path_dependent")
    # ValueError: ... PanelKit will not select it silently.
    ```

* **Interventional evaluates the model off-manifold.** Blending a row's
  `momentum` with a reference row's `price` can produce a combination that never
  occurred (high momentum, falling price). Keeping the background *past-only and
  fold-bound* is what keeps it close to the manifold; a background pooled from
  the whole dataset is both leaky *and* more off-manifold.

    The interventional engine for XGBoost / LightGBM / sklearn is `shap`'s exact
    C++ kernel (`pip install 'polars-features[explain]'`); CatBoost provides it
    natively via `reference_data` + `shap_calc_type="Independent"`.

## Choosing a background policy

| `background=` | Admissible references for `(e, t)` | Notes |
| --- | --- | --- |
| `"past"` **(default)** | Training rows with `t' < t`, minus the `embargo` most recent training dates. | The leak-safe default. |
| `"cross_sectional"` | The training fold's rows at exactly `t`; falls back to the latest strictly-past training date. | For per-date cross-sectional models. Never pools a future cross-section into a past date. |
| `"fold"` | The whole training fold, ignoring time. | What every other library does. Fold-bound but **not** past-only; warns unless you pass `i_accept_within_fold_lookahead=True`. |

Sampling is capped by `max_samples` (default 200) and seeded from
`(seed, n_admissible)` — never from the frame being explained — so the same
`(e, t)` always gets the same reference set.

Rows with no admissible past (the earliest dates) get **null** attributions and
one warning. Pass `on_empty="error"` to make that a hard failure instead.

## Group attribution: `additive` vs `joint`

Financial features are collinear, and per-feature SHAP is correspondingly
unstable. Grouping into factor buckets is the practical antidote — and it is
where a real conceptual error is usually made.

```python
groups = {"momentum": ["momentum"], "value": ["value"], "size": ["size"]}

additive = attr.group_attributions(test, groups, group_mode="additive")
joint = attr.group_attributions(test, groups, group_mode="joint")
```

| `group_mode` | Question answered | Cost |
| --- | --- | --- |
| `"additive"` (default) | "How much credit do these features receive **in total**, as a sum of individual credits?" Exact for marginal values by additivity of the value function. | Free — a `group_by` over columns you already have. |
| `"joint"` | "What would the prediction lose if this whole group were held out **together**?" A different game: each group is one coalition player. | `2^G` coalitions × background rows × explained rows. Keep `G` small. |

They coincide exactly when the groups do not interact, and diverge when they do
— which is the whole point of asking. A residual `shap_group_ungrouped` column
holds whatever the partition did not cover, so the group columns plus the base
value still reconstruct the prediction.

## Window attribution on each entity's own calendar

```python
rolling = attr.window_attributions(test, window=5, agg="abs_mean")
```

Windows are **per entity, on that entity's own observations**. An `int` window
counts that entity's own rows; a duration string (`"30d"`, `"3i"`) uses that
entity's own time values. Entities with ragged calendars — late listings, halted
names, mixed frequencies — are handled by construction, because no global time
grid is ever built.

`agg="abs_mean"` gives the classic "mean |SHAP|" importance profile through time;
`agg="sum"` preserves signed additivity within the window.

## Is that feature really drifting?

An attribution that moves is not automatically a signal. Redrawing the background
moves attributions too — especially for collinear features, where credit is
shared almost arbitrarily. PanelKit measures both on the same footing:

```python
from polars_features.explain import attribution_stability

attribution_stability(attr, test, seeds=(0, 1, 2, 3, 4, 5), threshold=2.0)
# feature │ mean_abs_attribution │ reference_sd │ temporal_sd │ drift_ratio │ verdict
```

* `reference_sd` — spread across independent draws of the reference set (same
  fold, same past-only policy, so nothing leaks). This is noise **in the
  explanation**.
* `temporal_sd` — spread of the cross-sectional mean attribution **across time**.
* `verdict` — `"regime-drift"` when `temporal_sd` exceeds `threshold ×
  reference_sd`, `"reference-noise"` when it does not, `"stable"` when the
  feature barely moves at all.

Do not tell a story about a `"reference-noise"` feature. The ratio is scale-free,
so it does not depend on the units of your target.

!!! note "Only meaningful under `mode="interventional"`"
    The path-dependent tree kernel behind `conditional` / `path_dependent`
    ignores the external reference set, so `reference_sd` is structurally zero,
    `drift_ratio` is infinite, and every feature reads as drift.
    `background_sensitivity` warns when you do this.

## Interactions (optional)

`max_order=k` gives any-order Shapley interactions via `shapiq` interop
(`pip install 'polars-features[explain]'`):

```python
from polars_features.explain import interaction_matrix, interaction_values

iv = interaction_values(
    model, test, background=attr.background_, max_order=2, max_rows=200
)
interaction_matrix(iv, agg="mean_abs")
```

The result is panel-keyed and long: `entity`, `time`, `order`, `features`
(`"|"`-joined), `interaction_value`. The background's `mode` selects the `shapiq`
engine — `interventional` → `TabularExplainer(imputer="marginal")` against the
fold-bound reference, `path_dependent` → `TreeExplainer` (TreeSHAP-IQ).

!!! note "The interactions theme"
    PanelKit uses one word — *interactions* — for higher-order structure on both
    sides of a model, with deliberately distinct keyword names so a call site is
    never ambiguous: **`max_order=k`** on the model side (feature interactions,
    here) and **`order=3|4`** on the data side (higher-order factor analysis in
    `polars_features.reduce`). Running `interaction_values` on HFA-derived
    factors is the flagship combination: "factor 2 × factor 5 synergy drove this
    forecast."

## Custom models

Any object exposing the documented hook is explained directly, with no booster
involved:

```python
def panelkit_shap_values(self, X, background):
    """Return (phi[n_rows, n_features], base[n_rows])."""
```

`background` is the reference matrix for that row group (`None` in
`path_dependent` mode). This is the extension point for a model PanelKit does not
wrap.

## Not implemented

Two items from the design are explicitly deferred, not hidden:

* **A native Arrow/Polars TreeSHAP kernel** (a `pyo3-polars` plugin). PanelKit
  dropped its Rust extension in 0.4.0 and is a pure-Python distribution; this
  would reintroduce a compiled build. The native library kernels are already
  exact and fast.
* **The sparse Möbius/Fourier (SPEX) interaction engine.** Research-stage; track
  it, do not depend on it. Use `shapiq` for interactions today.
