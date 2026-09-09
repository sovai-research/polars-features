# Attribution

Leak-safe, panel-aware feature attribution. Panelary does **not** reimplement
SHAP: exact TreeSHAP already ships inside XGBoost (`pred_contribs`), LightGBM
(`pred_contrib`) and CatBoost (`ShapValues`), the reference interventional engine
ships inside `shap`, and `shapiq` is the maintained hub for any-order
interactions. What no library ships — and what this subpackage is — is the
**reference set as a first-class, fold-bound, past-only object**.

`TimeAwareBackground` is the moat: for an observation at `(e, t)` the admissible
reference rows are `{(e', t') : t' < t}` drawn from the training fold only, with
an optional embargo and deterministic seeded sampling. `TreeAttributor` is a
`PanelTransformer`, so `fit(train)` freezes that reference and `transform(test)`
learns nothing — attribution inherits the split's leak-safety instead of quietly
undoing it.

Everything is Polars-native: `shap_<feature>` columns keyed by `(entity, time)`,
or a tidy long frame via `output="long"`.

Optional dependencies (all lazily imported): the boosters themselves,
`shap` for the interventional engine on non-CatBoost models, and `shapiq` for
interactions — `pip install 'panelary[explain]'`.

## What's here

| Your problem | Entry point |
| --- | --- |
| A past-only, fold-bound reference set | `TimeAwareBackground` |
| Per-row SHAP columns from a fitted booster | `TreeAttributor` |
| A tidy long frame instead of wide columns | `TreeAttributor(output="long")` |

## See also

- [Feature Attribution guide](../user-guide/attribution.md) — the workflow.
- [The background-set leak](../concepts/attribution-leakage.md) — why the reference set leaks.
- [`models`](models.md) — the panel estimators being explained.

## API

::: panelary.explain
