# Attribution (`explain`)

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

See [Feature attribution](../user-guide/attribution.md) for the workflow and
[The background-set leak](../concepts/attribution-leakage.md) for why the
reference set is where attribution leaks.

::: panelary.explain
