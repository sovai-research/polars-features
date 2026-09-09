# Feature registry (`FeatureSpec`)

`panelary.registry` is the single source of truth for **operators**: the metadata layer that
turns a loose collection of Polars expressions into an auditable, machine-introspectable
catalogue. Two objects do the work. `FeatureSpec` is a frozen, fully typed description of one
operator, and `FeatureRegistry` is the process-wide singleton (exported as `registry`) that
collects them.

A `FeatureSpec` carries more than a call signature. It records the **provenance and safety
contract**: which expression namespace the operator lives in, its input and output shape, its
`panel_safe` / `leakage_safe` declarations, where the implementation came from and under what
license. That last pair is how Panelary demonstrates it is an independent, permissively
licensed implementation rather than a copy of a copyleft upstream — `audit()` is a mechanical
check, not a claim in a README.

The registry is deliberately **decoupled from the implementations**. Namespaces register their
specs at import time via `register_feature`, but `panelary/registry.py` imports nothing from
the rest of Panelary, which keeps it cheap and side-effect-free to import from anywhere.

## What's here

| Your problem | Entry point |
| --- | --- |
| Describe one operator, with provenance and safety | `FeatureSpec` |
| Register an operator as an import side effect | `register_feature` |
| The process-wide catalogue | `registry` (a `FeatureRegistry`) |
| Look one operator up by name | `registry.get(name)` |
| Every spec, or every spec in one namespace | `registry.all()`, `registry.by_namespace(ns)` |
| Which namespaces exist? | `registry.namespaces()` |
| Serialise the catalogue | `registry.to_records()`, `registry.to_llms_txt()` |
| Check for missing safety metadata or copyleft | `registry.audit()` |
| Reset (tests only) | `registry.clear()` |
| The permissive-license allowlist and the tier vocabulary | `PERMISSIVE_LICENSES`, `VALID_TIERS` |

## What is registered today

`registry.namespaces()` returns **four** namespaces holding **56** operators in total:

| Namespace | Count | Scope | Examples |
| --- | ---: | --- | --- |
| `.ts` | 42 | Per-series time-series characteristics | `absolute_energy`, `max_drawdown`, `longest_winning_streak`, `cid_ce` |
| `.xs` | 7 | Cross-sectional, per date | `cs_zscore`, `demean`, `neutralize`, `quantile_bin`, `rank`, `standardize`, `winsorize` |
| `.factor` | 4 | Signal evaluation across the cross-section | `forward_return`, `ic`, `orthogonalize`, `portfolio_sort` |
| `.panel` | 3 | Per-entity, causal | `frac_diff`, `rs_vol`, `zscore` |

The `.factor` namespace is easy to miss — it is registered alongside the other three and is
the expression-level counterpart to [`panelary.factor`](factor.md).

## Reading `audit()`

`audit()` returns a dict of four lists. On the current tree three of them are empty
(`missing_leakage_safe`, `missing_provenance`, `non_permissive_license` — all 56 specs are
Apache-2.0), and one is not:

```python
>>> from panelary.registry import registry
>>> registry.audit()["missing_panel_safe"]
['ic', 'orthogonalize', 'portfolio_sort', 'demean', 'rank', 'standardize']
```

**These six are not a defect.** `panel_safe` means "respects entity boundaries", and all six
are deliberately cross-sectional: three `.factor` operators and three `.xs` operators whose
entire purpose is to compare entities *against each other* within a date. They declare
`panel_safe=False` honestly, and they remain `leakage_safe=True` because they only ever read a
single date's rows. `audit()` surfaces them so the exception stays visible rather than
becoming folklore — the flag to look for in the six is that each is scoped by `.over(time_col)`,
never `.over(entity_col)`.

## See also

- [`PanelFrame`](panel-frame.md) — the type these operators consume.
- [The two-tier API](../concepts/two-tier-api.md) — namespaces versus the fitted-transformer tier.
- [Leak-safety](../concepts/leak-safety.md) — what `panel_safe` and `leakage_safe` mean.
- [`factor`](factor.md) — the function-level surface behind the `.factor` namespace.

## API

::: panelary.registry
