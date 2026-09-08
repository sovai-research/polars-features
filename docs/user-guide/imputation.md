# Imputation

Missing values are the norm in panel data: entities enter and leave, sensors
drop out, a vendor stops reporting a field for a quarter. How you fill those
gaps quietly decides whether your features are honest. PanelKit ships a
leak-safe, model-based imputer built on **CAFE** (Causal Adaptive Factor
Estimation) that is *strictly point-in-time*: every filled cell uses only past
and contemporaneous information within its own entity.

## Why point-in-time imputation matters

The imputers you reach for by reflex are almost all forward-looking on a panel,
and the leakage is invisible because the filled frame looks perfectly clean:

- **Backward fill (`bfill`)** copies a *future* observation back over a gap. The
  filled cell literally contains a value that had not happened yet at that
  timestamp.
- **Linear `interpolate`** blends the observations on *both* sides of a gap, so
  an interpolated cell at time `t` depends on the value at some `t + k`.
- **Global / group mean or median** computes the statistic over the *whole*
  series (train and future alike), then writes it back into early rows. The mean
  of the entire column is a summary of data from the future.

Fit any of these on a full panel, then evaluate with walk-forward
cross-validation, and your "past" rows have been contaminated with information
from the validation window. Metrics look great in the notebook and collapse in
production. On a panel it is worse than on a single series, because a careless
imputer can also borrow *across entities* and blur entity boundaries.

CAFE is designed around the same leak-safe moat as the rest of PanelKit. Each
filled cell is reconstructed from information available *up to and including*
that `(entity, time)` — never after it — and imputation respects entity
boundaries. That is why both guarantees on the transformer are honestly `True`:

- `panel_safe = True` — fills never mix rows across entities.
- `leakage_safe = True` — no cell ever uses information from its own future.

Appending future rows to an entity therefore never changes an earlier cell's
imputed value (or its by-products), which is exactly the property that makes an
imputer safe to run inside cross-validation.

!!! note "Optional dependency"
    The imputer wraps the optional `cafe` package. Install it with
    `pip install polars_features[cafe]`. The rest of PanelKit imports fine
    without it; you only need it when you actually impute with CAFE.

## `cafe_impute` — the functional transformer

`cafe_impute` is a Polars-native panel transformer. Point it at a long panel
frame whose **first two columns are the entity and time keys**; every numeric
feature column is filled and the keys plus any non-numeric columns pass through
untouched, in their original order.

```python
import polars as pl
from polars_features.preprocessing import cafe_impute

# Long panel: entity, time, then numeric features (with gaps).
X = pl.DataFrame({
    "entity": ["A"] * 6 + ["B"] * 6,
    "time": list(range(6)) * 2,
    "sales":   [10.0, None, 12.0, 13.0, None, 15.0, 20.0, 21.0, None, 23.0, 24.0, None],
    "traffic": [100.0, 110.0, None, 130.0, 140.0, None, 200.0, None, 220.0, None, 240.0, 250.0],
})

filled = X.pipe(cafe_impute()).collect()
print(filled)
```

```text
shape: (12, 4)
┌────────┬──────┬───────────┬────────────┐
│ entity ┆ time ┆ sales     ┆ traffic    │
│ str    ┆ i64  ┆ f64       ┆ f64        │
╞════════╪══════╪═══════════╪════════════╡
│ A      ┆ 0    ┆ 10.0      ┆ 100.0      │
│ A      ┆ 1    ┆ 11.0      ┆ 110.0      │
│ A      ┆ 2    ┆ 12.0      ┆ 125.000017 │
│ A      ┆ 3    ┆ 13.0      ┆ 130.0      │
│ A      ┆ 4    ┆ 14.333333 ┆ 140.0      │
│ A      ┆ 5    ┆ 15.0      ┆ 150.000029 │
│ B      ┆ 0    ┆ 20.0      ┆ 200.0      │
│ B      ┆ 1    ┆ 21.0      ┆ 210.0      │
│ B      ┆ 2    ┆ 22.500002 ┆ 220.0      │
│ B      ┆ 3    ┆ 23.0      ┆ 235.000017 │
│ B      ┆ 4    ┆ 24.0      ┆ 240.0      │
│ B      ┆ 5    ┆ 25.333339 ┆ 250.0      │
└────────┴──────┴───────────┴────────────┘
```

`cafe_impute` returns a `LazyFrame`, so `.pipe(...)` composes with the rest of
your Polars pipeline and you `.collect()` when you are ready.

### Parameters

```python
cafe_impute(
    engine="joint",            # "joint" | "per_entity"
    add_uncertainty=False,     # emit <col>__cafe_sigma
    add_recoverability=False,  # emit <col>__cafe_recoverability
    add_anomaly=False,         # emit cafe_anomaly (one column, per row)
    add_missingness=False,     # emit <col>__cafe_was_imputed
    columns=None,              # restrict to a subset of numeric columns
)
```

- **`engine`** — the panel imputation strategy.
    - `"joint"` (default, validated) pools the contemporaneous cross-section, so
      entities that move together help fill each other's gaps *at the same
      timestamp* (still strictly point-in-time, never across time).
    - `"per_entity"` imputes each entity's series completely independently.
- **`columns`** — restrict imputation (and any by-products) to these numeric
  feature columns. Other numeric columns keep their **original** values,
  including their nulls. Defaults to every numeric feature column.

### By-product columns

Imputation destroys information — once a gap is filled you can no longer tell
which cells were real, or how confident the fill was. Each flag below adds that
information back as an extra column instead of hiding it:

| Flag | Column(s) emitted | Meaning |
| --- | --- | --- |
| `add_uncertainty` | `<col>__cafe_sigma` | Per-cell posterior standard deviation of the fill. `NaN` where the value was observed. |
| `add_recoverability` | `<col>__cafe_recoverability` | Per-cell recoverability certificate in `[0, 1]` — how identifiable that cell was. `NaN` where observed. |
| `add_anomaly` | `cafe_anomaly` | A single per-**row** outlier score in `[0, 1]` (0 = perfect fit, 1 = strong outlier), causal per entity. |
| `add_missingness` | `<col>__cafe_was_imputed` | Boolean flag preserving the original missing pattern — `true` exactly where the cell was filled. |

The uncertainty / recoverability / anomaly by-products come from a strictly
causal, per-entity traced pass, so they carry the same point-in-time guarantee
as the fill itself.

## Worked example: fill plus by-products

Ask for the fill *and* the confidence and provenance columns in one call:

```python
from polars_features.preprocessing import cafe_impute

out = X.pipe(
    cafe_impute(
        add_uncertainty=True,
        add_recoverability=True,
        add_missingness=True,
    )
).collect()

print(
    out.select(
        "entity", "time", "sales",
        "sales__cafe_sigma",
        "sales__cafe_recoverability",
        "sales__cafe_was_imputed",
    )
)
```

```text
shape: (12, 6)
┌────────┬──────┬───────────┬───────────────────┬────────────────────────────┬─────────────────────────┐
│ entity ┆ time ┆ sales     ┆ sales__cafe_sigma ┆ sales__cafe_recoverability ┆ sales__cafe_was_imputed │
│ str    ┆ i64  ┆ f64       ┆ f64               ┆ f64                        ┆ bool                    │
╞════════╪══════╪═══════════╪═══════════════════╪════════════════════════════╪═════════════════════════╡
│ A      ┆ 0    ┆ 10.0      ┆ NaN               ┆ NaN                        ┆ false                   │
│ A      ┆ 1    ┆ 11.0      ┆ 179.336772        ┆ 0.565479                   ┆ true                    │
│ A      ┆ 2    ┆ 12.0      ┆ NaN               ┆ NaN                        ┆ false                   │
│ A      ┆ 3    ┆ 13.0      ┆ NaN               ┆ NaN                        ┆ false                   │
│ A      ┆ 4    ┆ 14.333333 ┆ 93.422528         ┆ 0.485155                   ┆ true                    │
│ A      ┆ 5    ┆ 15.0      ┆ NaN               ┆ NaN                        ┆ false                   │
│ B      ┆ 0    ┆ 20.0      ┆ NaN               ┆ NaN                        ┆ false                   │
│ B      ┆ 1    ┆ 21.0      ┆ NaN               ┆ NaN                        ┆ false                   │
│ B      ┆ 2    ┆ 22.500002 ┆ 253.45163         ┆ 0.501038                   ┆ true                    │
│ B      ┆ 3    ┆ 23.0      ┆ NaN               ┆ NaN                        ┆ false                   │
│ B      ┆ 4    ┆ 24.0      ┆ NaN               ┆ NaN                        ┆ false                   │
│ B      ┆ 5    ┆ 25.333339 ┆ 163.685478        ┆ 0.476829                   ┆ true                    │
└────────┴──────┴───────────┴───────────────────┴────────────────────────────┴─────────────────────────┘
```

Read a filled row together with its by-products: `sales` at `(A, 1)` was filled
to `11.0`, `sales__cafe_was_imputed` is `true` there (and `false` on every
observed cell), and the `sigma` / `recoverability` columns quantify how much to
trust that fill. On observed cells the by-products are `NaN` — there was nothing
to impute.

### Imputing a subset of columns

Use `columns=` to fill only some features and leave the rest exactly as they
came in (nulls included):

```python
subset = X.pipe(cafe_impute(engine="per_entity", columns=["sales"])).collect()
# `sales` is filled; `traffic` keeps its original values and its nulls.
```

## `CafeImputer` — the pipeline transformer

For an sklearn-shaped `fit` / `transform` workflow (and to slot into PanelKit's
estimator layer), use `CafeImputer`. It is a `PanelTransformer`, so it accepts a
bare `pl.DataFrame` / `pl.LazyFrame` plus `entity=` / `time=` keys (or a
`PanelFrame`), and returns a `PanelFrame`.

```python
from polars_features.imputation import CafeImputer

imputer = CafeImputer(engine="joint", entity="entity", time="time")
imputer.fit(X)                      # records the feature columns; learns no fold state
panel = imputer.transform(X)        # returns a PanelFrame
filled = panel.collect()            # -> plain polars frame

imputer.panel_safe, imputer.leakage_safe   # (True, True)
```

The constructor signature is keyword-only:

```python
CafeImputer(*, engine="joint", entity=None, time=None)
```

`engine` behaves exactly as in `cafe_impute`. `entity` / `time` are the default
panel keys used when you pass a bare frame; they are ignored (and rejected if
they conflict) when you pass a `PanelFrame` that already carries its keys. A
`fit_transform(X)` convenience is available too.

!!! note
    `fit` is a near no-op: CAFE learns no cross-fold parameters because it is
    fully point-in-time at transform time. `fit` only records which feature
    columns exist; the strictly-causal fill happens in `transform`.

## `impute(method="cafe")` — the string alias

If you already route imputation through the generic `impute` transformer, the
`"cafe"` method is a drop-in alias for `cafe_impute()` with its defaults. These
two produce identical frames:

```python
from polars_features.preprocessing import impute, cafe_impute

a = X.pipe(impute(method="cafe")).collect()
b = X.pipe(cafe_impute()).collect()
assert a.equals(b)
```

The alias keeps every other `impute` method (`"mean"`, `"median"`, `"fill"`,
`"ffill"`, `"bfill"`, `"interpolate"`, or a constant) unchanged — it simply
adds a leak-safe, model-based option under the same door. Note that the alias
uses `cafe_impute()`'s defaults, so for by-products or `engine="per_entity"`
call `cafe_impute(...)` directly.

## Composing inside cross-validation

Because CAFE is strictly point-in-time, an earlier `(entity, time)` cell is
identical whether or not future rows exist. That means you can **fit the imputer
on the training fold and transform each fold** without leaking the validation
window back into training features:

```python
from polars_features.imputation import CafeImputer

# One walk-forward split: train on the past, validate on the future.
cutoff = 3
train = X.filter(pl.col("time") < cutoff)
valid = X.filter(pl.col("time") >= cutoff)

imputer = CafeImputer(entity="entity", time="time").fit(train)
train_filled = imputer.transform(train).collect()
valid_filled = imputer.transform(valid).collect()
```

Contrast this with `bfill` or a global mean, where filling the training rows
would pull in values from `valid`, silently inflating your cross-validated
scores. With CAFE the guarantee is structural (`leakage_safe = True`), not a
matter of remembering to be careful.

## See also

- [Preprocessing](preprocessing.md) — the other panel transformers
  (`diff`, `detrend`, `scale`, `roll`, …) that compose with imputation.
- API reference: `polars_features.preprocessing.cafe_impute`,
  `polars_features.preprocessing.impute`, and
  `polars_features.imputation.CafeImputer`.
