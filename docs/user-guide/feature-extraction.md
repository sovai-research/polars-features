# Feature Extraction

!!! info "This page has moved"
    Feature engineering in Panelary now lives in the **[Feature Engineering
    guide](features.md)**. That page covers the `.ts` extractors, the bulk
    `extract_features` API, catch22, per-entity vs. cross-sectional features, and how to
    discover operators through the registry. This page is a short pointer kept for
    backwards-compatible links.

Panelary exposes dozens of time-series feature extractors through a custom `ts`
(time-series) Polars namespace. Every extractor is an ordinary Polars expression, so it
works on a `Series`, a `LazyFrame`, and — most importantly for panel data — **per entity**
inside a `group_by`. Because each series is featurised independently, the results are
leak-safe by construction.

The `ts` namespace is registered automatically when you import the package:

```python
import numpy as np
import polars as pl
import panelary as pn  # registers the .ts / .panel / .xs namespaces
```

## A quick taste

```python
rng = np.random.default_rng(0)

# Works on LazyFrames with full query optimization
features = (
    pl.LazyFrame({
        "index": list(range(10)),
        "value": rng.normal(0, 1, 10),
    })
    .select(
        binned_entropy=pl.col("value").ts.binned_entropy(bin_count=5),
        longest_streak_above_mean=pl.col("value").ts.longest_streak_above_mean(),
        cid_ce=pl.col("value").ts.cid_ce(normalize=True),
    )
    .collect()
)

# Extract features per entity on a stacked panel using group_by
panel = pl.DataFrame({
    "ticker": ["A"] * 6 + ["B"] * 6,
    "date":   list(range(6)) * 2,
    "ret":    rng.normal(0, 1, 12),
})
per_entity = (
    panel.group_by("ticker", maintain_order=True)
    .agg(
        absolute_energy=pl.col("ret").ts.absolute_energy(),
        max_drawdown=pl.col("ret").ts.max_drawdown(),
        return_skew=pl.col("ret").ts.return_skew(),
    )
)
```

!!! warning "Named extractors, not raw reducers"
    Valid feature names are things like `absolute_energy`, `mean_abs_change`,
    `count_above_mean`, `longest_streak_above_mean`, `cid_ce`, `variation_coefficient`,
    `max_drawdown`, `return_skew`, `root_mean_square`, and `absolute_sum_of_changes` —
    **not** plain `mean` / `std` (use Polars' own `pl.col(...).mean()` for those).

## Batch extraction

For a one-call, tsfresh-style bulk extractor that produces a wide feature frame (one row
per entity), use `extract_features`:

```python
wide = pn.extract_features(
    panel, entity="ticker", time="date", column="ret",
    features=["absolute_energy", "mean_abs_change", "cid_ce"],
)
```

## Discovering what is available

Enumerate the registered extractors (and their parameters and safety flags) through the
registry rather than hard-coding names:

```python
from panelary.registry import registry

ts_features = [spec.name for spec in registry.by_namespace("ts")]
print(len(ts_features), ts_features[:5])
```

## Full guide

See **[Feature Engineering](features.md)** for:

- the `.ts` scalar extractors and their parametrised / list-valued variants,
- the `extract_features` batch API and its column-naming rules,
- the clean-room **catch22** feature set (`catch22_features`, `catch22_all`),
- **per-entity (`.panel`)** vs. **cross-sectional (`.xs`)** features,
- programmatic discovery and **feature-name provenance** via the registry,
- a realistic worked example producing a wide feature frame.

See also the [feature-extractors API reference](../api-reference/feature-extractors.md).
