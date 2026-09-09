# catch22

A clean-room, Polars-native implementation of the **catch22** feature set
(Lubba et al. 2019) — 22 canonical time-series characteristics selected from the
~7700 features of *hctsa* for high classification performance and low mutual
redundancy. The implementation is written directly from the published
algorithmic descriptions and does **not** vendor any GPL-licensed
`pycatch22`/`hctsa` code.

Every feature z-scores its input series before computing (the standard catch22
variant), and the Polars entry point `catch22_features` runs a **per-entity**
`group_by`, so the computation is leak-safe by construction: each feature only
ever sees the values of its own series.

## What's here

| Entry point | Purpose |
| --- | --- |
| `catch22_features` | All 22 features per entity, as a tidy Polars frame |
| The 22 individual expressions | Compose a subset yourself inside `.over(entity_col)` |

## See also

- [`feature_extractors`](feature-extractors.md) — the wider ~70-feature tsfresh-style library.
- [Feature Engineering guide](../user-guide/features.md) — the narrative walkthrough.

## API

::: panelary.catch22
