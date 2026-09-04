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

::: polars_features.catch22
