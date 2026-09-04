# Feature selection

Leak-safe feature selection for panels. Every routine is computed only on the
rows it is passed (in-fold), so importances and selections are free of
look-ahead when driven through a purged CV splitter.

- `mrmr` — minimum-Redundancy-Maximum-Relevance selection, computed in-fold.
- `mda` — Mean-Decrease-Accuracy (permutation) importance evaluated through a
  purged CV splitter.
- `mdi` — Mean-Decrease-Impurity importance from a fitted tree ensemble.
- `MRMRSelector` — a `PanelTransformer` wrapping `mrmr` for use as a `"select"`
  step in a `Pipeline`.

::: polars_features.select
