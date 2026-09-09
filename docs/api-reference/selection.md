# Feature selection

Leak-safe feature selection for panels. Every routine is computed only on the
rows it is passed (in-fold), so importances and selections are free of
look-ahead when driven through a purged CV splitter.

## What's here

- `mrmr` — minimum-Redundancy-Maximum-Relevance selection, computed in-fold.
- `mda` — Mean-Decrease-Accuracy (permutation) importance evaluated through a
  purged CV splitter.
- `mdi` — Mean-Decrease-Impurity importance from a fitted tree ensemble.
- `MRMRSelector` — a `PanelTransformer` wrapping `mrmr` for use as a `"select"`
  step in a `Pipeline`.

## See also

- [Feature Selection guide](../user-guide/selection.md) — the narrative walkthrough.
- [`validation`](validation.md) — the purged splitter `mda` evaluates through.
- [`explain`](explain.md) — attribution, once a model is fitted.

## API

::: panelary.select
