# Labeling

Leak-safe, Polars-native labeling for panel data, following López de Prado's
*Advances in Financial Machine Learning* (Ch. 3). The module ships the
triple-barrier method, fixed-horizon forward-return labels, and meta-labeling.

Every labeler emits a `t1` column (the event-end timestamp) in the same dtype as
the input time column. This shared span contract is the single leakage currency
consumed by purged cross-validation, so labels, sample weights, and CV purge all
derive from the same intervals and cannot desync.

## What's here

| Your problem | Entry point |
| --- | --- |
| Profit-take / stop-loss / time barrier, whichever is hit first | `triple_barrier` |
| A fixed-horizon forward return, sign or magnitude | `fixed_horizon` |
| Learn *when* to act on a primary signal | `meta_label` |

## See also

- [Labeling guide](../user-guide/labeling.md) — the narrative walkthrough.
- [`validation`](validation.md) — the purged splits that consume the `t1` spans.
- [`factor`](factor.md) — evaluating a signal cross-sectionally instead.

## API

::: panelary.label
