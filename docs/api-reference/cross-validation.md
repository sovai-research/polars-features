# Cross-validation splitters

`panelary.cross_validation` holds the **simple, chronological** panel splitters: a single
train/test cut and the two classic rolling-origin schemes. Each returns a callable that takes a
panel frame and yields `(train, test)` frames, split **within each entity** in time order, so a
test fold is always strictly later than its training fold for every entity.

These splitters purge nothing. They are the right tool when observations are point-in-time and
non-overlapping. The moment labels *span* time — a triple-barrier label, a multi-period forward
return — an adjacent train row overlaps its test fold and the split leaks; use the purged and
embargoed splitters in [`validation`](validation.md) instead.

## What's here

| Splitter | Shape |
| --- | --- |
| `train_test_split` | One chronological cut per entity, by count or fraction |
| `expanding_window_split` | Growing train window, fixed test window (rolling origin) |
| `sliding_window_split` | Fixed-length train window, fixed test window |

## See also

- [`validation`](validation.md) — purged / embargoed CV, CPCV, and backtest paths.
- [Validation guide](../user-guide/validation.md) — choosing between the two families.

## API

::: panelary.cross_validation
