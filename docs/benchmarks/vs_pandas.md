# Panelary vs pandas — panel feature generation

Panel feature engineering (many entities × time) is Panelary's core workload. These are
**measured, reproducible** numbers, not estimates: each computation is run **both** ways and
the results are asserted equal (within `1e-6`) *before* the timing is reported, so no speedup
comes from computing something cheaper.

Reproduce: `python benchmarks/bench_vs_pandas.py`

Environment: polars 1.44.1, pandas 3.0.2, Python 3.13, 15 threads (Apple Silicon). Best of 3 runs.

| Workload | 0.5M rows | 2.5M rows | Correctness |
|---|---:|---:|:--:|
| Rolling z-score, per entity (window 21) | **18.8×** | **10.7×** | match |
| Cross-sectional rank, per date | **14.9×** | **17.8×** | match |
| Bulk: 10 tsfresh-style features per entity | **16.6×** | **12.5×** | match |

Absolute times at 2.5M rows: rolling z-score 0.436s → 0.041s; cross-sectional rank
0.548s → 0.031s; 10-feature bulk 0.387s → 0.031s.

## Why it's faster

- **Native, multi-threaded expressions.** The whole computation stays in the Polars engine
  across all cores; pandas runs per-group Python for anything without a vectorised form
  (workload 3 — the realistic feature-engineering case — is exactly where tsfresh-style
  per-group Python loops dominate).
- **No per-group materialisation.** `.over(entity)` / `.over(time)` and a single
  `group_by().agg()` avoid the Python-level `groupby.apply` overhead.
- **Lazy, one pass.** `extract_features` compiles all requested features into one lazy plan.

## Honesty notes

- Compared against **pandas 3.0**, which has a fast groupby; older pandas shows larger gaps.
- The bulk comparison gives pandas its *best* path (a single `apply` returning a `Series`),
  not the slowest per-feature-lambda form — so the real-world gap is typically wider.
- Speedups grow further versus `tsfresh` (row-by-row Python) and shrink toward parity on
  trivially-vectorised single ops where pandas is already near-native.
