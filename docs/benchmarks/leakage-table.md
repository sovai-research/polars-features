# Borrowed accuracy — a scored table of preprocessing steps

Every preprocessing step can be written two ways: **permissively**, the way a practitioner
reaches for it, and **point-in-time**, constrained to information available at prediction
time. The out-of-sample gap between the two is the accuracy *borrowed* from data the method
will not have when it runs for real.

These are **measured** numbers. Each step is built both ways, both versions are scored on
*exactly the same rows* with *exactly the same* purged and embargoed folds, and every
permissive construction is independently cross-checked against
`panelary.testing.assert_no_lookahead` — so a row reports two instruments, not one, and the
table says where they disagree.

Reproduce: `python benchmarks/bench_leakage_table.py --long`

Environment: polars 1.44.2, numpy 2.5.3, Python 3.13, Apple Silicon (15 threads).
Panel: 1,250 entities × 120 steps = 150,000 rows, 9 seeds, `PurgedKFold(5, horizon=1, embargo=2)`.
Score: pooled out-of-sample R² of a numpy ridge on the step's output alone.

## The table

| Preprocessing step | permissive | point-in-time | **gap** | gap min | gap max | verifier flags permissive? |
|---|---:|---:|---:|---:|---:|:--:|
| _[control] same op both sides_ | _0.1223_ | _0.1223_ | _0.0000_ | _0.0000_ | _0.0000_ | _clean_ |
| Period aggregation, broadcast back | 0.1229 | 0.0247 | **+0.0974** | 0.0932 | 0.1023 | FLAG |
| Lag with a negative shift | 0.1896 | 0.1223 | **+0.0668** | 0.0649 | 0.0674 | FLAG |
| Rolling mean, centred vs trailing | 0.1475 | 0.0855 | **+0.0603** | 0.0575 | 0.0638 | FLAG |
| Target encoding of a categorical | 0.0544 | −0.0029 | **+0.0564** | 0.0552 | 0.0574 | FLAG |
| Whole pipeline fitted before the split | 0.1817 | 0.1357 | **+0.0469** | 0.0456 | 0.0474 | FLAG |
| Min–max scaling, per entity | 0.1241 | 0.0833 | **+0.0392** | 0.0374 | 0.0445 | FLAG |
| Robust scaling (median/IQR), per entity | 0.1531 | 0.1153 | **+0.0374** | 0.0337 | 0.0389 | FLAG |
| Standardisation (z-score), per entity | 0.1522 | 0.1150 | **+0.0368** | 0.0315 | 0.0383 | FLAG |
| Ranking, global vs cross-sectional | 0.1352 | 0.1266 | +0.0086 | 0.0042 | 0.0145 | FLAG |
| Backward vs forward fill | 0.1511 | 0.1441 | +0.0069 | 0.0058 | 0.0084 | FLAG |
| Linear interpolation | 0.1509 | 0.1441 | +0.0069 | 0.0063 | 0.0075 | FLAG |
| Rolling std, centred vs trailing | 0.0118 | 0.0078 | +0.0041 | 0.0035 | 0.0053 | FLAG |
| Differencing, forward vs backward | 0.0071 | 0.0057 | +0.0020 | 0.0010 | 0.0032 | FLAG |
| Winsorisation at global quantiles | 0.1478 | 0.1478 | +0.0000 | −0.0000 | 0.0001 | FLAG |
| Scaling before the train/test split | 0.1508 | 0.1508 | +0.0000 | 0.0000 | 0.0000 | FLAG |
| Mean imputation | 0.1319 | 0.1324 | −0.0004 | −0.0012 | 0.0005 | FLAG |

The **gap** column is a *paired* statistic: the median over seeds of the per-seed difference,
which is why it need not equal the difference of the two medians. `gap min` / `gap max` are
that difference's range over the nine seeds.

Every point-in-time construction the verifier can check — the twelve row-local steps plus
the control — comes back **clean**. The four fitted steps have no whole-panel point-in-time
form (they are defined relative to a fold), so the verifier has nothing to check and the
cell is reported as n/a rather than as a pass.

## What the numbers say

**The ordering is not the one the folklore predicts.** "Scaling before the train/test split"
is the canonical leakage sin, and on this DGP it borrows **exactly zero**. Aggregating over a
calendar period and broadcasting the result back over the days that made it up — a step
nobody writes a blog post about — reports five times the honest model's skill (R² 0.0247 →
0.1229). If you are spending your review budget on scalers, you are spending it in the wrong
place.

**Three clusters.**

- **Time-travel in the row itself** (period aggregation, negative shift, centred windows):
  0.06–0.10. These reach forward across rows and take the answer. Nothing about the model
  or the fold structure protects you.
- **Whole-history statistics, per entity** (z-score, min–max, robust scaling): a tight
  0.037–0.039 for all three. The leak is not the *statistic* — it is that the statistic tells
  you where the series is going to end up. Which of mean/std, min/max or median/IQR you pick
  changes the number by less than the seed spread.
- **Genuinely near-zero** (winsorisation, split-before-scaling, mean imputation,
  differencing): ≤ 0.002. Real information flows — the verifier flags all of them — but it
  buys nothing.

**Scaling borrows nothing here for a reason you can prove, not a fluke.** The ridge
standardises its features on the training fold, which makes it exactly invariant to any
global affine transform of a feature. A global scaler *is* a global affine transform, so the
two runs produce identical predictions and the gap is identically zero. The practical
corollary generalises past this benchmark: **global feature scaling can only borrow accuracy
from a learner that is not scale-invariant** — a regulariser with a fixed penalty on unscaled
features, a distance-based model, a net with a fixed initialisation. It is not that the
textbook is wrong about the information flow; it is that for a large class of models the
flow is worth nothing.

Note the contrast in the table: *per-entity* scaling borrows 0.037 while *global* scaling
borrows 0.000. Per-entity scaling is a different affine map for each entity, so the model
cannot undo it, and each map encodes that entity's future. In panel data the harmful version
of "standardise your features" is the one practitioners are most likely to write.

**Target encoding is the one that inverts the sign.** Honest target encoding, refit per fold,
scores **−0.0029** — worse than predicting the mean. The same encoder fitted on everything
scores +0.0544. The whole apparent signal is the encoder reading test-fold labels back to
itself. This is the most dangerous shape in the table: not a good feature made better, but a
useless feature made to look good.

**Relative inflation is not absolute inflation.** Centred rolling *std* borrows only 0.0041 —
but its honest score is 0.0078, so the permissive form reports 1.5× the real skill. Period
aggregation is 5.0×. A small gap on a weak feature can still be most of what you think you
have.

## Where the two instruments disagree

`assert_no_lookahead` flagged **all sixteen** permissive constructions and cleared **all
thirteen** checkable point-in-time ones (twelve steps plus the control). There is no row
where the gap found a leak the verifier missed, and none where the verifier rejected a
construction that is in fact point-in-time. On direction, they agree everywhere.

On magnitude, they agree nowhere, and that is the useful finding. Four flagged steps have
gaps at or below 0.002, and one of those (mean imputation) has a gap that is *negative* at
the median. So:

> `assert_no_lookahead` answers **"does information flow backwards?"** — and it answers it
> exactly. Borrowed accuracy answers **"is that flow worth anything?"** — and the answers are
> not correlated. A yes/no verifier cannot rank your pipeline's problems, and a gap cannot
> tell you whether a step is *correct*.

This is the argument for shipping both, and for `causalize` refusing on the verifier's
answer rather than on the gap: a step whose borrowed accuracy is zero on your data is still
a step whose output depends on rows you will not have, and the next dataset is not this one.

Two instrument artefacts worth recording, both found by running the cross-check:

- **A `0/0` reads as a leak.** Polars orders `NaN` above every float, so the verifier's
  `|a − b| > tol` test fires when it compares `NaN` to `NaN`. An expanding min–max scaler is
  perfectly causal but produces `0/0` at an entity's first observation, and was reported as
  leaky until the denominator was guarded. If the verifier flags something you are sure of,
  check for `NaN` before you check for look-ahead.
- **`cum_sum` is null-propagating.** `cum_sum() / cum_count()` — the obvious spelling of an
  expanding mean — returns null at exactly the rows an imputer needs to fill, so a
  point-in-time imputer built that way silently degenerates into "change nothing". The
  benchmark's first draft measured a gap of exactly 0.0000 for mean imputation for that
  reason. Count non-nulls separately (`is_not_null().cum_sum()`) over a zero-filled sum.

## Decomposing a pipeline

The "whole pipeline fitted before the split" row is four interacting stages reported as one
number. `panelary.leakage.borrowed_accuracy` turns it into a decomposition — all
`2⁴ = 16` subsets evaluated, exact Shapley values, nothing sampled:

| Stage | Shapley value |
|---|---:|
| Target encoder | **+0.0455** |
| Imputer | +0.0000 |
| Winsoriser | +0.0000 |
| Scaler | +0.0000 |
| **Total borrowed accuracy** | **+0.0456** |

`sum(φ) − total = 0.0e+00` — the efficiency axiom holds exactly, which is what makes this a
decomposition rather than a set of ablations. The attribution also reproduces the per-step
table from the other direction: the stages it prices at zero are the three whose standalone
rows are zero, and the stage it blames is the one whose standalone row is +0.056.

## Honesty notes

Read these before quoting a number.

- **Every figure is conditional on this data-generating process.** The panel has a persistent
  AR(1) idiosyncratic component (ρ = 0.90), a per-entity random-walk level, a persistent
  stochastic volatility, and a categorical with ~40 rows per level. Those four choices *are*
  the table: persistence is what makes a future observation informative, drift is what makes
  whole-sample statistics leak, and cardinality is what makes target encoding catastrophic.
  Change any of them and the ordering moves. Treat the table as a *target list* — which
  operations deserve a compiler rule — not as a set of constants.
- **The boring rows are results, not failures.** Four steps measure ≈ 0, one of them
  slightly negative. Nothing was tuned to make them look alarming, and they are reported at
  the same precision as the rest. A DGP on which all sixteen rows looked damaging would be a
  DGP built to flatter the argument.
- **Two rows are too weak to interpret.** Centred rolling std (0.0118 permissive) and forward
  differencing (0.0071) barely predict at all on this DGP. Their gaps are consistent across
  seeds and so are real, but a gap between two near-zero scores should not be read as a
  ranking against the strong rows.
- **A single seed is not evidence.** The `gap min` / `gap max` columns exist because of a real
  failure: an earlier draft parameterised the volatility AR(1) by its *innovation* sd rather
  than its process sd, so at ρ = 0.95 the process sd was 3.2× larger than intended, one seed
  in nine produced an entity whose series ran to five figures, and that seed alone moved the
  centred-rolling-mean gap to +0.82. The median across nine seeds hid it; the range did not.
  Report ranges.
- **The design matrix is the step's output and nothing else**, so each row isolates one
  operation. Real pipelines stack them, and stacked leaks are not additive — that is what the
  Shapley section is for.
- **Both modes are scored on the same rows.** Constructions have different warm-up and
  edge-effect profiles (a centred window loses both ends, a trailing one loses the start), so
  rows are restricted to where *both* are defined. Otherwise part of any gap would just be one
  side quietly evaluating on an easier subset.
- **The `[control]` row is the harness's own test.** It runs one identical construction on
  both sides, so its gap must be exactly zero. It is, to every printed digit. If it ever is
  not, the harness leaks and no other row in the table means anything.
- **The folds are purged and embargoed** (`horizon=1, embargo=2`), so fold adjacency is not a
  leak channel and the whole gap is attributable to the step.

## Reproducing

```bash
python benchmarks/bench_leakage_table.py                 # ~20k rows, 5 seeds, ~6s
python benchmarks/bench_leakage_table.py --long          # 150k rows, 9 seeds, ~17s
python benchmarks/bench_leakage_table.py --only shift_negative
python benchmarks/bench_leakage_table.py --rows 50000 --seed 100 --seeds 3
```

Deterministic (seeded RNG throughout), float64, numpy + polars only — no scikit-learn, so it
runs on the bare-core install.
