# Point-in-time compilation

> **Discipline does not scale.** A compiler does.

[Leak-safety](leak-safety.md) explains *what* leakage is on a panel and how every
operator Panelary ships is built, flagged and verified not to leak. This page is
about the other half of the problem: the code you write yourself, and the
libraries you reach for, which are not built that way and never declared
anything. `panelary.leakage` compiles that code into a point-in-time form, or
refuses.

## Why discipline does not scale

The rule is easy to state — *the value at time `t` may depend only on data at
times `<= t`* — and almost impossible to hold by hand, because nearly every
convenient preprocessing step violates it silently and none of them raise.

| What you wrote | What it actually reads |
| --- | --- |
| `x.mean()`, `x.std()` for standardisation | the whole column, test period included |
| `fill_null(strategy="backward")` | the next observed value, which is in the future |
| `interpolate()` | both neighbours, so half of it is the future |
| `rolling_mean(k, center=True)` | `k // 2` rows after `t` |
| `shift(-1)`, `cum_sum(reverse=True)` | explicitly backwards through time |
| `rank()` over the frame | every entity at every date, not the cross-section at `t` |
| target encoding by category mean | the target on rows that have not happened yet |
| a join on a dimension table | whatever the vendor knows *now*, not what they knew then |

None of these is a mistake in the sense of a typo. Each is the ordinary,
documented, i.i.d.-correct behaviour of the tool. scikit-learn's `TargetEncoder`
is the sharpest example: its internal cross-fitting is exactly the right defence
against target leakage when rows are exchangeable, and on time-ordered data it
encodes each row using folds drawn from its own future. The safe default for one
setting is the leak in another.

So the mitigation in practice is discipline: remember which of these reaches
forward, every time, in every notebook, on every branch, forever. That works
until a pipeline has thirty steps, or a second author, or a deadline.

### The incident that motivated this

This library shipped the bug. `cross_validation`'s `sliding_window_split`
computed its training offset as `pl.len() - cutoff - window_size`. When the
requested window exceeded the history available before the first test block,
that offset went negative — and **a negative Polars slice offset counts back
from the end of the group**. On a 20-step panel with `test_size=2, n_splits=5,
step_size=3, window_size=10`, fold 0 tested `t=6..7` and trained on `t=16..19`.

A public splitter, in a library whose stated value proposition is leak-safety,
trained on data ten steps *after* the block it was scoring. It was not found by
review or by the leakage suite — the existing tests only exercised the
panel-aware variants. It was found by accident, in commit
[`20ce293`](https://github.com/sovai-research/panelary/commit/20ce293), while
de-duplicating two splitters that were assumed to be the same code. See the
`Fixed` section of the [changelog](https://github.com/sovai-research/panelary/blob/main/CHANGELOG.md)
for the full write-up.

That is the case for a compiler rather than discipline. Careful people, a
correctness-focused codebase, an existing leakage test suite, and the bug still
sat in a shipped release. The class of error has to be made impossible by
machine, not unlikely by intention.

## The mechanism, concretely

Polars builds a lazy expression tree, and it will hand it to you as JSON:
`Expr.meta.serialize(format="json")` writes it out and `pl.Expr.deserialize`
reads a mutated one back. That round trip is the whole trick. The compiler walks
the serialised tree, matches each node against a rule table keyed by node kind,
rewrites the nodes that reach forward, and rebuilds a real `polars.Expr`.

Take a two-step expression that any analyst might write without a second thought:

```python
import polars as pl

df = pl.DataFrame({"x": [1.0, None, 3.0, 4.0, 5.0, 6.0]})
leaky = pl.col("x").fill_null(strategy="backward").rolling_mean(5, center=True)
```

Serialised, both leaks are named in the output: a `RollingExpr` node whose
`options.center` is `true`, holding as its input a `FillNullWithStrategy` node
carrying `{"Backward": null}`. Flip those two values in the JSON, deserialize,
and evaluate. Measured on Polars 1.44:

```text
x       [1.0,  None, 3.0, 4.0, 5.0, 6.0]
leaky   [None, None, 3.2, 4.2, None, None]
causal  [None, None, None, None, 2.8, 3.8]
```

Read the row at index 2. The leaky expression reports `3.2` there — a number
that is a function of `x[4]` and `x[5]`, rows that had not happened yet. The
causal expression reports `None`, because at index 2 a trailing five-row window
has not filled. The first honest value arrives at index 4, and it is a different
number. That gap between `3.2` and nothing is borrowed accuracy, in a single
cell, from an expression two operations long.

Note what the rewrite is *not*. It does not shift the leaky answer or
approximate it; it computes the quantity that was actually available at each
row, which is a genuinely different series with a different null prefix. A
point-in-time pipeline knows less than a permissive one. That is the point.

## Fail closed

The rule table covers roughly ten operations — the fill strategies, rolling
aggregates, shifts and cumulative reversals, whole-column aggregates and ranks,
`.over` scoping. **Anything the table does not recognise is refused, not
assumed safe.**

This is the only defensible default, and it is worth being explicit about why.
The compiler's output is a *guarantee*: if it returns an expression, that
expression does not read forward. A guarantee that holds only for nodes someone
remembered to enumerate is not a guarantee — it is the same discipline problem
relocated into the rule table, where it is *less* visible than it was in the
notebook. Fail-open also degrades exactly when it matters most: a new Polars
release adds a node kind, or you use an operation nobody anticipated, and the
compiler waves through precisely the thing it had never analysed. And the two
failure modes are not symmetric. A false refusal costs you an error message and
five minutes writing the operation a different way. A false pass costs you a
strategy.

So an unknown node kind is `REFUSE`, and the refusal names the node and the
position in the tree. Use `audit()` when you want the findings without the
exception.

## The subtle one: `.over(entity)` carries no order

This is the catch worth the price of admission on its own, because it is not a
mistake anyone makes — it is a mistake the API makes on your behalf.

```python
pl.col("close").panel.zscore(20).over("ticker")
```

`.over("ticker")` partitions by entity. It says nothing at all about *order*
within the partition. A trailing window inside that partition is therefore
trailing with respect to **whatever order the rows happen to sit in**. If the
frame is sorted by `(ticker, date)`, this is correct. If it arrived sorted by
`(date, ticker)`, or came back from a join, a `group_by`, a `sink_parquet` read
with multiple row groups, or a shuffle at any point in a longer pipeline, then
"trailing" means trailing in file order and the window reads whatever rows are
physically adjacent.

This is a precondition every panel library states in its docs and none of them
enforces, Panelary included. The frame-level `.panel` namespace guards the
*grouping* — it refuses to run without an entity key — but nothing checks the
sort, and nothing can, because sortedness is a property of the data rather than
of the expression.

The compiler handles it structurally. An `Over` node whose `order_by` is `null`
is not safe, so:

- when `Context.time` is known, the compiler **injects** `order_by=time`,
  turning an assumption into a declaration the engine enforces;
- when it is not known, the compiler **refuses**. It cannot know which column
  means time, and guessing which of your columns is the clock is the sort of
  helpfulness that produces a silently wrong backtest.

Pass the panel keys. `causalize(expr, time="date", entity="ticker")` repairs
this case; `causalize(expr)` will tell you it cannot.

## Honest limits

This is a v1 covering roughly ten operations. It is a rule table with a walker,
not a theorem prover, and it will not certify an arbitrary program.

**The serialised tree is not a stable public API.** Polars does not promise the
JSON shape across versions, and the compiler is built directly on it. The
mitigation is a pinned, tested set: the package records which Polars minor
versions its node shapes have been checked against (the module constant
`POLARS_TREE_FORMAT_TESTED`), golden-tree tests hold those shapes, and anything
unrecognised on a newer release refuses rather than misreads. A Polars upgrade
may turn working code into refusals until the table is re-verified. That is the
intended direction to fail in, but it is a real cost.

**`map_batches` and `map_elements` are opaque and always refused.** An arbitrary
Python callable cannot be analysed; there is no tree to walk. This collides with
Panelary's own escape hatch, since some shipped operators are implemented that
way — which is what the [operator registry](../api-reference/registry.md) is for.
A registered `FeatureSpec` already declares `panel_safe` / `leakage_safe`, so a
registered operator is a **trusted leaf**: the compiler reads the declaration
instead of the callable. Your own `map_batches` has no such declaration and will
be refused. Register it, or rewrite it in expressions.

**Exact rewrites only, by default.** An expanding mean *is* the point-in-time
mean, exactly. An expanding quantile is not the global quantile — it is a
defensible causal substitute, but it changes the numbers, so it is gated behind
`Context.allow_approximate` and off by default. A rewrite should never silently
change results.

**Structural analysis is not the proof.** The walker establishes coverage: it
shows that every node was classified. Soundness is established empirically —
each rewrite rule carries a test asserting the rewritten expression passes
[`assert_no_lookahead`](../api-reference/testing.md) *and* that the original
fails it. A rule whose "leaky" form passes the verifier was never a leak and
does not belong in the table.

**Refusal is not a verdict of leakage.** `REFUSE` means *not analysable here*,
which includes plenty of perfectly safe code. Read it as "prove this another
way" — with the verifier, or by registering the operator — not as an accusation.

## Measuring what is left

Prevention has a companion measurement. `borrowed_accuracy` runs a pipeline
twice, permissively and point-in-time, and reports the gap — the accuracy the
method borrowed from data it will not have. Because components interact, the gap
is attributed to each one by exact Shapley value over all `2^k` subsets, so the
per-component numbers sum to the total by construction rather than by assumption.
That is what turns "your backtest is optimistic" into "1.8 points of it came from
the scaler".

## See also

- [Leak-safety](leak-safety.md) — the two leak axes, the `panel_safe` /
  `leakage_safe` contract, and the future-perturbation verifier.
- [`leakage`](../api-reference/leakage.md) — the API: `causalize`, `audit`,
  `borrowed_accuracy`.
- [Leakage & correctness-by-construction](../leakage.md) — purge, embargo, CPCV,
  and the backtest-overfitting statistics.
- [Feature registry](../api-reference/registry.md) — why a registered operator is
  a trusted leaf.
