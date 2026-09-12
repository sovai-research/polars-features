# `pipeline.causalize()`: a point-in-time compiler

**Stage:** shipped — see `plans/done/leakage-build-contract.md` · **Priority:** 1 (with `borrowed-accuracy`) · **Home:** Panelary

## Pitch

Global means, backward fills, rolling functions, target encoding, joins and
normalisation all silently reach forward when applied to a panel.
scikit-learn's `TargetEncoder` — the field's safe default, correct for i.i.d.
leakage — uses future folds on time-ordered data. The current mitigation is
discipline, and discipline does not scale.

Polars has a lazy expression tree. Walk it, prove each operation
truncation-invariant or rewrite it into an expanding-window equivalent, and
refuse to compile what cannot be rewritten. This is a genuine Panelary
identity: not another `TimeSeriesSplit` but a compiler that makes an entire
class of error impossible.

First experiment: start with the ten operations that cause the most damage —
fill strategies, rolling aggregates, target encoding, scaling, joins. Ship
`causalize()` for those, with the borrowed-accuracy metric as the test suite.

## Assessment

**Feasibility is confirmed, not assumed.** Polars 1.44 exposes the tree via
`Expr.meta.serialize(format="json")`. For
`pl.col("x").fill_null(strategy="backward").rolling_mean(5, center=True).over("id")`
all three leaks are named in the output:

```
FillNullWithStrategy: Backward        -> rewrite to Forward
RollingExpr ... "center": true         -> rewrite to trailing window
Over ... "order_by": null              -> refuse, or inject order_by=time
```

The third is the most valuable catch. `.over(entity)` carries no time order,
so every "within-entity" op is only correct if the frame happens to be
sorted — exactly the precondition AGENTS.md states and nothing enforces.

**Risks, in order:**

1. **The serialised format is not a stable API.** Polars does not promise it
   across versions. Mitigation: a golden-tree test per supported Polars minor
   version, and fail closed (refuse) on any node kind the compiler does not
   recognise.
2. **Opaque nodes.** `map_batches` / `map_elements` are black boxes, so the
   compiler must refuse them — and AGENTS.md invariant 5 names `map_batches`
   as Panelary's own escape hatch, so it would refuse Panelary's own ops.
   **The registry fixes this:** every `FeatureSpec` already carries
   `panel_safe` / `leakage_safe`. Treat a registered op as a trusted leaf.
   The registry is already half of the compiler's type system.
3. **"Prove" is too strong in general.** Ship a whitelist of known-safe node
   kinds, rewrite rules for known-unsafe ones, and refusal for everything
   else. Back every rewrite with `assert_no_lookahead` as a property test —
   static checking for coverage, the empirical check for soundness.

**Target encoding** is the hard case in the first ten: the causal rewrite is
an expanding mean of the target per category, lagged by one period and
shrunk towards the expanding global mean. It should be a hand-written
registered op, not a tree rewrite.
