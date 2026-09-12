# Leakage (`causalize` / `borrowed_accuracy`)

A **point-in-time compiler** and a **decomposable leakage metric**, in one
subpackage. `panelary.leakage` prevents an entire class of look-ahead error and
measures what it would have cost you. Pure `numpy` + `polars`; it pulls in no
optional dependency and is eager on `import panelary`.

The compiler works on the serialised Polars expression tree
(`Expr.meta.serialize(format="json")`). It walks the tree, matches each node
against a rule table keyed by node kind, rewrites operations that reach forward
in time into their expanding-window equivalents, and **refuses to compile what
it cannot rewrite**. An unrecognised node is `REFUSE`, never `SAFE` — a compiler
that guesses is worse than no compiler.

The metric is the same idea reported as a number rather than an assertion. Run
the pipeline permissively, run it point-in-time, and report the gap; then
attribute that gap to each component by exact Shapley value, so the parts sum to
the whole by construction.

For the reasoning — the mechanism, the `.over` sort trap, the fail-closed
argument and the honest limits — read
[Point-in-time compilation](../concepts/point-in-time.md) first.

## What's here

| Your problem | Entry point |
| --- | --- |
| Make this expression point-in-time, or tell me you can't | `causalize` |
| The same, without raising — just the findings | `audit` |
| How much accuracy did this pipeline borrow from the future? | `borrowed_accuracy` |
| The gap, the per-component attribution and the coalition values | `BorrowedAccuracyReport` |
| One pipeline stage, runnable permissively or point-in-time | `Component` |
| Panel keys and options the rules decide against | `Context` |
| One decision, at one position in the tree | `Finding` |
| The outcome for a whole expression | `CompileResult`, `Verdict` |
| What the rule table said about one node | `Classification` |
| Raised when an expression cannot be made causal | `LeakageRefused` |

## `causalize` and `audit`

`causalize(expr, time=..., entity=...)` returns a rewritten `polars.Expr` or
raises `LeakageRefused`. `audit(...)` is the non-raising form: it returns a
`CompileResult` whose `findings` you can inspect, with `.refused` and
`.rewritten` narrowing it to the decisions that matter.

Both take the panel keys as keyword arguments — `time`, `entity`,
`allow_approximate` — and build the `Context` the rules decide against.

`time` is not optional in practice: an `.over(entity)` node carries no
`order_by`, so "within entity" is only correct if the frame happens to be
sorted. Given `time`, the compiler injects `order_by=time` and turns that
assumption into something the engine enforces; without it, the compiler refuses
rather than guessing which column is the clock. `allow_approximate` permits
rewrites that are causal but not numerically identical to the original — an
expanding quantile is not the global quantile — and is off by default so that a
rewrite never silently changes results.

The rule table covers roughly ten operations: the fill strategies, rolling
aggregates, shifts and reversed cumulative aggregations, whole-column aggregates
and ranks, interpolation, and `.over` scoping. `AnonymousFunction` nodes
(`map_batches`, `map_elements`) are opaque and always refused — unless the
operator is registered as a [`FeatureSpec`](registry.md), whose
`panel_safe` / `leakage_safe` declarations make it a trusted leaf.

## `borrowed_accuracy`

Give it a list of `Component`s, each runnable in two modes — *permissive* (fit
once, on everything) and *point-in-time* (refit using only rows at or before the
cut) — and an `evaluate` callable that scores one configuration.

Borrowed accuracy is `B = v(all permissive) − v(none permissive)`. Attribution
is the exact Shapley value

```text
phi_i = Σ over S ⊆ N\{i} of  |S|!(k−|S|−1)!/k!  ·  ( v(S ∪ {i}) − v(S) )
```

which costs `2^k` evaluations. At the component counts that occur in
practice — a scaler, an imputer, an encoder, a feature generator, model
selection, retrieval — that is exact and affordable, so **nothing is sampled**;
`max_components` (default 12) is the guard against the combinatorial blow-up.
`Σ phi_i == B` — Shapley efficiency — is what makes this a *decomposition*
rather than a set of ablations, and it is asserted as a property test alongside
null-player (a component with no leakage gets `phi == 0`) and symmetry.

The returned `BorrowedAccuracyReport` carries `total`, the per-component
`attribution`, both endpoint scores, and the full `coalition_values` map, so the
attribution can be re-derived rather than taken on trust.

The expensive part is point-in-time refitting. Refit at fold boundaries rather
than at every `t`; the error that introduces is itself measurable by refining
the grid.

## Scope

This is a v1. It is a rule table plus a walker over a tree format Polars does
not promise to keep stable — not a general theorem prover, and not a
certification of arbitrary Python. A refusal means *not analysable here*, which
includes a great deal of perfectly safe code. The
[limits section](../concepts/point-in-time.md#honest-limits) of the concept page
is the honest inventory.

## See also

- [Point-in-time compilation](../concepts/point-in-time.md) — the concepts, the
  measured example, and the limits.
- [Leak-safety](../concepts/leak-safety.md) — the `panel_safe` / `leakage_safe`
  contract the registry declares and this compiler trusts.
- [Leak verifiers](testing.md) — `assert_no_lookahead`, the empirical check
  every rewrite rule is proved against.
- [Feature registry](registry.md) — why a registered operator is a trusted leaf.
- [Leakage & correctness-by-construction](../leakage.md) — purge, embargo and
  CPCV, the fold-level half of the same problem.

## API

::: panelary.leakage
