# `panelary/leakage/` — build contract

**Stage:** in progress · Implements `plans/todo/borrowed-accuracy.md` (01) and
`plans/todo/causalize.md` (03). They are one package: 03 prevents, 01 measures,
and 01 is 03's test suite.

Pure **numpy + polars**. No scipy, sklearn, statsmodels, numba. The package is
eager in `panelary/__init__.py` (it costs ~0 ms and pulls no optional dep).

## Why this is buildable — verified, not assumed

`Expr.meta.serialize(format="json")` exposes the whole tree, and
`pl.Expr.deserialize` reads a mutated one back. Measured on Polars 1.44:

```
pl.col("x").fill_null(strategy="backward").rolling_mean(5, center=True)
  -> {"Function": {"function": {"RollingExpr": {"options": {"center": true ...
     "input": [{"Function": {"function": {"FillNullWithStrategy": {"Backward": null}}}}]}}
flip both in JSON, deserialize, evaluate:
  leaky [None, None, 3.2, 4.2, None, None]
  causal[None, None, None, None, 2.8, 3.8]
```

Node shapes confirmed on 1.44 (use these; do not guess):

| Expression | Qualified kind | Where the leak lives |
|---|---|---|
| `fill_null(strategy="backward")` | `Function.FillNullWithStrategy` | `{"Backward": null}` |
| `rolling_mean(5, center=True)` | `Function.RollingExpr` | `options.center == true` |
| `shift(-1)` | `Function.Shift` | `input[1]` literal `< 0` |
| `cum_sum(reverse=True)` | `Function.CumSum` | `{"reverse": true}` |
| `rank()` | `Function.Rank` | whole-column ranking |
| `interpolate()` | `Function.Interpolate` | bidirectional |
| `x.mean()` / `std()` / `quantile()` | `Agg.Mean` … | whole-column aggregate |
| `map_batches` | `AnonymousFunction` | opaque — always refuse |
| `.over("id")` | `Over` | `order_by == null` |

## Hard invariants

1. **Fail closed.** An unrecognised node kind is `REFUSE`, never `SAFE`. A
   compiler that guesses is worse than no compiler.
2. **A rewrite must be exact** unless `Context.allow_approximate` is set. An
   expanding mean is exactly the point-in-time mean; an expanding quantile is
   not exactly the global quantile, so it is approximate and off by default.
3. **Every rewrite is proved empirically, not just structurally.** Each rule
   that emits a rewrite carries a test asserting the rewritten expression
   passes `panelary.testing.assert_no_lookahead`, and that the original
   *fails* it. A rule whose "leaky" form passes the verifier is not a leak and
   should not be in the table.
4. **The registry is the type system for Panelary's own ops.** Every
   `FeatureSpec` carries `panel_safe` / `leakage_safe`. A registered op is a
   trusted leaf, which is what stops the compiler refusing Panelary's own
   `map_batches`-based operators (AGENTS.md invariant 5).
5. **`.over(entity)` with no `order_by` is not safe.** It is only correct if
   the frame happens to be sorted. Repair it by injecting `order_by=time` when
   `Context.time` is known; refuse when it is not. Do not assume sortedness.
6. Prefix invariance, float64, seeded RNG, no `rolling_map` — as everywhere.
7. Public API must not grow a name that collides with the eight verbs;
   `causal` is already taken, hence `leakage`.

## File ownership

| File | As built | Contents |
|---|---|---|
| `_types.py` | **landed** (200 lines) | `Rule`, `Context`, `Finding`, `CompileResult`, `Verdict`, `Classification`, `LeakageRefused`, `node_kind`, plus `POLARS_TREE_FORMAT_TESTED = ("1.44",)` |
| `_rules.py` | **landed late** (~890 lines) | `RULES`: 39 node kinds, 10 of them carrying a rewrite. Returns complete nodes, per the hardened `Rule.rewrite` contract |
| `_compile.py` | **landed** (~430 lines) | `audit()`, `causalize()`, `_walk`; keyword-arg API, not a positional `Context`. Refuses a rule that returns a bare payload instead of a node |
| `_borrowed.py` | **landed** (439 lines) | `Component`, `resolve_modes`, `BorrowedAccuracyReport`, `borrowed_accuracy()`, exact Shapley, `DEFAULT_MAX_COMPONENTS = 12` |
| `__init__.py` | **landed** (83 lines) | 11 re-exports + `__all__`; eager wiring into `panelary/__init__.py` behind `_warn_unavailable` |
| `docs/` + nav | **landed** | `docs/api-reference/leakage.md`, `docs/concepts/point-in-time.md`, two `mkdocs.yml` nav entries, one line each in `README.md` and `llms.txt` |
| `benchmarks/bench_leakage_table.py` | **landed late** (723 lines) | scored table of preprocessing steps, permissive vs point-in-time, cross-checked against `assert_no_lookahead`; numpy ridge so it runs bare-core. Not executed by me — no numbers verified |
| `core/pipeline.py` integration | **landed** (+522 lines) | `Pipeline.audit()`, `Pipeline.causalize()`, `StepAudit`, `PipelineAudit`, and the `leakage_exprs()` / `with_leakage_exprs()` step hooks |
| `tests/test_leakage_api.py` | **landed** (200 lines) | not in the original table; public-surface guard |
| `tests/test_leakage_compile.py` | **landed** (~600 lines) | not in the original table; green |
| `tests/test_leakage_borrowed.py` | **landed** (551 lines) | not in the original table |
| `tests/test_leakage_pipeline.py` | **landed** | not in the original table; covers the `Pipeline` facade |

## The borrowed-accuracy decomposition

`v(S)` = score with the components in `S` run permissively and the rest
point-in-time. Borrowed accuracy `B = v(all permissive) - v(none)`. Attribution
is the exact Shapley value

```
phi_i = sum over S subset of N\{i} of  |S|!(k-|S|-1)!/k!  * ( v(S + i) - v(S) )
```

which requires `2^k` evaluations — for `k <= 10` that is exact and affordable,
so **do not sample**. `sum(phi) == B` (efficiency) is the property test that
proves the implementation, alongside null-player (a component with no leakage
gets `phi == 0`) and symmetry.

## As built

Every file in the table above exists, the compiler works end to end, and the
leakage suite is green: **103 passed, 0 failed, 0 skipped**. `causalize` repairs
the contract's own three-leak example and the result is prefix-invariant.
Verified against the installed **polars 1.44.2**, inside the
`POLARS_TREE_FORMAT_TESTED = ("1.44",)` window.

This section was written across a moving build; the numbers below are from the
final pass, after `_rules.py`, `_compile.py` and `tests/test_leakage_compile.py`
stopped changing.

1. **`audit` / `causalize` take keyword arguments, not a `Context`.** The
   contract's node table implied `causalize(expr, ctx)`. The built signature is
   `audit(expr, *, time=None, entity=None, allow_approximate=False)` and the
   same for `causalize`; `Context` is constructed internally and stays public
   only as the value the rules decide against. `Pipeline.audit` /
   `Pipeline.causalize` mirror the three keywords and default `time` / `entity`
   to the keys configured on the pipeline. The `__init__.py` docstring example
   was corrected to the keyword form during the build and is now accurate; note
   that both of its interesting lines carry `# doctest: +SKIP`, which is what
   keeps the doctest suite green while *Not done* 1 stands.

2. **`_borrowed.py` is pure combinatorics, deliberately.** It never fits a
   model, touches a frame, or decides what "permissive" means: the caller passes
   `evaluate(selection: frozenset[str]) -> float` and the module does the `2**k`
   memoised evaluations and the Shapley weights. `Component` holds the
   permissive / point-in-time callable pair but the module never calls them —
   `resolve_modes(components, selection)` is the helper a caller uses when
   building its own `evaluate`. This is a better factoring than the contract
   described (it makes the Shapley axioms directly testable) but it does mean
   `borrowed_accuracy` is not a one-call backtest driver.

3. **Two guard rails not in the contract.** `max_components` (default 12, i.e.
   4096 evaluations) refuses rather than silently taking exponential time, and
   `higher_is_better=False` negates every difference so `total > 0` always reads
   "borrowed" whichever way the score points.

4. **The `Pipeline` integration is opt-in per step.** A step reaches the
   expression compiler only if it implements `leakage_exprs() -> dict[str, Expr]`;
   to be rewritten it must also implement
   `with_leakage_exprs(mapping) -> PanelTransformer`. A step that implements
   neither is audited from its declared `panel_safe` / `leakage_safe` class
   attributes, which is a weaker check — a declaration, not a proof — and the
   `StepAudit` says so. `Pipeline.causalize()` returns a new `Pipeline` and
   never mutates the original.

5. **The walker deserialises on every call, even when nothing changed.** A few
   microseconds, and it keeps the round trip on the tested path so a Polars
   format change surfaces as a loud `RuntimeError` naming
   `POLARS_TREE_FORMAT_TESTED` instead of a silently unrewritten expression.

6. **The rule table is wider and shallower than the contract implied.** It knows
   **39 node kinds**, not ten — but only **ten carry a rewrite**: the seven
   whole-column aggregates that become cumulative (`Agg.Mean`, `Sum`, `Min`,
   `Max`, `Std`, `Var`, `Count`), `Function.FillNullWithStrategy`,
   `Function.RollingExpr` and `Over`. The other 29 are classify-only: safe
   leaves (`Column`, `Literal`, `Cast`, `BinaryExpr`, `Alias`, `Ternary`,
   `Function.Abs`/`Boolean`/`Pow`) and refusals (`Rank`, `Interpolate`,
   `Reverse`, `Sort`, `Diff`, `PctChange`, `EwmMean`, `Quantile`, the
   `Function.Cum*` family, a negative `Shift`, `AnonymousFunction`,
   `Agg.First`/`Last`/`Median`/`NUnique`). "Roughly ten operations" in the
   contract is best read as "ten rewrites"; the breadth is in what it can
   *recognise* as safe without refusing.

Verified by running, not by reading:

- `sorted(panelary.leakage.__all__)` is exactly
  `['BorrowedAccuracyReport', 'Classification', 'CompileResult', 'Component',
  'Context', 'Finding', 'LeakageRefused', 'Verdict', 'audit',
  'borrowed_accuracy', 'causalize']` — 11 names, matching invariant 7 (no
  collision with the eight verbs; the package is `leakage`, not `causal`).
- `borrowed_accuracy` on a synthetic 3-component game with a deliberate
  pairwise interaction: `n_evaluations == 8 == 2**3`, efficiency
  `sum(phi) == total` to `< 1e-12`, and the null player scored exactly `0.0`.
- `Pipeline.audit` and `Pipeline.causalize` both exist with the three-keyword
  signature above.
- `len(RULES) == 39`, of which 10 carry a `rewrite` callable.
- Classification is correct on the cases that do not rewrite: `shift(-1)`,
  `cum_sum(reverse=True)`, `rank()` and `map_batches` each return
  `Verdict.REFUSED` with the expected qualified kind; `shift(1)` returns
  `Verdict.SAFE` with no findings.
- Every rewriting case round-trips: centred `rolling_mean`,
  `fill_null(strategy="backward")` and `.mean().over("id")` each return
  `Verdict.REWRITTEN`, and the compiled expression evaluates.
- `causalize` on the contract's three-leak example repairs all three leaks in
  one pass; the causal column differs from the leaky one and is prefix-invariant
  over the sample.
- `pytest tests/test_leakage_{api,borrowed,compile,pipeline}.py` →
  **103 passed** in 2.05s. No failures, no skips; the pipeline suite's earlier
  "compiler is not usable yet" auto-skips are gone.
- `mkdocs build --strict` is green with the two new pages in the nav.

## Not done

### 1. Nothing — the compiler and its suite are green

Kept as a numbered entry because the two items below were renumbered around it,
and because the history is worth recording: this slot held a genuine blocker for
most of the build, and it was fixed twice over.

Sequence, as observed:

1. Every rewrite produced a tree Polars could not deserialise. A rule returned
   the whole **node** where the walker expected the node's **payload**, and the
   walker re-wrapped it: `{"Function": {"Function": {...}}}`.
2. `_compile.py` was hardened to *detect* that mismatch and refuse loudly
   rather than emit an unloadable tree — fail-closed, the right direction —
   and `_rules.py` was updated to return complete nodes. The shipped table
   started working; the test file's **fixture** table, still returning bare
   payloads, went red in five places.
3. The fixture was repointed at the new contract and the suite went green.

The lesson worth keeping: the `Rule.rewrite` contract is "return a complete
node, never a bare payload", and sequence child slots take positional keys
(`'0'`), not `''`. Both halves of that were learned the hard way and are now
enforced by `_compile.py` with an explicit refusal message.

### 2. The golden-tree test per supported Polars minor

`causalize.md` names "a golden-tree test per supported Polars minor version" as
the mitigation for risk 1, the unstable serialised format. There is no such
test. The only thing pinning the format is the `RuntimeError` the walker raises
when the round trip fails, covered at `tests/test_leakage_compile.py:574`, and
`POLARS_TREE_FORMAT_TESTED = ("1.44",)` as a documented constant. That is
fail-closed but it is not a golden tree: nothing asserts the node *shapes* in
the contract's table above are still what Polars emits.

### 3. The registry-as-trusted-leaf escape hatch

Hard invariant 4 — a registered `FeatureSpec` is a trusted leaf, which is what
stops the compiler refusing Panelary's own `map_batches`-based operators — is
documented in `docs/api-reference/leakage.md` and
`docs/concepts/point-in-time.md` but implemented nowhere. Nothing under
`panelary/leakage/` imports `panelary.registry`; `AnonymousFunction` is an
unconditional refusal. So today the compiler *does* refuse Panelary's own
operators, which is the exact failure mode `causalize.md` risk 2 predicted and
claimed to have solved.

### 4. Target encoding

`causalize.md` names target encoding as the hard case in the first ten, to be a
hand-written registered op (expanding mean of the target per category, lagged
one period, shrunk toward the expanding global mean) rather than a tree rewrite.
No such op was added, and no `FeatureSpec` was registered for one.

### 5. Benchmark numbers

`benchmarks/bench_leakage_table.py` was written but not run as part of this
build, so no measured figures from it are recorded here or in `CHANGELOG.md`.
Its own docstring is appropriately careful that the gap for any step is a
property of its synthetic DGP and seeds, not a universal constant; whoever runs
it should keep that caveat attached to any number they quote.
