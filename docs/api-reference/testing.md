# Leak verifiers

Future-perturbation leak verifiers — the keystone of Panelary's leak-safety
guarantee. A transform is *leak-free* if its output at time `t` depends only on
data at times `<= t` (within each entity). The cleanest, model-agnostic way to
check that is a future-perturbation experiment: run the operation, corrupt every
value strictly in the future (per entity), run it again, and assert every output
cell in the past is bit-identical across the two runs. Any difference is a
look-ahead.

- `assert_no_lookahead` — split the shared time axis at a cut `t` and assert that
  perturbing `time > t` never changes any output at `time <= t`.
- `assert_no_train_test_leak` — perturb a *test* fold and assert the *train*-fold
  outputs are unchanged (the CV-boundary version of the same idea).

Both accept `op` as either a `polars.Expr` or a callable `frame -> frame`, and
failures name the first offending `(column, entity, time)`.

::: panelary.testing
