# Borrowed accuracy: a decomposable leakage metric

**Stage:** shipped — see `plans/done/leakage-build-contract.md` · **Priority:** 1 (build with `causalize`, below) · **Home:** Panelary

## Pitch

Every pipeline can be run twice — permissively, and with every operation
constrained to information available at prediction time. The gap is the
accuracy *borrowed* from data the method will not have. No published, named,
general version of this exists; fev-bench's leakage column is a
self-declaration covering same-frequency dataset overlap only.

The decomposition is the contribution: attribute the gap to the scaler, the
imputer, the encoder, the feature generator, model selection and retrieval
separately. Components interact, so the honest version is a Shapley
decomposition. Three independent papers have hit special cases of this
principle without naming it — the signature of a concept that is ready.

First experiment: formalise causality as output-at-t invariance to data after t.
Publish a reference implementation plus a scored table of the twenty most-used
preprocessing steps in scikit-learn and skrub. The commercial version answers:
how much of this vendor's backtest is look-ahead?

## Assessment

**The instrument already exists.** `panelary/testing.py::assert_no_lookahead`
*is* the formalisation in the pitch — perturb everything after a cut `t`,
assert outputs at `<= t` are bit-identical. It answers yes/no. Borrowed
accuracy is the same experiment reported as a number instead of an assertion.

**How to build it** (the part that looked unclear):

1. Let the pipeline have `k` components `c_1..c_k`, each runnable in two modes:
   *permissive* (fit once, on everything) and *point-in-time* (refit using only
   rows `<= t`, via the purged/embargoed splitters `core.model_selection`
   already has).
2. Borrowed accuracy `B = score(all permissive) − score(all point-in-time)`.
3. For attribution, evaluate the score for every subset `S` of components run
   permissively and the rest point-in-time: `v(S)`. That is `2^k` backtests.
   For the pitch's list (scaler, imputer, encoder, feature generator, model
   selection, retrieval) `k = 6`, so **64 runs — exact Shapley is affordable**,
   no sampling approximation needed. `phi_i = Σ_S |S|!(k−|S|−1)!/k! · (v(S∪{i}) − v(S))`.
4. The `phi_i` sum to `B` by construction (Shapley efficiency), which is what
   makes it a *decomposition* rather than a set of ablations.

The expensive part is point-in-time refitting. Refit at fold boundaries, not
at every `t`; the error that introduces is itself measurable by refining the
grid.

**Why it's credible here, specifically:** commit 20ce293 fixed a public
`sliding_window_split` that trained on `t=16..19` to predict `t=6..7`. It was
found by accident while deduplicating code. That is the case study for the
paper's motivation, and `B` for that bug is one computation away.

**Build order:** the scored table of twenty steps *is* the target list for
`causalize`. Build this first; it tells you which ten operations to compile.
