# Adaptive vector-leaf boosting

**Stage:** todo — needs a build contract before implementation · **Priority:** separate project · **Home:** not Panelary (needs XGBoost core)

## Pitch

XGBoost offers two extremes: one tree per output, or one shared tree for every
output. Vector leaves achieved lower held-out loss on nine of ten multiclass
datasets and roughly one ninth the serialised size at 32 outputs. But why
should all 32 outputs share? Twelve forecast horizons plausibly want three
groups, not one.

Learn the output-sharing graph jointly with the trees and you have a third
strategy — `multi_strategy='adaptive_output_tree'` — sitting between two
published extremes, with a hypothesis crisp enough to falsify in a month. Pair
it with exact vector-leaf SHAP interactions and you also get factor-pair to
horizon-vector attribution, which nothing currently provides.

First experiment: correlated-horizon return targets. Compare one-tree-per-output,
full sharing, and learned grouping. Report loss, model size and the recovered
output graph against the known correlation structure.

## Assessment

**The crispest hypothesis on the list, and the cheapest to falsify — without
touching XGBoost.** A joint learner is a C++ change to XGBoost. Panelary is
deliberately pure Python (AGENTS.md: no compiled extension), so this lives
elsewhere. But the month-one test needs no core change:

1. Fit targets with a known block structure (simulate 12 horizons in 3
   correlated groups).
2. Recover a grouping *before* training — cluster outputs on target
   correlation, or on leaf-value vectors from a fully-shared model.
3. Train one `multi_output_tree` model per group.

If this two-stage *fixed* grouping does not beat both extremes, joint learning
is unlikely to, and you have saved the C++ work. If it does, that is the
paper's baseline.

**Unverified:** the version-specific claims (the release that ships vector-leaf
SHAP interactions, the nine-of-ten and one-ninth figures) come from the source
document. Check them against the XGBoost changelog before citing.
