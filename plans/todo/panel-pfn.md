# PanelPFN: a foundation model pretrained on a panel prior

**Stage:** todo — needs a build contract before implementation · **Priority:** highest upside, highest cost · **Home:** its own repo; Panelary supplies the prior

## Pitch

Every tabular foundation model's prior is a structural causal model over
i.i.d. tables. BeyondArena shows those models lose on temporal, grouped, large
and high-dimensional data — exactly what the omission predicts. O'Prior then
proved the mechanism: hold architecture, optimizer and compute fixed, vary only
the task distribution, and downstream robustness moves, with gains
concentrated where the data is irregular.

CAFE is already a generator you can sample from. Generalise it —
regime-switching and stochastic-volatility factors, Student-t innovations,
asynchronous observation, informative missingness, entity entry and exit,
cross-sectional clusters, revisions, reporting lags, structural breaks, varying
N and T — and you have the prior nobody else can write, attacking BeyondArena
at the source rather than engineering around it.

First experiment: falsification first. Sample synthetic panels across the dial
space, pre-train a 20–50M parameter column-then-row model, evaluate in-context
on BeyondArena's temporal and grouped sub-benchmarks against tuned CatBoost.
If it loses there, the prior hypothesis is wrong and you stop. Honest caution:
realism is not automatically the objective.

## Assessment

**Split it in two; only the first half belongs in Panelary.**

- *The prior* — a seeded, dial-parameterised synthetic panel generator — is a
  natural Panelary module and valuable on its own: it is the test fixture every
  leakage suite here currently hand-rolls. Useful even if PanelPFN fails.
- *The model* — pretraining, GPU infrastructure, checkpoints — is a separate
  repo that depends on the generator.

**Check the CAFE premise.** In this repo CAFE is the optional third-party
`cafe` package (`cafe-impute` 0.1.0), wrapped for *imputation* via
`cafe.impute`. Whether it exposes a samplable generator needs checking; the
generator may have to be written rather than generalised.

**Keep the stopping rule.** Falsification-first with a pre-committed
kill condition is the right design for a project this expensive.
