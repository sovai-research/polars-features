# Econometrics

Panel and time-series econometric estimators, written **clean-room from the
published papers** and depending on nothing beyond `numpy` and `polars` — no
`scipy`, `statsmodels` or `linearmodels`. The chi-squared, Student-t and F tails
the tests need are computed from the incomplete gamma / beta expansions in
`polars_features.econ._common`.

Every class that learns parameters subclasses `PanelTransformer`, declares
`panel_safe` / `leakage_safe`, learns everything in `_fit` from the rows it is
handed, and applies frozen state in `_transform`. The narrative walkthrough is
the [Econometrics guide](../user-guide/econometrics.md).

## What's here

| Problem | Entry point |
| --- | --- |
| Absorb firm × date (× anything) fixed effects on millions of rows | `hdfe`, `HDFETransformer` |
| Cluster-robust / two-way / Driscoll-Kraay standard errors | `hdfe(vcov=…, cluster=…)` |
| Slopes differ by entity, and a common factor drives everyone | `cce_mg`, `cce_pooled`, `mean_group` |
| Common **long-run** relationship, entity-specific dynamics | `pmg` |
| Is this panel stationary? | `llc`, `ips`, `cips` |
| Have my fixed effects removed the common factor? | `pesaran_cd` |
| Cross-sectional risk premia over time | `fama_macbeth`, `FamaMacBethTransformer` |
| Who transmits shocks to whom? | `connectedness`, `rolling_connectedness`, `ConnectednessFeatures` |
| Predictability inference with a near-unit-root predictor | `ivx`, `ivx_screen`, `IVXSelector` |
| Effect of one feature, controlling for hundreds | `dml_partial_linear`, `post_double_selection` |

## High-dimensional fixed effects

`hdfe` absorbs an arbitrary number of fixed-effect dimensions by **alternating
projections** (von Neumann / Halperin; Gaure 2013 — the engine behind `reghdfe`
and `fixest`). Each individual projection is a group-mean demeaning, i.e. a
Polars `group_by`, and sweeping the dimensions in turn converges to the joint
projection. No dummy matrix is ever materialised, so the cost is linear in rows
rather than quadratic in levels.

By Frisch-Waugh-Lovell this reproduces a full dummy-variable OLS exactly —
coefficients *and* standard errors, verified to ~1e-8 against a hand-rolled
dense reference in `tests/test_econ_hdfe.py`. Available covariances:
`"classical"`, `"robust"` (HC1), one-way `"cluster"`, two-way clustering via
Cameron-Gelbach-Miller (`cluster=["firm", "date"]`, with an eigenvalue clip to
keep it PSD), and `"driscoll-kraay"` for arbitrary cross-sectional dependence.

`HDFETransformer` freezes the per-level offsets learned at `fit` time and applies
them at `transform` time, so the partialled-out columns are leak-controlled
features usable inside cross-validation. Levels unseen in training get a zero
offset (or `null`, with `unseen="null"`).

## Heterogeneous panels

`mean_group` (Pesaran-Smith 1995) fits one time-series regression per entity and
averages the slopes, using the *dispersion* of the slopes as the variance
estimator. `cce_mg` / `cce_pooled` (Pesaran 2006) add the per-date
**cross-sectional averages** of the dependent variable and regressors to each
entity's regression, which asymptotically annihilates unobserved common factors;
Mean Group without that augmentation is badly biased whenever a factor is
present. `pmg` (Pesaran-Shin-Smith 1999) estimates the ARDL(1,1)
error-correction model in which the long-run vector is common but the adjustment
speed, short-run dynamics and intercepts are entity-specific, by alternating the
per-entity and pooled concentrated least-squares steps.

`llc`, `ips` and `cips` are the LLC / IPS / CIPS panel unit-root tests and
`pesaran_cd` the CD test for cross-sectional dependence. Rather than hard-coding
the response-surface tables, the null distributions of the first three come from
a cached, seeded **Monte-Carlo simulation of the identical estimator pipeline**
under an independent-random-walk null for the same `(N, T, lags)`. That is
self-contained and exactly reproducible; raise `reps` for tighter tails.

`PanelSlopeFeatures` joins frozen train-sample per-entity coefficients back onto
the panel; `CrossSectionalAverages` emits the CCE factor proxies, which are
contemporaneous (date `t` uses only date-`t` rows) and therefore leak-safe.

## Fama-MacBeth

`fama_macbeth` runs one cross-sectional OLS per date via `group_by(time)` and
reports the time-series mean of the resulting lambdas with **Newey-West**
standard errors. Winsorising and standardising happen **per date, never
globally** — a pooled z-score would let the whole sample's mean and standard
deviation leak backwards into every earlier row.

## Diebold-Yilmaz connectedness

`connectedness` fits a VAR, inverts it to its Wold moving-average form, and
computes the **generalised** forecast-error variance decomposition (Koop-Pesaran-
Potter; Pesaran-Shin), which is invariant to the ordering of the series. The
normalised table is a directed network: row sums give `from_others`, column sums
give `to_others`, their difference is `net`, and the off-diagonal mass is the
total connectedness index. `rolling_connectedness` / `ConnectednessFeatures`
re-estimate on a **trailing** window ending at each date, so the emitted
`dy_to` / `dy_from` / `dy_net` / `dy_total` columns are backward-looking
cross-entity features.

## IVX

When a predictor is highly persistent *and* its innovation correlates with the
return innovation — true of essentially every valuation ratio — the OLS
t-statistic over-rejects "no predictability" badly, and the distortion does not
vanish with sample size. `ivx` instruments the predictor with a **mildly
integrated** transform of its own increments (Magdalinos-Phillips 2009;
Kostakis-Magdalinos-Stamatogiannis 2015), which yields a mixed-normal limit and a
chi-squared Wald statistic regardless of the degree of persistence. `ivx_screen`
and `IVXSelector` use that Wald statistic as a leak-aware screening score,
frozen at `fit` time.

## Double machine learning

`dml_partial_linear` estimates `theta` in `Y = theta D + g(X) + U` using the
Neyman-orthogonal residual-on-residual score with **cross-fitting**: the
nuisance predictions for a row come from a model that never saw it. The folds are
PanelKit's own purged and embargoed
[`PurgedKFold`](cross-validation.md), so they are leak-safe temporally as well.
`post_double_selection` is the Belloni-Chernozhukov-Hansen alternative: LASSO
`Y` on `X`, LASSO `D` on `X`, then OLS on `D` plus the union of the two selected
sets. The default nuisance learner is `"post-lasso"` — rigorous (plug-in) penalty
selection followed by an unpenalised OLS refit, because plain LASSO shrinkage
biases the predictions and that bias survives orthogonalisation. Any object with
`fit` / `predict` (a scikit-learn regressor, say) can be passed instead; import
it yourself, this module adds no dependency.

::: polars_features.econ
