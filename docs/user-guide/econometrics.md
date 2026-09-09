# Econometrics: panel estimators that are also feature generators

Most feature libraries stop at what one entity's own history can tell you.
`panelary.econ` adds the estimators that only exist because a panel has
**two** dimensions — and treats their by-products as first-class features:

- the residual after absorbing firm and date fixed effects,
- an entity's own slope, or its speed of adjustment to a long-run relationship,
- the cross-sectional average that proxies the common factor,
- how much of an entity's forecast uncertainty is imported from its peers,
- the part of a treatment that the other 400 features cannot explain.

Everything below is pure NumPy plus Polars, written clean-room from the papers,
and every fitted object obeys Panelary's leak-safety contract.

## The 60-second version

```python
import panelary.econ as econ

# 1. Absorb firm and date effects; cluster two ways.
res = econ.hdfe(
    panel, y="ret", x=["size", "bm"],
    absorb=["firm", "date"], cluster=["firm", "date"],
)
res.summary()

# 2. Keep the partialled-out columns as features.
tr = econ.HDFETransformer(columns=["ret", "size"], absorb=["firm", "date"])
features = tr.fit(train).transform(test)

# 3. Ask whether the fixed effects actually removed the common factor.
econ.pesaran_cd(panel_with_residuals, value="resid")   # rejects -> use CCE
```

## Fixed effects without a dummy matrix

A two-way fixed-effects regression on 5,000 firms and 3,000 dates has 8,000
nuisance parameters. Building that design matrix is hopeless; you do not need to.

Absorbing a fixed effect *is* projecting onto the orthogonal complement of its
dummy span, and that projection is nothing more than **demeaning within groups** —
a `group_by(...).mean()`. For several dimensions at once, sweep them in turn and
repeat: alternating projections converge to the joint projection (von Neumann /
Halperin; Gaure 2013; the engine inside `reghdfe` and `fixest`). Each sweep is
linear in rows.

Frisch-Waugh-Lovell then says: regressing the residualised `y` on the
residualised `x` gives *exactly* the dummy-variable OLS coefficients. Not
approximately — `tests/test_econ_hdfe.py` checks coefficients and standard errors
against a hand-rolled dense reference to 1e-8, including the clustered sandwich.

```python
res = econ.hdfe(
    panel, y="ret", x=["size", "bm"],
    absorb=["firm", "date", "industry"],     # N-way
    vcov="driscoll-kraay", lags=-1,          # or cluster=[...], "robust", ...
)
res.params, res.std_errors, res.r2_within, res.df_absorbed
res.wald_test(["size", "bm"])
```

### Choosing a covariance

| Situation | Setting |
| --- | --- |
| Textbook baseline | `vcov="classical"` |
| Heteroskedasticity only | `vcov="robust"` (HC1) |
| Shocks correlated within a firm over time | `cluster="firm"` |
| …and within a date across firms | `cluster=["firm", "date"]` (Cameron-Gelbach-Miller) |
| Arbitrary cross-sectional dependence, long `T` | `vcov="driscoll-kraay"` |

Two-way clustered and Driscoll-Kraay covariance matrices are not positive
semi-definite by construction; `psd_clip=True` (the default) clips negative
eigenvalues to zero so the reported standard errors are always real.

### Residuals as features

```python
tr = econ.HDFETransformer(columns=["ret", "volume"], absorb=["firm", "date"])
tr.fit(train)            # learns per-level offsets on TRAIN ROWS ONLY
tr.transform(test)       # applies the frozen offsets
```

`fit` accumulates, for each absorbed dimension, the mean it removed for every
level. Those offsets are the learned parameters. On the training rows,
`value - sum_g offset_g[level]` reproduces the alternating-projection residual
exactly; on held-out rows it is a genuine out-of-sample projection. A firm that
never appears in training gets an offset of zero (`unseen="zero"`) or `null`.

## When slopes are not the same for everyone

Pooled OLS assumes one slope. Panels rarely oblige, and if a **common factor**
drives both the regressor and the outcome, ignoring it biases everything.

```python
econ.mean_group(panel, y="y", x=["x"])   # heterogeneous slopes, no factor
econ.cce_mg(panel, y="y", x=["x"])       # + common factor  (recommended default)
econ.cce_pooled(panel, y="y", x=["x"])   # pooled variant, entity-clustered SEs
```

Pesaran's Common Correlated Effects trick is disarmingly simple: add the
**cross-sectional averages** of `y` and `x` at each date to every entity's
regression. Those averages span the unobserved factor space asymptotically, so
including them annihilates the factor loadings — no factor extraction, no
eigenvalue problem. In simulation, Mean Group misses the true coefficient by
70% while CCE-MG lands within 10%.

If the interesting relationship is a **long-run equilibrium** that everyone
shares while adjustment speeds differ, that is Pooled Mean Group:

```python
res = econ.pmg(panel, y="y", x=["x"])
res.params                      # the common long-run vector theta
res.per_entity["ec_speed"]      # each entity's speed of adjustment (a feature!)
```

`pmg` alternates two concentrated least-squares steps — per-entity short-run
regressions given `theta`, then one pooled update of `theta` given the short-run
parameters — until `theta` stops moving.

### Diagnostics

```python
econ.cips(panel, value="y")          # unit root, robust to a common factor
econ.ips(panel, value="y")           # heterogeneous roots
econ.llc(panel, value="y")           # one common root
econ.pesaran_cd(panel, value="resid")  # cross-sectional dependence
```

`pesaran_cd` on your *residuals* is the decision rule for the section above: if
it rejects after you have absorbed fixed effects, a factor is still there and you
want CCE rather than FE.

!!! note "Where the critical values come from"
    The LLC / IPS / CIPS null distributions are simulated, not tabulated: for the
    same `(N, T, lags, trend)` the package Monte-Carlos the identical estimator
    pipeline under an independent-random-walk null (cached, seeded, deterministic)
    and standardises against those moments. Increase `reps` for tighter tails.

## Fama-MacBeth, done per date

```python
res = econ.fama_macbeth(
    panel, y="fwd_ret", x=["beta", "size", "bm"],
    winsorize_limit=0.01, standardize=True, lags=-1,
)
res.summary()          # mean lambda, Newey-West SE, t, p
res.lambdas            # the per-date lambda series
```

The one thing that matters for leak-safety here: **winsorising and standardising
run inside each date's cross-section, never over the pooled sample.** A global
z-score makes an observation from 2011 depend on the mean and standard deviation
of data through 2024. A per-date z-score uses only that date. The test suite
proves it: perturbing the last date's characteristics leaves every earlier
date's lambda bit-identical.

`FamaMacBethTransformer` goes further and freezes the *train-sample* mean lambda,
emitting `fm_pred` (the premia-implied expected return) and `fm_resid` (the
characteristic-adjusted alpha) with coefficients that never saw a test row.

## Who transmits shocks to whom

```python
feats = econ.rolling_connectedness(
    panel, value="realised_vol", window=200, lags=2, horizon=12,
)
panel.join(feats, on=["entity", "time"], how="left")
```

Fit a VAR to the cross-section, invert it to its moving-average form, and split
each series' `H`-step forecast error variance into the shares attributable to
shocks in every series. The resulting table is a directed network:

- **row `i`** — where `i`'s uncertainty comes *from*;
- **column `j`** — how much `j` transmits *to* others;
- **net** — `to − from`: transmitter or receiver;
- **total** — the share of variance crossing entity boundaries.

The decomposition is **generalised** (Koop-Pesaran-Potter; Pesaran-Shin) rather
than Cholesky, so the answer does not depend on how you ordered the series —
which is exactly why Diebold and Yilmaz switched to it in 2012.

Rolling windows are strictly trailing, so `dy_net` at date `t` uses no
observation after `t`. Expect nulls for the first `window - 1` dates.

## Predictability when the predictor barely mean-reverts

The dividend yield has an autoregressive root near one, and its innovation is
strongly negatively correlated with returns. Under those conditions the OLS
t-test on "does the yield predict returns?" rejects a *true* null around 25% of
the time at a nominal 5%. That is not a small correction; it is most of the
predictability literature.

```python
res = econ.ivx(y=returns[1:], x=dividend_yield[:-1])
res.summary()      # ivx_wald, p_value, and the ols_t it replaces
```

IVX instruments the persistent regressor with a **mildly integrated** transform
of its own increments, `z_t = sum_j (1 - c/n^delta)^(t-j) Δx_j`. The instrument
stays correlated with the regressor but is far enough from a unit root that the
estimator has a mixed-normal limit and a chi-squared Wald statistic — uniformly,
with no pre-test on the persistence. In the package's own simulations the IVX
test rejects 5-7% of the time where OLS rejects 26%.

As a screen over many candidate predictors:

```python
sel = econ.IVXSelector(y="fwd_ret", k=20).fit(train)
sel.ranking_        # feature / ivx_wald / p_value / n_entities
sel.transform(test) # the frozen top-20
```

## The effect of one feature among hundreds

```python
res = econ.dml_partial_linear(
    panel, y="fwd_ret", d="my_signal",
    n_folds=5, horizon=5, embargo=5,        # overlapping 5-day target
)
res.theta, res.conf_int
```

Two ideas, both necessary:

1. **Orthogonalisation.** Partial the controls out of both `y` and `d` and
   regress residual on residual. The score's derivative with respect to the
   nuisance functions is zero at the truth, so first-order errors in them do not
   propagate into `theta`.
2. **Cross-fitting.** The nuisance predictions for a row come from a model that
   never saw that row — otherwise overfitting bias returns through the back door.

Panelary's contribution is the third idea: those cross-fitting folds are
**purged and embargoed**. A random `KFold` would put tomorrow in the training set
and today in the test set. With an overlapping-horizon target — an `h`-period
forward return, a local projection — pass `horizon=h` so overlapping labels are
purged, and set `embargo` to at least `h`.

`post_double_selection` is the LASSO-based alternative and is often the better
choice when you want to *see* which controls mattered:

```python
res = econ.post_double_selection(panel, y="fwd_ret", d="my_signal")
res.selected          # union of the two selections
```

Selecting on the treatment equation as well as the outcome equation is the whole
point. A control that predicts `d` strongly but `y` only weakly would be dropped
by single selection, and omitting it biases `theta` — the classic way naive
LASSO-then-OLS goes wrong.

!!! warning "LASSO shrinkage leaks into theta"
    The default nuisance learner is `"post-lasso"`: rigorous (plug-in) penalty
    selection followed by an **unpenalised OLS refit**. Plain LASSO predictions
    are shrunk, and that shrinkage bias survives orthogonalisation — in
    simulation it moved the estimate from 1.00 to 0.87 and dropped coverage from
    95% to 32%. Use `learner="lasso"` only if you know why you want it.

## Leak-safety checklist

| Rule | How it is enforced here |
| --- | --- |
| Fit on train rows only | Every estimator subclasses `PanelTransformer`; `_fit` sees only what you hand it |
| No refitting inside `transform` | `HDFETransformer`, `FamaMacBethTransformer`, `PanelSlopeFeatures`, `DoubleMLTransformer` all apply frozen state |
| Standardise per date, not globally | `fama_macbeth(standardize=True)` operates inside each `group_by(time)` |
| Rolling features look backwards only | `rolling_connectedness` windows end at the stamped date |
| Overlapping targets need an embargo | `dml_partial_linear(horizon=h, embargo=h)` |
| Cross-sectional contemporaneous is fine | `CrossSectionalAverages` uses date `t` rows only |

## See also

- [Econometrics API reference](../api-reference/econ.md)
- [Cross-validation](../api-reference/cross-validation.md) — the purged and
  embargoed splitters DML borrows
- [Latent factors](factors.md) — the other way to handle a common factor
