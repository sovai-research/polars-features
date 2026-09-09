# Roadmap & Vision

!!! warning "Forward-looking — not shipped"
    This page describes **planned and proposed** work, not the current feature
    set. Nothing here is a commitment, an API contract, or something you can
    `import` today. For what actually ships now, see the
    [API Reference](api-reference/feature-extractors.md). Names, signatures, and
    priorities below are design sketches and will change.

Panelary already owns the hard, leak-safe primitives — a clean-room
catch22/catch24, ~120 tsfresh-style extractors, a one-pass leak-safe
`extract_features`, purged / combinatorial-purged cross-validation, a
provenance-carrying feature registry, per-cross-section OLS neutralization, and
the `assert_no_lookahead` future-perturbation harness. The vision below is about
**packaging and completeness**: turning those primitives into "one obvious way"
to go from a raw panel to an honestly-validated model, without ever hand-wiring a
leak.

The organizing principle across every area is the same: **the safe path is the
default, the naive/pooled path is opt-in**, and leak-safety is *carried* by the
objects (flags on transforms, a `t1` span on the frame) rather than re-argued at
each call.

---

## Features

Make feature extraction collapse to naming a **preset** and getting a wide,
provenance-named, leak-safe frame.

- **Named presets** (`minimal` / `efficient` / `comprehensive`) as frozen,
  introspectable objects — turns kwarg soup into "start simple, tune later".
- **Cost/block tags** on every `FeatureSpec` — power preset expansion and cost
  ceilings without new numerics.
- **Mixed `features=` resolution** — accept presets, block names, and individual
  feature names in one list, deduped and stably ordered.
- **Always-on distributional moments block** — mean/std/skew/kurtosis/quantiles
  in pure Polars, so even `minimal` returns a strong, free baseline.
- **Welch spectral block** — entropy, centroid, spread, rolloff, flatness,
  dominant frequency, band-power ratios; fills the confirmed FFT gap via scipy
  structs computed inside the single agg.
- **catch24 block wiring + parity pass** — resolve the `NEEDS REVIEW` features to
  C-implementation numerical parity.
- **MiniRocket** (`RocketConfig` + `rocket_features()`) — a deterministic Rust
  PPV plugin over an equal-length windowing utility; best accuracy/speed, and
  deterministic so it is unit-testable against a reference vector.
- **Deterministic name provenance** — `<block>__<feature>[__param=value]` naming
  with a `from_columns()` round-trip for train/serve parity.

## Cross-sectional / Factor

Rank a signal, neutralize it, and evaluate IC and decile spread in ~6 lines,
with forward-return alignment done for you and per-date discipline impossible to
forget.

- **Rank-normalization presets** (`unit` [-1,1] GKX, `centered`, `uniform` FNW) —
  affine shifts of the existing rank; the single most-used op in the EAP
  literature.
- **`forward_return` leak-safe aligner** — the one audited negative-shift site,
  gap-guarded; the #1 caller-side leak, so it must exist before any evaluator is
  trustworthy.
- **`standardize` sugar** (winsorize -> z/rank) — compresses the universal "clean
  then scale" recipe to one call.
- **`ic` / rank-IC / ICIR evaluator** — field-default signal-quality metric as a
  per-date correlation with a five-number summary.
- **Industry group-demean/zscore fast paths** (`by=`) — cheap sector
  neutralization without OLS.
- **within / between / first-difference triad** and **gap-aware range-spec lag**
  — canonical panel decompositions as named, gap-safe transforms.
- **`portfolio_sort` / decile long-short** and **`fama_macbeth`** (per-date OLS +
  Newey-West t) — the canonical characteristic-sort backtest and the two-namespace
  showcase.
- **Two-way FE demeaning** (alternating projections) — a pure-Polars reference
  first, then a Rust MAP kernel as the speed flex.

## Labeling & Weights

Unify labels, sample weights, and CV purge under one invariant so they can never
disagree.

- **`_spans_from_t1` canonical span table** — one shared representation that
  purge, concurrency, and weights all read from; the architectural invariant of
  the whole area.
- **Event-stream concurrency** (`+1 @ t0`, `-1 @ t1+1`, `cum_sum`) — `O(N log N)`
  average-uniqueness without the `O(N^2)` mlfinlab loop.
- **`weights.return_attribution` + `attach`** — the headline sample-weighting
  layer, feeding the already-present `sample_weight=` estimator arg.
- **`weights.time_decay`** on cumulative uniqueness and **`class_weights`** — for
  non-stationarity and imbalance.
- **`label.trend_scanning`** — most-requested modern label via a closed-form
  max-|t-stat| slope sweep, yielding a natural weight.
- **`label.excess_over_median`** — a one-expression, panel-native cross-sectional
  label.
- **`sample.cusum`** event sampler and **`sequential_bootstrap`** — fewer, more
  unique labels and max-uniqueness bagging.
- **`sizing.bet_size`** (`m = 2*Phi(z) - 1`, average-active, discretized) —
  completes the side -> size story with meta-labeling.
- **CV weight plumbing** — thread a `weight=` column, re-weight **fold-locally**,
  and score weighted.

## Validation & Leak-Audit

This is Panelary's moat. Fix the numerics first, then build the one-call panel CV
runner and the panel leak auditor — the biggest white space no competitor
occupies.

- **`cross_validation(df, ...)` one-call runner** — returns a tidy long OOS frame
  plus `path_metrics()` for CPCV.
- **Three-unit embargo/horizon spec** — accept `"5D"`, an integer observation
  count, or a fraction of the time axis.
- **Performance-inference stats surface** — probabilistic Sharpe ratio, minimum
  track-record length, a minimum-backtest-length guard, `haircut_sharpe`
  (Bonferroni/Holm/BHY), `effective_n_trials` (ONC clustering), and
  stationary-bootstrap Sharpe CIs / difference tests.
- **`pn.audit()` panel leak auditor** — a named-check catalog on the perturbation
  engine, adding the panel-specific checks nobody else has: cross-sectional
  as-of leaks, cross-entity contamination, target leakage, label-horizon
  overlap, and **placebo/shuffle refutation** — all CI-gateable.

## Selection

Turn purged CV + per-cross-section OLS into a full, leak-safe selection catalog
where the safe path is the default.

- **`select_k_best` + `SelectKBest`** — the headline one-liner; a purged 5-fold is
  auto-built when `cv=` is omitted.
- **`clustered_importance` (CFI)** — cures the substitution effect that fools
  single-feature importance.
- **`relevance_table`** (FRESH + Benjamini-Yekutieli), with optional
  **block/cluster-robust p-values** — the genuinely novel panel upgrade.
- **`boruta` with panel shadows** and **`shap_rfe`** — all-relevant and
  wrapper-style selection under purged CV.
- **Prediction-stage neutralization** — `proportion=` on `.xs.neutralize`,
  `neutralize_predictions` / `FeatureNeutralizer`, and a
  `feature_neutral_correlation` metric (Numerai FNC).
- **feature-engine-style transformer catalog** — SelectByShuffling, RFE/RFA,
  SmartCorrelated/DropCorrelated, RelevanceFilter, with sklearn
  `get_feature_names_out` interop.

## Imputation

Do not rebuild classical imputers as rivals to CAFE; expose orthogonal
primitives, build the uncertainty moat, and fence every future-using method
behind a uniform leak gate.

- **`ffill_staleness`** — PIT forward-fill plus a `__staleness` signal and a
  `max_staleness` cap that kills stale-carry bias.
- **`missing_mask`** — first-class `__was_missing` (and optional gap-length)
  indicators; theoretically load-bearing under MNAR.
- **`CafeImputer` by-product kwargs** — surface per-cell `sigma`, recoverability,
  and anomaly flags, plus a `posterior()` hook.
- **`mi_cross_validate` (the moat)** — a Rubin's-rules multiple-imputation CV
  runner that propagates imputation uncertainty into backtest scores (within /
  between / total variance, FMI, relative efficiency); no mainstream lib does
  this.
- **`LowRankImputer`** (expanding-SVD safe vs batch leaky) and **`KalmanImputer`**
  (filter-only, with Gaussian posterior variance) — a CAFE-free fallback and a
  principled single-series UQ source.
- **`imputation_report`** — % missing, gap length, staleness, recoverability, and
  an `assert_no_lookahead` PASS/FAIL for the chosen imputer.

## Models & Stationarity

GBDT stays the default (trees dominate the small-fold, noisy-feature, regime
regime that purged CV produces); everything else is opt-in accuracy/uncertainty
tooling.

- **`RocketFeatures` + `PanelRocket*`** — MiniRocket -> ridge; the unsupervised
  transform is leak-free and drops into GBDT. Highest model ROI.
- **`PanelStackingRegressor/Classifier`** — OOF meta-features under the *same*
  purge+embargo; turns the CV moat into a model.
- **`PanelConformalRegressor`** — panel prediction intervals (EnbPI +
  cross-sectional split), calibration always purged, with a `coverage` attribute
  stating which guarantee holds.
- **`FracDiffAutoD` + `select_d_adf`** — fold-safe fractional differencing where
  the differencing order `d` is chosen on train rows only and frozen in
  transform; the signature quant-ML differentiator.
- **`PanelTabPFN*`** (small-fold, GBDT fallback) and caveated
  **`PanelForecastBaseline`** zero-shot adapters (with a loud pretraining-
  contamination warning).
- **Rust acceleration order** — rolling-reductions via incremental sufficient
  stats (`O(n*w) -> O(n)`, touches everything), grouped/rolling OLS via `faer`,
  spectral via RustFFT, all wrapped in a rayon per-entity fan-out.

## API & Developer Experience

Tie every area into one coherent product with a single mental model: load ->
schema -> impute -> features -> label+weights -> select -> model -> validate/audit.

- **sklearn compatibility** — `feature_names_in_`, `get_feature_names_out()`,
  `get_params`/`set_params` (with `step__param` in `Pipeline`), `set_output`, and
  `get_fitted_params` for free `clone` / `GridSearchCV` interop.
- **Typed frozen configs** (attrs) — `CVConfig`, `WindowConfig`, `LabelConfig`,
  `FeatureConfig`; a config fully determines a leak-safe run and serializes for
  reproducibility.
- **`PanelSchema` boundary validation** — validate the `(entity, time)` contract
  once, then stay lazy and trusted; makes the invariants leak-safety depends on
  explicit and enforceable.
- **`pn.presets.{area}.{minimal,default,comprehensive}`** — the same three-tier
  vocabulary across every area, keeping a growing surface learnable.
- **Registry `(namespace, name)` keying** — lets `panel.zscore` and `xs.zscore`
  coexist honestly and unblocks preset/block resolution.
- **`panel_safe` reconciliation** — a `requires_over` field and registry-derived
  class flags so the safety flags can never quietly drift from the code.

---

## Unsupervised methods & SovAI SDK ports

A dedicated design pass covers the unsupervised toolkit — clustering,
dimensionality reduction, unsupervised feature selection, anomaly/regime
detection, and pairwise networks — porting battle-tested methods from the SovAI
SDK. The governing principle: these methods almost universally *learn from data*
(fit a scaler, rotation, cluster centroids, or a detector threshold), so a naive
port would fit on the whole panel and leak the future into cross-validation.
Every port is therefore redesigned as a leak-safe `PanelTransformer` that fits on
the training fold only; genuinely offline/two-sided methods are shipped gated
behind `leakage_safe=False` (the pipeline refuses them across a train/test
boundary unless refit per fold) and paired with an expanding-window causal
companion.

- **Clustering (`cluster/`)** — a pure-NumPy **k-Shape** engine (FFT normalized
  cross-correlation + Shape-Based Distance) as `KShapeClusterer` for time-series
  shape, plus `CrossSectionalClusterer` for grouping entities by their same-date
  feature vectors. Centroids are learned state, so distances/labels become
  leak-safe features that refit per fold.
- **Dimensionality reduction (`reduce/`)** — `PanelPCA` and thin
  `PanelSVD`/`FactorAnalysis`/`RandomProjection`/`KernelPCA`/`NMF` subclasses
  (train-fit scaler + rotation, sign-fixed components, `pc_i` provenance),
  `StatisticalFactors` from a returns panel (expanding refit), optional
  `PanelUMAP`, and per-date `CrossSectionalPCA`.
- **Unsupervised selection (`select/` additions)** — **Principal Feature
  Analysis** (`PFASelector`: PCA loadings -> cluster feature vectors -> keep a
  representative per cluster), projection-energy importances (random projection,
  FastICA, TruncatedSVD, random Fourier), cheap variance/correlation selectors,
  and SHAP-based importance/RFE. Complements the supervised mRMR/MDA/MDI trio.
- **Anomaly, regime & change-point (`detect/`)** — reuses the existing Rust CUSUM
  for the causal path; adds adaptive `ImprovedCusumDetector`/`CusumRegime`,
  offline `PanelAnomaly` modes (local/global/cluster/reconstruction) and
  `RupturesRegime` (gated, with causal companions). Detector outputs
  (anomaly scores, regime labels, change-point flags) are first-class features.
- **Pairwise & networks (`pairwise/`)** — trailing-window `rolling_corr_network`,
  per-entity `connectedness`/centrality graph features, and a trailing `leadlag`
  similarity (fixing a forward-looking shift in the original). A rolling
  all-pairs co-moment Rust plugin is the highest-ROI acceleration in this set.
- **Signal evaluation (`factor/` additions)** — feature-set de-correlation
  (`orthogonalize`, per-date Gram-Schmidt/QR) distinct from Numerai-style
  neutralization, plus `turnover`, signal decay, and hit-rate metrics folded in
  behind the shared leak-safe forward-return alignment seam.

!!! note "First-party, leak-safe by construction"
    The SovAI methods are first-party code. The value of the port is not
    transcription — it is re-homing each method onto Panelary's fit-on-train
    contract and point-in-time discipline, so what was a descriptive research
    tool becomes a backtest-safe feature generator.

---

## Known correctness fixes in flight

These are confirmed defects with fixes designed and queued — the highest-priority
work because they affect the honesty of the numbers Panelary reports.

- **Deflated Sharpe `N` and `V`.** In the CPCV path reconstruction the Deflated
  Sharpe is deflated against `n_trials = n_paths` (a CV-geometry constant, not the
  number of strategy configurations searched) and `V` silently falls back to the
  single-strategy estimator variance instead of the empirical variance of the
  path Sharpes. Both errors inflate DSR — they make luck look like skill. Fix:
  thread an honest user-supplied `n_trials` (warn loudly and use `n_paths` only as
  a floor), pass `V = var(path_sharpes, ddof=1)`, centralise an
  annualisation-aware `_sharpe` that returns NaN on degenerate input, and rank PBO
  by Sharpe rather than block mean.
- **Frac-diff consolidation.** Three copies of the weight recursion across four
  surfaces can return different numbers. Fix: one `ffd_weights` + one causal
  `frac_diff_expr` in a dependency-free leaf module, route every surface through
  it, fix the Rust `threshold=0.0` footgun, and lock it all with an
  all-surfaces-agree parity test.
- **MDA within-entity permutation.** The permutation-importance shuffle currently
  permutes each feature column globally (across entities and time), which is
  off-manifold for panels and inflates importance. Fix: default to a
  **within-entity** shuffle (`permute_within="entity"`) so permuted values stay
  on the entity's own manifold.
- **`impute()` bfill/interpolate guard.** `bfill` and `interpolate` read the
  future but pass silently through CV because they are not `PanelTransformer`s.
  Fix: emit a `LeakageWarning` (gated behind an explicit `allow_leaky=True`) and
  steer users to the PIT `ffill_staleness` baseline.
