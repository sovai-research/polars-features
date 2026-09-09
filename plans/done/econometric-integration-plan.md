# Implementation Plan — Econometric integrations: honest validation, causal features, panel estimators

**Status:** design / ready to scope · **Owner:** TBD · **Target branch:** `feat/econometrics`
**Author of plan:** research + design session, 2026-09-04
**Sibling plans:** [factor-extraction-plan.md](factor-extraction-plan.md) (`reduce/` — HFA) · [shap-attribution-plan.md](shap-attribution-plan.md) (`explain/` — Shapley) · [interactions-theme-plan.md](interactions-theme-plan.md)

---

## 0. TL;DR — the decision

Add econometric capability along **three coherent workstreams**, in ROI order, plus a clear **depend-vs-port** stance on the exemplar Rust library `tsecon`:

1. **The leak-safe validation & selection layer (do first).** Cheap, mostly native, highest credibility-per-line: it turns "leak-safe" from a *claim* into a *measurable guarantee* — CPCV + Deflated Sharpe/PBO + Romano-Wolf/FDR + the DM→SPA→MCS chain. This is the finance-native complement to the existing leakage verifier and the natural home for the guarantees `explain/` and `reduce/` both rely on.
2. **Cheap causal feature generators.** Backward-looking by construction, all map to Polars `group_by`/`rolling`, all leak-safe by design: unit-root/long-memory screening feeding the existing `_ffd`, HAR-RV/realized-vol, STL/MSTL, Nelson-Siegel, Amihud/Roll liquidity, Fama-MacBeth + characteristic betas, EVT tails.
3. **Panel-native differentiators (the moat).** Fast **HDFE** fixed effects, **CCE/Mean-Group/PMG** heterogeneous panels, **Diebold-Yilmaz connectedness** — several with *no mature Python implementation*, whose by-products are first-class panel features.

> **This is the fourth pillar of the library:** features → **factors** (`reduce/`) → **attribution** (`explain/`) → **honest validation** (this plan). The validation layer under-writes the leak-safety claims of the other three.

### The `tsecon` stance (settles "what to port")
`tsecon` (https://github.com/cacoleman16/tsecon) is dual **MIT/Apache-2.0**, ships as a **single NumPy-only abi3 wheel**, and its **array-in/array-out boundary matches exactly how Panelary already crosses into its estimator layer**.

- **Depend on the wheel first** (near-zero friction) to (a) benchmark against and (b) **validate Panelary against tsecon's golden fixtures** — which directly fixes the two known bugs: **DSR N/V** and **frac-diff divergence** (see `panelkit-wave2-academic.md`; two independent agents flagged both).
- **Vendor source** (license permits) only for pieces Panelary needs **native + panel-partitioned**: the leak-safe CV/backtester, the panel trio (CCE/MG/PMG) + panel unit-root tests, fractional differencing (long-memory crate), and the shared **Philox-RNG / HAC / bootstrap / SSM** foundation crates that thicken Panelary's currently ~218-LOC Rust core.
- **Elsewhere prefer lighter sources:** `polars-ds` as the Polars-plugin *template* (and a dependency for basic stat tests / entropy / PSI); `augurs` for MSTL/ETS/DTW/changepoint; `arch` (wrap, don't rebuild) for SPA/MCS/bootstrap.

---

## 1. Depend-vs-port matrix (the actionable core)

| Capability | Decision | Why |
|---|---|---|
| Leak-safe CV / backtester contract | **vendor + reimplement native** (validate vs tsecon fixtures) | Panelary's founding thesis; must be panel-partitioned in-house |
| HAC / Newey-West | **vendor tsecon HAC crate** | shared primitive everything else calls |
| TS bootstrap (block/stationary/wild/sieve) | **vendor tsecon** or wrap `arch` | engine under CV bands + SPA/MCS |
| DM / SPA / Model Confidence Set | **wrap `arch`** | mature, don't rebuild |
| CPCV, Deflated Sharpe, PBO, Romano-Wolf, BH/BY | **build native** | small, and the differentiating "honesty" layer |
| Unit-root battery (incl. DF-GLS/Ng-Perron/Zivot-Andrews) | **cherry-pick from tsecon diagnostics** | many overlap existing extractors; take the gaps |
| Frac-diff + GPH/local-Whittle `d` | **vendor tsecon long-memory** | fixes the frac-diff bug; makes `_ffd` data-driven |
| HAR-RV / realized vol, Nelson-Siegel, Amihud/Roll, Fama-MacBeth, EVT | **build native** | cheap Polars `group_by`/`rolling`; no dependency warranted |
| STL/MSTL, ETS, DTW, changepoint | **depend on `augurs`** | pure-Rust, Polars-friendly, cheaper than tsecon's macro stack |
| HDFE fixed effects + clustered/Driscoll-Kraay SEs | **build native** | the "fast" headline; group-by demean + Rust |
| CCE / Mean-Group / PMG + panel unit roots | **vendor tsecon panel trio** | no mature Python equivalent — differentiator |
| Diebold-Yilmaz connectedness (FEVD) | **port (R-only)** | cross-entity network features; differentiator |
| IVX predictive regression | **port (R-only)** | valid predictability under persistence |
| Double ML + cross-fitting + PDS-LASSO | **build native** (reuse splitter) | reuses Panelary's own leak-safe splitting |
| TS conformal (ACI/PID, EnbPI, NexCP, CQR) | **build native** / vendor tsecon conformal | upgrades existing `conformal.py` to be temporally valid |
| basic stat tests / entropy / PSI / AR-coef features | **depend on `polars-ds`** | already pure-Rust Polars, reusable |

---

## 2. Module layout

Most of this lands in a new `validation/` package plus extensions to existing modules (not one monolith):

```
panelary/
    validation/                 # NEW — the honest validation & selection layer (workstream 1)
        __init__.py
        _cv.py                  # CPCV + purged/embargoed splitters (reconcile with cross_validation.py)
        _selection_stats.py     # Deflated Sharpe, PBO/CSCV, Romano-Wolf, BH/BY
        _forecast_tests.py      # DM, SPA, MCS (wrap arch), CRPS/pinball/PIT
        _bootstrap.py           # block/stationary/wild/sieve (vendor tsecon or arch)
    econ/                       # NEW — panel & TS econometric estimators (workstream 3)
        __init__.py
        _hdfe.py                # high-dim fixed effects + clustered/Driscoll-Kraay SEs
        _panel.py               # CCE / Mean-Group / PMG (vendor tsecon)
        _famamacbeth.py         # Fama-MacBeth cross-sectional regression
        _connectedness.py       # Diebold-Yilmaz FEVD spillover features (port)
        _ivx.py                 # IVX predictive regression (port)
        _dml.py                 # Double ML + cross-fitting + PDS-LASSO
    feature_extractors.py       # EXTEND — unit-root battery, HAR-RV, Nelson-Siegel, Amihud/Roll, EVT, d-estimation
    _ffd.py                     # EXTEND — wire GPH/local-Whittle d; adopt tsecon golden to fix divergence bug
    conformal.py                # EXTEND — ACI/PID, EnbPI, NexCP, CQR
```

**Reconcile, don't duplicate:** Panelary already has `cross_validation.py`, `backtesting.py`, `conformal.py`, `evaluation.py`, `testing.py`. Audit each against the plan — extend where a home exists (conformal, CV), add `validation/`/`econ/` only for genuinely new surface. Import boundary: `validation/` and `econ/` are Tier-2 (estimator layer) — may import `core/`, NumPy, optional `tsecon`/`arch`/`polars-ds`; Tier-1 `namespaces/` must not import them.

---

## 3. Workstream 1 — honest validation & selection (do first)

Everything here is cheap, mostly native, and consumes **out-of-sample** results only.

- **`_cv.py`** — Combinatorial Purged CV (López de Prado): test on all C(N,k) purged+embargoed group combinations → a *distribution* of backtest paths, not one. Reconcile with the existing purge/embargo in `cross_validation.py`; this becomes the default splitter feeding DML and conformal.
- **`_selection_stats.py`:**
  - **Deflated Sharpe Ratio** (Bailey–López de Prado): deflate Sharpe for #trials, sample length, skew/kurtosis. ⚠️ **Fix the known DSR N/V bug** on integration; gate with tsecon golden values.
  - **PBO / CSCV**: probability the in-sample-best underperforms the OOS median.
  - **Romano-Wolf stepdown** (FWER) + **BH/BY FDR** (BY valid under panel dependence): honest feature/strategy selection over correlated candidates.
- **`_forecast_tests.py`** — Diebold-Mariano, SPA, Model Confidence Set (**wrap `arch`**), plus proper scoring (**CRPS, pinball, PIT/reliability**) vectorized in Polars.
- **`_bootstrap.py`** — block/stationary/wild/sieve (vendor tsecon or `arch`); the resampling engine under CV bands and SPA/MCS. **Blocks must never straddle a fold boundary.**

**Acceptance:** CPCV lowers measured PBO vs walk-forward on a known-overfit fixture; DSR/PBO match tsecon (or published) golden values; Romano-Wolf controls FWER on a simulated multiple-testing panel.

---

## 4. Workstream 2 — cheap causal feature generators

All backward-looking, all Polars `group_by`/`rolling`, all leak-safe by construction. Extend `feature_extractors.py`.

- **Unit-root / stationarity battery** — ADF, KPSS, Phillips-Perron, **DF-GLS, Ng-Perron, Zivot-Andrews** (structural break). Dual use: preprocessing (choose differencing per entity, **train-only**) and cheap persistence/regime features (test stat, p-value, break date).
- **Long-memory `d` → `_ffd`** — GPH + local-Whittle estimators of the fractional-integration `d`; feed the existing fixed-width frac-diff so differencing is **data-driven per entity/window**. Vendoring tsecon's long-memory golden fixtures fixes the frac-diff divergence bug in the same change.
- **HAR-RV + realized vol** — realized variance, bipower variation, jump component, HAR daily/weekly/monthly terms. Pure lag-aggregation + one OLS per entity.
- **STL/MSTL** (via `augurs`) — trend/seasonal/remainder + seasonal-strength features. **Must be causal** (trailing-window or fit-on-train-then-project — naive two-sided STL leaks).
- **Nelson-Siegel** — level/slope/curvature from the maturity cross-section per date (contemporaneous → leak-safe).
- **Microstructure liquidity** — Amihud illiquidity, Roll spread (trailing rolling; ultra-cheap).
- **EVT tails** — POT-GPD / Hill tail index, EVT-VaR/ES per trailing window.
- **Fama-MacBeth + characteristics** — per-date cross-sectional regression (group-by-time), characteristic z-scores/sort buckets, rolling betas. Cross-sectional ranking per date is contemporaneous; **winsorize/standardize per-date, never globally.**

**Acceptance:** each feature recovers its known value on a simulated series; a leakage test proves each is invariant to future rows.

---

## 5. Workstream 3 — panel-native differentiators (the moat)

- **`_hdfe.py`** — high-dimensional fixed effects via **alternating projections** (reghdfe/Gaure) — near-linear, maps perfectly to Polars group-by demeaning + Rust. Deliver one/two-way + N-way absorption; **clustered (one/two-way) and Driscoll-Kraay** SEs. Residualized (partialled-out) columns are leak-controlled **features**. This is the "fast" headline.
- **`_panel.py`** — **CCE / CCE-MG, Mean-Group, PMG** (vendor tsecon panel trio) + panel unit-root (LLC/IPS/CIPS) and Pesaran CD. No mature Python equivalent → genuine differentiation. By-products are features: per-entity slopes, cross-sectional-average factor proxies (CCE augmentation terms).
- **`_famamacbeth.py`** — the finance staple (also feeds workstream 2); per-period λ time series as signals, Newey-West on the coefficient series.
- **`_connectedness.py`** — Diebold-Yilmaz generalized-FEVD spillover network (port from R): total/directional/net connectedness + per-entity centrality as **cross-entity features** no competing feature library offers. Needs a rolling VAR backend (reuse tsecon's or a native one).
- **`_ivx.py`** — IVX predictive regression (port from R): valid inference on predictability under persistent regressors; the IVX-Wald stat is a leak-aware **screening score** for candidate features.

**Acceptance:** HDFE matches `pyfixest`/`fixest` coefficients + SEs on a fixture; CCE/PMG match `plm`; connectedness matches the R `frequencyConnectedness` reference; all fit train-only inside CV.

---

## 6. Workstream 4 — modern inference & uncertainty (fast follow)

- **`_dml.py`** — Double/Debiased ML with **cross-fitting** (reuse `validation/_cv.py` so nuisance folds are themselves leak-safe) + **post-double-selection LASSO** with rigorous/plug-in penalty. Valid feature-effect inference over the wide panel feature matrix.
- **`conformal.py` extensions** — **ACI / Conformal-PID** (coverage under drift), **EnbPI** (ensemble batch intervals), **NexCP** (weighted/non-exchangeable), **CQR** (conformalized quantile regression). Upgrades naive (exchangeability-assuming) conformal to be genuinely time-series/panel-valid; calibration split must be purged/embargoed.

---

## 7. Leak-safety guardrails (bake in from day one)
1. **Filtered, not smoothed** — any Markov-switching/HMM regime feature must use `t|t` (filtered) not `t|T` (smoothed) probabilities.
2. **Causal, not offline** — STL and change-point detection must run on trailing windows / online, never two-sided over the full series.
3. **Train-only fitting inside CV** — transform choice (unit-root → differencing), factor extraction, GARCH params, covariance/shrinkage, Fama-MacBeth standardization stats.
4. **Bootstrap blocks never straddle a fold boundary**; block length respects autocorrelation.
5. **Overlapping-horizon targets** (local projections, long-horizon returns) require embargo of the horizon length.

Every estimator here that fits parameters subclasses `PanelTransformer` (or composes with it) and declares `panel_safe` / `leakage_safe`; the workstream-1 tests are the enforcement.

---

## 8. Milestones & acceptance criteria

| # | Milestone | Deliverable | Done when |
|---|---|---|---|
| M1 | **Honest validation layer** | `validation/` (CPCV, DSR⚠️fix, PBO, Romano-Wolf, BH/BY, DM/SPA/MCS via arch, CRPS) | CPCV lowers PBO vs walk-forward on overfit fixture; DSR matches golden; FWER controlled |
| M2 | **Frac-diff fix + memory features** | `_ffd` wired to GPH/local-Whittle; unit-root battery | frac-diff matches tsecon golden (bug closed); `d` recovered on ARFIMA sim |
| M3 | **Cheap feature battery** | HAR-RV, STL/MSTL (augurs), Nelson-Siegel, Amihud/Roll, EVT, Fama-MacBeth | each recovers known value + passes leakage test |
| M4 | **HDFE + SEs (the fast headline)** | `econ/_hdfe.py` | matches pyfixest coeffs/SEs; residual features leak-safe in CV |
| M5 | **Panel trio + connectedness** | `econ/_panel.py` (vendor tsecon), `_connectedness.py` (port) | match plm / frequencyConnectedness references |
| M6 | **Conformal upgrade + DML** | ACI/EnbPI/NexCP/CQR; `_dml.py` + PDS-LASSO | temporal coverage holds under drift; DML CI valid on sim |
| M7 | tsecon golden-fixture harness | CI job validating Panelary estimators vs tsecon | green, gating regressions |

**M1 is the must-ship, highest-ROI pillar.** M2 closes two known bugs. M3–M5 are the feature/moat build-out. M6 is the fast follow.

---

## 9. Dependencies & licensing
- **New optional deps:** `tsecon` (MIT/Apache — depend on wheel; vendor source per §1), `arch` (NCSA/BSD — SPA/MCS/bootstrap), `augurs` (MIT/Apache — MSTL/ETS/DTW), `polars-ds` (MIT — stats/entropy/PSI). All permissive.
- **Vendoring** tsecon crates is allowed under MIT/Apache with attribution (add to `THIRD-PARTY-LICENSES` / `NOTICE`).
- **Ports** (IVX, Diebold-Yilmaz) are clean-room from the papers (R sources are GPL — do **not** copy; implement from Kostakis-Magdalinos-Stamatogiannis and Diebold-Yilmaz directly).

---

## 10. Out of scope (scope discipline)
- Full DCC-GARCH and vine copulas (heavy, scale poorly with N) — revisit on demand.
- Bayesian VARs, FAVAR, threshold/STAR dynamics — tsecon's macro stack; depend on the wheel if ever needed, don't reimplement.
- Full ARFIMA MLE — use `d`-estimation + the existing fast FFD instead.
- Causal forests / modern DiD (Callaway-Sant'Anna, synthetic DiD) — a different (causal-inference) product; only if Panelary deliberately moves there.

---

## 11. Reference map
- **Exemplar:** [tsecon](https://github.com/cacoleman16/tsecon) (MIT/Apache, single wheel, golden-gated) · [augurs](https://github.com/grafana/augurs) · [polars-ds](https://github.com/abstractqqq/polars_ds_extension) · [arch](https://github.com/bashtage/arch).
- **Validation/selection:** López de Prado *Advances in Financial ML* (CPCV, Ch.7/12); Bailey–López de Prado (Deflated Sharpe, PBO); Romano–Wolf (Econometrica 2005); Hansen SPA / Hansen-Lunde-Nason MCS.
- **Panel:** `pyfixest`/`fixest`/`reghdfe` (HDFE); `plm`/`dcce`/`xtdcce2` (CCE/PMG); `linearmodels` (Fama-MacBeth, Driscoll-Kraay).
- **TS/features:** Corsi HAR-RV; Diebold-Li (Nelson-Siegel); GPH / Robinson local Whittle; Zivot-Andrews; Diebold-Yilmaz connectedness; Kostakis-Magdalinos-Stamatogiannis (IVX).
- **Modern inference:** Chernozhukov et al. DML; Belloni-Chernozhukov-Hansen PDS-LASSO; Gibbs-Candès ACI; Angelopoulos et al. Conformal-PID; Xu-Xie EnbPI; Barber et al. NexCP; Romano-Patterson-Candès CQR.
- **Known bugs to close:** DSR N/V and frac-diff divergence (`panelkit-wave2-academic.md`, a research memo kept under its pre-rename name).

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
