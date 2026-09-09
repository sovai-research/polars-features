# Implementation Plan — `explain`: leak-safe, panel-aware feature attribution (SHAP + interactions)

**Status:** design / ready to scope · **Owner:** TBD · **Target branch:** `feat/explain-attribution`
**Author of plan:** research + design session, 2026-09-04
**Companion:** [factor-extraction-plan.md](factor-extraction-plan.md) (the "interactions" theme connects the two)

---

> **Part of the "interactions" theme.** Shapley-IQ is the **order ≥ 2, model-side** half of a
> library-wide throughline: higher-order structure in the *model* (feature interactions) and
> in the *data* (HFA cumulant factors, [factor-extraction-plan.md](factor-extraction-plan.md)).
> See [interactions-theme-plan.md](interactions-theme-plan.md). The flagship demo of v3/M5 is
> the **factor-interactions** workflow — running Shapley interactions on HFA-derived factors to
> surface "factor 2 × factor 5 synergy drove this forecast," which no other library produces.

## 0. TL;DR — the decision

Do **not** reimplement SHAP. The pragmatic, defensible move for Panelary is a new **`explain/`** subpackage whose moat is **leak-safety and panel-awareness**, not the SHAP math:

> **Nobody ships leak-safe, panel-aware feature attribution.** Every SHAP library treats the background/reference set casually — which is precisely the dominant leakage surface. Panelary already has the leak-safe `PanelTransformer` contract; wrapping *exact, native* TreeSHAP inside it — with the background set as a fold-bound, time-respecting object — is a capability no other library has.

Three layers, shippable independently:

1. **v1 (the moat):** `TreeAttributor` — a leak-safe `PanelTransformer` that calls the **native exact TreeSHAP already built into the models you wrap** (XGBoost `pred_contribs`, LightGBM `CONTRIB`, CatBoost `ShapValues`, sklearn), emits attributions as Polars columns keyed by `(entity, time)`, with the **background set drawn only from the training fold and only from the past**, and **interventional-vs-conditional as an explicit knob**. Zero SHAP reimplementation; exact; fast.
2. **v2 (panel-native value-add):** group/window aggregation via Polars `group_by` (factor-bucket SHAP, per-entity rolling windows) exploiting SHAP additivity, plus **attribution-stability reporting** (separate reference-induced oscillation from real regime drift).
3. **v3 (the frontier, R&D):** any-order **Shapley interactions** via `shapiq`/TreeSHAP-IQ interop (ties directly to the HFA/factor "interactions" theme), and — the genuinely novel long game — a **native Arrow/Polars TreeSHAP kernel** as a `pyo3-polars` plugin (no Arrow-native SHAP exists anywhere), later extensible to the **sparse Möbius/Fourier (SPEX)** engine.

---

## 1. Why this shape (evidence from the research sweep)

- **Rust/Polars SHAP is greenfield.** No mature pure-Rust TreeSHAP crate exists; `rust_shap` is a brand-new undocumented KernelSHAP v0.1; `perpetual` is a Polars/Arrow-native Rust GBM that emits *contributions* but not confirmed-exact TreeSHAP; the only *real* exact TreeSHAP in Rust is FFI into XGBoost/LightGBM C++. **No Arrow/DuckDB-columnar-native SHAP kernel exists anywhere** — a genuine moat if we ever build one.
- **The leakage surface is the background set.** The subtlest, most common SHAP leak is sampling the reference/background from the whole dataset (or from the future). Even "no-background" path-dependent TreeSHAP leaks if the model saw future rows. A panel library that makes the background a **fold-bound, time-respecting, first-class object** is doing something no SHAP library does.
- **Financial features are collinear → interventional-vs-conditional is a real decision,** and **group/window aggregation** is the practical antidote (kills multicollinearity instability, cuts compute, maps to `group_by`). Marginal SHAP + a naive background evaluates the model off-manifold (e.g. high-momentum + falling-price) → nonsense attributions.
- **Don't reinvent the math.** Exact TreeSHAP (and pairwise interactions) already ship inside XGBoost/LightGBM/CatBoost. `shapiq` is the maintained hub for *any-order* interactions (TreeSHAP-IQ, KernelSHAP-IQ, SVARM-IQ) — interop, don't reimplement.
- **The frontier worth tracking:** sparse Möbius/Fourier recovery (**SPEX/ProxySPEX**) is the only line that scales *interactions* past ~20 features, and it has a native-Rust angle (`fwht`/`rustfft` + a SPRIGHT-style peeling decoder). That's a v3 differentiator, not a v1 dependency.

---

## 2. Design principles (inherit the moat)

1. **Leak-safety is the contract.** `TreeAttributor` subclasses [`PanelTransformer`](../../panelary/core/protocol.py). `_fit(train)` binds the explainer to **this fold's model + this fold's background**; `_transform(X)` emits attributions with no re-fitting and no future/other-fold information. Declare `panel_safe` / `leakage_safe`.
2. **Background set is a first-class, time-aware object** (see §4). This is the single most important design element.
3. **Reuse the models already wrapped.** `models.py` already adapts XGBoost/LightGBM/CatBoost/sklearn. The attributor consumes a *fitted* Panelary model and calls its native SHAP — it does not train anything.
4. **Polars-native output.** Attributions come back as columns (`shap_<feature>`), keyed by `(entity, time)`, ready for `group_by`. Exploit SHAP additivity so group/window SHAP is a cheap post-aggregation, not a recompute.
5. **No new hard deps for v1.** Native SHAP lives in the model libraries. `shapiq` is an *optional* extra for v3 interactions. Rust/`pyo3-polars` is v3 only.
6. **Interop over reimplementation.** Bridge to `shapiq` for interactions; port C++ reference algorithms only in the v3 Rust effort.

---

## 3. Module layout

New subpackage `panelary/explain/`, mirroring `select/` / the planned `reduce/`:

```
panelary/explain/
    __init__.py          # public API + docstring
    _background.py        # TimeAwareBackground: fold-bound, past-only reference sets
    _tree.py              # tree_attributions() core: dispatch to native TreeSHAP per model type
    _estimators.py        # TreeAttributor(PanelTransformer) + shared base
    _aggregate.py         # group SHAP / window SHAP over (entity, time) via group_by (v2)
    _stability.py         # attribution-stability / drift reporting (v2)
    _interactions.py      # shapiq / TreeSHAP-IQ interop (v3, optional import)
    _common.py            # shared helpers (reuse feature_matrix etc. from reduce/_common)
```

Register in `panelary/__init__.py`; add docs nav (see §9). Import boundary: `explain/` is **Tier-2 (estimator layer)** — may import `core/`, model libs, NumPy; must not be imported by Tier-1 `namespaces/`.

---

## 4. The background set — the leak-safe core (`_background.py`)

`TimeAwareBackground` is the object that makes this library different:

- **Fold-bound:** constructed from the training panel only; carries no rows from val/test/other folds.
- **Past-only (per entity):** for an observation at `(e, t)`, admissible background rows are `{(e', t') : t' < t}` (optionally `t' ≤ t − embargo` to honor the same purge/embargo used elsewhere in Panelary's CV). This forbids future distribution leaking into the explanation.
- **Sampling:** capped sample (default 100–1000) drawn from the admissible set; deterministic with a seed.
- **Modes:**
  - `interventional` (marginal) → requires a background; "true to the model," breaks correlations. **Default** for actionability/sparse models. Guard against off-manifold background rows.
  - `conditional`/`observational` → "true to the data"; spreads credit across correlated features. Offer, don't default.
  - `path_dependent` (tree-native, no external background) → **flag the hidden leak**: its implicit background is the training distribution baked into leaf counts, which leaks if the model was trained on future rows. Allow only with an explicit `i_accept_path_dependent_background=True` acknowledgement or a warning.
- **Cross-sectional variant:** for pure cross-sectional (per-date) attribution, background = the *same date's* cross-section, computed within each date before aggregating (never pool future cross-sections into a past date's background).

> **Test this hard.** The headline leak-safety test: an attribution at `(e, t)` must be identical whether or not future rows / other entities are present in the transform set, and must never change when future training data is added. See §8.

---

## 5. v1 — `TreeAttributor` (native exact TreeSHAP)

`tree_attributions(model, X, *, background, mode, interaction=False) -> pl.DataFrame`

Dispatch by wrapped-model type to the **native, exact** implementation (no reimplementation):
- **XGBoost:** `Booster.predict(..., pred_contribs=True)`; pairwise interactions via `pred_interactions=True`.
- **LightGBM:** `predict(..., pred_contrib=True)`.
- **CatBoost:** `get_feature_importance(type='ShapValues')`; pairwise via `'ShapInteractionValues'`; interventional when `reference_data` given.
- **sklearn / other:** fall back to `shap.TreeExplainer` if available, else a model-agnostic sampler (defer; see §7).

`TreeAttributor(PanelTransformer)`:
- ctor: `model` (a fitted Panelary estimator), `features`, `mode="interventional"`, `background="past"` (policy) / a `TimeAwareBackground`, `keep="all"`, `output="long"|"wide"`, `entity`, `time`.
- `_fit(panel)`: resolve features; build `TimeAwareBackground` from the training panel per the policy; bind to `model`. Store nothing that depends on transform-time data.
- `_transform(panel)`: compute native SHAP for each row against the frozen background; attach `shap_<feature>` columns preserving `(entity, time)`; optionally return a tidy long frame `(entity, time, feature, shap_value)`.
- `panel_safe = True`, `leakage_safe = True`.
- Sanity check available: SHAP row-sum + expected value ≈ model output (efficiency), surfaced as a `.check_efficiency()` helper.

**Deliverable value:** exact SHAP for the tree models users already train in Panelary, leak-safe inside the existing purged-CV, output ready for Polars aggregation — immediately useful, zero new math.

---

## 6. v2 — panel-native aggregation & stability

- **Group SHAP (`_aggregate.py`):** attributions summed over a user-defined feature partition (factor buckets: momentum/value/size…) via `group_by`. Exact for marginal values by additivity. **Caveat:** jointly-held-out *interventional group* values require treating the group as one coalition player (not a sum) — expose a separate `group_mode="additive"|"joint"`.
- **Window SHAP:** aggregate attributions over per-entity **rolling time windows** (respecting each entity's own calendar — never a global grid). Windowing also reduces dependence between attributed units → more stable than per-cell SHAP. Offer stationary/sliding window policies.
- **Attribution stability (`_stability.py`):** report, per feature/group, the variance of attribution across a **rolling background** vs across time, so users can distinguish reference-induced oscillation from genuine regime drift. Optionally out-of-sample significance testing (Group-Shapley-significance style) so factor-attribution stories aren't overstated.

This is the layer that makes it *panel* attribution rather than "SHAP with a `.over()`."

---

## 7. v3 — the frontier (R&D, sequenced after v1/v2 prove out)

1. **Any-order interactions via `shapiq` interop (`_interactions.py`):** optional dependency; wrap `shapiq.TreeExplainer` (TreeSHAP-IQ) to emit k-SII / Faith-Shap interaction values for the wrapped tree model, returned in the same leak-safe, panel-keyed, Polars-native shape. **This is the bridge to the factor/HFA "interactions" theme** — the library gains a coherent story: latent factor *interactions* (HFA) + feature *interactions* (Shapley). Mirror shapiq's imputer knob (Marginal/Conditional/Gaussian) onto our `mode`.
2. **Model-agnostic estimator** for non-tree models: prefer interop with `shapiq` (SVARM-IQ / KernelSHAP-IQ — current SOTA) over rolling our own; only consider Leverage SHAP if a native path is needed.
3. **Native Arrow/Polars TreeSHAP kernel (the novel long game):** a `pyo3-polars` `#[polars_expr]` (with `is_elementwise=False`, whole feature matrix as multiple `&[Series]`, model as serialized-bytes/path kwarg) that ports FastTreeSHAP v2 (offer the v1 memory-lean variant) over a **Treelite or ONNX TreeEnsemble** parse (one schema; honor explicit NaN branches; CatBoost oblivious trees need bespoke parsing). Reuse `gbdt-rs`/`perpetual`'s Arrow/Polars zero-copy loaders. Free WASM target falls out of pure Rust. **This is the thing that would make Panelary the only Arrow-native SHAP in existence** — but it's a large effort; gate it behind v1/v2 adoption.
4. **Sparse Möbius/Fourier (SPEX) engine, native:** the ambitious ceiling — Rust `fwht`/`rustfft` + a SPRIGHT-style sparse peeling decoder to recover interactions at scale. Track SPEX/ProxySPEX; do not commit until v3.3 lands.

---

## 8. Leak-safety & correctness tests (`tests/`)

- `tests/test_explain_leakage.py` — **the headline contract:**
  - Attribution at `(e, t)` is invariant to the presence/absence of future rows and other entities in the transform set.
  - Adding future rows to the *training* panel does not change a fitted explainer's attribution for a past `(e, t)` (background is past-only).
  - Path-dependent mode raises/warns without the explicit acknowledgement.
  - Drive it through the existing purged-CV splitter (see `tests/test_leakage_verifier.py`, `test_cv_runner.py`) to prove composability in a `Pipeline`.
- `tests/test_explain_tree.py` — efficiency (SHAP row-sum + E[f] ≈ prediction) for XGBoost/LightGBM/CatBoost; parity with the libraries' own `shap` output on a fixed model.
- `tests/test_explain_aggregate.py` — group SHAP additivity; window aggregation respects per-entity calendars.
- `tests/test_explain_interactions.py` (v3) — shapiq interop returns correct-shape k-SII; pairwise interaction row-sums reconcile with values.

---

## 9. Docs

- `docs/api-reference/explain.md` — API autodoc (match existing layout; add to `mkdocs.yml`).
- `docs/user-guide/attribution.md` — the leak-safety story (background-set pitfalls), interventional-vs-conditional decision table, group/window recipes, a FRED-MD / factor-bucket worked example.
- `docs/concepts/leakage.md` — extend with the **background-set leak** (currently the leakage docs cover feature/label leakage; this is a new, subtle case worth its own section).
- Update `docs/index.md`, `CHANGELOG.md`. `mkdocs build --strict` must stay clean.

---

## 10. Dependencies & licensing
- **v1:** no new hard deps (native SHAP ships in the model libs). `shap` optional for sklearn fallback.
- **v3:** `shapiq` (MIT) as an optional extra for interactions; `pyo3-polars` + maturin for the Rust kernel; Treelite (Apache-2.0) or ONNX for model parsing.
- **License note:** porting FastTreeSHAP (BSD-2) / `shap` `tree_shap.h` (MIT) is permissible with attribution; the algorithm is also published (Linear TreeSHAP, TreeSHAP-IQ) so a clean-room port from the papers is available if preferred.

---

## 11. Milestones & acceptance criteria

| # | Milestone | Deliverable | Done when |
|---|-----------|-------------|-----------|
| M1 | **Background object + `TreeAttributor` (the moat)** | `_background.py`, `_tree.py`, `_estimators.py` for XGBoost/LightGBM/CatBoost | Exact SHAP, leak-safe in purged CV; efficiency check passes; leakage tests green |
| M2 | Panel aggregation | group SHAP + window SHAP (`_aggregate.py`) | Additive group SHAP matches per-column sum; per-entity windows respected |
| M3 | Stability reporting | `_stability.py` | Separates reference-oscillation from regime drift on a simulated regime shift |
| M4 | Docs + polish | explain docs + background-leak concept page | `mkdocs build --strict` clean; decision table published |
| M5 (R&D) | shapiq interaction interop | `_interactions.py` | k-SII returned panel-keyed + leak-safe; ties to HFA interactions story |
| M6 (R&D) | Native Arrow/Polars TreeSHAP kernel | `pyo3-polars` plugin (Treelite/ONNX parse, FastTreeSHAP v2 port) | Parity with native libs; benchmark win; WASM builds |
| M7 (moonshot) | Sparse Möbius/Fourier (SPEX) native | Rust FWHT + peeling decoder | Recovers known sparse interactions; scales past ~20 features |

**M1 is the must-ship moat.** M2–M3 make it panel-native. M5 connects to the factor work. M6–M7 are the differentiating R&D bets, gated on adoption.

---

## 12. Explicitly out of scope
- Reimplementing KernelSHAP / TreeSHAP in Python (native versions exist and are exact).
- Causal / Asymmetric SHAP (needs a causal graph users won't have) — revisit only on demand.
- Glassbox models (EBM/GA2M, NAM) — a "compete, don't explain" paradigm; different product.
- Deep-net gradient attribution (Integrated Gradients/Hessians, Captum) — not Panelary's model class.
- Data-valuation Shapley (pyDVL/OpenDataVal) — different use case.

---

## 13. Reference map (consult, don't blindly copy)
- **Native exact SHAP:** XGBoost `pred_contribs`/`pred_interactions`, LightGBM `CONTRIB`, CatBoost `ShapValues`/`ShapInteractionValues`.
- **Interaction hub:** `shapiq` (TreeSHAP-IQ, KernelSHAP-IQ, SVARM-IQ, Faith-Shap, n-Shapley) — MIT, arXiv:2410.01649.
- **Semantics:** "True to the Model or True to the Data?" (arXiv:2006.16234); interventional vs conditional.
- **Panel/time-series:** TimeSHAP (arXiv:2012.00073), WindowSHAP (arXiv:2211.06507), Group Shapley w/ significance (arXiv:2501.03041).
- **Leak-safety:** SHAP-in-CV guidance; background/reference-set pitfalls; path-dependent implicit background.
- **Fast cores to port (v3/v6):** FastTreeSHAP v2 (BSD-2), `shap` `tree_shap.h` (MIT), Linear TreeSHAP (arXiv:2209.08192), TreeSHAP-IQ (arXiv:2401.12069).
- **Frontier (v7):** SPEX (arXiv:2502.13870), ProxySPEX (arXiv:2505.17495); Rust `fwht`/`rustfft`.
- **Substrate:** pyo3-polars expr plugins; Treelite; ONNX `ai.onnx.ml.TreeEnsemble`; `perpetual` (Arrow/Polars zero-copy Rust GBM).
- **Caveats to honor:** PNAS impossibility results (arXiv:2212.11870); Shapley Residuals (interaction diagnostic).

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
