# Design Vision — "Interactions" as a first-class, library-wide theme

**Status:** vision / cross-cutting design · **Owner:** TBD
**Author of plan:** research + design session, 2026-09-04
**Unifies:** [factor-extraction-plan.md](factor-extraction-plan.md) (`reduce/` — HFA) · [shap-attribution-plan.md](shap-attribution-plan.md) (`explain/` — Shapley)

---

## 0. The one-sentence thesis

> Ordinary tools stop at **first-order, additive** structure — PCA finds covariance factors, SHAP gives additive attributions. **PanelKit's throughline is higher-order structure:** discovering **interacting latent factors** in the data (HFA) and attributing **interacting features** in the model (Shapley interactions) — the same mathematical idea, applied on the input side and the output side, both leak-safe on panels.

This document is not a third module. It is the **connective tissue** that makes `reduce/` (HFA) and `explain/` (SHAP-IQ) read as two halves of one deliberate idea rather than two unrelated features. It defines the shared vocabulary, the shared math, the workflows that chain them, and the product narrative.

---

## 1. Why they are the same idea

Both modules answer **"where does higher-order structure live, and how much of it is there?"** — one about the *distribution of the inputs*, the other about the *function the model learned*.

| | `reduce/` — HFA | `explain/` — Shapley interactions |
|---|---|---|
| **Object analysed** | the feature *distribution* `X` | the model's *prediction function* `f` |
| **Supervision** | unsupervised (input side) | supervised (output side) |
| **First order (the baseline)** | PCA — covariance / 2nd-order | SHAP values — additive / order-1 |
| **Higher order (the point)** | higher-order **cumulants** → weak/non-Gaussian factors | higher-order **Möbius/interaction indices** → feature synergies |
| **"How many / how much" knob** | number of factors (Bai–Ng / eigenratio) | interaction order `k` (k-SII / Faith-Shap) |
| **Failure it fixes** | PCA misses weak non-Gaussian factors | SHAP folds synergies into main effects |

**The shared mathematical backbone is functional decomposition.** Cumulants (HFA), Möbius/Harsanyi dividends (Shapley interactions), and functional-ANOVA/Sobol terms are all *linearly related bases* for expressing a high-dimensional object as a sum of main effects + interaction terms. HFA reads higher-order structure out of the input distribution's cumulants; Shapley-IQ reads it out of the model function's Möbius transform. Same skeleton, two organs.

This is not a stretch we're imposing — the sparse-Fourier line (SPEX/ProxySPEX) in the attribution research and the higher-order-cumulant line (HFA) in the factor research are *literally the same decomposition mathematics* in different bases. Owen's theorem already ties Shapley values to Sobol/ANOVA; the cumulant↔Möbius↔Fourier equivalences close the loop.

---

## 2. What "make the linkage explicit" concretely means

Five deliverables, each cheap once both modules exist, that turn two modules into one theme.

### 2.1 A shared `order` vocabulary
Adopt one mental model across the library:
- **order 1** = additive baseline (PCA factors · SHAP values)
- **order ≥ 2** = interactions (HFA higher-cumulant factors · k-SII / Faith-Shap)

Surface it as consistent parameter naming: `reduce` uses `order=3|4` for HFA's cumulant order; `explain` uses `max_order=k` for interaction order. Document them side by side so a user who learned one immediately understands the other. **Do not force a single shared parameter** — the semantics differ (cumulant order vs interaction order) — but keep the *word* and the *concept* aligned.

### 2.2 A shared concepts page (`docs/concepts/interactions.md`)
One narrative page — the "why higher-order" story — that both API sections link into. It explains functional decomposition once, then branches: "on the data → HFA (`reduce/`)", "on the model → Shapley interactions (`explain/`)". This page is the single best artifact for making the theme legible to users; it is worth writing even before v3 of either module.

### 2.3 The chaining workflows (the real payoff)
The two modules become **composable in a `Pipeline`**, and the combinations are genuinely novel:

1. **Factors → attribution.** Extract HFA/PCA factors (`reduce`), train a model on them, then attribute predictions to *factors* (`explain`). You get "which latent factor drove this forecast," not just "which raw feature." Leak-safe end to end because both are `PanelTransformer`s bound to the training fold.
2. **Factor interactions.** Run Shapley *interactions* on the factor inputs → "factor 2 × factor 5 synergy drives the equity-premium forecast." This is the headline demo: latent-factor interactions, which neither PCA+SHAP nor any other library produces.
3. **Attribution-informed grouping.** Use Shapley interaction structure to define the feature *groups* for group-SHAP (or to seed HFA's feature pool) — interactions discovered on the model feed back into how the data is decomposed.
4. **Cross-check.** High Shapley-interaction mass among features that also load on the same HFA factor is a consistency signal (the model is using the latent structure the data actually has); divergence is a diagnostic. Ties naturally to the Shapley-Residuals interaction diagnostic already noted in the attribution plan.

### 2.4 A shared decomposition core (optional, only if it stays clean)
If, when both modules exist, there is real duplication in the Möbius/cumulant/basis-conversion helpers, factor them into a small internal `polars_features/_decomposition.py` (or extend `reduce/_common.py`). **Guardrail:** only do this if it removes real duplication — do not build an abstraction speculatively. The theme is a *narrative and API* commitment first, a code-sharing commitment only if earned.

### 2.5 The v3 convergence point
The attribution plan's v3 already routes any-order interactions through `shapiq`/TreeSHAP-IQ, and its moonshot is a native **sparse Möbius/Fourier (SPEX)** engine. The factor plan's HFA is higher-order **cumulant** eigenanalysis. Because Möbius, cumulant, and Fourier bases are interconvertible, a **single native Rust decomposition kernel** could eventually serve both — SPEX-style sparse recovery for model interactions and cumulant construction for factor extraction. This is the long-horizon reason to keep the two designs vocabulary- and basis-aware now, so the eventual kernel isn't two disjoint efforts. Explicitly a moonshot; not a near-term commitment.

---

## 3. The product narrative (positioning)

One line for the README / docs index:

> **PanelKit takes interactions seriously, end to end** — it finds the interacting latent factors hiding in your panel (HFA, where PCA gives up on weak non-Gaussian signal) and attributes your model's predictions to interacting features and factors (Shapley interactions), all leak-safe across entities and time.

Why this is defensible (from the research sweeps): **no library does leak-safe panel attribution**, **no library does higher-order-cumulant factor analysis outside R's `hofa`**, and **nothing connects the two.** The theme is not marketing gloss — it's three independently-verified gaps that happen to share one mathematical spine.

---

## 4. Sequencing (how the theme accretes)

The theme is emergent — you don't build it up front, you *protect the ability to have it* while building the two modules.

| Phase | Action | Depends on |
|---|---|---|
| P0 (now) | Adopt the `order` vocabulary + cross-link the two plans (this doc + §5 edits) | nothing |
| P1 | Ship `reduce/` M1–M2 and `explain/` M1 with aligned naming | the two plans |
| P2 | Write `docs/concepts/interactions.md`; wire the **factors → attribution** pipeline demo | P1 |
| P3 | **Factor-interactions demo** (Shapley interactions on HFA factors) via `explain/` M5 shapiq interop | `explain/` M5 |
| P4 | Evaluate a shared `_decomposition` core *iff* duplication is real | both modules mature |
| P5 (moonshot) | Unified native Rust decomposition kernel (cumulant + sparse Möbius/Fourier) | `explain/` M6–M7 |

**P0–P2 are nearly free and deliver most of the "one coherent library" value.** P3 is the demo that sells the whole thesis. P4–P5 are earned, not assumed.

---

## 5. Cross-references to add to the sibling plans
- `factor-extraction-plan.md` → add a short "Part of the interactions theme" note pointing here, framing HFA as the *order ≥ 2, input-side* half.
- `shap-attribution-plan.md` → add the same note, framing Shapley-IQ as the *order ≥ 2, model-side* half, and calling out the **factor-interactions** workflow (§2.3.2) as the flagship demo of v3/M5.

*(Both edits applied alongside this plan.)*

---

## 6. Open decisions
1. **How prominent in positioning?** Is "interactions, end to end" the library's *primary* tagline, or one theme among several (features / imputation / validation / models)? → Recommend making it a top-two headline, since it's the most differentiated and hardest-to-copy capability.
2. **Shared core now or later?** → Recommend **later** (P4), duplication-driven only.
3. **Does the `order` vocabulary extend to a third area** (e.g. higher-order moments in `feature_extractors`/catch22)? → Worth a scan; if catch22-style higher-moment features exist, fold them into the same narrative for free.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
