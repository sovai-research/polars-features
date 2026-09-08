# Interactions: higher-order structure, end to end

> **Ordinary tools stop at first-order, additive structure.** PCA finds covariance
> factors; SHAP gives additive attributions. PanelKit's throughline is what lies
> *above* first order: the **interacting latent factors** hiding in your data, and
> the **interacting features** driving your model — the same mathematical idea,
> applied on the input side and the output side, both leak-safe on panels.

This page is the connective tissue between two subpackages that look unrelated
until you see the shared spine:

- [`polars_features.reduce`](../api-reference/reduce.md) — higher-order **cumulant**
  factor analysis (HFA) on the *distribution of your inputs*.
- [`polars_features.explain`](../api-reference/explain.md) — Shapley **interaction**
  indices on the *function your model learned*.

## The `order` vocabulary

One mental model, used consistently across the library:

| | **order 1** — the additive baseline | **order ≥ 2** — the interactions |
|---|---|---|
| **Input side** (`reduce`) | PCA — covariance, 2nd-order moments | HFA — higher-order **cumulants**, weak / non-Gaussian factors |
| **Model side** (`explain`) | SHAP values — additive attributions | k-SII / Faith-Shap — feature **synergies** |
| **The knob** | — | `order=3\|4` (cumulant order) · `max_order=k` (interaction order) |
| **The failure it fixes** | — | PCA misses weak non-Gaussian factors · SHAP folds synergies into main effects |

The two knobs are deliberately *not* the same parameter — a cumulant order and an
interaction order mean different things — but they share the word and the concept,
so learning one teaches you the other.

## Why they are the same idea

Both modules answer one question: **where does higher-order structure live, and how
much of it is there?** One asks it of the *distribution of the inputs*, the other of
the *prediction function*.

The shared backbone is **functional decomposition**. Cumulants (HFA),
Möbius/Harsanyi dividends (Shapley interactions) and functional-ANOVA/Sobol terms
are *linearly related bases* for writing a high-dimensional object as a sum of main
effects plus interaction terms:

```
object  =  Σ main effects  +  Σ pairwise terms  +  Σ triple terms  +  …
             (order 1)          (order 2)           (order 3)
```

HFA reads that expansion out of the input distribution's cumulants. Shapley
interaction indices read it out of the model function's Möbius transform. Owen's
theorem already ties Shapley values to Sobol/ANOVA decompositions; the
cumulant ↔ Möbius ↔ Fourier equivalences close the loop. Same skeleton, two organs.

## On the data: HFA (`reduce`)

PCA is an eigendecomposition of the **covariance** matrix — a second-order object.
It recovers factors whose signal shows up in variance. It gives up on **weak
factors**: ones whose loadings are too diffuse to lift an eigenvalue above the
noise floor.

HFA does eigenanalysis on a higher-order **multi-cumulant** matrix instead. For
centered/standardized `X`:

```
G    = X @ X.T              # Gram
M3M  = X.T @ ((G * G) @ X)  # third-order multi-cumulant matrix
U    = top-r eigenvectors of M3M
F    = X @ U                # factor scores
```

Because it reads third- (or fourth-) order structure, it recovers factors that are
**non-Gaussian but weak** — exactly the ones PCA cannot see. The `order=` kwarg
selects the cumulant order.

```python
from polars_features.reduce import HFAFactors

hfa = HFAFactors(n_factors=3, order=3)
factors_train = hfa.fit_transform(train)   # loadings learned on train only
factors_test = hfa.transform(test)         # frozen loadings applied
```

Loadings and standardization statistics are learned in `fit` and frozen; eigenvector
signs are pinned by a fixed convention so a factor never flips between train and
test. See [Factor extraction](../user-guide/factors.md).

## On the model: Shapley interactions (`explain`)

A SHAP value answers "how much did feature *j* contribute?" — an order-1, additive
answer. When two features only matter *together*, that synergy has nowhere to go:
it gets smeared across both main effects, and the story you read off the plot is
wrong in a specific, confident-sounding way.

Shapley **interaction** indices (k-SII, Faith-Shap) restore the missing terms:
alongside each feature's main effect you get the pairwise (and higher) terms that
belong to *sets* of features. The `max_order=k` kwarg selects how far up the
expansion you go.

```python
from polars_features.explain import TreeAttributor

attr = TreeAttributor(model=fitted_model, max_order=2)   # main effects + pairs
attr.fit(train)
contributions = attr.transform(test)
```

The background/reference set is fold-bound and past-only, so the explanation itself
cannot see the future. See [Attribution](../user-guide/attribution.md) and
[the background-set leak](attribution-leakage.md).

## Chaining them: the workflows that pay off

The two modules are `PanelTransformer`s, so they compose in a `Pipeline` — and the
combinations are where the theme stops being a narrative and starts being a
capability.

**1. Factors → attribution.** Extract factors with `reduce`, train on them, then
attribute predictions to *factors* rather than raw features. You learn "which latent
factor drove this forecast," not "which of my 300 collinear columns nudged it."
Leak-safe end to end, because both stages are bound to the training fold.

**2. Factor interactions — the flagship.** Run Shapley *interactions* on the factor
inputs: *"factor 2 × factor 5 synergy drove the equity-premium forecast."* Latent
factor interactions are something neither PCA + SHAP nor any other library produces.

**3. Attribution-informed grouping.** Use the discovered interaction structure to
define the feature *groups* for group-SHAP, or to seed HFA's feature pool.
Interactions found in the model feed back into how the data gets decomposed.

**4. Cross-check.** High Shapley-interaction mass among features that also load on
the same HFA factor is a *consistency* signal — the model is using latent structure
the data actually has. Divergence is a diagnostic worth chasing.

## Why this is defensible

Three independently verified gaps that happen to share one mathematical spine:

- **No library does leak-safe panel attribution.** Every SHAP implementation treats
  the background set casually, and the background set is the dominant leakage surface.
- **No library does higher-order-cumulant factor analysis** outside R's `hofa`.
- **Nothing connects the two.**

## What is *not* claimed

The shared basis (cumulant ↔ Möbius ↔ Fourier) means a single decomposition kernel
could one day serve both engines. That is a long-horizon possibility, not a shipped
abstraction: the two modules deliberately keep their own math today, and a shared
core will only be factored out if real duplication earns it.

## See also

- [Factor extraction](../user-guide/factors.md) — the input side, in practice.
- [Attribution](../user-guide/attribution.md) — the model side, in practice.
- [Leak-safety](leak-safety.md) — the contract both sides inherit.
