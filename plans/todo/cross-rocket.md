# CrossROCKET: fixed random transforms across the cross-section

**Stage:** todo — needs a build contract before implementation · **Priority:** 3 · **Home:** Panelary

## Pitch

ROCKET applies fixed random operators along time. Nothing equivalent exists
across the cross-section. The natural object is a bank of fixed,
permutation-equivariant random operators over the entity dimension at each t:
indicators on deviation from the cross-sectional median, random peer-basket
deviations, random quantile-rank thresholds, random subset aggregates —
pooled the way MiniRocket pools convolution responses.

ROCKET along time plus ROCKET across entities is a general-purpose panel
representation. It is cheap, fixed, auditable, causal by construction at each
t, and a Panelary primitive that nothing else in the ecosystem has.

First experiment: build the operator bank, apply it to a cross-sectional
return-prediction task with a ridge head, and compare against handcrafted
cross-sectional features. If random operators match handcrafted ones, the same
thing that happened to time-series classification in 2020 happens to panel
feature engineering.

## Assessment

**Check prior art before claiming novelty.** The closest line I know of is
Kelly and Malamud's "virtue of complexity" work: random Fourier features plus
ridge on predictors, including a factor-pricing version over stock
characteristics. It is not the same object — as I understand it, those random
features act on each asset's *own* characteristics, not as operators *across*
entities — but a reviewer will raise it. **Add it as a baseline**: handcrafted
vs random-features-on-characteristics vs CrossROCKET. Multivariate MiniRocket's
random channel combinations are the other near neighbour; channels have fixed
identity, entities are exchangeable, so equivariance is the distinguishing
property to emphasise. (I have not verified these papers' details — read them
first.)

**Panelary already has the primitives.** `.xs.rank`, `.xs.demean`,
`.xs.quantile_bin` and `.xs.winsorize` are the operator vocabulary; the bank is
random compositions of them with fixed seeds. Two constraints from AGENTS.md
apply directly: seeded RNG only (invariant 2), and "causal by construction at
each t" holds only if every operator is purely `.over(time_col)`. Peer baskets
defined from *trailing* correlations need care — build them from data `< t`.

**Pooling is the open design question.** MiniRocket pools along time; across
entities, there is no ordering to pool along, so the pooled statistic must be
permutation-invariant (PPV over entities is — a good default).
