# Prefix-safe temporal ROCKET

**Stage:** todo — needs a build contract before implementation · **Priority:** 2 (small, fast, uses borrowed-accuracy as its instrument) · **Home:** Panelary

## Pitch

ROCKET's proportion-of-positive-values (PPV) pooling is computed over the
entire series. In a rolling backtest the feature at time t depends on values
after t. No causal variant of MiniRocket exists in any library, and the same
applies to max, min, slope and local volatility summaries.

ROCKET is the best-performing family in both time-series classification and
extrinsic regression — window to scalar, the shape of every return-prediction
problem — and its canonical implementation leaks in exactly the setting
finance uses it. Fixing it produces stage one of later items, and the
measurement alone is publishable.

First experiment: expanding-window PPV against batch PPV on one panel; report
the borrowed-accuracy delta. If it is non-zero, every published ROCKET
backtest in finance is inflated.

## Assessment

**Sharpen the leak claim before building — there are two channels, not one.**

1. *Pooling.* Only leaks if ROCKET runs over the full series and the result is
   used as a feature at earlier `t`. Applied to a trailing window ending at
   `t`, PPV over that window is already causal. So the claim holds for
   *how finance uses it*, not for ROCKET itself — the paper has to show
   people use it the leaky way.
2. *Bias fitting — likely the larger and less-known channel.* MiniRocket's
   `fit` draws each kernel's bias from quantiles of convolution outputs on
   training examples. Fit on the whole sample, and future data sets the
   thresholds that define every feature. That leaks even with trailing
   windows.

Measure both separately; that is a two-component borrowed-accuracy
decomposition and a clean first use of that metric.

**Build is cheap and fits Panelary's invariants exactly.** Expanding PPV is a
cumulative count of positives over `t`, which is prefix-invariant by
construction (AGENTS.md invariant 1). Expanding max/min are native
`cum_max`/`cum_min`. No `rolling_map` needed.

**Honest caveat:** "every published ROCKET backtest is inflated" follows only
if the delta is non-zero *and* the papers used the leaky form. A null result
is still reportable, just as a smaller paper.
