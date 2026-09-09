# Custom Forecasters and Transformers

This guide shows how to build new **transformers** and **forecasters** on Panelary's own
primitives. If you would like your implementation to ship inside Panelary, please open a
[draft pull request on GitHub](https://github.com/sovai-research/panelary/pulls) — all
contributions are welcome. See the [Contributing guide](contributing.md) for the development
setup, and [Leakage & Correctness](../leakage.md) for the rules any new component must obey.

Panelary has a [two-tier API](../concepts/two-tier-api.md), and which tier you extend depends
on whether your component **learns anything**:

| Your component | Extend | Lives in |
| --- | --- | --- |
| Stateless reshaping / windowing of a panel | the `@transformer` decorator | `panelary.preprocessing` |
| Anything with parameters learned from data | `PanelTransformer` | the relevant subpackage |
| A model that terminates a pipeline | `PanelEstimator` / `Forecaster` | `panelary.models`, `panelary.forecasting` |

## The one rule

**No lookahead, ever.** Within-entity work goes `.over(entity_col)` in time order;
cross-sectional work goes `.over(time_col)`. Anything with a `fit` is fit **per fold on
training data only** — never globally, and never on the test fold. Every new component is
expected to come with a future-perturbation test built from
[`panelary.testing`](../api-reference/testing.md):

```python
from panelary.testing import assert_no_lookahead

assert_no_lookahead(panel, op=my_transform, entity="entity", time="time")
```

## Build your own stateless `transformer`

`panelary.base.transformer.transformer` is a decorator that turns a **curried** function into a
composable `Transformer`. Specify the parameters in the outer function, and return an inner
`transform(X: pl.LazyFrame) -> dict` that produces `{"X_new": <LazyFrame>}`.

```python
import polars as pl

from panelary.base.transformer import transformer


@transformer
def lag(lags: list[int], is_sorted: bool = False):
    """Apply a lag transformation to a panel LazyFrame.

    Parameters
    ----------
    lags : list of int
        The lag values to apply.
    is_sorted : bool
        Skip the sort when the frame is already ordered by (entity, time).
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col = X.columns[0]
        time_col = X.columns[1]
        max_lag = max(lags)
        lagged_series = (
            (
                pl.all()
                .exclude([entity_col, time_col])
                .shift(lag)
                # `.over(entity_col)` is what keeps one entity's rows out of
                # another's — the leak-safety of the whole transform rests here.
                .over(entity_col)
                .name.suffix(f"__lag_{lag}")
            )
            for lag in lags
        )
        # Pre-sorting improves performance by ~20%.
        X_new = X if is_sorted else X.sort(by=[entity_col, time_col])
        X_new = X_new.select(
            pl.col(entity_col).set_sorted(),
            pl.col(time_col).set_sorted(),
            *lagged_series,
        ).filter(pl.col(time_col).arg_sort().over(entity_col) >= max_lag)

        artifacts = {"X_new": X_new}
        return artifacts

    return transform
```

Key points:

1. Declare every parameter on the **outer** function; it is what the user calls.
2. The curried inner `transform` takes a panel `LazyFrame` — entity column first, time column
   second — and returns a dict containing at least the `X_new` key.
3. Stay lazy. Returning a `LazyFrame` lets the caller fuse your step into one query plan.
4. Put any state the inverse needs into the returned dict, and supply an `invert` function if
   the transform is invertible.

## Build your own fitted `PanelTransformer`

When a component *learns* something — a scaler's moments, a factor loading matrix, a per-entity
coefficient — the stateless decorator is the wrong tool, because there is nowhere to say which
rows the state was learned from. Subclass
`PanelTransformer` (from `panelary.core.protocol`) instead. It requires two hooks and two
**mandatory** class-level declarations; forgetting either declaration is a loud error, by
design.

```python
from panelary.core.protocol import PanelTransformer


class MyScaler(PanelTransformer):
    # Both are required. `panel_safe` = respects entity boundaries.
    # `leakage_safe` = no look-ahead, and fits only on the rows given to `fit`.
    panel_safe = True
    leakage_safe = True

    def _fit(self, panel):
        # Learn from `panel` (the training fold) and ONLY from `panel`.
        # Store learned state on trailing-underscore attributes.
        self.means_ = ...

    def _transform(self, panel):
        # Apply the frozen state. Learn nothing here.
        return panel.with_columns(...)
```

The public `fit` / `transform` / `fit_transform` wrap these hooks with contract checks (input
typing, fitted-state tracking and an optional leakage assertion), so you only write the two
underscore methods. The guarantee this buys: if `leakage_safe is True`, then for any
walk-forward or purged split, `t.fit(train).transform(test)` produces test-fold features that
depend only on parameters learned from `train` and each test row's own past within its entity.

If your transform genuinely cannot be made leak-safe, declare `leakage_safe = False` and say
why in the docstring. That is honest and useful; a false `True` is neither.

## Build your own `forecaster`

Subclass [`Forecaster`](../api-reference/forecasting.md) and implement its `_fit` / `_predict`
hooks. Recursive forecasting must build the features for horizon `h` from observations at or
before the forecast origin, feeding each step its own prediction forward rather than reading
the realised value.

🚧 A full walkthrough is under construction. Until it lands, `panelary/forecasting/` contains
the reference implementations — `naive` for the minimal shape, `lightgbm` for the full
lagged-feature path.
