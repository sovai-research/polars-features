# Installation

PanelKit is published to PyPI as the `polars_features` package (the public rename to
`panelkit` is planned but not yet effective). To install the latest release, run:

```bash
pip install polars_features
```

Then import it as:

```python
import polars_features as pk   # PanelKit
```

## Requirements

- **Python 3.10+**.
- **Polars** is a required dependency and is installed automatically. PanelKit is developed and
  tested against **Polars 1.x** (the `.panel` / `.xs` / `ts` namespaces register on
  `polars.Expr`, `polars.DataFrame`, and `polars.LazyFrame`).
- The Rust-accelerated feature extractors ship as prebuilt wheels for common platforms; no
  Rust toolchain is needed to install a release.

## Extras

`polars_features` ships optional feature sets as extras. For example, to install with the
large-language-model (LLM) analysis and CAFE imputation features:

```bash
pip install "polars_features[llm,cafe]"
```

- `llm`: LLM-powered forecast analyst (OpenAI/Anthropic SDKs, tokenizers, retries).
- `cafe`: CAFE imputation backend for `cafe_impute` / `CafeImputer` (MIT, Sov.ai).
- `gpu`: Polars GPU engine via the `cudf-polars` backend (`.collect(engine="gpu")`).
- `viz`: plotting utilities, powered by [`plotly`](https://plotly.com/python/).
- `signatures`: path-signature transforms via `iisignature`.
- `explain`: feature-attribution fallbacks for `polars_features.explain` --
  [`shap`](https://shap.readthedocs.io/) for non-booster models and
  [`shapiq`](https://shapiq.readthedocs.io/) for any-order Shapley interactions.
  The `TreeAttributor` fast path needs neither: it calls the boosters' own
  native exact TreeSHAP.
- `all`: convenience extra unioning every optional feature set above.
- `dev`: development / CI tooling (ruff, pytest, hypothesis, mypy, pre-commit, maturin).

Install everything at once with:

```bash
pip install "polars_features[all]"
```

!!! note "CAFE imputation"
    The [Quickstart](./quickstart.md) imputation step uses `CafeImputer` / `cafe_impute`,
    which require the `cafe` extra (`pip install "polars_features[cafe]"`). Everything else in
    the quickstart works with the core install.

## Verify your install

```python
import polars as pl
import polars_features as pk

print("PanelKit", pk.__version__)

df = pl.DataFrame(
    {"ticker": ["A", "A", "B", "B"], "day": [0, 1, 0, 1], "close": [1.0, 2.0, 3.0, 4.0]}
)
panel = pk.PanelFrame(df, entity="ticker", time="day")

# A causal within-entity op — passes the leak-safety check.
pk.assert_no_lookahead(
    pl.col("close").panel.zscore(window=2).over("ticker").alias("z"),
    panel,
)
print("OK:", panel.feature_cols)   # OK: ['close']
```

If this prints the version and `OK: ['close']` with no errors, PanelKit is installed
correctly. Continue with the [Quickstart](./quickstart.md).
