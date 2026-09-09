# Installation

PanelKit is published to PyPI as the `polars_features` package (the public rename to
`panelkit` is planned but not yet effective). To install the latest release, run:

```bash
pip install polars_features
```

Prefer [uv](https://docs.astral.sh/uv/)? It installs the same wheel, just much faster:

```bash
uv pip install polars_features       # into an existing environment
uv add polars_features               # into a uv-managed project (pyproject.toml)
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
- **Pure Python** since 0.4.0: the distribution is a single universal `py3-none-any` wheel, so
  there is no compiler, no Rust toolchain and no platform-specific build -- it installs the same
  way on every OS and every supported Python.

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
- `signatures`: **reserved / not yet implemented.** No module imports `iisignature` today;
  the extra name exists only so the `_deps` install hint stays resolvable. Installing it
  currently enables nothing.
- `explain`: feature-attribution fallbacks for `polars_features.explain` --
  [`shap`](https://shap.readthedocs.io/) for non-booster models and
  [`shapiq`](https://shapiq.readthedocs.io/) for any-order Shapley interactions.
  The `TreeAttributor` fast path needs neither: it calls the boosters' own
  native exact TreeSHAP.
- `all`: convenience extra unioning every optional feature set above (except the reserved
  `signatures`).
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
