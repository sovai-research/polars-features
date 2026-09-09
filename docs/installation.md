# Installation

Panelary is published to PyPI as the `panelary` package. To install the latest release, run:

```bash
pip install panelary
```

Prefer [uv](https://docs.astral.sh/uv/)? It installs the same wheel, just much faster:

```bash
uv pip install panelary       # into an existing environment
uv add panelary               # into a uv-managed project (pyproject.toml)
```

Then import it as:

```python
import panelary as pn
```

## Requirements

- **Python 3.10+**.
- **Polars** is a required dependency and is installed automatically. Panelary is developed and
  tested against **Polars 1.x** (the `.panel` / `.xs` / `ts` namespaces register on
  `polars.Expr`, `polars.DataFrame`, and `polars.LazyFrame`).
- **Pure Python** since 0.4.0: the distribution is a single universal `py3-none-any` wheel, so
  there is no compiler, no Rust toolchain and no platform-specific build — it installs the same
  way on every OS and every supported Python.

## Extras

The core install is deliberately light: **numpy and Polars only**. Everything heavier is an
optional extra, imported lazily at first use — if a method needs something you have not
installed, it raises with the exact `pip install 'panelary[<extra>]'` command to fix it.

Install one or several at a time:

```bash
pip install 'panelary[ml]'
pip install 'panelary[ml,scipy,cafe]'
```

### Bundles

Most users want one of these two rather than a hand-picked list.

- `recommended` — the batteries-included bundle (`ml`, `scipy`, `seasonality`, `cafe`). This
  is the everyday install: it covers the estimators, selection, reduction, clustering and
  scientific kernels that most of the documentation assumes.
- `all` — every real optional feature set below, including the boosters, LLM and GPU paths.

```bash
pip install 'panelary[recommended]'   # the usual choice
pip install 'panelary[all]'           # the kitchen sink
```

### Feature sets

| Extra | Pulls in | Unlocks |
| --- | --- | --- |
| `ml` | scikit-learn | Panel estimators, feature selection, reduction, clustering |
| `scipy` | scipy | Scientific kernels in feature extractors, catch22, evaluation, metrics |
| `cafe` | cafe-impute (MIT, Sov.ai) | CAFE imputation: `cafe_impute` / `CafeImputer` |
| `seasonality` | holidays | Calendar/holiday seasonality effects (Fourier terms need nothing) |
| `explain` | shap, shapiq | Attribution fallbacks in `panelary.explain` for non-booster models, plus any-order Shapley interactions |
| `dimreduce` | umap-learn | Non-linear reduction (`reduce.PanelUMAP`); PCA/SVD/FA/NMF only need `ml` |
| `viz` | plotly | Plotting utilities |
| `llm` | openai, anthropic, tiktoken, tenacity | LLM-powered forecast analyst |
| `gpu` | cudf-polars-cu12 | Polars GPU engine (`.collect(engine="gpu")`) |
| `fast` | numba | Optional JIT acceleration for the CUSUM change-point kernel |
| `progress` | tqdm | Progress bars in forecasting loops (a no-op without it) |

### Modelling back-ends

| Extra | Pulls in | Unlocks |
| --- | --- | --- |
| `lightgbm` / `catboost` / `xgboost` | the matching booster | Gradient-boosting back-ends for forecasting and panel models |
| `forecasting` | flaml, tqdm | Legacy autoregressive forecasting (functime-derived) with hyper-parameter tuning |
| `automl` | flaml[automl], lightgbm | The FLAML AutoML forecasting path |
| `ann` | pylance | Approximate-nearest-neighbour reduction in `forecasting.lance` |

### Toolchain

| Extra | Unlocks |
| --- | --- |
| `dev` | Development / CI tooling: ruff, pytest, hypothesis, mypy, pre-commit, pytest-xdist, pytest-benchmark |
| `docs` | The MkDocs Material toolchain that builds this site |

!!! note "CAFE imputation"
    The [Quickstart](./quickstart.md) imputation step uses `CafeImputer` / `cafe_impute`,
    which require the `cafe` extra (`pip install 'panelary[cafe]'`). Everything else in the
    quickstart works with the core install.

## Verify your install

```python
import polars as pl
import panelary as pn

print("Panelary", pn.__version__)

df = pl.DataFrame(
    {"ticker": ["A", "A", "B", "B"], "day": [0, 1, 0, 1], "close": [1.0, 2.0, 3.0, 4.0]}
)
panel = pn.PanelFrame(df, entity="ticker", time="day")

# A causal within-entity op — passes the leak-safety check.
pn.assert_no_lookahead(
    pl.col("close").panel.zscore(window=2).over("ticker").alias("z"),
    panel,
)
print("OK:", panel.feature_cols)   # OK: ['close']
```

If this prints the version and `OK: ['close']` with no errors, Panelary is installed
correctly. Continue with the [Quickstart](./quickstart.md).
