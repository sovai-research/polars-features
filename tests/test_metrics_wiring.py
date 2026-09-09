"""Guard the reachability of `panelary.metrics.multi_objective`.

The module was documented (`docs/api-reference/multi-objective.md`, navigated by
`mkdocs.yml`) long before it was importable as an attribute of `panelary.metrics`
-- defect B2 in `plans/todo/layout-build-contract.md`. These tests fail if the
re-export is ever dropped again, or if the docs page starts promising a name the
package does not export.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import polars as pl
import pytest

import panelary as pn

DOCUMENTED = ("Metrics", "score_forecast", "score_backtest", "summarize_scores")
DOCS_PAGE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "api-reference"
    / "multi-objective.md"
)


def test_submodule_is_reachable_as_an_attribute():
    assert hasattr(pn.metrics, "multi_objective")
    assert pn.metrics.multi_objective.__name__ == "panelary.metrics.multi_objective"
    assert "multi_objective" in pn.metrics.__all__


@pytest.mark.parametrize("name", DOCUMENTED)
def test_documented_names_are_exported(name):
    assert hasattr(pn.metrics.multi_objective, name)
    assert getattr(pn.metrics, name) is getattr(pn.metrics.multi_objective, name)
    assert name in pn.metrics.__all__


def test_all_is_actually_importable():
    """Every name in ``__all__`` must resolve -- catches typos in the list."""
    for name in pn.metrics.__all__:
        assert hasattr(pn.metrics, name), name


def test_docs_directive_target_resolves():
    """`mkdocs build --strict` fails on an unresolvable `:::` target."""
    text = DOCS_PAGE.read_text()
    targets = re.findall(r"^::: (\S+)$", text, flags=re.MULTILINE)
    assert targets, "docs page lost its mkdocstrings directive"
    for target in targets:
        head, *parts = target.split(".")
        assert head == "panelary", target
        obj = pn
        for part in parts:
            assert hasattr(obj, part), f"{target} does not resolve at {part!r}"
            obj = getattr(obj, part)


def test_score_forecast_round_trip():
    """The wired module actually runs -- not just imports."""
    train_times = [date(2020, 1, d) for d in range(1, 7)]
    y_train = pl.DataFrame(
        {
            "entity": ["a"] * 6 + ["b"] * 6,
            "time": train_times * 2,
            "target": [1.0, 2, 3, 4, 5, 6, 2.0, 4, 6, 8, 10, 12],
        }
    )
    test_times = [date(2020, 1, 7), date(2020, 1, 8)]
    y_true = pl.DataFrame(
        {
            "entity": ["a"] * 2 + ["b"] * 2,
            "time": test_times * 2,
            "target": [7.0, 8.0, 14.0, 16.0],
        }
    )
    y_pred = y_true.with_columns(pl.col("target") + 0.5)

    scores = pn.metrics.score_forecast(y_true, y_pred, y_train)
    assert scores.height == 2
    for field in ("mae", "mase", "mse", "rmse", "rmsse", "smape"):
        assert field in scores.columns

    summary = pn.metrics.summarize_scores(scores)
    assert isinstance(summary, pn.metrics.Metrics)
    assert summary.mae == pytest.approx(0.5)

    y_preds = pl.concat(
        [y_pred.with_columns(pl.lit(split).alias("split")) for split in (0, 1)]
    )
    backtest = pn.metrics.score_backtest(y_true, y_preds, agg_method="mean")
    assert backtest.height == 2
    assert "mae" in backtest.columns
