"""Ergonomics tests for the two-tier API.

Tier 1 (Polars-native extension): the ``.panel`` / ``.xs`` namespaces work on
bare ``pl.LazyFrame`` / ``pl.DataFrame`` with no Panelary objects involved.

Tier 2 (estimator layer): transformers and Pipelines accept bare frames with
``entity`` / ``time`` supplied, so ``PanelFrame`` is optional sugar — while the
explicit ``PanelFrame`` path keeps working unchanged.
"""

from __future__ import annotations

import polars as pl
import pytest

import panelary.namespaces  # noqa: F401  (registers the namespaces)
from panelary.core import PanelFrame
from panelary.transform import (
    CrossSectionalRank,
    TimeSeriesScaler,
)


@pytest.fixture
def panel_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "e": ["A", "A", "A", "B", "B", "B"],
            "d": [1, 2, 3, 1, 2, 3],
            "ret": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


# --------------------------------------------------------------------------- #
# Tier 1 — frame-level namespaces on bare Polars frames
# --------------------------------------------------------------------------- #
def test_panel_namespace_on_dataframe(panel_df: pl.DataFrame) -> None:
    out = panel_df.panel.frac_diff("ret", d=0.4, over="e", alias="ret_fd")
    assert "ret_fd" in out.columns
    assert out.height == panel_df.height


def test_panel_namespace_on_lazyframe(panel_df: pl.DataFrame) -> None:
    out = panel_df.lazy().panel.zscore("ret", window=2, over="e")
    assert isinstance(out, pl.LazyFrame)
    assert out.collect().height == panel_df.height


def test_xs_namespace_rank_within_date(panel_df: pl.DataFrame) -> None:
    out = panel_df.xs.rank("ret", over="d")
    assert out.height == panel_df.height


def test_xs_namespace_requires_over(panel_df: pl.DataFrame) -> None:
    # Cross-sectional ops are meaningless without a cross-section key.
    with pytest.raises(ValueError):
        panel_df.xs.demean("ret")


def test_namespace_does_not_mix_entities(panel_df: pl.DataFrame) -> None:
    # A per-entity op grouped by `over="e"` must not bleed across entities:
    # the first observation of each entity has no predecessor -> null.
    out = panel_df.panel.frac_diff("ret", d=1.0, over="e", alias="fd").sort("e", "d")
    fd = out.get_column("fd")
    # one leading null per entity (2 entities)
    assert fd.is_null().sum() >= 2


# --------------------------------------------------------------------------- #
# Tier 2 — estimators accept bare frames; PanelFrame stays optional
# --------------------------------------------------------------------------- #
def test_transformer_on_bare_frame_keys_in_call(panel_df: pl.DataFrame) -> None:
    out = CrossSectionalRank().fit_transform(panel_df, entity="e", time="d")
    assert isinstance(out, PanelFrame)
    assert out.collect().height == panel_df.height


def test_transformer_on_bare_frame_keys_in_constructor(panel_df: pl.DataFrame) -> None:
    out = TimeSeriesScaler(entity="e", time="d").fit_transform(panel_df)
    assert isinstance(out, PanelFrame)
    assert out.collect().height == panel_df.height


def test_transformer_panelframe_path_unchanged(panel_df: pl.DataFrame) -> None:
    panel = PanelFrame(panel_df, entity="e", time="d")
    out = CrossSectionalRank().fit_transform(panel)
    assert isinstance(out, PanelFrame)
    assert out.collect().height == panel_df.height


def test_conflicting_keys_raise(panel_df: pl.DataFrame) -> None:
    panel = PanelFrame(panel_df, entity="e", time="d")
    with pytest.raises(ValueError):
        # passing a PanelFrame AND conflicting keys is ambiguous
        TimeSeriesScaler().fit_transform(panel, entity="d", time="e")


def test_lazyframe_input_stays_lazy(panel_df: pl.DataFrame) -> None:
    out = TimeSeriesScaler(entity="e", time="d").fit_transform(panel_df.lazy())
    # transform returns a PanelFrame; its native form is recoverable and lazy
    assert isinstance(out.to_native(lazy=True), pl.LazyFrame)
