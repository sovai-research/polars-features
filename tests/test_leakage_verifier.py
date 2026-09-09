"""Tests for the future-perturbation leak verifier (``panelary.testing``).

The verifier is the keystone that makes Panelary's leak-safety claim checkable:
perturb the future, re-run the op, and assert the past is bit-identical. These
tests (property-based where useful) confirm it *catches* an injected look-ahead
(``shift(-1)``) and *passes* a genuinely walk-forward op (``shift(1).over``).
"""

from __future__ import annotations

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from panelary.core.panel_frame import PanelFrame
from panelary.core.pipeline import Pipeline
from panelary.core.protocol import PanelTransformer
from panelary.testing import assert_no_lookahead, assert_no_train_test_leak

ENTITIES = ["A", "B", "C"]


def make_panel(n_times: int, entities=ENTITIES, seed: int = 0) -> PanelFrame:
    """A dense panel with a distinct per-(entity,time) value, integer time axis."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows_e, rows_t, rows_v = [], [], []
    for e in entities:
        for t in range(n_times):
            rows_e.append(e)
            rows_t.append(t)
            rows_v.append(float(rng.standard_normal()))
    df = pl.DataFrame({"entity": rows_e, "time": rows_t, "value": rows_v})
    return PanelFrame(df, entity="entity", time="time")


# --------------------------------------------------------------------------- #
# Expression ops
# --------------------------------------------------------------------------- #
def _safe_expr() -> pl.Expr:
    # Backward-looking lag within each entity -> no look-ahead.
    return pl.col("value").shift(1).over("entity").alias("lag")


def _leaky_expr() -> pl.Expr:
    # Forward-looking lead within each entity -> pure look-ahead.
    return pl.col("value").shift(-1).over("entity").alias("lead")


@settings(max_examples=30, deadline=None)
@given(n_times=st.integers(min_value=4, max_value=25))
def test_safe_expr_passes(n_times):
    panel = make_panel(n_times)
    # Must not raise: shift(1).over(entity) is walk-forward.
    assert_no_lookahead(_safe_expr(), panel)


@settings(max_examples=30, deadline=None)
@given(n_times=st.integers(min_value=4, max_value=25))
def test_leaky_expr_is_detected(n_times):
    panel = make_panel(n_times)
    with pytest.raises(AssertionError, match="LOOK-AHEAD LEAK DETECTED"):
        assert_no_lookahead(_leaky_expr(), panel)


def test_trailing_rolling_mean_is_safe():
    panel = make_panel(15)
    op = pl.col("value").rolling_mean(window_size=3).over("entity").alias("roll")
    assert_no_lookahead(op, panel)


def test_centered_rolling_mean_leaks():
    panel = make_panel(15)
    # A centered window peeks forward -> leak.
    op = (
        pl.col("value")
        .rolling_mean(window_size=3, center=True)
        .over("entity")
        .alias("roll_c")
    )
    with pytest.raises(AssertionError):
        assert_no_lookahead(op, panel)


def test_failure_message_names_column_and_keys():
    panel = make_panel(8)
    with pytest.raises(AssertionError) as exc:
        assert_no_lookahead(_leaky_expr(), panel)
    msg = str(exc.value)
    assert "lead" in msg  # the offending column
    assert "entity=" in msg and "time=" in msg  # the offending row keys


# --------------------------------------------------------------------------- #
# Callable ops (frame -> frame)
# --------------------------------------------------------------------------- #
def test_callable_dataframe_op_safe():
    panel = make_panel(12)

    def op(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("value").shift(2).over("entity").alias("lag2"))

    assert_no_lookahead(op, panel)


def test_callable_dataframe_op_leaky():
    panel = make_panel(12)

    def op(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("value").shift(-2).over("entity").alias("lead2"))

    with pytest.raises(AssertionError):
        assert_no_lookahead(op, panel)


def test_callable_panelframe_op_safe():
    panel = make_panel(12)

    def op(pf: PanelFrame) -> PanelFrame:
        return pf.with_columns(pl.col("value").cum_sum().over("entity").alias("cum"))

    # cumulative sum is a trailing aggregation -> no look-ahead.
    assert_no_lookahead(op, panel)


# --------------------------------------------------------------------------- #
# Bare frames + explicit keys
# --------------------------------------------------------------------------- #
def test_accepts_bare_dataframe_with_keys():
    panel = make_panel(10)
    df = panel.collect()
    assert_no_lookahead(_safe_expr(), df, entity="entity", time="time")
    with pytest.raises(AssertionError):
        assert_no_lookahead(_leaky_expr(), df, entity="entity", time="time")


def test_explicit_cut_controls_split():
    panel = make_panel(10)
    # A cut well inside the axis still detects the lead leak at the boundary.
    with pytest.raises(AssertionError):
        assert_no_lookahead(_leaky_expr(), panel, cut=3)
    assert_no_lookahead(_safe_expr(), panel, cut=3)


def test_requires_two_times():
    panel = make_panel(1)
    with pytest.raises(ValueError, match="two distinct"):
        assert_no_lookahead(_safe_expr(), panel)


# --------------------------------------------------------------------------- #
# Train/test-boundary variant
# --------------------------------------------------------------------------- #
def _walk_forward_split(panel: PanelFrame, cut: int, test_len: int):
    train = panel.filter(pl.col("time") < cut)
    test = panel.filter((pl.col("time") >= cut) & (pl.col("time") < cut + test_len))
    return train, test


def test_train_test_leak_safe_op_passes():
    panel = make_panel(20)
    split = _walk_forward_split(panel, cut=10, test_len=4)
    # Backward lag on a train-before-test split cannot see the perturbed test.
    assert_no_train_test_leak(_safe_expr(), panel, split)


def test_train_test_leak_leaky_op_detected():
    panel = make_panel(20)
    split = _walk_forward_split(panel, cut=10, test_len=4)
    with pytest.raises(AssertionError, match="LOOK-AHEAD LEAK DETECTED"):
        assert_no_train_test_leak(_leaky_expr(), panel, split)


def test_train_test_leak_rejects_bad_split():
    panel = make_panel(10)
    with pytest.raises(ValueError, match="train, test"):
        assert_no_train_test_leak(_safe_expr(), panel, (panel,))


# --------------------------------------------------------------------------- #
# Pipeline enforcement (_check_leakage wired into the flow)
# --------------------------------------------------------------------------- #
class _Safe(PanelTransformer):
    panel_safe = True
    leakage_safe = True

    def _fit(self, panel):
        pass

    def _transform(self, panel):
        return panel


class _Unsafe(PanelTransformer):
    panel_safe = True
    leakage_safe = False

    def _fit(self, panel):
        pass

    def _transform(self, panel):
        return panel


def test_pipeline_refuses_unsafe_step_across_boundary():
    panel = make_panel(12)
    train = panel.filter(pl.col("time") < 8)
    test = panel.filter(pl.col("time") >= 8)
    pipe = Pipeline([("u", _Unsafe())])
    pipe.fit(train)
    # Transforming the train fold (no boundary) is fine.
    pipe.transform(train)
    # Transforming a test fold crosses the boundary -> refused.
    with pytest.raises(RuntimeError, match="leakage_safe = False"):
        pipe.transform(test)


def test_pipeline_allows_safe_step_across_boundary():
    panel = make_panel(12)
    train = panel.filter(pl.col("time") < 8)
    test = panel.filter(pl.col("time") >= 8)
    pipe = Pipeline([("s", _Safe())])
    pipe.fit(train)
    # A leakage-safe pipeline transforms the test fold without complaint.
    out = pipe.transform(test)
    assert isinstance(out, PanelFrame)
