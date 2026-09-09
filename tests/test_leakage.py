"""Leak-safety tests: the brand-defining guarantees of the Panelary core.

Covers purge+embargo (PurgedKFold), CPCV split counts / paths, walk-forward
no-lookahead, and the end-to-end Pipeline + CV leakage guarantee. Pure-Python
core only; no compiled Rust extension required.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from panelary.core.model_selection import (
    CombinatorialPurgedCV,
    PurgedKFold,
    expanding_window_split,
    sliding_window_split,
)
from panelary.core.panel_frame import PanelFrame
from panelary.core.pipeline import Pipeline
from panelary.core.protocol import PanelTransformer

ENTITIES = ["A", "B", "C"]


def make_panel(n_times: int, entities=ENTITIES) -> PanelFrame:
    """Build a dense panel with a shared integer time axis 0..n_times-1."""
    rows_entity = []
    rows_time = []
    rows_val = []
    for e in entities:
        for t in range(n_times):
            rows_entity.append(e)
            rows_time.append(t)
            rows_val.append(float(t))
    df = pl.DataFrame({"entity": rows_entity, "time": rows_time, "value": rows_val})
    return PanelFrame(df, entity="entity", time="time")


# --------------------------------------------------------------------------- #
# PurgedKFold: purge + embargo guarantee
# --------------------------------------------------------------------------- #
@settings(max_examples=50, deadline=None)
@given(
    n_times=st.integers(min_value=6, max_value=40),
    n_splits=st.integers(min_value=2, max_value=5),
    horizon=st.integers(min_value=0, max_value=4),
    embargo=st.integers(min_value=0, max_value=4),
)
def test_purgedkfold_purge_embargo_guarantee(n_times, n_splits, horizon, embargo):
    # Need at least n_splits unique time steps.
    if n_times < n_splits:
        return
    panel = make_panel(n_times)
    cv = PurgedKFold(
        n_splits=n_splits, horizon=horizon, embargo=embargo, return_indices=True
    )
    for train_pos, test_pos in cv.split(panel):
        train = {int(p) for p in train_pos}
        test = {int(p) for p in test_pos}

        # No train index within `horizon` of any test index (purge band).
        for i in test:
            for j in range(i - horizon, i + horizon + 1):
                assert j not in train, (
                    f"train pos {j} lies in purge band of test pos {i} "
                    f"(horizon={horizon})"
                )

        # No train index in the embargo band after a contiguous test block.
        test_sorted = sorted(test)
        # Identify contiguous block ends.
        block_ends = [
            test_sorted[k]
            for k in range(len(test_sorted))
            if k == len(test_sorted) - 1 or test_sorted[k + 1] != test_sorted[k] + 1
        ]
        for end in block_ends:
            for j in range(end + 1, min(n_times, end + embargo + 1)):
                assert j not in train, (
                    f"train pos {j} lies in embargo band after test block "
                    f"ending at {end} (embargo={embargo})"
                )


@settings(max_examples=50, deadline=None)
@given(
    n_times=st.integers(min_value=6, max_value=40),
    n_splits=st.integers(min_value=2, max_value=5),
    horizon=st.integers(min_value=0, max_value=4),
    embargo=st.integers(min_value=0, max_value=4),
)
def test_purgedkfold_train_test_disjoint(n_times, n_splits, horizon, embargo):
    if n_times < n_splits:
        return
    panel = make_panel(n_times)
    cv = PurgedKFold(
        n_splits=n_splits, horizon=horizon, embargo=embargo, return_indices=True
    )
    for train_pos, test_pos in cv.split(panel):
        assert {int(p) for p in train_pos}.isdisjoint({int(p) for p in test_pos})


def test_purgedkfold_panel_output_disjoint_times():
    """When returning PanelFrames, train/test must share no (entity,time)
    times, and every entity must appear on each side's selected times."""
    panel = make_panel(20)
    cv = PurgedKFold(n_splits=4, horizon=1, embargo=1)
    for train, test in cv.split(panel):
        tr_times = set(train.collect()["time"].to_list())
        te_times = set(test.collect()["time"].to_list())
        assert tr_times.isdisjoint(te_times)
        # Panel grouping respected: all entities present on the test slice.
        te_entities = set(test.collect()["entity"].to_list())
        if te_times:
            assert te_entities == set(ENTITIES)


# --------------------------------------------------------------------------- #
# CombinatorialPurgedCV: split counts, paths, leakage
# --------------------------------------------------------------------------- #
@settings(max_examples=50, deadline=None)
@given(
    n_groups=st.integers(min_value=2, max_value=8),
    k=st.integers(min_value=1, max_value=7),
)
def test_cpcv_n_splits_and_paths_formula(n_groups, k):
    if not (1 <= k < n_groups):
        return
    cv = CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=k)
    assert cv.n_splits == math.comb(n_groups, k)
    assert cv.n_paths == math.comb(n_groups, k) * k // n_groups


@settings(max_examples=40, deadline=None)
@given(
    n_groups=st.integers(min_value=3, max_value=6),
    k=st.integers(min_value=1, max_value=3),
    horizon=st.integers(min_value=0, max_value=3),
    embargo=st.integers(min_value=0, max_value=3),
)
def test_cpcv_split_count_matches_combinations(n_groups, k, horizon, embargo):
    if not (1 <= k < n_groups):
        return
    n_times = n_groups * 4  # plenty of times per group
    panel = make_panel(n_times)
    cv = CombinatorialPurgedCV(
        n_groups=n_groups,
        n_test_groups=k,
        horizon=horizon,
        embargo=embargo,
        return_indices=True,
    )
    folds = list(cv.split(panel))
    assert len(folds) == math.comb(n_groups, k)
    for train_pos, test_pos in folds:
        train = {int(p) for p in train_pos}
        test = {int(p) for p in test_pos}
        assert train.isdisjoint(test)
        # Purge guarantee holds for CPCV too.
        for i in test:
            for j in range(i - horizon, i + horizon + 1):
                assert j not in train


def test_cpcv_backtest_paths_cover_all_groups():
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    paths = cv.backtest_paths()
    assert len(paths) == cv.n_paths
    for path in paths:
        # Each path assigns exactly one (split, group) per group 0..N-1.
        groups_covered = sorted(g for _, g in path)
        assert groups_covered == list(range(cv.n_groups))
        # Each (split, group) reference must actually test that group.
        combos = list(__import__("itertools").combinations(range(6), 2))
        for split_idx, g in path:
            assert g in combos[split_idx]


# --------------------------------------------------------------------------- #
# Walk-forward: no look-ahead
# --------------------------------------------------------------------------- #
@settings(max_examples=40, deadline=None)
@given(
    n_times=st.integers(min_value=8, max_value=30),
    test_size=st.integers(min_value=1, max_value=3),
    n_splits=st.integers(min_value=2, max_value=4),
)
def test_expanding_window_no_lookahead(n_times, test_size, n_splits):
    # Ensure enough history for the deepest split.
    needed = test_size + (n_splits - 1) + test_size
    if n_times < needed + 1:
        return
    panel = make_panel(n_times)
    try:
        folds = expanding_window_split(
            test_size=test_size, n_splits=n_splits, step_size=1
        )(panel)
    except ValueError:
        return  # not enough history for this combination
    for train, test in folds:
        tr_times = train.collect()["time"].to_list()
        te_times = test.collect()["time"].to_list()
        if tr_times and te_times:
            assert max(tr_times) < min(te_times), (
                "expanding window train block must be entirely before test"
            )


@settings(max_examples=40, deadline=None)
@given(
    n_times=st.integers(min_value=10, max_value=30),
    test_size=st.integers(min_value=1, max_value=3),
    n_splits=st.integers(min_value=2, max_value=4),
    window_size=st.integers(min_value=2, max_value=6),
)
def test_sliding_window_no_lookahead_and_bounded(
    n_times, test_size, n_splits, window_size
):
    panel = make_panel(n_times)
    try:
        folds = sliding_window_split(
            test_size=test_size,
            n_splits=n_splits,
            step_size=1,
            window_size=window_size,
        )(panel)
    except ValueError:
        return
    for train, test in folds:
        tr_times = train.collect()["time"].to_list()
        te_times = test.collect()["time"].to_list()
        if tr_times and te_times:
            assert max(tr_times) < min(te_times)
            # Sliding window training span is bounded by window_size time steps.
            assert (max(tr_times) - min(tr_times) + 1) <= window_size


# --------------------------------------------------------------------------- #
# Pipeline + CV: end-to-end leakage guarantee
# --------------------------------------------------------------------------- #
class RecordingTransformer(PanelTransformer):
    """A leakage-safe transformer that records the time set seen during _fit."""

    panel_safe = True
    leakage_safe = True

    def __init__(self):
        super().__init__()
        self.fit_times_ = None

    def _fit(self, panel: PanelFrame) -> None:
        self.fit_times_ = set(panel.collect()[panel.time_col].to_list())

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        return panel.with_columns((pl.col("value") + 1.0).alias("value_plus"))


class NotLeakageSafe(PanelTransformer):
    """A transformer that explicitly declares it is not leakage-safe."""

    panel_safe = True
    leakage_safe = False

    def _fit(self, panel: PanelFrame) -> None:
        pass

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        return panel


def test_pipeline_fit_only_sees_train_fold_rows():
    panel = make_panel(20)
    cv = PurgedKFold(n_splits=4, horizon=1, embargo=1)
    for train, test in cv.split(panel):
        rec = RecordingTransformer()
        pipe = Pipeline([("rec", rec)])
        pipe.fit(train)

        train_times = set(train.collect()["time"].to_list())
        test_times = set(test.collect()["time"].to_list())

        # fit only ever saw train-fold times, never any test time.
        assert rec.fit_times_ == train_times
        assert rec.fit_times_.isdisjoint(test_times)

        # transform on the test fold does not re-fit (recorded times unchanged).
        before = set(rec.fit_times_)
        out = pipe.transform(test)
        assert rec.fit_times_ == before
        assert isinstance(out, PanelFrame)


def test_pipeline_safety_flags_aggregate_from_steps():
    safe = Pipeline([("a", RecordingTransformer()), ("b", RecordingTransformer())])
    assert safe.leakage_safe is True
    assert safe.panel_safe is True

    unsafe = Pipeline([("a", RecordingTransformer()), ("b", NotLeakageSafe())])
    assert unsafe.leakage_safe is False


def test_leakage_safe_false_is_rejected_across_boundary():
    """The protocol's _check_leakage hook must reject a not-leakage-safe
    transform applied across a train/test boundary."""
    panel = make_panel(10)
    t = NotLeakageSafe()
    t.fit(panel)
    # Applied within the same data (no boundary) -> allowed.
    t._check_leakage(panel, test=None)
    # Applied across a train/test boundary -> must raise.
    with pytest.raises(RuntimeError) as exc:
        t._check_leakage(panel, test=panel)
    assert "leakage_safe = False" in str(exc.value)

    # A leakage-safe transform must NOT raise across the boundary.
    safe = RecordingTransformer()
    safe.fit(panel)
    safe._check_leakage(panel, test=panel)  # no exception


def test_protocol_enforces_declaration_of_flags():
    """__init_subclass__ must reject a concrete transformer that forgets to
    declare panel_safe / leakage_safe."""
    with pytest.raises(TypeError) as exc:

        class Forgetful(PanelTransformer):
            # neither panel_safe nor leakage_safe declared
            def _fit(self, panel):
                pass

            def _transform(self, panel):
                return panel

    msg = str(exc.value)
    assert "panel_safe" in msg or "leakage_safe" in msg
