from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import polars as pl
import pytest

from panelary.cross_validation import (
    expanding_window_split,
    sliding_window_split,
    train_test_split,
)


@pytest.fixture(params=[6, 12], ids=lambda x: f"test_size({x})")
def test_size(request):
    return request.param


@pytest.fixture(params=[3, 5, 10], ids=lambda x: f"n_splits({x})")
def n_splits(request):
    return request.param


@pytest.fixture(params=[1, 3], ids=lambda x: f"step_size({x})")
def step_size(request):
    return request.param


def test_train_test_split_int_size(test_size, pl_y, benchmark):
    def _split(y):
        y_train, y_test = train_test_split(test_size)(y)
        return pl.collect_all([y_train, y_test])

    y_train, y_test = benchmark(_split, pl_y)

    # Check column names
    entity_col, time_col = pl_y.columns[:2]
    assert y_train.columns == y_test.columns

    # Check train window lengths
    ts_lengths = (
        pl_y.group_by(entity_col, maintain_order=True).agg(pl.count(time_col)).collect()
    )
    train_lengths = y_train.group_by(entity_col, maintain_order=True).agg(
        pl.count(time_col)
    )
    assert (
        ((ts_lengths.select("time") - train_lengths.select("time")) == test_size)
        .to_series()
        .all()
    )

    # Check test window lengths
    test_lengths = y_test.group_by(entity_col).agg(pl.count(time_col))
    assert (test_lengths.select("time") == test_size).to_series().all()


@pytest.mark.parametrize(
    "float_test_size,context",
    [
        (0.1, does_not_raise()),
        (0.5, does_not_raise()),
        (1.1, pytest.raises(ValueError)),
        (-0.1, pytest.raises(ValueError)),
    ],
)
def test_train_test_split_float_size(pl_y, float_test_size, context):
    with context as exc_info:
        y_train, y_test = train_test_split(float_test_size)(pl_y)

    if exc_info:
        assert "`test_size` must be between 0 and 1" in str(exc_info.value)

    else:
        entity_col, time_col = pl_y.columns[:2]
        assert y_train.columns == y_test.columns

        # Check train window lengths
        ts_lengths = (
            pl_y.group_by(entity_col, maintain_order=True)
            .agg(pl.count(time_col))
            .collect()
        )

        test_lengths = (
            y_test.group_by(entity_col, maintain_order=True)
            .agg(pl.count(time_col))
            .collect()
        )

        assert (
            (
                (test_lengths.select("time") / ts_lengths.select("time"))
                == float_test_size
            )
            .to_series()
            .all()
        )


def test_expanding_window_split(test_size, n_splits, step_size, pl_y, benchmark):
    def _split(y):
        cv = expanding_window_split(
            test_size=test_size, n_splits=n_splits, step_size=step_size
        )
        splits = cv(y)
        return {i: pl.collect_all(s) for i, s in splits.items()}

    splits = benchmark(_split, pl_y)
    entity_col, time_col = pl_y.columns[:2]

    for split in splits.values():
        _, y_test = split
        # Check test window lengths
        test_lengths = y_test.group_by(entity_col, maintain_order=True).agg(
            pl.count(time_col)
        )
        assert (test_lengths.select("time") == test_size).select(pl.all().all())[0, 0]


def test_sliding_window_split(test_size, n_splits, step_size, pl_y, benchmark):
    def _split(y):
        cv = sliding_window_split(
            test_size=test_size,
            n_splits=n_splits,
            step_size=step_size,
        )
        splits = cv(y)
        return {i: pl.collect_all(s) for i, s in splits.items()}

    splits = benchmark(_split, pl_y)
    entity_col, time_col = pl_y.columns[:2]

    for split in splits.values():
        _, y_test = split
        # Check test window lengths
        test_lengths = y_test.group_by(entity_col, maintain_order=True).agg(
            pl.count(time_col)
        )
        assert (test_lengths.select("time") == test_size).select(pl.all().all())[0, 0]


# --------------------------------------------------------------------------- #
# D3: the row-based splitters here vs. the panel-aware splitters in
# `panelary.core.model_selection`. These pin down exactly where the two agree
# (so the shared schedule kernel can never drift) and exactly where they do
# not (so nobody merges two different behaviours by accident).
# --------------------------------------------------------------------------- #
def _balanced_panel(n_entities: int = 3, n_times: int = 20) -> pl.DataFrame:
    """Balanced panel: every entity observed on every time step."""
    return pl.DataFrame(
        {
            "entity": [e for e in range(n_entities) for _ in range(n_times)],
            "time": [t for _ in range(n_entities) for t in range(n_times)],
            "y": [
                float(e * 100 + t) for e in range(n_entities) for t in range(n_times)
            ],
        }
    )


def _keys(frame) -> list[tuple]:
    """The (entity, time) keys a fold selected, order-insensitive."""
    if hasattr(frame, "to_frame"):  # PanelFrame
        frame = frame.to_frame()
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    return sorted(frame.select("entity", "time").iter_rows())


_GRID = [
    (n_times, test_size, n_splits, step_size)
    for n_times in (20, 30)
    for test_size in (2, 3, 5)
    for n_splits in (2, 3, 5)
    for step_size in (1, 2, 3)
]


@pytest.mark.parametrize("n_times,ts,ns,ss", _GRID)
def test_expanding_agrees_with_panel_splitter(n_times, ts, ns, ss):
    """On a balanced panel the two expanding splitters place identical folds."""
    from panelary.core.model_selection import (
        expanding_window_split as panel_expanding,
    )

    df = _balanced_panel(n_times=n_times)
    row_folds = expanding_window_split(test_size=ts, n_splits=ns, step_size=ss)(
        df.lazy()
    )
    panel_folds = panel_expanding(test_size=ts, n_splits=ns, step_size=ss)(df)

    assert len(row_folds) == len(panel_folds) == ns
    for i in range(ns):
        row_train, row_test = row_folds[i]
        panel_train, panel_test = panel_folds[i]
        assert _keys(row_train) == _keys(panel_train), f"train mismatch at split {i}"
        assert _keys(row_test) == _keys(panel_test), f"test mismatch at split {i}"


@pytest.mark.parametrize("n_times,ts,ns,ss", _GRID)
@pytest.mark.parametrize("ws", [3, 5, 10])
def test_sliding_agrees_with_panel_splitter(n_times, ts, ns, ss, ws):
    """On a balanced panel the two sliding splitters place identical folds.

    Includes the short-history corner (``window_size`` longer than the history
    before the first test block), where the training window must be *truncated
    at the start of history* rather than wrapping around to the end of it.
    """
    from panelary.core.model_selection import sliding_window_split as panel_sliding

    df = _balanced_panel(n_times=n_times)
    row_folds = sliding_window_split(
        test_size=ts, n_splits=ns, step_size=ss, window_size=ws
    )(df.lazy())
    panel_folds = panel_sliding(
        test_size=ts, n_splits=ns, step_size=ss, window_size=ws
    )(df)

    assert len(row_folds) == len(panel_folds) == ns
    for i in range(ns):
        row_train, row_test = row_folds[i]
        panel_train, panel_test = panel_folds[i]
        assert _keys(row_train) == _keys(panel_train), f"train mismatch at split {i}"
        assert _keys(row_test) == _keys(panel_test), f"test mismatch at split {i}"


@pytest.mark.parametrize("ws", [3, 5, 10, 25])
@pytest.mark.parametrize("ts,ns,ss", [(2, 5, 3), (3, 5, 2), (5, 3, 3), (5, 5, 3)])
def test_sliding_window_train_never_reaches_into_the_future(ws, ts, ns, ss):
    """No lookahead: the training window is always strictly before the test block.

    A negative slice offset in Polars counts back from the *end* of the group, so
    a window longer than the available history silently trained on rows after the
    test block. Every fold must satisfy ``max(train_time) < min(test_time)``.
    """
    df = _balanced_panel(n_times=20)
    folds = sliding_window_split(
        test_size=ts, n_splits=ns, step_size=ss, window_size=ws
    )(df.lazy())
    for i, (train, test) in folds.items():
        train_times = [t for _, t in _keys(train)]
        test_times = [t for _, t in _keys(test)]
        assert test_times, f"split {i}: empty test block"
        if not train_times:
            continue
        assert max(train_times) < min(test_times), (
            f"split {i}: sliding training window reaches into the future "
            f"(max train {max(train_times)} >= min test {min(test_times)})"
        )
        span = max(train_times) - min(train_times) + 1
        assert span <= ws, f"split {i}: training span {span} exceeds window_size {ws}"


def test_row_and_panel_splitters_diverge_on_a_ragged_panel():
    """The deliberate semantic difference: which axis is sliced.

    The row-based splitters here slice **each entity's own rows**, so on a panel
    whose entities end on different dates each entity is tested on its own last
    ``test_size`` observations. The panel-aware splitters in
    ``core.model_selection`` slice the **shared unique-time index**, so every
    entity is tested on the same dates. This is not a bug in either; it is why
    they are two functions and must not be collapsed into one.
    """
    from panelary.core.model_selection import (
        expanding_window_split as panel_expanding,
    )

    keys = [(0, t) for t in range(20)] + [(1, t) for t in range(15)]
    df = pl.DataFrame(
        {
            "entity": [e for e, _ in keys],
            "time": [t for _, t in keys],
            "y": [0.0] * len(keys),
        }
    )
    row_folds = expanding_window_split(test_size=3, n_splits=2, step_size=2)(df.lazy())
    panel_folds = panel_expanding(test_size=3, n_splits=2, step_size=2)(df)

    # Row-based: entity-relative, so the short entity is tested on earlier dates.
    assert _keys(row_folds[0][1]) == [
        (0, 15),
        (0, 16),
        (0, 17),
        (1, 10),
        (1, 11),
        (1, 12),
    ]
    # Panel-aware: common dates only, so the short entity contributes nothing.
    assert _keys(panel_folds[0][1]) == [(0, 15), (0, 16), (0, 17)]


def test_shared_walk_forward_schedule_is_the_only_copy():
    """Both splitter families take their fold schedule from one kernel."""
    from panelary.core import model_selection as ms
    from panelary.cross_validation import _walk_forward_cutoffs

    assert ms._walk_forward_cutoffs is _walk_forward_cutoffs
    # test_size=3, n_splits=4, step_size=2: test blocks start 9, 7, 5, 3 from the
    # end of the index, oldest first.
    assert _walk_forward_cutoffs(3, 4, 2) == [9, 7, 5, 3]
    assert _walk_forward_cutoffs(2, 1, 5) == [2]
