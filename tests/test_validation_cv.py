"""Tests for ``panelary.validation._cv``.

Covers the positional splitters (purge/embargo actually removes the right
positions), the purged conformal calibration split, CPCV path reconstruction,
and the §3 acceptance criterion: **CPCV lowers measured PBO relative to
walk-forward on a known-overfit fixture.**
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import polars as pl
import pytest

from panelary.core.panel_frame import PanelFrame
from panelary.validation import (
    CombinatorialPurgedCV,
    IndexSplit,
    cpcv_backtest_paths,
    cpcv_splits,
    fold_boundaries,
    probability_of_backtest_overfitting,
    purged_calibration_split,
    walk_forward_backtest_path,
    walk_forward_splits,
)


# --------------------------------------------------------------------------- #
# cpcv_splits
# --------------------------------------------------------------------------- #
def test_cpcv_splits_count_and_disjointness():
    splits = cpcv_splits(24, n_groups=6, n_test_groups=2)
    assert len(splits) == math.comb(6, 2)
    for sp in splits:
        assert isinstance(sp, IndexSplit)
        assert np.intersect1d(sp.train, sp.test).size == 0
        assert sp.test.size == 8  # 2 groups of 4


def test_cpcv_splits_test_groups_cover_every_combination():
    splits = cpcv_splits(20, n_groups=5, n_test_groups=2)
    combos = {sp.test_groups for sp in splits}
    assert combos == set(itertools.combinations(range(5), 2))


def test_cpcv_splits_match_the_panel_splitter_exactly():
    """The positional wrapper must reuse core's purge/embargo, not re-derive it."""
    n_times = 30
    df = pl.DataFrame(
        {
            "entity": ["a"] * n_times + ["b"] * n_times,
            "time": list(range(n_times)) * 2,
            "x": np.arange(2 * n_times, dtype=float),
        }
    )
    pf = PanelFrame(df, entity="entity", time="time")
    cv = CombinatorialPurgedCV(
        n_groups=5, n_test_groups=2, horizon=2, embargo=1, return_indices=True
    )
    reference = [(tr, te) for tr, te in cv.split(pf)]
    mine = cpcv_splits(n_times, n_groups=5, n_test_groups=2, horizon=2, embargo=1)
    assert len(reference) == len(mine)
    for (tr, te), sp in zip(reference, mine, strict=True):
        np.testing.assert_array_equal(tr, sp.train)
        np.testing.assert_array_equal(te, sp.test)


def test_cpcv_purge_removes_the_horizon_neighbourhood():
    horizon = 3
    splits = cpcv_splits(30, n_groups=5, n_test_groups=1, horizon=horizon)
    for sp in splits:
        for j in sp.train:
            assert np.min(np.abs(sp.test - j)) > horizon


def test_cpcv_embargo_removes_positions_after_each_test_block():
    embargo = 2
    splits = cpcv_splits(30, n_groups=5, n_test_groups=1, embargo=embargo)
    for sp in splits:
        end = int(sp.test.max())
        banned = set(range(end + 1, min(30, end + embargo + 1)))
        assert banned.isdisjoint(set(sp.train.tolist()))


# --------------------------------------------------------------------------- #
# walk_forward_splits
# --------------------------------------------------------------------------- #
def test_walk_forward_is_strictly_backward_looking():
    splits = walk_forward_splits(60, n_splits=4, test_size=10)
    assert len(splits) == 4
    prev_start = -1
    for sp in splits:
        assert sp.train.max() < sp.test.min()
        assert int(sp.test.min()) > prev_start
        prev_start = int(sp.test.min())


def test_walk_forward_purges_and_embargoes_the_boundary():
    horizon, embargo = 3, 2
    splits = walk_forward_splits(
        60, n_splits=3, test_size=10, horizon=horizon, embargo=embargo
    )
    for sp in splits:
        gap = int(sp.test.min()) - int(sp.train.max())
        assert gap > max(horizon, embargo)


def test_walk_forward_rolling_window_bounds_the_training_length():
    splits = walk_forward_splits(
        80, n_splits=4, test_size=10, expanding=False, window_size=15
    )
    for sp in splits:
        assert sp.train.size <= 15
    expanding = walk_forward_splits(80, n_splits=4, test_size=10, expanding=True)
    assert expanding[-1].train.size > splits[-1].train.size


def test_walk_forward_rejects_impossible_geometry():
    with pytest.raises(ValueError, match="no training history"):
        walk_forward_splits(20, n_splits=4, test_size=10)
    with pytest.raises(ValueError, match="n_splits"):
        walk_forward_splits(20, n_splits=0)


# --------------------------------------------------------------------------- #
# purged_calibration_split
# --------------------------------------------------------------------------- #
def test_purged_calibration_split_is_temporal_and_gapped():
    train, calib = purged_calibration_split(
        100, calibration_size=0.2, horizon=4, embargo=3
    )
    assert calib.size == 20
    assert train.max() < calib.min()
    assert int(calib.min()) - int(train.max()) > 4


def test_purged_calibration_split_accepts_absolute_size():
    train, calib = purged_calibration_split(50, calibration_size=10)
    assert calib.size == 10
    assert train.size == 40


def test_purged_calibration_split_validates():
    with pytest.raises(ValueError, match="calibration_size"):
        purged_calibration_split(10, calibration_size=10)
    with pytest.raises(ValueError, match="no training positions"):
        purged_calibration_split(10, calibration_size=5, horizon=20)


# --------------------------------------------------------------------------- #
# fold_boundaries
# --------------------------------------------------------------------------- #
def test_fold_boundaries_are_the_test_block_edges():
    splits = walk_forward_splits(60, n_splits=3, test_size=10)
    assert fold_boundaries(splits) == [30, 40, 50, 60]


def test_fold_boundaries_handles_non_contiguous_cpcv_tests():
    splits = cpcv_splits(24, n_groups=4, n_test_groups=2)
    cuts = fold_boundaries(splits)
    assert cuts == [0, 6, 12, 18, 24]


# --------------------------------------------------------------------------- #
# Backtest paths
# --------------------------------------------------------------------------- #
def test_cpcv_backtest_paths_cover_every_time_step_on_every_path():
    values = np.arange(24, dtype=float)
    paths = cpcv_backtest_paths(
        24, lambda tr, te: values[te], n_groups=4, n_test_groups=2
    )
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=2)
    assert paths.shape == (24, cv.n_paths)
    assert not np.isnan(paths).any()
    # A deterministic "prediction" makes every path identical to the input.
    for p in range(paths.shape[1]):
        np.testing.assert_array_equal(paths[:, p], values)


def test_cpcv_backtest_paths_differ_when_predictions_depend_on_the_train_set():
    def fit_predict(train, test):
        return np.full(test.shape[0], float(train.size))

    paths = cpcv_backtest_paths(
        30, fit_predict, n_groups=6, n_test_groups=2, horizon=2, embargo=1
    )
    assert paths.shape[1] == CombinatorialPurgedCV(n_groups=6, n_test_groups=2).n_paths
    # Different paths stitch different splits, so the columns are not identical.
    assert np.unique(paths, axis=1).shape[1] > 1


def test_cpcv_backtest_paths_validates_the_callback():
    with pytest.raises(ValueError, match="returned"):
        cpcv_backtest_paths(24, lambda tr, te: np.zeros(1), n_groups=4)


def test_walk_forward_backtest_path_returns_the_oos_tail():
    values = np.arange(60, dtype=float)
    pos, vals = walk_forward_backtest_path(
        60, lambda tr, te: values[te], n_splits=3, test_size=10
    )
    np.testing.assert_array_equal(pos, np.arange(30, 60))
    np.testing.assert_array_equal(vals, values[30:])


# --------------------------------------------------------------------------- #
# Acceptance (§3): CPCV lowers measured PBO vs walk-forward
# --------------------------------------------------------------------------- #
def _overfit_fixture(seed: int) -> np.ndarray:
    """One genuinely skilled strategy hidden among seven look-alike noise ones.

    A search over these eight candidates is exactly the situation PBO is meant
    to diagnose: the winner is only identifiable with enough out-of-sample data.
    """
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((240, 8)) * 0.01
    returns[:, 0] += 0.0035  # the one real edge
    return returns


def _pbo_under_cpcv(returns: np.ndarray) -> float:
    n_times, n_strats = returns.shape
    cols = [
        cpcv_backtest_paths(
            n_times,
            lambda train, test, j=j: returns[test, j],
            n_groups=6,
            n_test_groups=2,
        ).mean(axis=1)
        for j in range(n_strats)
    ]
    return probability_of_backtest_overfitting(
        np.column_stack(cols), n_partitions=8, statistic="sharpe"
    )


def _pbo_under_walk_forward(returns: np.ndarray) -> float:
    splits = walk_forward_splits(returns.shape[0], n_splits=3, test_size=40)
    oos = np.concatenate([sp.test for sp in splits])
    return probability_of_backtest_overfitting(
        returns[oos], n_partitions=8, statistic="sharpe"
    )


def test_cpcv_lowers_measured_pbo_versus_walk_forward():
    """Acceptance criterion from the econometric-integration plan, §3.

    Walk-forward leaves most of the sample un-tested, so the in-sample winner is
    resolved from a short out-of-sample record and often loses to a lucky rival.
    CPCV tests every observation on some path, which sharpens the selection and
    drives the measured PBO down.
    """
    results = np.array(
        [
            (_pbo_under_cpcv(r), _pbo_under_walk_forward(r))
            for r in (_overfit_fixture(s) for s in range(12))
        ]
    )
    cpcv_pbo, wf_pbo = results[:, 0], results[:, 1]
    assert cpcv_pbo.mean() < wf_pbo.mean()
    # And CPCV is never *worse* on any of the fixed seeds.
    assert np.all(cpcv_pbo <= wf_pbo)
