"""Tests for the CPCV validation runner and label-driven (t1) purge.

Covers:
* ``cross_validate`` end-to-end with a trivial sklearn-shaped stub estimator on
  a synthetic panel, for both PurgedKFold and CombinatorialPurgedCV.
* The CPCV path reconstruction / overfitting diagnostics on the ``CVReport``.
* The optional per-row event-end (``t1``) purge on PurgedKFold /
  CombinatorialPurgedCV.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_features.core.model_selection import (
    CombinatorialPurgedCV,
    CVReport,
    PurgedKFold,
    cross_validate,
    validate,
)
from polars_features.core.panel_frame import PanelFrame

ENTITIES = ["A", "B", "C"]


# --------------------------------------------------------------------------- #
# Fixtures / stubs
# --------------------------------------------------------------------------- #
class EchoRegressor:
    """A trivial sklearn-shaped estimator: predict = first feature (+ mean bias).

    Deterministic, needs no real training, and produces varying predictions so
    the CPCV path returns are non-degenerate.
    """

    def fit(self, X, y=None):
        self.bias_ = 0.0 if y is None else float(np.mean(y))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, 0] + self.bias_


def make_panel(n_times: int, entities=ENTITIES, seed: int = 0) -> PanelFrame:
    rng = np.random.default_rng(seed)
    rows_e, rows_t, x1, y = [], [], [], []
    for e in entities:
        for t in range(n_times):
            rows_e.append(e)
            rows_t.append(t)
            x1.append(float(rng.standard_normal()))
            y.append(float(rng.standard_normal()))
    df = pl.DataFrame({"entity": rows_e, "time": rows_t, "x1": x1, "target": y})
    return PanelFrame(df, entity="entity", time="time")


# --------------------------------------------------------------------------- #
# cross_validate with PurgedKFold
# --------------------------------------------------------------------------- #
def test_cross_validate_purged_kfold_runs():
    panel = make_panel(24)
    cv = PurgedKFold(n_splits=4, horizon=1, embargo=1)
    report = cross_validate(EchoRegressor(), panel, "target", cv)
    assert isinstance(report, CVReport)
    assert report.n_splits == 4
    assert len(report.fold_scores) == 4
    assert all(np.isfinite(s) for s in report.fold_scores)
    # No paths for a plain PurgedKFold.
    assert report.n_paths is None
    summ = report.summary()
    assert summ["metric"] == "neg_mean_squared_error"
    assert summ["n_splits"] == 4


def test_cross_validate_accepts_array_target():
    panel = make_panel(18)
    y = panel.sort_panel().collect().get_column("target").to_numpy()
    cv = PurgedKFold(n_splits=3)
    report = cross_validate(EchoRegressor(), panel.select("x1"), y, cv)
    assert report.n_splits == 3


def test_cross_validate_custom_metric():
    panel = make_panel(20)

    def mae(y_true, y_pred):
        return -float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    cv = PurgedKFold(n_splits=4)
    report = cross_validate(EchoRegressor(), panel, "target", cv, metric=mae)
    assert report.metric_name == "mae"


def test_cross_validate_rejects_index_splitter():
    panel = make_panel(12)
    cv = PurgedKFold(n_splits=3, return_indices=True)
    with pytest.raises(ValueError, match="return_indices=False"):
        cross_validate(EchoRegressor(), panel, "target", cv)


# --------------------------------------------------------------------------- #
# cross_validate with CombinatorialPurgedCV -> paths + diagnostics
# --------------------------------------------------------------------------- #
def test_cross_validate_cpcv_reconstructs_paths():
    panel = make_panel(48)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, horizon=1, embargo=1)
    report = cross_validate(EchoRegressor(), panel, "target", cv)

    assert report.n_splits == cv.n_splits
    assert report.n_paths == cv.n_paths
    assert report.path_scores is not None
    assert len(report.path_sharpes) == cv.n_paths
    # Each split records which groups it tested.
    assert all(g is not None for g in report.fold_test_groups)

    # Performance matrix is (n_periods, n_paths).
    M = report.performance_matrix
    assert M is not None
    assert M.shape[1] == cv.n_paths

    # Diagnostics are wired and in range.
    if report.deflated_sharpe is not None:
        assert 0.0 <= report.deflated_sharpe <= 1.0
    if report.pbo is not None:
        assert 0.0 <= report.pbo <= 1.0

    summ = report.summary()
    assert summ["n_paths"] == cv.n_paths
    assert "deflated_sharpe" in summ and "pbo" in summ


def test_validate_cpcv_convenience():
    panel = make_panel(40)
    report = validate.cpcv(
        EchoRegressor(), panel, "target", n_groups=5, n_test_groups=2, embargo=1
    )
    assert isinstance(report, CVReport)
    assert report.n_paths == CombinatorialPurgedCV(5, 2).n_paths


def test_validate_purged_kfold_convenience():
    panel = make_panel(20)
    report = validate.purged_kfold(
        EchoRegressor(), panel, "target", n_splits=4, horizon=1
    )
    assert report.n_splits == 4


# --------------------------------------------------------------------------- #
# Label-driven (t1) purge
# --------------------------------------------------------------------------- #
def _find_fold_with_test(cv, panel, target_test: set[int]):
    for train_pos, test_pos in cv.split(panel):
        if {int(p) for p in test_pos} == target_test:
            return {int(p) for p in train_pos}, {int(p) for p in test_pos}
    raise AssertionError(f"no fold tested exactly {target_test}")


def test_t1_purge_removes_overlapping_label():
    # times 0..9; 2 folds -> test folds are {0..4} and {5..9}.
    panel = make_panel(10)
    n = 10

    # Baseline point labels (t1 == time): position 4's label [4,4] does NOT
    # overlap the test block {5..9}, so it stays in train.
    t1_point = np.arange(n)
    cv = PurgedKFold(n_splits=2, t1=t1_point, return_indices=True)
    train, _ = _find_fold_with_test(cv, panel, set(range(5, 10)))
    assert 4 in train

    # Now stretch position 4's label to end at time 6: [4,6] overlaps the test
    # block -> position 4 must be purged.
    t1_overlap = np.arange(n).astype(float)
    t1_overlap[4] = 6.0
    cv2 = PurgedKFold(n_splits=2, t1=t1_overlap, return_indices=True)
    train2, _ = _find_fold_with_test(cv2, panel, set(range(5, 10)))
    assert 4 not in train2
    # A far-away train point (position 0) is untouched.
    assert 0 in train2


def test_t1_scalar_horizon_still_default():
    # Without t1, the scalar-horizon path is used and position 4 stays in train
    # for the {5..9} test fold when horizon=0.
    panel = make_panel(10)
    cv = PurgedKFold(n_splits=2, horizon=0, return_indices=True)
    train, _ = _find_fold_with_test(cv, panel, set(range(5, 10)))
    assert 4 in train


def test_t1_from_column_name():
    # A panel carrying a per-row `t1` column; aggregated to max end-time per time.
    base = make_panel(10).sort_panel().collect()
    # Give the time==4 rows an end time of 6 (overlaps the {5..9} test block).
    t1_vals = [6 if t == 4 else t for t in base.get_column("time").to_list()]
    base = base.with_columns(pl.Series("t1", t1_vals))
    panel = PanelFrame(base, entity="entity", time="time")

    cv = PurgedKFold(n_splits=2, t1="t1", return_indices=True)
    train, _ = _find_fold_with_test(cv, panel, set(range(5, 10)))
    assert 4 not in train


def test_t1_length_mismatch_raises():
    panel = make_panel(10)
    cv = PurgedKFold(n_splits=2, t1=np.arange(3))  # wrong length
    with pytest.raises(ValueError, match="align with the sorted unique-time"):
        list(cv.split(panel))


def test_cpcv_t1_purges_overlap():
    panel = make_panel(24)
    n = 24
    t1 = np.arange(n).astype(float)
    # Stretch a label near a group boundary to force overlap-based purging.
    t1[11] = 20.0
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=1, t1=t1, return_indices=True)
    for train_pos, test_pos in cv.split(panel):
        test_set = {int(p) for p in test_pos}
        train_set = {int(p) for p in train_pos}
        # If any test position falls within [11, 20], position 11 must be purged.
        if any(11 <= p <= 20 for p in test_set) and 11 not in test_set:
            assert 11 not in train_set
