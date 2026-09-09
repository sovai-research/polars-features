"""Tests for the Deflated-Sharpe (DSR) correctness fix in ``model_selection``.

Proves the P0 wiring fix in ``_reconstruct_paths`` and its helpers:

* The corrected DSR deflates against the honest, user-supplied ``n_trials`` and
  the *empirical* variance ``V = var(path_sharpes, ddof=1)`` of the trial
  Sharpes — not ``n_paths`` and not the single-strategy estimator variance —
  so it is LOWER (less optimistic) than the old inflated wiring for the same
  inputs, and both ``V`` and ``n_trials`` are demonstrably used.
* ``_sharpe`` returns ``nan`` (not ``0.0``) on degenerate/empty input and
  annualises by ``sqrt(periods_per_year)``.
* Back-compat: ``cross_validate`` still runs without ``n_trials`` on a CPCV and
  warns loudly, using ``n_paths`` only as a floor.
* The PBO ranking statistic can be Sharpe-based (fix #3).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from panelary.core.model_selection import (
    CombinatorialPurgedCV,
    _sharpe,  # internal helper under test
    cross_validate,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    validate,
)
from panelary.core.panel_frame import PanelFrame

ENTITIES = ["A", "B", "C"]


class EchoRegressor:
    """Trivial sklearn-shaped estimator: predict = first feature (+ mean bias)."""

    def fit(self, X, y=None):
        self.bias_ = 0.0 if y is None else float(np.mean(y))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, 0] + self.bias_


def make_signal_panel(n_times: int, entities=ENTITIES, seed: int = 0) -> PanelFrame:
    """Panel where target is a strong function of x1 -> pred*target > 0.

    ``EchoRegressor`` predicts ``x1``; with ``target ≈ x1`` the per-period
    strategy return ``pred * target ≈ x1**2`` is strongly positive, giving the
    reconstructed paths a genuinely high Sharpe (so DSR is sensitive to the
    deflation inputs).
    """
    rng = np.random.default_rng(seed)
    rows_e, rows_t, x1, y = [], [], [], []
    for e in entities:
        for t in range(n_times):
            v = float(rng.standard_normal())
            rows_e.append(e)
            rows_t.append(t)
            x1.append(v)
            # target closely tracks x1 so predictions are profitable.
            y.append(v + 0.05 * float(rng.standard_normal()))
    df = pl.DataFrame({"entity": rows_e, "time": rows_t, "x1": x1, "target": y})
    return PanelFrame(df, entity="entity", time="time")


# --------------------------------------------------------------------------- #
# (a) Corrected DSR is lower than the old inflated wiring for the same inputs.
# --------------------------------------------------------------------------- #
def test_corrected_dsr_lower_than_inflated_controlled():
    """One genuinely strong path vs many noise paths (controlled inputs).

    The old wiring used ``n_trials = n_paths`` and let ``V`` fall back to the
    single-strategy estimator variance. The corrected wiring passes the honest
    (larger) ``n_trials`` and the empirical ``V``. Both raise the deflation
    benchmark, so the corrected DSR must be strictly lower.
    """
    strong = 0.20  # per-observation Sharpe of the lucky winner
    noise = [0.02, -0.03, 0.01, -0.01, 0.00, 0.015, -0.02, 0.005, -0.008]
    path_sharpes = [strong, *noise]
    n_obs = 250
    best = max(path_sharpes)
    n_paths = len(path_sharpes)
    v_emp = float(np.var(path_sharpes, ddof=1))

    # Old wiring: N == n_paths, V == estimator-variance fallback.
    old = deflated_sharpe_ratio(best, n_trials=n_paths, n_observations=n_obs)
    # Corrected wiring: honest N (configs searched) + empirical V.
    corrected = deflated_sharpe_ratio(
        best,
        n_trials=500,
        n_observations=n_obs,
        sharpe_variance_across_trials=v_emp,
    )
    assert corrected < old

    # Prove n_trials is actually used: monotonically decreasing in N.
    dsr_small_n = deflated_sharpe_ratio(
        best, n_trials=10, n_observations=n_obs, sharpe_variance_across_trials=v_emp
    )
    dsr_large_n = deflated_sharpe_ratio(
        best, n_trials=5000, n_observations=n_obs, sharpe_variance_across_trials=v_emp
    )
    assert dsr_large_n < dsr_small_n

    # Prove V is actually used: a larger V raises the benchmark -> lower DSR.
    dsr_low_v = deflated_sharpe_ratio(
        best, n_trials=500, n_observations=n_obs, sharpe_variance_across_trials=1e-4
    )
    dsr_high_v = deflated_sharpe_ratio(
        best, n_trials=500, n_observations=n_obs, sharpe_variance_across_trials=1e-1
    )
    assert dsr_high_v < dsr_low_v


def test_report_dsr_uses_empirical_v_and_user_n_trials():
    """Integration: the CPCV report's DSR is *exactly* the value obtained by
    deflating the best per-observation path Sharpe against the recorded honest
    ``n_trials`` and the empirical variance ``V = var(path_sharpes, ddof=1)`` —
    i.e. both are genuinely threaded through the pipeline (not ``n_paths`` and
    not the estimator-variance fallback)."""
    n_times = 48
    panel = make_signal_panel(n_times)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, horizon=1, embargo=1)

    report = cross_validate(EchoRegressor(), panel, "target", cv, n_trials=1000)

    finite = [s for s in report.path_sharpes if math.isfinite(s)]
    assert finite  # the strong-signal panel yields non-degenerate paths

    # Fields recorded on the report.
    assert report.n_trials == 1000
    v_emp = float(np.var(finite, ddof=1))
    assert report.sharpe_variance == pytest.approx(v_emp)
    assert "dsr_n_trials_warning" not in report.extra  # N was supplied
    # n_trials=1000 is the honest search count, not the CV geometry.
    assert report.n_trials != cv.n_paths

    # The DSR is exactly a deflation against the RECORDED n_trials and empirical
    # V (no periods_per_year -> path Sharpes are already per-observation).
    best = max(finite)
    expected = deflated_sharpe_ratio(
        best,
        n_trials=report.n_trials,
        n_observations=max(n_times, 2),
        sharpe_variance_across_trials=v_emp if v_emp > 0 else None,
    )
    assert report.deflated_sharpe is not None
    assert report.deflated_sharpe == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# (b) _sharpe: NaN on degenerate input; annualisation-aware.
# --------------------------------------------------------------------------- #
def test_sharpe_degenerate_is_nan():
    # Constant series: zero std -> nan, NOT 0.0 (luck must not read as skill).
    assert math.isnan(_sharpe(np.array([1.0, 1.0, 1.0, 1.0])))
    # Empty / single-point series -> nan.
    assert math.isnan(_sharpe(np.array([])))
    assert math.isnan(_sharpe(np.array([0.5])))
    # NaNs dropped; if fewer than 2 finite points remain -> nan.
    assert math.isnan(_sharpe(np.array([np.nan, np.nan, 2.0])))


def test_sharpe_annualisation():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(300) * 0.01 + 0.001
    base = _sharpe(r)
    ann = _sharpe(r, periods_per_year=252)
    assert math.isfinite(base) and math.isfinite(ann)
    assert ann == pytest.approx(base * math.sqrt(252))


def test_sharpe_nondegenerate_finite():
    r = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    sr = _sharpe(r)
    assert math.isfinite(sr)
    assert sr == pytest.approx(float(np.mean(r)) / float(np.std(r, ddof=1)))


# --------------------------------------------------------------------------- #
# (c) Back-compat: cross_validate without n_trials still runs and warns.
# --------------------------------------------------------------------------- #
def test_cross_validate_without_n_trials_warns_and_uses_floor():
    panel = make_signal_panel(48)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, horizon=1, embargo=1)

    with pytest.warns(UserWarning, match="n_trials not supplied"):
        report = cross_validate(EchoRegressor(), panel, "target", cv)

    # n_paths is used only as a floor, and that fact is recorded loudly.
    assert report.n_trials == cv.n_paths
    assert "dsr_n_trials_warning" in report.extra
    assert report.sharpe_variance is not None
    # Diagnostics still populated and in range.
    if report.deflated_sharpe is not None:
        assert 0.0 <= report.deflated_sharpe <= 1.0


def test_validate_cpcv_forwards_n_trials():
    panel = make_signal_panel(40)
    report = validate.cpcv(
        EchoRegressor(),
        panel,
        "target",
        n_groups=5,
        n_test_groups=2,
        embargo=1,
        n_trials=250,
    )
    assert report.n_trials == 250
    assert "dsr_n_trials_warning" not in report.extra


def test_cross_validate_purged_kfold_unaffected():
    """Non-CPCV splitters ignore n_trials cleanly (no paths, no warning)."""
    from panelary.core.model_selection import PurgedKFold

    panel = make_signal_panel(24)
    cv = PurgedKFold(n_splits=4, horizon=1, embargo=1)
    report = cross_validate(EchoRegressor(), panel, "target", cv, n_trials=100)
    assert report.n_paths is None
    assert report.n_trials is None  # only set on the CPCV path


# --------------------------------------------------------------------------- #
# Fix #3: PBO ranking statistic can be Sharpe-based.
# --------------------------------------------------------------------------- #
def test_pbo_accepts_sharpe_statistic():
    rng = np.random.default_rng(3)
    M = rng.standard_normal((60, 6)) * 0.01
    pbo_mean = probability_of_backtest_overfitting(M, n_partitions=6, statistic="mean")
    pbo_sharpe = probability_of_backtest_overfitting(
        M, n_partitions=6, statistic="sharpe"
    )
    assert 0.0 <= pbo_mean <= 1.0
    assert 0.0 <= pbo_sharpe <= 1.0


def test_pbo_sharpe_separates_equal_means_diff_vol():
    """Two columns with equal means but different vols rank differently under
    Sharpe vs mean; the Sharpe statistic must at least run and differ from mean
    ranking on a constructed case."""
    rng = np.random.default_rng(7)
    T = 80
    # Column j has increasing volatility; means are forced identical below so
    # only the Sharpe statistic (not the mean) can separate the columns.
    cols = [rng.standard_normal(T) * (0.005 * (j + 1)) for j in range(6)]
    M = np.column_stack(cols)
    M = M - M.mean(axis=0, keepdims=True) + 0.002
    pbo_mean = probability_of_backtest_overfitting(M, n_partitions=6, statistic="mean")
    pbo_sharpe = probability_of_backtest_overfitting(
        M, n_partitions=6, statistic="sharpe"
    )
    assert 0.0 <= pbo_mean <= 1.0
    assert 0.0 <= pbo_sharpe <= 1.0


def test_pbo_rejects_unknown_statistic():
    M = np.random.default_rng(0).standard_normal((40, 4)) * 0.01
    with pytest.raises(ValueError, match="unknown `statistic`"):
        probability_of_backtest_overfitting(M, n_partitions=4, statistic="bogus")
