"""Tests for the time-series conformal extensions in ``polars_features.conformal``.

The headline requirement: **temporal coverage must hold under drift where naive
(exchangeability-assuming) split conformal fails.** The calibration split used to
produce the scores must itself be purged and embargoed.
"""

from __future__ import annotations

import numpy as np
import pytest

from polars_features.conformal import (
    adaptive_conformal_intervals,
    conformal_calibration_split,
    conformal_pid_intervals,
    conformal_quantile,
    conformalized_quantile_regression,
    cqr_scores,
    nexcp_intervals,
    nexcp_quantile,
    split_conformal_interval,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _drifting_scores(seed: int = 0, n: int = 800) -> np.ndarray:
    """Absolute residuals whose scale ramps up over the second half.

    A textbook distribution shift: a model calibrated on the calm regime is
    badly over-confident once volatility rises.
    """
    rng = np.random.default_rng(seed)
    scale = np.concatenate([np.ones(n // 2), np.linspace(1.0, 6.0, n - n // 2)])
    return np.abs(rng.standard_normal(n)) * scale


def _stationary_scores(seed: int = 0, n: int = 600) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.abs(rng.standard_normal(n))


# --------------------------------------------------------------------------- #
# Conformal quantile
# --------------------------------------------------------------------------- #
def test_conformal_quantile_uses_the_finite_sample_correction():
    scores = np.arange(1.0, 101.0)
    # ceil(101 * 0.9) = 91 -> the 91st order statistic.
    assert conformal_quantile(scores, 0.1) == 91.0
    # Too few points for the level -> honestly infinite.
    assert conformal_quantile(np.arange(5.0), 0.01) == float("inf")


def test_conformal_quantile_attains_nominal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    draws = np.abs(rng.standard_normal((3000, 51)))
    hits = [row[50] <= conformal_quantile(row[:50], 0.1) for row in draws]
    assert np.mean(hits) >= 0.88


def test_conformal_quantile_validates():
    with pytest.raises(ValueError, match="no finite values"):
        conformal_quantile(np.array([np.nan]), 0.1)
    with pytest.raises(ValueError, match="alpha"):
        conformal_quantile(np.ones(10), 0.0)


def test_split_conformal_interval_is_a_constant_band():
    lo, hi = split_conformal_interval(np.arange(1.0, 101.0), np.array([0.0, 5.0]), 0.1)
    np.testing.assert_allclose(hi - lo, 2 * 91.0)


# --------------------------------------------------------------------------- #
# Acceptance: coverage under drift
# --------------------------------------------------------------------------- #
def test_naive_split_conformal_loses_coverage_under_drift():
    """The failure mode the adaptive methods exist to fix."""
    scores = _drifting_scores()
    radius = conformal_quantile(scores[:300], 0.1)
    assert (scores[400:] <= radius).mean() < 0.7  # nominal is 0.9


@pytest.mark.parametrize(
    ("name", "fn"),
    [
        ("aci", lambda s: adaptive_conformal_intervals(s, alpha=0.1, gamma=0.03)),
        ("pid", lambda s: conformal_pid_intervals(s, alpha=0.1, k_p=0.3, k_i=0.03)),
    ],
)
def test_adaptive_conformal_holds_coverage_under_drift(name, fn):
    """ACI and conformal PID both track the moving target; naive conformal cannot."""
    scores = _drifting_scores()
    res = fn(scores)
    assert res.target_coverage == pytest.approx(0.9)
    assert abs(res.coverage - 0.9) < 0.04, name
    # And specifically over the drifted tail, not just on average.
    assert res.covered[400:].mean() > 0.82, name


def test_adaptive_conformal_beats_naive_over_the_drifted_tail():
    scores = _drifting_scores()
    naive_radius = conformal_quantile(scores[:300], 0.1)
    naive_tail = (scores[400:] <= naive_radius).mean()
    aci = adaptive_conformal_intervals(scores, alpha=0.1, gamma=0.03)
    assert aci.covered[400:].mean() > naive_tail + 0.15


def test_adaptive_conformal_is_calibrated_on_stationary_data_too():
    res = adaptive_conformal_intervals(_stationary_scores(), alpha=0.1, gamma=0.01)
    assert abs(res.coverage - 0.9) < 0.05


def test_aci_widens_intervals_when_coverage_is_missed():
    scores = _drifting_scores()
    res = adaptive_conformal_intervals(scores, alpha=0.1, gamma=0.05)
    finite = np.isfinite(res.radius)
    early = res.radius[finite][50:150].mean()
    late = res.radius[finite][-150:].mean()
    assert late > 2.0 * early


def test_aci_alpha_recursion_is_the_gibbs_candes_update():
    scores = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
    gamma, alpha = 0.1, 0.2
    res = adaptive_conformal_intervals(scores, alpha=alpha, gamma=gamma, warmup=0)
    for t in range(len(scores) - 1):
        err = 0.0 if res.covered[t] else 1.0
        assert res.alpha_t[t + 1] == pytest.approx(
            res.alpha_t[t] + gamma * (alpha - err)
        )


def test_aci_rolling_window_forgets_the_distant_past():
    scores = _drifting_scores()
    full = adaptive_conformal_intervals(scores, alpha=0.1, gamma=0.01)
    rolling = adaptive_conformal_intervals(scores, alpha=0.1, gamma=0.01, window=100)
    assert not np.allclose(full.radius[300:], rolling.radius[300:])
    assert abs(rolling.coverage - 0.9) < 0.06


def test_pid_reduces_to_quantile_tracking_when_p_and_d_are_zero():
    scores = _stationary_scores()
    res = conformal_pid_intervals(scores, alpha=0.1, k_p=0.0, k_i=0.05, k_d=0.0)
    assert np.all(np.isnan(res.alpha_t))
    assert abs(res.coverage - 0.9) < 0.06


def test_pid_scorecaster_shifts_the_radius_and_is_compensated_for():
    """The scorecast enters the radius directly; the controller then absorbs it."""
    scores = _stationary_scores()
    plain = conformal_pid_intervals(scores, alpha=0.1)
    cast = conformal_pid_intervals(
        scores, alpha=0.1, scorecast=np.full(scores.size, 1.0)
    )
    # Before any feedback, the radius is exactly shifted by the forecast.
    assert cast.radius[0] == pytest.approx(plain.radius[0] + 1.0)
    # The controller then pulls the tracked quantile back down, so coverage
    # stays on target instead of drifting to 1.
    assert abs(cast.coverage - 0.9) < 0.06
    assert not np.allclose(cast.radius, plain.radius + 1.0)


def test_pid_integral_is_clipped_against_windup():
    scores = np.full(300, 1e6)  # every step misses
    res = conformal_pid_intervals(scores, alpha=0.1, k_i=1.0, integral_clip=5.0)
    # With the clip, the radius grows at most k_i * clip per step.
    assert np.max(np.diff(res.radius)) <= 5.0 + 1e-9


def test_adaptive_methods_validate():
    with pytest.raises(ValueError, match="must not be empty"):
        adaptive_conformal_intervals(np.array([]))
    with pytest.raises(ValueError, match="gamma"):
        adaptive_conformal_intervals(np.ones(5), gamma=0.0)
    with pytest.raises(ValueError, match="alpha"):
        conformal_pid_intervals(np.ones(5), alpha=1.0)
    with pytest.raises(ValueError, match="integral_clip"):
        conformal_pid_intervals(np.ones(5), integral_clip=0.0)
    with pytest.raises(ValueError, match="scorecast"):
        conformal_pid_intervals(np.ones(5), scorecast=np.ones(4))


# --------------------------------------------------------------------------- #
# NexCP
# --------------------------------------------------------------------------- #
def test_nexcp_with_rho_one_reduces_to_split_conformal():
    scores = np.arange(1.0, 101.0)
    assert nexcp_quantile(scores, 0.1, rho=1.0) == conformal_quantile(scores, 0.1)


def test_nexcp_weighting_helps_under_drift():
    scores = _drifting_scores()
    weighted = nexcp_intervals(scores, alpha=0.1, rho=0.98)
    unweighted = nexcp_intervals(scores, alpha=0.1, rho=1.0)
    assert weighted.coverage > unweighted.coverage
    assert weighted.covered[400:].mean() > unweighted.covered[400:].mean()


def test_nexcp_smaller_rho_forgets_faster():
    scores = _drifting_scores()
    fast = nexcp_intervals(scores, alpha=0.1, rho=0.95)
    slow = nexcp_intervals(scores, alpha=0.1, rho=0.999)
    assert fast.radius[-1] > slow.radius[-1]


def test_nexcp_is_calibrated_on_exchangeable_data():
    res = nexcp_intervals(_stationary_scores(), alpha=0.1, rho=0.995)
    assert abs(res.coverage - 0.9) < 0.06
    assert np.all(res.alpha_t == 0.1)


def test_nexcp_validates():
    with pytest.raises(ValueError, match="rho"):
        nexcp_quantile(np.ones(10), 0.1, rho=1.5)
    with pytest.raises(ValueError, match="no finite values"):
        nexcp_quantile(np.array([np.nan]), 0.1)
    with pytest.raises(ValueError, match="must not be empty"):
        nexcp_intervals(np.array([]))


# --------------------------------------------------------------------------- #
# CQR
# --------------------------------------------------------------------------- #
def test_cqr_scores_are_negative_inside_and_positive_outside():
    y = np.array([0.0, 5.0, -5.0])
    lo, hi = np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    s = cqr_scores(y, lo, hi)
    np.testing.assert_allclose(s, [-1.0, 4.0, 4.0])
    with pytest.raises(ValueError, match="same shape"):
        cqr_scores(y, lo[:2], hi)


def test_cqr_inflates_a_too_narrow_band_and_shrinks_a_too_wide_one():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(500)
    narrow_lo, narrow_hi = np.full(500, -0.5), np.full(500, 0.5)
    lo, hi = conformalized_quantile_regression(
        y, narrow_lo, narrow_hi, narrow_lo[:1], narrow_hi[:1], alpha=0.1
    )
    assert (hi - lo)[0] > 1.0
    wide_lo, wide_hi = np.full(500, -5.0), np.full(500, 5.0)
    lo2, hi2 = conformalized_quantile_regression(
        y, wide_lo, wide_hi, wide_lo[:1], wide_hi[:1], alpha=0.1
    )
    assert (hi2 - lo2)[0] < 10.0


def test_cqr_attains_nominal_coverage_and_keeps_heteroskedastic_shape():
    rng = np.random.default_rng(1)
    n = 2000
    # Volatility varies across observations but is exchangeable in time, so the
    # calibration block is representative of the test block.
    sigma = rng.uniform(0.2, 3.0, n)
    y = rng.standard_normal(n) * sigma
    # A quantile model that knows the shape but is mis-scaled by 40%.
    band = 0.6 * 1.6448536269514722 * sigma
    calib, test = slice(0, 1000), slice(1000, n)
    lo, hi = conformalized_quantile_regression(
        y[calib],
        -band[calib],
        band[calib],
        -band[test],
        band[test],
        alpha=0.1,
    )
    coverage = float(np.mean((y[test] >= lo) & (y[test] <= hi)))
    assert coverage == pytest.approx(0.9, abs=0.04)
    # The band still tracks sigma (a constant-radius interval would not).
    width = hi - lo
    assert np.corrcoef(width, sigma[test])[0, 1] > 0.99


# --------------------------------------------------------------------------- #
# The calibration split must be purged and embargoed
# --------------------------------------------------------------------------- #
def test_conformal_calibration_split_is_purged_and_embargoed():
    train, calib = conformal_calibration_split(
        200, calibration_size=0.25, horizon=5, embargo=3
    )
    assert calib.size == 50
    assert train.max() < calib.min()
    assert int(calib.min()) - int(train.max()) > 5
    assert np.intersect1d(train, calib).size == 0


def test_conformal_calibration_split_delegates_to_validation():
    from polars_features.validation import purged_calibration_split

    a = conformal_calibration_split(120, calibration_size=30, horizon=2, embargo=1)
    b = purged_calibration_split(120, calibration_size=30, horizon=2, embargo=1)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_conformal_pipeline_end_to_end_on_a_purged_split():
    """Fit on train, calibrate on the purged block, then adapt online."""
    rng = np.random.default_rng(0)
    n = 400
    y = np.cumsum(rng.standard_normal(n) * 0.1)
    train, calib = conformal_calibration_split(
        n, calibration_size=0.3, horizon=2, embargo=2
    )
    level = float(np.mean(y[train]))  # a "model" fitted on train rows only
    calib_scores = np.abs(y[calib] - level)
    radius = conformal_quantile(calib_scores, 0.1)
    assert np.isfinite(radius)
    online = adaptive_conformal_intervals(calib_scores, alpha=0.1, gamma=0.05)
    assert 0.0 <= online.coverage <= 1.0
    assert online.n_warmup >= 0
