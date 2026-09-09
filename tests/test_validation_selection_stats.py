"""Tests for ``panelary.validation._selection_stats``.

Includes the two §3 acceptance criteria that live here:

* **DSR matches hand-computed golden values.** The golden numbers below were
  produced from the published Bailey & Lopez de Prado formulae evaluated with an
  independent high-precision normal CDF/PPF (SciPy), then frozen. They pin both
  halves of the ``(N, V)`` parameterisation — the expected-maximum threshold
  ``SR_0`` and the PSR evaluated at it.
* **Romano-Wolf controls the FWER** on a simulated correlated panel where every
  null is true.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from panelary.validation import (
    benjamini_hochberg,
    benjamini_yekutieli,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    holm_bonferroni,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    romano_wolf,
    romano_wolf_mean_test,
)
from panelary.validation._forecast_tests import _t_sf

# --------------------------------------------------------------------------- #
# Golden values (independent high-precision evaluation of the published
# formulae; see the module docstring).  (n_trials, V, SR_hat, T) -> (SR_0, DSR)
# --------------------------------------------------------------------------- #
DSR_GOLDEN = [
    ((100, 0.01, 0.10, 1001), (0.2530602893201685, 6.890846633e-07)),
    ((10, 0.0025, 0.05, 251), (0.0787299150672875, 0.3249229797565907)),
    ((1000, 0.04, 0.30, 501), (0.6510243027305447, 8.1e-15)),
]


@pytest.mark.parametrize(("inputs", "expected"), DSR_GOLDEN)
def test_expected_maximum_sharpe_matches_golden(inputs, expected):
    n_trials, v, _, _ = inputs
    sr0, _ = expected
    assert expected_maximum_sharpe(n_trials, v) == pytest.approx(sr0, rel=1e-8)


@pytest.mark.parametrize(("inputs", "expected"), DSR_GOLDEN)
def test_deflated_sharpe_ratio_matches_golden(inputs, expected):
    n_trials, v, sr, n_obs = inputs
    _, dsr = expected
    got = deflated_sharpe_ratio(
        sr,
        n_trials=n_trials,
        n_observations=n_obs,
        sharpe_variance_across_trials=v,
    )
    assert got == pytest.approx(dsr, rel=1e-6, abs=1e-16)


def test_deflated_sharpe_ratio_matches_golden_with_skew_and_kurtosis():
    """A fat-tailed, negatively skewed series raises sigma(SR) and lowers DSR."""
    got = deflated_sharpe_ratio(
        0.12,
        n_trials=50,
        n_observations=1001,
        sharpe_variance_across_trials=0.01,
        skewness=-0.5,
        kurtosis=6.0,
    )
    assert got == pytest.approx(0.0005224997636586, rel=1e-6)


def test_dsr_is_psr_evaluated_at_the_expected_maximum():
    """The N/V parameterisation: DSR == PSR(SR_0(N, V)), exactly."""
    sr, n_obs, n_trials, v = 0.14, 750, 250, 0.02
    sr0 = expected_maximum_sharpe(n_trials, v)
    assert deflated_sharpe_ratio(
        sr, n_trials=n_trials, n_observations=n_obs, sharpe_variance_across_trials=v
    ) == pytest.approx(
        probabilistic_sharpe_ratio(sr, n_observations=n_obs, benchmark_sharpe=sr0)
    )


def test_expected_maximum_sharpe_uses_both_n_and_v():
    base = expected_maximum_sharpe(100, 0.01)
    assert expected_maximum_sharpe(1000, 0.01) > base  # more trials -> higher bar
    assert expected_maximum_sharpe(100, 0.04) > base  # more dispersion -> higher bar
    # Scale: SR_0 is proportional to sqrt(V).
    assert expected_maximum_sharpe(100, 0.04) == pytest.approx(2.0 * base)


def test_expected_maximum_sharpe_edge_cases():
    assert expected_maximum_sharpe(1, 0.25) == 0.0
    assert expected_maximum_sharpe(500, 0.0) == 0.0
    with pytest.raises(ValueError, match="n_trials"):
        expected_maximum_sharpe(0, 0.01)
    with pytest.raises(ValueError, match="must be >= 0"):
        expected_maximum_sharpe(10, -1.0)


def test_expected_maximum_sharpe_tracks_the_true_gumbel_maximum():
    """Sanity check against a Monte-Carlo maximum of N standard normals."""
    rng = np.random.default_rng(0)
    for n_trials in (10, 100, 1000):
        draws = rng.standard_normal((20000, n_trials)).max(axis=1)
        assert expected_maximum_sharpe(n_trials, 1.0) == pytest.approx(
            draws.mean(), abs=0.08
        )


# --------------------------------------------------------------------------- #
# PSR / MinTRL
# --------------------------------------------------------------------------- #
def test_psr_is_monotone_in_sample_length_and_benchmark():
    assert probabilistic_sharpe_ratio(0.1, n_observations=2000) > (
        probabilistic_sharpe_ratio(0.1, n_observations=100)
    )
    assert probabilistic_sharpe_ratio(
        0.1, n_observations=500, benchmark_sharpe=0.05
    ) < (probabilistic_sharpe_ratio(0.1, n_observations=500))


def test_psr_penalises_negative_skew_and_fat_tails():
    clean = probabilistic_sharpe_ratio(0.1, n_observations=500)
    ugly = probabilistic_sharpe_ratio(
        0.1, n_observations=500, skewness=-1.5, kurtosis=9.0
    )
    assert ugly < clean


def test_min_track_record_length_inverts_the_psr():
    sr, conf = 0.08, 0.95
    n = minimum_track_record_length(sr, confidence=conf)
    assert probabilistic_sharpe_ratio(
        sr, n_observations=int(math.ceil(n))
    ) == pytest.approx(conf, abs=1e-3)


def test_min_track_record_length_requires_an_edge():
    with pytest.raises(ValueError, match="must exceed"):
        minimum_track_record_length(0.05, benchmark_sharpe=0.05)


# --------------------------------------------------------------------------- #
# PBO
# --------------------------------------------------------------------------- #
def test_pbo_is_high_for_pure_noise_and_low_for_a_dominant_strategy():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((200, 8)) * 0.01
    assert probability_of_backtest_overfitting(noise, n_partitions=8) > 0.25
    dominant = noise.copy()
    dominant[:, 0] += 0.02
    assert probability_of_backtest_overfitting(dominant, n_partitions=8) == 0.0


# --------------------------------------------------------------------------- #
# Romano-Wolf
# --------------------------------------------------------------------------- #
def _unadjusted_rejections(x: np.ndarray, alpha: float) -> int:
    t = x.mean(axis=0) / (x.std(axis=0, ddof=1) / math.sqrt(x.shape[0]))
    p = np.array([2.0 * _t_sf(abs(v), x.shape[0] - 1) for v in t])
    return int(np.count_nonzero(p <= alpha))


def test_romano_wolf_controls_fwer_on_a_correlated_panel():
    """Acceptance criterion from the plan, §3.

    Ten cross-correlated candidate signals, **all** with a true mean of zero.
    A family-wise error is any rejection at all. Unadjusted testing errs far
    more often than the nominal 5%; the stepdown does not.
    """
    alpha = 0.05
    reps = 40
    fwe_stepdown = 0
    fwe_unadjusted = 0
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        common = rng.standard_normal((120, 1))
        x = 0.7 * common + 0.7 * rng.standard_normal((120, 10))
        result = romano_wolf_mean_test(x, alpha=alpha, n_boot=199, seed=r)
        fwe_stepdown += result.n_rejected > 0
        fwe_unadjusted += _unadjusted_rejections(x, alpha) > 0
    assert fwe_stepdown / reps <= 0.15  # generous MC allowance around alpha=0.05
    assert fwe_unadjusted / reps > 0.25  # the problem being corrected
    assert fwe_stepdown < fwe_unadjusted


def test_romano_wolf_still_finds_a_genuine_signal():
    rng = np.random.default_rng(7)
    x = 0.5 * rng.standard_normal((200, 1)) + 0.5 * rng.standard_normal((200, 10))
    x[:, 3] += 0.6  # a large, unmistakable effect
    result = romano_wolf_mean_test(x, alpha=0.05, n_boot=299, seed=0)
    assert result.rejected[3]
    assert result.adjusted_pvalues[3] < 0.05


def test_romano_wolf_is_more_powerful_than_holm_under_strong_correlation():
    """The point of the stepdown: it learns the dependence instead of assuming it."""
    rng = np.random.default_rng(11)
    common = rng.standard_normal((300, 1))
    x = 0.95 * common + 0.05 * rng.standard_normal((300, 20)) + 0.115
    rw = romano_wolf_mean_test(x, alpha=0.05, n_boot=399, seed=1)
    t = x.mean(axis=0) / (x.std(axis=0, ddof=1) / math.sqrt(300))
    p = np.array([2.0 * _t_sf(abs(v), 299) for v in t])
    holm = holm_bonferroni(p, alpha=0.05)
    assert rw.n_rejected >= holm.n_rejected
    assert np.all(rw.adjusted_pvalues <= holm.adjusted_pvalues + 1e-12)


def test_romano_wolf_adjusted_pvalues_are_monotone_in_the_statistics():
    rng = np.random.default_rng(3)
    stats = np.array([4.0, 3.0, 2.0, 1.0, 0.5])
    null = rng.standard_normal((500, 5))
    res = romano_wolf(stats, null, alpha=0.05)
    assert np.all(np.diff(res.adjusted_pvalues) >= 0)


def test_romano_wolf_validates_shapes():
    with pytest.raises(ValueError, match="columns"):
        romano_wolf([1.0, 2.0], np.zeros((10, 3)))
    with pytest.raises(ValueError, match="2-D"):
        romano_wolf([1.0], np.zeros(10))
    with pytest.raises(ValueError, match="alpha"):
        romano_wolf([1.0], np.zeros((10, 1)), alpha=1.5)


def test_romano_wolf_mean_test_honours_fold_boundaries():
    """Passing boundaries changes the null draw, and stays deterministic."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((120, 4))
    a = romano_wolf_mean_test(x, n_boot=99, seed=0)
    b = romano_wolf_mean_test(x, n_boot=99, seed=0, boundaries=[40, 80])
    c = romano_wolf_mean_test(x, n_boot=99, seed=0, boundaries=[40, 80])
    np.testing.assert_allclose(b.adjusted_pvalues, c.adjusted_pvalues)
    assert not np.allclose(a.adjusted_pvalues, b.adjusted_pvalues)


# --------------------------------------------------------------------------- #
# FDR
# --------------------------------------------------------------------------- #
def test_benjamini_hochberg_matches_the_hand_worked_example():
    p = [0.001, 0.02, 0.04, 0.6]
    res = benjamini_hochberg(p, alpha=0.05)
    np.testing.assert_allclose(
        res.adjusted_pvalues, [0.004, 0.04, 0.0533333333, 0.6], rtol=1e-8
    )
    assert res.rejected.tolist() == [True, True, False, False]


def test_benjamini_yekutieli_is_uniformly_more_conservative():
    p = np.array([0.001, 0.008, 0.02, 0.04, 0.2, 0.6])
    bh = benjamini_hochberg(p, alpha=0.05)
    by = benjamini_yekutieli(p, alpha=0.05)
    assert np.all(by.adjusted_pvalues >= bh.adjusted_pvalues - 1e-12)
    assert by.n_rejected <= bh.n_rejected
    # BY multiplies by the harmonic number c(m).
    c_m = float(np.sum(1.0 / np.arange(1, p.size + 1)))
    np.testing.assert_allclose(
        by.adjusted_pvalues, np.minimum(bh.adjusted_pvalues * c_m, 1.0), rtol=1e-10
    )


def test_fdr_procedures_are_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = np.sort(rng.random(50))
    for res in (benjamini_hochberg(p), benjamini_yekutieli(p), holm_bonferroni(p)):
        assert np.all(np.diff(res.adjusted_pvalues) >= -1e-12)
        assert np.all((res.adjusted_pvalues >= 0) & (res.adjusted_pvalues <= 1))


def test_benjamini_hochberg_controls_fdr_on_a_simulation():
    rng = np.random.default_rng(2)
    m, m_true_null = 200, 180
    false_discovery_rates = []
    for _ in range(20):
        z = rng.standard_normal(m)
        z[m_true_null:] += 3.5
        p = 2.0 * (
            1.0
            - np.vectorize(lambda v: 0.5 * (1 + math.erf(v / math.sqrt(2))))(np.abs(z))
        )
        res = benjamini_hochberg(p, alpha=0.10)
        rejected = np.flatnonzero(res.rejected)
        if rejected.size:
            false_discovery_rates.append(float(np.mean(rejected < m_true_null)))
    assert np.mean(false_discovery_rates) <= 0.15


def test_multiple_test_result_reports_counts_and_method():
    res = benjamini_hochberg([0.001, 0.9])
    assert res.n_rejected == 1
    assert res.method == "benjamini-hochberg"
    assert res.alpha == 0.05


def test_pvalue_validation():
    with pytest.raises(ValueError, match="not be empty"):
        benjamini_hochberg([])
    with pytest.raises(ValueError, match="finite and in"):
        benjamini_hochberg([0.5, 1.5])


# --------------------------------------------------------------------------- #
# Honest (N, V) pairing through the CV runner
# --------------------------------------------------------------------------- #
class _EchoRegressor:
    """Trivial sklearn-shaped estimator: predict = first feature (+ mean bias)."""

    def fit(self, X, y=None):
        self.bias_ = 0.0 if y is None else float(np.mean(y))
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float)[:, 0] + self.bias_


def _signal_panel(n_times: int = 48, seed: int = 0):
    import polars as pl

    from panelary.core.panel_frame import PanelFrame

    rng = np.random.default_rng(seed)
    rows_e, rows_t, x1, y = [], [], [], []
    for entity in ("A", "B", "C"):
        for t in range(n_times):
            v = float(rng.standard_normal())
            rows_e.append(entity)
            rows_t.append(t)
            x1.append(v)
            y.append(v + 0.05 * float(rng.standard_normal()))
    return PanelFrame(
        pl.DataFrame({"entity": rows_e, "time": rows_t, "x1": x1, "target": y}),
        entity="entity",
        time="time",
    )


def test_cross_validate_uses_trial_sharpes_for_a_matched_n_and_v():
    """``V`` must be the variance of the *trial* Sharpes, paired with that ``N``.

    Without ``trial_sharpes`` the runner falls back to the dispersion of this
    strategy's CPCV paths, which is a proxy for trial dispersion rather than the
    quantity Bailey & de Prado define.
    """
    from panelary.core.model_selection import (
        CombinatorialPurgedCV,
        cross_validate,
    )

    panel = _signal_panel()
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, horizon=1, embargo=1)
    trials = [0.30, 0.02, -0.03, 0.01, -0.01, 0.0, 0.015, -0.02, 0.005, -0.008]

    report = cross_validate(_EchoRegressor(), panel, "target", cv, trial_sharpes=trials)
    assert report.n_trials == len(trials)
    assert report.sharpe_variance == pytest.approx(float(np.var(trials, ddof=1)))
    assert "dsr_n_trials_warning" not in report.extra

    # And the reported DSR is exactly the deflation implied by that (N, V).
    finite = [s for s in report.path_sharpes if math.isfinite(s)]
    expected = deflated_sharpe_ratio(
        max(finite),
        n_trials=len(trials),
        n_observations=48,
        sharpe_variance_across_trials=float(np.var(trials, ddof=1)),
    )
    assert report.deflated_sharpe == pytest.approx(expected)


def test_explicit_n_trials_overrides_the_trial_sample_size():
    from panelary.core.model_selection import (
        CombinatorialPurgedCV,
        cross_validate,
    )

    panel = _signal_panel()
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, horizon=1, embargo=1)
    report = cross_validate(
        _EchoRegressor(),
        panel,
        "target",
        cv,
        n_trials=5000,
        trial_sharpes=[0.1, 0.0, -0.05],
    )
    assert report.n_trials == 5000
    assert report.sharpe_variance == pytest.approx(
        float(np.var([0.1, 0.0, -0.05], ddof=1))
    )
