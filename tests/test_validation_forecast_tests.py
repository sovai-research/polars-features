"""Tests for ``panelary.validation._forecast_tests``.

Diebold-Mariano (HAC + Harvey-Leybourne-Newbold), Hansen's SPA, the Model
Confidence Set, and the proper scoring rules. The distribution helpers are
checked against closed forms rather than another library, so the suite stays
dependency-free.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from panelary.validation import (
    crps_ensemble,
    crps_from_quantiles,
    crps_gaussian,
    diebold_mariano,
    interval_score,
    model_confidence_set,
    newey_west_variance,
    pinball_loss,
    pinball_loss_expr,
    pit_histogram,
    pit_values,
    score_quantile_forecasts,
    superior_predictive_ability,
)
from panelary.validation._forecast_tests import _betainc, _t_sf


# --------------------------------------------------------------------------- #
# Distribution helpers
# --------------------------------------------------------------------------- #
def test_regularized_incomplete_beta_hits_known_values():
    assert _betainc(1.0, 1.0, 0.3) == pytest.approx(0.3)
    assert _betainc(2.0, 2.0, 0.5) == pytest.approx(0.5)
    assert _betainc(0.5, 0.5, 0.5) == pytest.approx(0.5)
    assert _betainc(3.0, 1.0, 0.4) == pytest.approx(0.4**3)


def test_student_t_survival_function_matches_closed_forms():
    # Cauchy (df=1): P(T > t) = 0.5 - atan(t)/pi
    for t in (0.0, 0.5, 2.0, 10.0, -3.0):
        assert _t_sf(t, 1.0) == pytest.approx(0.5 - math.atan(t) / math.pi, abs=1e-12)
    # df=2: P(T > t) = 0.5 * (1 - t / sqrt(2 + t^2))
    for t in (0.0, 1.0, 4.0, -2.5):
        assert _t_sf(t, 2.0) == pytest.approx(
            0.5 * (1.0 - t / math.sqrt(2.0 + t * t)), abs=1e-12
        )
    assert _t_sf(0.0, 17.0) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Newey-West
# --------------------------------------------------------------------------- #
def test_newey_west_with_zero_lags_is_the_sample_variance():
    x = np.array([1.0, 2.0, 3.0, 10.0, -4.0])
    assert newey_west_variance(x, lags=0) == pytest.approx(float(np.var(x)))


def test_newey_west_inflates_variance_for_a_persistent_series():
    rng = np.random.default_rng(0)
    n = 3000
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + rng.standard_normal()
    short = newey_west_variance(x, lags=0)
    long_run = newey_west_variance(x, lags=40)
    assert long_run > 3.0 * short  # true ratio is 1/(1-0.8)^2 = 25 on gamma_0 terms


def test_newey_west_validates():
    with pytest.raises(ValueError, match="at least 2"):
        newey_west_variance(np.array([1.0]))
    with pytest.raises(ValueError, match="lags"):
        newey_west_variance(np.arange(10.0), lags=-1)


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #
def test_dm_detects_a_clearly_better_model():
    rng = np.random.default_rng(0)
    loss_a = rng.standard_normal(300) ** 2
    loss_b = loss_a * 0.4
    res = diebold_mariano(loss_a, loss_b, alternative="greater")
    assert res.statistic > 0
    assert res.pvalue < 0.001
    assert res.mean_loss_differential > 0


def test_dm_does_not_reject_for_equally_good_models():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(400) ** 2
    loss_a = base + 0.01 * rng.standard_normal(400)
    loss_b = base + 0.01 * rng.standard_normal(400)
    res = diebold_mariano(loss_a, loss_b)
    assert res.pvalue > 0.10


def test_dm_is_antisymmetric_in_its_arguments():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(200) ** 2
    b = rng.standard_normal(200) ** 2
    ab = diebold_mariano(a, b)
    ba = diebold_mariano(b, a)
    assert ab.statistic == pytest.approx(-ba.statistic)
    assert ab.pvalue == pytest.approx(ba.pvalue)


def test_harvey_correction_shrinks_the_statistic_and_is_reported():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(60) ** 2
    b = a * 0.7
    raw = diebold_mariano(a, b, horizon=4, harvey_correction=False)
    corrected = diebold_mariano(a, b, horizon=4, harvey_correction=True)
    assert abs(corrected.statistic) < abs(raw.statistic)
    assert corrected.harvey_correction is True
    assert corrected.pvalue > raw.pvalue
    assert corrected.horizon == 4


def test_dm_uses_more_hac_lags_for_longer_horizons():
    rng = np.random.default_rng(4)
    n = 400
    e = rng.standard_normal(n + 5)
    # An MA(4) loss differential: ignoring the serial correlation overstates
    # significance, so the h=5 statistic must be smaller in magnitude.
    d = sum(e[k : k + n] for k in range(5)) / 5.0 + 0.15
    a = np.abs(d) + 1.0
    b = a - d
    h1 = diebold_mariano(a, b, horizon=1, harvey_correction=False)
    h5 = diebold_mariano(a, b, horizon=5, harvey_correction=False)
    assert abs(h5.statistic) < abs(h1.statistic)


def test_dm_validates_inputs():
    with pytest.raises(ValueError, match="same shape"):
        diebold_mariano(np.ones(5), np.ones(6))
    with pytest.raises(ValueError, match="horizon"):
        diebold_mariano(np.ones(5), np.zeros(5), horizon=0)
    with pytest.raises(ValueError, match="unknown `alternative`"):
        diebold_mariano(np.ones(5), np.zeros(5), alternative="up")
    with pytest.raises(ValueError, match="at least 3"):
        diebold_mariano(np.ones(2), np.zeros(2))
    with pytest.raises(ValueError, match="identical losses"):
        diebold_mariano(np.arange(10.0), np.arange(10.0))


# --------------------------------------------------------------------------- #
# SPA
# --------------------------------------------------------------------------- #
def _spa_fixture(seed: int, edge: float, n_models: int = 5):
    rng = np.random.default_rng(seed)
    bench = 1.0 + 0.3 * rng.standard_normal(200)
    models = bench[:, None] + 0.3 * rng.standard_normal((200, n_models))
    models[:, 0] -= edge
    return bench, models


def test_spa_rejects_when_a_model_genuinely_beats_the_benchmark():
    bench, models = _spa_fixture(0, edge=0.4)
    res = superior_predictive_ability(bench, models, n_boot=299, seed=0)
    assert res.statistic > 0
    assert res.pvalue_consistent < 0.05
    assert res.n_models == 5


def test_spa_does_not_reject_when_every_model_is_worse():
    rng = np.random.default_rng(1)
    bench = 1.0 + 0.3 * rng.standard_normal(200)
    models = bench[:, None] + 0.3 * rng.standard_normal((200, 5)) + 0.3
    res = superior_predictive_ability(bench, models, n_boot=299, seed=0)
    assert res.statistic == 0.0
    assert res.pvalue_consistent > 0.5


def test_spa_pvalues_are_ordered_lower_consistent_upper():
    bench, models = _spa_fixture(2, edge=0.05, n_models=20)
    res = superior_predictive_ability(bench, models, n_boot=499, seed=2)
    assert res.pvalue_lower <= res.pvalue_consistent <= res.pvalue_upper
    assert res.pvalue == res.pvalue_consistent


def test_spa_penalises_a_larger_search():
    """Adding worthless models must not make the winner look better."""
    rng = np.random.default_rng(5)
    bench = 1.0 + 0.3 * rng.standard_normal(300)
    good = bench - 0.06 + 0.3 * rng.standard_normal(300)
    junk = bench[:, None] + 0.3 * rng.standard_normal((300, 40)) + 0.02
    small = superior_predictive_ability(
        bench, good[:, None], n_boot=499, seed=1
    ).pvalue_upper
    large = superior_predictive_ability(
        bench, np.column_stack([good, junk]), n_boot=499, seed=1
    ).pvalue_upper
    assert large >= small


def test_spa_is_deterministic_and_validates():
    bench, models = _spa_fixture(3, edge=0.2)
    a = superior_predictive_ability(bench, models, n_boot=199, seed=9)
    b = superior_predictive_ability(bench, models, n_boot=199, seed=9)
    assert a == b
    with pytest.raises(ValueError, match="aligned"):
        superior_predictive_ability(np.ones(10), np.ones((9, 2)))
    with pytest.raises(ValueError, match="at least 3 periods"):
        superior_predictive_ability(np.ones(2), np.ones((2, 2)))


# --------------------------------------------------------------------------- #
# Model Confidence Set
# --------------------------------------------------------------------------- #
def test_mcs_keeps_only_the_dominant_model():
    rng = np.random.default_rng(0)
    losses = rng.standard_normal((300, 5)) ** 2
    losses[:, 2] *= 0.25
    res = model_confidence_set(losses, alpha=0.10, n_boot=299, seed=1)
    assert res.included == [2]
    assert set(res.excluded) == {0, 1, 3, 4}
    assert res.pvalues[2] > res.alpha


def test_mcs_keeps_everything_when_models_are_indistinguishable():
    rng = np.random.default_rng(3)
    losses = rng.standard_normal((300, 4)) ** 2
    res = model_confidence_set(losses, alpha=0.10, n_boot=299, seed=3)
    assert res.included == [0, 1, 2, 3]
    assert res.excluded == []
    assert np.all(res.pvalues > 0.10)


def test_mcs_trange_agrees_with_tmax_on_a_clear_case():
    rng = np.random.default_rng(0)
    losses = rng.standard_normal((300, 5)) ** 2
    losses[:, 2] *= 0.25
    a = model_confidence_set(losses, alpha=0.10, n_boot=299, seed=1)
    b = model_confidence_set(losses, alpha=0.10, n_boot=299, seed=1, statistic="trange")
    assert a.included == b.included


def test_mcs_pvalues_are_monotone_along_the_elimination_order():
    rng = np.random.default_rng(4)
    losses = rng.standard_normal((250, 6)) ** 2 * np.linspace(1.0, 2.5, 6)
    res = model_confidence_set(losses, alpha=0.05, n_boot=299, seed=0)
    seq = [res.pvalues[i] for i in res.excluded]
    assert all(b >= a - 1e-12 for a, b in zip(seq[:-1], seq[1:], strict=True))
    assert res.names is None


def test_mcs_echoes_names_and_validates():
    rng = np.random.default_rng(0)
    losses = rng.standard_normal((100, 3)) ** 2
    res = model_confidence_set(losses, n_boot=99, seed=0, names=["ar", "rw", "ml"])
    assert res.names == ["ar", "rw", "ml"]
    with pytest.raises(ValueError, match="at least 2 models"):
        model_confidence_set(losses[:, :1], n_boot=9)
    with pytest.raises(ValueError, match="unknown `statistic`"):
        model_confidence_set(losses, n_boot=9, statistic="tsum")
    with pytest.raises(ValueError, match="alpha"):
        model_confidence_set(losses, n_boot=9, alpha=0.0)


# --------------------------------------------------------------------------- #
# Proper scoring rules
# --------------------------------------------------------------------------- #
def test_crps_ensemble_is_zero_for_a_perfect_point_forecast():
    assert crps_ensemble(np.array([3.0]), np.full((1, 5), 3.0))[0] == pytest.approx(0.0)


def test_crps_ensemble_matches_the_gaussian_closed_form():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((1, 200_000))
    got = crps_ensemble(np.array([0.5]), samples)[0]
    expected = crps_gaussian(np.array([0.5]), np.array([0.0]), np.array([1.0]))[0]
    assert got == pytest.approx(expected, abs=0.01)


def test_crps_gaussian_matches_the_analytic_value_at_the_mean():
    # CRPS(N(0,1), 0) = 2 phi(0) - 1/sqrt(pi)
    expected = 2.0 / math.sqrt(2 * math.pi) - 1.0 / math.sqrt(math.pi)
    assert crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([1.0]))[
        0
    ] == pytest.approx(expected)


def test_crps_gaussian_scales_with_sigma():
    a = crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([1.0]))[0]
    b = crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([3.0]))[0]
    assert b == pytest.approx(3.0 * a)
    with pytest.raises(ValueError, match="strictly positive"):
        crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([0.0]))


def test_crps_is_minimised_by_the_true_distribution():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(4000)
    honest = crps_gaussian(y, np.zeros_like(y), np.ones_like(y)).mean()
    overconfident = crps_gaussian(y, np.zeros_like(y), np.full_like(y, 0.3)).mean()
    underconfident = crps_gaussian(y, np.zeros_like(y), np.full_like(y, 3.0)).mean()
    biased = crps_gaussian(y, np.full_like(y, 1.5), np.ones_like(y)).mean()
    assert honest < min(overconfident, underconfident, biased)


def test_pinball_loss_is_asymmetric_and_matches_by_hand():
    # tau = 0.9, under-prediction (y > q) is penalised 9x more than over.
    assert pinball_loss(np.array([1.0]), np.array([0.0]), 0.9)[0] == pytest.approx(0.9)
    assert pinball_loss(np.array([0.0]), np.array([1.0]), 0.9)[0] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="quantile"):
        pinball_loss(np.array([0.0]), np.array([0.0]), 1.0)


def test_pinball_loss_is_minimised_at_the_true_quantile():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(20000)
    tau = 0.75
    true_q = float(np.quantile(y, tau))
    best = pinball_loss(y, np.full_like(y, true_q), tau).mean()
    for offset in (-0.4, -0.1, 0.1, 0.4):
        assert pinball_loss(y, np.full_like(y, true_q + offset), tau).mean() > best


def test_pinball_expr_matches_the_numpy_implementation():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(50)
    q = rng.standard_normal(50)
    df = pl.DataFrame({"y": y, "q": q})
    got = df.select(pinball_loss_expr("y", "q", 0.3).alias("loss"))["loss"].to_numpy()
    np.testing.assert_allclose(got, pinball_loss(y, q, 0.3))


def test_crps_from_quantiles_approximates_the_gaussian_crps():
    levels = np.linspace(0.01, 0.99, 99)
    from panelary.core.model_selection import _norm_ppf

    q = np.array([[_norm_ppf(t) for t in levels]])
    got = crps_from_quantiles(np.array([0.0]), q, levels)[0]
    expected = crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([1.0]))[0]
    assert got == pytest.approx(expected, abs=0.01)


def test_crps_from_quantiles_validates():
    with pytest.raises(ValueError, match="levels"):
        crps_from_quantiles(np.zeros(2), np.zeros((2, 3)), [0.1, 0.9])
    with pytest.raises(ValueError, match="must lie in"):
        crps_from_quantiles(np.zeros(2), np.zeros((2, 2)), [0.0, 0.9])


def test_interval_score_penalises_misses_and_rewards_tightness():
    y = np.array([0.0, 5.0])
    lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    scores = interval_score(y, lo, hi, alpha=0.1)
    assert scores[0] == pytest.approx(2.0)  # inside: just the width
    assert scores[1] == pytest.approx(2.0 + 20.0 * 4.0)
    tight = interval_score(np.array([0.0]), np.array([-0.5]), np.array([0.5]), 0.1)
    assert tight[0] < scores[0]


def test_pit_values_are_uniform_for_a_calibrated_forecast():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(4000)
    samples = rng.standard_normal((4000, 400))
    u = pit_values(y, samples)
    assert u.min() >= 0.0 and u.max() <= 1.0
    assert u.mean() == pytest.approx(0.5, abs=0.02)
    hist = pit_histogram(u, n_bins=10)
    assert hist.height == 10
    assert hist["frequency"].sum() == pytest.approx(1.0)
    assert np.all(np.abs(hist["frequency"].to_numpy() - 0.1) < 0.03)


def test_pit_histogram_is_u_shaped_when_forecasts_are_overconfident():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(4000)
    samples = rng.standard_normal((4000, 400)) * 0.3  # far too narrow
    hist = pit_histogram(pit_values(y, samples), n_bins=10)
    freq = hist["frequency"].to_numpy()
    assert freq[0] + freq[-1] > 0.5  # mass piles up in the tails


def test_pit_histogram_validates():
    with pytest.raises(ValueError, match="no finite"):
        pit_histogram(np.array([np.nan]))
    with pytest.raises(ValueError, match="n_bins"):
        pit_histogram(np.array([0.5]), n_bins=0)


# --------------------------------------------------------------------------- #
# Polars scoring
# --------------------------------------------------------------------------- #
def _quantile_frame() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    y = rng.standard_normal(n)
    return pl.DataFrame(
        {
            "entity": ["a"] * (n // 2) + ["b"] * (n // 2),
            "y": y,
            "q10": np.full(n, -1.2815515655446004),
            "q50": np.zeros(n),
            "q90": np.full(n, 1.2815515655446004),
        }
    )


def test_score_quantile_forecasts_recovers_nominal_coverage():
    df = _quantile_frame()
    out = score_quantile_forecasts(
        df, y_true="y", quantile_cols=["q10", "q50", "q90"], levels=[0.1, 0.5, 0.9]
    )
    assert out.height == 1
    assert out["coverage_0.1"][0] == pytest.approx(0.1, abs=0.05)
    assert out["coverage_0.9"][0] == pytest.approx(0.9, abs=0.05)
    assert out["crps"][0] > 0


def test_score_quantile_forecasts_groups_and_matches_numpy():
    df = _quantile_frame()
    out = score_quantile_forecasts(
        df,
        y_true="y",
        quantile_cols=["q10", "q90"],
        levels=[0.1, 0.9],
        by="entity",
    )
    assert out.height == 2
    assert out["entity"].to_list() == ["a", "b"]
    sub = df.filter(pl.col("entity") == "a")
    expected = pinball_loss(sub["y"].to_numpy(), sub["q10"].to_numpy(), 0.1).mean()
    assert out.filter(pl.col("entity") == "a")["pinball_0.1"][0] == pytest.approx(
        expected
    )


def test_score_quantile_forecasts_accepts_lazyframes_and_validates():
    df = _quantile_frame()
    out = score_quantile_forecasts(
        df.lazy(), y_true="y", quantile_cols=["q50"], levels=[0.5]
    )
    assert out.height == 1
    with pytest.raises(ValueError, match="levels"):
        score_quantile_forecasts(
            df, y_true="y", quantile_cols=["q10"], levels=[0.1, 0.9]
        )
