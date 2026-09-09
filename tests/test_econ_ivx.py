"""Tests for IVX predictive regression (``econ._ivx``).

The acceptance criterion is **correct size under a persistent-regressor null**:
with a local-to-unity, endogenous predictor and no true predictability, the IVX
test must reject at roughly its nominal rate while the OLS t-test over-rejects
badly. That contrast is the entire reason IVX exists, so the tests below assert
both halves of it.

The Monte-Carlo tests use modest replication counts (they must stay fast); the
thresholds are set with enough slack to be stable across platforms while still
failing loudly if the estimator regresses.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.econ import IVXSelector, ivx, ivx_instrument, ivx_screen


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def _simulate(
    n: int,
    *,
    c: float,
    corr: float,
    beta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Local-to-unity predictor with endogenous innovations.

    ``x_t = (1 - c/n) x_{t-1} + e_t``, ``y_t = beta x_{t-1} + u_t``,
    ``corr(u_t, e_t) = corr``.
    """
    e = rng.standard_normal(n)
    u = corr * e + np.sqrt(1.0 - corr**2) * rng.standard_normal(n)
    rho = 1.0 - c / n
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    y = beta * np.concatenate([[0.0], x[:-1]]) + u
    return y[1:], x[:-1]


def _rejection_rates(
    *, n: int, c: float, corr: float, beta: float, reps: int, seed: int
) -> tuple[float, float]:
    """Empirical rejection rate at the 5% level for IVX and for OLS."""
    rng = np.random.default_rng(seed)
    ivx_rej = ols_rej = 0
    for _ in range(reps):
        y, x = _simulate(n, c=c, corr=corr, beta=beta, rng=rng)
        res = ivx(y, x)
        ivx_rej += res.pvalues[0] < 0.05
        ols_rej += abs(res.ols_tstats[0]) > 1.96
    return ivx_rej / reps, ols_rej / reps


# --------------------------------------------------------------------------- #
# The instrument
# --------------------------------------------------------------------------- #
def test_instrument_matches_its_defining_recursion():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(200))[:, None]
    z = ivx_instrument(x, delta=0.9, c_z=1.0)
    rho = 1.0 - 1.0 / (200**0.9)
    dx = np.diff(x, axis=0)
    manual = np.zeros_like(dx)
    acc = np.zeros(1)
    for i in range(dx.shape[0]):
        acc = rho * acc + dx[i]
        manual[i] = acc
    np.testing.assert_allclose(z, manual, atol=1e-12)
    assert z.shape == (199, 1)


def test_instrument_is_mildly_integrated():
    """``rho_z`` sits strictly between 0 and 1 and approaches 1 as n grows."""
    rng = np.random.default_rng(1)
    small = ivx(*_simulate(200, c=2.0, corr=-0.9, beta=0.0, rng=rng))
    large = ivx(*_simulate(2000, c=2.0, corr=-0.9, beta=0.0, rng=rng))
    assert 0.0 < small.rho_z < large.rho_z < 1.0


def test_instrument_validates_its_tuning_constants():
    x = np.arange(50.0)
    with pytest.raises(ValueError, match=r"`delta` must lie in \(0, 1\)"):
        ivx_instrument(x, delta=1.5)
    with pytest.raises(ValueError, match="`c_z` must be positive"):
        ivx_instrument(x, c_z=0.0)
    with pytest.raises(ValueError, match="at least 3 observations"):
        ivx_instrument(np.array([1.0, 2.0]))


# --------------------------------------------------------------------------- #
# Size: the headline acceptance test
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("c", [1.0, 5.0, 20.0])
def test_size_under_a_persistent_endogenous_null(c):
    ivx_rate, ols_rate = _rejection_rates(
        n=400, c=c, corr=-0.95, beta=0.0, reps=300, seed=100 + int(c)
    )
    # Nominal 5%: IVX stays close; the exact rate drifts up slightly as the root
    # approaches unity, which matches the published simulation evidence.
    assert ivx_rate < 0.12, f"IVX over-rejects at c={c}: {ivx_rate:.3f}"
    assert ivx_rate <= ols_rate + 1e-9, "IVX must not be worse sized than OLS"


def test_ols_over_rejects_where_ivx_does_not():
    ivx_rate, ols_rate = _rejection_rates(
        n=400, c=1.0, corr=-0.95, beta=0.0, reps=300, seed=7
    )
    assert ols_rate > 0.15  # the classic Stambaugh-type size distortion
    assert ivx_rate < ols_rate / 2.0


def test_size_is_close_to_nominal_without_endogeneity():
    ivx_rate, _ = _rejection_rates(n=400, c=5.0, corr=0.0, beta=0.0, reps=300, seed=11)
    assert 0.01 < ivx_rate < 0.11


def test_has_power_against_true_predictability():
    ivx_rate, _ = _rejection_rates(
        n=400, c=5.0, corr=-0.9, beta=0.08, reps=150, seed=13
    )
    assert ivx_rate > 0.8


# --------------------------------------------------------------------------- #
# Estimation surface
# --------------------------------------------------------------------------- #
def test_recovers_the_coefficient_of_a_stationary_predictor():
    rng = np.random.default_rng(3)
    y, x = _simulate(4000, c=200.0, corr=-0.5, beta=0.3, rng=rng)
    res = ivx(y, x, names=["dp"])
    assert res.names == ["dp"]
    assert abs(res.params[0] - 0.3) < 0.05
    assert res.pvalues[0] < 1e-6


def test_result_surface():
    rng = np.random.default_rng(4)
    y, x = _simulate(500, c=5.0, corr=-0.9, beta=0.05, rng=rng)
    res = ivx(y, x, names=["dp"])
    summary = res.summary()
    assert summary.columns == [
        "term",
        "estimate",
        "std_error",
        "ivx_wald",
        "p_value",
        "ols_estimate",
        "ols_t",
    ]
    assert res.nobs == 498
    # The individual Wald statistic is the squared studentised coefficient.
    np.testing.assert_allclose(
        res.wald[0], (res.params[0] / res.std_errors[0]) ** 2, rtol=1e-10
    )
    assert 0.0 <= res.joint_pvalue <= 1.0


def test_multivariate_ivx():
    rng = np.random.default_rng(5)
    n = 1500
    e = rng.standard_normal((n, 2))
    x = np.zeros((n, 2))
    for t in range(1, n):
        x[t] = np.array([0.99, 0.9]) * x[t - 1] + e[t]
    u = -0.8 * e[:, 0] + 0.6 * rng.standard_normal(n)
    y = 0.05 * x[:-1, 0] + u[1:]
    res = ivx(y, x[:-1], names=["p1", "p2"])
    assert res.params.shape == (2,)
    assert res.vcov.shape == (2, 2)
    assert res.pvalues[0] < 0.1
    assert res.joint_wald > 0


def test_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="observations"):
        ivx(np.zeros(10), np.zeros(9))
    with pytest.raises(ValueError, match="at least"):
        ivx(np.zeros(3), np.zeros(3))


# --------------------------------------------------------------------------- #
# Panel screening
# --------------------------------------------------------------------------- #
def _screen_panel(
    n_entities: int = 6, n_times: int = 300, seed: int = 0
) -> pl.DataFrame:
    """One genuinely predictive column, one persistent-but-useless one."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_entities):
        y, good = _simulate(n_times, c=5.0, corr=-0.5, beta=0.15, rng=rng)
        _, junk = _simulate(n_times, c=1.0, corr=-0.95, beta=0.0, rng=rng)
        frames.append(
            pl.DataFrame(
                {
                    "id": np.full(y.size, i),
                    "t": np.arange(y.size),
                    "y": y,
                    "good": good,
                    "junk": junk,
                }
            )
        )
    return pl.concat(frames)


def test_screen_ranks_the_predictive_feature_first():
    ranking = ivx_screen(
        _screen_panel(seed=21),
        y="y",
        candidates=["good", "junk"],
        entity="id",
        time="t",
    )
    assert ranking.columns == [
        "feature",
        "ivx_wald",
        "p_value",
        "n_entities",
        "mean_beta",
    ]
    assert ranking["feature"][0] == "good"
    assert ranking["p_value"][0] < 0.01
    assert ranking["n_entities"][0] == 6


def test_screen_rejects_unknown_columns():
    with pytest.raises(ValueError, match="not found"):
        ivx_screen(
            _screen_panel(n_entities=2, n_times=60),
            y="y",
            candidates=["nope"],
            entity="id",
            time="t",
        )


def test_selector_freezes_its_selection():
    df = _screen_panel(seed=22)
    train = df.filter(pl.col("t") < 150)
    sel = IVXSelector(y="y", k=1).fit(train, entity="id", time="t")
    assert IVXSelector.panel_safe is True
    assert IVXSelector.leakage_safe is True
    assert sel.selected_ == ["good"]
    frozen = list(sel.selected_)
    out = sel.transform(df).collect()
    assert sel.selected_ == frozen  # transform never re-selects
    assert set(out.columns) == {"id", "t", "good"}


def test_selector_alpha_mode_and_validation():
    df = _screen_panel(n_entities=4, n_times=250, seed=23)
    sel = IVXSelector(y="y", candidates=["good", "junk"], alpha=0.01).fit(
        df, entity="id", time="t"
    )
    assert "good" in sel.selected_
    assert sel.ranking_.height == 2
    with pytest.raises(ValueError, match="`k` must be"):
        IVXSelector(y="y", k=0)
    with pytest.raises(ValueError, match="`alpha` must"):
        IVXSelector(y="y", alpha=1.5)
