"""Tests for heterogeneous-panel estimators and panel tests (``econ._panel``).

Acceptance for M5: CCE / MG / PMG must **recover known DGP coefficients in
simulation**, and the unit-root / dependence tests must have the right sign of
behaviour under their nulls and alternatives.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.econ import (
    CrossSectionalAverages,
    PanelSlopeFeatures,
    cce_mg,
    cce_pooled,
    cips,
    ips,
    llc,
    mean_group,
    pesaran_cd,
    pmg,
)


# --------------------------------------------------------------------------- #
# Data generating processes
# --------------------------------------------------------------------------- #
def _factor_panel(
    n_entities: int = 60,
    n_times: int = 90,
    *,
    beta: float = 1.0,
    seed: int = 0,
) -> pl.DataFrame:
    """A CCE-style panel: a common factor drives both ``x`` and ``y``.

    ``y_it = b_i x_it + g_i f_t + e_it`` with ``x_it = 0.6 f_t + v_it`` and
    ``mean(b_i) = beta``. Mean Group is inconsistent here (it ignores ``f``);
    CCE is consistent.
    """
    rng = np.random.default_rng(seed)
    f = rng.normal(size=n_times)
    frames = []
    for i in range(n_entities):
        v = 0.5 * rng.normal(size=n_times)
        x = 0.6 * f + v
        b_i = beta + 0.2 * rng.normal()
        g_i = 0.8 + 0.2 * rng.normal()
        y = b_i * x + g_i * f + 0.3 * rng.normal(size=n_times)
        frames.append(
            pl.DataFrame(
                {
                    "id": np.full(n_times, i),
                    "t": np.arange(n_times),
                    "x": x,
                    "y": y,
                }
            )
        )
    return pl.concat(frames)


def _plain_panel(
    n_entities: int = 40, n_times: int = 60, *, beta: float = 2.0, seed: int = 0
) -> pl.DataFrame:
    """A heterogeneous-slope panel with **no** common factor: MG is consistent."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_entities):
        x = rng.normal(size=n_times)
        b_i = beta + 0.3 * rng.normal()
        y = 1.0 + b_i * x + 0.4 * rng.normal(size=n_times)
        frames.append(
            pl.DataFrame(
                {"id": np.full(n_times, i), "t": np.arange(n_times), "x": x, "y": y}
            )
        )
    return pl.concat(frames)


def _ardl_panel(
    n_entities: int = 25,
    n_times: int = 90,
    *,
    theta: float = 2.0,
    seed: int = 0,
) -> pl.DataFrame:
    """A PMG data generating process with a **common long-run** coefficient.

    ``dy_it = phi_i (y_{i,t-1} - theta x_{i,t-1}) + delta_i dx_it + mu_i + e_it``.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_entities):
        phi = -rng.uniform(0.2, 0.6)
        delta = rng.normal(0.5, 0.1)
        mu = 0.2 * rng.normal()
        x = np.cumsum(0.5 * rng.normal(size=n_times))
        y = np.zeros(n_times)
        y[0] = theta * x[0]
        for t in range(1, n_times):
            y[t] = (
                y[t - 1]
                + phi * (y[t - 1] - theta * x[t - 1])
                + delta * (x[t] - x[t - 1])
                + mu
                + 0.2 * rng.normal()
            )
        frames.append(
            pl.DataFrame(
                {"id": np.full(n_times, i), "t": np.arange(n_times), "x": x, "y": y}
            )
        )
    return pl.concat(frames)


def _unit_root_panel(
    n_entities: int = 15, n_times: int = 60, *, rho: float = 1.0, seed: int = 0
) -> pl.DataFrame:
    """Independent AR(1) series; ``rho=1`` is the unit-root null."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_entities):
        e = rng.normal(size=n_times)
        y = np.zeros(n_times)
        for t in range(1, n_times):
            y[t] = rho * y[t - 1] + e[t]
        frames.append(
            pl.DataFrame({"id": np.full(n_times, i), "t": np.arange(n_times), "y": y})
        )
    return pl.concat(frames)


# --------------------------------------------------------------------------- #
# Mean Group
# --------------------------------------------------------------------------- #
def test_mean_group_recovers_the_average_slope():
    df = _plain_panel(beta=2.0, seed=1)
    res = mean_group(df, y="y", x=["x"], entity="id", time="t")
    assert res.method == "MG"
    assert abs(res.params[0] - 2.0) < 0.1
    # The true slope is inside the confidence interval.
    ci = res.conf_int()
    assert ci["lower"][0] < 2.0 < ci["upper"][0]
    assert res.n_entities == 40
    assert res.per_entity.height == 40
    assert "beta_x" in res.per_entity.columns


def test_mean_group_variance_reflects_slope_dispersion():
    """The MG standard error is the dispersion of the per-entity slopes."""
    df = _plain_panel(n_entities=50, seed=2)
    res = mean_group(df, y="y", x=["x"], entity="id", time="t")
    betas = res.per_entity["beta_x"].to_numpy()
    expected = np.std(betas, ddof=1) / np.sqrt(betas.size)
    np.testing.assert_allclose(res.std_errors[0], expected, rtol=1e-10)


def test_mean_group_is_biased_when_a_common_factor_is_ignored():
    df = _factor_panel(beta=1.0, seed=3)
    mg = mean_group(df, y="y", x=["x"], entity="id", time="t")
    cce = cce_mg(df, y="y", x=["x"], entity="id", time="t")
    assert abs(mg.params[0] - 1.0) > 0.3  # MG is badly biased
    assert abs(cce.params[0] - 1.0) < 0.1  # CCE is not


# --------------------------------------------------------------------------- #
# CCE
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_cce_mg_recovers_the_dgp_coefficient(beta):
    df = _factor_panel(beta=beta, seed=int(10 * beta))
    res = cce_mg(df, y="y", x=["x"], entity="id", time="t")
    assert res.method == "CCEMG"
    assert abs(res.params[0] - beta) < 0.12
    assert res.extra["csa_columns"] == ["y_csa", "x_csa"]


def test_cce_pooled_recovers_the_dgp_coefficient():
    df = _factor_panel(beta=1.0, seed=7)
    res = cce_pooled(df, y="y", x=["x"], entity="id", time="t")
    assert res.method == "CCEP"
    assert abs(res.params[0] - 1.0) < 0.12
    assert np.isfinite(res.std_errors[0]) and res.std_errors[0] > 0


def test_cce_converges_as_the_panel_grows():
    small = cce_mg(_factor_panel(30, 50, seed=5), y="y", x=["x"], entity="id", time="t")
    large = cce_mg(
        _factor_panel(120, 150, seed=5), y="y", x=["x"], entity="id", time="t"
    )
    assert abs(large.params[0] - 1.0) <= abs(small.params[0] - 1.0) + 0.05
    assert large.std_errors[0] < small.std_errors[0]


# --------------------------------------------------------------------------- #
# PMG
# --------------------------------------------------------------------------- #
def test_pmg_recovers_the_common_long_run_coefficient():
    df = _ardl_panel(theta=2.0, seed=4)
    res = pmg(df, y="y", x=["x"], entity="id", time="t")
    assert res.method == "PMG"
    assert abs(res.params[0] - 2.0) < 0.1
    assert res.extra["converged"]
    # The error-correction speeds must be negative (stable adjustment).
    speeds = res.per_entity["ec_speed"].to_numpy()
    assert (speeds < 0).all()
    assert -1.0 < res.extra["ec_speed"] < 0.0


def test_pmg_emits_entity_level_features():
    df = _ardl_panel(theta=1.5, seed=9)
    res = pmg(df, y="y", x=["x"], entity="id", time="t")
    assert set(res.per_entity.columns) == {"entity", "ec_speed", "intercept", "d_x"}
    assert res.per_entity.height == res.n_entities


def test_pmg_raises_when_no_entity_has_enough_history():
    df = _ardl_panel(n_entities=3, n_times=5, seed=1)
    with pytest.raises(ValueError, match="enough observations"):
        pmg(df, y="y", x=["x"], entity="id", time="t", min_obs=50)


# --------------------------------------------------------------------------- #
# Panel unit-root tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("test_fn", [llc, ips, cips])
def test_panel_unit_root_tests_do_not_reject_under_the_null(test_fn):
    df = _unit_root_panel(rho=1.0, seed=31)
    res = test_fn(df, value="y", entity="id", time="t", reps=200, seed=1)
    assert res.p_value > 0.10
    assert np.isfinite(res.statistic)


@pytest.mark.parametrize("test_fn", [llc, ips, cips])
def test_panel_unit_root_tests_reject_for_stationary_panels(test_fn):
    df = _unit_root_panel(rho=0.5, seed=31)
    res = test_fn(df, value="y", entity="id", time="t", reps=200, seed=1)
    assert res.p_value < 0.01


def test_ips_and_cips_expose_per_entity_statistics():
    df = _unit_root_panel(rho=0.6, seed=12)
    res_ips = ips(df, value="y", entity="id", time="t", reps=100, seed=2)
    res_cips = cips(df, value="y", entity="id", time="t", reps=100, seed=2)
    assert res_ips.extra["per_entity"].columns == ["entity", "adf_t"]
    assert res_cips.extra["per_entity"].columns == ["entity", "cadf_t"]
    # t-bar is by construction the mean of the individual statistics.
    np.testing.assert_allclose(
        res_ips.statistic, res_ips.extra["per_entity"]["adf_t"].mean(), rtol=1e-12
    )


def test_null_moment_simulation_is_deterministic():
    df = _unit_root_panel(seed=3)
    a = ips(df, value="y", entity="id", time="t", reps=150, seed=99)
    b = ips(df, value="y", entity="id", time="t", reps=150, seed=99)
    assert a.z_statistic == b.z_statistic


# --------------------------------------------------------------------------- #
# Pesaran CD
# --------------------------------------------------------------------------- #
def test_pesaran_cd_does_not_reject_for_independent_series():
    rng = np.random.default_rng(0)
    n_entities, n_times = 20, 80
    frames = [
        pl.DataFrame(
            {
                "id": np.full(n_times, i),
                "t": np.arange(n_times),
                "y": rng.normal(size=n_times),
            }
        )
        for i in range(n_entities)
    ]
    res = pesaran_cd(pl.concat(frames), value="y", entity="id", time="t")
    assert res.p_value > 0.05
    assert abs(res.statistic) < 2.0


def test_pesaran_cd_rejects_under_a_common_factor():
    df = _factor_panel(30, 80, seed=2)
    res = pesaran_cd(df, value="y", entity="id", time="t")
    assert res.p_value < 1e-6
    assert res.extra["mean_abs_corr"] > 0.3


def test_pesaran_cd_needs_two_entities():
    df = pl.DataFrame({"id": [0, 0, 0], "t": [0, 1, 2], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="at least two entities"):
        pesaran_cd(df, value="y", entity="id", time="t")


# --------------------------------------------------------------------------- #
# Feature transformers
# --------------------------------------------------------------------------- #
def test_panel_slope_features_freeze_train_coefficients():
    df = _plain_panel(n_entities=20, n_times=60, seed=6)
    train = df.filter(pl.col("t") < 30)
    tr = PanelSlopeFeatures(y="y", x=["x"], method="mg").fit(
        train, entity="id", time="t"
    )
    frozen = tr.coef_table_.clone()
    out_train = tr.transform(train).collect()
    out_full = tr.transform(df).collect()
    assert "mg_beta_x" in out_train.columns
    # Transforming a larger frame must not change the fitted coefficients.
    assert tr.coef_table_.equals(frozen)
    joined = out_full.filter(pl.col("t") < 30).sort("id", "t")["mg_beta_x"].to_numpy()
    np.testing.assert_allclose(
        joined, out_train.sort("id", "t")["mg_beta_x"].to_numpy(), atol=1e-12
    )


def test_panel_slope_features_fills_unseen_entities():
    df = _plain_panel(n_entities=10, n_times=40, seed=7)
    train = df.filter(pl.col("id") < 6)
    tr = PanelSlopeFeatures(y="y", x=["x"], method="mg", unseen="mean").fit(
        train, entity="id", time="t"
    )
    out = tr.transform(df).collect()
    assert out["mg_beta_x"].null_count() == 0

    tr_null = PanelSlopeFeatures(y="y", x=["x"], method="mg", unseen="null").fit(
        train, entity="id", time="t"
    )
    out_null = tr_null.transform(df).collect().filter(pl.col("id") >= 6)
    assert out_null["mg_beta_x"].null_count() == out_null.height


def test_panel_slope_features_validates_method():
    with pytest.raises(ValueError, match="unknown `method"):
        PanelSlopeFeatures(y="y", x=["x"], method="ols")


def test_cross_sectional_averages_are_contemporaneous():
    df = _factor_panel(10, 20, seed=8)
    tr = CrossSectionalAverages(["x", "y"], demean=True).fit(df, entity="id", time="t")
    out = tr.transform(df).collect().sort("id", "t")
    assert tr.output_names == ["x_csa", "y_csa", "x_csa_dev", "y_csa_dev"]
    # The proxy on date t is exactly the cross-sectional mean on date t.
    expected = df.group_by("t").agg(pl.col("x").mean().alias("m")).sort("t")
    got = out.group_by("t").agg(pl.col("x_csa").first().alias("m")).sort("t")
    np.testing.assert_allclose(
        got["m"].to_numpy(), expected["m"].to_numpy(), atol=1e-12
    )
    # Truncating the future must not change earlier dates' proxies.
    head = tr.transform(df.filter(pl.col("t") < 10)).collect().sort("id", "t")
    np.testing.assert_allclose(
        head["x_csa"].to_numpy(),
        out.filter(pl.col("t") < 10)["x_csa"].to_numpy(),
        atol=1e-12,
    )


def test_transformers_declare_the_leak_safety_contract():
    for cls in (PanelSlopeFeatures, CrossSectionalAverages):
        assert cls.panel_safe is True
        assert cls.leakage_safe is True
