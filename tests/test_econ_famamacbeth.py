"""Tests for the Fama-MacBeth two-pass regression (``econ._famamacbeth``).

Two things matter here: (1) the estimator recovers a known risk premium and its
Newey-West standard errors are right, and (2) winsorising / standardising happen
**per date** -- the plan's leak-safety rule 7.3. A global z-score would make an
early row's feature depend on the whole sample's mean, which the invariance test
below would catch.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.econ import (
    FamaMacBethTransformer,
    fama_macbeth,
)
from panelary.econ._common import newey_west_scalar


def _cross_section_panel(
    n_entities: int = 60,
    n_times: int = 120,
    *,
    lam: float = 0.02,
    seed: int = 0,
) -> pl.DataFrame:
    """``r_it = lambda * beta_i + noise``: a known cross-sectional premium."""
    rng = np.random.default_rng(seed)
    beta = rng.normal(1.0, 0.4, size=n_entities)
    size = rng.normal(0.0, 1.0, size=n_entities)
    frames = []
    for t in range(n_times):
        ret = lam * beta + 0.01 * rng.normal(size=n_entities)
        frames.append(
            pl.DataFrame(
                {
                    "id": np.arange(n_entities),
                    "date": np.full(n_entities, t),
                    "beta": beta + 0.05 * rng.normal(size=n_entities),
                    "size": size,
                    "ret": ret,
                }
            )
        )
    return pl.concat(frames)


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #
def test_recovers_a_known_risk_premium():
    df = _cross_section_panel(lam=0.02, seed=1)
    res = fama_macbeth(df, y="ret", x=["beta", "size"], entity="id", time="date")
    assert res.names == ["const", "beta", "size"]
    idx = res.names.index("beta")
    assert abs(res.params[idx] - 0.02) < 0.004
    ci = res.conf_int()
    assert ci["lower"][idx] < 0.02 < ci["upper"][idx]
    # `size` carries no premium in this DGP.
    assert res.pvalues[res.names.index("size")] > 0.05


def test_lambda_series_has_one_row_per_date():
    df = _cross_section_panel(n_times=50, seed=2)
    res = fama_macbeth(df, y="ret", x=["beta"], entity="id", time="date")
    assert res.n_periods == 50
    assert res.lambdas.height == 50
    assert res.lambdas.columns == ["date", "const", "beta", "n_obs", "r2"]
    # The reported coefficient is exactly the mean of the lambda series.
    np.testing.assert_allclose(res.params[1], res.lambdas["beta"].mean(), rtol=1e-12)


def test_standard_error_is_the_newey_west_se_of_the_lambda_mean():
    df = _cross_section_panel(n_times=80, seed=3)
    res = fama_macbeth(df, y="ret", x=["beta"], entity="id", time="date", lags=4)
    expected = np.sqrt(newey_west_scalar(res.lambdas["beta"].to_numpy(), 4))
    np.testing.assert_allclose(res.std_errors[1], expected, rtol=1e-12)
    assert res.lags == 4


def test_zero_lags_reproduces_the_plain_standard_error():
    df = _cross_section_panel(n_times=60, seed=4)
    res = fama_macbeth(df, y="ret", x=["beta"], entity="id", time="date", lags=0)
    lam = res.lambdas["beta"].to_numpy()
    plain = np.std(lam, ddof=0) / np.sqrt(lam.size)
    np.testing.assert_allclose(res.std_errors[1], plain, rtol=1e-10)


def test_no_intercept_option():
    df = _cross_section_panel(n_times=40, seed=5)
    res = fama_macbeth(
        df, y="ret", x=["beta"], entity="id", time="date", intercept=False
    )
    assert res.names == ["beta"]
    assert res.lambdas.columns == ["date", "beta", "n_obs", "r2"]


def test_summary_and_lambda_features():
    df = _cross_section_panel(n_times=40, seed=6)
    res = fama_macbeth(df, y="ret", x=["beta"], entity="id", time="date")
    assert res.summary().columns == [
        "term",
        "estimate",
        "std_error",
        "t_stat",
        "p_value",
    ]
    feats = res.lambda_features(prefix="fm_")
    assert "fm_beta" in feats.columns
    assert feats.columns[0] == "date"


# --------------------------------------------------------------------------- #
# Per-date, never global (leak safety)
# --------------------------------------------------------------------------- #
def test_standardization_is_per_date_not_global():
    """Doubling one date's characteristics must not move any other date's lambda."""
    df = _cross_section_panel(n_times=30, seed=7)
    base = fama_macbeth(
        df, y="ret", x=["beta"], entity="id", time="date", standardize=True
    )
    perturbed_df = df.with_columns(
        pl.when(pl.col("date") == 29)
        .then(pl.col("beta") * 100.0 + 50.0)
        .otherwise(pl.col("beta"))
        .alias("beta")
    )
    perturbed = fama_macbeth(
        perturbed_df, y="ret", x=["beta"], entity="id", time="date", standardize=True
    )
    a = base.lambdas.filter(pl.col("date") < 29)["beta"].to_numpy()
    b = perturbed.lambdas.filter(pl.col("date") < 29)["beta"].to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-12)
    # A per-date z-score is scale/location invariant, so even date 29 is unchanged.
    np.testing.assert_allclose(
        base.lambdas["beta"].to_numpy()[-1],
        perturbed.lambdas["beta"].to_numpy()[-1],
        rtol=1e-8,
    )


def test_winsorizing_is_per_date():
    df = _cross_section_panel(n_entities=100, n_times=20, seed=8)
    outlier = df.with_columns(
        pl.when((pl.col("date") == 19) & (pl.col("id") == 0))
        .then(pl.lit(1e6))
        .otherwise(pl.col("beta"))
        .alias("beta")
    )
    base = fama_macbeth(
        df, y="ret", x=["beta"], entity="id", time="date", winsorize_limit=0.05
    )
    perturbed = fama_macbeth(
        outlier, y="ret", x=["beta"], entity="id", time="date", winsorize_limit=0.05
    )
    a = base.lambdas.filter(pl.col("date") < 19)["beta"].to_numpy()
    b = perturbed.lambdas.filter(pl.col("date") < 19)["beta"].to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_dates_with_too_small_a_cross_section_are_skipped():
    df = _cross_section_panel(n_entities=8, n_times=10, seed=9)
    thin = df.filter(~((pl.col("date") == 5) & (pl.col("id") > 1)))
    res = fama_macbeth(thin, y="ret", x=["beta"], entity="id", time="date")
    assert 5 not in res.lambdas["date"].to_list()
    assert res.n_periods == 9


def test_raises_when_no_date_is_usable():
    df = _cross_section_panel(n_entities=2, n_times=5, seed=10)
    with pytest.raises(ValueError, match="no date produced"):
        fama_macbeth(
            df, y="ret", x=["beta"], entity="id", time="date", min_cross_section=50
        )


def test_requires_at_least_one_characteristic():
    df = _cross_section_panel(n_times=5, seed=11)
    with pytest.raises(ValueError, match="at least one characteristic"):
        fama_macbeth(df, y="ret", x=[], entity="id", time="date")


# --------------------------------------------------------------------------- #
# Transformer
# --------------------------------------------------------------------------- #
def test_transformer_freezes_train_premia():
    df = _cross_section_panel(n_times=60, seed=12)
    train = df.filter(pl.col("date") < 30)
    tr = FamaMacBethTransformer(y="ret", x=["beta"]).fit(
        train, entity="id", time="date"
    )
    frozen = np.array(tr.params_, copy=True)
    out = tr.transform(df).collect()
    assert tr.output_names == ["fm_pred", "fm_resid"]
    assert {"fm_pred", "fm_resid"}.issubset(out.columns)
    np.testing.assert_allclose(tr.params_, frozen)
    # Prediction is exactly `const + beta * lambda_beta` with the frozen lambdas.
    expected = frozen[0] + frozen[1] * out["beta"].to_numpy()
    np.testing.assert_allclose(out["fm_pred"].to_numpy(), expected, atol=1e-12)


def test_transformer_output_is_invariant_to_future_rows():
    df = _cross_section_panel(n_times=40, seed=13)
    train = df.filter(pl.col("date") < 20)
    tr = FamaMacBethTransformer(y="ret", x=["beta", "size"]).fit(
        train, entity="id", time="date"
    )
    on_train = tr.transform(train).collect().sort("id", "date")["fm_resid"].to_numpy()
    on_full = (
        tr.transform(df)
        .collect()
        .filter(pl.col("date") < 20)
        .sort("id", "date")["fm_resid"]
        .to_numpy()
    )
    np.testing.assert_allclose(on_train, on_full, atol=1e-12)


def test_transformer_omits_the_residual_when_the_target_is_absent():
    df = _cross_section_panel(n_times=20, seed=14)
    tr = FamaMacBethTransformer(y="ret", x=["beta"]).fit(df, entity="id", time="date")
    out = tr.transform(df.drop("ret")).collect()
    assert "fm_pred" in out.columns
    assert "fm_resid" not in out.columns


def test_transformer_declares_the_contract():
    assert FamaMacBethTransformer.panel_safe is True
    assert FamaMacBethTransformer.leakage_safe is True
    tr = FamaMacBethTransformer(y="ret", x=["beta"])
    with pytest.raises(RuntimeError, match="not fitted"):
        tr.transform(_cross_section_panel(n_times=3), entity="id", time="date")
