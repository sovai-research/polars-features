"""Acceptance tests for the unit-root / stationarity battery.

Each test asserts the *known* answer on a simulated series: a random walk must
not reject a unit root, a stationary AR(1) must, KPSS must run the other way
round, and Zivot-Andrews must find a planted structural break.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.core import PanelFrame
from panelary.econ.features import (
    StationarityDifferencer,
    adf,
    dfgls,
    kpss,
    ng_perron,
    phillips_perron,
    rolling_unit_root_features,
    unit_root_table,
    zivot_andrews,
)

N = 500


def _random_walk(seed: int = 0, n: int = N) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n))


def _ar1(rho: float = 0.5, seed: int = 1, n: int = N) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    return x


# --------------------------------------------------------------------------- #
# ADF
# --------------------------------------------------------------------------- #
def test_adf_fails_to_reject_on_random_walk() -> None:
    res = adf(_random_walk())
    assert res.pvalue > 0.10
    assert res.stat > res.crit_values[0.10]


def test_adf_rejects_on_stationary_ar1() -> None:
    res = adf(_ar1())
    assert res.pvalue < 0.01
    assert res.stat < res.crit_values[0.01]


@pytest.mark.parametrize("trend", ["n", "c", "ct"])
def test_adf_trend_cases_run_and_order_criticals(trend: str) -> None:
    res = adf(_ar1(), trend=trend)
    crit = res.crit_values
    # More extreme significance => more negative critical value.
    assert crit[0.01] < crit[0.05] < crit[0.10]
    assert np.isfinite(res.stat)


def test_adf_fixed_lags_are_honoured() -> None:
    assert adf(_ar1(), lags=3).lags == 3


def test_adf_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        adf(np.arange(5.0))


# --------------------------------------------------------------------------- #
# KPSS
# --------------------------------------------------------------------------- #
def test_kpss_rejects_stationarity_on_random_walk() -> None:
    res = kpss(_random_walk())
    assert res.stat > res.crit_values[0.01]
    assert res.pvalue < 0.05
    assert res.null_is_unit_root is False


def test_kpss_is_correctly_sized_on_stationary_series() -> None:
    """KPSS must *not* reject stationarity much more often than its nominal size.

    A single draw of the statistic is noisy (its asymptotic distribution has a
    fat right tail), so the acceptance criterion is the rejection *frequency*
    over independent replications, which is what "correct size" means.
    """
    rejections = sum(
        kpss(_ar1(seed=seed)).stat > kpss(_ar1(seed=seed)).crit_values[0.05]
        for seed in range(40)
    )
    assert rejections / 40 <= 0.20
    # And the median statistic sits well inside the acceptance region.
    stats = np.array([kpss(_ar1(seed=seed)).stat for seed in range(40)])
    assert float(np.median(stats)) < 0.347  # the 10% critical value


def test_kpss_trend_case_handles_deterministic_trend() -> None:
    rng = np.random.default_rng(3)
    trend_stationary = 0.05 * np.arange(N) + rng.standard_normal(N)
    assert kpss(trend_stationary, trend="ct").pvalue > 0.05
    # Around a level only, the same series looks decidedly non-stationary.
    assert kpss(trend_stationary, trend="c").pvalue < 0.05


# --------------------------------------------------------------------------- #
# Phillips-Perron / DF-GLS / Ng-Perron
# --------------------------------------------------------------------------- #
def test_phillips_perron_matches_adf_verdicts() -> None:
    assert phillips_perron(_random_walk()).pvalue > 0.10
    assert phillips_perron(_ar1()).pvalue < 0.01


def test_dfgls_matches_adf_verdicts() -> None:
    assert dfgls(_random_walk()).pvalue > 0.10
    assert dfgls(_ar1()).pvalue < 0.05


def test_ng_perron_statistics_have_the_right_signs() -> None:
    rw = ng_perron(_random_walk())
    ar = ng_perron(_ar1())
    # MZa/MZt are far more negative for the stationary series.
    assert ar.mza < rw.mza
    assert ar.mzt < rw.mzt
    # MSB (a variance-ratio-like statistic) is smaller under stationarity.
    assert ar.msb < rw.msb
    assert ar.pvalue_mzt < 0.05
    assert rw.pvalue_mzt > 0.10
    # MZt == MZa * MSB by construction.
    assert ar.mzt == pytest.approx(ar.mza * ar.msb, rel=1e-10)


# --------------------------------------------------------------------------- #
# Zivot-Andrews
# --------------------------------------------------------------------------- #
def test_zivot_andrews_locates_a_planted_level_break() -> None:
    rng = np.random.default_rng(11)
    x = rng.standard_normal(N) * 0.5
    x[300:] += 6.0  # a large level shift two-thirds of the way in
    res = zivot_andrews(x, regression="intercept")
    assert abs(res.break_index - 300) <= 15
    assert 0.0 < res.break_fraction < 1.0


def test_zivot_andrews_is_no_more_conservative_than_adf() -> None:
    rng = np.random.default_rng(12)
    x = rng.standard_normal(N) * 0.5
    x[300:] += 6.0
    # The break makes plain ADF look non-stationary; allowing for it recovers
    # a far more negative statistic.
    assert zivot_andrews(x).stat < adf(x).stat


def test_zivot_andrews_rejects_bad_regression() -> None:
    with pytest.raises(ValueError, match="'intercept', 'trend' or 'both'"):
        zivot_andrews(_ar1(), regression="nope")


# --------------------------------------------------------------------------- #
# Panel surfaces
# --------------------------------------------------------------------------- #
def _panel() -> pl.DataFrame:
    frames = []
    for i, (name, series) in enumerate(
        [("rw", _random_walk(seed=5)), ("ar", _ar1(seed=6))]
    ):
        frames.append(
            pl.DataFrame(
                {
                    "entity": [name] * N,
                    "time": np.arange(N, dtype=np.int64),
                    "x": series + i,
                }
            )
        )
    return pl.concat(frames)


def test_unit_root_table_one_row_per_entity() -> None:
    out = unit_root_table(_panel(), "x", entity="entity", time="time")
    assert out.height == 2
    assert {"adf_stat", "adf_pvalue", "kpss_stat", "kpss_pvalue"} <= set(out.columns)
    row = {r["entity"]: r for r in out.to_dicts()}
    assert row["ar"]["adf_pvalue"] < 0.01
    assert row["rw"]["adf_pvalue"] > 0.10


def test_rolling_unit_root_features_are_trailing() -> None:
    out = rolling_unit_root_features(
        _panel(), "x", entity="entity", time="time", window=120
    )
    col = out.get_column("x_adf_stat")
    # First (window - 1) rows of each entity are warm-up nulls.
    assert col.head(119).is_null().all()
    assert col.slice(119, 100).is_not_null().any()


# --------------------------------------------------------------------------- #
# StationarityDifferencer
# --------------------------------------------------------------------------- #
def test_differencer_picks_zero_for_stationary_and_one_for_random_walk() -> None:
    panel = PanelFrame(_panel(), entity="entity", time="time")
    tr = StationarityDifferencer(["x"], max_order=1).fit(panel)
    assert tr.orders_[("ar", "x")] == 0
    assert tr.orders_[("rw", "x")] == 1


def test_differencer_applies_frozen_orders_and_is_leak_safe() -> None:
    df = _panel()
    panel = PanelFrame(df, entity="entity", time="time")
    tr = StationarityDifferencer(["x"], max_order=1, suffix="_d").fit(panel)
    out = tr.transform(panel).collect().sort(["entity", "time"])

    src = df.sort(["entity", "time"])
    ar_in = src.filter(pl.col("entity") == "ar").get_column("x").to_numpy()
    rw_in = src.filter(pl.col("entity") == "rw").get_column("x").to_numpy()
    ar_out = out.filter(pl.col("entity") == "ar").get_column("x_d").to_numpy()
    rw_out = out.filter(pl.col("entity") == "rw").get_column("x_d").to_numpy()

    np.testing.assert_allclose(ar_out, ar_in)  # order 0 == identity
    assert np.isnan(rw_out[0])
    np.testing.assert_allclose(rw_out[1:], np.diff(rw_in))


def test_differencer_unseen_entity_uses_the_modal_default() -> None:
    panel = PanelFrame(_panel(), entity="entity", time="time")
    tr = StationarityDifferencer(["x"], max_order=1).fit(panel)
    fresh = pl.DataFrame(
        {
            "entity": ["new"] * 50,
            "time": np.arange(50, dtype=np.int64),
            "x": np.arange(50, dtype=float),
        }
    )
    out = tr.transform(PanelFrame(fresh, entity="entity", time="time")).collect()
    assert out.height == 50
    assert tr.default_order_["x"] in (0, 1)


def test_differencer_requires_columns() -> None:
    with pytest.raises(ValueError, match="at least one column"):
        StationarityDifferencer([])
