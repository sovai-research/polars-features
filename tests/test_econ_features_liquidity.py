"""Acceptance tests for the microstructure liquidity block.

Amihud and Roll are checked against hand-computed values on tiny inputs, then
against their known behaviour on a simulated bid-ask-bounce price process where
the true spread is known.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.econ.features import (
    amihud_illiquidity,
    amivest_liquidity,
    liquidity_features,
    roll_spread,
)


def _tiny() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity": ["A", "A", "A", "A"],
            "time": [0, 1, 2, 3],
            "ret": [0.01, -0.02, 0.03, -0.04],
            "dv": [100.0, 200.0, 400.0, 800.0],
        }
    )


def test_amihud_matches_a_hand_computed_reference() -> None:
    out = amihud_illiquidity(
        _tiny(),
        entity="entity",
        time="time",
        returns="ret",
        dollar_volume="dv",
        window=2,
        scale=1.0,
    )
    got = out.get_column("amihud_2").to_numpy()
    assert np.isnan(got[0]) or got[0] is None or out["amihud_2"][0] is None
    # mean(|0.01|/100, |-0.02|/200) = mean(1e-4, 1e-4) = 1e-4
    assert got[1] == pytest.approx(1e-4)
    # mean(1e-4, 0.03/400 = 7.5e-5)
    assert got[2] == pytest.approx((1e-4 + 7.5e-5) / 2)
    # mean(7.5e-5, 0.04/800 = 5e-5)
    assert got[3] == pytest.approx((7.5e-5 + 5e-5) / 2)


def test_amihud_scaling_is_linear() -> None:
    kwargs = {
        "entity": "entity",
        "time": "time",
        "returns": "ret",
        "dollar_volume": "dv",
        "window": 2,
    }
    raw = amihud_illiquidity(_tiny(), scale=1.0, **kwargs).get_column("amihud_2")
    scaled = amihud_illiquidity(_tiny(), scale=1e6, **kwargs).get_column("amihud_2")
    np.testing.assert_allclose(
        scaled.drop_nulls().to_numpy(), raw.drop_nulls().to_numpy() * 1e6
    )


def test_amihud_ignores_non_positive_volume() -> None:
    df = _tiny().with_columns(pl.Series("dv", [100.0, 0.0, 400.0, 800.0]))
    out = amihud_illiquidity(
        df,
        entity="entity",
        time="time",
        returns="ret",
        dollar_volume="dv",
        window=2,
        min_periods=1,
        scale=1.0,
    )
    # The zero-volume row contributes nothing; the window falls back to the
    # single valid observation rather than dividing by zero.
    assert np.isfinite(out.get_column("amihud_2").to_numpy()).all()


def test_amivest_is_the_reciprocal_flavoured_companion() -> None:
    df = _tiny().rename({"dv": "vol"})
    out = amivest_liquidity(
        df, entity="entity", time="time", returns="ret", volume="vol", window=2
    )
    got = out.get_column("amivest_2").to_numpy()
    # mean(100/0.01, 200/0.02) = mean(10000, 10000)
    assert got[1] == pytest.approx(10000.0)


# --------------------------------------------------------------------------- #
# Roll spread
# --------------------------------------------------------------------------- #
def test_roll_spread_recovers_a_known_bid_ask_bounce() -> None:
    """Simulate a random walk observed with a bid-ask bounce of half-spread ``c``.

    Roll's model gives ``Cov(dp_t, dp_{t-1}) = -c**2``, so the estimator
    ``2 * sqrt(-Cov)`` recovers the full spread ``2c``.
    """
    rng = np.random.default_rng(0)
    n = 4000
    c = 0.05  # half-spread
    efficient = np.cumsum(rng.standard_normal(n) * 0.01)
    direction = rng.choice([-1.0, 1.0], size=n)
    price = efficient + c * direction
    df = pl.DataFrame(
        {
            "entity": ["A"] * n,
            "time": np.arange(n, dtype=np.int64),
            "px": price,
        }
    )
    out = roll_spread(df, entity="entity", time="time", price="px", window=1000)
    est = out.get_column("roll_spread_1000").drop_nulls().to_numpy()[-1]
    assert est == pytest.approx(2 * c, rel=0.15)


def test_roll_spread_is_null_when_autocovariance_is_positive() -> None:
    # A pure trend has strongly positive first-order autocovariance in dp.
    n = 60
    df = pl.DataFrame(
        {
            "entity": ["A"] * n,
            "time": np.arange(n, dtype=np.int64),
            "px": np.arange(n, dtype=float) ** 1.5,
        }
    )
    out = roll_spread(df, entity="entity", time="time", price="px", window=20)
    assert out.get_column("roll_spread_20").drop_nulls().len() == 0
    clipped = roll_spread(
        df, entity="entity", time="time", price="px", window=20, clip_positive=True
    )
    vals = clipped.get_column("roll_spread_20").drop_nulls().to_numpy()
    assert vals.size > 0
    np.testing.assert_allclose(vals, 0.0)


def test_roll_spread_requires_exactly_one_price_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        roll_spread(_tiny(), entity="entity", time="time")


def test_roll_spread_never_crosses_entity_boundaries() -> None:
    n = 40
    rng = np.random.default_rng(2)
    df = pl.concat(
        [
            pl.DataFrame(
                {
                    "entity": [ent] * n,
                    "time": np.arange(n, dtype=np.int64),
                    "px": 100.0 * (i + 1) + np.cumsum(rng.standard_normal(n)),
                }
            )
            for i, ent in enumerate(("A", "B"))
        ]
    )
    out = roll_spread(df, entity="entity", time="time", price="px", window=10).sort(
        ["entity", "time"]
    )
    b = out.filter(pl.col("entity") == "B")
    # B's first row has no within-entity predecessor, so its price change (and
    # therefore its window warm-up) must be null, not a 100-point jump from A.
    assert b.get_column("roll_spread_10").head(9).is_null().all()


# --------------------------------------------------------------------------- #
# Combined block
# --------------------------------------------------------------------------- #
def test_liquidity_features_emit_every_available_measure() -> None:
    n = 300
    rng = np.random.default_rng(5)
    df = pl.DataFrame(
        {
            "entity": ["A"] * n,
            "time": np.arange(n, dtype=np.int64),
            "ret": rng.standard_normal(n) * 0.01,
            "px": 100.0 + np.cumsum(rng.standard_normal(n) * 0.1),
            "dv": np.abs(rng.standard_normal(n)) * 1e6 + 1e5,
            "vol": np.abs(rng.standard_normal(n)) * 1e4 + 1e3,
            "shares": np.full(n, 1e7),
        }
    )
    out = liquidity_features(
        df,
        entity="entity",
        time="time",
        returns="ret",
        price="px",
        dollar_volume="dv",
        volume="vol",
        shares_outstanding="shares",
        windows=(21, 63),
    )
    for w in (21, 63):
        for stem in ("amihud", "roll_spread", "amivest", "turnover", "ret_vol"):
            assert f"{stem}_{w}" in out.columns
    assert out.get_column("amihud_21").drop_nulls().len() > 0


def test_liquidity_features_only_emit_what_the_columns_support() -> None:
    n = 60
    rng = np.random.default_rng(6)
    df = pl.DataFrame(
        {
            "entity": ["A"] * n,
            "time": np.arange(n, dtype=np.int64),
            "ret": rng.standard_normal(n) * 0.01,
        }
    )
    out = liquidity_features(
        df, entity="entity", time="time", returns="ret", windows=(21,)
    )
    assert "ret_vol_21" in out.columns
    assert "amihud_21" not in out.columns
    assert "roll_spread_21" not in out.columns
