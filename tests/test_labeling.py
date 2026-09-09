"""Tests for the leak-safe López de Prado labeling module."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from panelary.label import fixed_horizon, meta_label, triple_barrier


def _price_frame(prices, entity="A", start=None):
    """Build a single-entity (entity, time, close) frame from a price list."""
    n = len(prices)
    start = start or dt.date(2020, 1, 1)
    times = [start + dt.timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "entity": [entity] * n,
            "time": times,
            "close": [float(p) for p in prices],
        }
    )


def test_profit_take_hit():
    """A clear up-move after a low-vol run must touch the profit-take barrier."""
    # 20 low-vol rows to seed a small trailing vol, then a sharp jump up.
    base = [100.0 + (i % 2) * 0.05 for i in range(20)]
    prices = base + [101.0, 108.0]  # big up-move at the end
    df = _price_frame(prices)

    out = triple_barrier(
        df,
        price="close",
        pt=2.0,
        sl=1.0,
        max_holding=3,
        vol_lookback=10,
    )

    row = out.filter(pl.col("time") == dt.date(2020, 1, 21))  # index 20, close 101
    assert row.get_column("label").item() == 1
    # touch on the jump to 108 (index 21) -> next day
    assert row.get_column("t1").item() == dt.date(2020, 1, 22)
    assert row.get_column("ret").item() > 0


def test_vertical_barrier_zero_label():
    """Flat path -> neither horizontal barrier hit -> label 0 at vertical."""
    prices = [100.0 + (i % 2) * 0.01 for i in range(30)]
    df = _price_frame(prices)

    out = triple_barrier(
        df, price="close", pt=5.0, sl=5.0, max_holding=3, vol_lookback=10
    )

    row = out.filter(pl.col("time") == dt.date(2020, 1, 15))
    assert row.get_column("label").item() == 0
    # vertical barrier is exactly max_holding steps ahead
    assert row.get_column("t1").item() == dt.date(2020, 1, 18)


def test_t1_bounds():
    """t1 must satisfy t <= t1 <= t + max_holding for every row."""
    rng = np.random.default_rng(0)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.5, size=40))
    df = _price_frame(prices)

    max_holding = 5
    out = triple_barrier(
        df, price="close", pt=2.0, sl=2.0, max_holding=max_holding, vol_lookback=10
    ).with_columns(
        (pl.col("t1") - pl.col("time")).dt.total_days().alias("gap"),
    )

    gaps = out.get_column("gap")
    assert (gaps >= 0).all()
    assert (gaps <= max_holding).all()


def test_t1_dtype_matches_time():
    """t1 must share the time column dtype (purging contract)."""
    df = _price_frame([100.0 + i for i in range(25)])
    out = triple_barrier(df, price="close", max_holding=4, vol_lookback=5)
    assert out.schema["t1"] == out.schema["time"]


def test_point_in_time_no_future_leak():
    """Perturbing a price strictly beyond t1 must not change the row's label."""
    base = [100.0 + (i % 2) * 0.05 for i in range(20)]
    prices = base + [101.0, 108.0, 90.0, 200.0]
    df = _price_frame(prices)

    kwargs = {
        "price": "close",
        "pt": 2.0,
        "sl": 1.0,
        "max_holding": 3,
        "vol_lookback": 10,
    }
    out = triple_barrier(df, **kwargs)

    target_time = dt.date(2020, 1, 21)  # index 20
    row = out.filter(pl.col("time") == target_time)
    label0 = row.get_column("label").item()
    t1 = row.get_column("t1").item()

    # Perturb every price strictly after t1 to an extreme value.
    perturbed = df.with_columns(
        pl.when(pl.col("time") > t1)
        .then(pl.col("close") * 10.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    out2 = triple_barrier(perturbed, **kwargs)
    row2 = out2.filter(pl.col("time") == target_time)

    assert row2.get_column("label").item() == label0
    assert row2.get_column("t1").item() == t1


def test_vol_estimate_is_trailing():
    """The vol estimate must not use future returns: a row's label is unchanged
    when a far-future price (beyond its t1) moves, confirming trailing vol."""
    prices = [100.0] * 15 + [100.0, 101.0, 102.0, 103.0, 104.0]
    df = _price_frame(prices)
    kwargs = {
        "price": "close",
        "pt": 2.0,
        "sl": 2.0,
        "max_holding": 2,
        "vol_lookback": 5,
    }

    out = triple_barrier(df, **kwargs)
    early = out.filter(pl.col("time") == dt.date(2020, 1, 5))
    lbl = early.get_column("label").item()
    t1 = early.get_column("t1").item()

    perturbed = df.with_columns(
        pl.when(pl.col("time") > t1)
        .then(pl.lit(1e6))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    out2 = triple_barrier(perturbed, **kwargs)
    early2 = out2.filter(pl.col("time") == dt.date(2020, 1, 5))
    assert early2.get_column("label").item() == lbl


def test_multi_entity_independence():
    """Barriers are computed per entity (.over(entity)); entities do not mix."""
    a = _price_frame([100.0 + i for i in range(10)], entity="A")
    b = _price_frame([100.0 - i for i in range(10)], entity="B")
    df = pl.concat([a, b])

    out = triple_barrier(
        df, price="close", pt=1.0, sl=1.0, max_holding=3, vol_lookback=3
    )
    # last row of each entity has t1 == its own time (no forward rows)
    last_a = out.filter(
        (pl.col("entity") == "A") & (pl.col("time") == dt.date(2020, 1, 10))
    )
    assert last_a.get_column("t1").item() == dt.date(2020, 1, 10)
    assert out.height == df.height


def test_fixed_horizon_continuous():
    """Continuous fixed-horizon return and t1 = t + horizon steps."""
    prices = [100.0, 110.0, 121.0, 133.1]
    df = _price_frame(prices)
    out = fixed_horizon(df, price="close", horizon=1)

    first = out.filter(pl.col("time") == dt.date(2020, 1, 1))
    assert first.get_column("ret").item() == pytest.approx(0.10)
    assert first.get_column("label").item() == pytest.approx(0.10)
    assert first.get_column("t1").item() == dt.date(2020, 1, 2)
    # tail row lacks a full horizon -> null t1
    last = out.filter(pl.col("time") == dt.date(2020, 1, 4))
    assert last.get_column("t1").item() is None


def test_fixed_horizon_threshold_sign():
    """Threshold produces a ternary sign label with a dead-band."""
    prices = [100.0, 105.0, 105.2, 90.0]
    df = _price_frame(prices)
    out = fixed_horizon(df, price="close", horizon=1, threshold=0.02)

    vals = {
        r["time"]: r["label"] for r in out.select("time", "label").iter_rows(named=True)
    }
    assert vals[dt.date(2020, 1, 1)] == 1  # +5% > 2%
    assert vals[dt.date(2020, 1, 2)] == 0  # ~+0.19% within band
    assert vals[dt.date(2020, 1, 3)] == -1  # big drop
    assert out.schema["label"] == pl.Int64


def test_meta_label():
    """Meta-label is 1 (act) only when side agrees with a non-zero outcome."""
    df = pl.DataFrame(
        {
            "side": [1, 1, -1, -1, 1, 0],
            "label": [1, -1, -1, 1, 0, 1],
        }
    )
    out = meta_label(df, primary_signal="side", label="label")
    assert out.get_column("meta_label").to_list() == [1, 0, 1, 0, 0, 0]
    assert out.schema["meta_label"] == pl.Int64


def test_accepts_lazyframe():
    """LazyFrame input is accepted and materialized."""
    df = _price_frame([100.0 + i for i in range(12)]).lazy()
    out = triple_barrier(df, price="close", max_holding=3, vol_lookback=4)
    assert isinstance(out, pl.DataFrame)
    assert {"label", "ret", "t1"}.issubset(out.columns)
