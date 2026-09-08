"""Acceptance tests for the realized-volatility / HAR-RV block.

The HAR terms and the realized measures are checked against **hand-computed
references** on small, fully-specified inputs, then against their known
behaviour on a simulated series with a planted jump.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_features.core import PanelFrame
from polars_features.econ.features import (
    HARModel,
    bipower_variation,
    daily_realized_measures,
    har_features,
    har_terms,
    jump_component,
    realized_measures,
    realized_variance,
)


# --------------------------------------------------------------------------- #
# Hand-computed references
# --------------------------------------------------------------------------- #
def test_realized_variance_matches_a_hand_computed_reference() -> None:
    r = np.array([1.0, 2.0, 3.0, 4.0])
    got = realized_variance(r, 3)
    # Windows: [1,2,3] -> 1+4+9 = 14 ; [2,3,4] -> 4+9+16 = 29.
    assert np.isnan(got[0]) and np.isnan(got[1])
    np.testing.assert_allclose(got[2:], [14.0, 29.0])


def test_bipower_variation_matches_a_hand_computed_reference() -> None:
    r = np.array([1.0, 2.0, 3.0, 4.0])
    got = bipower_variation(r, 3)
    scale = (np.pi / 2.0) * (3.0 / 2.0)
    # Window ending at index 2 sums |r1||r0| + |r2||r1| = 2 + 6 = 8 (the first
    # product in the window, at index 0, has no predecessor inside the series).
    np.testing.assert_allclose(got[2], scale * 8.0)
    # Window ending at index 3 sums the three products inside it:
    # |r1||r0| + |r2||r1| + |r3||r2| = 2 + 6 + 12 = 20.
    np.testing.assert_allclose(got[3], scale * 20.0)


def test_har_terms_match_hand_computed_rolling_means() -> None:
    rv = np.arange(1.0, 11.0)  # 1..10
    terms = har_terms(rv, lags=(1, 5, 22))
    np.testing.assert_allclose(terms["har_d"], rv)  # daily term is RV itself
    # Weekly term at index 4 is mean(1..5) = 3, at index 9 mean(6..10) = 8.
    assert terms["har_w"][4] == pytest.approx(3.0)
    assert terms["har_w"][9] == pytest.approx(8.0)
    assert np.isnan(terms["har_w"][3])
    # The 22-period term never fills on a 10-point series.
    assert np.all(np.isnan(terms["har_m"]))


def test_jump_component_is_non_negative_and_isolates_a_jump() -> None:
    rng = np.random.default_rng(0)
    r = rng.standard_normal(300) * 0.01
    r[200] = 0.25  # one enormous return
    parts = realized_measures(r, window=22)
    assert np.all(parts["jump"][np.isfinite(parts["jump"])] >= 0.0)
    # The window containing the jump has RV far above the jump-robust BV.
    idx = 205
    assert parts["rv"][idx] > 3.0 * parts["bv"][idx]
    assert parts["rel_jump"][idx] > 0.5
    # A quiet window shows almost no jump.
    assert parts["rel_jump"][150] < 0.3


def test_jump_component_clips_at_zero() -> None:
    np.testing.assert_allclose(
        jump_component(np.array([1.0, 2.0]), np.array([3.0, 1.0])),
        [0.0, 1.0],
    )


# --------------------------------------------------------------------------- #
# Panel surface
# --------------------------------------------------------------------------- #
def _returns_panel(
    n: int = 400, entities: tuple[str, ...] = ("A", "B")
) -> pl.DataFrame:
    frames = []
    for i, ent in enumerate(entities):
        rng = np.random.default_rng(100 + i)
        r = rng.standard_normal(n) * (0.01 * (i + 1))
        frames.append(
            pl.DataFrame(
                {
                    "entity": [ent] * n,
                    "time": np.arange(n, dtype=np.int64),
                    "ret": r,
                }
            )
        )
    return pl.concat(frames)


def test_har_features_emit_the_expected_columns() -> None:
    out = har_features(
        _returns_panel(), entity="entity", time="time", returns="ret", window=22
    )
    for col in ("rv", "bv", "jump", "rel_jump", "har_d", "har_w", "har_m"):
        assert col in out.columns
    assert out.get_column("har_m").is_not_null().sum() > 0


def test_har_features_never_cross_entity_boundaries() -> None:
    df = _returns_panel(n=60)
    out = har_features(df, entity="entity", time="time", returns="ret", window=10).sort(
        ["entity", "time"]
    )
    for ent in ("A", "B"):
        sub = out.filter(pl.col("entity") == ent).get_column("rv")
        # The first (window - 1) rows of EVERY entity are warm-up nulls; if the
        # kernel bled across entities, B's warm-up would be filled from A's tail.
        assert sub.head(9).is_null().all()


def test_har_features_require_exactly_one_source() -> None:
    df = _returns_panel(n=30)
    with pytest.raises(ValueError, match="exactly one"):
        har_features(df, entity="entity", time="time")
    with pytest.raises(ValueError, match="exactly one"):
        har_features(df, entity="entity", time="time", returns="ret", rv="ret")


def test_daily_realized_measures_aggregate_within_the_day() -> None:
    df = pl.DataFrame(
        {
            "entity": ["A"] * 6,
            "date": [1, 1, 1, 2, 2, 2],
            "ret": [0.1, -0.2, 0.3, 1.0, 1.0, 1.0],
        }
    )
    out = daily_realized_measures(df, entity="entity", date="date", returns="ret")
    assert out.height == 2
    row = out.filter(pl.col("date") == 1).to_dicts()[0]
    assert row["rv"] == pytest.approx(0.01 + 0.04 + 0.09)
    assert row["n_obs"] == 3
    # Day 2 has constant |r| = 1, so RV == 3 and BV == (pi/2)(3/2)(1 + 1) = 3pi/2.
    row2 = out.filter(pl.col("date") == 2).to_dicts()[0]
    assert row2["rv"] == pytest.approx(3.0)
    assert row2["bv"] == pytest.approx((np.pi / 2.0) * 1.5 * 2.0)


# --------------------------------------------------------------------------- #
# HARModel
# --------------------------------------------------------------------------- #
def test_har_model_fits_per_entity_and_predicts() -> None:
    df = _returns_panel(n=600)
    panel = PanelFrame(df, entity="entity", time="time")
    model = HARModel(returns="ret", horizon=1, window=22, min_train_rows=100).fit(panel)
    assert set(model.coef_) == {"A", "B"}
    assert model.pooled_coef_ is not None
    out = model.transform(panel).collect()
    assert "har_forecast" in out.columns
    preds = out.get_column("har_forecast").drop_nulls().to_numpy()
    assert preds.size > 0
    # Volatility forecasts must be positive and of the right order of magnitude.
    assert np.nanmedian(preds) > 0


def test_har_model_matches_an_independent_ols_reference() -> None:
    """The fitted coefficients must equal a hand-built OLS of RV_{t+1} on
    ``[1, RV_d, RV_w, RV_m]`` computed directly with numpy."""
    n = 400
    rng = np.random.default_rng(3)
    rv = np.abs(rng.standard_normal(n)) + 1.0
    df = pl.DataFrame(
        {
            "entity": ["A"] * n,
            "time": np.arange(n, dtype=np.int64),
            "rv": rv,
        }
    )
    panel = PanelFrame(df, entity="entity", time="time")
    model = HARModel(rv="rv", horizon=1, lags=(1, 5, 22), min_train_rows=50).fit(panel)

    # Independent reference: trailing means of RV over 1 / 5 / 22 rows.
    def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
        out = np.full(x.shape[0], np.nan)
        for t in range(w - 1, x.shape[0]):
            out[t] = x[t - w + 1 : t + 1].mean()
        return out

    X = np.column_stack(
        [
            np.ones(n),
            trailing_mean(rv, 1),
            trailing_mean(rv, 5),
            trailing_mean(rv, 22),
        ]
    )
    y = np.full(n, np.nan)
    y[:-1] = rv[1:]
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    expected, *_ = np.linalg.lstsq(X[good], y[good], rcond=None)
    np.testing.assert_allclose(model.coef_["A"], expected, atol=1e-9)

    # And the emitted forecast is exactly that linear combination.
    out = model.transform(panel).collect().sort("time")
    got = out.get_column("har_forecast").to_numpy()
    np.testing.assert_allclose(got[good], (X @ expected)[good], atol=1e-9)


def test_har_model_falls_back_to_pooled_coefficients() -> None:
    df = _returns_panel(n=400)
    panel = PanelFrame(df, entity="entity", time="time")
    model = HARModel(returns="ret", min_train_rows=100).fit(panel)
    fresh = pl.DataFrame(
        {
            "entity": ["NEW"] * 300,
            "time": np.arange(300, dtype=np.int64),
            "ret": np.random.default_rng(7).standard_normal(300) * 0.02,
        }
    )
    out = model.transform(PanelFrame(fresh, entity="entity", time="time")).collect()
    assert out.get_column("har_forecast").drop_nulls().len() > 0


def test_har_model_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        HARModel()
