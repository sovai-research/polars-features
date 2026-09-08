"""The leak-safety acceptance test for every econ feature generator.

The plan's criterion for workstream 2 is: *"a leakage test proves each is
invariant to future rows"*. That is exactly what this file does, mechanically,
for every generator in :mod:`polars_features.econ.features`.

The protocol
------------
Build a panel, compute the features, then **append future rows** to every entity
and recompute. Every feature value on a row that existed in the short panel must
be **bit-for-bit identical**. If a generator peeked forward -- a centred moving
average, a full-sample quantile, a two-sided STL, a whole-sample standardisation
-- appending rows would move those earlier values and the comparison would fail.

The cross-sectional (per-date) generators are tested the other way round: their
values must be unchanged when *other dates* are removed, since a per-date fit
must never pool across dates.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_features.core import PanelFrame
from polars_features.econ.features import (
    AutoFracDiff,
    CausalSeasonalDecomposer,
    HARModel,
    StationarityDifferencer,
    decompose_features,
    evt_features,
    har_features,
    liquidity_features,
    nelson_siegel_factors,
    rolling_beta,
    rolling_long_memory_features,
    rolling_unit_root_features,
)

ENTITIES = ("A", "B")
N_SHORT = 400
N_EXTRA = 120


def _panel(n: int, *, seed: int = 0) -> pl.DataFrame:
    """A long panel with everything the generators might need."""
    frames = []
    for i, ent in enumerate(ENTITIES):
        rng = np.random.default_rng(seed + i)
        ret = rng.standard_normal(n) * 0.01 + 0.0002
        px = 100.0 * np.exp(np.cumsum(ret))
        frames.append(
            pl.DataFrame(
                {
                    "entity": [ent] * n,
                    "time": np.arange(n, dtype=np.int64),
                    "ret": ret,
                    "px": px,
                    "mkt": rng.standard_normal(n) * 0.01,
                    "dv": np.abs(rng.standard_normal(n)) * 1e6 + 1e5,
                    "vol": np.abs(rng.standard_normal(n)) * 1e4 + 1e3,
                    # A clean seasonal series for the decomposition.
                    "seasonal": (
                        np.sin(2 * np.pi * np.arange(n) / 12.0)
                        + 0.02 * np.arange(n)
                        + rng.standard_normal(n) * 0.1
                    ),
                }
            )
        )
    return pl.concat(frames).sort(["entity", "time"])


@pytest.fixture(scope="module")
def extended_panel() -> pl.DataFrame:
    """The full panel: ``N_SHORT`` rows plus ``N_EXTRA`` *future* rows per entity."""
    return _panel(N_SHORT + N_EXTRA, seed=0)


@pytest.fixture(scope="module")
def short_panel(extended_panel: pl.DataFrame) -> pl.DataFrame:
    """The same panel truncated to its first ``N_SHORT`` rows per entity.

    Derived by slicing the long panel rather than re-simulating, so the two
    frames agree *exactly* on the shared rows -- otherwise the comparison would
    be testing the random-number stream, not the feature generators.
    """
    return extended_panel.filter(pl.col("time") < N_SHORT)


def _rows_for(frame: pl.DataFrame, entity: str, upto: int) -> pl.DataFrame:
    return frame.filter((pl.col("entity") == entity) & (pl.col("time") < upto)).sort(
        "time"
    )


def assert_future_rows_do_not_change_the_past(
    short: pl.DataFrame, extended: pl.DataFrame, columns: list[str]
) -> None:
    """Every ``columns`` value on a pre-existing row must be unchanged."""
    assert columns, "no feature columns to compare"
    for ent in ENTITIES:
        a = _rows_for(short, ent, N_SHORT)
        b = _rows_for(extended, ent, N_SHORT)
        assert a.height == b.height == N_SHORT
        for col in columns:
            left = a.get_column(col).to_numpy().astype(float)
            right = b.get_column(col).to_numpy().astype(float)
            np.testing.assert_array_equal(
                np.isnan(left),
                np.isnan(right),
                err_msg=f"{col}: null/NaN mask changed for entity {ent}",
            )
            np.testing.assert_allclose(
                left,
                right,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
                err_msg=(
                    f"{col}: appending future rows changed a PAST value for "
                    f"entity {ent} -- this generator leaks."
                ),
            )


def _new_columns(before: pl.DataFrame, after: pl.DataFrame) -> list[str]:
    return [c for c in after.columns if c not in before.columns]


# --------------------------------------------------------------------------- #
# Rolling / trailing generators
# --------------------------------------------------------------------------- #
def test_har_features_are_invariant_to_future_rows(short_panel, extended_panel):
    kwargs = {"entity": "entity", "time": "time", "returns": "ret", "window": 22}
    a = har_features(short_panel, **kwargs)
    b = har_features(extended_panel, **kwargs)
    assert_future_rows_do_not_change_the_past(a, b, _new_columns(short_panel, a))


def test_liquidity_features_are_invariant_to_future_rows(short_panel, extended_panel):
    kwargs = {
        "entity": "entity",
        "time": "time",
        "returns": "ret",
        "price": "px",
        "dollar_volume": "dv",
        "volume": "vol",
        "windows": (21, 63),
    }
    a = liquidity_features(short_panel, **kwargs)
    b = liquidity_features(extended_panel, **kwargs)
    assert_future_rows_do_not_change_the_past(a, b, _new_columns(short_panel, a))


def test_evt_features_are_invariant_to_future_rows(short_panel, extended_panel):
    kwargs = {"entity": "entity", "time": "time", "window": 252}
    a = evt_features(short_panel, "ret", **kwargs)
    b = evt_features(extended_panel, "ret", **kwargs)
    assert_future_rows_do_not_change_the_past(a, b, _new_columns(short_panel, a))


def test_rolling_unit_root_features_are_invariant_to_future_rows(
    short_panel, extended_panel
):
    kwargs = {
        "entity": "entity",
        "time": "time",
        "window": 200,
        "tests": ("adf", "kpss"),
    }
    a = rolling_unit_root_features(short_panel, "px", **kwargs)
    b = rolling_unit_root_features(extended_panel, "px", **kwargs)
    assert_future_rows_do_not_change_the_past(a, b, _new_columns(short_panel, a))


def test_rolling_long_memory_features_are_invariant_to_future_rows(
    short_panel, extended_panel
):
    kwargs = {"entity": "entity", "time": "time", "window": 256}
    a = rolling_long_memory_features(short_panel, "ret", **kwargs)
    b = rolling_long_memory_features(extended_panel, "ret", **kwargs)
    assert_future_rows_do_not_change_the_past(a, b, _new_columns(short_panel, a))


def test_causal_decomposition_is_invariant_to_future_rows(short_panel, extended_panel):
    """The decisive test for the decomposition.

    A classical two-sided STL would fail this outright: its trend at time ``t``
    averages observations on both sides of ``t``, so appending rows rewrites
    history. The trailing-window decomposition must not move at all.
    """
    kwargs = {"entity": "entity", "time": "time", "period": 12, "strength_window": 36}
    a = decompose_features(short_panel, "seasonal", **kwargs)
    b = decompose_features(extended_panel, "seasonal", **kwargs)
    assert_future_rows_do_not_change_the_past(a, b, _new_columns(short_panel, a))


def test_two_sided_smoother_would_fail_this_test(short_panel, extended_panel):
    """Control: prove the leakage test has teeth.

    A *centred* moving average -- the smoother a classical STL uses -- is
    computed here on the same data and must FAIL the invariance check, showing
    that the passing tests above are not vacuous.
    """

    def centred(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("seasonal")
            .rolling_mean(13, center=True, min_samples=1)
            .over("entity")
            .alias("two_sided_trend")
        )

    with pytest.raises(AssertionError, match="leaks"):
        assert_future_rows_do_not_change_the_past(
            centred(short_panel), centred(extended_panel), ["two_sided_trend"]
        )


def test_rolling_beta_is_invariant_to_future_rows(short_panel, extended_panel):
    kwargs = {"entity": "entity", "time": "time", "y": "ret", "x": "mkt", "window": 60}
    a = rolling_beta(short_panel, **kwargs)
    b = rolling_beta(extended_panel, **kwargs)
    assert_future_rows_do_not_change_the_past(a, b, _new_columns(short_panel, a))


# --------------------------------------------------------------------------- #
# Fitted transformers: fitting on train must not depend on the future either
# --------------------------------------------------------------------------- #
def test_har_model_predictions_are_invariant_to_future_rows(
    short_panel, extended_panel
):
    """Fit ONCE on the short panel, then apply to both panels.

    A fitted transformer's leak-safety claim is that a test-fold row's feature
    depends only on frozen parameters plus its own past -- so applying the same
    fitted model to a longer panel must reproduce the short panel's values.
    """
    train = PanelFrame(short_panel, entity="entity", time="time")
    model = HARModel(returns="ret", window=22, min_train_rows=100).fit(train)
    a = model.transform(train).collect()
    b = model.transform(
        PanelFrame(extended_panel, entity="entity", time="time")
    ).collect()
    assert_future_rows_do_not_change_the_past(a, b, ["har_forecast", "rv", "har_w"])


def test_auto_fracdiff_output_is_invariant_to_future_rows(short_panel, extended_panel):
    train = PanelFrame(short_panel, entity="entity", time="time")
    tr = AutoFracDiff(["px"], bandwidth_exponent=0.7).fit(train)
    a = tr.transform(train).collect()
    b = tr.transform(PanelFrame(extended_panel, entity="entity", time="time")).collect()
    assert_future_rows_do_not_change_the_past(a, b, ["px_ffd"])


def test_stationarity_differencer_output_is_invariant_to_future_rows(
    short_panel, extended_panel
):
    train = PanelFrame(short_panel, entity="entity", time="time")
    tr = StationarityDifferencer(["px"], suffix="_d").fit(train)
    a = tr.transform(train).collect()
    b = tr.transform(PanelFrame(extended_panel, entity="entity", time="time")).collect()
    assert_future_rows_do_not_change_the_past(a, b, ["px_d"])


def test_causal_decomposer_step_is_invariant_to_future_rows(
    short_panel, extended_panel
):
    train = PanelFrame(short_panel, entity="entity", time="time")
    tr = CausalSeasonalDecomposer(["seasonal"], period=12).fit(train)
    a = tr.transform(train).collect()
    b = tr.transform(PanelFrame(extended_panel, entity="entity", time="time")).collect()
    assert_future_rows_do_not_change_the_past(
        a, b, ["seasonal_trend", "seasonal_seasonal", "seasonal_remainder"]
    )


def test_every_transformer_declares_the_leak_safety_contract() -> None:
    """The metaclass check only fires on definition; assert the values too."""
    for cls in (
        StationarityDifferencer,
        AutoFracDiff,
        HARModel,
        CausalSeasonalDecomposer,
    ):
        assert cls.panel_safe is True, cls.__name__
        assert cls.leakage_safe is True, cls.__name__


def test_fitted_state_does_not_change_when_transforming_new_data(
    short_panel, extended_panel
):
    """Re-fitting inside ``transform`` is the other half of the leak.

    Transforming a longer panel must leave every learned parameter untouched.
    """
    train = PanelFrame(short_panel, entity="entity", time="time")
    later = PanelFrame(extended_panel, entity="entity", time="time")

    diff = StationarityDifferencer(["px"]).fit(train)
    before = dict(diff.orders_)
    diff.transform(later)
    assert diff.orders_ == before

    har = HARModel(returns="ret", min_train_rows=100).fit(train)
    coefs = {k: v.copy() for k, v in har.coef_.items()}
    har.transform(later)
    for key, value in coefs.items():
        np.testing.assert_array_equal(har.coef_[key], value)


# --------------------------------------------------------------------------- #
# Per-date (cross-sectional) generators
# --------------------------------------------------------------------------- #
def test_nelson_siegel_factors_are_unaffected_by_other_dates() -> None:
    """A per-date fit must never pool across dates.

    Removing every date after ``t`` (and every date before it) must leave date
    ``t``'s factors bit-for-bit identical.
    """
    maturities = np.array([3.0, 12.0, 36.0, 60.0, 120.0])
    rng = np.random.default_rng(0)
    rows = []
    for t in range(40):
        base = 3.0 + 0.5 * rng.standard_normal()
        for mat in maturities:
            rows.append(
                {
                    "maturity": float(mat),
                    "date": t,
                    "yield": base + 2.0 * (1 - np.exp(-0.06 * mat)) / (0.06 * mat),
                }
            )
    df = pl.DataFrame(rows)

    full = nelson_siegel_factors(
        df, date="date", maturity="maturity", yield_col="yield"
    )
    for t in (0, 17, 39):
        single = nelson_siegel_factors(
            df.filter(pl.col("date") == t),
            date="date",
            maturity="maturity",
            yield_col="yield",
        )
        for col in ("ns_level", "ns_slope", "ns_curvature"):
            assert (
                single.get_column(col)[0]
                == full.filter(pl.col("date") == t).get_column(col)[0]
            )
