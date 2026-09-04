"""Tests for the clean-room catch22 implementation (:mod:`polars_features.catch22`).

The reference ``pycatch22`` library is *not* a dependency, so these tests do not
assert exact numerical parity.  Instead they check, for a handful of
deterministic inputs (a sine wave, a fixed-seed random walk, a constant series),
that every feature returns a finite float of the right sign / rough magnitude,
that the aggregate helpers return the correct number of features, and -- where a
feature has a closed-form answer -- they sanity-check a couple of hand-computed
values.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from polars_features import catch22 as c22
from polars_features.catch22 import (
    CATCH22_NAMES,
    CO_trev_1_num,
    DN_HistogramMode_5,
    MD_hrv_classic_pnn40,
    SB_BinaryStats_diff_longstretch0,
    SB_BinaryStats_mean_longstretch1,
    SB_MotifThree_quantile_hh,
    catch22_all,
    catch22_all_expr,
    catch22_features,
)


# ---------------------------------------------------------------------------
# Deterministic fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sine() -> np.ndarray:
    t = np.linspace(0, 8 * np.pi, 500)
    return np.sin(t)


@pytest.fixture(scope="module")
def random_walk() -> np.ndarray:
    rng = np.random.default_rng(1234)
    return np.cumsum(rng.standard_normal(600))


@pytest.fixture(scope="module")
def constant() -> np.ndarray:
    return np.full(300, 3.0)


# ---------------------------------------------------------------------------
# catch22_all: shape, keys, finiteness
# ---------------------------------------------------------------------------
def test_catch22_all_returns_22_keys(sine):
    out = catch22_all(sine)
    assert len(out) == 22
    assert tuple(out) == CATCH22_NAMES
    assert all(isinstance(v, float) for v in out.values())


def test_catch24_adds_mean_and_std(random_walk):
    out = catch22_all(random_walk, catch24=True)
    assert len(out) == 24
    assert "DN_Mean" in out and "DN_Spread_Std" in out
    # DN_Mean / DN_Spread_Std are computed on the RAW series
    assert out["DN_Mean"] == pytest.approx(float(random_walk.mean()))
    assert out["DN_Spread_Std"] == pytest.approx(float(random_walk.std(ddof=1)))


@pytest.mark.parametrize("fixture", ["sine", "random_walk"])
def test_all_features_finite_on_wellposed_inputs(fixture, request):
    x = request.getfixturevalue(fixture)
    out = catch22_all(x)
    non_finite = {k: v for k, v in out.items() if not math.isfinite(v)}
    assert not non_finite, f"non-finite features on {fixture}: {non_finite}"


def test_constant_series_does_not_raise(constant):
    # A zero-variance series is degenerate: features may be NaN, but the call
    # must still return 22 floats without raising.
    out = catch22_all(constant)
    assert len(out) == 22
    assert all(isinstance(v, float) for v in out.values())


def test_single_feature_functions_are_standalone(sine):
    # Each public feature function z-scores internally and works on raw input.
    assert math.isfinite(DN_HistogramMode_5(sine))
    assert math.isfinite(CO_trev_1_num(sine))


# ---------------------------------------------------------------------------
# Hand-computed / closed-form sanity checks
# ---------------------------------------------------------------------------
def test_pnn40_short_ramp_is_one():
    # A short linear ramp has constant successive differences; after z-scoring
    # they are 1/std ~ 0.17 for n=20, which exceeds the 0.04 threshold, so
    # every difference counts -> pNN40 == 1.0.
    assert MD_hrv_classic_pnn40(np.arange(20.0)) == pytest.approx(1.0)


def test_pnn40_long_ramp_is_zero():
    # For a long ramp (n=200) the z-scored constant difference (~0.017) is below
    # the 0.04 threshold, so no difference counts -> pNN40 == 0.0.
    assert MD_hrv_classic_pnn40(np.arange(200.0)) == pytest.approx(0.0)


def test_mean_longstretch1_on_ramp():
    # arange(10) has mean 4.5; the values above the mean are 5..9, a single
    # run of length 5.
    assert SB_BinaryStats_mean_longstretch1(np.arange(10.0)) == pytest.approx(5.0)


def test_diff_longstretch0_monotone_series():
    # Strictly increasing: all diffs > 0 -> binary all 1 -> no run of 0s.
    assert SB_BinaryStats_diff_longstretch0(np.arange(10.0)) == pytest.approx(0.0)
    # Strictly decreasing: all diffs < 0 -> binary all 0 -> one run of length 9.
    assert SB_BinaryStats_diff_longstretch0(np.arange(10.0)[::-1]) == pytest.approx(9.0)


def test_trev_sign_reflects_temporal_asymmetry():
    ramp = np.arange(50.0)
    assert CO_trev_1_num(ramp) > 0  # increasing -> positive cubed diffs
    assert CO_trev_1_num(ramp[::-1]) < 0  # time-reversed -> negative
    # trev is exactly sign-antisymmetric under time reversal here
    assert CO_trev_1_num(ramp) == pytest.approx(-CO_trev_1_num(ramp[::-1]))


def test_dn_histogram_mode_matches_manual_histogram(random_walk):
    # Independent recomputation of the 5-bin histogram mode on the z-scored data.
    x = random_walk
    z = (x - x.mean()) / x.std(ddof=1)
    counts, edges = np.histogram(z, bins=5)
    centers = (edges[:-1] + edges[1:]) / 2.0
    expected = centers[counts == counts.max()].mean()
    assert DN_HistogramMode_5(x) == pytest.approx(expected)


def test_motif_three_entropy_bounds(random_walk):
    hh = SB_MotifThree_quantile_hh(random_walk)
    # Entropy of a 3x3 word distribution is in [0, log 9].
    assert 0.0 <= hh <= math.log(9) + 1e-9


def test_f1ecac_positive_and_reasonable(sine):
    # The 1/e crossing of a slowly-decaying oscillatory ACF is a positive lag.
    val = c22.CO_f1ecac(sine)
    assert math.isfinite(val)
    assert val > 0


def test_fluctanal_props_in_unit_interval(random_walk):
    dfa = c22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(random_walk)
    rsr = c22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(random_walk)
    for v in (dfa, rsr):
        assert math.isfinite(v)
        assert 0.0 <= v <= 1.0


def test_spectral_centroid_within_range(sine):
    centroid = c22.SP_Summaries_welch_rect_centroid(sine)
    area = c22.SP_Summaries_welch_rect_area_5_1(sine)
    assert 0.0 <= centroid <= math.pi + 1e-9
    assert 0.0 <= area <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Polars panel entry points
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    frames = []
    for ent in ("aaa", "bbb", "ccc"):
        n = 250
        frames.append(
            pl.DataFrame(
                {
                    "entity": [ent] * n,
                    "t": np.arange(n),
                    "value": np.cumsum(rng.standard_normal(n)),
                }
            )
        )
    return pl.concat(frames)


def test_catch22_features_one_row_per_entity(panel):
    out = catch22_features(panel, entity="entity", time="t", column="value")
    assert out.height == 3
    # entity column + 22 feature columns
    assert out.width == 23
    assert set(out.get_column("entity").to_list()) == {"aaa", "bbb", "ccc"}
    for name in CATCH22_NAMES:
        assert name in out.columns


def test_catch22_features_catch24(panel):
    out = catch22_features(
        panel, entity="entity", time="t", column="value", catch24=True
    )
    assert out.width == 25  # entity + 24
    assert "DN_Mean" in out.columns and "DN_Spread_Std" in out.columns


def test_catch22_features_subset(panel):
    which = ["DN_HistogramMode_5", "CO_trev_1_num"]
    out = catch22_features(
        panel, entity="entity", time="t", column="value", which=which
    )
    assert out.columns == ["entity", *which]
    assert out.height == 3


def test_catch22_features_no_entity_single_row(panel):
    one = panel.filter(pl.col("entity") == "aaa")
    out = catch22_features(one, time="t", column="value")
    assert out.height == 1
    assert out.width == 22


def test_catch22_features_is_leak_safe(panel):
    # Per-entity features must not depend on other entities: computing on one
    # entity in isolation must equal that entity's row in the panel result.
    full = catch22_features(panel, entity="entity", time="t", column="value")
    only_b = panel.filter(pl.col("entity") == "bbb")
    solo = catch22_features(only_b, entity="entity", time="t", column="value")
    row_full = full.filter(pl.col("entity") == "bbb").drop("entity").row(0)
    row_solo = solo.drop("entity").row(0)
    for a, b in zip(row_full, row_solo):
        assert a == pytest.approx(b, nan_ok=True)


def test_catch22_lazyframe_input(panel):
    out = catch22_features(panel.lazy(), entity="entity", time="t", column="value")
    assert out.height == 3


def test_catch22_all_expr_matches_function(panel):
    expr_out = (
        panel.sort("entity", "t")
        .group_by("entity", maintain_order=True)
        .agg(catch22_all_expr("value"))
        .unnest("catch22")
    )
    fn_out = catch22_features(panel, entity="entity", time="t", column="value")
    assert expr_out.height == 3
    assert expr_out.width == fn_out.width
    # Same entities, same feature columns
    assert set(expr_out.columns) == set(fn_out.columns)


def test_expr_namespace_registered():
    # The convenience namespace is registered exactly once and callable.
    assert hasattr(pl.Expr, "catch22")
    expr = pl.col("value").catch22.all()
    assert isinstance(expr, pl.Expr)


def test_unknown_feature_name_raises(panel):
    with pytest.raises(ValueError):
        catch22_features(
            panel, entity="entity", time="t", column="value", which=["not_a_feature"]
        )
