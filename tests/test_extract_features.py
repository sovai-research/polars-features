from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from panelary.feature_extractors import (
    _TS_SCALAR_AGGS,
    benford_correlation,
    extract_features,
    fft_coefficients,
    max_drawdown,
    return_kurtosis,
    return_skew,
)
from panelary.registry import registry


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _panel() -> pl.DataFrame:
    rng = np.random.RandomState(1)
    return pl.DataFrame(
        {
            "eid": ["a"] * 8 + ["b"] * 8,
            "t": list(range(8)) * 2,
            "v": rng.uniform(1.0, 10.0, 16),
        }
    )


# ---------------------------------------------------------------------------
# Task 2 — registry catalogue
# ---------------------------------------------------------------------------
def test_ts_features_are_registered():
    ts_specs = registry.by_namespace("ts")
    assert len(ts_specs) >= 40
    names = {s.name for s in ts_specs}
    assert {"absolute_energy", "max_drawdown", "benford_correlation"} <= names
    for spec in ts_specs:
        assert spec.namespace == "ts"
        assert spec.panel_safe is True
        assert spec.leakage_safe is True
        assert spec.source == "Panelary"


def test_registered_ts_specs_pass_license_audit():
    audit = registry.audit()
    for name in _TS_SCALAR_AGGS:
        assert name not in audit["non_permissive_license"]
        assert name not in audit["missing_panel_safe"]
        assert name not in audit["missing_leakage_safe"]
        assert name not in audit["missing_provenance"]


# ---------------------------------------------------------------------------
# Task 2 — bulk extract_features
# ---------------------------------------------------------------------------
def test_extract_features_all_one_row_per_entity():
    df = _panel()
    out = extract_features(df, entity="eid", time="t")
    assert isinstance(out, pl.DataFrame)
    # one row per entity
    assert out.height == 2
    assert set(out["eid"].to_list()) == {"a", "b"}
    # one column per registered scalar feature (+ the entity key)
    assert out.width == len(_TS_SCALAR_AGGS) + 1
    for name in _TS_SCALAR_AGGS:
        assert name in out.columns


def test_extract_features_matches_hand_composed_agg():
    df = _panel()
    feats = ["absolute_energy", "max_drawdown", "mean_change", "benford_correlation"]
    out = extract_features(df, entity="eid", time="t", features=feats).sort("eid")

    hand = (
        df.lazy()
        .group_by("eid", maintain_order=True)
        .agg(
            pl.col("v").ts.absolute_energy().alias("absolute_energy"),
            pl.col("v").ts.max_drawdown().alias("max_drawdown"),
            pl.col("v").ts.mean_change().alias("mean_change"),
            pl.col("v").ts.benford_correlation().alias("benford_correlation"),
        )
        .collect()
        .sort("eid")
    )
    assert_frame_equal(out, hand)


def test_extract_features_subset():
    df = _panel()
    out = extract_features(df, entity="eid", time="t", features=["mean_change"])
    assert out.columns == ["eid", "mean_change"]


def test_extract_features_lazy_in_lazy_out():
    df = _panel()
    out = extract_features(df.lazy(), entity="eid", time="t", features=["mean_change"])
    assert isinstance(out, pl.LazyFrame)
    assert out.collect().height == 2


def test_extract_features_defaults_entity_time_to_first_two_columns():
    df = _panel()
    out = extract_features(df, features=["mean_change"])
    # entity defaults to first column ("eid")
    assert "eid" in out.columns
    assert out.height == 2


def test_extract_features_multi_column_namespacing():
    df = _panel().with_columns((pl.col("v") * 2.0).alias("w"))
    out = extract_features(df, entity="eid", time="t", features=["mean_change"])
    assert "v__mean_change" in out.columns
    assert "w__mean_change" in out.columns


def test_extract_features_unknown_feature_raises():
    df = _panel()
    with pytest.raises(ValueError, match="Unknown ts feature"):
        extract_features(df, entity="eid", time="t", features=["not_a_feature"])


# ---------------------------------------------------------------------------
# Task 1 — bug-fix regression tests
# ---------------------------------------------------------------------------
def test_fft_phase_equals_np_angle():
    x = pl.Series("x", [1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 0.5, 2.2])
    out = fft_coefficients(x)
    expected_deg = np.degrees(np.angle(np.fft.rfft(x.to_numpy())))
    np.testing.assert_allclose(out["angle"], expected_deg, rtol=1e-9, atol=1e-9)


def test_benford_handles_scientific_notation():
    # Values whose float repr uses scientific notation ("1.23e-05") used to be
    # mis-parsed by the old string-stripping leading-digit extraction.
    data = [1.23e-5, 2.5e10, 3.14, 9.99e-8, 1.0, 8e3, 4.2, 7.7e-2, 5e-9]
    series_val = benford_correlation(pl.Series("v", data))
    expr_val = (
        pl.DataFrame({"v": data})
        .select(benford_correlation(pl.col("v")).alias("c"))
        .item()
    )
    # Series and expression paths must agree, and both must be finite (not NaN
    # from a mis-parse collapsing every value onto the same fake leading digit).
    assert np.isfinite(series_val)
    assert np.isclose(series_val, expr_val)

    # Hand-computed leading digits: 1, 2, 3, 9, 1, 8, 4, 7, 5 -> compare to a
    # direct numeric bincount correlation.
    lead = [1, 2, 3, 9, 1, 8, 4, 7, 5]
    counts = np.bincount(lead, minlength=10)[1:10]
    benford = np.log10(1.0 + 1.0 / np.arange(1, 10))
    expected = np.corrcoef(counts, benford)[0, 1]
    assert np.isclose(series_val, expected)


def test_benford_scientific_vs_plain_are_identical():
    # 1e-05 and 0.00001 are the same number; leading digit must be 1 for both.
    sci = pl.Series("v", [1e-5, 2e-5, 3e-5, 9e-5, 5e-5, 4e-5, 7e-5, 8e-5, 6e-5])
    plain = pl.Series(
        "v",
        [
            0.00001,
            0.00002,
            0.00003,
            0.00009,
            0.00005,
            0.00004,
            0.00007,
            0.00008,
            0.00006,
        ],
    )
    assert np.isclose(benford_correlation(sci), benford_correlation(plain))


def test_max_drawdown_module_equals_namespace():
    prices = pl.Series("p", [100.0, 110.0, 90.0, 95.0, 80.0, 120.0, 60.0])
    module_val = max_drawdown(prices)
    ns_val = (
        pl.DataFrame({"p": prices})
        .select(pl.col("p").ts.max_drawdown().alias("d"))
        .item()
    )
    assert np.isclose(module_val, ns_val)
    # ratio form: peak 120 -> trough 60 => 60/120 - 1 = -0.5
    assert np.isclose(module_val, -0.5)


def test_return_skew_kurtosis_module_equals_namespace():
    prices = pl.Series("p", [100.0, 110.0, 90.0, 95.0, 80.0, 120.0, 60.0])
    df = pl.DataFrame({"p": prices})
    ns_skew = df.select(pl.col("p").ts.return_skew().alias("s")).item()
    ns_kurt = df.select(pl.col("p").ts.return_kurtosis().alias("k")).item()
    assert np.isclose(return_skew(prices), ns_skew)
    assert np.isclose(return_kurtosis(prices), ns_kurt)
