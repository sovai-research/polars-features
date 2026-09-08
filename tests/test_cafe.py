"""Tests for the CAFE-backed, leak-safe imputer.

Covered:
* ``cafe_impute`` fills nulls, preserves observed values / schema / column order,
* opt-in by-product columns (sigma / recoverability / anomaly / missingness),
* the point-in-time (leak-safety) property: truncating future rows never changes
  an earlier ``(entity, time)`` fill,
* the ``CafeImputer`` fit/transform contract.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("cafe")

from polars_features.imputation import CafeImputer  # noqa: E402
from polars_features.preprocessing import cafe_impute, impute  # noqa: E402


def _panel() -> pl.DataFrame:
    """A small long-format panel with interior nulls and a passthrough column."""
    entities = ["A", "B", "C"]
    n_t = 10
    rng = np.random.default_rng(0)
    rows = []
    for e_i, ent in enumerate(entities):
        for t in range(n_t):
            x = float(1.0 + e_i + 0.5 * t + rng.normal(0, 0.05))
            y = float(10.0 - e_i + t + rng.normal(0, 0.05))
            rows.append({"entity": ent, "time": t, "x": x, "y": y, "grp": ent.lower()})
    df = pl.DataFrame(rows)
    # Punch interior nulls (never at the tail, so a truncated prefix still fills them).
    df = df.with_columns(
        pl.when((pl.col("time") == 3) & (pl.col("entity") == "A"))
        .then(None)
        .otherwise(pl.col("x"))
        .alias("x"),
        pl.when((pl.col("time") == 4) & (pl.col("entity") == "B"))
        .then(None)
        .otherwise(pl.col("y"))
        .alias("y"),
    )
    return df


def test_fills_nulls_and_preserves_schema() -> None:
    df = _panel()
    assert df.null_count().select(pl.sum_horizontal(pl.all())).item() > 0

    result = cafe_impute()(df.lazy()).collect()

    # Column order and schema preserved; passthrough column untouched.
    assert result.columns == df.columns
    assert result.schema == df.schema
    assert result.get_column("grp").to_list() == df.get_column("grp").to_list()

    # All numeric nulls filled.
    assert result.select(pl.col("x", "y")).null_count().to_numpy().sum() == 0

    # Observed (non-null) values are preserved exactly.
    observed = df.get_column("x").is_not_null().to_numpy()
    np.testing.assert_allclose(
        result.get_column("x").to_numpy()[observed],
        df.get_column("x").to_numpy()[observed],
    )


def test_impute_cafe_alias_matches_direct() -> None:
    df = _panel()
    via_alias = impute("cafe")(df.lazy()).collect()
    direct = cafe_impute()(df.lazy()).collect()
    assert via_alias.columns == direct.columns
    np.testing.assert_allclose(
        via_alias.get_column("x").to_numpy(),
        direct.get_column("x").to_numpy(),
    )


def test_byproduct_columns_appear_when_requested() -> None:
    df = _panel()
    result = cafe_impute(
        add_uncertainty=True,
        add_recoverability=True,
        add_anomaly=True,
        add_missingness=True,
    )(df.lazy()).collect()

    for col in ("x", "y"):
        assert f"{col}__cafe_sigma" in result.columns
        assert f"{col}__cafe_recoverability" in result.columns
        assert f"{col}__cafe_was_imputed" in result.columns
    assert "cafe_anomaly" in result.columns

    # The was-imputed indicator matches the original null pattern.
    assert (
        result.get_column("x__cafe_was_imputed").to_list()
        == df.get_column("x").is_null().to_list()
    )
    # Sigma is finite (>= 0) exactly where a value was imputed, NaN where observed.
    sigma = result.get_column("x__cafe_sigma").to_numpy()
    was_imputed = df.get_column("x").is_null().to_numpy()
    assert np.all(np.isfinite(sigma[was_imputed]))
    assert np.all(np.isnan(sigma[~was_imputed]))
    # Anomaly score is a per-row value in [0, 1].
    anomaly = result.get_column("cafe_anomaly").to_numpy()
    assert anomaly.shape[0] == df.height
    assert np.all((anomaly >= 0) & (anomaly <= 1))


def test_no_byproducts_by_default() -> None:
    df = _panel()
    result = cafe_impute()(df.lazy()).collect()
    assert result.columns == df.columns


def test_columns_restricts_imputation() -> None:
    df = _panel()
    # Only impute x; y should keep its original null.
    result = cafe_impute(columns=["x"])(df.lazy()).collect()
    assert result.get_column("x").null_count() == 0
    assert result.get_column("y").null_count() == df.get_column("y").null_count()


def test_leak_safety_point_in_time() -> None:
    """Truncating future rows must not change any earlier (entity, time) fill."""
    df = _panel()
    cutoff = 6

    full = cafe_impute()(df.lazy()).collect()
    prefix = cafe_impute()(df.filter(pl.col("time") <= cutoff).lazy()).collect()

    keys = ["entity", "time"]
    full_early = full.filter(pl.col("time") <= cutoff).sort(keys)
    prefix_sorted = prefix.sort(keys)

    for col in ("x", "y"):
        np.testing.assert_allclose(
            full_early.get_column(col).to_numpy(),
            prefix_sorted.get_column(col).to_numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"future rows leaked into earlier {col} fills",
        )


def test_cafe_imputer_fit_transform() -> None:
    df = _panel()
    imp = CafeImputer(entity="entity", time="time")
    assert imp.panel_safe is True
    assert imp.leakage_safe is True

    imp.fit(df)
    assert imp.is_fitted
    assert imp.feature_cols_ == ["x", "y", "grp"]

    out = imp.transform(df)
    filled = out.collect()
    assert filled.columns == df.columns
    assert filled.select(pl.col("x", "y")).null_count().to_numpy().sum() == 0

    # fit_transform is equivalent on training data.
    ft = CafeImputer(entity="entity", time="time").fit_transform(df).collect()
    np.testing.assert_allclose(
        ft.get_column("x").to_numpy(), filled.get_column("x").to_numpy()
    )
