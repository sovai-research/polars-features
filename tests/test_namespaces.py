"""Hardening tests for the frame-level ``.panel`` / ``.xs`` namespaces.

Focus areas:

* **Leak safety (Task 1).** The frame-level ``.panel`` ops must *require* an
  ``over`` entity key — computing a per-entity time-series op across the whole
  frame silently bleeds one entity's history into the next.
* **Reference equivalence.** Frame ops must equal the hand-composed
  ``pl.col(...).<ns>.<op>().over(key)`` expression they are sugar for.
* **Ergonomics (Task 2).** Multi-column + ``suffix`` engineering in one call.
* **Cross-sectional ops (Task 3).** ``winsorize`` / ``quantile_bin`` /
  ``zscore`` / ``neutralize`` computed per time-slice (contemporaneous only).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_features.namespaces  # noqa: F401  (registers the namespaces)
from polars_features.registry import registry


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def panel_df() -> pl.DataFrame:
    # Two entities, distinct level per entity so a whole-frame (leaky) computation
    # differs sharply from the correct per-entity one.
    return pl.DataFrame(
        {
            "e": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "d": [1, 2, 3, 4, 1, 2, 3, 4],
            "ret": [0.1, 0.2, 0.3, 0.4, 10.0, 11.0, 12.0, 13.0],
            "vol": [1.0, 2.0, 1.5, 2.5, 5.0, 4.0, 6.0, 3.0],
        }
    )


@pytest.fixture
def xs_df() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[tuple[str, int, float, float, float]] = []
    for d in (1, 2, 3):
        for i in range(25):
            beta = float(rng.normal())
            size = float(rng.normal())
            ret = 0.4 * beta - 0.6 * size + 0.1 * float(rng.normal())
            rows.append((f"e{i}", d, ret, beta, size))
    return pl.DataFrame(rows, schema=["e", "d", "ret", "beta", "size"], orient="row")


# --------------------------------------------------------------------------- #
# Task 1 — the frame-level .panel ops must REQUIRE `over` (no cross-entity leak)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("op", "kwargs"),
    [
        ("frac_diff", {"d": 0.4}),
        ("zscore", {"window": 2}),
        ("rs_vol", {"window": 2}),
    ],
)
def test_panel_frame_op_requires_over(
    panel_df: pl.DataFrame, op: str, kwargs: dict
) -> None:
    method = getattr(panel_df.panel, op)
    with pytest.raises(ValueError, match="requires an `over`"):
        method("ret", **kwargs)  # no over= -> must refuse
    # LazyFrame path refuses too (error surfaces at call time, pre-collect).
    with pytest.raises(ValueError, match="requires an `over`"):
        getattr(panel_df.lazy().panel, op)("ret", **kwargs)


def test_panel_over_error_mentions_entity_crossing(panel_df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="cross-entity leak"):
        panel_df.panel.zscore("ret", window=2)


def test_panel_specs_declare_panel_safe() -> None:
    for name in ("frac_diff", "zscore", "rs_vol"):
        assert registry.get(name).panel_safe is True


# --------------------------------------------------------------------------- #
# Reference equivalence — frame op == manual .over(entity)
# --------------------------------------------------------------------------- #
def test_panel_frame_op_equals_manual_over_entity(panel_df: pl.DataFrame) -> None:
    got = panel_df.panel.zscore("ret", window=2, over="e", alias="z")
    ref = panel_df.with_columns(
        z=pl.col("ret").panel.zscore(2).over("e"),
    )
    assert got["z"].to_list() == pytest.approx(ref["z"].to_list(), nan_ok=True)


def test_panel_over_differs_from_whole_frame(panel_df: pl.DataFrame) -> None:
    # The per-entity result must NOT equal the (leaky) whole-frame result.
    per_entity = panel_df.with_columns(z=pl.col("ret").panel.zscore(2).over("e"))[
        "z"
    ].to_list()
    whole_frame = panel_df.with_columns(z=pl.col("ret").panel.zscore(2))["z"].to_list()
    assert per_entity != pytest.approx(whole_frame, nan_ok=True)


def test_panel_frac_diff_no_cross_entity_bleed(panel_df: pl.DataFrame) -> None:
    # First observation of each entity has no in-entity predecessor -> null.
    out = panel_df.panel.frac_diff("ret", d=1.0, over="e", alias="fd").sort("e", "d")
    fd = out.get_column("fd")
    # exactly one leading null per entity (2 entities)
    assert fd.is_null().sum() == 2


# --------------------------------------------------------------------------- #
# Task 2 — multi-column + suffix ergonomics
# --------------------------------------------------------------------------- #
def test_panel_multi_column_suffix(panel_df: pl.DataFrame) -> None:
    out = panel_df.panel.zscore(["ret", "vol"], window=2, over="e", suffix="_z")
    assert "ret_z" in out.columns and "vol_z" in out.columns
    # originals untouched
    assert out["ret"].to_list() == panel_df["ret"].to_list()
    # each equals its manual per-column reference
    ref = panel_df.with_columns(
        ret_z=pl.col("ret").panel.zscore(2).over("e"),
        vol_z=pl.col("vol").panel.zscore(2).over("e"),
    )
    assert out["ret_z"].to_list() == pytest.approx(ref["ret_z"].to_list(), nan_ok=True)
    assert out["vol_z"].to_list() == pytest.approx(ref["vol_z"].to_list(), nan_ok=True)


def test_xs_multi_column_suffix(xs_df: pl.DataFrame) -> None:
    out = xs_df.xs.rank(["ret", "beta"], over="d", suffix="_rk")
    assert "ret_rk" in out.columns and "beta_rk" in out.columns
    assert out["ret"].to_list() == xs_df["ret"].to_list()  # originals kept


def test_single_string_backward_compatible(panel_df: pl.DataFrame) -> None:
    # A bare string column + alias still works exactly as before.
    out = panel_df.panel.frac_diff("ret", d=0.4, over="e", alias="ret_fd")
    assert "ret_fd" in out.columns and out.height == panel_df.height


def test_alias_and_suffix_are_mutually_exclusive(panel_df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="either `alias` or `suffix`"):
        panel_df.panel.zscore("ret", window=2, over="e", alias="z", suffix="_z")


def test_alias_rejected_for_multiple_columns(panel_df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="only valid for a single column"):
        panel_df.panel.zscore(["ret", "vol"], window=2, over="e", alias="z")


# --------------------------------------------------------------------------- #
# Task 3 — cross-sectional ops match a manual .over(date) reference
# --------------------------------------------------------------------------- #
def test_xs_requires_over(xs_df: pl.DataFrame) -> None:
    for call in (
        lambda: xs_df.xs.zscore("ret"),
        lambda: xs_df.xs.winsorize("ret", limits=0.1),
        lambda: xs_df.xs.quantile_bin("ret", q=4),
        lambda: xs_df.xs.neutralize("ret", by="beta"),
    ):
        with pytest.raises(ValueError, match="requires an `over`"):
            call()


def test_xs_zscore_matches_manual_over_date(xs_df: pl.DataFrame) -> None:
    got = xs_df.xs.zscore("ret", over="d", alias="z")["z"].to_list()
    ref = xs_df.with_columns(z=pl.col("ret").xs.zscore().over("d"))["z"].to_list()
    assert got == pytest.approx(ref, nan_ok=True)
    # Each date's cross-section is standardized (mean ~0, pop-std ~1).
    for d in (1, 2, 3):
        z = np.asarray(
            xs_df.xs.zscore("ret", over="d", alias="z")
            .filter(pl.col("d") == d)["z"]
            .to_list()
        )
        assert z.mean() == pytest.approx(0.0, abs=1e-9)
        assert z.std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_xs_winsorize_matches_manual_and_clips(xs_df: pl.DataFrame) -> None:
    got = xs_df.xs.winsorize("ret", limits=(0.1, 0.9), over="d", alias="w")
    ref = xs_df.with_columns(w=pl.col("ret").xs.winsorize((0.1, 0.9)).over("d"))
    assert got["w"].to_list() == pytest.approx(ref["w"].to_list(), nan_ok=True)
    # Winsorized values never exceed the per-date quantile band. Use polars'
    # quantile (default "nearest" interpolation) so the band matches the op.
    for d in (1, 2, 3):
        raw = xs_df.filter(pl.col("d") == d)["ret"]
        lo, hi = raw.quantile(0.1), raw.quantile(0.9)
        w = np.asarray(got.filter(pl.col("d") == d)["w"].to_list())
        assert w.min() >= lo - 1e-9 and w.max() <= hi + 1e-9


def test_xs_quantile_bin_matches_manual(xs_df: pl.DataFrame) -> None:
    got = xs_df.xs.quantile_bin("ret", q=5, over="d", alias="b")
    ref = xs_df.with_columns(b=pl.col("ret").xs.quantile_bin(5).over("d"))
    assert got["b"].to_list() == ref["b"].to_list()
    # 5 equal-count buckets over 25 entities -> each bucket used, indices in [0,5).
    for d in (1, 2, 3):
        bins = got.filter(pl.col("d") == d)["b"].to_list()
        assert set(bins) == {0, 1, 2, 3, 4}


def test_xs_rank_matches_manual_over_date(xs_df: pl.DataFrame) -> None:
    got = xs_df.xs.rank("ret", over="d", normalize=True, alias="r")["r"].to_list()
    ref = xs_df.with_columns(r=pl.col("ret").xs.rank(normalize=True).over("d"))[
        "r"
    ].to_list()
    assert got == pytest.approx(ref, nan_ok=True)


# --------------------------------------------------------------------------- #
# Task 3 — neutralize residual orthogonality (per date)
# --------------------------------------------------------------------------- #
def test_xs_neutralize_residual_orthogonal_to_factors(xs_df: pl.DataFrame) -> None:
    out = xs_df.xs.neutralize("ret", by=["beta", "size"], over="d")
    assert "ret_neutral" in out.columns  # default suffix keeps original + adds col
    assert out["ret"].to_list() == xs_df["ret"].to_list()
    for d in (1, 2, 3):
        sub = out.filter(pl.col("d") == d)
        resid = np.asarray(sub["ret_neutral"].to_list())
        beta = np.asarray(sub["beta"].to_list())
        size = np.asarray(sub["size"].to_list())
        # Residual is orthogonal to each factor and (with intercept) de-meaned.
        assert float(resid @ beta) == pytest.approx(0.0, abs=1e-8)
        assert float(resid @ size) == pytest.approx(0.0, abs=1e-8)
        assert float(resid.mean()) == pytest.approx(0.0, abs=1e-9)


def test_xs_neutralize_matches_manual_over_date(xs_df: pl.DataFrame) -> None:
    got = xs_df.xs.neutralize("ret", by=["beta", "size"], over="d", alias="n")[
        "n"
    ].to_list()
    ref = xs_df.with_columns(n=pl.col("ret").xs.neutralize(["beta", "size"]).over("d"))[
        "n"
    ].to_list()
    assert got == pytest.approx(ref, nan_ok=True)


def test_xs_neutralize_no_intercept_not_demeaned() -> None:
    # Without an intercept, the residual is neutral to the factor but need not
    # be de-meaned; orthogonality to the factor still holds per date.
    rng = np.random.default_rng(1)
    rows = []
    for d in (1, 2):
        for i in range(15):
            beta = float(rng.normal())
            ret = 1.0 + 0.5 * beta + 0.1 * float(rng.normal())
            rows.append((f"e{i}", d, ret, beta))
    df = pl.DataFrame(rows, schema=["e", "d", "ret", "beta"], orient="row")
    out = df.xs.neutralize("ret", by="beta", over="d", add_intercept=False)
    for d in (1, 2):
        sub = out.filter(pl.col("d") == d)
        resid = np.asarray(sub["ret_neutral"].to_list())
        beta = np.asarray(sub["beta"].to_list())
        assert float(resid @ beta) == pytest.approx(0.0, abs=1e-8)


# --------------------------------------------------------------------------- #
# Registry — new specs are present with the declared safety flags
# --------------------------------------------------------------------------- #
def test_new_xs_specs_registered() -> None:
    # zscore method registers under a collision-free name (flat registry key).
    for name in ("winsorize", "quantile_bin", "neutralize", "cs_zscore"):
        spec = registry.get(name)
        assert spec.namespace == "xs"
        assert spec.leakage_safe is True
        assert spec.panel_safe is True
