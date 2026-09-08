"""Tests for the leak-safe ``factor/`` toolkit and the ``.xs`` factor sugar.

The two load-bearing invariants: **forward-return alignment is a backward shift
only** (no fabricated future, gaps rejected) and **all standardization is
per-date, never global**. The rest checks the evaluators recover known answers
on toy panels.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_features.namespaces  # noqa: F401  (registers .xs / .panel)
from polars_features import factor as fc
from polars_features.registry import registry


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def signal_panel() -> pl.DataFrame:
    """A panel whose signal is monotonically related to its forward return."""
    rng = np.random.default_rng(7)
    rows: list[tuple[str, int, float, float]] = []
    for d in range(1, 13):
        for i in range(40):
            sig = float(rng.normal())
            # forward return increases in the signal, plus a little noise
            fwd = 0.05 * sig + 0.01 * float(rng.normal())
            rows.append((f"e{i}", d, sig, fwd))
    return pl.DataFrame(rows, schema=["e", "d", "sig", "fwd"], orient="row")


# --------------------------------------------------------------------------- #
# forward_return: backward shift, horizon edge, gap guard
# --------------------------------------------------------------------------- #
def test_forward_return_equals_manual_backward_shift() -> None:
    df = pl.DataFrame(
        {
            "e": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "d": [1, 2, 3, 4, 1, 2, 3, 4],
            "ret": [0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 4.0],
        }
    )
    out = fc.forward_return(df, entity="e", time="d", ret="ret", horizon=1)
    manual = df.sort(["e", "d"]).with_columns(
        pl.col("ret").shift(-1).over("e").alias("fwd_ret")
    )
    assert out.equals(manual)


def test_forward_return_null_at_horizon_edge() -> None:
    df = pl.DataFrame({"e": ["A", "A", "A"], "d": [1, 2, 3], "ret": [0.1, 0.2, 0.3]})
    out = fc.forward_return(df, entity="e", time="d", ret="ret", horizon=1)
    # last row per entity has no future -> null; earlier rows carry the next ret.
    assert out.sort("d").get_column("fwd_ret").to_list() == [0.2, 0.3, None]

    out2 = fc.forward_return(df, entity="e", time="d", ret="ret", horizon=2)
    # last 2 rows null.
    assert out2.sort("d").get_column("fwd_ret").to_list() == [0.3, None, None]


def test_forward_return_from_price() -> None:
    df = pl.DataFrame(
        {"e": ["A", "A", "A"], "d": [1, 2, 3], "px": [100.0, 110.0, 121.0]}
    )
    out = fc.forward_return(df, entity="e", time="d", price="px", horizon=1)
    fwd = out.sort("d").get_column("fwd_ret").to_list()
    assert fwd[0] == pytest.approx(0.10)
    assert fwd[1] == pytest.approx(0.10)
    assert fwd[2] is None


def test_forward_return_rejects_gaps() -> None:
    # Entity A skips period 3 -> a shift would jump the gap and leak.
    df = pl.DataFrame({"e": ["A", "A", "A"], "d": [1, 2, 4], "ret": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="irregular"):
        fc.forward_return(df, entity="e", time="d", ret="ret", horizon=1)
    # allow_gaps bypasses the guard.
    out = fc.forward_return(
        df, entity="e", time="d", ret="ret", horizon=1, allow_gaps=True
    )
    assert "fwd_ret" in out.columns


def test_forward_return_requires_exactly_one_source() -> None:
    df = pl.DataFrame({"e": ["A"], "d": [1], "ret": [0.1], "px": [1.0]})
    with pytest.raises(ValueError, match="exactly one"):
        fc.forward_return(df, entity="e", time="d")
    with pytest.raises(ValueError, match="exactly one"):
        fc.forward_return(df, entity="e", time="d", ret="ret", price="px")


# --------------------------------------------------------------------------- #
# ic / ic_summary
# --------------------------------------------------------------------------- #
def test_ic_perfect_correlation_is_one() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for d in range(1, 8):
        for i in range(30):
            s = float(rng.normal())
            rows.append((f"e{i}", d, s, s))  # fwd == signal
    df = pl.DataFrame(rows, schema=["e", "d", "sig", "fwd"], orient="row")
    ic = fc.ic(df, signal="sig", forward_return="fwd", time="d", method="spearman")
    assert ic.get_column("ic").mean() == pytest.approx(1.0)
    ic_p = fc.ic(df, signal="sig", forward_return="fwd", time="d", method="pearson")
    assert ic_p.get_column("ic").mean() == pytest.approx(1.0)


def test_ic_summary_fields(signal_panel: pl.DataFrame) -> None:
    ic = fc.ic(signal_panel, signal="sig", forward_return="fwd", time="d")
    s = fc.ic_summary(ic)
    assert s["mean_ic"] > 0.0
    assert s["n_periods"] == 12.0
    assert 0.0 <= s["hit_rate"] <= 1.0
    assert np.isfinite(s["icir"])
    # t_stat == icir * sqrt(T)
    assert s["t_stat"] == pytest.approx(s["icir"] * np.sqrt(12))


def test_ic_ignores_horizon_edge_nulls() -> None:
    # A real forward return (with edge nulls) must not break the correlation.
    df = pl.DataFrame(
        {
            "e": ["A", "A", "A", "B", "B", "B"],
            "d": [1, 2, 3, 1, 2, 3],
            "ret": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "sig": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    aligned = fc.forward_return(df, entity="e", time="d", ret="ret", horizon=1)
    ic = fc.ic(aligned, signal="sig", forward_return="fwd_ret", time="d")
    # dates 1 and 2 have both entities defined; date 3 is all-null and dropped.
    assert ic.height == 2


# --------------------------------------------------------------------------- #
# portfolio_sort
# --------------------------------------------------------------------------- #
def test_portfolio_sort_spread_positive_and_monotone(
    signal_panel: pl.DataFrame,
) -> None:
    res = fc.portfolio_sort(
        signal_panel, signal="sig", forward_return="fwd", time="d", q=5
    )
    assert res.mean_spread > 0.0
    assert res.monotonicity > 0.9
    assert res.n_periods == 12
    # per-bucket mean returns increase with bucket index.
    rets = res.bucket_means.sort("bucket").get_column("mean_ret").to_list()
    assert rets == sorted(rets)
    assert set(res.summary()) == {
        "mean_spread",
        "t_stat",
        "monotonicity",
        "n_periods",
        "q",
    }


# --------------------------------------------------------------------------- #
# orthogonalize
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["gram_schmidt", "qr"])
def test_orthogonalize_outputs_uncorrelated_per_date(method: str) -> None:
    rng = np.random.default_rng(3)
    rows = []
    for d in range(1, 5):
        for i in range(50):
            a = float(rng.normal())
            b = 0.8 * a + 0.4 * float(rng.normal())  # strongly correlated with a
            c = 0.5 * a - 0.5 * b + 0.3 * float(rng.normal())
            rows.append((f"e{i}", d, a, b, c))
    df = pl.DataFrame(rows, schema=["e", "d", "a", "b", "c"], orient="row")

    # inputs are correlated
    corr_in = np.corrcoef(df.get_column("a"), df.get_column("b"))[0, 1]
    assert abs(corr_in) > 0.5

    out = fc.orthogonalize(df, ["a", "b", "c"], time="d", method=method)
    names = ["a_orth", "b_orth", "c_orth"]
    for d in range(1, 5):
        sub = out.filter(pl.col("d") == d)
        mat = np.column_stack([sub.get_column(n).to_numpy() for n in names])
        corr = np.corrcoef(mat, rowvar=False)
        off = corr[~np.eye(3, dtype=bool)]
        assert np.abs(off).max() < 1e-8


def test_orthogonalize_by_subgroup() -> None:
    rng = np.random.default_rng(5)
    rows = []
    for d in (1, 2):
        for sec in ("X", "Y"):
            for i in range(30):
                a = float(rng.normal())
                b = 0.9 * a + 0.2 * float(rng.normal())
                rows.append((f"e{i}", d, sec, a, b))
    df = pl.DataFrame(rows, schema=["e", "d", "sec", "a", "b"], orient="row")
    out = fc.orthogonalize(df, ["a", "b"], by="sec", time="d", method="gram_schmidt")
    for d in (1, 2):
        for sec in ("X", "Y"):
            sub = out.filter((pl.col("d") == d) & (pl.col("sec") == sec))
            c = np.corrcoef(
                sub.get_column("a_orth").to_numpy(),
                sub.get_column("b_orth").to_numpy(),
            )[0, 1]
            assert abs(c) < 1e-8


# --------------------------------------------------------------------------- #
# .xs.standardize + rank normalize presets — documented ranges, per-date
# --------------------------------------------------------------------------- #
def test_xs_standardize_is_zero_mean_unit_std_per_date() -> None:
    df = pl.DataFrame(
        {"d": [1, 1, 1, 1, 2, 2, 2, 2], "v": [1.0, 2.0, 3.0, 4.0, 10, 20, 30, 40]}
    )
    out = df.xs.standardize("v", over="d", suffix="_z")
    for d in (1, 2):
        z = out.filter(pl.col("d") == d).get_column("v_z")
        assert z.mean() == pytest.approx(0.0, abs=1e-9)
        assert z.std(ddof=0) == pytest.approx(1.0)


def test_xs_standardize_per_date_differs_from_global() -> None:
    # date 2 lives on a very different scale; a per-date z must differ from a
    # single pooled/global z computed across both dates.
    df = pl.DataFrame(
        {"d": [1, 1, 1, 2, 2, 2], "v": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0]}
    )
    per_date = df.xs.standardize("v", over="d", suffix="_z").get_column("v_z")
    v = df.get_column("v").to_numpy()
    glob = (v - v.mean()) / v.std()
    assert not np.allclose(per_date.to_numpy(), glob)
    # per-date is scale-invariant: both dates map to the same standardized shape.
    assert per_date.to_list()[:3] == pytest.approx(per_date.to_list()[3:])


def test_xs_standardize_winsor_clips_per_date() -> None:
    df = pl.DataFrame({"d": [1] * 5, "v": [1.0, 2.0, 3.0, 4.0, 1000.0]})
    no_w = df.xs.standardize("v", over="d", suffix="_z").get_column("v_z")
    with_w = df.xs.standardize("v", over="d", winsor=0.2, suffix="_z").get_column("v_z")
    # winsorizing pulls the extreme value in, shrinking the top z-score.
    assert with_w.max() < no_w.max()


def test_rank_normalize_presets_ranges() -> None:
    df = pl.DataFrame({"d": [1] * 5, "v": [10.0, 20.0, 30.0, 40.0, 50.0]})

    unit = df.with_columns(
        pl.col("v").xs.rank(normalize="unit").over("d").alias("r")
    ).get_column("r")
    assert unit.min() == pytest.approx(-1.0)
    assert unit.max() == pytest.approx(1.0)

    centered = df.with_columns(
        pl.col("v").xs.rank(normalize="centered").over("d").alias("r")
    ).get_column("r")
    assert centered.mean() == pytest.approx(0.0, abs=1e-9)
    assert centered.min() > -0.5 and centered.max() < 0.5

    uniform = df.with_columns(
        pl.col("v").xs.rank(normalize="uniform").over("d").alias("r")
    ).get_column("r")
    assert uniform.min() > 0.0 and uniform.max() == pytest.approx(1.0)

    # bool back-compat: True == (0, 1) via rank / (N + 1).
    uplus = df.with_columns(
        pl.col("v").xs.rank(normalize=True).over("d").alias("r")
    ).get_column("r")
    assert uplus.min() > 0.0 and uplus.max() < 1.0
    assert uplus.to_list() == pytest.approx([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])

    # False (default) is the raw rank, unchanged behaviour.
    raw = df.with_columns(pl.col("v").xs.rank().over("d").alias("r")).get_column("r")
    assert raw.to_list() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_rank_unit_guards_single_entity_date() -> None:
    # N == 1 cross-section -> null (no divide-by-zero), matching zscore.
    df = pl.DataFrame({"d": [1, 2, 2], "v": [5.0, 1.0, 2.0]})
    out = df.with_columns(pl.col("v").xs.rank(normalize="unit").over("d").alias("r"))
    r = out.sort(["d", "v"]).get_column("r").to_list()
    assert r[0] is None  # the lone entity on date 1


def test_rank_invalid_normalize_raises() -> None:
    with pytest.raises(ValueError, match="normalize"):
        pl.DataFrame({"v": [1.0, 2.0]}).with_columns(
            pl.col("v").xs.rank(normalize="nonsense")
        )


# --------------------------------------------------------------------------- #
# registry provenance
# --------------------------------------------------------------------------- #
def test_factor_specs_registered_leak_safe() -> None:
    for name in ("forward_return", "ic", "portfolio_sort", "orthogonalize"):
        spec = registry.get(name)
        assert spec.namespace == "factor"
        assert spec.leakage_safe is True
    assert registry.get("standardize").namespace == "xs"
