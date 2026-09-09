"""Parity between the two cross-sectional neutralization surfaces.

``panelary.transform.Neutralize`` (a :class:`PanelTransformer`) and
``pl.col(...).xs.neutralize(...)`` (the expression namespace) must compute the
*same* per-date OLS residual. They used to hold two independent copies of that
arithmetic — the same regression written twice, free to drift. This module

1. pins them to bit-identical outputs across numeric / categorical factors,
   intercept on and off, nulls, NaN/inf, and the rank-deficient edge case, and
2. asserts structurally that both now route through the single shared kernel
   :mod:`panelary.namespaces._neutralize_kernel`, so the duplication cannot be
   quietly reintroduced.

It also re-checks the ``leakage_safe`` contract at this seam: a cross-sectional
residual for date ``t`` must depend on date ``t``'s rows alone.
"""

from __future__ import annotations

import inspect

import numpy as np
import polars as pl
import pytest

import panelary.namespaces  # noqa: F401  (registers the `.xs` namespace)
import panelary.namespaces.xs as xs_mod
import panelary.transform.neutralize as tf_mod
from panelary.core import PanelFrame
from panelary.namespaces import _neutralize_kernel as kernel
from panelary.transform import Neutralize

SEED = 20240917


# --------------------------------------------------------------------------
# Fixtures / case matrix
# --------------------------------------------------------------------------
def _panel(seed: int = SEED, n_dates: int = 4, n_entities: int = 9) -> pl.DataFrame:
    """A small, fully deterministic panel with mixed factor dtypes."""
    rng = np.random.default_rng(seed)
    n = n_dates * n_entities
    return pl.DataFrame(
        {
            "d": np.repeat(np.arange(n_dates), n_entities),
            "e": [f"e{i}" for i in range(n_entities)] * n_dates,
            "ret": rng.normal(size=n),
            "beta": rng.normal(size=n),
            "size": rng.normal(size=n),
            "sector": [["x", "y", "z"][i % 3] for i in range(n)],
            "flag": [i % 2 == 0 for i in range(n)],
            "icount": np.arange(n),
        }
    )


def _with_nulls(df: pl.DataFrame, col: str, every: int) -> pl.DataFrame:
    idx = pl.int_range(0, pl.len()).over(pl.lit(1))
    return df.with_columns(
        pl.when(idx % every == 0).then(None).otherwise(pl.col(col)).alias(col)
    )


# (case id, frame, factors)
def _cases() -> list[tuple[str, pl.DataFrame, list[str]]]:
    base = _panel()
    return [
        ("numeric-one", base, ["beta"]),
        ("numeric-two", base, ["beta", "size"]),
        ("categorical", base, ["sector"]),
        ("mixed", base, ["beta", "sector"]),
        ("mixed-reordered", base, ["sector", "beta"]),
        ("boolean", base, ["flag"]),
        ("integer", base, ["icount"]),
        ("everything", base, ["beta", "size", "sector", "flag", "icount"]),
        ("float32", base.with_columns(pl.col("beta").cast(pl.Float32)), ["beta"]),
        (
            "categorical-dtype",
            base.with_columns(pl.col("sector").cast(pl.Categorical)),
            ["sector"],
        ),
        (
            "enum-dtype",
            base.with_columns(pl.col("sector").cast(pl.Enum(["x", "y", "z"]))),
            ["sector"],
        ),
        ("null-target", _with_nulls(base, "ret", 5), ["beta", "sector"]),
        ("null-numeric-factor", _with_nulls(base, "beta", 4), ["beta", "size"]),
        ("null-categorical-factor", _with_nulls(base, "sector", 3), ["sector", "beta"]),
        (
            "nan-target",
            base.with_columns(
                pl.when(pl.col("e") == "e2")
                .then(float("nan"))
                .otherwise(pl.col("ret"))
                .alias("ret")
            ),
            ["beta"],
        ),
        (
            "inf-factor",
            base.with_columns(
                pl.when(pl.col("e") == "e3")
                .then(float("inf"))
                .otherwise(pl.col("size"))
                .alias("size")
            ),
            ["size"],
        ),
        (
            "collinear",
            base.with_columns((pl.col("beta") * 2.0).alias("size")),
            ["beta", "size"],
        ),
        ("constant-factor", base.with_columns(pl.lit(1.0).alias("beta")), ["beta"]),
    ]


CASES = _cases()
CASE_IDS = [c[0] for c in CASES]


def _transformer_residual(
    df: pl.DataFrame, factors: list[str], *, add_intercept: bool
) -> np.ndarray:
    panel = PanelFrame(df, entity="e", time="d")
    out = (
        Neutralize("ret", factors=factors, add_intercept=add_intercept)
        .fit_transform(panel)
        .collect()
    )
    return out.get_column("ret_neutral").to_numpy().astype(np.float64)


def _namespace_residual(
    df: pl.DataFrame, factors: list[str], *, add_intercept: bool
) -> np.ndarray:
    out = df.with_columns(
        pl.col("ret")
        .xs.neutralize(factors, add_intercept=add_intercept)
        .over("d")
        .alias("n")
    )
    return out.get_column("n").to_numpy().astype(np.float64)


# --------------------------------------------------------------------------
# 1. The two surfaces agree, bit for bit
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "add_intercept", [True, False], ids=["intercept", "no-intercept"]
)
@pytest.mark.parametrize(("case", "df", "factors"), CASES, ids=CASE_IDS)
def test_transformer_matches_namespace_bitwise(
    case: str, df: pl.DataFrame, factors: list[str], add_intercept: bool
) -> None:
    """`Neutralize` and `.xs.neutralize` must produce identical residuals."""
    a = _transformer_residual(df, factors, add_intercept=add_intercept)
    b = _namespace_residual(df, factors, add_intercept=add_intercept)
    assert a.shape == b.shape
    assert np.array_equal(a, b, equal_nan=True), (
        f"{case}: transformer and namespace diverged\ntransformer={a}\nnamespace  ={b}"
    )


@pytest.mark.parametrize(
    "add_intercept", [True, False], ids=["intercept", "no-intercept"]
)
def test_rank_deficient_cross_section_is_null_on_both_surfaces(
    add_intercept: bool,
) -> None:
    """Fewer usable rows than parameters -> nulls, identically on both paths.

    One-hot encoding happens *within* a cross-section, so the parameter count is
    per-date too: date 0 has 3 rows but 3 sector dummies plus ``beta`` (4
    parameters, 5 with an intercept) and is rank deficient either way; date 1
    has 6 rows for the same design and is fittable either way.
    """
    df = pl.DataFrame(
        {
            "d": [0, 0, 0, 1, 1, 1, 1, 1, 1],
            "e": ["a", "b", "c", "a", "b", "c", "f", "g", "h"],
            "ret": [1.0, 2.0, 3.0, 0.5, -0.5, 1.5, -1.5, 0.25, 2.5],
            "beta": [1.0, 0.5, 1.5, 1.0, 0.5, 1.5, 0.75, 1.25, 0.9],
            "sector": ["x", "y", "z", "x", "y", "z", "x", "y", "z"],
        }
    )
    a = _transformer_residual(df, ["beta", "sector"], add_intercept=add_intercept)
    b = _namespace_residual(df, ["beta", "sector"], add_intercept=add_intercept)
    assert np.array_equal(a, b, equal_nan=True)
    assert np.isnan(a[:3]).all(), "rank-deficient date must yield nulls"
    assert not np.isnan(a[3:]).any(), "fittable date must yield residuals"


def test_all_null_target_cross_section_is_null_on_both_surfaces() -> None:
    df = pl.DataFrame(
        {
            "d": [0, 0, 0],
            "e": ["a", "b", "c"],
            "ret": pl.Series([None, None, None], dtype=pl.Float64),
            "beta": [1.0, 2.0, 3.0],
        }
    )
    a = _transformer_residual(df, ["beta"], add_intercept=True)
    b = _namespace_residual(df, ["beta"], add_intercept=True)
    assert np.isnan(a).all()
    assert np.array_equal(a, b, equal_nan=True)


# --------------------------------------------------------------------------
# 2. There is exactly ONE implementation
# --------------------------------------------------------------------------
def test_both_call_sites_use_the_same_kernel_object() -> None:
    """Identity, not merely equality: one function object, two importers."""
    assert tf_mod.cross_section_residuals is kernel.cross_section_residuals
    assert xs_mod.cross_section_residuals is kernel.cross_section_residuals
    assert tf_mod.build_design is kernel.build_design


def test_neither_call_site_reimplements_the_least_squares_solve() -> None:
    """Guard against the duplication being reintroduced by a future edit.

    Prose may still *mention* ``numpy.linalg.lstsq``; what must not come back is
    a second call to it (or a second per-date one-hot encoding).
    """
    for mod in (tf_mod, xs_mod):
        src = inspect.getsource(mod)
        assert "np.linalg.lstsq(" not in src, (
            f"{mod.__name__} reimplements the OLS solve; it must delegate to "
            "panelary.namespaces._neutralize_kernel instead."
        )
        assert ".to_dummies(" not in src, (
            f"{mod.__name__} reimplements the per-date one-hot encoding; it must"
            "delegate to panelary.namespaces._neutralize_kernel instead."
        )


def test_kernel_is_a_leaf_module() -> None:
    """The kernel may import numpy/polars only -- nothing from panelary.

    That is what lets ``namespaces`` (which must not import ``transform``) and
    ``transform`` share it without an import cycle.
    """
    src = inspect.getsource(kernel)
    assert "import panelary" not in src
    assert "from panelary" not in src


# --------------------------------------------------------------------------
# 3. The leakage contract at this seam
# --------------------------------------------------------------------------
def test_neutralize_declares_the_safety_contract() -> None:
    assert Neutralize.panel_safe is True
    assert Neutralize.leakage_safe is True


@pytest.mark.parametrize(
    "add_intercept", [True, False], ids=["intercept", "no-intercept"]
)
def test_residuals_do_not_depend_on_other_dates(add_intercept: bool) -> None:
    """Prefix invariance: appending future dates cannot change earlier rows.

    If any information crossed the ``time_col`` boundary -- a pooled fit, a
    global mean, a shared one-hot vocabulary -- truncating the panel would move
    the surviving residuals. It must not.
    """
    full = _panel(n_dates=5)
    prefix = full.filter(pl.col("d") < 3)
    factors = ["beta", "sector"]

    for residual_of in (_transformer_residual, _namespace_residual):
        r_full = residual_of(full, factors, add_intercept=add_intercept)
        r_prefix = residual_of(prefix, factors, add_intercept=add_intercept)
        n = prefix.height
        assert np.allclose(r_full[:n], r_prefix, equal_nan=True), (
            f"{residual_of.__name__}: later dates changed earlier residuals"
        )


def test_residual_is_orthogonal_to_factors_within_each_date() -> None:
    """The defining property of the OLS residual, checked per cross-section."""
    df = _panel()
    out = df.with_columns(
        pl.col("ret").xs.neutralize(["beta", "size"]).over("d").alias("n")
    )
    for (_date,), block in out.group_by("d", maintain_order=True):
        resid = block.get_column("n").to_numpy()
        for f in ("beta", "size"):
            x = block.get_column(f).to_numpy()
            assert abs(float(np.dot(resid - resid.mean(), x - x.mean()))) < 1e-9


def test_kernel_rejects_nothing_but_returns_nan_for_undefined_rows() -> None:
    """Direct kernel check: null/NaN target or factor -> NaN, others fitted."""
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, None, 4.0, 5.0],
            "x": [1.0, 2.0, 3.0, None, 5.0],
        }
    )
    r = kernel.cross_section_residuals(df, "y", ["x"], add_intercept=True)
    assert r.dtype == np.float64
    assert np.isnan(r[2]) and np.isnan(r[3])
    assert not np.isnan(r[[0, 1, 4]]).any()
    # y == x on the valid rows, so with an intercept the residual is ~0.
    assert np.allclose(r[[0, 1, 4]], 0.0, atol=1e-12)
