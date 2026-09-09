"""Leakage-guard tests for ``preprocessing.impute``.

The ``bfill`` and ``interpolate`` imputation methods read *future* rows to fill a
missing value, so they leak look-ahead information and pass silently through
cross-validation. They must emit a :class:`~panelary.preprocessing.LeakageWarning`
by default, and be silent when the caller explicitly opts in with
``allow_leaky=True``. Leak-safe methods must never warn.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from panelary.preprocessing import LeakageWarning, impute


def _panel_with_gaps() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "entity": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "time": [0, 1, 2, 3, 0, 1, 2, 3],
            "px": [1.0, None, None, 4.0, 10.0, None, 12.0, None],
        }
    ).lazy()


@pytest.mark.parametrize("method", ["bfill", "interpolate"])
def test_leaky_methods_warn_by_default(method: str) -> None:
    # The @transformer factory defers execution, so the guard fires when the
    # leaky op actually runs (pipe + collect).
    with pytest.warns(LeakageWarning, match="future"):
        _panel_with_gaps().pipe(impute(method)).collect()


@pytest.mark.parametrize("method", ["bfill", "interpolate"])
def test_leaky_methods_silent_with_allow_leaky(method: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        # Full construct + execute path must be silent.
        out = _panel_with_gaps().pipe(impute(method, allow_leaky=True)).collect()
    assert out.height == 8


@pytest.mark.parametrize("method", ["mean", "median", "fill", "ffill"])
def test_safe_methods_never_warn(method: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _panel_with_gaps().pipe(impute(method)).collect()
    assert out.height == 8


def test_constant_fill_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _panel_with_gaps().pipe(impute(0.0)).collect()
    assert out.height == 8


def test_allow_leaky_is_keyword_only() -> None:
    # Positional passing of allow_leaky must fail (keyword-only guard). The
    # @transformer factory defers, so the TypeError surfaces at execution.
    with pytest.raises(TypeError):
        _panel_with_gaps().pipe(impute("bfill", True)).collect()  # type: ignore[misc]


def test_leaky_still_produces_output_when_acknowledged() -> None:
    # bfill fills backward: A -> [1, 4, 4, 4]; the leak is real, which is why it
    # is gated, but the operation itself must remain functional.
    transformer = impute("bfill", allow_leaky=True)
    out = _panel_with_gaps().pipe(transformer).collect().sort("entity", "time")
    a = out.filter(pl.col("entity") == "A").get_column("px").to_list()
    assert a[:2] == [1.0, 4.0]  # the null at t=1 was filled from the future (t=3)
