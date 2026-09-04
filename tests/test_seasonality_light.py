"""Light-import tests for the seasonality subpackage.

These verify that the heavy optional ``holidays`` dependency is not pulled in at
import time, that the numpy/polars-only Fourier path works without it, and that
the holiday-calendar path still works when ``holidays`` is installed.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import polars as pl
import pytest

from polars_features._deps import have


def test_importing_seasonality_does_not_import_holidays():
    """Importing the seasonality subpackage must not eagerly load ``holidays``.

    Run in a fresh interpreter so a prior import in this process can't mask a
    leak.
    """
    code = (
        "import polars_features.seasonality, sys; "
        "assert 'holidays' not in sys.modules, 'holidays imported eagerly'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_importing_package_does_not_import_holidays():
    """``import polars_features`` (reaches seasonality via preprocessing) stays clean."""
    code = (
        "import polars_features, sys; "
        "assert 'holidays' not in sys.modules, 'holidays imported eagerly'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_add_fourier_terms_works_without_holidays():
    """The Fourier path is numpy/polars only and must work regardless of holidays."""
    from polars_features.seasonality import add_fourier_terms

    df = pl.DataFrame(
        {
            "entity": ["a"] * 12,
            "time": pl.datetime_range(
                pl.datetime(2020, 1, 1),
                pl.datetime(2020, 12, 1),
                interval="1mo",
                eager=True,
            ),
            "value": list(range(12)),
        }
    )

    transf = add_fourier_terms(sp=12, K=3)
    out = transf(df)
    if isinstance(out, pl.LazyFrame):
        out = out.collect()

    for name in ("cos_12_1", "sin_12_1", "cos_12_3", "sin_12_3"):
        assert name in out.columns
    assert out.height == df.height


@pytest.mark.skipif(not have("holidays"), reason="holidays not installed")
def test_add_holiday_effects_works_when_holidays_installed():
    """When ``holidays`` is installed the calendar path still works end-to-end."""
    from polars_features.seasonality import add_holiday_effects

    df = pl.DataFrame(
        {
            "entity": ["a"] * 5,
            "time": pl.date_range(
                pl.date(2020, 12, 23),
                pl.date(2020, 12, 27),
                interval="1d",
                eager=True,
            ),
            "value": list(range(5)),
        }
    )

    transf = add_holiday_effects(country_codes=["US"])
    out = transf(df)
    if isinstance(out, pl.LazyFrame):
        out = out.collect()

    assert "holiday__US" in out.columns
    assert out.height == df.height


def test_add_holiday_effects_raises_helpful_error_without_holidays(monkeypatch):
    """When ``holidays`` is absent, using the calendar path raises a guided ImportError."""
    from polars_features.seasonality import add_holiday_effects

    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "holidays" or name.startswith("holidays."):
            raise ImportError("No module named 'holidays'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(
        "polars_features._deps.importlib.import_module", fake_import_module
    )

    df = pl.DataFrame(
        {
            "entity": ["a"] * 3,
            "time": pl.date_range(
                pl.date(2020, 12, 24),
                pl.date(2020, 12, 26),
                interval="1d",
                eager=True,
            ),
            "value": [0, 1, 2],
        }
    )

    transf = add_holiday_effects(country_codes=["US"])
    with pytest.raises(ImportError) as excinfo:
        out = transf(df)
        if isinstance(out, pl.LazyFrame):
            out.collect()

    msg = str(excinfo.value)
    assert "holidays" in msg
    assert "polars-features[seasonality]" in msg
