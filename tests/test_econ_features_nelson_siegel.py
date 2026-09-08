"""Acceptance tests for the Nelson-Siegel(-Svensson) curve factors.

The curve is generated from *known* betas and then recovered, which is the
strongest possible form of the plan's "recovers its known value" criterion.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_features.core import PanelFrame
from polars_features.econ.features import (
    NelsonSiegel,
    nelson_siegel_factors,
    nelson_siegel_fit,
    nelson_siegel_loadings,
)

MATURITIES = np.array([3.0, 6.0, 12.0, 24.0, 36.0, 60.0, 84.0, 120.0])
LAM = 0.0609


def _curve(b0: float, b1: float, b2: float, lam: float = LAM) -> np.ndarray:
    load = nelson_siegel_loadings(MATURITIES, lam)
    return load @ np.array([b0, b1, b2])


# --------------------------------------------------------------------------- #
# Exact recovery
# --------------------------------------------------------------------------- #
def test_fit_recovers_known_betas_exactly_without_noise() -> None:
    fit = nelson_siegel_fit(MATURITIES, _curve(5.0, -2.0, 1.5), lam=LAM)
    assert fit.level == pytest.approx(5.0, abs=1e-9)
    assert fit.slope == pytest.approx(2.0, abs=1e-9)  # slope == -b1
    assert fit.curvature == pytest.approx(1.5, abs=1e-9)
    assert fit.rmse == pytest.approx(0.0, abs=1e-9)
    assert fit.nobs == MATURITIES.size


def test_fit_recovers_known_betas_under_small_noise() -> None:
    rng = np.random.default_rng(0)
    y = _curve(4.0, -1.0, 2.0) + rng.standard_normal(MATURITIES.size) * 0.01
    fit = nelson_siegel_fit(MATURITIES, y, lam=LAM)
    assert fit.level == pytest.approx(4.0, abs=0.1)
    assert fit.slope == pytest.approx(1.0, abs=0.1)
    assert fit.curvature == pytest.approx(2.0, abs=0.3)


def test_level_is_the_long_rate_and_slope_the_spread() -> None:
    """Diebold-Li interpretation: ``b0`` is the asymptotic long rate and the
    reported slope (``-b1``) is the long-minus-short spread."""
    y = _curve(6.0, -3.0, 0.0)
    fit = nelson_siegel_fit(MATURITIES, y, lam=LAM)
    long_end = y[-1]
    short_end = y[0]
    assert fit.level == pytest.approx(6.0, abs=1e-9)
    # The level sits at the long end of the curve, not the short end.
    assert abs(fit.level - long_end) < abs(fit.level - short_end)
    # An upward-sloping curve gives a positive slope of the right magnitude.
    assert fit.slope > 0
    assert fit.slope == pytest.approx(long_end - short_end, abs=0.7)


def test_svensson_adds_a_fourth_factor() -> None:
    load = nelson_siegel_loadings(MATURITIES, LAM, lam2=2 * LAM)
    assert load.shape == (MATURITIES.size, 4)
    y = load @ np.array([5.0, -2.0, 1.0, 0.75])
    fit = nelson_siegel_fit(MATURITIES, y, lam=LAM, lam2=2 * LAM)
    assert fit.beta[3] == pytest.approx(0.75, abs=1e-8)


def test_loadings_reject_bad_inputs() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        nelson_siegel_loadings(np.array([0.0, 1.0]), LAM)
    with pytest.raises(ValueError, match="`lam` must be positive"):
        nelson_siegel_loadings(MATURITIES, 0.0)
    with pytest.raises(ValueError, match="same length"):
        nelson_siegel_fit(MATURITIES, np.array([1.0]))


def test_fit_degrades_gracefully_on_too_few_maturities() -> None:
    fit = nelson_siegel_fit(np.array([12.0, 24.0]), np.array([1.0, 2.0]))
    assert np.isnan(fit.level)
    assert fit.nobs == 2


# --------------------------------------------------------------------------- #
# Per-date panel surface
# --------------------------------------------------------------------------- #
def _curve_panel(n_dates: int = 60) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(1)
    levels = 4.0 + np.cumsum(rng.standard_normal(n_dates) * 0.05)
    rows = []
    for t in range(n_dates):
        y = _curve(levels[t], -2.0 + 0.01 * t, 1.0)
        for mat, yield_ in zip(MATURITIES, y):
            rows.append({"maturity": float(mat), "date": t, "yield": float(yield_)})
    return pl.DataFrame(rows), levels


def test_factors_are_estimated_per_date() -> None:
    df, levels = _curve_panel()
    out = nelson_siegel_factors(
        df, date="date", maturity="maturity", yield_col="yield", lam=LAM
    )
    assert out.height == 60
    np.testing.assert_allclose(out.get_column("ns_level").to_numpy(), levels, atol=1e-8)
    # Slope was constructed to drift; the fitted slope must track it.
    slope = out.get_column("ns_slope").to_numpy()
    assert slope[0] > slope[-1]


def test_per_date_fit_is_unaffected_by_other_dates() -> None:
    """Contemporaneous by construction: dropping later dates must not move an
    earlier date's factors by one ulp."""
    df, _ = _curve_panel()
    full = nelson_siegel_factors(
        df, date="date", maturity="maturity", yield_col="yield"
    )
    truncated = nelson_siegel_factors(
        df.filter(pl.col("date") < 30),
        date="date",
        maturity="maturity",
        yield_col="yield",
    )
    np.testing.assert_array_equal(
        full.head(30).get_column("ns_level").to_numpy(),
        truncated.get_column("ns_level").to_numpy(),
    )


# --------------------------------------------------------------------------- #
# Transformer
# --------------------------------------------------------------------------- #
def test_transformer_broadcasts_factors_onto_every_row_of_a_date() -> None:
    df, levels = _curve_panel(n_dates=20)
    panel = PanelFrame(df, entity="maturity", time="date")
    tr = NelsonSiegel("yield").fit(panel)
    out = tr.transform(panel).collect().sort(["date", "maturity"])
    assert {"ns_level", "ns_slope", "ns_curvature"} <= set(out.columns)
    # Every maturity row of a date carries that date's single set of factors.
    for t in range(20):
        vals = out.filter(pl.col("date") == t).get_column("ns_level").unique()
        assert vals.len() == 1
        assert vals[0] == pytest.approx(levels[t], abs=1e-8)


def test_transformer_freezes_lambda_learned_on_train() -> None:
    df, _ = _curve_panel(n_dates=40)
    train = PanelFrame(df.filter(pl.col("date") < 20), entity="maturity", time="date")
    test = PanelFrame(df.filter(pl.col("date") >= 20), entity="maturity", time="date")
    tr = NelsonSiegel("yield", fit_lambda=True).fit(train)
    learned = tr.lambda_
    # The data were generated at LAM, so the grid search must land nearby.
    assert learned == pytest.approx(LAM, rel=0.35)
    tr.transform(test)
    assert tr.lambda_ == learned  # transform never re-fits


def test_transformer_rejects_a_missing_yield_column() -> None:
    df, _ = _curve_panel(n_dates=5)
    panel = PanelFrame(df, entity="maturity", time="date")
    with pytest.raises(ValueError, match="not found in panel"):
        NelsonSiegel("nope").fit(panel)
