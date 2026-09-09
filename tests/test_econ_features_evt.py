"""Acceptance tests for the EVT tail block.

The headline criterion from the plan: the **Hill index recovers a known Pareto
tail**. The GPD fit is likewise checked against a sample drawn from a GPD with
known ``(xi, sigma)``, and the POT VaR against the analytic quantile.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.econ.features import (
    evt_features,
    gpd_fit,
    hill_index,
    pot_var_es,
)


def _pareto(alpha: float, n: int, seed: int = 0) -> np.ndarray:
    """Draw from a Pareto(alpha) with unit scale: ``P(X > x) = x**-alpha``."""
    rng = np.random.default_rng(seed)
    return (1.0 - rng.random(n)) ** (-1.0 / alpha)


def _gpd_sample(xi: float, sigma: float, n: int, seed: int = 0) -> np.ndarray:
    """Inverse-CDF draw from GPD(xi, sigma)."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    if abs(xi) < 1e-12:
        return -sigma * np.log(1.0 - u)
    return sigma / xi * ((1.0 - u) ** (-xi) - 1.0)


# --------------------------------------------------------------------------- #
# Hill
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [2.0, 3.0, 4.0])
def test_hill_recovers_a_known_pareto_tail_index(alpha: float) -> None:
    estimates = [
        hill_index(_pareto(alpha, 20_000, seed=s), tail="upper") for s in range(5)
    ]
    assert float(np.mean(estimates)) == pytest.approx(alpha, rel=0.10)


def test_hill_reads_the_lower_tail_by_default() -> None:
    heavy_losses = -_pareto(2.5, 20_000, seed=1)
    assert hill_index(heavy_losses, tail="lower") == pytest.approx(2.5, rel=0.12)
    # Looking at the (thin, bounded) upper tail of the same sample gives a very
    # different answer, so the orientation flag genuinely matters.
    assert hill_index(heavy_losses, tail="upper") != pytest.approx(2.5, rel=0.12)


def test_hill_is_monotone_in_tail_heaviness() -> None:
    heavy = hill_index(_pareto(1.5, 20_000, seed=2), tail="upper")
    light = hill_index(_pareto(5.0, 20_000, seed=2), tail="upper")
    assert heavy < light  # alpha is smaller for a heavier tail


def test_hill_returns_nan_on_a_degenerate_sample() -> None:
    assert np.isnan(hill_index(np.ones(5)))
    assert np.isnan(hill_index(np.ones(200)))


def test_hill_rejects_an_unknown_tail() -> None:
    with pytest.raises(ValueError, match="'upper' or 'lower'"):
        hill_index(np.arange(100.0), tail="sideways")


# --------------------------------------------------------------------------- #
# GPD / POT
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("xi", "sigma"), [(0.2, 1.0), (0.3, 2.0), (0.1, 0.5)])
def test_gpd_pwm_recovers_known_shape_and_scale(xi: float, sigma: float) -> None:
    shapes, scales = [], []
    for seed in range(6):
        fit = gpd_fit(
            _gpd_sample(xi, sigma, 20_000, seed=seed),
            threshold=0.0,
            tail="upper",
        )
        shapes.append(fit.xi)
        scales.append(fit.sigma)
    assert float(np.mean(shapes)) == pytest.approx(xi, abs=0.05)
    assert float(np.mean(scales)) == pytest.approx(sigma, rel=0.10)


def test_gpd_fit_reports_the_threshold_and_exceedance_count() -> None:
    sample = _pareto(3.0, 5_000, seed=3)
    fit = gpd_fit(sample, threshold_quantile=0.90, tail="upper")
    assert fit.nobs == 5_000
    assert fit.n_exceed == pytest.approx(500, rel=0.05)
    assert fit.threshold > 0


def test_gpd_fit_degrades_on_a_tiny_sample() -> None:
    fit = gpd_fit(np.arange(10.0))
    assert np.isnan(fit.xi)
    assert fit.n_exceed == 0


def test_pot_var_recovers_the_analytic_pareto_quantile() -> None:
    """For Pareto(alpha), ``VaR_q = (1 - q) ** (-1 / alpha)``."""
    alpha = 3.0
    q = 0.99
    var, es, fit = pot_var_es(
        _pareto(alpha, 40_000, seed=4), q=q, threshold_quantile=0.90, tail="upper"
    )
    analytic = (1.0 - q) ** (-1.0 / alpha)
    assert var == pytest.approx(analytic, rel=0.08)
    # A Pareto tail has xi = 1 / alpha.
    assert fit.xi == pytest.approx(1.0 / alpha, abs=0.06)
    # ES always exceeds VaR for a heavy tail.
    assert es > var


def test_pwm_shape_saturates_below_one_on_an_infinite_mean_tail() -> None:
    """Documented limitation: PWM's shape estimate is bounded above by 1.

    A Pareto with ``alpha = 0.7`` has ``xi = 1 / alpha = 1.43`` and no finite
    mean, but PWM cannot report a shape above 1 -- it saturates. The Hill index,
    which has no such bound, still reads the tail correctly, which is why the
    docstring points heavy-tail users at it.
    """
    sample = _pareto(0.7, 20_000, seed=5)
    _, _, fit = pot_var_es(sample, tail="upper")
    assert 0.9 < fit.xi < 1.0
    assert hill_index(sample, tail="upper") == pytest.approx(0.7, rel=0.15)


def test_pot_var_es_rejects_a_bad_confidence_level() -> None:
    with pytest.raises(ValueError, match="must lie in"):
        pot_var_es(np.arange(100.0), q=1.5)


# --------------------------------------------------------------------------- #
# Panel surface
# --------------------------------------------------------------------------- #
def _tail_panel(n: int = 800) -> pl.DataFrame:
    frames = []
    for i, (ent, alpha) in enumerate((("heavy", 2.0), ("light", 6.0))):
        frames.append(
            pl.DataFrame(
                {
                    "entity": [ent] * n,
                    "time": np.arange(n, dtype=np.int64),
                    "ret": -_pareto(alpha, n, seed=10 + i) / 100.0,
                }
            )
        )
    return pl.concat(frames)


def test_evt_features_emit_the_expected_columns_and_warm_up() -> None:
    out = evt_features(_tail_panel(), "ret", entity="entity", time="time", window=252)
    for stem in (
        "hill_alpha",
        "gpd_xi",
        "gpd_sigma",
        "evt_var",
        "evt_es",
        "evt_n_exceed",
    ):
        assert f"ret_{stem}" in out.columns
    col = out.get_column("ret_hill_alpha")
    assert col.head(251).is_null().all()
    assert col.slice(251, 100).is_not_null().all()


def test_evt_features_separate_a_heavy_tail_from_a_light_one() -> None:
    out = evt_features(_tail_panel(), "ret", entity="entity", time="time", window=400)
    med = out.group_by("entity").agg(pl.col("ret_hill_alpha").median()).to_dicts()
    by_entity = {r["entity"]: r["ret_hill_alpha"] for r in med}
    # A smaller Hill alpha means a heavier tail.
    assert by_entity["heavy"] < by_entity["light"]
