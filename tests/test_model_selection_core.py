"""Sanity / monotonicity tests for de Prado performance & overfitting metrics.

Covers ``deflated_sharpe_ratio`` and ``probability_of_backtest_overfitting``
from the pure-Python core. No compiled Rust extension required.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polars_features.core.model_selection import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio
# --------------------------------------------------------------------------- #
def test_dsr_in_unit_interval():
    dsr = deflated_sharpe_ratio(0.1, n_trials=10, n_observations=250)
    assert 0.0 <= dsr <= 1.0


@settings(max_examples=50, deadline=None)
@given(
    sr=st.floats(min_value=0.02, max_value=0.5),
    n_obs=st.integers(min_value=50, max_value=2000),
    n_low=st.integers(min_value=1, max_value=20),
    extra=st.integers(min_value=1, max_value=200),
)
def test_dsr_decreases_with_more_trials(sr, n_obs, n_low, extra):
    """More trials -> higher selection bar -> DSR should not increase."""
    n_high = n_low + extra
    dsr_low = deflated_sharpe_ratio(sr, n_trials=n_low, n_observations=n_obs)
    dsr_high = deflated_sharpe_ratio(sr, n_trials=n_high, n_observations=n_obs)
    assert dsr_high <= dsr_low + 1e-9


@settings(max_examples=50, deadline=None)
@given(
    sr=st.floats(min_value=0.05, max_value=0.5),
    n_trials=st.integers(min_value=2, max_value=100),
    n_low=st.integers(min_value=30, max_value=500),
    extra=st.integers(min_value=50, max_value=2000),
)
def test_dsr_increases_with_more_observations(sr, n_trials, n_low, extra):
    """More observations -> tighter Sharpe estimator -> DSR should not
    decrease (a positive observed Sharpe becomes more credible)."""
    n_high = n_low + extra
    dsr_low = deflated_sharpe_ratio(sr, n_trials=n_trials, n_observations=n_low)
    dsr_high = deflated_sharpe_ratio(sr, n_trials=n_trials, n_observations=n_high)
    assert dsr_high >= dsr_low - 1e-9


def test_dsr_single_trial_benchmark_is_zero():
    # With one trial the benchmark Sharpe is 0; DSR > 0.5 for a positive SR.
    dsr = deflated_sharpe_ratio(0.2, n_trials=1, n_observations=500)
    assert dsr > 0.5


# --------------------------------------------------------------------------- #
# Probability of Backtest Overfitting
# --------------------------------------------------------------------------- #
def test_pbo_in_unit_interval():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((128, 8))
    pbo = probability_of_backtest_overfitting(M, n_partitions=8)
    assert 0.0 <= pbo <= 1.0


def test_pbo_high_for_pure_noise():
    """A matrix of i.i.d. noise strategies -> selection is luck -> PBO should
    be substantial (around / above 0.5)."""
    rng = np.random.default_rng(42)
    M = rng.standard_normal((256, 20))
    pbo = probability_of_backtest_overfitting(M, n_partitions=10)
    assert pbo >= 0.4


def test_pbo_low_when_one_strategy_dominates():
    """If one strategy genuinely dominates everywhere, the IS-best is also
    OS-best -> PBO should be low."""
    rng = np.random.default_rng(7)
    T, S = 256, 10
    noise = rng.standard_normal((T, S)) * 0.1
    # Strategy 0 has a large, persistent positive drift; others ~0.
    drift = np.zeros((T, S))
    drift[:, 0] = 2.0
    M = noise + drift
    pbo = probability_of_backtest_overfitting(M, n_partitions=10)
    assert pbo <= 0.1


def test_pbo_rejects_single_strategy():
    M = np.random.default_rng(1).standard_normal((64, 1))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(M, n_partitions=8)


def test_pbo_rejects_odd_partitions():
    M = np.random.default_rng(1).standard_normal((64, 4))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(M, n_partitions=7)


def test_pbo_higher_is_better_flag_flips_selection():
    """For a dominating-loss matrix, flipping higher_is_better should change
    which strategy is selected and hence the PBO outcome."""
    rng = np.random.default_rng(3)
    T, S = 200, 8
    noise = rng.standard_normal((T, S)) * 0.1
    drift = np.zeros((T, S))
    drift[:, 0] = 2.0  # column 0 has the highest values everywhere
    M = noise + drift
    # higher_is_better: col 0 dominates -> low PBO
    pbo_high = probability_of_backtest_overfitting(
        M, n_partitions=8, higher_is_better=True
    )
    # lower_is_better: col 0 is now the worst -> selection picks a noisy col,
    # so PBO is not driven to ~0 by a true dominator.
    pbo_low = probability_of_backtest_overfitting(
        M, n_partitions=8, higher_is_better=False
    )
    assert pbo_high <= 0.1
    assert pbo_low > pbo_high
