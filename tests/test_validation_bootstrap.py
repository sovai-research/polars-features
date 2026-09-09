"""Tests for ``panelary.validation._bootstrap``.

The load-bearing property is the leak-safety guardrail: **a resampled block must
never straddle a fold boundary**. Everything else (determinism, shape, moment
recovery) protects the numerics.
"""

from __future__ import annotations

import numpy as np
import pytest

from panelary.validation import (
    block_bootstrap_indices,
    circular_block_bootstrap,
    moving_block_bootstrap,
    resolve_segments,
    sieve_bootstrap,
    stationary_bootstrap,
    wild_bootstrap,
)


# --------------------------------------------------------------------------- #
# Segments
# --------------------------------------------------------------------------- #
def test_resolve_segments_defaults_to_one_segment():
    assert resolve_segments(10, None) == [(0, 10)]
    assert resolve_segments(10, []) == [(0, 10)]


def test_resolve_segments_ignores_trivial_edges():
    assert resolve_segments(10, [0, 4, 10]) == [(0, 4), (4, 10)]


def test_resolve_segments_rejects_out_of_range():
    with pytest.raises(ValueError, match="outside"):
        resolve_segments(10, [12])


# --------------------------------------------------------------------------- #
# Leak-safety: blocks never straddle a fold boundary
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scheme", ["moving", "circular", "stationary"])
def test_blocks_never_straddle_a_fold_boundary(scheme):
    """Every resampled slot draws from the segment that slot belongs to.

    This is the exact statement of the guardrail: because slot ``t`` is filled
    only from ``t``'s own segment, no block can span a cut, and a fold's
    resampled series can never contain another fold's observations.
    """
    n, boundaries = 60, [20, 40]
    idx = block_bootstrap_indices(
        n,
        block_length=7,
        n_boot=60,
        scheme=scheme,
        boundaries=boundaries,
        seed=0,
    )
    for lo, hi in resolve_segments(n, boundaries):
        drawn = idx[:, lo:hi]
        assert drawn.min() >= lo, f"{scheme}: leaked from before segment [{lo},{hi})"
        assert drawn.max() < hi, f"{scheme}: leaked from after segment [{lo},{hi})"
        # No run of consecutive positions inside the segment crosses ``hi``.
        for row in drawn:
            breaks = np.flatnonzero(np.diff(row) != 1) + 1
            for block in np.split(row, breaks):
                assert int(block.max()) < hi


def test_output_positions_stay_within_their_own_segment():
    """Position ``t`` of the resample must come from ``t``'s own segment.

    Otherwise a fold's resampled series would contain another fold's data even
    if no individual block straddled a cut.
    """
    n, boundaries = 60, [20, 40]
    idx = block_bootstrap_indices(
        n, block_length=5, n_boot=30, scheme="moving", boundaries=boundaries, seed=1
    )
    for lo, hi in resolve_segments(n, boundaries):
        block = idx[:, lo:hi]
        assert block.min() >= lo
        assert block.max() < hi


def test_moving_blocks_are_contiguous_runs_of_the_requested_length():
    idx = block_bootstrap_indices(
        48, block_length=6, n_boot=20, scheme="moving", seed=3
    )
    for row in idx:
        for start in range(0, 48, 6):
            block = row[start : start + 6]
            assert np.all(np.diff(block) == 1)


# --------------------------------------------------------------------------- #
# Determinism / shapes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fn", [moving_block_bootstrap, circular_block_bootstrap, stationary_bootstrap]
)
def test_seeded_bootstraps_are_deterministic(fn):
    x = np.arange(40.0)
    a = fn(x, block_length=5, n_boot=8, seed=7)
    b = fn(x, block_length=5, n_boot=8, seed=7)
    c = fn(x, block_length=5, n_boot=8, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize(
    "fn", [moving_block_bootstrap, circular_block_bootstrap, stationary_bootstrap]
)
def test_bootstrap_preserves_length_and_support(fn):
    x = np.arange(40.0)
    out = fn(x, block_length=4, n_boot=6, seed=0)
    assert out.shape == (6, 40)
    assert set(np.unique(out)).issubset(set(x))


def test_bootstrap_handles_2d_input():
    x = np.arange(60.0).reshape(20, 3)
    out = moving_block_bootstrap(x, block_length=4, n_boot=5, seed=0)
    assert out.shape == (5, 20, 3)
    # Rows are resampled jointly: the (col1 - col0) offset is preserved.
    assert np.all(out[..., 1] - out[..., 0] == 1)


def test_circular_bootstrap_can_reach_the_final_observation_from_any_start():
    """The wrap is what removes the moving-block edge bias."""
    idx = block_bootstrap_indices(
        20, block_length=6, n_boot=200, scheme="circular", seed=0
    )
    # A wrapping block produces a -19 step (19 -> 0).
    assert np.any(np.diff(idx, axis=1) == -19)


def test_stationary_bootstrap_block_lengths_vary():
    idx = block_bootstrap_indices(
        200, block_length=8, n_boot=5, scheme="stationary", seed=0
    )
    run_lengths = set()
    for row in idx:
        breaks = np.flatnonzero(np.diff(row) != 1) + 1
        run_lengths.update(len(b) for b in np.split(row, breaks))
    assert len(run_lengths) > 3  # geometric, not fixed


def test_block_bootstrap_rejects_bad_arguments():
    with pytest.raises(ValueError, match="block_length"):
        block_bootstrap_indices(10, block_length=0)
    with pytest.raises(ValueError, match="unknown `scheme`"):
        block_bootstrap_indices(10, block_length=2, scheme="nope")
    with pytest.raises(ValueError, match="n_boot"):
        block_bootstrap_indices(10, block_length=2, n_boot=0)


# --------------------------------------------------------------------------- #
# Statistical sanity
# --------------------------------------------------------------------------- #
def test_block_bootstrap_preserves_autocorrelation_better_than_iid():
    """A strongly autocorrelated AR(1) keeps its lag-1 correlation under blocks."""
    rng = np.random.default_rng(0)
    n = 600
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.standard_normal()

    def rho1(v):
        v = v - v.mean()
        return float(np.dot(v[1:], v[:-1]) / np.dot(v, v))

    blocks = moving_block_bootstrap(x, block_length=40, n_boot=50, seed=0)
    iid = x[rng.integers(0, n, size=(50, n))]
    rho_block = np.mean([rho1(r) for r in blocks])
    rho_iid = np.mean([rho1(r) for r in iid])
    assert rho_block > 0.75
    assert abs(rho_iid) < 0.1


def test_stationary_bootstrap_is_unbiased_for_the_mean():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(400) + 2.0
    boot = stationary_bootstrap(x, block_length=10, n_boot=400, seed=2)
    assert boot.mean() == pytest.approx(x.mean(), abs=0.05)


# --------------------------------------------------------------------------- #
# Wild bootstrap
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dist", ["rademacher", "mammen", "normal"])
def test_wild_bootstrap_multipliers_have_unit_variance_and_zero_mean(dist):
    e = np.ones(4000)
    out = wild_bootstrap(e, n_boot=20, distribution=dist, seed=0)
    assert out.shape == (20, 4000)
    assert out.mean() == pytest.approx(0.0, abs=0.02)
    assert out.std() == pytest.approx(1.0, abs=0.03)


def test_wild_bootstrap_preserves_heteroskedasticity():
    e = np.concatenate([np.full(500, 0.1), np.full(500, 5.0)])
    out = wild_bootstrap(e, n_boot=100, seed=0)
    assert out[:, :500].std() < out[:, 500:].std() / 10


def test_wild_bootstrap_adds_fitted_values():
    e = np.ones(10)
    fitted = np.arange(10.0)
    out = wild_bootstrap(e, n_boot=3, distribution="rademacher", fitted=fitted, seed=0)
    assert np.all(np.abs(out - fitted) == 1.0)


def test_wild_bootstrap_rejects_unknown_distribution_and_bad_fitted():
    with pytest.raises(ValueError, match="unknown `distribution`"):
        wild_bootstrap(np.ones(5), distribution="bogus")
    with pytest.raises(ValueError, match="must match"):
        wild_bootstrap(np.ones(5), fitted=np.ones(4))


# --------------------------------------------------------------------------- #
# Sieve bootstrap
# --------------------------------------------------------------------------- #
def test_sieve_bootstrap_recovers_ar1_persistence():
    rng = np.random.default_rng(0)
    n = 500
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.standard_normal()
    boot = sieve_bootstrap(x, n_boot=60, order=1, seed=0)
    assert boot.shape == (60, n)

    def rho1(v):
        v = v - v.mean()
        return float(np.dot(v[1:], v[:-1]) / np.dot(v, v))

    assert np.mean([rho1(r) for r in boot]) == pytest.approx(0.7, abs=0.12)


def test_sieve_bootstrap_selects_an_order_and_is_deterministic():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(200)
    a = sieve_bootstrap(x, n_boot=5, seed=11)
    b = sieve_bootstrap(x, n_boot=5, seed=11)
    np.testing.assert_allclose(a, b)


def test_sieve_bootstrap_requires_enough_data():
    with pytest.raises(ValueError, match="at least 4"):
        sieve_bootstrap(np.ones(3))
