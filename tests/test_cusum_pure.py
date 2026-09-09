"""Parity tests for the pure-Python/numpy CUSUM change-point filter.

These lock the pure-Python (and optional numba) implementation of
``.ts.cusum(...)`` to the output of the former Rust plugin. The baseline event
lists in ``cusum_baseline_<thr>_<drift>.json`` were captured from the Rust
version before it was removed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from panelary._deps import have
from panelary.feature_extractors import _cusum_events, _cusum_events_py

# Directory holding the Rust-captured parity baselines.
#
# These were captured from the Rust plugin that 0.4.0 removed, so they cannot be
# regenerated -- and the path they were read from was an ephemeral per-session
# scratchpad that no longer exists, which meant these six tests had been failing
# with FileNotFoundError on every machine and in every CI run.
#
# The files belong next to the repo's other golden fixture
# (tests/data/perf_parity_golden.json). Drop
# `cusum_baseline_<threshold>_<drift>.json` in there and the tests light up
# again; until then they skip loudly rather than fail silently or, worse, get
# "fixed" by regenerating the expectations from the implementation they are
# supposed to be checking.
_BASELINE_DIR = Path(__file__).parent / "data"

_WARMUP = 10
_CASES = [(1.0, 0.0), (2.0, 0.5), (5.0, 0.0)]


def _make_signal() -> np.ndarray:
    """Rebuild the exact 120-point signal used to capture the baselines."""
    rng = np.random.default_rng(0)
    return np.concatenate([rng.standard_normal(60), rng.standard_normal(60) + 3.0])


_HAVE_BASELINES = all(
    (_BASELINE_DIR / f"cusum_baseline_{thr}_{drift}.json").exists()
    for thr, drift in _CASES
)
_NEEDS_BASELINES = pytest.mark.skipif(
    not _HAVE_BASELINES,
    reason=(
        "Rust parity baselines missing from tests/data/. They were captured "
        "from the Rust plugin removed in 0.4.0 and cannot be regenerated; see "
        "the note on _BASELINE_DIR."
    ),
)


@_NEEDS_BASELINES
@pytest.mark.parametrize(("threshold", "drift"), _CASES)
def test_matches_rust_baseline(threshold: float, drift: float) -> None:
    baseline = json.loads(
        (_BASELINE_DIR / f"cusum_baseline_{threshold}_{drift}.json").read_text()
    )
    signal = _make_signal()
    got = _cusum_events(signal.copy(), threshold, _WARMUP, drift)
    assert got.dtype == np.int32
    assert got.tolist() == baseline


@_NEEDS_BASELINES
@pytest.mark.parametrize(("threshold", "drift"), _CASES)
def test_expr_matches_rust_baseline(threshold: float, drift: float) -> None:
    """The ``.ts.cusum`` expression reproduces the baseline exactly."""
    baseline = json.loads(
        (_BASELINE_DIR / f"cusum_baseline_{threshold}_{drift}.json").read_text()
    )
    df = pl.DataFrame({"v": _make_signal()})
    out = df.select(pl.col("v").ts.cusum(threshold, _WARMUP, drift).alias("e"))["e"]
    assert out.dtype == pl.Int32
    assert out.to_list() == baseline


def test_over_entity_no_cross_bleed() -> None:
    """``.over(entity)`` must run CUSUM per group with no cross-entity bleed."""
    signal = _make_signal()
    threshold, drift = 2.0, 0.5

    # Two entities with distinct signals; each group scored independently.
    sig_a = signal
    sig_b = signal[::-1]  # a genuinely different series
    df = pl.DataFrame(
        {
            "entity": ["a"] * len(sig_a) + ["b"] * len(sig_b),
            "v": np.concatenate([sig_a, sig_b]),
        }
    )
    out = df.with_columns(
        pl.col("v").ts.cusum(threshold, _WARMUP, drift).over("entity").alias("e")
    )

    exp_a = _cusum_events(sig_a.copy(), threshold, _WARMUP, drift).tolist()
    exp_b = _cusum_events(sig_b.copy(), threshold, _WARMUP, drift).tolist()
    got_a = out.filter(pl.col("entity") == "a")["e"].to_list()
    got_b = out.filter(pl.col("entity") == "b")["e"].to_list()
    assert got_a == exp_a
    assert got_b == exp_b


def test_null_nan_handling() -> None:
    """Nulls (NaN) are treated as the Rust ``None``: not collected, never events."""
    threshold, drift = 2.0, 0.0
    signal = _make_signal()

    # Inject nulls at a few positions after warmup.
    values = signal.astype(np.float64).copy()
    null_idx = [12, 45, 90]
    with_nan = values.copy()
    for i in null_idx:
        with_nan[i] = np.nan

    # numpy-level: NaN positions must always be 0 (no event) and not raise.
    ev = _cusum_events(with_nan, threshold, _WARMUP, drift)
    for i in null_idx:
        assert ev[i] == 0

    # Expression-level with real polars nulls must match the numpy path.
    s = pl.Series("v", values)
    for i in null_idx:
        s[i] = None
    df = pl.DataFrame({"v": s})
    out = df.select(pl.col("v").ts.cusum(threshold, _WARMUP, drift).alias("e"))["e"]
    assert out.to_list() == ev.tolist()


def test_zero_sigma_no_guard_does_not_raise() -> None:
    """Constant warmup window => sigma==0; IEEE division must not raise."""
    # First 10 identical values -> sigma == 0; then a jump.
    values = np.array([1.0] * 10 + [1.0, 5.0, 5.0, 5.0], dtype=np.float64)
    ev = _cusum_events(values, threshold=2.0, warmup_period=10, drift=0.0)
    assert ev.dtype == np.int32
    assert len(ev) == len(values)


@pytest.mark.skipif(not have("numba"), reason="numba (fast extra) not installed")
def test_numba_matches_python() -> None:
    from panelary.feature_extractors import _get_cusum_numba

    kernel = _get_cusum_numba()
    assert kernel is not None
    rng = np.random.default_rng(123)
    for _ in range(10):
        n = int(rng.integers(30, 200))
        arr = rng.standard_normal(n)
        # sprinkle some NaNs
        mask = rng.random(n) < 0.05
        arr[mask] = np.nan
        threshold = float(rng.uniform(0.5, 5.0))
        drift = float(rng.uniform(0.0, 1.0))
        warmup = int(rng.integers(1, 15))
        exp = _cusum_events_py(
            np.ascontiguousarray(arr, dtype=np.float64), threshold, warmup, drift
        )
        got = np.asarray(
            kernel(
                np.ascontiguousarray(arr, dtype=np.float64), threshold, warmup, drift
            )
        )
        assert got.tolist() == exp.tolist()
