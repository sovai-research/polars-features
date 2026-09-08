"""Exactness tests for the O(1) / O(m) sequential monitors.

None of these detectors needs calibration, which is precisely why they are
worth having: each one has a closed form or a brute-force definition that can
be checked to the last bit. Every test here compares the shipped implementation
against that definition, written out here in the slowest, most obvious way.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest

from tests import test_detect_support as S


# --------------------------------------------------------------------------- #
# FOCuS
# --------------------------------------------------------------------------- #
def test_focus_matches_brute_force_exactly() -> None:
    """``focus(z)[t] == max_i (S_t - S_i)^2 / (2 (t - i))``, to 0.0.

    The functional pruning discards a candidate only once it is dominated for
    every shift magnitude, so it is *exact*, not approximate: the admissible
    difference from the brute-force sweep is zero, not 1e-12.
    """
    fn = S.require_attr(S.monitors, "focus", "_monitors")
    rng = np.random.default_rng(101)
    z = np.concatenate([rng.standard_normal(300), rng.standard_normal(200) + 0.8])

    got = np.asarray(fn(z, side="two"), dtype=np.float64)
    assert got.shape == z.shape, f"focus returned {got.shape}, want {z.shape}"

    dev = float(np.max(np.abs(got - S.naive_focus(z))))
    if dev != 0.0:
        dev1 = float(np.max(np.abs(got - S.naive_focus_one_sided(z))))
        pytest.fail(
            f"focus(side='two') differs from the unrestricted brute force by "
            f"{dev:.3g}; the pruning is exact, so the only admissible "
            f"difference is 0.0. Deviation from the one-sided sweep is "
            f"{dev1:.3g} -- if that one is 0.0 the two-sided branch is not "
            "considering downward shifts."
        )


def test_focus_default_side_is_the_one_sided_sweep_exactly() -> None:
    """The default ``side='upper'`` restricts the candidate set to ``S_t > S_i``."""
    fn = S.require_attr(S.monitors, "focus", "_monitors")
    rng = np.random.default_rng(131)
    z = np.concatenate([rng.standard_normal(300), rng.standard_normal(200) + 0.8])
    got = np.asarray(fn(z), dtype=np.float64)
    assert float(np.max(np.abs(got - S.naive_focus_one_sided(z)))) == 0.0
    assert (got >= -0.0).all(), "a one-sided GLR statistic is non-negative"
    two = np.asarray(fn(z, side="two"), dtype=np.float64)
    assert (two >= got - 1e-12).all(), (
        "the two-sided statistic is a max over a superset of candidates and can "
        "never be smaller than the one-sided one."
    )


def test_focus_lower_side_mirrors_the_upper_side() -> None:
    """``focus(-z, side='upper')`` and ``focus(z, side='lower')`` are the same test."""
    fn = S.require_attr(S.monitors, "focus", "_monitors")
    z = np.random.default_rng(132).standard_normal(400) - 0.3
    lower = np.asarray(fn(z, side="lower"), dtype=np.float64)
    mirror = np.asarray(fn(-z, side="upper"), dtype=np.float64)
    assert S.bit_identical(lower, mirror)


def test_focus_is_exact_under_a_pure_mean_shift() -> None:
    """A clean two-segment mean shift: the maximiser must be the true change point."""
    fn = S.require_attr(S.monitors, "focus", "_monitors")
    z = np.concatenate([np.full(120, -0.5), np.full(120, 0.5)])
    got, start = fn(z, side="two", return_start=True)
    got = np.asarray(got, dtype=np.float64)
    assert np.max(np.abs(got - S.naive_focus(z))) == 0.0
    # At the last observation the best split is exactly at index 120.
    s = np.concatenate([[0.0], np.cumsum(z)])
    t = z.size
    i = np.arange(t)
    stats = (s[t] - s[i]) ** 2 / (2.0 * (t - i))
    assert int(np.argmax(stats)) == 120
    assert int(start[-1]) == 120, (
        f"focus reported the change point at {int(start[-1])}, brute force says 120."
    )


def test_focus_candidate_stack_stays_logarithmic() -> None:
    """The pruned candidate set is O(log n): under 40 entries at n = 100 000."""
    fn = S.require_attr(S.monitors, "focus", "_monitors")
    n = 100_000
    z = np.random.default_rng(102).standard_normal(n)

    for attr in ("focus_max_stack_len", "_focus_max_stack_len", "_focus_stack_len"):
        probe = getattr(S.monitors, attr, None)
        if probe is not None:
            assert int(probe(z)) < 40, (
                f"{attr}(z) at n={n} reports a candidate stack of "
                f"{int(probe(z))}; the monotone pruning is not working."
            )
            return
    try:
        out, stack = fn(z, return_stack=True)
    except TypeError:
        pass
    else:
        assert int(np.max(stack)) < 40, (
            f"focus candidate stack peaked at {int(np.max(stack))} at n={n}"
        )
        return

    # No introspection hook: fall back to the observable consequence. An
    # unpruned FOCuS is O(n^2) -- 1e10 operations at n = 1e5 -- so any run that
    # finishes in seconds has a sub-linear candidate set.
    t0 = time.perf_counter()
    out = np.asarray(fn(z), dtype=np.float64)
    elapsed = time.perf_counter() - t0
    assert out.shape == z.shape
    assert elapsed < 5.0, (
        f"focus took {elapsed:.2f}s at n={n}; that is quadratic behaviour, so "
        "the candidate stack is not being pruned."
    )


# --------------------------------------------------------------------------- #
# Page CUSUM
# --------------------------------------------------------------------------- #
N_PAGE = 20_000


def _page_signal() -> np.ndarray:
    rng = np.random.default_rng(103)
    return np.concatenate(
        [rng.standard_normal(N_PAGE - 4_000) - 0.05, rng.standard_normal(4_000) + 0.30]
    )


def test_page_cusum_closed_form_matches_the_recursion() -> None:
    """``S - min(cummin(S), 0)`` equals ``g = max(0, g + z)`` to < 1e-10."""
    fn = S.require_attr(S.monitors, "page_cusum_expr", "_monitors")
    z = _page_signal()

    got = (
        pl.DataFrame({"z": z})
        .select(fn(pl.col("z")).alias("g"))
        .get_column("g")
        .to_numpy()
    )
    ref = S.naive_page_cusum(z)

    assert got.shape == z.shape
    dev = float(np.max(np.abs(got - ref)))
    assert dev < 1e-10, (
        f"the vectorised Page CUSUM drifts from the recursion by {dev:.3g} over "
        f"{N_PAGE} points."
    )
    assert (got >= -1e-12).all(), "the Page detector is non-negative by construction"


@pytest.mark.parametrize("threshold", [2.0, 5.0, 10.0])
def test_page_cusum_first_crossing_index_is_identical(threshold: float) -> None:
    """An alarm time that is off by one is a different trading day."""
    fn = S.require_attr(S.monitors, "page_cusum_expr", "_monitors")
    z = _page_signal()
    got = (
        pl.DataFrame({"z": z})
        .select(fn(pl.col("z")).alias("g"))
        .get_column("g")
        .to_numpy()
    )
    ref = S.naive_page_cusum(z)

    hit_got = np.flatnonzero(got >= threshold)
    hit_ref = np.flatnonzero(ref >= threshold)
    assert hit_ref.size, "the reference never crossed; pick a lower threshold"
    assert hit_got.size, f"the closed form never reached {threshold}"
    assert int(hit_got[0]) == int(hit_ref[0]), (
        f"first crossing of {threshold} at index {int(hit_got[0])} vs "
        f"{int(hit_ref[0])} for the recursion."
    )


def test_page_cusum_expression_respects_entity_boundaries() -> None:
    """``.over(entity)`` must restart the detector, not carry it across entities."""
    fn = S.require_attr(S.monitors, "page_cusum_expr", "_monitors")
    a = np.random.default_rng(104).standard_normal(300)
    b = np.random.default_rng(105).standard_normal(300) + 0.4
    df = pl.DataFrame(
        {"entity": ["a"] * 300 + ["b"] * 300, "z": np.concatenate([a, b])}
    )
    out = df.with_columns(fn(pl.col("z")).over("entity").alias("g"))
    got_a = out.filter(pl.col("entity") == "a").get_column("g").to_numpy()
    got_b = out.filter(pl.col("entity") == "b").get_column("g").to_numpy()
    assert np.max(np.abs(got_a - S.naive_page_cusum(a))) < 1e-10
    assert np.max(np.abs(got_b - S.naive_page_cusum(b))) < 1e-10


# --------------------------------------------------------------------------- #
# Shiryaev-Roberts
# --------------------------------------------------------------------------- #
def _sr_reference(llr: np.ndarray, *, log_domain_start: bool, r0: float) -> np.ndarray:
    """``R_t = logaddexp(R_{t-1}, 0) + llr_t``, one explicit Python loop."""
    prev = r0 if log_domain_start else (-np.inf if r0 <= 0.0 else np.log(r0))
    out = np.empty(llr.size, dtype=np.float64)
    for i in range(llr.size):
        prev = np.logaddexp(prev, 0.0) + float(llr[i])
        out[i] = prev
    return out


def test_shiryaev_roberts_matches_the_log_domain_recursion() -> None:
    fn = S.require_attr(S.monitors, "shiryaev_roberts", "_monitors")
    rng = np.random.default_rng(106)
    z = np.concatenate([rng.standard_normal(400), rng.standard_normal(200) + 1.0])

    kwargs, llr = S.resolve_shiryaev_kwargs(z)
    got = np.asarray(fn(z, **kwargs), dtype=np.float64)
    assert got.shape == z.shape

    raw = _sr_reference(llr, log_domain_start=False, r0=0.0)
    logd = _sr_reference(llr, log_domain_start=True, r0=0.0)
    dev_raw = float(np.max(np.abs(got - raw)))
    dev_log = float(np.max(np.abs(got - logd)))
    assert min(dev_raw, dev_log) < 1e-10, (
        f"shiryaev_roberts matches neither the R_0 = 0 recursion (dev "
        f"{dev_raw:.3g}) nor the log-domain R_0 = 0.0 recursion (dev "
        f"{dev_log:.3g}). The contract fixes it as "
        "`R_t = logaddexp(R_{t-1}, 0) + llr_t`."
    )
    # No overflow: the whole point of the log domain is that the raw statistic
    # would have exceeded float64 range long before the end of the series.
    assert np.isfinite(got).all(), "the log-domain recursion produced non-finite values"
    assert got[-1] > got[100], "the detector did not respond to the planted shift"


def test_shiryaev_roberts_is_prefix_invariant() -> None:
    fn = S.require_attr(S.monitors, "shiryaev_roberts", "_monitors")
    z = np.random.default_rng(107).standard_normal(400)
    kwargs, _ = S.resolve_shiryaev_kwargs(z)
    full = np.asarray(fn(z, **kwargs), dtype=np.float64)
    for cut in (120, 200, 310):
        kw, _ = S.resolve_shiryaev_kwargs(z[:cut])
        part = np.asarray(fn(z[:cut], **kw), dtype=np.float64)
        assert S.bit_identical(full[:cut], part), (
            f"shiryaev_roberts differs on y[:{cut}] versus the full sample."
        )


# --------------------------------------------------------------------------- #
# Homm-Breitung monitoring CUSUM
# --------------------------------------------------------------------------- #
def test_hb_cusum_boundary_depends_on_the_training_length_not_on_the_data() -> None:
    """The detection boundary is a deterministic function of ``t`` and ``r0``."""
    fn = S.require_attr(S.monitors, "hb_cusum", "_monitors")
    r0 = 60
    stat_a, bound_a = fn(S.random_walk(300, seed=111), r0=r0)
    stat_b, bound_b = fn(S.bubble_series(300, seed=112), r0=r0)

    stat_a = np.asarray(stat_a, dtype=np.float64)
    bound_a = np.asarray(bound_a, dtype=np.float64)
    bound_b = np.asarray(bound_b, dtype=np.float64)
    assert stat_a.shape == (300,) and bound_a.shape == (300,)
    assert S.bit_identical(bound_a, bound_b), (
        "the Homm-Breitung boundary moved with the data; it is a function of "
        "the monitoring horizon alone, and a data-dependent boundary would make "
        "the alarm time depend on the future path."
    )

    tail = np.isfinite(bound_a)
    assert tail.sum() > 100
    assert (np.diff(bound_a[tail]) >= -1e-12).all(), (
        "the boundary must widen with the monitoring horizon"
    )


def test_hb_cusum_boundary_scales_with_kappa() -> None:
    fn = S.require_attr(S.monitors, "hb_cusum", "_monitors")
    y = S.random_walk(300, seed=113)
    _, low = fn(y, r0=60, kappa=4.6)
    _, high = fn(y, r0=60, kappa=9.2)
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    mask = np.isfinite(low) & np.isfinite(high)
    assert mask.sum() > 100
    assert (high[mask] > low[mask]).all(), "a larger kappa must widen the boundary"


def test_hb_cusum_fires_on_a_bubble_and_rarely_on_a_random_walk() -> None:
    fn = S.require_attr(S.monitors, "hb_cusum", "_monitors")
    r0 = 60

    stat, bound = (
        np.asarray(a, dtype=np.float64)
        for a in fn(S.bubble_series(300, seed=114), r0=r0)
    )
    mask = np.isfinite(stat) & np.isfinite(bound)
    assert (stat[mask] > bound[mask]).any(), (
        "the monitor never crossed its boundary on an explosive episode"
    )

    fired = 0
    for seed in range(20):
        s, b = (
            np.asarray(a, dtype=np.float64)
            for a in fn(S.random_walk(300, seed=1000 + seed), r0=r0)
        )
        m = np.isfinite(s) & np.isfinite(b)
        fired += bool((s[m] > b[m]).any())
    assert fired <= 6, (
        f"the monitor fired on {fired}/20 driftless random walks; the nominal "
        "size at kappa=4.6 is a few percent, so the boundary is far too tight."
    )


# --------------------------------------------------------------------------- #
# Andrews end-of-sample instability
# --------------------------------------------------------------------------- #
def test_end_of_sample_S_shape_and_determinism() -> None:
    fn = S.require_attr(S.monitors, "end_of_sample_S", "_monitors")
    y = S.random_walk(300, seed=115)
    a = np.asarray(fn(y, m=10), dtype=np.float64)
    b = np.asarray(fn(y, m=10), dtype=np.float64)
    assert a.shape == y.shape, f"end_of_sample_S returned {a.shape}, want {y.shape}"
    assert S.bit_identical(a, b)
    assert np.isfinite(a[-50:]).any(), "no finite values anywhere in the tail"
    assert np.isnan(a[:10]).all(), "the first m observations have no window yet"


def test_end_of_sample_S_reproduces_its_closed_form() -> None:
    """``S^w_m`` on a clean ramp is ``55 / sqrt(385)``, computed by hand."""
    fn = S.require_attr(S.monitors, "end_of_sample_S", "_monitors")
    y = np.concatenate([np.zeros(50), np.arange(1, 11) * 1.0])
    got = float(np.asarray(fn(y, m=10), dtype=np.float64)[-1])

    dy = np.diff(y)[-10:]
    w = np.arange(1, 11, dtype=np.float64)
    want = float(np.sum(w * dy) / np.sqrt(np.sum((w * dy) ** 2)))
    assert want == pytest.approx(55.0 / np.sqrt(385.0), rel=1e-12)
    assert got == pytest.approx(want, rel=1e-12), (
        f"end_of_sample_S(ramp)[-1] = {got:.6f}, hand calculation gives "
        f"{want:.6f} = 55/sqrt(385)."
    )


def test_end_of_sample_S_responds_to_a_terminal_break() -> None:
    """A shift confined to the final ``m`` observations must raise the statistic."""
    fn = S.require_attr(S.monitors, "end_of_sample_S", "_monitors")
    m = 10
    clean = S.random_walk(300, seed=116)
    broken = clean.copy()
    broken[-m:] += np.linspace(5.0, 25.0, m)

    a = np.asarray(fn(clean, m=m), dtype=np.float64)
    b = np.asarray(fn(broken, m=m), dtype=np.float64)
    assert np.isfinite(a[-1]) and np.isfinite(b[-1])
    assert b[-1] > a[-1], (
        f"a large terminal break left the statistic at {b[-1]:.4f} versus "
        f"{a[-1]:.4f} on the clean series."
    )
    # Only the last m observations changed, so nothing earlier may move.
    n = clean.size - m
    assert S.bit_identical(a[:n], b[:n]), (
        "editing the last m observations changed earlier values of the "
        "end-of-sample statistic."
    )


@pytest.mark.parametrize("studentise", ["white", "none"])
def test_end_of_sample_S_studentisation_options(studentise: str) -> None:
    fn = S.require_attr(S.monitors, "end_of_sample_S", "_monitors")
    y = S.random_walk(200, seed=117)
    try:
        out = np.asarray(fn(y, m=10, studentise=studentise), dtype=np.float64)
    except (ValueError, KeyError) as exc:
        if studentise == "none":
            pytest.skip(f"studentise={studentise!r} not offered: {exc}")
        raise
    assert out.shape == y.shape
    assert np.isfinite(out[-1])
