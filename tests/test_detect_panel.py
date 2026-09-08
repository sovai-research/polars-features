"""Cross-sectional aggregation: breadth, ranks, residualisation, panel critical values.

The headline test here is :func:`test_panel_mean_critical_value_is_near_or_below_zero`.
It looks wrong the first time you read it -- a critical value at *or below zero*
for a statistic whose single-series 95% cut-off is around 2.2 -- and that is
exactly why it earns its place. Averaging a backward sup-ADF across ``N``
entities destroys almost all of its variance, so the panel null distribution
collapses onto the *mean* of the single-series distribution rather than onto its
upper tail. A panel routine that silently fails to accumulate across entities
(the reference R package shipped this bug for years) reproduces the
single-series cut-off instead, and the difference is the whole test.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import test_detect_support as S

LEVELS = (0.90, 0.95, 0.99)


# --------------------------------------------------------------------------- #
# 8. Panel null sign
# --------------------------------------------------------------------------- #
def _panel_mean_null_draws(
    *, n_entities: int, n_time: int, min_window: int, nrep: int, seed: int
) -> np.ndarray:
    """MC draws of the published MEAN panel statistic at the final endpoint."""
    bs = S.bsadf.bsadf_panel
    pm = S.panel.panel_mean
    rng = np.random.default_rng(seed)
    out = np.empty(nrep, dtype=np.float64)
    for r in range(nrep):
        mat = np.cumsum(rng.standard_normal((n_entities, n_time)), axis=1)
        stat = np.asarray(
            bs(mat, min_window=min_window, lag=0, grid=32), dtype=np.float64
        )
        out[r] = float(np.asarray(pm(stat, limit=0.0), dtype=np.float64)[-1])
    return out


def _single_null_draws(
    *, n_time: int, min_window: int, nrep: int, seed: int
) -> np.ndarray:
    fn = S.bsadf.bsadf_sequence
    rng = np.random.default_rng(seed)
    out = np.empty(nrep, dtype=np.float64)
    for r in range(nrep):
        y = np.cumsum(rng.standard_normal(n_time))
        out[r] = float(np.asarray(fn(y, min_window=min_window, lag=0, grid=32))[-1])
    return out


def test_panel_mean_critical_value_is_near_or_below_zero() -> None:
    """The panel MEAN statistic's 95% null cut-off must collapse towards zero."""
    S.require("_bsadf", "_panel")
    S.require_attr(S.bsadf, "bsadf_panel", "_bsadf")
    S.require_attr(S.panel, "panel_mean", "_panel")

    bound = float(getattr(S.panel, "PANEL_MEAN_CV_UPPER_BOUND", 0.5))
    n_entities, n_time, min_window, nrep = 20, 150, 25, 150
    panel_draws = _panel_mean_null_draws(
        n_entities=n_entities,
        n_time=n_time,
        min_window=min_window,
        nrep=nrep,
        seed=201,
    )
    single_draws = _single_null_draws(
        n_time=n_time, min_window=min_window, nrep=nrep * 4, seed=202
    )

    cv_panel = float(np.quantile(panel_draws, 0.95))
    cv_single = float(np.quantile(single_draws, 0.95))

    assert cv_panel <= bound, (
        f"the 95% null cut-off for the MEAN panel statistic over "
        f"N={n_entities} entities is {cv_panel:.3f}, above the module's own "
        f"PANEL_MEAN_CV_UPPER_BOUND of {bound}. Averaging across the "
        "cross-section removes nearly all of the variance, so the panel null "
        "concentrates near the mean ADF t-statistic -- at or below zero -- not "
        f"in the single-series tail ({cv_single:.3f}). A cut-off up there means "
        "the statistic is not accumulating across entities, which is exactly "
        "the bug the reference R package shipped."
    )
    assert cv_panel < 0.5 * cv_single, (
        f"the panel cut-off ({cv_panel:.3f}) is not materially below the "
        f"single-series cut-off ({cv_single:.3f})."
    )
    # And the variance really does shrink: that is the mechanism, so assert it.
    assert panel_draws.std() < 0.5 * single_draws.std(), (
        f"panel sd {panel_draws.std():.3f} vs single-series sd "
        f"{single_draws.std():.3f}; averaging N={n_entities} entities should "
        "cut the spread by roughly sqrt(N)."
    )


def test_panel_mean_upper_bound_constant_is_published() -> None:
    """The counter-intuitive bound is a named constant, not a magic number."""
    S.require("_panel")
    bound = getattr(S.panel, "PANEL_MEAN_CV_UPPER_BOUND", None)
    assert bound is not None, (
        "`_panel` must publish PANEL_MEAN_CV_UPPER_BOUND so callers can assert "
        "the panel mean's null cut-off is near zero."
    )
    assert 0.0 < float(bound) <= 1.0


def test_breadth_is_the_recommended_panel_statistic_and_is_bounded() -> None:
    """Unlike the mean, breadth sits on a fixed [0, 1] scale under the null."""
    S.require("_bsadf", "_panel")
    bs = S.require_attr(S.bsadf, "bsadf_panel", "_bsadf")
    br = S.require_attr(S.panel, "breadth", "_panel")
    rng = np.random.default_rng(220)
    mat = np.cumsum(rng.standard_normal((20, 150)), axis=1)
    stat = np.asarray(bs(mat, min_window=25, lag=0, grid=32), dtype=np.float64)
    out = np.asarray(br(stat, 1.5), dtype=np.float64)
    fin = np.isfinite(out)
    assert fin.any()
    assert ((out[fin] >= 0.0) & (out[fin] <= 1.0)).all()
    assert float(np.mean(out[fin])) < 0.35, (
        f"a pure-null panel shows breadth {float(np.mean(out[fin])):.3f} at a "
        "1.5 threshold; that is far too much participation under H0."
    )


# --------------------------------------------------------------------------- #
# `sieve_bootstrap_cv`
# --------------------------------------------------------------------------- #
def test_sieve_bootstrap_cv_is_seeded_and_ordered() -> None:
    fn = S.require_attr(S.panel, "sieve_bootstrap_cv", "_panel")
    mat = S.panel_matrix(n_entities=10, n_time=140, seed=203)  # (N, T)
    kw = {"min_window": 25, "lag": 0, "nboot": 60, "levels": LEVELS, "entity_nrep": 200}

    a = np.asarray(fn(mat, seed=0, **kw), dtype=np.float64)
    b = np.asarray(fn(mat, seed=0, **kw), dtype=np.float64)
    assert S.bit_identical(a, b), "sieve_bootstrap_cv is not deterministic at a seed"

    c = np.asarray(fn(mat, seed=1, **kw), dtype=np.float64)
    assert not S.bit_identical(a, c), "sieve_bootstrap_cv ignores its `seed`"

    assert a.shape[0] == len(LEVELS), (
        f"expected the leading axis to be the {len(LEVELS)} levels, got {a.shape}"
    )
    flat = a.reshape(len(LEVELS), -1)
    cols = np.isfinite(flat).all(axis=0)
    assert cols.any(), "every bootstrap critical value came back NaN"
    assert (flat[0, cols] <= flat[1, cols] + 1e-12).all()
    assert (flat[1, cols] <= flat[2, cols] + 1e-12).all()
    # The default statistic is `breadth`, a fraction.
    assert ((flat[:, cols] >= 0.0) & (flat[:, cols] <= 1.0)).all(), (
        "the default (breadth) bootstrap critical values left [0, 1]"
    )


def test_sieve_bootstrap_mean_critical_value_is_near_or_below_zero() -> None:
    """Assertion 8 again, this time through the shipped calibration routine."""
    fn = S.require_attr(S.panel, "sieve_bootstrap_cv", "_panel")
    bound = float(getattr(S.panel, "PANEL_MEAN_CV_UPPER_BOUND", 0.5))
    mat = S.panel_matrix(n_entities=15, n_time=150, seed=213)
    cv = np.asarray(
        fn(
            mat,
            min_window=25,
            lag=0,
            nboot=60,
            seed=3,
            levels=LEVELS,
            statistic="mean",
        ),
        dtype=np.float64,
    )
    flat = cv.reshape(len(LEVELS), -1)
    cols = np.isfinite(flat).all(axis=0)
    assert cols.any()
    peak = float(np.max(flat[1, cols]))
    assert peak <= bound, (
        f"the bootstrap 95% critical value for statistic='mean' peaks at "
        f"{peak:.3f}, above PANEL_MEAN_CV_UPPER_BOUND ({bound}). A panel mean "
        "cut-off in the single-series range (~2.2) means the cross-sectional "
        "average is not being formed before the quantile is taken."
    )


def test_sieve_bootstrap_reacts_to_cross_sectional_dependence() -> None:
    """Row resampling preserves the contemporaneous cross-sectional covariance.

    A panel with a dominant common factor has a very different null from a panel
    of independent series; if entities were resampled independently the
    dependence would be destroyed and the two calibrations would coincide.
    """
    fn = S.require_attr(S.panel, "sieve_bootstrap_cv", "_panel")
    rng = np.random.default_rng(204)
    n_ent, n_time = 12, 150
    common = rng.standard_normal(n_time)
    idio = rng.standard_normal((n_ent, n_time))

    correlated = np.cumsum(3.0 * common[None, :] + idio, axis=1)
    independent = np.cumsum(rng.standard_normal((n_ent, n_time)), axis=1)

    kw = {
        "min_window": 25,
        "lag": 0,
        "nboot": 80,
        "seed": 5,
        "levels": LEVELS,
        "entity_nrep": 200,
    }
    a = np.asarray(fn(correlated, **kw), dtype=np.float64)
    b = np.asarray(fn(independent, **kw), dtype=np.float64)
    fa = a.reshape(len(LEVELS), -1)
    fb = b.reshape(len(LEVELS), -1)
    cols = np.isfinite(fa).all(axis=0) & np.isfinite(fb).all(axis=0)
    assert cols.any()
    assert not S.bit_identical(fa[:, cols], fb[:, cols]), (
        "a strongly factor-driven panel and an independent one produced "
        "identical bootstrap critical values; the cross-sectional dependence is "
        "being destroyed, so whole rows are not being resampled."
    )
    assert float(np.mean(fa[1, cols])) > float(np.mean(fb[1, cols])), (
        f"the correlated panel's 95% breadth threshold "
        f"({float(np.mean(fa[1, cols])):.3f}) is not above the independent "
        f"panel's ({float(np.mean(fb[1, cols])):.3f}); common-factor dependence "
        "must widen the null, because the effective cross-section shrinks."
    )


# --------------------------------------------------------------------------- #
# `breadth`
# --------------------------------------------------------------------------- #
def test_breadth_counts_exceedances_and_is_monotone_in_the_critical_value() -> None:
    fn = S.require_attr(S.panel, "breadth", "_panel")
    rng = np.random.default_rng(205)
    n_ent, n_time = 12, 80
    stat = rng.standard_normal((n_ent, n_time))

    # A non-finite `cv` marks the entity/date as missing, so the saturating
    # cases use large finite thresholds instead.
    none = np.asarray(fn(stat, np.full(n_time, 1e12)), dtype=np.float64)
    allof = np.asarray(fn(stat, np.full(n_time, -1e12)), dtype=np.float64)
    assert none.shape == (n_time,), f"breadth returned {none.shape}, want ({n_time},)"
    assert np.allclose(none, 0.0), (
        "an unreachable critical value must give zero breadth"
    )
    assert np.allclose(allof, 1.0) or np.allclose(allof, float(n_ent)), (
        f"with an always-exceeded cv breadth returned {allof[0]}, expected 1.0 "
        f"(a fraction) or {n_ent} (a count)"
    )
    scale = 1.0 if np.allclose(allof, 1.0) else float(n_ent)

    cv = np.full(n_time, 0.5)
    got = np.asarray(fn(stat, cv), dtype=np.float64)
    want = (stat > cv).sum(axis=0) / n_ent * scale
    assert np.allclose(got, want), "breadth does not count strict exceedances"

    # Strict comparison: a statistic exactly at its critical value must not count.
    tied = np.full((n_ent, n_time), 1.0)
    assert np.allclose(np.asarray(fn(tied, 1.0), dtype=np.float64), 0.0), (
        "breadth counted a statistic exactly equal to its critical value"
    )

    prev = allof
    for level in (-1.0, 0.0, 1.0, 2.0):
        cur = np.asarray(fn(stat, np.full(n_time, level)), dtype=np.float64)
        assert (cur <= prev + 1e-12).all(), "breadth increased with the cut-off"
        prev = cur


def test_breadth_ignores_missing_entities_rather_than_counting_them() -> None:
    fn = S.require_attr(S.panel, "breadth", "_panel")
    n_ent, n_time = 8, 40
    stat = np.full((n_ent, n_time), 3.0)
    stat[:, :10] = np.nan  # every entity still in its warm-up
    cv = np.zeros(n_time)
    got = np.asarray(fn(stat, cv), dtype=np.float64)
    assert np.isnan(got[:10]).all() or np.allclose(got[:10], 0.0), (
        f"breadth over an all-NaN column returned {got[0]}; it must be NaN or 0, "
        "never a count of entities that have not started reporting."
    )
    assert np.isfinite(got[10:]).all()


# --------------------------------------------------------------------------- #
# `cross_sectional_rank`
# --------------------------------------------------------------------------- #
def test_cross_sectional_rank_is_a_per_date_rank_in_the_unit_interval() -> None:
    fn = S.require_attr(S.panel, "cross_sectional_rank", "_panel")
    rng = np.random.default_rng(206)
    stat = rng.standard_normal((10, 60))
    got = np.asarray(fn(stat), dtype=np.float64)

    assert got.shape == stat.shape
    fin = np.isfinite(got)
    assert fin.all()
    assert (got >= 0.0).all() and (got <= 1.0).all(), (
        f"ranks outside [0, 1]: min {got.min():.4f}, max {got.max():.4f}"
    )
    # Within each date, the rank order must match the statistic order.
    for t in range(stat.shape[1]):
        assert np.array_equal(np.argsort(got[:, t]), np.argsort(stat[:, t]))


def test_cross_sectional_rank_never_pools_across_dates() -> None:
    """Editing one date must leave every other date bit-identical."""
    fn = S.require_attr(S.panel, "cross_sectional_rank", "_panel")
    rng = np.random.default_rng(207)
    stat = rng.standard_normal((10, 60))
    base = np.asarray(fn(stat), dtype=np.float64)

    edited = stat.copy()
    edited[:, 30] *= 1e6
    got = np.asarray(fn(edited), dtype=np.float64)

    keep = [t for t in range(60) if t != 30]
    assert S.bit_identical(base[:, keep], got[:, keep]), (
        "blowing up a single date changed the ranks on other dates; the rank is "
        "being taken over the pooled panel, which leaks both across time and "
        "across the cross-section."
    )


def test_cross_sectional_rank_is_invariant_to_monotone_rescaling() -> None:
    fn = S.require_attr(S.panel, "cross_sectional_rank", "_panel")
    rng = np.random.default_rng(208)
    stat = rng.standard_normal((10, 60))
    a = np.asarray(fn(stat), dtype=np.float64)
    b = np.asarray(fn(np.exp(stat)), dtype=np.float64)
    assert S.bit_identical(a, b), "a rank must be invariant to a monotone transform"


def test_cross_sectional_rank_is_equivariant_to_entity_order() -> None:
    fn = S.require_attr(S.panel, "cross_sectional_rank", "_panel")
    rng = np.random.default_rng(209)
    stat = rng.standard_normal((10, 60))
    perm = rng.permutation(10)
    a = np.asarray(fn(stat), dtype=np.float64)[perm]
    b = np.asarray(fn(stat[perm]), dtype=np.float64)
    assert S.bit_identical(a, b)


# --------------------------------------------------------------------------- #
# `residualise`
# --------------------------------------------------------------------------- #
MIN_PERIODS = 40


@pytest.fixture(scope="module")
def residualise_case():
    """A ``(N, T)`` panel with one dominant common factor, plus its residuals."""
    fn = S.require_attr(S.panel, "residualise", "_panel")
    rng = np.random.default_rng(210)
    n_ent, n_time = 8, 200
    common = rng.standard_normal(n_time)
    returns = 1.5 * common[None, :] + rng.standard_normal((n_ent, n_time))
    out = np.asarray(
        fn(returns, n_factors=1, min_periods=MIN_PERIODS), dtype=np.float64
    )
    assert out.shape == returns.shape, (
        f"residualise returned {out.shape} for a {returns.shape} input"
    )
    return fn, returns, out


def test_residualise_uses_the_modules_entity_major_layout(residualise_case) -> None:
    """Warm-up NaNs must sit on the time axis, which is axis 1 across ``detect``."""
    _, _, out = residualise_case
    assert np.isnan(out[:, :MIN_PERIODS]).all(), (
        "the first min_periods *columns* are not NaN, so time is not on axis 1. "
        "`bsadf_panel`, `breadth` and `cross_sectional_rank` all take (N, T); a "
        "transposed `residualise` would silently residualise across time."
    )
    assert not np.isnan(out[0, MIN_PERIODS:]).all()


def test_residualise_respects_min_periods(residualise_case) -> None:
    _, _, out = residualise_case
    assert np.isnan(out[:, :MIN_PERIODS]).all(), (
        f"the first {MIN_PERIODS} periods must be NaN: a PCA fitted on fewer "
        "than min_periods observations is not defined."
    )
    assert np.isfinite(out[:, MIN_PERIODS:]).all()


def test_residualise_betas_are_frozen_at_t(residualise_case) -> None:
    """Truncating the future must not move a single past residual."""
    fn, returns, out = residualise_case
    for cut in (80, 120, 160):
        part = np.asarray(
            fn(returns[:, :cut], n_factors=1, min_periods=MIN_PERIODS),
            dtype=np.float64,
        )
        assert S.bit_identical(out[:, :cut], part), (
            f"residuals on returns[:, :{cut}] differ from the same columns of "
            f"the full sample (max dev "
            f"{S.max_abs_dev(out[:, :cut], part):.3g}). The PCA is not "
            "backward-looking: betas must be frozen using data up to t only."
        )


def test_residualise_refit_cadence_is_anchored_to_index_zero(residualise_case) -> None:
    """Refits happen at ``min_periods + k * refit_every``, never at ``len(y)``."""
    fn, returns, _ = residualise_case
    a = np.asarray(
        fn(returns, n_factors=1, min_periods=MIN_PERIODS, refit_every=21),
        dtype=np.float64,
    )
    b = np.asarray(
        fn(returns, n_factors=1, min_periods=MIN_PERIODS, refit_every=5),
        dtype=np.float64,
    )
    assert not S.bit_identical(a, b), "`refit_every` had no effect at all"
    # The first block is identical either way: both refit exactly at min_periods.
    first = slice(MIN_PERIODS, MIN_PERIODS + 5)
    assert S.bit_identical(a[:, first], b[:, first]), (
        "the two cadences disagree inside their common first block, so the "
        "refit schedule is not anchored at index 0."
    )


def test_residualise_is_equivariant_to_entity_order(residualise_case) -> None:
    fn, returns, out = residualise_case
    perm = np.random.default_rng(211).permutation(returns.shape[0])
    got = np.asarray(
        fn(returns[perm], n_factors=1, min_periods=MIN_PERIODS), dtype=np.float64
    )
    dev = S.max_abs_dev(out[perm], got)
    assert dev < 1e-9, (
        f"permuting the entity rows changed the residuals by {dev:.3g}; the "
        "principal subspace is permutation-equivariant, so they must not move."
    )


def test_residualise_removes_the_common_factor(residualise_case) -> None:
    _, returns, out = residualise_case
    raw = returns[:, MIN_PERIODS + 20 :]
    res = out[:, MIN_PERIODS + 20 :]

    def _mean_abs_offdiag_corr(x: np.ndarray) -> float:
        c = np.corrcoef(x, rowvar=True)
        off = ~np.eye(c.shape[0], dtype=bool)
        return float(np.mean(np.abs(c[off])))

    before = _mean_abs_offdiag_corr(raw)
    after = _mean_abs_offdiag_corr(res)
    assert before > 0.4, f"the fixture lost its common factor (corr {before:.3f})"
    assert after < 0.5 * before, (
        f"mean |cross-correlation| only fell from {before:.3f} to {after:.3f}; "
        "the first principal component is not being projected out."
    )
