"""Golden-baseline parity tests for the Wave 2/3 vectorisation round.

Every numeric change in that round (catch22 loop removal, the scipy-free
Chebyshev neighbour counter behind sample/approximate entropy, the batched
k-Shape NCC, and the single-conversion ``CrossSectionalPCA`` transform) was made
under one rule: **capture the output first, then assert the optimised code
reproduces it**.

``tests/data/perf_parity_golden.json`` is that capture -- produced by the
pre-change implementations on a fixed bank of synthetic series -- and these
tests are the ratchet that keeps the optimised code pinned to it.  Most of the
comparisons are bit-exact; the few that are not carry a documented tolerance and
a reason.

A note on what "bit-exact" can and cannot mean here.  A comparison against a
*stored* baseline is only reproducible bit-for-bit where the arithmetic is
platform-independent.  Pure-NumPy elementwise reductions are; anything routed
through the platform's BLAS/LAPACK (``dot``, ``eigh``) or through pocketfft is
not, because the CI matrix spans Linux/macOS/Windows and therefore OpenBLAS and
Accelerate.  Comparisons in that second category use an explicit tolerance
derived from the conditioning of the step involved, never a "whatever makes it
pass" epsilon.  The bit-exactness of the *vectorisation itself* is still pinned,
but by same-process tests (``test_batched_distance_matrix_matches_pairwise_ncc``)
where both sides run on the same LAPACK.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import polars as pl
import pytest

import panelary.catch22 as c22
from panelary import feature_extractors as fe
from panelary._internal._numpy_stats import chebyshev_neighbour_counts
from panelary.cluster._kshape import KShapeCore, _distance_matrix, ncc

GOLDEN_PATH = pathlib.Path(__file__).parent / "data" / "perf_parity_golden.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    with GOLDEN_PATH.open() as fh:
        return json.load(fh)


def series_bank() -> dict[str, np.ndarray]:
    """The fixed series the golden baseline was captured on.

    Deliberately varied: gaussian noise, random walks, a noisy sine, a smoothed
    AR-ish series, heavy tails, a run-constant series (heavy ties, which is the
    interesting case for the count-based rewrites), a short series and a trend.
    """
    rng = np.random.default_rng(20260908)
    bank: dict[str, np.ndarray] = {}
    bank["gauss_256"] = rng.standard_normal(256)
    bank["walk_512"] = rng.standard_normal(512).cumsum()
    bank["walk_1024"] = rng.standard_normal(1024).cumsum()
    t = np.arange(700)
    bank["sine_700"] = np.sin(2 * np.pi * t / 37.0) + 0.3 * rng.standard_normal(700)
    bank["ar1_800"] = np.convolve(rng.standard_normal(900), np.ones(5) / 5)[:800]
    bank["heavy_tail_400"] = rng.standard_t(2.5, 400)
    bank["const_run_300"] = np.repeat(rng.standard_normal(30), 10)
    bank["short_40"] = rng.standard_normal(40)
    bank["trend_600"] = np.linspace(0, 5, 600) + 0.5 * rng.standard_normal(600)
    bank["spiky_500"] = rng.standard_normal(500)
    bank["spiky_500"][::37] += 8.0
    return bank


def _same(got: float, ref: float) -> bool:
    if ref is None or got is None:
        return got is ref
    if np.isnan(ref) and np.isnan(got):
        return True
    if np.isinf(ref) and np.isinf(got):
        return np.sign(ref) == np.sign(got)
    return bool(got == ref)


# --------------------------------------------------------------------------- #
# catch22 -- all 24 features, all series
# --------------------------------------------------------------------------- #
#: Empty, and it should stay that way.  `PD_PeriodicityWang_th0_01` used to live
#: here: it spline-detrended via `scipy.interpolate.LSQUnivariateSpline` and
#: silently fell back to a *zero* spline when SciPy was absent, changing the
#: answer (140.0 -> 0.0 on `walk_512`).  The spline is now fitted in pure NumPy
#: by `catch22._lsq_spline_fit`, so every feature is SciPy-independent and the
#: golden baseline -- captured with SciPy present -- applies unconditionally.
#: See `test_lsq_spline_matches_scipy`.
_SCIPY_DEPENDENT_FEATURES: frozenset[str] = frozenset()


# --------------------------------------------------------------------------- #
# How each catch22 feature is compared, and why.
#
# The rewrite this baseline guards (loops -> `flatnonzero`/`bincount`/sliding
# windows) preserved the arithmetic exactly, so on one machine the outputs are
# bit-identical.  What is *not* portable is float64 reduction order: NumPy's
# pairwise summation, its BLAS, and pocketfft all dispatch on CPU features and
# build flags, and the CI matrix spans Linux/macOS/Windows.  Re-running this
# exact fixture bank under NumPy 1.26.4 (OpenBLAS) instead of NumPy 2.5.3
# (Accelerate) moves 17 of the 240 values, by 1 to 32 ulps -- 1.2e-16 to 5.1e-15
# relative.  None of them is a wrong answer; all are last-bit drift.
#
# So each feature is compared according to what it actually is.  Anything whose
# output is a *decision* (a lag, a run length, a count over n) stays exact --
# those cannot drift, and if one ever does it is a real change, not noise.
# Everything else gets a tolerance derived from that specific kernel's error
# analysis.  The derivations below use eps = 2.22e-16 and n <= 1024, so
# pairwise summation contributes <= eps*log2(n) ~ 2.2e-15 relative and an FFT of
# length <= 2048 contributes ~ eps*log2(2048) ~ 2.4e-15; a feature composes a
# handful of these, so ~1e-14 relative is the generic ceiling.
# --------------------------------------------------------------------------- #

#: Features whose value is a discrete decision, compared **exactly**.  Either an
#: integer (a lag, a period, a run length) or an exact ratio of two integers
#: (a count over n, a ratio of two zero-crossing lags) -- in IEEE754 both are
#: reproduced bit-for-bit on every platform once the underlying comparison goes
#: the same way, and `test_catch22_discrete_decisions_are_not_close_calls`
#: shows those comparisons clear their thresholds by >= 5.6e-5, i.e. ten orders
#: of magnitude more than the drift above.  A mismatch here is therefore a real
#: behavioural change and must not be absorbed into a tolerance.
_CATCH22_EXACT: frozenset[str] = frozenset(
    {
        "CO_FirstMin_ac",  # lag of the first ACF local minimum
        "IN_AutoMutualInfoStats_40_gaussian_fmmi",  # lag of the first AMI minimum
        "PD_PeriodicityWang_th0_01",  # detected period
        "SB_BinaryStats_diff_longstretch0",  # run length
        "SB_BinaryStats_mean_longstretch1",  # run length
        "MD_hrv_classic_pnn40",  # count / (n - 1)
        "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",  # count / n_scales
        "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",  # count / n_scales
        "FC_LocalSimple_mean1_tauresrat",  # ratio of two integer lags
        "DN_OutlierInclude_n_001_mdrmd",  # median of index/n ratios
        "DN_OutlierInclude_p_001_mdrmd",  # median of index/n ratios
    }
)

#: Generic relative tolerance for the well-conditioned continuous features --
#: means, standard deviations, spectra, entropies.  Bound above: ~1e-14
#: relative.  1e-12 is ~100x that and ~1e12 short of any change of formula.
#: Measured worst drift across the bank under a different BLAS: 4.5e-16.
CATCH22_REDUCTION_RTOL = 1e-12

#: Absolute floor paired with the relative tolerance, for the features whose
#: value legitimately passes near zero (`DN_Mean` of a centred series,
#: `SB_TransitionMatrix_3ac_sumdiagcov`), where a purely relative bound is
#: meaningless.  Set to the *absolute* error bound of an O(1) z-scored
#: reduction, eps*log2(n) ~ 2.2e-15, rounded up.
CATCH22_REDUCTION_ATOL = 1e-14

#: `CO_trev_1_num` = mean((y[t+1] - y[t])**3) is a **cancelling** sum, so it is
#: the one feature where a relative tolerance is the wrong instrument.  Measured
#: cancellation kappa = sum|d^3| / |sum d^3| over this bank reaches 1871
#: (`sine_700`, an almost time-reversible series), so the relative error bound
#: is kappa*eps*log2(n) = 4.2e-12 -- CATCH22_REDUCTION_RTOL would be *tighter
#: than the arithmetic*, and would fail on a runner that rounds the other way.
#: The honest bound is absolute and set by the summands, not the answer:
#: eps*log2(n) * mean|d^3| <= 2.2e-15 * 6.3 = 1.4e-14.  1e-12 is ~70x that.
#: Measured worst drift: 3.3e-16.
CATCH22_TREV_ATOL = 1e-12

#: `DN_HistogramMode_5` / `_10` return a bin *centre*, so the scale that governs
#: their rounding is the z-scored range (max|y| = 6.29 over this bank), not the
#: returned value.  That is why the observed 4.4e-16 drift looks alarming as
#: "16 ulps of the answer" (the answer is -0.224) and is in fact sub-ulp of the
#: quantity that produced it.  Bound: eps*log2(n) * max|y| = 1.4e-14; 1e-12 is
#: ~70x.  Note this tolerance is *not* how a bin-boundary flip is caught -- see
#: `test_catch22_discrete_decisions_are_not_close_calls`, which shows no
#: observation lies within 1.2e-4 of a bin edge, so the counts are identical on
#: every platform.  Were a bin to flip anyway, the answer would move by roughly
#: half a bin width (~0.4 here), 1e12 times this tolerance, and the assertion
#: below would fire exactly as it should.
CATCH22_HISTOGRAM_MODE_ATOL = 1e-12

#: `CO_f1ecac` interpolates the 1/e crossing as `i + (1/e - acf[i]) / slope`,
#: which amplifies ACF noise by `1 / |slope|`.  The crossing *index* is safe
#: (the ACF clears 1/e by >= 1.1e-3 there), but the interpolation is not: the
#: flattest crossing in this bank has |slope| = 3.9e-3 (`walk_1024`), so a delta
#: of 16*eps = 3.6e-15 on the ACF becomes 9.1e-13 on the answer.  This is the
#: one feature whose bound exceeds CATCH22_REDUCTION_RTOL, which is why it gets
#: its own number; 1e-11 leaves ~11x.  Measured worst drift: 1.4e-14.
CATCH22_F1ECAC_ATOL = 1e-11


def _catch22_tolerance(key: str) -> tuple[float, float]:
    """Return ``(rtol, atol)`` for one catch22 feature; ``(0.0, 0.0)`` is exact."""
    if key in _CATCH22_EXACT:
        return 0.0, 0.0
    if key == "CO_trev_1_num":
        return 0.0, CATCH22_TREV_ATOL
    if key.startswith("DN_HistogramMode_"):
        return 0.0, CATCH22_HISTOGRAM_MODE_ATOL
    if key == "CO_f1ecac":
        return 0.0, CATCH22_F1ECAC_ATOL
    return CATCH22_REDUCTION_RTOL, CATCH22_REDUCTION_ATOL


def _within(got: float, ref: float, rtol: float, atol: float) -> bool:
    """``_same`` for the non-finite cases, an explicit tolerance otherwise."""
    if got is None or ref is None:
        return _same(got, ref)
    if not (np.isfinite(ref) and np.isfinite(got)):
        return _same(got, ref)
    if rtol == 0.0 and atol == 0.0:
        return bool(got == ref)
    return bool(abs(got - ref) <= atol + rtol * abs(ref))


def test_catch22_exact_set_is_spelt_correctly(golden):
    """A typo in `_CATCH22_EXACT` would silently downgrade a feature to a tolerance.

    Cheap guard against the failure mode that would quietly remove the teeth:
    every name in the exact set must actually be a catch22 feature.
    """
    keys = set(golden["catch22"]["gauss_256"])
    assert keys >= _CATCH22_EXACT, sorted(_CATCH22_EXACT - keys)


@pytest.mark.parametrize("name", sorted(series_bank()))
def test_catch22_matches_golden(name, golden):
    """The vectorised catch22 helpers reproduce the loop versions.

    Exactly for the features that are decisions; to a per-kernel tolerance for
    the ones that are float64 reductions.  See the block comment above for the
    classification and every tolerance's derivation.
    """
    arr = series_bank()[name]
    got = c22.catch22_all(arr, catch24=True)
    reference = golden["catch22"][name]
    assert set(got) == set(reference)
    skip = _SCIPY_DEPENDENT_FEATURES
    mismatched = {}
    for key in reference:
        if key in skip:
            continue
        rtol, atol = _catch22_tolerance(key)
        if not _within(got[key], reference[key], rtol, atol):
            allowed = atol + rtol * abs(reference[key] or 0.0)
            mismatched[key] = (
                f"golden {reference[key]!r} -> got {got[key]!r} "
                f"(drift {abs(got[key] - reference[key]):.3e} > allowed {allowed:.3e})"
                if isinstance(reference[key], float) and isinstance(got[key], float)
                else f"golden {reference[key]!r} -> got {got[key]!r}"
            )
    assert not mismatched, f"catch22 drift on {name}: {mismatched}"


#: Floors for the discrete decisions inside catch22, in the natural units of
#: each comparison.  Every one of these is enforced by
#: `test_catch22_discrete_decisions_are_not_close_calls`; the measured value on
#: this fixture bank is quoted beside it.  They exist so that comparing the
#: discrete features exactly is a *justified* choice rather than a lucky one:
#: if a future edit to `series_bank` produced a series that straddles one of
#: these thresholds, the exact comparisons would become a platform coin-flip,
#: and this test says so directly instead of leaving a mystifying red build on
#: one runner.  All are ~1e10 times the ~1e-15 last-bit drift they must survive.
_HISTOGRAM_EDGE_CLEARANCE_FLOOR = 1e-6  # measured 1.240e-4 (spiky_500)
_PNN40_THRESHOLD_MARGIN_FLOOR = 1e-6  # measured 5.646e-5 (walk_512)
_ACF_ZERO_MARGIN_FLOOR = 1e-6  # measured 5.346e-4 (walk_1024)
_OUTLIER_BIN_MARGIN_FLOOR = 1e-6  # measured 1.653e-2 (const_run_300)
_F1ECAC_CROSSING_MARGIN_FLOOR = 1e-6  # measured 1.081e-3 (walk_1024)
#: The slope floor is different in kind: it is not a tie risk, it is the
#: conditioning number that `CATCH22_F1ECAC_ATOL` is derived from.  Measured
#: 3.920e-3 (walk_1024).  If it ever drops, that tolerance must be recomputed.
_F1ECAC_SLOPE_FLOOR = 1e-3


@pytest.mark.parametrize("name", sorted(series_bank()))
def test_catch22_discrete_decisions_are_not_close_calls(name):
    """Nothing in catch22 decides a discrete question by a hair, on this bank.

    This is the companion to `_CATCH22_EXACT`, and the reason a bin-boundary
    flip is not something a tolerance is asked to paper over.  Each check below
    is the actual comparison the implementation makes, measured in its own
    units; a value near zero would mean the feature's output depends on which
    way the platform's last bit rounds.
    """
    y = c22._zscore(series_bank()[name])

    # DN_HistogramMode_5 / _10: an observation crossing a bin edge would change
    # the counts, which is the only way the mode can move by more than rounding.
    for n_bins in (5, 10):
        _, edges = np.histogram(y, bins=n_bins)
        interior = edges[1:-1]
        clearance = float(np.abs(y[:, None] - interior[None, :]).min())
        assert clearance >= _HISTOGRAM_EDGE_CLEARANCE_FLOOR, (
            f"{name}: an observation sits {clearance:.3e} from a {n_bins}-bin "
            "histogram edge, so the bin counts -- and therefore the reported "
            "mode -- are no longer platform-independent."
        )

    # MD_hrv_classic_pnn40: fraction of |diff| over a hard 0.04 threshold.
    diffs = np.abs(np.diff(y))
    margin = float(np.abs(diffs - 0.04).min())
    assert margin >= _PNN40_THRESHOLD_MARGIN_FLOOR, (
        f"{name}: a successive difference is {margin:.3e} from the pNN40 "
        "threshold; the count is no longer reproducible."
    )

    # DN_OutlierInclude_*: `n_thresh = int(max_val / 0.01) + 1` truncates, so a
    # max sitting on an exact multiple of the increment is a flip risk.
    for sign in (1, -1):
        scaled = float((sign * y).max()) / 0.01
        frac = abs(scaled - round(scaled))
        assert frac >= _OUTLIER_BIN_MARGIN_FLOOR, (
            f"{name}: the sign={sign} extreme is {frac:.3e} of an increment "
            "from an exact threshold boundary; `int()` may truncate either way."
        )

    # FC_LocalSimple_mean1_tauresrat / CO_FirstMin_ac: first ACF zero crossing.
    acf = c22._acf(y)
    negative = np.flatnonzero(acf < 0)
    if negative.size:
        i = int(negative[0])
        crossing = float(min(abs(acf[i - 1]), abs(acf[i])))
        assert crossing >= _ACF_ZERO_MARGIN_FLOOR, (
            f"{name}: the ACF passes within {crossing:.3e} of zero at its first "
            "sign change; the first-zero lag could differ by one."
        )

    # CO_f1ecac: the 1/e crossing index, plus the slope its interpolation is
    # divided by -- the conditioning that CATCH22_F1ECAC_ATOL is derived from.
    below = np.flatnonzero(acf[1:] < 1.0 / np.e)
    if below.size:
        i = int(below[0])
        margin = float(min(abs(acf[i] - 1.0 / np.e), abs(acf[i + 1] - 1.0 / np.e)))
        assert margin >= _F1ECAC_CROSSING_MARGIN_FLOOR, (
            f"{name}: the ACF sits {margin:.3e} from 1/e at the crossing; the "
            "crossing index itself could differ between platforms."
        )
        slope = float(abs(acf[i + 1] - acf[i]))
        assert slope >= _F1ECAC_SLOPE_FLOOR, (
            f"{name}: the ACF slope at the 1/e crossing is {slope:.3e}, so the "
            f"interpolation amplifies ACF noise by {1 / slope:.0f}x -- "
            "CATCH22_F1ECAC_ATOL was derived assuming at least "
            f"{_F1ECAC_SLOPE_FLOOR:.0e} and must be recomputed."
        )


def test_lsq_spline_matches_scipy():
    """The NumPy least-squares spline reproduces SciPy's, so catch22 is SciPy-free.

    `PD_PeriodicityWang_th0_01` detrends with a cubic LSQ spline on 3 interior
    knots.  That used to come from `scipy.interpolate.LSQUnivariateSpline`, with
    a zero-spline fallback that silently changed the feature's value on the bare
    numpy+polars core.  `catch22._lsq_spline_fit` now does it in NumPy; this
    test is the parity evidence, checked against SciPy where SciPy is installed.
    """
    LSQUnivariateSpline = pytest.importorskip("scipy.interpolate").LSQUnivariateSpline
    rng = np.random.default_rng(0)
    worst = 0.0
    for n in (8, 9, 13, 50, 100, 512, 1000, 2048):
        for series in (
            rng.standard_normal(n),
            np.cumsum(rng.standard_normal(n)),
            np.sin(np.arange(n) / 7.0) + 0.01 * rng.standard_normal(n),
            np.arange(n, dtype=float) ** 1.5,
        ):
            y = (series - series.mean()) / (series.std(ddof=1) or 1.0)
            t = np.arange(n, dtype=float)
            interior = np.linspace(0, n - 1, 5)[1:-1]
            reference = LSQUnivariateSpline(t, y, interior, k=3)(t)
            got = c22._lsq_spline_fit(t, y, interior, k=3)
            worst = max(worst, float(np.max(np.abs(reference - got))))
    assert worst < 1e-10, f"NumPy LSQ spline drifted from SciPy by {worst:.3e}"


def test_periodicity_wang_is_scipy_independent():
    """The feature returns the same number whether or not SciPy is importable.

    Guards the specific regression that motivated the NumPy spline: a silent
    dependency on an optional extra changing a feature's *value*.
    """
    import builtins

    series = series_bank()["walk_512"]
    with_scipy = c22.PD_PeriodicityWang_th0_01(series)

    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.split(".")[0] == "scipy":
            raise ImportError("scipy blocked for this test")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = no_scipy
    try:
        without_scipy = c22.PD_PeriodicityWang_th0_01(series)
    finally:
        builtins.__import__ = real_import

    assert with_scipy == without_scipy


# --------------------------------------------------------------------------- #
# sample / approximate entropy -- scipy-free counter, bit-exact
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(series_bank()))
def test_entropy_matches_golden_bitwise(name, golden):
    arr = series_bank()[name]
    s = pl.Series(name, arr)
    reference = golden["entropy"][name]
    for key, ref in reference.items():
        if isinstance(ref, str):  # the capture recorded an error for this case
            continue
        if key.startswith("sampen_m"):
            with np.errstate(divide="ignore", invalid="ignore"):
                got = float(fe.sample_entropy(s, ratio=0.2, m=int(key[-1])))
        else:
            _, run_length, level, scaled = key.split("_")
            got = float(
                fe.approximate_entropy(
                    s,
                    run_length=int(run_length),
                    filtering_level=float(level),
                    scale_by_std=scaled == "True",
                )
            )
        assert _same(got, ref), f"{name}/{key}: {ref} -> {got}"


def test_chebyshev_counts_match_kdtree():
    """The numpy neighbour counter is exact against SciPy's k-d tree.

    Both answer ``max_d |x_d - y_d| <= r`` on the same floats, so the counts are
    equal as integers -- no tolerance involved.  SciPy stays the preferred
    backend for speed (see `feature_extractors._chebyshev_counter`); this test
    pins the fallback that makes the feature work in the bare core.
    """
    spatial = pytest.importorskip("scipy.spatial")
    rng = np.random.default_rng(11)
    for trial in range(25):
        n = int(rng.integers(1, 300))
        dim = int(rng.integers(1, 4))
        scale = float(rng.choice([0.01, 1.0, 100.0]))
        points = rng.standard_normal((n, dim)) * scale
        if trial % 4 == 0:  # force heavy ties
            points = np.round(points, 0)
        radius = float(rng.uniform(0.01, 2.0)) * scale
        reference = np.asarray(
            spatial.KDTree(points).query_ball_point(
                points, radius, p=np.inf, return_length=True
            )
        ).ravel()
        assert np.array_equal(chebyshev_neighbour_counts(points, radius), reference)


def test_chebyshev_counts_edge_cases():
    assert chebyshev_neighbour_counts(np.empty((0, 2)), 1.0).size == 0
    single = chebyshev_neighbour_counts(np.array([[1.0, 2.0]]), 0.5)
    assert single.tolist() == [1]
    # 1-D input is treated as one dimension
    flat = chebyshev_neighbour_counts(np.array([0.0, 0.4, 5.0]), 0.5)
    assert flat.tolist() == [2, 2, 1]


# --------------------------------------------------------------------------- #
# k-Shape -- batched NCC.  Bit-exact against the pairwise definition in-process;
# LAPACK/FFT tolerance against the stored baseline (see the module docstring).
# --------------------------------------------------------------------------- #
def _kshape_fixture():
    rng = np.random.default_rng(7)
    return rng.standard_normal((24, 96)).cumsum(axis=1)


#: Absolute ceiling on the LAPACK-level drift of a k-Shape centroid between
#: BLAS implementations.  Derived, not tuned:
#:
#: ``_extract_shape`` takes the top eigenvector of ``M = P S P`` where
#: ``S = Y^T Y`` comes out of a BLAS ``syrk``.  Davis-Kahan bounds the angle the
#: eigenvector can rotate under a perturbation ``E`` of ``M`` by
#: ``sin(theta) <= ||E|| / gap``; a different BLAS gives ``||E|| ~ eps * ||M||``.
#: Measured over this fixture's twelve ``eigh`` calls: ``||M||_2`` in
#: 3.2e2..8.5e2 and the leading eigengap in 2.0e2..8.2e2, so
#: ``sin(theta) <= 3.9e-16``.  The centroid is that unit eigenvector z-scored to
#: norm ``sqrt(95) = 9.75``, so an element may move by ``9.75 * 3.9e-16 =
#: 3.8e-15`` per iteration, and the Lloyd loop runs four of them.
#:
#: Confirmed empirically: rerunning this exact fixture against NumPy 1.26.4
#: (OpenBLAS) instead of NumPy 2.5 (Accelerate) moves 225 of the 288 centroid
#: entries, by at most 6.8e-15.
#:
#: 1e-12 sits ~150x above that and ~1e11 below any change of *algorithm* -- a
#: different alignment shift, a different eigenvector, or an unbatched/batched
#: mismatch all move a centroid by O(1e-2) or more.  It is a ratchet, not a
#: rubber stamp.
KSHAPE_CENTROID_ATOL = 1e-12

#: Same reasoning propagated to the SBD matrix.  ``d = 1 - max_shift NCC`` and
#: ``|dNCC| <= ||delta_c|| / ||c||``, so the centroid drift above contributes
#: ``sqrt(96) * 3.8e-15 / 9.75 = 3.8e-15``, on top of pocketfft's own
#: ``eps * log2(256) = 1.8e-15``.  Measured cross-BLAS drift: 4.4e-16.
KSHAPE_DISTANCE_ATOL = 1e-12

#: The clustering is only allowed to be compared *exactly* because its discrete
#: decisions are not close calls.  This is the measured floor on
#: ``second-best minus best`` SBD over every assignment round of the fixture
#: (min 3.8e-5, 5th percentile 3.9e-3), i.e. ten orders of magnitude above the
#: drift above.  ``test_kshape_labels_are_not_a_close_call`` enforces it, so the
#: justification for exact label equality is itself under test.
KSHAPE_MIN_ASSIGNMENT_MARGIN = 1e-6


def test_kshape_fit_matches_golden(golden):
    """Batching the forward transforms does not change what ``fit`` learns.

    Labels are compared **exactly**: they are discrete, and the assignment is
    not a close call (see ``KSHAPE_MIN_ASSIGNMENT_MARGIN``), so a label flip
    here means a genuinely different clustering, never rounding.

    Centroids are compared to ``KSHAPE_CENTROID_ATOL``.  They are eigenvectors
    produced by LAPACK ``eigh`` on a BLAS-formed scatter matrix, so a stored
    baseline cannot be bit-exact across the OpenBLAS/Accelerate CI matrix; the
    tolerance is derived from Davis-Kahan at this fixture's conditioning, not
    from what makes the test pass.
    """
    core = KShapeCore(3, seed=11, max_iter=20).fit(_kshape_fixture())
    reference = golden["kshape"]
    assert core.labels_.tolist() == reference["labels"]
    got = np.asarray(core.centroids_).ravel()
    ref = np.asarray(reference["centroids"])
    assert got.shape == ref.shape
    drift = float(np.max(np.abs(got - ref)))
    assert drift <= KSHAPE_CENTROID_ATOL, (
        f"k-Shape centroids drifted from the golden baseline by {drift:.3e}, "
        f"over the {KSHAPE_CENTROID_ATOL:.0e} LAPACK-noise ceiling -- that is "
        "too large to be a BLAS difference, so treat it as a real change."
    )


def test_kshape_is_deterministic():
    """The seed is the only source of randomness; two fits agree bit-for-bit.

    Rules out the other explanation for a golden mismatch: unseeded RNG.
    ``KShapeCore(seed=...)`` threads a ``default_rng`` through both the initial
    assignment and empty-cluster reseeding, so repeated fits in one process are
    identical, and the same seed must give the same answer every time.
    """
    a = KShapeCore(3, seed=11, max_iter=20).fit(_kshape_fixture())
    b = KShapeCore(3, seed=11, max_iter=20).fit(_kshape_fixture())
    assert a.labels_.tolist() == b.labels_.tolist()
    assert np.array_equal(np.asarray(a.centroids_), np.asarray(b.centroids_))


def test_kshape_labels_are_not_a_close_call():
    """The margin that licenses comparing labels exactly.

    If a future change made the fixture's assignments near-ties, exact label
    equality would become a platform coin-flip and this test would say so
    *before* the golden comparison starts failing mysteriously on one runner.
    """
    core = KShapeCore(3, seed=11, max_iter=20).fit(_kshape_fixture())
    dist = _distance_matrix(KShapeCore._as3d(_kshape_fixture()), core.centroids_)
    ordered = np.sort(dist, axis=1)
    margin = float(np.min(ordered[:, 1] - ordered[:, 0]))
    assert margin >= KSHAPE_MIN_ASSIGNMENT_MARGIN, (
        f"nearest/second-nearest SBD differ by only {margin:.3e}; the fixture's "
        "clustering is now a near-tie, so exact label comparison against the "
        "golden baseline is no longer portable."
    )


def test_kshape_distance_matrix_matches_golden(golden):
    core = KShapeCore(3, seed=11, max_iter=20).fit(_kshape_fixture())
    dist = _distance_matrix(KShapeCore._as3d(_kshape_fixture()), core.centroids_)
    ref = np.asarray(golden["kshape"]["distances"])
    assert dist.size == ref.size
    drift = float(np.max(np.abs(dist.ravel() - ref)))
    assert drift <= KSHAPE_DISTANCE_ATOL, (
        f"k-Shape SBD matrix drifted from the golden baseline by {drift:.3e}, "
        f"over the {KSHAPE_DISTANCE_ATOL:.0e} FFT/LAPACK-noise ceiling."
    )


def test_batched_distance_matrix_matches_pairwise_ncc():
    """Independent, bit-exact check that batching changed nothing.

    This is where the vectorisation claim is actually pinned: both sides are
    computed in the same process on the same LAPACK/pocketfft, so ``==`` is the
    right comparison and no tolerance is warranted.  It is also what makes the
    tolerances in the two golden comparisons above safe -- those absorb only
    cross-platform noise, never a change in the batched arithmetic.
    """
    X = KShapeCore._as3d(_kshape_fixture())
    core = KShapeCore(3, seed=11, max_iter=20).fit(_kshape_fixture())
    centroids = np.asarray(core.centroids_)
    pairwise = np.array(
        [
            [1.0 - ncc(X[p], centroids[q]).max() for q in range(centroids.shape[0])]
            for p in range(X.shape[0])
        ]
    )
    assert np.array_equal(_distance_matrix(X, centroids), pairwise)


# --------------------------------------------------------------------------- #
# CrossSectionalPCA -- same per-date arithmetic, LAPACK-level tolerance
# --------------------------------------------------------------------------- #
def _xs_fixture() -> pl.DataFrame:
    rng = np.random.default_rng(3)
    n_ent, n_t, n_f = 40, 30, 6
    entities = np.repeat([f"e{i}" for i in range(n_ent)], n_t)
    times = np.tile(np.arange(n_t), n_ent)
    feats = {f"f{j}": rng.standard_normal(n_ent * n_t) for j in range(n_f)}
    feats["f0"][::53] = np.nan
    frame = pl.DataFrame({"entity": entities, "time": times, **feats})
    # Drop three entities on one date so the panel is genuinely unbalanced.
    return frame.filter(
        ~((pl.col("time") == 5) & (pl.col("entity").is_in(["e0", "e1", "e2"])))
    )


@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("sign_fix", [True, False])
def test_cross_sectional_pca_matches_golden(standardize, sign_fix, golden):
    """Hoisting the per-date polars conversion leaves the scores unchanged.

    Tolerance rather than bit-exact: the per-date matrix is now a slice of one
    big ``to_numpy()`` instead of a per-group one, and LAPACK's blocking makes
    the last couple of digits of the decomposition depend on that layout.  The
    measured drift is ~1e-13, i.e. decomposition noise, not a different
    computation -- 1e-11 is the documented ceiling.
    """
    pytest.importorskip("sklearn")
    from panelary.reduce.xs import CrossSectionalPCA

    frame = _xs_fixture()
    reducer = CrossSectionalPCA(
        n_components=3,
        standardize=standardize,
        sign_fix=sign_fix,
        entity="entity",
        time="time",
    )
    result = reducer.fit(frame).transform(frame).collect().sort(["entity", "time"])
    reference = golden["xs_pca"][f"std{int(standardize)}_sf{int(sign_fix)}"]
    for column in ("cspc_1", "cspc_2", "cspc_3"):
        got = [
            None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
            for v in result.get_column(column).to_list()
        ]
        ref = reference[column]
        assert len(got) == len(ref)
        for a, b in zip(got, ref, strict=True):
            if a is None or b is None:
                assert a is None and b is None, column
            else:
                assert abs(a - b) <= 1e-11, (column, a, b)
