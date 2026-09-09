"""Golden-baseline parity tests for the Wave 2/3 vectorisation round.

Every numeric change in that round (catch22 loop removal, the scipy-free
Chebyshev neighbour counter behind sample/approximate entropy, the batched
k-Shape NCC, and the single-conversion ``CrossSectionalPCA`` transform) was made
under one rule: **capture the output first, then assert the optimised code
reproduces it**.

``tests/data/perf_parity_golden.json`` is that capture -- produced by the
pre-change implementations on a fixed bank of synthetic series -- and these
tests are the ratchet that keeps the optimised code pinned to it.  Most of the
comparisons are bit-exact; the two that are not carry a documented tolerance and
a reason.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import polars as pl
import pytest

import panelary.catch22 as c22
from panelary import feature_extractors as fe
from panelary._numpy_stats import chebyshev_neighbour_counts
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
# catch22 -- all 24 features, all series, bit-exact
# --------------------------------------------------------------------------- #
#: Empty, and it should stay that way.  `PD_PeriodicityWang_th0_01` used to live
#: here: it spline-detrended via `scipy.interpolate.LSQUnivariateSpline` and
#: silently fell back to a *zero* spline when SciPy was absent, changing the
#: answer (140.0 -> 0.0 on `walk_512`).  The spline is now fitted in pure NumPy
#: by `catch22._lsq_spline_fit`, so every feature is SciPy-independent and the
#: golden baseline -- captured with SciPy present -- applies unconditionally.
#: See `test_lsq_spline_matches_scipy`.
_SCIPY_DEPENDENT_FEATURES: frozenset[str] = frozenset()


@pytest.mark.parametrize("name", sorted(series_bank()))
def test_catch22_matches_golden_bitwise(name, golden):
    """The vectorised catch22 helpers reproduce the loop versions exactly.

    Bit-exact is achievable here because every rewrite preserved the arithmetic:
    the loops became `flatnonzero`/`bincount`/sliding-window reductions over the
    same values in the same order, not a different formula.
    """
    arr = series_bank()[name]
    got = c22.catch22_all(arr, catch24=True)
    reference = golden["catch22"][name]
    assert set(got) == set(reference)
    skip = _SCIPY_DEPENDENT_FEATURES
    mismatched = {
        key: (reference[key], got[key])
        for key in reference
        if key not in skip and not _same(got[key], reference[key])
    }
    assert not mismatched, f"catch22 drift on {name}: {mismatched}"


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
# k-Shape -- batched NCC, bit-exact
# --------------------------------------------------------------------------- #
def _kshape_fixture():
    rng = np.random.default_rng(7)
    return rng.standard_normal((24, 96)).cumsum(axis=1)


def test_kshape_fit_matches_golden_bitwise(golden):
    """Batching the forward transforms does not perturb fit at all.

    Labels are discrete, so anything but bit-exact here would risk a different
    clustering; the batched form reuses the same ``fft``/``ifft`` calls on the
    same inputs, so it is.
    """
    core = KShapeCore(3, seed=11, max_iter=20).fit(_kshape_fixture())
    reference = golden["kshape"]
    assert core.labels_.tolist() == reference["labels"]
    assert np.array_equal(
        np.asarray(core.centroids_).ravel(), np.asarray(reference["centroids"])
    )


def test_kshape_distance_matrix_matches_golden_bitwise(golden):
    core = KShapeCore(3, seed=11, max_iter=20).fit(_kshape_fixture())
    dist = _distance_matrix(KShapeCore._as3d(_kshape_fixture()), core.centroids_)
    assert np.array_equal(dist.ravel(), np.asarray(golden["kshape"]["distances"]))


def test_batched_distance_matrix_matches_pairwise_ncc():
    """Independent check: the batched matrix equals the pairwise definition."""
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
