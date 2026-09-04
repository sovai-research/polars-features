"""Parity tests: :mod:`polars_features._numpy_stats` vs SciPy.

These pin the dependency-free numpy replacements to SciPy's reference output to
the tolerances measured in the D3 investigation.  SciPy is a test-only import
here; the point of ``_numpy_stats`` is that the runtime feature modules no
longer need it.
"""

from __future__ import annotations

import numpy as np
import pytest

from polars_features import _numpy_stats as ns

scipy_stats = pytest.importorskip("scipy.stats")
scipy_signal = pytest.importorskip("scipy.signal")
scipy_linalg = pytest.importorskip("scipy.linalg")


RNG = np.random.default_rng(0)


def test_lstsq_matches_scipy():
    max_err = 0.0
    for _ in range(10):
        a = RNG.standard_normal((50, 4))
        y = RNG.standard_normal(50)
        c_sp = scipy_linalg.lstsq(a, y, cond=None)[0]
        c_np = ns.lstsq(a, y, cond=None)[0]
        max_err = max(max_err, float(np.max(np.abs(c_sp - c_np))))
    assert max_err <= 1e-12


def test_norm_ppf_matches_scipy():
    ps = np.array([0.001, 0.01, 0.025, 0.1, 0.5, 0.9, 0.975, 0.99, 0.999])
    err = np.max(np.abs(ns.norm_ppf(ps) - scipy_stats.norm.ppf(ps)))
    assert err <= 1e-13
    # scalar input returns a scalar-like value
    assert float(ns.norm_ppf(0.975)) == pytest.approx(
        float(scipy_stats.norm.ppf(0.975)), abs=1e-13
    )


def test_norm_cdf_matches_scipy():
    xs = np.linspace(-6, 6, 201)
    err = np.max(np.abs(ns.norm_cdf(xs) - scipy_stats.norm.cdf(xs)))
    assert err <= 1e-15


def test_skew_matches_scipy():
    max_err = 0.0
    for _ in range(20):
        d = RNG.standard_normal(RNG.integers(20, 300)) * RNG.uniform(0.5, 5)
        max_err = max(max_err, abs(ns.skew(d) - scipy_stats.skew(d)))
    assert max_err <= 1e-12


def test_kurtosis_matches_scipy():
    max_err = 0.0
    for _ in range(20):
        d = RNG.standard_normal(RNG.integers(20, 300)) * RNG.uniform(0.5, 5)
        max_err = max(max_err, abs(ns.kurtosis(d) - scipy_stats.kurtosis(d)))
    assert max_err <= 1e-12


def test_normaltest_stat_matches_scipy():
    max_err = 0.0
    for _ in range(20):
        d = RNG.standard_normal(RNG.integers(30, 500))
        max_err = max(
            max_err, abs(ns.normaltest_stat(d) - float(scipy_stats.normaltest(d)[0]))
        )
    assert max_err <= 1e-12


def test_welch_hann_default_matches_scipy():
    """The parameterization used in feature_extractors (default hann welch)."""
    max_f, max_p = 0.0, 0.0
    for _ in range(10):
        n = int(RNG.integers(300, 1200))
        t = np.arange(n)
        x = np.sin(2 * np.pi * 0.1 * t) + 0.5 * RNG.standard_normal(n)
        nps = min(256, n)
        f_sp, p_sp = scipy_signal.welch(x, nperseg=nps)
        f_np, p_np = ns.welch(x, nperseg=nps)
        max_f = max(max_f, float(np.max(np.abs(f_sp - f_np))))
        max_p = max(max_p, float(np.max(np.abs(p_sp - p_np))))
    assert max_f <= 1e-15
    assert max_p <= 1e-12


def test_welch_boxcar_matches_scipy():
    """The parameterization used in catch22 (_welch_spectrum)."""
    max_p = 0.0
    for _ in range(10):
        n = int(RNG.integers(64, 600))
        x = np.cumsum(RNG.standard_normal(n))
        f_sp, p_sp = scipy_signal.welch(
            x,
            window="boxcar",
            nperseg=n,
            noverlap=0,
            detrend=False,
            return_onesided=True,
            scaling="density",
        )
        f_np, p_np = ns.welch(
            x,
            window="boxcar",
            nperseg=n,
            noverlap=0,
            detrend=False,
            return_onesided=True,
            scaling="density",
        )
        assert np.max(np.abs(f_sp - f_np)) <= 1e-15
        max_p = max(max_p, float(np.max(np.abs(p_sp - p_np))))
    assert max_p <= 1e-10
