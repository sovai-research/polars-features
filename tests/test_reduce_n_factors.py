"""Tests for the shared factor-count selectors (Bai--Ng + eigenvalue ratio).

Covers:

(a) both Bai--Ng criteria recover the true ``k`` on simulated rank-``k`` data
    across several ``(T, N, k)`` regimes;
(b) the eigenvalue-ratio rule recovers ``k`` from a covariance spectrum and from
    a raw spectrum handed in directly (which is how HFA uses it);
(c) the search cap (``max_r``, ``DEFAULT_MAX_FACTORS``) is honoured;
(d) argument validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from panelary.reduce import bai_ng, eigenvalue_ratio, n_factors
from panelary.reduce._n_factors import DEFAULT_MAX_FACTORS


def _rank_k_matrix(n=500, p=30, k=3, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, k))
    L = rng.standard_normal((k, p))
    return F @ L + noise * rng.standard_normal((n, p))


# --------------------------------------------------------------------------- #
# (a) Bai-Ng recovers the true k
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("criterion", ["IC_p1", "IC_p2"])
@pytest.mark.parametrize(
    ("n", "p", "k"),
    [(500, 30, 3), (600, 40, 3), (400, 25, 4), (800, 50, 5)],
)
def test_bai_ng_recovers_true_rank(criterion, n, p, k):
    hits = [
        bai_ng(_rank_k_matrix(n, p, k, 1.5, seed), criterion=criterion) == k
        for seed in range(5)
    ]
    assert sum(hits) >= 4, f"{criterion} picked the wrong k in {5 - sum(hits)}/5 sims"


def test_n_factors_dispatches_to_bai_ng_by_default():
    X = _rank_k_matrix(k=3)
    assert n_factors(X) == bai_ng(X) == 3


def test_bai_ng_rejects_unknown_criterion():
    with pytest.raises(ValueError, match="criterion"):
        bai_ng(_rank_k_matrix(), criterion="AIC")


def test_n_factors_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        n_factors(_rank_k_matrix(), method="magic")


# --------------------------------------------------------------------------- #
# (b) eigenvalue ratio
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_eigenratio_recovers_true_rank_from_covariance(k):
    hits = [
        n_factors(_rank_k_matrix(600, 40, k, 1.5, seed), method="eigenratio") == k
        for seed in range(5)
    ]
    assert sum(hits) >= 4


def test_eigenratio_on_a_raw_spectrum():
    # The HFA path: a spectrum, not a data matrix. Big gap after the 2nd value.
    spectrum = np.array([10.0, 8.0, 0.2, 0.15, 0.1, 0.05])
    assert eigenvalue_ratio(spectrum) == 2


def test_eigenratio_is_order_insensitive_and_clips_negatives():
    spectrum = np.array([0.05, 10.0, -0.3, 8.0, 0.2, 0.1])
    assert eigenvalue_ratio(spectrum) == 2


def test_eigenratio_degenerate_spectrum_returns_one():
    assert eigenvalue_ratio(np.array([3.0])) == 1
    assert eigenvalue_ratio(np.zeros(5)) == 1


# --------------------------------------------------------------------------- #
# (c) caps
# --------------------------------------------------------------------------- #
def test_max_r_caps_the_search():
    X = _rank_k_matrix(600, 40, 5, 1.5, 0)
    assert bai_ng(X, max_r=2) <= 2
    assert n_factors(X, method="eigenratio", max_r=2) <= 2


def test_default_cap_is_respected():
    X = _rank_k_matrix(600, 40, 3, 1.5, 0)
    assert bai_ng(X) <= DEFAULT_MAX_FACTORS
    assert n_factors(X, method="eigenratio") <= DEFAULT_MAX_FACTORS


def test_max_r_must_be_positive():
    with pytest.raises(ValueError, match="max_r"):
        bai_ng(_rank_k_matrix(), max_r=0)


# --------------------------------------------------------------------------- #
# (d) selectors never import scipy/sklearn
# --------------------------------------------------------------------------- #
def test_selectors_are_pure_numpy():
    import subprocess
    import sys

    code = (
        "import sys; "
        "import numpy as np; "
        "from panelary.reduce import n_factors; "
        "rng = np.random.default_rng(0); "
        "X = rng.standard_normal((200, 20)); "
        "n_factors(X); n_factors(X, method='eigenratio'); "
        "assert 'scipy' not in sys.modules, 'scipy imported'; "
        "assert 'sklearn' not in sys.modules, 'sklearn imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
