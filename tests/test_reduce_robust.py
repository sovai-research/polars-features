"""Tests for Huber (robust) PCA factors.

(a) under row contamination -- a minority of rows replaced by heavy-tailed junk
    -- the Huber-reweighted subspace stays much closer to the truth than the
    ordinary least-squares one;
(b) on clean Gaussian data the two agree closely (``c = 1.345`` costs almost
    nothing), so robustness is not paid for with bias;
(c) the row weights are a usable outlier diagnostic: contaminated rows get
    systematically lower weight;
(d) the shared API (shapes, names, `keep=`, determinism, leak-safety flags) and
    argument validation;
(e) no new dependency -- the whole routine runs on numpy alone.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from panelary.reduce import RobustPCAFactors, robust_pca_factors
from panelary.reduce._estimators import pca_factors


def _subspace_score(A: np.ndarray, B: np.ndarray) -> float:
    Qa, _ = np.linalg.qr(np.asarray(A, dtype=float))
    Qb, _ = np.linalg.qr(np.asarray(B, dtype=float))
    cos = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.mean(np.clip(cos, 0.0, 1.0) ** 2))


def _contaminated(seed: int, *, n=1200, p=15, k=2, frac=0.08, blow=25.0):
    """A clean rank-`k` factor model with `frac` of the rows blown up."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, k))
    L = rng.standard_normal((p, k))
    X = F @ L.T + 0.4 * rng.standard_normal((n, p))
    bad = rng.choice(n, size=int(frac * n), replace=False)
    X[bad] += blow * rng.standard_normal((bad.size, p))
    return X, L, bad


def _panel_frame(X: np.ndarray, n_entities: int = 10) -> pl.DataFrame:
    n, p = X.shape
    per = n // n_entities
    data = {
        "id": np.repeat([f"e{i}" for i in range(n_entities)], per),
        "t": np.tile(np.arange(per), n_entities),
    }
    for j in range(p):
        data[f"x{j}"] = X[: n_entities * per, j]
    return pl.DataFrame(data)


# --------------------------------------------------------------------------- #
# (a) beats PCA under contamination
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(5))
def test_robust_pca_beats_pca_under_row_contamination(seed):
    X, L, _bad = _contaminated(seed)
    _f_rob, u_rob, _ = robust_pca_factors(X, 2)
    _f_pca, u_pca, _ = pca_factors(X, 2)

    rob = _subspace_score(u_rob, L)
    pca = _subspace_score(u_pca, L)
    assert rob > pca, f"seed={seed}: robust {rob:.3f} did not beat PCA {pca:.3f}"
    assert rob > 0.85


# --------------------------------------------------------------------------- #
# (b) near-free on clean data
# --------------------------------------------------------------------------- #
def test_agrees_with_pca_on_clean_gaussian_data():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((2000, 3))
    L = rng.standard_normal((3, 12))
    X = F @ L + rng.standard_normal((2000, 12))
    _f_rob, u_rob, _ = robust_pca_factors(X, 3)
    _f_pca, u_pca, _ = pca_factors(X, 3)
    assert _subspace_score(u_rob, u_pca) > 0.99


def test_large_c_converges_towards_plain_pca():
    X, _L, _bad = _contaminated(0)
    _f_rob, u_rob, _ = robust_pca_factors(X, 2, c=1e6)
    _f_pca, u_pca, _ = pca_factors(X, 2)
    assert _subspace_score(u_rob, u_pca) > 0.999


# --------------------------------------------------------------------------- #
# (c) the weights are an outlier diagnostic
# --------------------------------------------------------------------------- #
def test_contaminated_rows_are_downweighted():
    X, _L, bad = _contaminated(0)
    _f, _u, extra = robust_pca_factors(X, 2)
    w = extra["weights"]
    mask = np.zeros(X.shape[0], dtype=bool)
    mask[bad] = True
    assert w[mask].mean() < 0.5 * w[~mask].mean()
    assert w.min() >= 0.0
    assert w.max() <= 1.0


def test_estimator_exposes_row_weights():
    X, _L, _bad = _contaminated(0)
    df = _panel_frame(X)
    ext = RobustPCAFactors(2, entity="id", time="t").fit(df)
    assert ext.row_weights_ is not None
    assert ext.row_weights_.size == df.height
    assert ext.row_weights_.max() <= 1.0


# --------------------------------------------------------------------------- #
# (d) shared API + validation
# --------------------------------------------------------------------------- #
def test_estimator_shapes_names_and_keep():
    X, _L, _bad = _contaminated(0)
    df = _panel_frame(X)
    ext = RobustPCAFactors(2, entity="id", time="t").fit(df)
    assert ext.loadings_.shape == (15, 2)
    assert ext.get_feature_names_out() == ["factor_1", "factor_2"]
    assert ext.explained_variance_ratio_ is not None
    assert ext.explained_variance_ratio_.shape == (2,)
    assert np.all(ext.explained_variance_ratio_ >= 0)

    only = (
        RobustPCAFactors(2, keep="factors", entity="id", time="t")
        .fit_transform(df)
        .collect()
    )
    assert only.columns == ["id", "t", "factor_1", "factor_2"]


def test_auto_n_factors_resolves_via_bai_ng():
    rng = np.random.default_rng(3)
    F = rng.standard_normal((800, 3))
    L = rng.standard_normal((3, 30))
    X = F @ L + 1.5 * rng.standard_normal((800, 30))
    ext = RobustPCAFactors(entity="id", time="t").fit(_panel_frame(X))
    assert ext.n_factors_ == 3


def test_declares_the_leakage_contract():
    assert RobustPCAFactors.panel_safe is True
    assert RobustPCAFactors.leakage_safe is True


def test_c_must_be_positive():
    X, _L, _bad = _contaminated(0, n=200, p=6)
    with pytest.raises(ValueError, match="`c` must be positive"):
        robust_pca_factors(X, 2, c=0.0)


def test_n_factors_must_be_positive():
    with pytest.raises(ValueError, match="n_factors"):
        RobustPCAFactors(0)


def test_iteration_budget_is_respected():
    X, _L, _bad = _contaminated(0)
    _f, _u, extra = robust_pca_factors(X, 2, max_iter=3, tol=0.0)
    assert extra["n_iter"] <= 3


# --------------------------------------------------------------------------- #
# (e) numpy only
# --------------------------------------------------------------------------- #
def test_robust_path_needs_no_optional_dependency():
    code = (
        "import sys; "
        "import numpy as np; "
        "from panelary.reduce import robust_pca_factors; "
        "rng = np.random.default_rng(0); "
        "robust_pca_factors(rng.standard_normal((300, 10)), 2); "
        "assert 'sklearn' not in sys.modules, 'sklearn imported'; "
        "assert 'scipy' not in sys.modules, 'scipy imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
