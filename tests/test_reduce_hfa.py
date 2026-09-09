"""Tests for HFA -- higher-order multi-cumulant factor analysis.

The headline claim (and the module's whole reason to exist) is (a): when a
**non-Gaussian** factor is masked by a larger Gaussian one, PCA locks onto the
Gaussian mask while HFA -- whose cumulant matrix is blind to anything Gaussian
-- recovers the factor of interest. Everything else here defends the
implementation around that claim.

(a) recovery: HFA(order=3) beats PCA on a skewed weak/masked factor, and
    HFA(order=4) beats PCA on a symmetric heavy-tailed one (where order 3 has
    nothing to find, the third cumulant being zero by symmetry);
(b) the cumulant matrix is symmetric PSD at order 3, eigenvalues come back
    sorted descending, and ``eigh`` (not ``eig``) is what is being used;
(c) the blocked accumulation path reproduces the dense matrix exactly;
(d) the memory guard fires with an actionable message above the hard cap;
(e) shape / API / ``get_feature_names_out`` / ``keep=`` for ``HFAFactors``;
(f) the mandatory sign convention (loadings, hence factors, cannot flip);
(g) ``order`` validation and the ``order=4`` Gaussian correction actually
    changing the answer.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.reduce import HFAFactors, hfa_cumulant_matrix, hfa_factors
from panelary.reduce._common import prepare_matrix
from panelary.reduce._estimators import pca_factors
from panelary.reduce._hfa import HARD_ROW_CAP


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _subspace_score(A: np.ndarray, B: np.ndarray) -> float:
    """Mean squared canonical correlation between span(A) and span(B).

    1.0 == identical subspaces, 0.0 == orthogonal. This is the standard
    principal-angle measure: the singular values of ``Qa' Qb`` for orthonormal
    bases are the cosines of the principal angles.
    """
    Qa, _ = np.linalg.qr(np.asarray(A, dtype=float))
    Qb, _ = np.linalg.qr(np.asarray(B, dtype=float))
    cos = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.mean(np.clip(cos, 0.0, 1.0) ** 2))


def _masked_factor_panel(seed: int, *, kind: str = "skew", n: int = 2000, p: int = 20):
    """A weak non-Gaussian factor hiding behind a stronger Gaussian one.

    ``X = f @ lam' + g @ mu' + e`` with ``f`` non-Gaussian and weakly loaded,
    ``g`` Gaussian and strongly loaded, ``e`` Gaussian idiosyncratic noise.
    PCA's leading eigenvector tracks ``mu``; the truth of interest is ``lam``.
    """
    rng = np.random.default_rng(seed)
    if kind == "skew":  # exponential: skewness 2, excess kurtosis 6
        f = rng.standard_exponential(n) - 1.0
        weak, strong = 0.6, 1.5
    else:  # symmetric heavy tail: zero skewness, positive excess kurtosis
        f = rng.standard_t(4, n)
        f = f / f.std()
        weak, strong = 0.9, 1.5
    g = rng.standard_normal(n)
    lam = rng.standard_normal((p, 1)) * weak
    mu = rng.standard_normal((p, 1)) * strong
    X = f[:, None] @ lam.T + g[:, None] @ mu.T + rng.standard_normal((n, p))
    return X, lam


def _panel_frame(X: np.ndarray, n_entities: int = 10) -> pl.DataFrame:
    n, p = X.shape
    per = n // n_entities
    ids = np.repeat([f"e{i}" for i in range(n_entities)], per)
    times = np.tile(np.arange(per), n_entities)
    data = {"id": ids, "t": times}
    for j in range(p):
        data[f"x{j}"] = X[: n_entities * per, j]
    return pl.DataFrame(data)


# --------------------------------------------------------------------------- #
# (a) THE headline claim: HFA recovers what PCA cannot
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(5))
def test_hfa_order3_beats_pca_on_a_skewed_masked_factor(seed):
    X, lam = _masked_factor_panel(seed, kind="skew")
    _f_hfa, u_hfa, _ = hfa_factors(X, 1, order=3)
    _f_pca, u_pca, _ = pca_factors(X, 1)

    hfa_score = _subspace_score(u_hfa, lam)
    pca_score = _subspace_score(u_pca, lam)
    assert hfa_score > pca_score, (
        f"seed={seed}: HFA {hfa_score:.3f} did not beat PCA {pca_score:.3f}"
    )
    assert hfa_score > 0.70, f"seed={seed}: HFA failed to recover the subspace"
    assert pca_score < 0.60, f"seed={seed}: PCA was not actually fooled"


@pytest.mark.parametrize("seed", range(5))
def test_hfa_order4_beats_pca_on_a_symmetric_heavy_tailed_factor(seed):
    # Third cumulant is zero by symmetry here, so this is order 4's job.
    X, lam = _masked_factor_panel(seed, kind="heavy", n=3000)
    _f_hfa, u_hfa, _ = hfa_factors(X, 1, order=4)
    _f_pca, u_pca, _ = pca_factors(X, 1)

    hfa_score = _subspace_score(u_hfa, lam)
    pca_score = _subspace_score(u_pca, lam)
    assert hfa_score > pca_score, (
        f"seed={seed}: HFA4 {hfa_score:.3f} did not beat PCA {pca_score:.3f}"
    )
    assert hfa_score > 0.30


def test_hfa_matches_the_true_loadings_up_to_sign_on_a_clean_signal():
    rng = np.random.default_rng(0)
    n, p = 4000, 8
    f = rng.standard_exponential(n) - 1.0
    lam = rng.standard_normal((p, 1))
    X = f[:, None] @ lam.T + 0.1 * rng.standard_normal((n, p))
    # standardize=False keeps the loadings in the original column scale, so the
    # recovered direction is directly comparable to `lam`.
    _f, u, _extra = hfa_factors(X, 1, standardize=False)
    assert _subspace_score(u, lam) > 0.98


# --------------------------------------------------------------------------- #
# (b) the cumulant matrix is a symmetric PSD object, decomposed with eigh
# --------------------------------------------------------------------------- #
def test_order3_cumulant_matrix_is_symmetric_and_psd():
    X, _ = _masked_factor_panel(0)
    M = hfa_cumulant_matrix(prepare_matrix(X), order=3)
    assert np.allclose(M, M.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(M)
    assert eigvals.min() > -1e-8, "order-3 cumulant matrix should be PSD"


def test_eigenvalues_come_back_sorted_descending():
    X, _ = _masked_factor_panel(0)
    _f, _u, extra = hfa_factors(X, 4)
    eigvals = extra["eigenvalues"]
    assert np.all(np.diff(eigvals) <= 1e-12)
    spectrum = extra["spectrum"]
    assert np.all(np.diff(spectrum) <= 1e-12)
    assert np.allclose(eigvals, spectrum[:4])


def test_loadings_are_orthonormal_eigenvectors():
    X, _ = _masked_factor_panel(0)
    _f, u, _extra = hfa_factors(X, 3)
    assert np.allclose(u.T @ u, np.eye(3), atol=1e-8)
    assert np.isrealobj(u), "eigh must be used, not eig (which can go complex)"


# --------------------------------------------------------------------------- #
# (c) blocked accumulation == dense accumulation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize("block_rows", [1, 7, 97, 512, 10_000])
def test_blocked_path_reproduces_the_dense_matrix(order, block_rows):
    X, _ = _masked_factor_panel(0, n=500, p=12)
    Z = prepare_matrix(X)
    dense = hfa_cumulant_matrix(Z, order=order)
    blocked = hfa_cumulant_matrix(Z, order=order, block_rows=block_rows)
    assert np.allclose(dense, blocked, rtol=1e-10, atol=1e-12)


def test_blocked_path_gives_the_same_factors():
    X, _ = _masked_factor_panel(0, n=800, p=10)
    f_dense, u_dense, _ = hfa_factors(X, 2)
    f_block, u_block, _ = hfa_factors(X, 2, block_rows=64)
    assert np.allclose(u_dense, u_block, atol=1e-8)
    assert np.allclose(f_dense, f_block, atol=1e-8)


# --------------------------------------------------------------------------- #
# (d) the memory guard
# --------------------------------------------------------------------------- #
def test_hard_row_cap_raises_an_actionable_error_without_block_rows():
    from panelary.reduce._hfa import _resolve_block_rows

    with pytest.raises(ValueError) as exc:
        _resolve_block_rows(HARD_ROW_CAP + 1, None)
    message = str(exc.value)
    assert "block_rows" in message
    assert "subsample" in message
    assert "cross-sectionally" in message


def test_explicit_block_rows_bypasses_the_hard_cap():
    from panelary.reduce._hfa import _resolve_block_rows

    assert _resolve_block_rows(HARD_ROW_CAP + 1, 1024) == 1024


def test_auto_blocking_between_the_limits():
    from panelary.reduce._hfa import DENSE_ROW_LIMIT, _resolve_block_rows

    assert _resolve_block_rows(DENSE_ROW_LIMIT, None) == DENSE_ROW_LIMIT
    auto = _resolve_block_rows(20_000, None)
    assert 0 < auto < 20_000


def test_block_rows_must_be_positive():
    X, _ = _masked_factor_panel(0, n=200, p=5)
    with pytest.raises(ValueError, match="block_rows"):
        hfa_factors(X, 1, block_rows=0)


# --------------------------------------------------------------------------- #
# (e) the estimator: shapes, names, keep=
# --------------------------------------------------------------------------- #
def test_estimator_shapes_and_feature_names():
    X, _ = _masked_factor_panel(0, n=600, p=8)
    df = _panel_frame(X)
    ext = HFAFactors(3, entity="id", time="t").fit(df)
    assert ext.n_factors_ == 3
    assert ext.loadings_.shape == (8, 3)
    assert ext.get_feature_names_out() == ["factor_1", "factor_2", "factor_3"]
    out = ext.transform(df).collect()
    assert out.height == df.height
    assert set(df.columns).issubset(out.columns)
    assert out.select("factor_1").to_series().std() > 0


def test_keep_factors_returns_keys_plus_factors_only():
    X, _ = _masked_factor_panel(0, n=600, p=8)
    df = _panel_frame(X)
    out = (
        HFAFactors(2, keep="factors", entity="id", time="t").fit_transform(df).collect()
    )
    assert out.columns == ["id", "t", "factor_1", "factor_2"]


def test_keep_is_validated():
    with pytest.raises(ValueError, match="keep"):
        HFAFactors(2, keep="everything")


def test_auto_n_factors_uses_the_cumulant_spectrum():
    X, _ = _masked_factor_panel(0, n=1500, p=12)
    ext = HFAFactors(entity="id", time="t").fit(_panel_frame(X, n_entities=10))
    assert 1 <= ext.n_factors_ <= 12
    assert ext.cumulant_spectrum_ is not None
    assert ext.cumulant_spectrum_.size == 12


def test_declares_the_leakage_contract():
    assert HFAFactors.panel_safe is True
    assert HFAFactors.leakage_safe is True


def test_transform_before_fit_raises():
    X, _ = _masked_factor_panel(0, n=200, p=5)
    with pytest.raises(RuntimeError, match="not fitted"):
        HFAFactors(1, entity="id", time="t").transform(_panel_frame(X, 5))


# --------------------------------------------------------------------------- #
# (f) the mandatory sign convention
# --------------------------------------------------------------------------- #
def test_loading_columns_obey_the_sign_convention():
    X, _ = _masked_factor_panel(0, n=800, p=10)
    _f, u, _extra = hfa_factors(X, 3)
    for j in range(u.shape[1]):
        col = u[:, j]
        assert col[np.argmax(np.abs(col))] > 0


def test_signs_are_stable_across_row_shuffles():
    X, _ = _masked_factor_panel(0, n=800, p=10)
    rng = np.random.default_rng(7)
    perm = rng.permutation(X.shape[0])
    _f1, u1, _ = hfa_factors(X, 3)
    _f2, u2, _ = hfa_factors(X[perm], 3)
    assert np.allclose(u1, u2, atol=1e-6)


# --------------------------------------------------------------------------- #
# (g) order validation + the order-4 Gaussian correction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", [0, 1, 2, 5, 6])
def test_unsupported_order_raises(order):
    X, _ = _masked_factor_panel(0, n=200, p=5)
    with pytest.raises(ValueError, match="order"):
        hfa_factors(X, 1, order=order)


def test_order4_gaussian_correction_changes_the_answer():
    X, lam = _masked_factor_panel(0, kind="heavy", n=3000)
    _f_on, u_on, _ = hfa_factors(X, 1, order=4, gaussian_correction=True)
    _f_off, u_off, _ = hfa_factors(X, 1, order=4, gaussian_correction=False)
    assert not np.allclose(u_on, u_off, atol=1e-3)
    # Uncorrected, the order-4 matrix is a polynomial in the covariance and so
    # re-finds the Gaussian mask; corrected, it finds the heavy-tailed factor.
    assert _subspace_score(u_on, lam) > _subspace_score(u_off, lam)


def test_order3_ignores_the_gaussian_correction_flag():
    X, _ = _masked_factor_panel(0, n=600, p=8)
    Z = prepare_matrix(X)
    a = hfa_cumulant_matrix(Z, order=3, gaussian_correction=True)
    b = hfa_cumulant_matrix(Z, order=3, gaussian_correction=False)
    assert np.array_equal(a, b)
