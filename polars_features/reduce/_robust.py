"""Robust (Huber) PCA factors -- a leading subspace that outliers cannot drag.

The fourth corner of the factor family. Ordinary PCA minimises a *squared*
reconstruction error, so a handful of contaminated rows -- a bad print, a
corporate action, a fat-tailed shock -- can rotate the leading subspace by an
arbitrary amount. Huber PCA replaces the square with Huber's loss, which is
quadratic near zero and **linear** in the tail, bounding any single row's
influence.

Algorithm (iteratively reweighted PCA, pure NumPy)
--------------------------------------------------
Given a centred/standardised ``X``:

1. initialise ``U`` with the ordinary top-``r`` eigenvectors of ``X'X / n``;
2. compute each row's orthogonal residual to the current subspace,
   ``d_i = || x_i - U U' x_i ||``;
3. estimate the residual scale robustly, ``s = median(d) / 0.6745``;
4. apply Huber weights ``w_i = min(1, c*s / d_i)`` -- rows inside the tuning
   radius keep full weight, rows outside are downweighted in proportion to how
   far out they are;
5. re-estimate ``U`` from the weighted covariance ``sum_i w_i x_i x_i' /
   sum_i w_i`` and repeat until the projector ``U U'`` stops moving.

The default ``c = 1.345`` is Huber's classic constant (95% asymptotic efficiency
under Gaussianity), so on clean data this converges to something very close to
plain PCA -- you pay almost nothing for the insurance.

No new dependency: the whole routine is ``numpy.linalg.eigh`` in a loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from polars_features.reduce._common import (
    covariance,
    fix_signs,
    prepare_matrix,
    top_eigenvectors,
)
from polars_features.reduce._n_factors import n_factors as _resolve_n_factors

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["robust_pca_factors"]

#: Consistency constant making ``median(|z|) / 0.6745`` a Gaussian scale estimate.
_MAD_CONSISTENCY = 0.6744897501960817


def robust_pca_factors(
    X: NDArray[Any],
    r: int | None = None,
    *,
    standardize: bool = True,
    c: float = 1.345,
    max_iter: int = 50,
    tol: float = 1e-6,
    max_r: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Extract factors from a Huber-reweighted covariance matrix.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n_rows, n_features)`` observation matrix; NaNs are mean-filled.
    r : int, optional
        Number of factors. ``None`` (default) uses the Bai--Ng ``IC_p2``
        criterion.
    standardize : bool, default True
        Scale columns to unit variance. Centring always happens.
    c : float, default 1.345
        Huber tuning constant, in units of the robust residual scale. Smaller is
        more aggressive; ``c -> inf`` recovers ordinary PCA.
    max_iter : int, default 50
        Maximum IRLS iterations.
    tol : float, default 1e-6
        Convergence tolerance on the Frobenius change of the projector ``U U'``.
    max_r : int, optional
        Cap on the automatically selected factor count.

    Returns
    -------
    (factors, loadings, extra) : tuple
        ``loadings`` is the sign-fixed ``(n_features, n_factors)`` matrix;
        ``extra`` carries ``"weights"`` (the final per-row Huber weights),
        ``"eigenvalues"``, ``"explained_variance_ratio"`` (of the weighted
        covariance), ``"n_iter"`` and ``"n_factors"``.
    """
    if c <= 0:
        raise ValueError(f"`c` must be positive, got {c!r}.")
    Z = prepare_matrix(X, standardize=standardize)
    n, p = Z.shape
    k = _resolve_n_factors(Z, method="bai_ng", max_r=max_r) if r is None else int(r)
    if k < 1:
        raise ValueError(f"`r` must be a positive integer or None, got {r!r}.")
    k = min(k, p)

    S = covariance(Z)
    eigvals, U = top_eigenvectors(S, k)
    weights = np.ones(n, dtype=np.float64)
    projector = U @ U.T
    n_iter = 0

    for iteration in range(1, int(max_iter) + 1):
        n_iter = iteration
        resid = Z - (Z @ U) @ U.T
        d = np.linalg.norm(resid, axis=1)
        scale = float(np.median(d)) / _MAD_CONSISTENCY
        if not np.isfinite(scale) or scale <= 0.0:
            break
        cutoff = c * scale
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(d > cutoff, cutoff / np.maximum(d, 1e-300), 1.0)
        weights = np.clip(np.nan_to_num(weights, nan=1.0), 0.0, 1.0)
        wsum = float(weights.sum())
        if wsum <= 0.0:  # pragma: no cover - defensive
            break
        S = (Z * weights[:, None]).T @ Z / wsum
        eigvals, U_new = top_eigenvectors(S, k)
        projector_new = U_new @ U_new.T
        shift = float(np.linalg.norm(projector_new - projector))
        U, projector = U_new, projector_new
        if shift < tol:
            break

    all_eigvals = np.linalg.eigvalsh(0.5 * (S + S.T))
    total = float(np.clip(all_eigvals, 0.0, None).sum())
    evr = (
        np.clip(eigvals, 0.0, None) / total
        if total > 0
        else np.zeros(k, dtype=np.float64)
    )

    loadings, _signs = fix_signs(U)
    factors = Z @ loadings
    extra: dict[str, Any] = {
        "weights": weights,
        "eigenvalues": eigvals,
        "explained_variance_ratio": np.asarray(evr, dtype=np.float64),
        "n_iter": n_iter,
        "n_factors": k,
    }
    return factors, loadings, extra
