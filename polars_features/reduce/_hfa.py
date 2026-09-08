"""HFA -- factor extraction from higher-order **multi-cumulants**.

Part of PanelKit's **interactions theme**. The theme has two halves and one
vocabulary: ``order=k`` here selects the order of the *data* interaction (which
cumulant is eigendecomposed), exactly as ``max_order=k`` in the attribution
layer selects the order of the *model* interaction (Shapley interactions).
HFA is the order >= 2, input-side half.

Why it exists
-------------
PCA eigendecomposes the **covariance** matrix, so it can only see a factor whose
variance contribution stands clear of the noise bulk. Two very common panel
situations defeat that:

* a **weak** factor -- its eigenvalue is buried inside the Marchenko--Pastur
  bulk of the idiosyncratic noise;
* a **masked** factor -- a larger, uninteresting Gaussian source (a level, a
  market factor, correlated measurement noise) owns the leading eigenvector.

Gaussian variables have **zero cumulants above order two**. So if you replace
the covariance matrix with a higher-order cumulant matrix, everything Gaussian
-- bulk and mask alike -- drops out of the population object, and the leading
eigenvectors span the non-Gaussian factors regardless of how small their
variance share is. That is the whole idea.

The estimator (order 3)
-----------------------
For a centred/standardised ``X`` of shape ``(n, p)``::

    G    = X @ X.T                  # (n, n) Gram
    M3M  = X.T @ ((G * G) @ X)      # (p, p) third-order multi-cumulant matrix
    U    = top-r eigenvectors(M3M)  # (p, r) loadings, via eigh
    F    = X @ U                    # (n, r) factor scores

``M3M`` is a Hadamard square of a Gram matrix sandwiched by ``X``, hence
symmetric positive semi-definite (Schur product theorem), so
:func:`numpy.linalg.eigh` is the correct solver -- never ``eig``.

Writing ``M3M = sum_{i,j} (x_i . x_j)^2 x_i x_j^T`` makes the general order
obvious: order ``m`` uses the ``(m-1)``-th Hadamard power of the Gram matrix.
The population expectation of the ``i != j`` terms is
``E[(x.y)^{m-1} x y^T]`` for independent ``x, y``. For ``m = 3`` that is an odd
moment and vanishes exactly under Gaussianity -- no correction needed. For
``m = 4`` it does not: by Isserlis' theorem a Gaussian contributes
``3 tr(S^2) S^2 + 6 S^4`` with ``S`` the covariance, and
:func:`hfa_cumulant_matrix` subtracts that estimated Gaussian part so ``order=4``
measures *excess kurtosis* structure rather than re-discovering the covariance.

Use ``order=3`` for **skewed** factors (the default, and the strongest signal in
financial panels); use ``order=4`` for **symmetric heavy-tailed** factors, where
the third cumulant is zero by symmetry and order 3 has nothing to find.

Memory
------
``G`` is ``(n, n)``. A panel is ``entities x dates``, so ``n`` gets large fast
and a dense Gram is not always affordable. Because the accumulation
``sum_i x_i (row_i of G^{o(m-1)}) X`` decomposes over **row blocks** of ``G``,
:func:`hfa_cumulant_matrix` computes the identical matrix in blocks with peak
memory ``O(block_rows * n)`` instead of ``O(n^2)``. Blocking switches on
automatically above :data:`DENSE_ROW_LIMIT`, and above :data:`HARD_ROW_CAP` the
function refuses to run without an explicit ``block_rows=`` (the work is
inherently ``O(n^2 p)``, so the cap is a runtime guard as much as a memory one).

Clean-room notice
-----------------
Implemented from the published equations above. The authors' R package ``hofa``
ships **no LICENSE file** and is therefore all-rights-reserved; none of its
source was consulted or translated.

References
----------
Huang, Lu et al., "Estimation of Factors Using Higher-Order Multi-Cumulants in
Weak Factor Models", *Journal of Business & Economic Statistics* (forthcoming).
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
from polars_features.reduce._n_factors import eigenvalue_ratio

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "hfa_factors",
    "hfa_cumulant_matrix",
    "DENSE_ROW_LIMIT",
    "HARD_ROW_CAP",
]

#: Rows above which blocked accumulation switches on automatically.
DENSE_ROW_LIMIT = 5_000
#: Rows above which an explicit ``block_rows=`` is required.
HARD_ROW_CAP = 50_000
#: Target element count for an auto-chosen Gram block (~80 MB in float64).
_BLOCK_ELEMENTS = 10_000_000

_SUPPORTED_ORDERS = (3, 4)


def _resolve_block_rows(n_rows: int, block_rows: int | None) -> int:
    """Pick the Gram row-block size, guarding the ``O(n^2)`` dense path."""
    if block_rows is not None:
        if int(block_rows) < 1:
            raise ValueError(
                f"`block_rows` must be a positive integer or None, got {block_rows!r}."
            )
        return min(int(block_rows), n_rows)
    if n_rows <= DENSE_ROW_LIMIT:
        return n_rows
    if n_rows > HARD_ROW_CAP:
        raise ValueError(
            f"HFA: refusing to build a {n_rows} x {n_rows} Gram matrix "
            f"({n_rows * n_rows * 8 / 1e9:.1f} GB dense) for {n_rows} rows, which "
            f"exceeds the hard cap of {HARD_ROW_CAP}. The cumulant contraction is "
            "O(n^2 * p) in time regardless of blocking, so this is a runtime guard "
            "too. Choose one of:\n"
            "  * pass `block_rows=` explicitly (e.g. block_rows=2048) to accept "
            "the cost with bounded memory;\n"
            "  * subsample or aggregate rows before fitting (factor loadings are "
            "usually stable on a sample);\n"
            "  * fit cross-sectionally per date instead of pooling the panel."
        )
    return max(256, min(n_rows, _BLOCK_ELEMENTS // n_rows))


def hfa_cumulant_matrix(
    X: NDArray[Any],
    *,
    order: int = 3,
    block_rows: int | None = None,
    gaussian_correction: bool = True,
) -> NDArray[np.float64]:
    """The ``(p, p)`` higher-order multi-cumulant matrix of a centred ``X``.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n_rows, n_features)`` matrix. **Assumed already centred** (and
        usually standardised) -- use :func:`hfa_factors` for the full pipeline.
    order : {3, 4}, default 3
        Cumulant order. ``order=m`` contracts the ``(m-1)``-th Hadamard power of
        the Gram matrix.
    block_rows : int, optional
        Row-block size for the Gram accumulation. ``None`` (default) uses a
        dense Gram below :data:`DENSE_ROW_LIMIT` rows and an automatically sized
        block above it. The blocked result is mathematically identical to the
        dense one (it is the same sum, re-associated).
    gaussian_correction : bool, default True
        For ``order=4`` only: subtract the Isserlis Gaussian part
        ``3 tr(S^2) S^2 + 6 S^4``. Without it the leading eigenvector of the
        order-4 matrix is simply the leading eigenvector of the covariance, i.e.
        PCA with extra steps. Ignored for ``order=3``, whose Gaussian part is
        exactly zero.

    Returns
    -------
    numpy.ndarray
        The symmetric ``(n_features, n_features)`` cumulant matrix, normalised
        by ``n_rows ** 2``.

    Raises
    ------
    ValueError
        For an unsupported ``order``, a non-positive ``block_rows``, or a row
        count above :data:`HARD_ROW_CAP` with no explicit ``block_rows``.
    """
    if int(order) not in _SUPPORTED_ORDERS:
        raise ValueError(
            f"HFA `order` must be one of {_SUPPORTED_ORDERS}, got {order!r}. "
            "order=3 targets skewed factors, order=4 symmetric heavy-tailed "
            "ones; higher orders are not implemented."
        )
    Z = np.asarray(X, dtype=np.float64)
    n, p = Z.shape
    block = _resolve_block_rows(n, block_rows)
    hadamard_power = int(order) - 1

    acc = np.zeros((p, p), dtype=np.float64)
    for start in range(0, n, block):
        Xb = Z[start : start + block]
        Gb = Xb @ Z.T  # (b, n)
        Hb = Gb.copy()
        for _ in range(hadamard_power - 1):
            Hb *= Gb
        acc += Xb.T @ (Hb @ Z)
    acc /= float(n) ** 2
    acc = 0.5 * (acc + acc.T)

    if int(order) == 4 and gaussian_correction:
        S = covariance(Z)
        S2 = S @ S
        acc = acc - (3.0 * float(np.trace(S2)) * S2 + 6.0 * (S2 @ S2))
        acc = 0.5 * (acc + acc.T)
    return acc


def hfa_factors(
    X: NDArray[Any],
    r: int | None = None,
    *,
    order: int = 3,
    standardize: bool = True,
    block_rows: int | None = None,
    gaussian_correction: bool = True,
    max_r: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Extract higher-order multi-cumulant (HFA) factors from a matrix.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n_rows, n_features)`` observation matrix. NaNs are filled with the
        column mean before the decomposition.
    r : int, optional
        Number of factors. ``None`` (default) resolves it with the
        eigenvalue-ratio rule applied to the **cumulant** spectrum -- a cumulant
        matrix has no explained-variance interpretation, so an information
        criterion like Bai--Ng does not apply to it.
    order : {3, 4}, default 3
        Cumulant order. See the module docstring: 3 for skewed factors, 4 for
        symmetric heavy-tailed ones. The kwarg is deliberately named ``order`` to
        match ``max_order`` in the attribution half of the interactions theme.
    standardize : bool, default True
        Scale columns to unit variance. Centring always happens.
    block_rows : int, optional
        Gram row-block size; see :func:`hfa_cumulant_matrix`.
    gaussian_correction : bool, default True
        ``order=4`` Gaussian-part subtraction; see :func:`hfa_cumulant_matrix`.
    max_r : int, optional
        Cap on the automatically selected factor count.

    Returns
    -------
    (factors, loadings, extra) : tuple
        ``factors`` is ``(n_rows, r)``, ``loadings`` is ``(n_features, r)`` and
        sign-fixed (largest-magnitude entry of each column forced positive), and
        ``extra`` carries ``"eigenvalues"`` (the top-``r`` cumulant eigenvalues),
        ``"spectrum"`` (the full cumulant spectrum, descending), ``"n_factors"``
        and ``"order"``.

    Examples
    --------
    >>> import numpy as np
    >>> from polars_features.reduce import hfa_factors
    >>> rng = np.random.default_rng(0)
    >>> f = rng.standard_exponential(2000) - 1.0        # skewed weak factor
    >>> lam = rng.standard_normal((10, 1)) * 0.6
    >>> X = f[:, None] @ lam.T + rng.standard_normal((2000, 10))
    >>> F, U, extra = hfa_factors(X, 1)
    >>> F.shape, U.shape
    ((2000, 1), (10, 1))
    """
    Z = prepare_matrix(X, standardize=standardize)
    M = hfa_cumulant_matrix(
        Z,
        order=order,
        block_rows=block_rows,
        gaussian_correction=gaussian_correction,
    )
    spectrum = np.sort(np.linalg.eigvalsh(0.5 * (M + M.T)))[::-1]
    k = eigenvalue_ratio(spectrum, max_r=max_r) if r is None else int(r)
    if k < 1:
        raise ValueError(f"`r` must be a positive integer or None, got {r!r}.")
    k = min(k, Z.shape[1])

    eigvals, eigvecs = top_eigenvectors(M, k)
    loadings, _signs = fix_signs(eigvecs)
    factors = Z @ loadings
    extra: dict[str, Any] = {
        "eigenvalues": eigvals,
        "spectrum": spectrum,
        "n_factors": k,
        "order": int(order),
    }
    return factors, loadings, extra
