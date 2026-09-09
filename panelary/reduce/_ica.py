"""ICA factors -- independent, maximally non-Gaussian components.

The third corner of the factor family's decision square: where HFA looks for
non-Gaussian factors through a single higher-order **cumulant** matrix, ICA
searches directly for a rotation whose components are as statistically
*independent* (equivalently, as non-Gaussian) as possible. Same output contract
as every other extractor here -- frozen loadings ``(n_features, n_factors)``,
scores ``F = X_standardised @ loadings`` -- so it drops into the same
:class:`~panelary.core.protocol.PanelTransformer` wrapper and the same
pipelines.

``scikit-learn``'s ``FastICA`` does the heavy lifting and is **imported lazily**
through :func:`panelary._internal._deps.require`, so ``import panelary``
stays numpy+polars only. Install it with ``pip install 'panelary[ml]'``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from panelary._internal._deps import require
from panelary.reduce._common import fix_signs, prepare_matrix
from panelary.reduce._n_factors import n_factors as _resolve_n_factors

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["ica_factors"]


def ica_factors(
    X: NDArray[Any],
    r: int | None = None,
    *,
    standardize: bool = True,
    random_state: int = 0,
    max_iter: int = 500,
    tol: float = 1e-4,
    max_r: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Extract independent components as factors.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n_rows, n_features)`` observation matrix; NaNs are mean-filled.
    r : int, optional
        Number of components. ``None`` (default) resolves it with the Bai--Ng
        ``IC_p2`` criterion -- ICA cannot choose its own dimension, so the
        covariance-based selector supplies one.
    standardize : bool, default True
        Scale columns to unit variance. Centring always happens, which is what
        lets the fitted unmixing matrix be applied as a plain linear map.
    random_state : int, default 0
        Seed for FastICA's initialisation. Fixed by default so repeated fits are
        bit-identical.
    max_iter : int, default 500
        FastICA iteration budget.
    tol : float, default 1e-4
        FastICA convergence tolerance.
    max_r : int, optional
        Cap on the automatically selected component count.

    Returns
    -------
    (factors, loadings, extra) : tuple
        ``loadings`` is the sign-fixed ``(n_features, n_factors)`` unmixing
        matrix (``FastICA.components_.T``); ``extra`` carries ``"mixing"``,
        ``"n_factors"`` and ``"n_iter"``.

    Raises
    ------
    ImportError
        If ``scikit-learn`` is not installed.
    """
    Z = prepare_matrix(X, standardize=standardize)
    k = _resolve_n_factors(Z, method="bai_ng", max_r=max_r) if r is None else int(r)
    if k < 1:
        raise ValueError(f"`r` must be a positive integer or None, got {r!r}.")
    k = min(k, Z.shape[1])

    decomposition = require(
        "sklearn.decomposition", feature="ICA factor extraction (ICAFactors)"
    )
    ica = decomposition.FastICA(
        n_components=k,
        random_state=random_state,
        max_iter=max_iter,
        tol=tol,
        whiten="unit-variance",
    )
    ica.fit(Z)
    loadings, _signs = fix_signs(np.asarray(ica.components_, dtype=np.float64).T)
    factors = Z @ loadings
    extra: dict[str, Any] = {
        "mixing": np.asarray(getattr(ica, "mixing_", np.empty((0, 0)))),
        "n_factors": k,
        "n_iter": int(getattr(ica, "n_iter_", 0) or 0),
    }
    return factors, loadings, extra
