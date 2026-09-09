"""Shared, dependency-free helpers for the leak-safe factor-extraction family.

Part of Panelary's **interactions theme**: higher-order structure in the *data*
(the cumulant factors of :mod:`panelary.reduce._hfa`, ``order=3|4``) and
higher-order structure in the *model* (Shapley interactions, ``max_order=k``).
This module is the plumbing both halves of the factor side share.

Everything here is pure NumPy/Polars -- no ``scipy``, no ``sklearn``, no lazy
imports -- so the whole "matrix in, loadings out" path works on a bare
``numpy + polars`` install.

The helpers exist to make the **fit-on-train** contract mechanical:

* :func:`feature_matrix` / :func:`resolve_features` turn a panel into a dense
  ``(n_rows, n_features)`` float matrix;
* :func:`column_means` + :func:`impute_column_mean` split "learn the fill value"
  from "apply the fill value", so a transform can never impute test rows with
  test-fold means;
* :func:`standardize_fit` / :func:`standardize_apply` do the same split for
  centering/scaling;
* :func:`fix_signs` pins the otherwise-arbitrary sign of every eigenvector /
  loading column, so factors computed at ``fit`` time and at ``transform`` time
  can never flip relative to one another.

Notes
-----
:mod:`panelary.select._unsupervised` carries near-identical private
``_feature_matrix`` / ``_impute_column_mean`` helpers. They are deliberately
*not* refactored away here: ``select`` is a separate, already-shipped public
surface and this module is additive. Consolidating the two is a follow-up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from panelary.core.panel_frame import PanelFrame

__all__ = [
    "resolve_features",
    "feature_matrix",
    "column_means",
    "impute_column_mean",
    "standardize_fit",
    "standardize_apply",
    "sign_of_max_abs",
    "fix_signs",
    "prepare_matrix",
    "top_eigenvectors",
    "covariance",
]


# --------------------------------------------------------------------------- #
# Panel -> matrix
# --------------------------------------------------------------------------- #
def resolve_features(
    panel: PanelFrame,
    features: Sequence[str] | None = None,
    *,
    exclude: Sequence[str] | None = None,
    caller: str = "factor extraction",
) -> list[str]:
    """Resolve the numeric feature columns to factor-analyse.

    Parameters
    ----------
    panel : PanelFrame
        The panel to read columns from.
    features : sequence of str, optional
        Explicit column list. ``None`` (default) uses every non-key column.
    exclude : sequence of str, optional
        Columns to drop from the resolved list.
    caller : str
        Name used in error messages.

    Returns
    -------
    list of str
        Numeric feature column names, in schema order.

    Raises
    ------
    ValueError
        If a requested column is missing or no numeric column survives.
    """
    dropped = set(exclude or ())
    schema = panel.schema
    if features is not None:
        feats = list(features)
        missing = [c for c in feats if c not in panel]
        if missing:
            raise ValueError(
                f"{caller}: feature column(s) {missing} not found in panel. "
                f"Available columns: {panel.columns}."
            )
    else:
        feats = list(panel.feature_cols)
    feats = [c for c in feats if c not in dropped]
    numeric = [c for c in feats if schema[c].is_numeric()]
    if not numeric:
        raise ValueError(
            f"{caller}: no numeric feature columns available (considered "
            f"{feats}). Encode/cast features to numeric first, or pass an "
            "explicit `features=` list."
        )
    return numeric


def feature_matrix(panel: PanelFrame, feats: Sequence[str]) -> NDArray[np.float64]:
    """Collect ``feats`` into a dense ``(n_rows, n_features)`` float matrix."""
    frame = panel.lazy().select(list(feats)).collect()
    return frame.to_numpy().astype(np.float64, copy=False)


# --------------------------------------------------------------------------- #
# Imputation (learn / apply split)
# --------------------------------------------------------------------------- #
def column_means(mat: NDArray[Any]) -> NDArray[np.float64]:
    """Per-column mean ignoring NaNs; all-NaN columns map to ``0.0``.

    This is the *learned* half of the imputation: call it on the training
    matrix and hand the result to :func:`impute_column_mean` at transform time.
    """
    arr = np.asarray(mat, dtype=np.float64)
    if arr.size == 0:  # pragma: no cover - defensive
        return np.zeros(arr.shape[1], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(arr, axis=0)
    return np.nan_to_num(np.asarray(means, dtype=np.float64), nan=0.0)


def impute_column_mean(
    mat: NDArray[Any], means: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """Fill NaNs with ``means`` (or this matrix's own column means).

    Parameters
    ----------
    mat : numpy.ndarray
        The ``(n_rows, n_features)`` matrix to fill.
    means : numpy.ndarray, optional
        Pre-learned fill values from :func:`column_means`. Passing them is what
        keeps a ``transform`` leak-safe -- ``None`` (compute from ``mat``) is
        only correct at ``fit`` time.

    Returns
    -------
    numpy.ndarray
        A float copy with no NaNs (the input is returned unchanged when it
        already has none and needs no cast).
    """
    arr = np.asarray(mat, dtype=np.float64)
    if not np.isnan(arr).any():
        return arr
    fill = column_means(arr) if means is None else np.asarray(means, dtype=np.float64)
    out = arr.copy()
    nan_idx = np.where(np.isnan(out))
    out[nan_idx] = np.take(fill, nan_idx[1])
    return out


# --------------------------------------------------------------------------- #
# Standardization (learn / apply split)
# --------------------------------------------------------------------------- #
def standardize_fit(
    mat: NDArray[Any], *, scale: bool = True
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Learn centering/scaling statistics from ``mat`` (training rows only).

    Parameters
    ----------
    mat : numpy.ndarray
        Training matrix, ``(n_rows, n_features)``.
    scale : bool, default True
        Whether to scale to unit standard deviation. **Centering always
        happens**: every method in this subpackage (PCA, HFA, ICA, robust PCA)
        is defined on centred data, so ``scale=False`` means "centre but do not
        rescale", not "leave the data alone".

    Returns
    -------
    (mean, std) : tuple of numpy.ndarray
        Per-column statistics. Zero (or non-finite) standard deviations are
        replaced by ``1.0`` so constant columns pass through as exact zeros
        instead of NaNs.
    """
    arr = np.asarray(mat, dtype=np.float64)
    mean = arr.mean(axis=0)
    if scale:
        std = arr.std(axis=0)
        std = np.where(np.isfinite(std) & (std > 0.0), std, 1.0)
    else:
        std = np.ones(arr.shape[1], dtype=np.float64)
    return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)


def standardize_apply(
    mat: NDArray[Any], mean: NDArray[np.float64], std: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply frozen ``mean`` / ``std`` from :func:`standardize_fit`."""
    arr = np.asarray(mat, dtype=np.float64)
    return (arr - mean) / std


def prepare_matrix(X: NDArray[Any], *, standardize: bool = True) -> NDArray[np.float64]:
    """Centre (and optionally scale) ``X`` in one shot, for the free functions.

    The estimator classes split this into
    :func:`standardize_fit`/:func:`standardize_apply` so the statistics can be
    frozen; the standalone ``*_factors`` functions, which fit and score the same
    matrix, use this convenience instead.
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"expected a 2-D (n_rows, n_features) matrix, got shape {arr.shape}."
        )
    if arr.shape[0] < 2 or arr.shape[1] < 1:
        raise ValueError(
            "need at least 2 rows and 1 feature to extract factors, got shape "
            f"{arr.shape}."
        )
    arr = impute_column_mean(arr)
    mean, std = standardize_fit(arr, scale=standardize)
    return standardize_apply(arr, mean, std)


# --------------------------------------------------------------------------- #
# Sign convention -- mandatory for leak-safety
# --------------------------------------------------------------------------- #
def sign_of_max_abs(vec: NDArray[Any]) -> float:
    """Sign (``+1.0``/``-1.0``) of the largest-magnitude entry of ``vec``.

    Ties break on the first occurrence (:func:`numpy.argmax`) and an exact-zero
    winner maps to ``+1.0``, so the result is fully deterministic.
    """
    arr = np.asarray(vec, dtype=np.float64)
    if arr.size == 0:  # pragma: no cover - defensive
        return 1.0
    idx = int(np.argmax(np.abs(arr)))
    s = float(np.sign(arr[idx]))
    return s if s != 0.0 else 1.0


def fix_signs(
    loadings: NDArray[Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pin the sign of every loading column (the §6.4 convention).

    Eigenvectors are only defined up to sign, and ``numpy.linalg.eigh`` makes no
    promise about which one it returns. Without a fixed convention, refitting --
    or simply comparing factors learned at ``fit`` time against factors emitted
    at ``transform`` time by a *different* fitted instance -- can silently flip a
    factor's sign, which flips the sign of every downstream coefficient.

    The convention: **flip each column so that its largest-magnitude entry is
    positive**.

    Parameters
    ----------
    loadings : numpy.ndarray
        ``(n_features, n_factors)`` loading matrix.

    Returns
    -------
    (loadings, signs) : tuple of numpy.ndarray
        The sign-fixed copy and the ``(n_factors,)`` vector of applied signs.
    """
    arr = np.array(loadings, dtype=np.float64, copy=True)
    if arr.ndim != 2:  # pragma: no cover - defensive
        raise ValueError(f"loadings must be 2-D, got shape {arr.shape}.")
    signs = np.array(
        [sign_of_max_abs(arr[:, j]) for j in range(arr.shape[1])], dtype=np.float64
    )
    arr *= signs
    return arr, signs


# --------------------------------------------------------------------------- #
# Small linear-algebra utilities (numpy only -- no scipy)
# --------------------------------------------------------------------------- #
def covariance(X: NDArray[Any]) -> NDArray[np.float64]:
    """``X.T @ X / n`` for an already-centred ``X`` (the ML covariance)."""
    arr = np.asarray(X, dtype=np.float64)
    return (arr.T @ arr) / float(arr.shape[0])


def top_eigenvectors(
    mat: NDArray[Any], r: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Top-``r`` eigenvectors of a **symmetric** matrix, by descending eigenvalue.

    Uses :func:`numpy.linalg.eigh` (never ``eig``) and symmetrises the input
    first, so floating-point asymmetry from a blocked accumulation cannot leak
    complex eigenvalues into the result.

    Returns
    -------
    (eigenvalues, eigenvectors) : tuple of numpy.ndarray
        ``eigenvalues`` is ``(r,)`` sorted descending; ``eigenvectors`` is
        ``(n, r)`` with matching column order. Signs are **not** fixed here --
        call :func:`fix_signs` on the result.
    """
    arr = np.asarray(mat, dtype=np.float64)
    sym = 0.5 * (arr + arr.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    order = np.argsort(eigvals)[::-1]
    k = max(1, min(int(r), arr.shape[0]))
    keep = order[:k]
    return eigvals[keep], eigvecs[:, keep]
