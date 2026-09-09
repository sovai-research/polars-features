"""Leak-safe :class:`PanelTransformer` wrappers for the factor family.

One base, four methods, one decision table:

======================================  ===========================
your problem                            the extractor
======================================  ===========================
"just give me the baseline"             :class:`PCAFactors`
weak / masked **non-Gaussian** factors  :class:`HFAFactors` (``order=3|4``)
maximally **independent** components    :class:`ICAFactors`
heavy tails / contaminated rows         :class:`RobustPCAFactors`
======================================  ===========================

All four share :class:`~panelary.reduce._estimators._FactorExtractor`, which owns the entire leak-safety
story so no individual method can get it wrong:

* ``_fit`` resolves the feature columns, learns the NaN fill values, learns the
  centring/scaling statistics, calls the method's ``_extract`` to get loadings,
  and **fixes the loading signs**;
* ``_transform`` re-uses every one of those frozen quantities and does nothing
  but ``F = ((X - mean) / std) @ loadings``.

Because ``_transform`` is a pure linear map with frozen parameters, a row's
factors depend only on that row -- never on which other rows happen to be in the
same test fold -- and the sign convention guarantees factor ``j`` means the same
thing at fit time and at transform time.

Note that a factor extractor **emits new columns** (``factor_1 .. factor_r``),
which is what separates this family from :mod:`panelary.select`, whose
selectors project onto a *subset of existing* columns.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary._internal._deps import require
from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer
from panelary.reduce._common import (
    column_means,
    feature_matrix,
    fix_signs,
    impute_column_mean,
    prepare_matrix,
    resolve_features,
    standardize_apply,
    standardize_fit,
    top_eigenvectors,
)
from panelary.reduce._hfa import hfa_cumulant_matrix
from panelary.reduce._ica import ica_factors
from panelary.reduce._n_factors import eigenvalue_ratio
from panelary.reduce._n_factors import n_factors as _resolve_n_factors
from panelary.reduce._robust import robust_pca_factors

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = [
    "pca_factors",
    "PCAFactors",
    "HFAFactors",
    "ICAFactors",
    "RobustPCAFactors",
]

_KEEP_CHOICES = ("all", "factors")


# --------------------------------------------------------------------------- #
# PCA functional core (the baseline every other method is judged against)
# --------------------------------------------------------------------------- #
def pca_factors(
    X: NDArray[Any],
    r: int | None = None,
    *,
    standardize: bool = True,
    svd_solver: str = "auto",
    random_state: int = 0,
    max_r: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Extract principal-component factors from a matrix.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n_rows, n_features)`` observation matrix; NaNs are mean-filled.
    r : int, optional
        Number of factors. ``None`` (default) uses the Bai--Ng ``IC_p2``
        criterion (:func:`panelary.reduce.n_factors`).
    standardize : bool, default True
        Scale columns to unit variance. Centring always happens.
    svd_solver : {"auto", "full", "randomized"}, default "auto"
        ``"auto"``/``"full"`` use :func:`numpy.linalg.svd` and need nothing
        beyond NumPy. ``"randomized"`` lazily imports
        ``sklearn.utils.extmath.randomized_svd`` for wide matrices.
    random_state : int, default 0
        Seed for the randomized solver.
    max_r : int, optional
        Cap on the automatically selected factor count.

    Returns
    -------
    (factors, loadings, extra) : tuple
        ``loadings`` is the sign-fixed ``(n_features, n_factors)`` matrix;
        ``extra`` carries ``"eigenvalues"``, ``"explained_variance_ratio"``,
        ``"singular_values"`` and ``"n_factors"``.
    """
    Z = prepare_matrix(X, standardize=standardize)
    n, p = Z.shape
    k = _resolve_n_factors(Z, method="bai_ng", max_r=max_r) if r is None else int(r)
    if k < 1:
        raise ValueError(f"`r` must be a positive integer or None, got {r!r}.")
    k = min(k, p)

    if svd_solver == "randomized":
        extmath = require(
            "sklearn.utils.extmath", feature="PCA with svd_solver='randomized'"
        )
        _u, svals, vt = extmath.randomized_svd(
            Z, n_components=k, random_state=random_state
        )
        components = np.asarray(vt, dtype=np.float64)[:k]
        full_svals = np.linalg.svd(Z, compute_uv=False)
    elif svd_solver in ("auto", "full"):
        _u, svals, vt = np.linalg.svd(Z, full_matrices=False)
        components = np.asarray(vt, dtype=np.float64)[:k]
        full_svals = svals
    else:
        raise ValueError(
            f"unknown `svd_solver={svd_solver!r}`; expected 'auto', 'full' or "
            "'randomized'."
        )

    loadings, _signs = fix_signs(components.T)
    factors = Z @ loadings
    eigvals = (np.asarray(svals, dtype=np.float64)[:k] ** 2) / float(n)
    total = float((np.asarray(full_svals, dtype=np.float64) ** 2).sum() / n)
    evr = eigvals / total if total > 0 else np.zeros(k, dtype=np.float64)
    extra: dict[str, Any] = {
        "eigenvalues": eigvals,
        "explained_variance_ratio": evr,
        "singular_values": np.asarray(svals, dtype=np.float64)[:k],
        "n_factors": k,
    }
    return factors, loadings, extra


# --------------------------------------------------------------------------- #
# Shared estimator base
# --------------------------------------------------------------------------- #
class _FactorExtractor(PanelTransformer):
    """Shared, leak-safe machinery for every factor extractor.

    Subclasses implement one hook, :meth:`_extract`, which turns a standardised
    training matrix into ``(loadings, extra)``. Everything else -- column
    resolution, imputation, standardisation, the sign convention, the emitted
    column names, the panel round-trip -- lives here.

    Parameters
    ----------
    n_factors : int, optional
        Number of factors to extract. ``None`` (default) resolves the count on
        the training rows using the method's own default selector (Bai--Ng for
        the covariance-based methods, the eigenvalue ratio of the cumulant
        spectrum for HFA).
    features : sequence of str, optional
        Columns to factor-analyse. ``None`` uses every numeric non-key column.
    standardize : bool, default True
        Scale columns to unit variance using **training** statistics. Centring
        always happens.
    keep : {"all", "factors"}, default "all"
        ``"all"`` appends the factor columns to the input panel (so the
        extractor drops into a :class:`~panelary.core.pipeline.Pipeline`
        in front of a model); ``"factors"`` returns only the ``(entity, time)``
        keys plus the factors.
    factor_prefix : str, default "factor"
        Prefix for the emitted columns (``factor_1 .. factor_r``).
    max_factors : int, optional
        Cap on the automatically selected factor count.
    entity, time : str, optional
        Default panel keys, used when a bare polars frame is passed.

    Attributes
    ----------
    loadings_ : numpy.ndarray
        Sign-fixed ``(n_features, n_factors)`` loadings learned at fit time.
    n_factors_ : int
        The resolved factor count.
    feature_names_in_ : list of str
        Input columns, in the order fed to the extractor.
    factor_names_ : list of str
        Emitted column names.
    mean_, std_ : numpy.ndarray
        Frozen standardisation statistics from the training rows.
    column_mean_ : numpy.ndarray
        Frozen NaN fill values from the training rows.
    explained_variance_ratio_ : numpy.ndarray or None
        Per-factor variance share, where the method defines one.
    eigenvalues_ : numpy.ndarray or None
        Top-``r`` eigenvalues of whichever matrix the method decomposed.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        n_factors: int | None = None,
        *,
        features: Sequence[str] | None = None,
        standardize: bool = True,
        keep: str = "all",
        factor_prefix: str = "factor",
        max_factors: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if n_factors is not None and (not isinstance(n_factors, int) or n_factors < 1):
            raise ValueError(
                f"{type(self).__name__}: `n_factors` must be a positive integer "
                f"or None, got {n_factors!r}."
            )
        if keep not in _KEEP_CHOICES:
            raise ValueError(
                f"{type(self).__name__}: `keep` must be one of {_KEEP_CHOICES}, "
                f"got {keep!r}."
            )
        self.n_factors = n_factors
        self.features = list(features) if features is not None else None
        self.standardize = bool(standardize)
        self.keep = keep
        self.factor_prefix = factor_prefix
        self.max_factors = max_factors
        # learned state
        self.loadings_: NDArray[np.float64] | None = None
        self.n_factors_: int = 0
        self.feature_names_in_: list[str] = []
        self.factor_names_: list[str] = []
        self.mean_: NDArray[np.float64] | None = None
        self.std_: NDArray[np.float64] | None = None
        self.column_mean_: NDArray[np.float64] | None = None
        self.explained_variance_ratio_: NDArray[np.float64] | None = None
        self.eigenvalues_: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------ #
    # Subclass hook
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _extract(
        self, Z: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Return ``(loadings, extra)`` for an already-standardised ``Z``."""
        raise NotImplementedError

    def _resolve_r(self, Z: NDArray[np.float64], *, method: str = "bai_ng") -> int:
        """Resolve the factor count, honouring ``n_factors`` when given."""
        if self.n_factors is not None:
            return min(int(self.n_factors), Z.shape[1])
        return min(
            _resolve_n_factors(
                Z, method=method, max_r=self.max_factors, standardize=False
            ),
            Z.shape[1],
        )

    # ------------------------------------------------------------------ #
    # PanelTransformer hooks
    # ------------------------------------------------------------------ #
    def _fit(self, panel: PanelFrame) -> None:
        feats = resolve_features(
            panel, self.features, caller=f"{type(self).__name__}.fit"
        )
        X = feature_matrix(panel, feats)
        if X.shape[0] < 2:
            raise ValueError(
                f"{type(self).__name__}.fit: need at least 2 rows to extract "
                f"factors, got {X.shape[0]}."
            )
        self.column_mean_ = column_means(X)
        X = impute_column_mean(X, self.column_mean_)
        self.mean_, self.std_ = standardize_fit(X, scale=self.standardize)
        Z = standardize_apply(X, self.mean_, self.std_)

        loadings, extra = self._extract(Z)
        loadings, _signs = fix_signs(np.asarray(loadings, dtype=np.float64))

        self.loadings_ = loadings
        self.n_factors_ = int(loadings.shape[1])
        self.feature_names_in_ = feats
        self.factor_names_ = [
            f"{self.factor_prefix}_{i}" for i in range(1, self.n_factors_ + 1)
        ]
        evr = extra.get("explained_variance_ratio")
        self.explained_variance_ratio_ = (
            None if evr is None else np.asarray(evr, dtype=np.float64)
        )
        eig = extra.get("eigenvalues")
        self.eigenvalues_ = None if eig is None else np.asarray(eig, dtype=np.float64)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        if self.loadings_ is None:  # pragma: no cover - guarded by _check_fitted
            raise RuntimeError(f"{type(self).__name__} is not fitted.")
        missing = [c for c in self.feature_names_in_ if c not in panel]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.transform: feature column(s) {missing} "
                f"not found in panel. Available columns: {panel.columns}."
            )
        full = panel.collect()
        X = full.select(self.feature_names_in_).to_numpy().astype(np.float64)
        # Frozen train-fold statistics only: fill values, then centring/scaling.
        X = impute_column_mean(X, self.column_mean_)
        Z = standardize_apply(X, self.mean_, self.std_)
        F = Z @ self.loadings_

        factor_cols = [
            pl.Series(name=name, values=F[:, i])
            for i, name in enumerate(self.factor_names_)
        ]
        if self.keep == "all":
            out = full.with_columns(factor_cols)
        else:
            out = full.select([panel.entity_col, panel.time_col]).with_columns(
                factor_cols
            )
        return PanelFrame(
            out, entity=panel.entity_col, time=panel.time_col, validate=False
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def get_feature_names_out(self) -> list[str]:
        """Names of the emitted factor columns (``factor_1 .. factor_r``)."""
        self._check_fitted("get_feature_names_out")
        return list(self.factor_names_)


# --------------------------------------------------------------------------- #
# Concrete extractors
# --------------------------------------------------------------------------- #
class PCAFactors(_FactorExtractor):
    """Principal-component factors -- the baseline.

    Eigendecomposes the training **covariance** matrix and freezes the loadings.
    ``n_factors=None`` resolves the count with the Bai--Ng ``IC_p2`` criterion.

    Parameters
    ----------
    svd_solver : {"auto", "full", "randomized"}, default "auto"
        ``"randomized"`` lazily requires ``scikit-learn``; the other two are
        pure NumPy.
    random_state : int, default 0
        Seed for the randomized solver.

    Notes
    -----
    The shared parameters (``n_factors``, ``features``, ``standardize``,
    ``keep``, ``factor_prefix``, ``max_factors``, ``entity``, ``time``) are
    documented on the base class,
    :class:`~panelary.reduce._estimators._FactorExtractor`.

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.reduce import PCAFactors
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["a"] * 6,
    ...         "t": range(6),
    ...         "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ...         "y": [2.0, 4.1, 5.9, 8.2, 9.8, 12.1],
    ...     }
    ... )
    >>> ext = PCAFactors(1, entity="id", time="t").fit(df)
    >>> ext.get_feature_names_out()
    ['factor_1']
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        n_factors: int | None = None,
        *,
        svd_solver: str = "auto",
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_factors, **kwargs)
        self.svd_solver = svd_solver
        self.random_state = random_state

    def _extract(
        self, Z: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        k = self._resolve_r(Z, method="bai_ng")
        _factors, loadings, extra = pca_factors(
            Z,
            k,
            standardize=False,
            svd_solver=self.svd_solver,
            random_state=self.random_state,
        )
        return loadings, extra


class HFAFactors(_FactorExtractor):
    """Higher-order multi-cumulant factors (HFA) -- the differentiator.

    Eigendecomposes a higher-order **cumulant** matrix instead of the covariance
    matrix, so Gaussian structure -- however large its variance share -- drops
    out and weak or masked non-Gaussian factors become visible. See
    :mod:`panelary.reduce._hfa` for the estimator equations, the memory
    story and the clean-room notice.

    Parameters
    ----------
    order : {3, 4}, default 3
        Cumulant order. 3 targets **skewed** factors; 4 targets symmetric
        **heavy-tailed** ones. The name matches ``max_order`` in the attribution
        half of Panelary's interactions theme.
    block_rows : int, optional
        Row-block size for the Gram accumulation, bounding peak memory at
        ``O(block_rows * n_rows)``. ``None`` blocks automatically above 5,000
        rows and refuses to run above 50,000 without an explicit value.
    gaussian_correction : bool, default True
        ``order=4`` only: subtract the Isserlis Gaussian part so the result
        measures excess kurtosis rather than re-deriving the covariance.

    Notes
    -----
    ``n_factors=None`` resolves the count with the **eigenvalue-ratio** rule on
    the cumulant spectrum (not Bai--Ng, which presumes an explained-variance
    interpretation a cumulant matrix does not have).

    The shared parameters (``n_factors``, ``features``, ``standardize``,
    ``keep``, ``factor_prefix``, ``max_factors``, ``entity``, ``time``) are
    documented on the base class,
    :class:`~panelary.reduce._estimators._FactorExtractor`.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        n_factors: int | None = None,
        *,
        order: int = 3,
        block_rows: int | None = None,
        gaussian_correction: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_factors, **kwargs)
        self.order = int(order)
        self.block_rows = block_rows
        self.gaussian_correction = bool(gaussian_correction)
        self.cumulant_spectrum_: NDArray[np.float64] | None = None

    def _extract(
        self, Z: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        M = hfa_cumulant_matrix(
            Z,
            order=self.order,
            block_rows=self.block_rows,
            gaussian_correction=self.gaussian_correction,
        )
        spectrum = np.sort(np.linalg.eigvalsh(0.5 * (M + M.T)))[::-1]
        self.cumulant_spectrum_ = spectrum
        if self.n_factors is not None:
            k = min(int(self.n_factors), Z.shape[1])
        else:
            k = min(eigenvalue_ratio(spectrum, max_r=self.max_factors), Z.shape[1])
        eigvals, eigvecs = top_eigenvectors(M, k)
        return eigvecs, {"eigenvalues": eigvals, "order": self.order}


class ICAFactors(_FactorExtractor):
    """Independent-component factors (FastICA), frozen as a linear map.

    Requires ``scikit-learn`` (lazily imported; ``pip install
    'panelary[ml]'``). ``n_factors=None`` uses Bai--Ng ``IC_p2``.

    Parameters
    ----------
    random_state : int, default 0
        FastICA seed; fixed by default so refits are bit-identical.
    max_iter : int, default 500
        FastICA iteration budget.
    tol : float, default 1e-4
        FastICA convergence tolerance.

    Notes
    -----
    The shared parameters (``n_factors``, ``features``, ``standardize``,
    ``keep``, ``factor_prefix``, ``max_factors``, ``entity``, ``time``) are
    documented on the base class,
    :class:`~panelary.reduce._estimators._FactorExtractor`.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        n_factors: int | None = None,
        *,
        random_state: int = 0,
        max_iter: int = 500,
        tol: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_factors, **kwargs)
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    def _extract(
        self, Z: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        k = self._resolve_r(Z, method="bai_ng")
        _factors, loadings, extra = ica_factors(
            Z,
            k,
            standardize=False,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        return loadings, extra


class RobustPCAFactors(_FactorExtractor):
    """Huber-reweighted PCA factors -- a subspace outliers cannot drag.

    Pure NumPy IRLS; see :mod:`panelary.reduce._robust` for the
    algorithm. ``n_factors=None`` uses Bai--Ng ``IC_p2``.

    Parameters
    ----------
    c : float, default 1.345
        Huber tuning constant in units of the robust residual scale.
    max_iter : int, default 50
        Maximum IRLS iterations.
    tol : float, default 1e-6
        Convergence tolerance on the projector.

    Attributes
    ----------
    row_weights_ : numpy.ndarray
        Final Huber weight of each **training** row (1.0 = untouched). A useful
        outlier diagnostic in its own right.

    Notes
    -----
    The shared parameters (``n_factors``, ``features``, ``standardize``,
    ``keep``, ``factor_prefix``, ``max_factors``, ``entity``, ``time``) are
    documented on the base class,
    :class:`~panelary.reduce._estimators._FactorExtractor`.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        n_factors: int | None = None,
        *,
        c: float = 1.345,
        max_iter: int = 50,
        tol: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_factors, **kwargs)
        self.c = c
        self.max_iter = max_iter
        self.tol = tol
        self.row_weights_: NDArray[np.float64] | None = None

    def _extract(
        self, Z: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        k = self._resolve_r(Z, method="bai_ng")
        _factors, loadings, extra = robust_pca_factors(
            Z,
            k,
            standardize=False,
            c=self.c,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.row_weights_ = np.asarray(extra["weights"], dtype=np.float64)
        return loadings, extra
