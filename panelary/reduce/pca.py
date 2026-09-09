"""Concrete leak-safe panel reducers (PCA family + friends).

Every class here is a thin subclass of
:class:`panelary.reduce._base._PanelReducer`: it only names the sklearn
estimator to build. The base supplies the leak-safe fit-on-train contract
(scaler + rotation + auto-``n_components`` learned on training rows only),
deterministic component sign-fixing, ``pc_1 .. pc_k`` naming and provenance
attributes.

The set mirrors SovAI's ``reducer_methods`` (``pca``, ``truncated_svd``,
``factor_analysis``, ``gaussian_random_projection``) plus ``kernel_pca`` and
``nmf``, and adds an optional, dependency-guarded :class:`PanelUMAP`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from panelary.reduce._base import _PanelReducer

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl
    from numpy.typing import NDArray

    from panelary.core.panel_frame import PanelFrame

__all__ = [
    "PanelPCA",
    "PanelSVD",
    "PanelFactorAnalysis",
    "PanelRandomProjection",
    "PanelKernelPCA",
    "PanelNMF",
    "PanelUMAP",
    "reduce_features",
]


class PanelPCA(_PanelReducer):
    """Leak-safe panel PCA (:class:`sklearn.decomposition.PCA`).

    Learns the standardiser, the orthogonal rotation and -- when
    ``n_components`` is left as ``None`` -- the component count (from
    ``explained_variance``) on the **training** rows only, then projects any
    panel onto ``pc_1 .. pc_k``. This is the leak-safe replacement for SovAI's
    ``dimensionality_reduction(method="pca")``, which fits all three on the full
    sample.

    See :class:`~panelary.reduce._base._PanelReducer` for the full
    parameter list.
    """

    _supports_explained_variance = True

    def _make_reducer(self, n_components: int) -> Any:
        from sklearn.decomposition import PCA

        return PCA(n_components=n_components, random_state=self.random_state)


class PanelSVD(_PanelReducer):
    """Leak-safe panel truncated SVD (:class:`sklearn.decomposition.TruncatedSVD`).

    LSA-style SVD (no centering; sparse-friendly). ``TruncatedSVD`` requires
    ``n_components < n_features``, so the admissible count is capped one below the
    feature count.
    """

    _supports_explained_variance = True

    def _cap_components(self, n_samples: int, n_features: int) -> int:
        return max(1, min(n_samples, n_features - 1))

    def _make_reducer(self, n_components: int) -> Any:
        from sklearn.decomposition import TruncatedSVD

        return TruncatedSVD(n_components=n_components, random_state=self.random_state)


class PanelFactorAnalysis(_PanelReducer):
    """Leak-safe panel factor analysis (:class:`sklearn.decomposition.FactorAnalysis`)."""

    def _make_reducer(self, n_components: int) -> Any:
        from sklearn.decomposition import FactorAnalysis

        return FactorAnalysis(n_components=n_components, random_state=self.random_state)


class PanelRandomProjection(_PanelReducer):
    """Leak-safe panel Gaussian random projection.

    Wraps :class:`sklearn.random_projection.GaussianRandomProjection`. The
    projection matrix is drawn from ``random_state`` and is **data-independent**,
    so only the upstream standardiser could leak -- and that is fit on train rows
    only. For a fixed ``random_state`` the projection is identical regardless of
    the data.
    """

    def _make_reducer(self, n_components: int) -> Any:
        from sklearn.random_projection import GaussianRandomProjection

        return GaussianRandomProjection(
            n_components=n_components, random_state=self.random_state
        )


class PanelKernelPCA(_PanelReducer):
    """Leak-safe panel kernel PCA (:class:`sklearn.decomposition.KernelPCA`).

    Nonlinear PCA in an (RBF by default) kernel space. The landmark rows that
    define the kernel basis are fit on the training rows only. ``KernelPCA``
    exposes no ``components_``, so sign-fixing falls back to forcing each
    component's largest-magnitude *score* positive.

    Parameters
    ----------
    kernel : str, default "rbf"
        Kernel passed to :class:`sklearn.decomposition.KernelPCA`.
    **kwargs
        See :class:`~panelary.reduce._base._PanelReducer`.
    """

    def __init__(self, *, kernel: str = "rbf", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.kernel = kernel

    def _make_reducer(self, n_components: int) -> Any:
        from sklearn.decomposition import KernelPCA

        return KernelPCA(
            n_components=n_components,
            kernel=self.kernel,
            random_state=self.random_state,
        )


class PanelNMF(_PanelReducer):
    """Leak-safe panel non-negative matrix factorisation.

    Wraps :class:`sklearn.decomposition.NMF`. NMF requires a non-negative input
    matrix; :meth:`_validate_matrix` rejects negative values with a clear error.
    Because standardisation would introduce negatives, ``standardize`` defaults
    to ``False`` here.
    """

    def __init__(self, *, standardize: bool = False, **kwargs: Any) -> None:
        super().__init__(standardize=standardize, **kwargs)

    def _validate_matrix(self, X: NDArray[Any]) -> None:
        import numpy as np

        if np.nanmin(X) < 0.0:
            raise ValueError(
                f"{type(self).__name__}: NMF requires a non-negative input "
                "matrix, but negative values were found. Shift/clip the features "
                "to be non-negative first, or use PanelPCA/PanelSVD instead."
            )

    def _make_reducer(self, n_components: int) -> Any:
        from sklearn.decomposition import NMF

        return NMF(
            n_components=n_components,
            random_state=self.random_state,
            init="nndsvda",
            max_iter=500,
        )


def _require_umap() -> Any:
    """Lazily import the optional ``umap-learn`` dependency with an actionable error."""
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(
            "PanelUMAP requires the optional `umap-learn` dependency, which is "
            "not installed. Install it with `pip install panelary[umap]` "
            "(or `pip install umap-learn`)."
        ) from exc
    return umap


class PanelUMAP(_PanelReducer):
    """Leak-safe panel UMAP manifold embedding (optional dependency).

    Wraps :class:`umap.UMAP`, imported lazily so importing this module never
    requires ``umap-learn``. The embedding is fit on the training rows only and
    applied to new rows via ``UMAP.transform``. UMAP exposes no ``components_``,
    so sign-fixing falls back to the score-based rule.

    Parameters
    ----------
    n_components : int, default 2
        Embedding dimensionality (UMAP has no explained-variance target).
    n_neighbors : int, default 15
        UMAP local-neighbourhood size.
    min_dist : float, default 0.1
        UMAP minimum embedding distance.
    **kwargs
        See :class:`~panelary.reduce._base._PanelReducer`.

    Raises
    ------
    ImportError
        At :meth:`fit` time if ``umap-learn`` is not installed.
    """

    def __init__(
        self,
        *,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_components=n_components, **kwargs)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def _make_reducer(self, n_components: int) -> Any:
        umap = _require_umap()

        return umap.UMAP(
            n_components=n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
        )


_METHODS: dict[str, type[_PanelReducer]] = {
    "pca": PanelPCA,
    "truncated_svd": PanelSVD,
    "svd": PanelSVD,
    "factor_analysis": PanelFactorAnalysis,
    "gaussian_random_projection": PanelRandomProjection,
    "random_projection": PanelRandomProjection,
    "kernel_pca": PanelKernelPCA,
    "nmf": PanelNMF,
    "umap": PanelUMAP,
}


def reduce_features(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    method: str = "pca",
    n_components: int | None = None,
    columns: Sequence[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
    **kwargs: Any,
) -> PanelFrame:
    """Fit-and-transform convenience over the panel reducers.

    Builds the reducer named by ``method`` and returns
    ``reducer.fit_transform(data)``. Because this both fits and transforms on the
    same rows it is only appropriate on training data (or exploratory single
    frames); inside cross-validation, construct the reducer and use the
    ``fit`` / ``transform`` split explicitly.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
        The panel to reduce.
    method : str, default "pca"
        One of ``pca``, ``truncated_svd``/``svd``, ``factor_analysis``,
        ``gaussian_random_projection``/``random_projection``, ``kernel_pca``,
        ``nmf``, ``umap``.
    n_components, columns, entity, time, **kwargs
        Forwarded to the reducer constructor (see
        :class:`~panelary.reduce._base._PanelReducer`).

    Returns
    -------
    PanelFrame
        Keys plus the component columns (or the panel with components appended
        when ``keep_features=True``).
    """
    key = method.lower()
    if key not in _METHODS:
        raise ValueError(
            f"reduce_features: unknown method {method!r}. "
            f"Choose one of {sorted(_METHODS)}."
        )
    reducer = _METHODS[key](
        n_components=n_components,
        columns=columns,
        entity=entity,
        time=time,
        **kwargs,
    )
    return reducer.fit_transform(data, entity=entity, time=time)
