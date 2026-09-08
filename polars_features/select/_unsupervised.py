"""Leak-safe **unsupervised** feature selection for panels.

These selectors need **no target** ``y``. They rank / prune feature *columns*
of a long panel using only the redundancy / energy structure of the features
themselves, and they are leak-safe in exactly the same way as the supervised
selectors in :mod:`polars_features.select._methods`: everything is computed
**only on the rows passed in** (the training fold), and the wrapping
transformers freeze ``selected_`` at ``fit`` time so the chosen column set can
never absorb test-fold structure.

Public surface
--------------
Free functions (fit-on-input, deterministic when seeded):

* :func:`pfa` -- Principal Feature Analysis: PCA the standardized feature
  matrix, cluster the per-feature loading vectors with KMeans, keep the feature
  closest to each cluster centroid (one minimally-redundant representative per
  cluster). Returns the selected feature names.
* :func:`variance` -- keep the ``k`` highest-variance features.
* :func:`correlation` -- drop one of every pair of features whose absolute
  correlation exceeds a threshold, keeping the higher-variance member.
* :func:`projection_importance` -- score each feature's energy in a random /
  learned projection space (``"gaussian"``, ``"sparse"``, ``"ica"``, ``"svd"``)
  and return a ``feature`` / ``importance`` / ``importance_percentile`` table.
* :func:`select_top` -- pick features from a ranking table by top-``k`` or by
  cumulative-importance ``variability``.

Transformer classes (``"select"`` pipeline steps, mirroring
:class:`~polars_features.select._methods.MRMRSelector`):

* :class:`PFASelector`, :class:`VarianceSelector`, :class:`CorrelationSelector`.

All are ``panel_safe = True`` and ``leakage_safe = True``.

Notes
-----
No ``scipy`` dependency: ``percentileofscore`` is reimplemented in numpy.
``sklearn`` is used (lazily imported) only for PCA / KMeans / random projections.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from polars_features.core.panel_frame import PanelFrame, as_panel
from polars_features.core.protocol import PanelTransformer

__all__ = [
    "pfa",
    "variance",
    "correlation",
    "projection_importance",
    "select_top",
    "PFASelector",
    "VarianceSelector",
    "CorrelationSelector",
]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _resolve_unsup_features(
    panel: PanelFrame,
    features: Sequence[str] | None,
    exclude: set[str] | None = None,
) -> list[str]:
    """Resolve the numeric candidate feature columns (no target needed)."""
    exclude = set(exclude or ())
    schema = panel.schema
    if features is not None:
        feats = list(features)
        missing = [c for c in feats if c not in panel]
        if missing:
            raise ValueError(
                f"feature column(s) {missing} not found in panel. "
                f"Available columns: {panel.columns}."
            )
    else:
        feats = list(panel.feature_cols)
    feats = [c for c in feats if c not in exclude]
    numeric = [c for c in feats if schema[c].is_numeric()]
    if not numeric:
        raise ValueError(
            "no numeric feature columns available for selection "
            f"(considered {feats}). Encode/scale features to numeric first."
        )
    return numeric


def _feature_matrix(panel: PanelFrame, feats: Sequence[str]) -> np.ndarray:
    """Collect ``feats`` into a float ``(n_rows, n_features)`` numpy matrix."""
    frame = panel.lazy().select(list(feats)).collect()
    return frame.to_numpy().astype(float, copy=False)


def _impute_column_mean(mat: np.ndarray) -> np.ndarray:
    """Fill NaNs with the (train-only) per-column mean; all-NaN columns -> 0."""
    if not np.isnan(mat).any():
        return mat
    out = mat.copy()
    with np.errstate(invalid="ignore"):
        col_mean = np.nanmean(out, axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    nan_idx = np.where(np.isnan(out))
    out[nan_idx] = np.take(col_mean, nan_idx[1])
    return out


def _percentileofscore(values: np.ndarray) -> np.ndarray:
    """Percentile rank (``kind="weak"``) of each value among ``values``.

    Reimplements ``scipy.stats.percentileofscore(values, values)`` so the
    package carries no ``scipy`` dependency: the percentile of ``x`` is the
    percentage of entries ``<= x``.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return v
    # For each value, fraction of entries <= it (broadcast comparison).
    leq = (v[None, :] <= v[:, None]).sum(axis=1)
    return 100.0 * leq / n


def select_top(
    importance_df: pl.DataFrame,
    *,
    k: int | None = None,
    variability: float | None = None,
    feature_col: str = "feature",
    importance_col: str = "importance",
) -> list[str]:
    """Select features from a ranking table by top-``k`` or cumulative importance.

    Parameters
    ----------
    importance_df : polars.DataFrame
        A ranking table with a ``feature`` column and a numeric importance
        column (e.g. the output of :func:`projection_importance`).
    k : int, optional
        If given, keep the ``k`` highest-importance features.
    variability : float, optional
        If given (and ``k`` is ``None``), keep the smallest set of top features
        whose cumulative share of the total importance reaches ``variability``
        (e.g. ``0.90`` for 90%).
    feature_col, importance_col : str
        Column names in ``importance_df``.

    Returns
    -------
    list of str
        Selected feature names, most important first.
    """
    ranked = importance_df.sort(importance_col, descending=True)
    names = ranked.get_column(feature_col).to_list()
    if k is not None:
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"`k` must be a positive integer, got {k!r}.")
        return names[:k]
    if variability is not None:
        if not 0.0 < variability <= 1.0:
            raise ValueError(f"`variability` must be in (0, 1], got {variability!r}.")
        imp = ranked.get_column(importance_col).to_numpy().astype(float)
        total = float(imp.sum())
        if total <= 0.0:
            return names
        cum = np.cumsum(imp) / total
        n = int(np.searchsorted(cum, variability) + 1)
        return names[: max(1, min(n, len(names)))]
    return names


# --------------------------------------------------------------------------- #
# Principal Feature Analysis
# --------------------------------------------------------------------------- #
def pfa(
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    k: int,
    *,
    features: Sequence[str] | None = None,
    n_components: int | None = None,
    variance_threshold: float = 0.95,
    random_state: int = 0,
    entity: str | None = None,
    time: str | None = None,
) -> list[str]:
    """Select ``k`` non-redundant features by Principal Feature Analysis (PFA).

    Algorithm (Lu et al., 2007), reimplemented leak-safely:

    1. Standardize the feature matrix using **train-only** per-column mean/std.
    2. Fit PCA and keep the leading ``q`` components -- enough to retain
       ``variance_threshold`` of the variance (or ``n_components`` if given).
    3. Form the loading matrix ``A_q = components_.T`` (one row per **feature**,
       ``q`` columns): each row is that feature's coordinates in PC space.
    4. KMeans-cluster the feature loading vectors into ``k`` clusters. Features
       in the same cluster load on the principal components alike, i.e. they are
       mutually redundant.
    5. From each cluster keep the single feature whose loading vector is closest
       to the cluster centroid -- one minimally-redundant representative.

    Everything is computed **only from the rows in** ``X``, so calling ``pfa``
    on a training fold cannot leak test-fold structure. It is deterministic for
    a fixed ``random_state``.

    Parameters
    ----------
    X : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format panel. Bare frames are wrapped via
        :func:`~polars_features.core.panel_frame.as_panel`.
    k : int
        Number of features (clusters) to select.
    features : sequence of str, optional
        Restrict the candidate pool. Defaults to every numeric feature column.
    n_components : int, optional
        Number of principal components to keep. If ``None``, the smallest number
        of components reaching ``variance_threshold`` is used (at least 1).
    variance_threshold : float, default=0.95
        Cumulative explained-variance target used when ``n_components`` is None.
    random_state : int, default=0
        Seed for PCA / KMeans (determinism).
    entity, time : str, optional
        Panel keys, used only when ``X`` is a bare polars frame.

    Returns
    -------
    list of str
        The selected feature names (one representative per cluster).

    References
    ----------
    Lu, Y., Cohen, I., Zhou, X. S., & Tian, Q. (2007). "Feature selection using
    principal feature analysis." *ACM Multimedia*.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    panel = as_panel(X, entity, time)
    feats = _resolve_unsup_features(panel, features)

    if not isinstance(k, int) or k < 1:
        raise ValueError(f"`k` must be a positive integer, got {k!r}.")
    if k > len(feats):
        raise ValueError(
            f"`k`={k} exceeds the number of candidate features ({len(feats)}: {feats})."
        )

    mat = _impute_column_mean(_feature_matrix(panel, feats))
    xs = StandardScaler().fit_transform(mat)
    n_features = xs.shape[1]

    if n_components is None:
        pca = PCA(random_state=random_state).fit(xs)
        cum = np.cumsum(pca.explained_variance_ratio_)
        q = int(np.searchsorted(cum, variance_threshold) + 1)
        q = max(1, min(q, n_features))
        loadings = pca.components_[:q]
    else:
        q = max(1, min(int(n_components), n_features))
        loadings = PCA(n_components=q, random_state=random_state).fit(xs).components_

    a_q = loadings.T  # (n_features, q): one loading vector per feature.

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(a_q)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    selected: list[tuple[int, float]] = []
    for c in range(k):
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        dists = np.linalg.norm(a_q[idx] - centers[c], axis=1)
        best_local = int(np.argmin(dists))
        selected.append((int(idx[best_local]), float(dists[best_local])))

    # Return names ordered by original feature order for stability.
    chosen = sorted(i for i, _ in selected)
    return [feats[i] for i in chosen]


# --------------------------------------------------------------------------- #
# Variance
# --------------------------------------------------------------------------- #
def variance(
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    k: int,
    *,
    features: Sequence[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
) -> list[str]:
    """Select the ``k`` highest-variance features (Polars-native).

    Variance is computed **only on the rows in** ``X``. Ties break on the
    original column order.

    Parameters
    ----------
    X : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format panel.
    k : int
        Number of features to keep.
    features : sequence of str, optional
        Candidate pool (defaults to all numeric feature columns).
    entity, time : str, optional
        Panel keys for bare frames.

    Returns
    -------
    list of str
        The ``k`` highest-variance feature names, highest first.
    """
    panel = as_panel(X, entity, time)
    feats = _resolve_unsup_features(panel, features)
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"`k` must be a positive integer, got {k!r}.")
    if k > len(feats):
        raise ValueError(
            f"`k`={k} exceeds the number of candidate features ({len(feats)}: {feats})."
        )
    var_row = (
        panel.lazy().select([pl.col(f).var().alias(f) for f in feats]).collect().row(0)
    )
    variances = {
        f: (0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
        for f, v in zip(feats, var_row, strict=True)
    }
    # Stable order: higher variance first, original order breaks ties.
    order = sorted(range(len(feats)), key=lambda i: (-variances[feats[i]], i))
    return [feats[i] for i in order[:k]]


# --------------------------------------------------------------------------- #
# Correlation pruning
# --------------------------------------------------------------------------- #
def correlation(
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    threshold: float = 0.95,
    *,
    features: Sequence[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
) -> list[str]:
    """Drop redundant features by absolute-correlation pruning (Polars-native).

    Features are scanned from highest to lowest variance; a feature is kept
    unless its absolute Pearson correlation with an already-kept feature exceeds
    ``threshold``. Of every highly-correlated pair the **higher-variance** member
    therefore survives. Correlations are computed **only on the rows in** ``X``.

    Parameters
    ----------
    X : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format panel.
    threshold : float, default=0.95
        Absolute-correlation cutoff in ``(0, 1]``. Pairs above it are pruned.
    features : sequence of str, optional
        Candidate pool (defaults to all numeric feature columns).
    entity, time : str, optional
        Panel keys for bare frames.

    Returns
    -------
    list of str
        The retained feature names (original column order).
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"`threshold` must be in (0, 1], got {threshold!r}.")
    panel = as_panel(X, entity, time)
    feats = _resolve_unsup_features(panel, features)

    mat = _impute_column_mean(_feature_matrix(panel, feats))
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(mat, rowvar=False)
    abs_corr = np.nan_to_num(np.abs(np.atleast_2d(corr)), nan=0.0)
    var = np.nan_to_num(np.nanvar(mat, axis=0), nan=0.0)

    # Scan highest-variance first so the survivor of a redundant pair is the
    # higher-variance feature.
    order = sorted(range(len(feats)), key=lambda i: (-var[i], i))
    kept: list[int] = []
    for i in order:
        if all(abs_corr[i, j] <= threshold for j in kept):
            kept.append(i)
    # Return in original column order for stability.
    return [feats[i] for i in sorted(kept)]


# --------------------------------------------------------------------------- #
# Projection-based importance
# --------------------------------------------------------------------------- #
def projection_importance(
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    method: str = "gaussian",
    *,
    n_components: int | None = None,
    random_state: int = 42,
    features: Sequence[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
) -> pl.DataFrame:
    """Score feature energy in a projected space (unsupervised importance).

    Each feature's importance is the total energy of its column in the
    projection's loading matrix -- a cheap, model-free redundancy/energy
    ranking. Computed **only on the rows in** ``X`` and deterministic for a
    fixed ``random_state``.

    Parameters
    ----------
    X : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format panel.
    method : {"gaussian", "sparse", "ica", "svd"}, default="gaussian"
        Projector:

        * ``"gaussian"`` -- :class:`~sklearn.random_projection.GaussianRandomProjection`.
        * ``"sparse"`` -- :class:`~sklearn.random_projection.SparseRandomProjection`.
        * ``"ica"`` -- :class:`~sklearn.decomposition.FastICA`.
        * ``"svd"`` -- :class:`~sklearn.decomposition.TruncatedSVD`.
    n_components : int, optional
        Number of projected components. Defaults to a sensible value per method.
    random_state : int, default=42
        Seed for the projector.
    features : sequence of str, optional
        Candidate pool (defaults to all numeric feature columns).
    entity, time : str, optional
        Panel keys for bare frames.

    Returns
    -------
    polars.DataFrame
        Columns ``feature``, ``importance`` and ``importance_percentile``,
        sorted by ``importance`` descending.
    """
    valid = ("gaussian", "sparse", "ica", "svd")
    if method not in valid:
        raise ValueError(f"`method` must be one of {valid}, got {method!r}.")

    panel = as_panel(X, entity, time)
    feats = _resolve_unsup_features(panel, features)
    mat = np.nan_to_num(_feature_matrix(panel, feats), nan=0.0)
    n_features = mat.shape[1]

    if n_components is None:
        # A reduced projection; SVD requires strictly fewer components.
        nc = max(1, min(n_features, max(2, n_features)))
    else:
        nc = max(1, int(n_components))

    if method in ("gaussian", "sparse"):
        from sklearn.random_projection import (
            GaussianRandomProjection,
            SparseRandomProjection,
        )

        cls = (
            GaussianRandomProjection if method == "gaussian" else SparseRandomProjection
        )
        proj = cls(n_components=min(nc, n_features), random_state=random_state).fit(mat)
        comp = proj.components_  # (n_components, n_features), possibly sparse
        if hasattr(comp, "power"):  # scipy sparse (SparseRandomProjection)
            energy = np.asarray(comp.power(2).sum(axis=0)).ravel()
        else:
            energy = np.sum(np.asarray(comp) ** 2, axis=0)
    elif method == "ica":
        from sklearn.decomposition import FastICA

        proj = FastICA(
            n_components=min(nc, n_features),
            random_state=random_state,
            max_iter=1000,
        ).fit(mat)
        energy = np.linalg.norm(proj.components_, axis=0)
    else:  # svd
        from sklearn.decomposition import TruncatedSVD

        nc_svd = max(1, min(nc, n_features - 1)) if n_features > 1 else 1
        proj = TruncatedSVD(n_components=nc_svd, random_state=random_state).fit(mat)
        energy = np.sum(proj.components_**2, axis=0)

    energy = np.asarray(energy, dtype=float)
    out = pl.DataFrame(
        {
            "feature": feats,
            "importance": energy,
            "importance_percentile": _percentileofscore(energy),
        }
    )
    return out.sort("importance", descending=True)


# --------------------------------------------------------------------------- #
# Pipeline-friendly transformers
# --------------------------------------------------------------------------- #
class _UnsupervisedSelector(PanelTransformer):
    """Shared base for unsupervised ``"select"`` steps.

    Concrete subclasses implement :meth:`_fit` (computing ``self.selected_``)
    and declare ``panel_safe`` / ``leakage_safe``. This base is abstract (it does
    not implement ``_fit``) and supplies the common constructor, the projecting
    :meth:`_transform`, and the sklearn-style :meth:`get_feature_names_out` /
    :attr:`support_`.
    """

    def __init__(
        self,
        *,
        target: str | None = None,
        features: Sequence[str] | None = None,
        keep_target: bool = True,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self.target = target
        self.features = list(features) if features is not None else None
        self.keep_target = bool(keep_target)
        self.selected_: list[str] = []
        self.target_: str | None = None
        self.feature_names_in_: list[str] = []

    def _candidate_features(self, panel: PanelFrame) -> list[str]:
        exclude = {self.target} if self.target else set()
        return _resolve_unsup_features(panel, self.features, exclude=exclude)

    def _resolve_target(self, panel: PanelFrame) -> str | None:
        if self.target is None:
            return None
        if self.target not in panel:
            raise ValueError(
                f"target column {self.target!r} not found in panel. "
                f"Available columns: {panel.columns}."
            )
        return self.target

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        missing = [c for c in self.selected_ if c not in panel]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.transform: selected column(s) {missing} "
                f"not found in panel. Available columns: {panel.columns}."
            )
        keep = list(self.selected_)
        if self.keep_target and self.target_ is not None and self.target_ in panel:
            keep.append(self.target_)
        return panel.select(*(pl.col(c) for c in keep))

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> list[str]:
        """Return the selected feature names (sklearn-style)."""
        self._check_fitted("get_feature_names_out")
        return list(self.selected_)

    @property
    def support_(self) -> list[bool]:
        """Boolean mask over the fitted candidate pool (sklearn-style)."""
        self._check_fitted("support_")
        chosen = set(self.selected_)
        return [f in chosen for f in self.feature_names_in_]


class PFASelector(_UnsupervisedSelector):
    """Keep ``k`` non-redundant features chosen by :func:`pfa` (a ``"select"`` step).

    Fits by running :func:`pfa` on the training panel and remembering the chosen
    feature names; transforms by projecting any panel onto ``(entity, time)`` +
    the selected features (+ the target, if configured). Selection happens
    **only on the fit panel**, so the step is leak-safe inside cross-validation.

    Parameters
    ----------
    k : int
        Number of features (clusters) to select.
    target : str, optional
        A target column to *exclude* from selection candidates and (if
        ``keep_target``) retain in the transformed panel so a downstream
        estimator can still see it.
    features : sequence of str, optional
        Candidate pool (see :func:`pfa`).
    keep_target : bool, default=True
        Retain ``target`` in the transformed panel.
    n_components, variance_threshold, random_state
        Forwarded to :func:`pfa`.
    entity, time : str, optional
        Default panel keys for bare polars frames.

    Attributes
    ----------
    selected_ : list of str
        The chosen feature names (available after :meth:`fit`).
    feature_names_in_ : list of str
        The candidate pool considered at fit time.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        k: int,
        *,
        target: str | None = None,
        features: Sequence[str] | None = None,
        keep_target: bool = True,
        n_components: int | None = None,
        variance_threshold: float = 0.95,
        random_state: int = 0,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(
            target=target,
            features=features,
            keep_target=keep_target,
            entity=entity,
            time=time,
        )
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"`k` must be a positive integer, got {k!r}.")
        self.k = k
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.random_state = random_state

    def _fit(self, panel: PanelFrame) -> None:
        self.target_ = self._resolve_target(panel)
        self.feature_names_in_ = self._candidate_features(panel)
        self.selected_ = pfa(
            panel,
            self.k,
            features=self.feature_names_in_,
            n_components=self.n_components,
            variance_threshold=self.variance_threshold,
            random_state=self.random_state,
        )


class VarianceSelector(_UnsupervisedSelector):
    """Keep the ``k`` highest-variance features (a ``"select"`` step).

    Wraps :func:`variance`. Selection is frozen at fit time, so it is leak-safe
    inside cross-validation.

    Parameters
    ----------
    k : int
        Number of features to keep.
    target, features, keep_target, entity, time
        See :class:`PFASelector`.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        k: int,
        *,
        target: str | None = None,
        features: Sequence[str] | None = None,
        keep_target: bool = True,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(
            target=target,
            features=features,
            keep_target=keep_target,
            entity=entity,
            time=time,
        )
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"`k` must be a positive integer, got {k!r}.")
        self.k = k

    def _fit(self, panel: PanelFrame) -> None:
        self.target_ = self._resolve_target(panel)
        self.feature_names_in_ = self._candidate_features(panel)
        self.selected_ = variance(panel, self.k, features=self.feature_names_in_)


class CorrelationSelector(_UnsupervisedSelector):
    """Drop redundant features by absolute-correlation pruning (a ``"select"`` step).

    Wraps :func:`correlation`. Selection is frozen at fit time, so it is
    leak-safe inside cross-validation.

    Parameters
    ----------
    threshold : float, default=0.95
        Absolute-correlation cutoff (see :func:`correlation`).
    target, features, keep_target, entity, time
        See :class:`PFASelector`.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        threshold: float = 0.95,
        *,
        target: str | None = None,
        features: Sequence[str] | None = None,
        keep_target: bool = True,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(
            target=target,
            features=features,
            keep_target=keep_target,
            entity=entity,
            time=time,
        )
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"`threshold` must be in (0, 1], got {threshold!r}.")
        self.threshold = threshold

    def _fit(self, panel: PanelFrame) -> None:
        self.target_ = self._resolve_target(panel)
        self.feature_names_in_ = self._candidate_features(panel)
        self.selected_ = correlation(
            panel, self.threshold, features=self.feature_names_in_
        )
