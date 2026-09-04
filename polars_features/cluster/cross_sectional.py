"""Leak-safe cross-sectional entity clusterer (mode a).

:class:`CrossSectionalClusterer` clusters entities by their **same-date feature
vectors** using scikit-learn (KMeans or agglomerative). Each ``(entity, time)``
row is one observation vector; a scaler and the cluster centres are fit on the
**training rows only**, then frozen. At predict time every row is standardised
with the frozen scaler and assigned to its nearest frozen centre, so the label
at a row depends solely on that row's own features and train-learned state --
inherently time-leak-safe, exactly like the ``.xs`` cross-sectional operations.

Unlike the k-Shape clusterer this needs no temporal window: a cross-sectional
label uses only contemporaneous features, so ``leakage_safe = True`` holds by
construction as long as the scaler/centres come from ``fit`` alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from polars_features.core.panel_frame import PanelFrame
from polars_features.core.protocol import PanelEstimator

__all__ = ["CrossSectionalClusterer"]


class CrossSectionalClusterer(PanelEstimator):
    """Cluster entities by their same-date feature vectors (KMeans/agglomerative).

    Parameters
    ----------
    features : str or sequence of str
        Feature columns forming each entity-at-date vector.
    n_clusters : int, default=8
        Number of clusters.
    algo : {"kmeans", "agglomerative"}, default="kmeans"
        Clustering algorithm (scikit-learn, imported lazily). For
        ``"agglomerative"`` -- which has no native ``predict`` -- cluster centres
        are computed as the per-label means of the scaled training points and
        used for nearest-centre assignment at predict time.
    scaler : {"standard", "none"}, default="standard"
        Whether to fit a :class:`sklearn.preprocessing.StandardScaler` on the
        training rows (applied to every row) before clustering.
    seed : int, default=42
        Random state for KMeans (deterministic).
    emit : {"label", "distances", "both"}, default="label"
        Which feature columns to emit.
    prefix : str, default="xcl"
        Prefix for emitted names: ``f"{prefix}_label"`` and
        ``f"{prefix}_dist_{i}"``.
    entity, time : str, optional
        Default panel keys used when a bare polars frame is passed.

    Attributes
    ----------
    scaler_ : object or None
        The fitted scaler (or ``None`` when ``scaler="none"``).
    cluster_centers_ : numpy.ndarray
        ``(k, n_features)`` frozen cluster centres (in scaled space).
    features_ : list of str
        The resolved feature columns.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        features: str | Sequence[str],
        *,
        n_clusters: int = 8,
        algo: str = "kmeans",
        scaler: str = "standard",
        seed: int = 42,
        emit: str = "label",
        prefix: str = "xcl",
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self.features = features
        self.n_clusters = int(n_clusters)
        if algo not in ("kmeans", "agglomerative"):
            raise ValueError(f"algo must be 'kmeans' or 'agglomerative', got {algo!r}.")
        self.algo = algo
        if scaler not in ("standard", "none"):
            raise ValueError(f"scaler must be 'standard' or 'none', got {scaler!r}.")
        self.scaler = scaler
        self.seed = int(seed)
        if emit not in ("label", "distances", "both"):
            raise ValueError(
                f"emit must be 'label', 'distances' or 'both', got {emit!r}."
            )
        self.emit = emit
        self.prefix = prefix
        # learned state
        self.scaler_: Any | None = None
        self.cluster_centers_: np.ndarray | None = None
        self.features_: list[str] = []

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _resolve_features(self, panel: PanelFrame) -> list[str]:
        cols = (
            [self.features] if isinstance(self.features, str) else list(self.features)
        )
        missing = [c for c in cols if c not in panel]
        if missing:
            raise ValueError(
                f"{type(self).__name__}: feature column(s) {missing} not found in "
                f"panel. Available columns: {panel.columns}."
            )
        return cols

    def _matrix(self, panel: PanelFrame, cols: list[str]) -> np.ndarray:
        frame = (
            panel.lazy().select([pl.col(c).cast(pl.Float64) for c in cols]).collect()
        )
        return frame.to_numpy()

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is None:
            return X
        return self.scaler_.transform(X)

    # ------------------------------------------------------------------ #
    # PanelEstimator hooks
    # ------------------------------------------------------------------ #
    def _fit(self, panel: PanelFrame) -> None:
        cols = self._resolve_features(panel)
        X = self._matrix(panel, cols)
        # Drop rows with any null/NaN feature for fitting only.
        mask = ~np.isnan(X).any(axis=1)
        Xfit = X[mask]
        if Xfit.shape[0] == 0:
            raise ValueError(
                f"{type(self).__name__}: no complete rows to fit on "
                f"(all rows have a null in features {cols})."
            )
        k = max(1, min(self.n_clusters, Xfit.shape[0]))

        if self.scaler == "standard":
            from sklearn.preprocessing import StandardScaler

            self.scaler_ = StandardScaler().fit(Xfit)
            Xfit = self.scaler_.transform(Xfit)
        else:
            self.scaler_ = None

        if self.algo == "kmeans":
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=k, random_state=self.seed, n_init=10).fit(Xfit)
            centers = model.cluster_centers_
        else:
            from sklearn.cluster import AgglomerativeClustering

            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(Xfit)
            centers = np.vstack([Xfit[labels == c].mean(axis=0) for c in range(k)])

        self.cluster_centers_ = centers
        self.features_ = cols

    def _predict(self, panel: PanelFrame) -> PanelFrame:
        if self.cluster_centers_ is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted.")
        ent, tim = panel.entity_col, panel.time_col
        missing = [c for c in self.features_ if c not in panel]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.predict: feature column(s) {missing} not "
                f"found in panel. Available columns: {panel.columns}."
            )
        keys = panel.lazy().select([ent, tim]).collect()
        X = self._matrix(panel, self.features_)
        valid = ~np.isnan(X).any(axis=1)

        centers = self.cluster_centers_
        k = centers.shape[0]
        n = X.shape[0]
        dist = np.full((n, k), np.nan)
        if valid.any():
            Xs = self._scale(X[valid])
            # Euclidean distance to each frozen centre.
            d = np.linalg.norm(Xs[:, None, :] - centers[None, :, :], axis=2)
            dist[valid] = d

        labels = np.full(n, np.nan)
        good = ~np.isnan(dist).any(axis=1)
        if good.any():
            labels[good] = dist[good].argmin(axis=1)

        dist_cols = [f"{self.prefix}_dist_{i}" for i in range(k)]
        label_col = f"{self.prefix}_label"
        out = keys.with_columns(
            [pl.Series(dist_cols[i], dist[:, i]) for i in range(k)]
        ).with_columns(pl.Series(label_col, labels))
        out = out.with_columns(
            [
                pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
                for c in dist_cols
            ]
        ).with_columns(
            pl.when(pl.col(label_col).is_nan())
            .then(None)
            .otherwise(pl.col(label_col))
            .cast(pl.Int32)
            .alias(label_col)
        )

        if self.emit == "label":
            emit_cols = [label_col]
        elif self.emit == "distances":
            emit_cols = dist_cols
        else:
            emit_cols = [label_col, *dist_cols]

        out = out.select([ent, tim, *emit_cols])
        return PanelFrame(out, entity=ent, time=tim, validate=False)
