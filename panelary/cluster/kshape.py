"""Leak-safe panel k-Shape clusterer (time-series-shape clustering, mode b).

:class:`KShapeClusterer` clusters each entity's time-series *shape* using the
numpy k-Shape engine (:mod:`panelary.cluster._kshape`) and emits, per
``(entity, time)`` row, the Shape-Based Distance to each learned centroid plus a
hard cluster label.

Leak-safety
-----------
* ``fit`` learns centroids from the **training rows only**; they become frozen
  state (``self.centroids_``), so under a purged / walk-forward split the
  centroids are refit per fold automatically (``leakage_safe = True``).
* The tensor is built causally (forward-fill only, expanding z-norm) via
  :func:`panelary.cluster._tensor.build_tensor` -- no ``bfill``, no
  full-sample scaler.
* In the row-level (``"expanding"`` / integer-window) modes, a row's distance at
  time ``t`` uses only that entity's window **ending at ``t``**, so appending
  future rows never changes an earlier row's assignment.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from panelary.cluster._kshape import KShapeCore, ncc
from panelary.cluster._tensor import build_tensor
from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelEstimator

__all__ = ["KShapeClusterer", "k_from_n_entities"]


def k_from_n_entities(n_entities: int) -> int:
    """Heuristic number of clusters from the entity count (SovAI-derived).

    Returns 3 up to 10 entities, grows roughly linearly to ~9 by 100 entities,
    then eases toward a hard cap of 12. Always at least 1.

    Ported from ``clustering.calculate_number_of_clusters`` with the original's
    dead post-``return`` code removed and an explicit floor/cap applied.
    """
    import math

    n = int(n_entities)
    if n <= 10:
        k = 3
    elif n <= 100:
        k = int(5 + 6 * (n - 10) / 90)
    else:
        k = int(9 + 3 / (1 + math.exp(-0.03 * (n - 100))))
    return max(1, min(k, 12))


class KShapeClusterer(PanelEstimator):
    """Cluster entity time-series shapes with k-Shape; emit distances + labels.

    Parameters
    ----------
    value : str or sequence of str
        Feature column(s) whose per-entity shape is clustered. Multiple columns
        are clustered jointly as a multivariate series.
    n_clusters : int, optional
        Number of shape clusters. If ``None`` (default) it is chosen from the
        training entity count via :func:`k_from_n_entities` (and clamped to at
        most the number of training entities).
    window : {"static", "expanding"} or int, default="expanding"
        How predict-time distances are scored against the frozen centroids:

        * ``"static"`` -- one distance vector / label per entity, from that
          entity's whole series (a per-entity summary, constant over time).
        * ``"expanding"`` -- walk-forward: at each row use the entity's window
          *ending at that time*. Causal.
        * ``int`` -- like ``"expanding"`` but with a trailing window of the given
          fixed length.
    z_normalize : bool, default=True
        Apply per-entity expanding (causal) z-normalisation when building the
        tensor.
    max_iter : int, default=100
        Maximum k-Shape Lloyd iterations.
    centroid_init : {"zero", "random"}, default="zero"
        Centroid initialisation strategy.
    seed : int, default=42
        Seed for the k-Shape engine (deterministic results).
    emit : {"both", "distances", "label"}, default="both"
        Which feature columns to emit.
    prefix : str, default="ksh"
        Prefix for emitted column names: ``f"{prefix}_dist_{i}"`` and
        ``f"{prefix}_label"``.
    min_history : int, default=2
        Minimum observations before a row is scored; earlier rows emit nulls.
    entity, time : str, optional
        Default panel keys used when a bare polars frame is passed.

    Attributes
    ----------
    centroids_ : numpy.ndarray
        ``(k, length, n_values)`` frozen shape centroids learned on train.
    labels_ : numpy.ndarray
        Training-entity hard labels.
    k_ : int
        The resolved number of clusters.
    value_cols_ : list of str
        The resolved value columns.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        value: str | Sequence[str],
        *,
        n_clusters: int | None = None,
        window: str | int = "expanding",
        z_normalize: bool = True,
        max_iter: int = 100,
        centroid_init: str = "zero",
        seed: int = 42,
        emit: str = "both",
        prefix: str = "ksh",
        min_history: int = 2,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self.value = value
        self.n_clusters = n_clusters
        self.window = window
        self.z_normalize = z_normalize
        self.max_iter = max_iter
        self.centroid_init = centroid_init
        self.seed = seed
        if emit not in ("both", "distances", "label"):
            raise ValueError(
                f"emit must be 'both', 'distances' or 'label', got {emit!r}."
            )
        self.emit = emit
        self.prefix = prefix
        self.min_history = min_history
        if not (window == "static" or window == "expanding" or isinstance(window, int)):
            raise ValueError(
                f"window must be 'static', 'expanding' or an int, got {window!r}."
            )
        # learned state
        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.k_: int | None = None
        self.value_cols_: list[str] = []

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _resolve_value_cols(self, panel: PanelFrame) -> list[str]:
        cols = [self.value] if isinstance(self.value, str) else list(self.value)
        missing = [c for c in cols if c not in panel]
        if missing:
            raise ValueError(
                f"{type(self).__name__}: value column(s) {missing} not found in "
                f"panel. Available columns: {panel.columns}."
            )
        return cols

    @staticmethod
    def _coerce_len(win: np.ndarray, length: int) -> np.ndarray:
        """Coerce a ``(w, dims)`` window to ``(length, dims)`` (last-``length`` / left-pad)."""
        w = win.shape[0]
        if w == length:
            return win
        if w > length:
            return win[-length:]
        pad = np.zeros((length - w, win.shape[1]))
        return np.concatenate([pad, win], axis=0)

    def _dist_to_centroids(self, win: np.ndarray) -> np.ndarray:
        assert self.centroids_ is not None
        k = self.centroids_.shape[0]
        return np.array([1.0 - ncc(win, self.centroids_[q]).max() for q in range(k)])

    # ------------------------------------------------------------------ #
    # PanelEstimator hooks
    # ------------------------------------------------------------------ #
    def _fit(self, panel: PanelFrame) -> None:
        value_cols = self._resolve_value_cols(panel)
        pt = build_tensor(
            panel,
            value_cols,
            forward_fill=True,
            z_normalize=self.z_normalize,
            min_history=self.min_history,
        )
        n_ent = pt.tensor.shape[0]
        k = self.n_clusters if self.n_clusters is not None else k_from_n_entities(n_ent)
        k = max(1, min(k, n_ent))

        core = KShapeCore(
            k,
            centroid_init=self.centroid_init,
            max_iter=self.max_iter,
            seed=self.seed,
        ).fit(pt.tensor)  # NaNs -> 0 inside the engine

        self.centroids_ = core.centroids_
        self.labels_ = core.labels_
        self.k_ = k
        self.value_cols_ = value_cols

    def _predict(self, panel: PanelFrame) -> PanelFrame:
        if self.centroids_ is None or self.k_ is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted.")
        ent, tim = panel.entity_col, panel.time_col
        pt = build_tensor(
            panel,
            self.value_cols_,
            forward_fill=True,
            z_normalize=self.z_normalize,
            min_history=self.min_history,
        )
        tensor = np.nan_to_num(pt.tensor)
        n_ent, n_times, _ = tensor.shape
        length = self.centroids_.shape[1]
        k = self.k_

        dist = np.full((n_ent, n_times, k), np.nan)
        for i in range(n_ent):
            series = tensor[i]  # (n_times, n_values)
            if self.window == "static":
                dvec = self._dist_to_centroids(self._coerce_len(series, length))
                if n_times >= self.min_history:
                    dist[i, :, :] = dvec
                continue
            for tj in range(n_times):
                available = tj + 1
                if isinstance(self.window, int):
                    start = max(0, tj - self.window + 1)
                    available = min(available, self.window)
                else:  # "expanding"
                    start = 0
                if available < self.min_history:
                    continue
                win = self._coerce_len(series[start : tj + 1], length)
                dist[i, tj, :] = self._dist_to_centroids(win)

        flat = dist.reshape(n_ent * n_times, k)
        labels = np.full(n_ent * n_times, np.nan)
        good = ~np.isnan(flat).any(axis=1)
        if good.any():
            labels[good] = flat[good].argmin(axis=1)

        dist_cols = [f"{self.prefix}_dist_{i}" for i in range(k)]
        label_col = f"{self.prefix}_label"
        dense = pt.keys.with_columns(
            [pl.Series(dist_cols[i], flat[:, i]) for i in range(k)]
        ).with_columns(pl.Series(label_col, labels))
        # NaN -> null; label -> Int32.
        dense = dense.with_columns(
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

        if self.emit == "distances":
            emit_cols = dist_cols
        elif self.emit == "label":
            emit_cols = [label_col]
        else:
            emit_cols = [*dist_cols, label_col]

        keys = panel.lazy().select([ent, tim]).collect()
        out = keys.join(dense.select([ent, tim, *emit_cols]), on=[ent, tim], how="left")
        return PanelFrame(out, entity=ent, time=tim, validate=False)

    def _get_params_repr(self) -> dict[str, Any]:  # pragma: no cover - cosmetic
        return {"value": self.value, "k": self.k_, "window": self.window}
