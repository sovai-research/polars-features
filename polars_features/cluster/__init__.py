"""Leak-safe panel clustering for PanelKit.

Two clustering modes, both honouring the
:class:`~polars_features.core.protocol.PanelEstimator` contract (fit-on-train,
frozen state, per-fold refit under purged CV):

* :class:`KShapeClusterer` -- mode (b): cluster each entity's **time-series
  shape** with the numpy k-Shape engine and emit, per ``(entity, time)`` row,
  the Shape-Based Distance to each learned centroid plus a hard cluster label.
* :class:`CrossSectionalClusterer` -- mode (a): cluster **entities by their
  same-date feature vectors** with scikit-learn (KMeans / agglomerative), scaler
  and centres fit on the training rows only.

Also exported: :func:`sbd` / :func:`ncc` (numpy Shape-Based-Distance /
normalized-cross-correlation utilities) and :func:`k_from_n_entities` (the
cluster-count heuristic).

Ported from Sov.ai's ``core_kshape.py`` / ``clustering.py`` (first-party; reuse
authorized) with the original leaks removed -- see :mod:`._tensor`.

Public symbols are guarded: if an optional dependency (e.g. scikit-learn) is
missing, the affected symbol is simply omitted from :data:`__all__` rather than
breaking the import.
"""

from __future__ import annotations

import warnings

__all__: list[str] = []


def _warn_unavailable(component: str, exc: Exception) -> None:
    warnings.warn(
        f"polars_features.cluster: optional component {component!r} is "
        f"unavailable ({exc.__class__.__name__}: {exc}).",
        stacklevel=3,
    )


# Numpy-only k-Shape engine + utilities (no optional deps).
try:
    from polars_features.cluster._kshape import ncc, sbd
    from polars_features.cluster.kshape import KShapeClusterer, k_from_n_entities
except ImportError as exc:  # pragma: no cover - defensive
    _warn_unavailable("kshape", exc)
else:
    __all__ += ["KShapeClusterer", "k_from_n_entities", "ncc", "sbd"]

# Cross-sectional clusterer (needs scikit-learn, imported lazily inside fit).
try:
    from polars_features.cluster.cross_sectional import CrossSectionalClusterer
except ImportError as exc:  # pragma: no cover - defensive
    _warn_unavailable("cross_sectional", exc)
else:
    __all__ += ["CrossSectionalClusterer"]
