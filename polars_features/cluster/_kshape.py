"""Clean-room, dependency-free numpy port of the k-Shape clustering engine.

k-Shape (Paparrizos & Gravano, SIGMOD 2015) clusters time series by *shape*
rather than by amplitude or phase. Two ingredients do the work:

* **Normalized cross-correlation (NCC)** computed via the FFT
  (:func:`ncc`) -- the shift-invariant similarity kernel. This is the dominant
  runtime cost (``m * k`` evaluations per Lloyd iteration) and the natural
  target for a future Rust ``rustfft`` plugin; the numpy version here is the
  reference fallback.
* **Shape-Based Distance (SBD)** (:func:`sbd`) -- ``1 - max(NCC)`` over all
  circular shifts, so two series that differ only by a phase shift have distance
  ~0.

The centroid update (:func:`_extract_shape`) solves the maximisation of
Rayleigh-quotient-style scatter via the top eigenvector of ``M = P S P``.

This port intentionally drops the SovAI original's dependencies on
``sklearn.base`` and the default ``ProcessPoolExecutor`` (which broke
determinism and pickling), and replaces the module-level ``np.random`` calls
with a seeded :class:`numpy.random.Generator` so results are reproducible.

Only the numpy building blocks live here. The leak-safe, Polars-native panel
wrapper is :class:`polars_features.cluster.kshape.KShapeClusterer`.

Source: Sov.ai ``sovai/extensions/core_kshape.py`` (first-party; reuse
authorized). Leaks present in the surrounding pandas orchestration
(``bfill``, full-sample ``StandardScaler``) are *not* ported -- they live only
in the Polars tensor builder, which is causal.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft, ifft
from numpy.linalg import eigh, norm

__all__ = ["KShapeCore", "ncc", "sbd", "zscore", "roll_zeropad"]


def zscore(a: np.ndarray, axis: int = 0, ddof: int = 0) -> np.ndarray:
    """Z-normalise ``a`` along ``axis``, mapping the 0/0 case to 0.

    Mirrors ``scipy.stats.zscore`` but wraps the result in ``nan_to_num`` so a
    constant (zero-variance) series becomes all-zeros instead of NaN.
    """
    a = np.asanyarray(a, dtype=float)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)
    if axis and mns.ndim < a.ndim:
        res = (a - np.expand_dims(mns, axis=axis)) / np.expand_dims(sstd, axis=axis)
    else:
        res = (a - mns) / sstd
    return np.nan_to_num(res)


def roll_zeropad(a: np.ndarray, shift: int, axis: int | None = None) -> np.ndarray:
    """Shift ``a`` by ``shift`` positions, padding vacated slots with zeros.

    Unlike :func:`numpy.roll`, values shifted off one end are dropped (not
    wrapped) and replaced with zeros -- the alignment primitive used by SBD.
    """
    a = np.asanyarray(a)
    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n - shift), axis))
        res = np.concatenate((a.take(np.arange(n - shift, n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n - shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n - shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    return res


def _as2d(x: np.ndarray) -> np.ndarray:
    """Coerce a series to a ``(length, dims)`` float array."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return x


def ncc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation of ``x`` and ``y`` over all shifts (FFT).

    Both inputs are ``(length,)`` or ``(length, dims)``. Returns the length
    ``2 * len(x) - 1`` vector of correlation coefficients; ``ncc(x, y).max()``
    is the shift-invariant similarity in ``[-1, 1]`` (1 == identical shape).

    This is the k-Shape distance kernel and the hot loop; a batched Rust
    ``rustfft`` implementation is the intended future replacement behind this
    same signature.
    """
    x = _as2d(x)
    y = _as2d(y)
    den = norm(x) * norm(y)
    if den < 1e-9:
        den = np.inf
    x_len = x.shape[0]
    fft_size = 1 << (2 * x_len - 1).bit_length()
    cc = ifft(fft(x, fft_size, axis=0) * np.conj(fft(y, fft_size, axis=0)), axis=0)
    cc = np.concatenate((cc[-(x_len - 1) :], cc[:x_len]), axis=0)
    return np.real(cc).sum(axis=-1) / den


def sbd(x: np.ndarray, y: np.ndarray) -> float:
    """Shape-Based Distance ``1 - max(NCC(x, y))``.

    Zero (up to floating error) when ``x`` and ``y`` have the same shape,
    including when one is a shifted copy of the other -- SBD is shift-invariant.
    """
    return float(1.0 - ncc(x, y).max())


def _sbd_align(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return ``y`` shifted to best align with ``x`` (zero-padded)."""
    nc = ncc(x, y)
    idx = int(np.argmax(nc))
    shift = (idx + 1) - max(len(x), len(y))
    return roll_zeropad(y, shift)


def _collect_shift(series: np.ndarray, cur_center: np.ndarray) -> np.ndarray:
    """Align ``series`` to ``cur_center`` (identity if the centroid is all-zero)."""
    if np.all(cur_center == 0):
        return series
    return _sbd_align(cur_center, series)


def _extract_shape(
    idx: np.ndarray,
    x: np.ndarray,
    j: int,
    cur_center: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Update the centroid for cluster ``j`` as the top eigenvector of ``P S P``.

    ``x`` is ``(m, length, 1)`` (single dimension; the Lloyd loop calls this
    once per feature dimension). Members are shift-aligned to the current
    centroid, z-normalised, and the leading eigenvector of the scatter matrix is
    taken as the new shape, with a sign fix that minimises total distance.
    """
    members = [_collect_shift(x[i], cur_center) for i in range(len(idx)) if idx[i] == j]
    if len(members) == 0:
        # Empty cluster: reseed from a random member (deterministic via rng).
        i = int(rng.integers(0, x.shape[0]))
        return np.squeeze(x[i].copy())

    a = np.array(members)
    columns = a.shape[1]
    y = zscore(a, axis=1, ddof=1)
    s = np.dot(y[:, :, 0].transpose(), y[:, :, 0])
    p = np.eye(columns) - np.full((columns, columns), 1.0 / columns)
    m = np.dot(np.dot(p, s), p)
    _, vec = eigh(m)
    centroid = vec[:, -1]

    dist_pos = np.sum(np.linalg.norm(a - centroid.reshape((columns, 1)), axis=(1, 2)))
    dist_neg = np.sum(np.linalg.norm(a + centroid.reshape((columns, 1)), axis=(1, 2)))
    if dist_pos >= dist_neg:
        centroid = -centroid
    return zscore(centroid, ddof=1)


def _distance_matrix(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return the ``(m, k)`` SBD distance matrix ``1 - max NCC(series, centroid)``."""
    m = x.shape[0]
    k = centroids.shape[0]
    dist = np.empty((m, k))
    for p in range(m):
        for q in range(k):
            dist[p, q] = 1.0 - ncc(x[p], centroids[q]).max()
    return dist


def _kshape(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    centroid_init: str = "zero",
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the k-Shape Lloyd loop; return ``(labels, centroids)``.

    ``x`` is ``(m, length, dims)``. Deterministic for a given ``rng``.
    """
    m, length, dims = x.shape
    idx = rng.integers(0, k, size=m)
    if centroid_init == "zero":
        centroids = np.zeros((k, length, dims))
    elif centroid_init == "random":
        indices = rng.choice(m, k, replace=m < k)
        centroids = x[indices].copy()
    else:
        raise ValueError(
            f"centroid_init must be 'zero' or 'random', got {centroid_init!r}."
        )

    for _ in range(max_iter):
        old_idx = idx.copy()
        for j in range(k):
            for d in range(dims):
                centroids[j, :, d] = _extract_shape(
                    idx,
                    x[:, :, d : d + 1],
                    j,
                    centroids[j, :, d : d + 1],
                    rng,
                )
        dist = _distance_matrix(x, centroids)
        idx = dist.argmin(1)
        if np.array_equal(old_idx, idx):
            break
    return idx, centroids


class KShapeCore:
    """Seeded, numpy-only k-Shape clusterer with an sklearn-style fit/predict.

    Parameters
    ----------
    n_clusters : int
        Number of shape clusters ``k``.
    centroid_init : {"zero", "random"}, default="zero"
        Centroid initialisation strategy.
    max_iter : int, default=100
        Maximum Lloyd iterations.
    seed : int, default=42
        Seed for the internal :class:`numpy.random.Generator`, making both
        ``fit`` (initial assignment / empty-cluster reseeding) deterministic.

    Attributes
    ----------
    labels_ : numpy.ndarray
        Hard cluster label per training series. Available after :meth:`fit`.
    centroids_ : numpy.ndarray
        ``(k, length, dims)`` learned shape centroids. Frozen state used by
        :meth:`predict` -- this is the leak-safe seam.
    """

    def __init__(
        self,
        n_clusters: int,
        *,
        centroid_init: str = "zero",
        max_iter: int = 100,
        seed: int = 42,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.centroid_init = centroid_init
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.labels_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None

    @staticmethod
    def _as3d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[:, :, None]
        if X.ndim != 3:
            raise ValueError(
                "X must be (m, length) or (m, length, dims); "
                f"got array with shape {X.shape}."
            )
        return np.nan_to_num(X)

    def fit(self, X: np.ndarray) -> KShapeCore:
        """Learn ``n_clusters`` shape centroids from ``X`` (``(m, length[, dims])``)."""
        X = self._as3d(X)
        rng = np.random.default_rng(self.seed)
        idx, centroids = _kshape(
            X,
            self.n_clusters,
            rng,
            centroid_init=self.centroid_init,
            max_iter=self.max_iter,
        )
        self.labels_ = idx
        self.centroids_ = centroids
        return self

    def distances(self, X: np.ndarray) -> np.ndarray:
        """Return the ``(m, k)`` SBD distance matrix of ``X`` to frozen centroids."""
        if self.centroids_ is None:
            raise RuntimeError("KShapeCore is not fitted; call `fit` first.")
        X = self._as3d(X)
        return _distance_matrix(X, self.centroids_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each series in ``X`` to its nearest frozen centroid."""
        return self.distances(X).argmin(1)
