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


def _ncc_many(x: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """NCC of one series against many, batched.

    ``x`` is ``(length, dims)``, ``ys`` is ``(n, length, dims)``; the result is
    ``(n, 2 * length - 1)`` and row ``i`` equals ``ncc(x, ys[i])`` exactly -- the
    same forward transforms, products and inverse transform, just evaluated for
    every ``i`` in one call instead of re-transforming ``x`` each time.
    """
    x = _as2d(x)
    n = ys.shape[0]
    x_len = x.shape[0]
    if n == 0:
        return np.empty((0, 2 * x_len - 1))
    fft_size = 1 << (2 * x_len - 1).bit_length()
    den_x = norm(x)
    den = den_x * np.array([norm(ys[i]) for i in range(n)])
    den = np.where(den < 1e-9, np.inf, den)

    fx = fft(x, fft_size, axis=0)  # (fft_size, dims)
    out = np.empty((n, 2 * x_len - 1))
    chunk = max(1, int(_NCC_TILE_BUDGET // max(fft_size * ys.shape[2], 1)))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        fy = np.conj(fft(ys[start:stop], fft_size, axis=1))
        cc = ifft(fx[None, :, :] * fy, axis=1)
        cc = np.concatenate((cc[:, -(x_len - 1) :], cc[:, :x_len]), axis=1)
        out[start:stop] = cc.real.sum(axis=-1) / den[start:stop, None]
    return out


def _collect_shift_many(series: np.ndarray, cur_center: np.ndarray) -> np.ndarray:
    """Align every series in ``series`` (``(n, length, dims)``) to ``cur_center``.

    Batched equivalent of ``[_collect_shift(s, cur_center) for s in series]``:
    the centroid's forward transform is computed once rather than once per
    member, which is the dominant cost of the k-Shape centroid update.
    """
    if series.shape[0] == 0:
        return series
    if np.all(cur_center == 0):
        return series
    nc = _ncc_many(cur_center, series)
    shifts = nc.argmax(axis=1) + 1 - max(cur_center.shape[0], series.shape[1])
    return np.array(
        [roll_zeropad(series[i], int(shifts[i])) for i in range(series.shape[0])]
    )


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
    member_rows = np.flatnonzero(np.asarray(idx) == j)
    if member_rows.size == 0:
        # Empty cluster: reseed from a random member (deterministic via rng).
        i = int(rng.integers(0, x.shape[0]))
        return np.squeeze(x[i].copy())

    a = _collect_shift_many(x[member_rows], cur_center)
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


#: Rough ceiling on the complex128 working set of one :func:`_distance_matrix`
#: tile (~64 MB), used to pick the series-chunk size.
_NCC_TILE_BUDGET = 4_000_000


def _distance_matrix(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return the ``(m, k)`` SBD distance matrix ``1 - max NCC(series, centroid)``.

    Mathematically identical to ``1 - ncc(x[p], centroids[q]).max()`` for every
    pair, but the transforms are computed **once each** instead of once per
    pair: the naive double loop evaluates ``fft(series)`` ``k`` times and
    ``fft(centroid)`` ``m`` times.  Here the ``m + k`` forward transforms are
    done up front and only the ``m * k`` products and inverse transforms remain
    -- the dominant cost of every Lloyd iteration and of ``predict``.

    The pairwise tile is chunked over series so the complex working set stays
    bounded regardless of ``m``.
    """
    m = x.shape[0]
    k = centroids.shape[0]
    if m == 0 or k == 0:
        return np.empty((m, k))

    x_len = x.shape[1]
    fft_size = 1 << (2 * x_len - 1).bit_length()

    # Per-series / per-centroid Frobenius norms, computed exactly as `ncc` does
    # (`numpy.linalg.norm` on the 2-D slice) so the denominators are bit-identical.
    nx = np.array([norm(x[p]) for p in range(m)])
    nc = np.array([norm(centroids[q]) for q in range(k)])
    den = nx[:, None] * nc[None, :]
    den = np.where(den < 1e-9, np.inf, den)

    fx = fft(x, fft_size, axis=1)  # (m, fft_size, dims)
    fc = np.conj(fft(centroids, fft_size, axis=1))  # (k, fft_size, dims)

    dist = np.empty((m, k))
    chunk = max(1, int(_NCC_TILE_BUDGET // max(k * fft_size * x.shape[2], 1)))
    for start in range(0, m, chunk):
        stop = min(start + chunk, m)
        # (chunk, k, fft_size, dims)
        cc = ifft(fx[start:stop, None] * fc[None, :], axis=2).real
        # `ncc` keeps shifts -(x_len-1) .. x_len-1, i.e. the tail then the head
        # of the circular correlation. Only the maximum is needed here, so take
        # it from each piece instead of materialising the concatenation.
        best = cc[:, :, :x_len].sum(axis=-1).max(axis=-1)
        if x_len > 1:
            tail = cc[:, :, -(x_len - 1) :].sum(axis=-1).max(axis=-1)
            best = np.maximum(best, tail)
        dist[start:stop] = 1.0 - best / den[start:stop]
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
