"""Backward sup-ADF (BSADF) sequences from the prefix-sum moment engine.

For each endpoint ``t2`` the backward sup-ADF statistic is

.. math::

    BSADF_{t_2}(r_0) = \\sup_{t_1 \\le t_2 - w_{\\min} + 1} ADF_{t_1}^{t_2},

the supremum of the right-tailed ADF ``t``-statistic over every backward
expanding window ending at ``t2``. The recursive *sup* of these, over ``t2``, is
the GSADF statistic of Phillips, Shi & Yu (2015); the sequence itself is what
date-stamps the origination and collapse of an explosive episode.

Everything here is a thin, careful driver on top of
:mod:`polars_features.detect._moments`: build the blocked cumulative sufficient
statistics once, then evaluate whole rectangles of ``(end, start)`` pairs with
elementwise NumPy. Three knobs control the cost / accuracy trade:

``grid``
    Instead of every start, test a **geometric ladder** of window *lengths*
    ``w_j = ceil(w_0 * gamma^j)``. Over 20 random walks of length 600 with
    ``w_min = 60``, 32 rungs cost a mean shortfall of ``0.025`` ``t``-units
    (worst series ``0.050``) against the exhaustive supremum, correlation
    ``0.998``, for ``10x`` less work -- and because the ladder is a
    deterministic function of the endpoint alone, it stays exactly
    point-in-time.
``chunk_windows``
    The dense ``(K, k, k)`` Gram batch grows quadratically in ``T``
    (``T = 5000`` exhaustive is ``1.57 GB`` at once), so endpoints are processed
    in chunks of roughly this many windows.
``refine_top_k``
    Keep the ``k`` best candidates per endpoint from the cheap pass and
    recompute *only those* by QR on the raw observations, reporting the QR
    value. Windows the prefix pass flags as numerically suspect (negative
    ``SSR``, non-positive-definite Gram) are escalated the same way. Normal
    equations square the condition number; at ``cond(X) = 1.7e8`` they returned
    ``+1.000e+00`` where QR gives ``+2.8e-09``. On a strongly drifting series
    (``T = 4000``, level ``4000``, local ``sd`` ``0.0016``) the unrefined sup was
    ``5.9e-09`` away from a full-QR sweep; ``refine_top_k=1`` closed that to
    ``6.0e-12`` for ``+25%`` runtime.

Invariants
----------
Nothing depends on ``len(y)``. The window ladder at endpoint ``t`` is a function
of ``t``, ``min_window`` and ``max_window`` only, the accumulator blocks are
anchored at row 0, and the anchor is ``y[0]``; appending future rows therefore
leaves every earlier value bitwise identical. Chunk boundaries change how the
work is grouped, never any value: the reduction is a maximum, which is
associative and order independent.

References
----------
Phillips, P. C. B., Shi, S. & Yu, J. (2015). "Testing for Multiple Bubbles:
Historical Episodes of Exuberance and Collapse in the S&P 500."
*International Economic Review* 56(4), 1043-1078. DOI 10.1111/iere.12132.

Clean-room implementation from the published equations.
"""

from __future__ import annotations

import numpy as np

from polars_features.detect._moments import (
    Moments,
    cumulative_moments,
    window_adf_full,
    window_adf_qr,
)

__all__ = ["bsadf_sequence", "bsadf_panel", "gsadf", "min_admissible_window"]

#: Default number of ``(start, end)`` pairs evaluated per batch. Sized so the
#: dense Gram batch stays in the tens of megabytes for the lag orders that
#: matter (``k <= 6``).
_CHUNK_WINDOWS = 131_072


def min_admissible_window(lag: int = 0, trend: str = "c") -> int:
    """Shortest window with at least one residual degree of freedom.

    A window of length ``w`` yields ``n = w - lag - 1`` regression rows against
    ``k`` regressors, so ``w >= lag + k + 2`` is required for ``n > k``.

    Parameters
    ----------
    lag : int, default=0
        Augmenting lags ``p``.
    trend : {"c", "ct"}, default="c"
        Deterministic terms.

    Returns
    -------
    int
    """
    k = int(lag) + 2 + (1 if trend == "ct" else 0)
    return int(lag) + k + 2


# --------------------------------------------------------------------------- #
# Window ladders
# --------------------------------------------------------------------------- #
def _ragged_starts(lo: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenated ``arange(lo[i], lo[i] + counts[i])`` without a Python loop."""
    total = int(counts.sum())
    offsets = np.zeros(counts.size, dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    within = np.arange(total, dtype=np.int64) - np.repeat(offsets, counts)
    return np.repeat(lo, counts) + within


def _geometric_lengths(w_top: np.ndarray, w_min: int, rungs: int) -> np.ndarray:
    """Geometric ladder of window lengths, one row per endpoint.

    ``w_j = ceil(w_min * gamma^j)`` with ``gamma = (w_top / w_min)^(1/(g-1))``,
    clipped into ``[w_min, w_top]`` and pinned at both ends so the shortest and
    the longest admissible window are always tested.

    Parameters
    ----------
    w_top : numpy.ndarray, shape (E,)
        Longest window available at each endpoint.
    w_min : int
        Shortest window tested.
    rungs : int
        Number of ladder rungs.

    Returns
    -------
    numpy.ndarray of int64, shape (E, rungs)
        Duplicate rungs are left in place: they cost one extra ``O(1)`` window
        each and cannot change a maximum.
    """
    top = w_top.astype(np.float64)
    if rungs <= 1:
        return top.astype(np.int64)[:, None]
    gamma = (top / float(w_min)) ** (1.0 / (rungs - 1))
    powers = np.arange(rungs, dtype=np.float64)
    lengths = np.ceil(float(w_min) * gamma[:, None] ** powers[None, :])
    lengths = np.clip(lengths, float(w_min), top[:, None]).astype(np.int64)
    lengths[:, 0] = w_min
    lengths[:, -1] = w_top
    return lengths


def _chunk_bounds(counts: np.ndarray, budget: int) -> list[tuple[int, int]]:
    """Split endpoints into runs whose window counts sum to about ``budget``."""
    cum = np.cumsum(counts)
    bounds: list[tuple[int, int]] = []
    start = 0
    n = counts.size
    while start < n:
        base = cum[start - 1] if start else 0
        stop = int(np.searchsorted(cum, base + budget, side="right"))
        stop = max(stop, start + 1)
        stop = min(stop, n)
        bounds.append((start, stop))
        start = stop
    return bounds


# --------------------------------------------------------------------------- #
# Core driver
# --------------------------------------------------------------------------- #
def _bsadf_core(
    mom: Moments,
    *,
    min_window: int,
    max_window: int | None,
    grid: int | None,
    refine_top_k: int,
    chunk_windows: int,
) -> np.ndarray:
    """BSADF for a (possibly batched) ``Moments``; returns ``(N, T)``."""
    n_entities = mom.cin.shape[0]
    n_obs = mom.n_obs
    out = np.full((n_entities, n_obs), np.nan, dtype=np.float64)

    w_min = max(int(min_window), min_admissible_window(mom.lag, mom.trend))
    w_cap = n_obs if max_window is None else int(max_window)
    if w_cap < w_min:
        return out

    endpoints = np.arange(n_obs, dtype=np.int64)
    w_top = np.minimum(endpoints + 1, w_cap)
    usable = w_top >= w_min
    endpoints = endpoints[usable]
    w_top = w_top[usable]
    if endpoints.size == 0:
        return out

    if grid is None:
        counts = (w_top - w_min + 1).astype(np.int64)
    else:
        counts = np.full(endpoints.size, max(int(grid), 1), dtype=np.int64)

    budget = max(int(chunk_windows) // max(n_entities, 1), 1)
    for lo_e, hi_e in _chunk_bounds(counts, budget):
        ends_c = endpoints[lo_e:hi_e]
        wtop_c = w_top[lo_e:hi_e]
        counts_c = counts[lo_e:hi_e]

        if grid is None:
            first_start = ends_c - wtop_c + 1
            starts = _ragged_starts(first_start, counts_c)
            ends = np.repeat(ends_c, counts_c)
        else:
            lengths = _geometric_lengths(wtop_c, w_min, int(grid))
            starts = (ends_c[:, None] - lengths + 1).ravel()
            ends = np.repeat(ends_c, counts_c)

        stat, escalate = window_adf_full(mom, starts, ends)
        vals = np.where(np.isfinite(stat), stat, -np.inf)

        offsets = np.zeros(counts_c.size, dtype=np.int64)
        np.cumsum(counts_c[:-1], out=offsets[1:])

        if refine_top_k > 0:
            best = _reduce_with_refinement(
                mom,
                vals=vals,
                escalate=escalate,
                starts=starts,
                ends=ends,
                offsets=offsets,
                counts=counts_c,
                top_k=int(refine_top_k),
            )
        else:
            best = _segment_max(vals, offsets)

        out[:, ends_c] = np.where(np.isfinite(best), best, np.nan)
    return out


def _segment_max(vals: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Maximum of ``vals[:, offsets[i]:offsets[i+1]]`` for each segment ``i``."""
    return np.maximum.reduceat(vals, offsets, axis=1)


def _reduce_with_refinement(
    mom: Moments,
    *,
    vals: np.ndarray,
    escalate: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    offsets: np.ndarray,
    counts: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Segment maximum with the top-``k`` (and any suspect) windows redone by QR.

    The reported maximum is the larger of (a) the cheap statistic over the
    windows that were *not* recomputed and (b) the QR statistic over the ones
    that were, so a candidate whose normal-equation value was spuriously large
    cannot survive, and a genuine runner-up is not lost either.
    """
    n_entities, n_windows = vals.shape
    n_seg = counts.size

    # (a) baseline over the non-candidates, candidates knocked out to -inf.
    kept = vals.copy()
    refined = np.full((n_entities, n_seg), -np.inf, dtype=np.float64)

    for e_i in range(n_entities):
        row = vals[e_i]
        picks: list[np.ndarray] = []
        for s_i in range(n_seg):
            lo = int(offsets[s_i])
            size = int(counts[s_i])
            take = min(top_k, size)
            seg = row[lo : lo + size]
            if take >= size:
                idx = np.arange(size, dtype=np.int64)
            else:
                idx = np.argpartition(seg, size - take)[size - take :]
            picks.append(idx.astype(np.int64) + lo)
        cand = np.concatenate(picks) if picks else np.empty(0, dtype=np.int64)
        flagged = np.flatnonzero(escalate[e_i]).astype(np.int64)
        cand = np.unique(np.concatenate([cand, flagged])) if flagged.size else cand
        if cand.size == 0:
            continue
        kept[e_i, cand] = -np.inf
        qr_vals = window_adf_qr(
            mom.y[e_i], starts[cand], ends[cand], lag=mom.lag, trend=mom.trend
        )
        qr_vals = np.where(np.isfinite(qr_vals), qr_vals, -np.inf)
        seg_of = np.searchsorted(offsets, cand, side="right") - 1
        np.maximum.at(refined[e_i], seg_of, qr_vals)

    return np.maximum(_segment_max(kept, offsets), refined)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def bsadf_sequence(
    y: np.ndarray,
    *,
    min_window: int,
    lag: int = 0,
    max_window: int | None = None,
    grid: int | None = 32,
    refine_top_k: int = 0,
    trend: str = "c",
    block: int = 1000,
    chunk_windows: int = _CHUNK_WINDOWS,
) -> np.ndarray:
    """Backward sup-ADF statistic at every endpoint of a single series.

    Parameters
    ----------
    y : numpy.ndarray, shape (T,)
        Series in time order. Cast to float64; ``nan`` propagates.
    min_window : int
        Shortest backward window, ``r_0 * T`` in the Phillips-Shi-Yu notation.
        Pass an **absolute** number of observations, never a fraction of the
        sample: a window that grows with ``len(y)`` would break prefix
        invariance. Raised to :func:`min_admissible_window` if smaller.
    lag : int, default=0
        Augmenting lagged differences ``p`` in the ADF regression.
    max_window : int, optional
        Longest backward window. ``None`` lets the window expand to the whole
        history available at each endpoint (the usual PSY choice).
    grid : int or None, default=32
        ``None`` evaluates every admissible start (exhaustive supremum). An
        integer evaluates a geometric ladder of that many window lengths, which
        turns the ``O(T^2)`` sweep into ``O(T * grid)``.
    refine_top_k : int, default=0
        Recompute this many best candidates per endpoint by QR on the raw
        observations and report the QR value. ``0`` disables refinement (but
        numerically suspect windows are then simply dropped as ``nan``).
    trend : {"c", "ct"}, default="c"
        Deterministic terms in the ADF regression. ``"c"`` is the PSY
        specification.
    block : int, default=1000
        Accumulator block length; see :func:`~polars_features.detect._moments.cumulative_moments`.
    chunk_windows : int, default=131072
        Approximate number of windows evaluated per batch. Lower it to cap peak
        memory, raise it for a little more speed.

    Returns
    -------
    numpy.ndarray, shape (T,)
        float64. ``nan`` at every endpoint with no admissible window (that is,
        for the first ``min_window - 1`` rows).

    Raises
    ------
    ValueError
        If ``y`` is not 1-D, or is too short for the requested lag order.

    Examples
    --------
    >>> import numpy as np
    >>> from polars_features.detect._bsadf import bsadf_sequence
    >>> rng = np.random.default_rng(0)
    >>> y = np.cumsum(rng.standard_normal(400))
    >>> s = bsadf_sequence(y, min_window=40, grid=None)
    >>> s.shape
    (400,)
    >>> bool(np.all(np.isnan(s[:39])))
    True

    Notes
    -----
    Exhaustive ``T = 1000``, ``lag = 0`` costs ``0.012 s`` here -- 452_676
    windows, ``38`` million windows a second -- against ``0.78 s`` for a
    compiled per-window reference. ``lag = 1`` is ``0.049 s``, ``lag = 3`` is
    ``0.24 s``, and the default ``grid = 32`` ladder is ``0.0012 s``.
    """
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"`y` must be 1-D, got shape {arr.shape}.")
    mom = cumulative_moments(arr[None, :], lag=lag, block=block, trend=trend)
    out = _bsadf_core(
        mom,
        min_window=min_window,
        max_window=max_window,
        grid=grid,
        refine_top_k=refine_top_k,
        chunk_windows=chunk_windows,
    )
    return out[0]


def bsadf_panel(
    y: np.ndarray,
    *,
    min_window: int,
    lag: int = 0,
    max_window: int | None = None,
    grid: int | None = 32,
    refine_top_k: int = 0,
    trend: str = "c",
    block: int = 1000,
    chunk_windows: int = _CHUNK_WINDOWS,
) -> np.ndarray:
    """BSADF sequences for a whole panel, vectorised over entities.

    Parameters
    ----------
    y : numpy.ndarray, shape (N, T)
        One series per row, all on a common, aligned time axis.
    min_window, lag, max_window, grid, refine_top_k, trend, block, chunk_windows
        As :func:`bsadf_sequence`.

    Returns
    -------
    numpy.ndarray, shape (N, T)

    Notes
    -----
    The cumulative sums are taken along ``axis=1`` and every window gather,
    Cholesky and solve runs once for the whole ``(N, K)`` batch, so the entity
    dimension costs no extra Python. ``chunk_windows`` is divided by ``N`` so
    peak memory is independent of the panel width. ``refine_top_k`` is the one
    part that loops per entity, since QR is a per-series operation.

    Entities are treated independently -- this is the input to a breadth or
    cross-sectional-rank aggregation, not a pooled test.
    """
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"`y` must be 2-D (N, T), got shape {arr.shape}.")
    mom = cumulative_moments(arr, lag=lag, block=block, trend=trend)
    return _bsadf_core(
        mom,
        min_window=min_window,
        max_window=max_window,
        grid=grid,
        refine_top_k=refine_top_k,
        chunk_windows=chunk_windows,
    )


def gsadf(y: np.ndarray, *, min_window: int, lag: int = 0, **kwargs) -> float:
    """Generalised sup-ADF statistic: the maximum of the BSADF sequence.

    Parameters
    ----------
    y : numpy.ndarray, shape (T,)
        Series in time order.
    min_window : int
        Shortest backward window, in observations.
    lag : int, default=0
        Augmenting lags.
    **kwargs
        Forwarded to :func:`bsadf_sequence`.

    Returns
    -------
    float
        ``nan`` if no admissible window exists.
    """
    seq = bsadf_sequence(y, min_window=min_window, lag=lag, **kwargs)
    finite = seq[np.isfinite(seq)]
    return float(finite.max()) if finite.size else float("nan")
