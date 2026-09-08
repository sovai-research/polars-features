"""Prefix-sum sufficient statistics for recursive right-tailed unit-root tests.

Recursive bubble detection (Phillips, Shi & Yu 2015) evaluates the augmented
Dickey-Fuller regression

.. math::

    \\Delta y_t = \\alpha + \\beta\\, y_{t-1}
                 + \\sum_{i=1}^{p} \\psi_i\\, \\Delta y_{t-i} + \\varepsilon_t

on *every* backward-expanding window ``[t1, t2]`` and takes the supremum of the
right-tailed ``t``-statistic on ``beta``. Done naively that is
``O(T^2)`` regressions, each ``O(n k^2)`` -- hopeless past a few hundred rows.

The algebra that makes it cheap
-------------------------------
Stack the regressors as ``z_t = (1, y_{t-1}, dy_{t-1}, ..., dy_{t-p})`` with
``k = p + 2`` and write ``d_t = dy_t``. The normal equations for the window
``[t1, t2]`` depend on the data **only** through three sums

``A = sum z z'`` (``k x k``),  ``b = sum z d`` (``k``),  ``s = sum d^2``,

each of which is a *window* sum and therefore a difference of two cumulative
sums. Build the cumulative sums once in ``O(T k^2)`` and every window costs
``O(1)`` gathers plus an ``O(k^3)`` solve on a tiny fixed-size system.

The second trick removes the residual pass. Because the least-squares solution
satisfies ``A beta_hat = b``,

.. math::

    SSR = s - 2\\,\\hat\\beta'b + \\hat\\beta'A\\hat\\beta = s - \\hat\\beta'b ,

so the residual sum of squares is available from the same three sums -- no
second pass over the observations. With the Cholesky factor ``A = L L'`` and the
forward solve ``L z = b`` this is even simpler: ``beta_hat'b = z'z``. A second
forward solve against ``e_2`` gives ``(A^{-1})_{22}``, and

``sigma2_hat = SSR / (n - k)``,  ``t = beta_hat_2 / sqrt(sigma2_hat (A^{-1})_{22})``.

Numerical care
--------------
Normal equations inherit ``cond(X)^2``, and long windows of a near-unit-root
series in levels are badly conditioned. Three defences are built in:

* **Anchoring.** The series is shifted to ``y - y[0]`` before anything is
  accumulated. The intercept absorbs the shift exactly, so the statistic is
  unchanged, but the accumulated magnitudes are of the order of the *variation*
  of the series rather than its level. Unlike mean-centring, ``y[0]`` is known
  at every point in time, so anchoring is exactly point-in-time.
* **Block resets.** The accumulator is reset every ``block`` rows and a short
  per-block ladder is kept alongside it, so a window sum never cancels two
  numbers larger than one block's worth of mass, however long the sample.
  Measured on a random walk at ``T = 500_000``, the window statistic lands
  ``1.6e-12`` from an extended-precision reference with the reset on and
  ``8.3e-8`` from it with a plain unblocked ``cumsum`` -- five orders of
  magnitude, for about ``+15%`` build cost.
* **Escalation.** Windows whose ``SSR`` goes negative, or whose Gram matrix is
  not numerically positive definite, are flagged so the caller can recompute
  them from the raw observations by QR (:func:`window_adf_qr`), which is
  backward stable in ``cond(X)`` rather than ``cond(X)^2``. At
  ``cond(X) = 1.7e8`` the normal equations returned ``+1.000e+00`` for a
  coefficient whose true value is zero; QR returned ``+2.8e-09``.

If a linear trend is requested it is normalised as ``(t - t1) / n`` *within the
window*, never as a raw time index: a raw trend column would make the design
depend on where the window sits in the sample and would blow the conditioning
up like ``T^2``. The raw ``sum t z`` / ``sum t^2`` / ``sum t d`` moments are
accumulated and the window-local normalisation is applied at assembly time.

References
----------
Phillips, P. C. B., Shi, S. & Yu, J. (2015). "Testing for Multiple Bubbles:
Historical Episodes of Exuberance and Collapse in the S&P 500."
*International Economic Review* 56(4), 1043-1078. DOI 10.1111/iere.12132.

This is a clean-room implementation written from the published equations.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "Moments",
    "cumulative_moments",
    "window_adf",
    "window_adf_full",
    "window_adf_qr",
    "cholesky_batch",
]

#: Relative floor below which ``SSR = s - beta'b`` is treated as destroyed by
#: cancellation and the window is escalated to the QR path.
_SSR_TOL = 1e-10


class Moments(NamedTuple):
    """Blocked cumulative sufficient statistics for an ADF regression.

    Opaque; build with :func:`cumulative_moments` and consume with
    :func:`window_adf`. The moment vector stored per regression row is the
    concatenation of the upper triangle of ``z z'`` (``kb (kb + 1) / 2``
    entries, row-major over :func:`numpy.triu_indices`), the cross-products
    ``z d`` (``kb`` entries) and ``d ** 2`` (one entry); when ``trend == "ct"``
    the raw trend moments ``r z`` (``kb``), ``r ** 2`` and ``r d`` follow.

    Attributes
    ----------
    cin : numpy.ndarray, shape (N, nb, block + 1, m)
        Exclusive prefix sums *within* each block: ``cin[:, q, u]`` is the sum
        of rows ``q * block ... q * block + u - 1``.
    bfull : numpy.ndarray, shape (N, nb, m)
        Per-block totals.
    bc : numpy.ndarray, shape (N, nb + 1, m)
        Exclusive cumulative sums of ``bfull`` over blocks.
    block : int
        Rows per accumulator block.
    n_rows : int
        Number of usable regression rows ``R = T - 1 - lag``.
    n_obs : int
        Length ``T`` of the input series.
    lag : int
        Number of augmenting lags ``p``.
    kb : int
        Size of the base regressor stack, ``p + 2``.
    k : int
        Total number of regressors (``kb``, or ``kb + 1`` when a trend is on).
    trend : str
        ``"c"`` or ``"ct"``.
    m_gram : int
        Number of stored Gram entries, ``kb (kb + 1) / 2``.
    iu : tuple of numpy.ndarray
        ``numpy.triu_indices(kb)``, the (row, col) decoding of the Gram block.
    y : numpy.ndarray, shape (N, T)
        The anchored series ``y - y[:, :1]``, kept for QR escalation.
    y0 : numpy.ndarray, shape (N, 1)
        The anchor that was removed.
    batched : bool
        ``True`` when the caller passed a 2-D ``(N, T)`` panel.
    """

    cin: np.ndarray
    bfull: np.ndarray
    bc: np.ndarray
    block: int
    n_rows: int
    n_obs: int
    lag: int
    kb: int
    k: int
    trend: str
    m_gram: int
    iu: tuple[np.ndarray, np.ndarray]
    y: np.ndarray
    y0: np.ndarray
    batched: bool

    @property
    def min_window(self) -> int:
        """Shortest window with strictly positive residual degrees of freedom."""
        return self.lag + self.k + 2


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def cumulative_moments(
    y: np.ndarray,
    *,
    lag: int = 0,
    block: int = 1000,
    trend: str = "c",
) -> Moments:
    """Accumulate the blocked prefix sums that summarise every ADF window.

    Parameters
    ----------
    y : numpy.ndarray
        Series in time order, shape ``(T,)``, or a panel of shape ``(N, T)``
        whose rows are accumulated independently along ``axis=1``.
    lag : int, default=0
        Number of augmenting lagged differences ``p``.
    block : int, default=1000
        Accumulator block length. The accumulator is reset every ``block``
        rows, which bounds the cancellation in a window sum by one block's mass
        instead of the whole sample's.
    trend : {"c", "ct"}, default="c"
        Deterministic terms. ``"c"`` is the Phillips-Shi-Yu specification (an
        intercept only); ``"ct"`` adds a window-normalised linear trend.

    Returns
    -------
    Moments
        Opaque struct consumed by :func:`window_adf`.

    Raises
    ------
    ValueError
        If the series is too short for the requested lag order, or the
        arguments are out of range.

    Notes
    -----
    Cost is ``O(N T k^2)`` time and memory. The result is *prefix invariant*:
    because the blocks are anchored at row 0 and the anchor is ``y[0]``,
    accumulating ``y[:T]`` and ``y[:T + h]`` produces bitwise identical entries
    for every row ``< T``. ``block`` must therefore be a fixed constant and
    never a fraction of the sample.

    Shrinking ``block`` buys accuracy on the hardest case -- a *short* window
    sitting at a level far from ``y[0]``, where ``sum x^2`` over the window is a
    tiny difference of two large prefixes. On a drifting random walk whose worst
    window had ``mean(x) / sd(x) = 58``, the error against an extended-precision
    reference fell from ``1.8e-11`` at ``block = 1000`` to ``2.0e-12`` at
    ``block = 25``, at a proportional cost in the ladder. The default is a good
    trade for series that are not dominated by drift; when they are, prefer
    ``refine_top_k`` on the caller side, which fixes the reported extremum
    exactly rather than everywhere approximately.
    """
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
        batched = False
    elif arr.ndim == 2:
        batched = True
    else:
        raise ValueError(f"`y` must be 1-D or 2-D, got {arr.ndim} dimensions.")
    n_entities, n_obs = arr.shape

    lag = int(lag)
    block = int(block)
    if lag < 0:
        raise ValueError(f"`lag` must be >= 0, got {lag!r}.")
    if block < 1:
        raise ValueError(f"`block` must be >= 1, got {block!r}.")
    if trend not in ("c", "ct"):
        raise ValueError(f"`trend` must be 'c' or 'ct', got {trend!r}.")

    kb = lag + 2
    k = kb + 1 if trend == "ct" else kb
    n_rows = n_obs - 1 - lag
    if n_rows <= k:
        raise ValueError(
            f"series of length {n_obs} is too short for lag={lag}, trend={trend!r}: "
            f"it yields {max(n_rows, 0)} regression rows for {k} regressors."
        )

    # Anchor: point-in-time (uses only y[0]) and absorbed exactly by the
    # intercept, but keeps every accumulated magnitude O(variation).
    y0 = arr[:, :1].copy()
    ya = arr - y0
    dy = np.diff(ya, axis=1)

    stack = np.empty((n_entities, n_rows, kb), dtype=np.float64)
    stack[:, :, 0] = 1.0
    stack[:, :, 1] = ya[:, lag : lag + n_rows]
    for i in range(1, lag + 1):
        stack[:, :, 1 + i] = dy[:, lag - i : lag - i + n_rows]
    target = dy[:, lag : lag + n_rows]

    iu = np.triu_indices(kb)
    m_gram = int(iu[0].size)
    m = m_gram + kb + 1
    if trend == "ct":
        m += kb + 2

    moments = np.empty((n_entities, n_rows, m), dtype=np.float64)
    moments[:, :, :m_gram] = stack[:, :, iu[0]] * stack[:, :, iu[1]]
    moments[:, :, m_gram : m_gram + kb] = stack * target[:, :, None]
    moments[:, :, m_gram + kb] = target * target
    if trend == "ct":
        rr = np.arange(n_rows, dtype=np.float64)
        off = m_gram + kb + 1
        moments[:, :, off : off + kb] = stack * rr[None, :, None]
        moments[:, :, off + kb] = rr * rr
        moments[:, :, off + kb + 1] = rr * target
    del stack

    cin, bfull, bc = _blocked_prefix(moments, block)
    del moments

    return Moments(
        cin=cin,
        bfull=bfull,
        bc=bc,
        block=block,
        n_rows=n_rows,
        n_obs=n_obs,
        lag=lag,
        kb=kb,
        k=k,
        trend=trend,
        m_gram=m_gram,
        iu=iu,
        y=ya,
        y0=y0,
        batched=batched,
    )


def _blocked_prefix(
    values: np.ndarray, block: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-level (within-block, across-block) exclusive prefix sums.

    Parameters
    ----------
    values : numpy.ndarray, shape (N, R, m)
        Per-row moment vectors.
    block : int
        Block length.

    Returns
    -------
    cin : numpy.ndarray, shape (N, nb, block + 1, m)
    bfull : numpy.ndarray, shape (N, nb, m)
    bc : numpy.ndarray, shape (N, nb + 1, m)
    """
    n_entities, n_rows, m = values.shape
    n_blocks = max(1, -(-n_rows // block))
    # Physical block width. When the whole sample fits in one block there is
    # nothing to pad -- important, because `block` may legitimately be set far
    # larger than the sample to disable the reset. When there are several
    # blocks the padding is strictly less than one block.
    width = block if n_blocks > 1 else n_rows
    pad = n_blocks * width - n_rows
    if pad:
        padded = np.zeros((n_entities, n_rows + pad, m), dtype=np.float64)
        padded[:, :n_rows, :] = values
    else:
        padded = values
    padded = padded.reshape(n_entities, n_blocks, width, m)
    # One allocation, one pass: the leading zero column makes `cin` an
    # *exclusive* prefix, and its last column is exactly the block total.
    cin = np.empty((n_entities, n_blocks, width + 1, m), dtype=np.float64)
    cin[:, :, 0, :] = 0.0
    np.cumsum(padded, axis=2, out=cin[:, :, 1:, :])
    bfull = np.ascontiguousarray(cin[:, :, width, :])
    bc = np.zeros((n_entities, n_blocks + 1, m), dtype=np.float64)
    np.cumsum(bfull, axis=1, out=bc[:, 1:, :])
    return cin, bfull, bc


def _window_sums(mom: Moments, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Sum the moment vectors over regression rows ``[lo, hi)``, block-aware.

    Within a single block the answer is the plain difference of two small
    within-block prefixes. Across blocks it is assembled as
    ``(tail of the first block) + (whole intervening blocks) + (head of the
    last block)``, so no subtraction ever involves more than one block's mass
    plus the short block ladder.

    Both shortcuts below return exactly the ``within`` branch of the general
    expression, bit for bit, so they change the cost and nothing else.
    """
    n_entities, n_blocks, stride, m = mom.cin.shape
    flat = mom.cin.reshape(n_entities, n_blocks * stride, m)
    if n_blocks == 1:
        return flat[:, hi, :] - flat[:, lo, :]

    blk = mom.block
    qlo = np.minimum(lo // blk, n_blocks - 1)
    ulo = lo - qlo * blk
    qhi = np.minimum(hi // blk, n_blocks - 1)
    uhi = hi - qhi * blk

    # One linear gather per endpoint beats a two-axis fancy index.
    head = flat[:, qlo * stride + ulo, :]
    tail = flat[:, qhi * stride + uhi, :]
    within = tail - head
    same = qlo == qhi
    if bool(same.all()):
        return within
    across = (
        (mom.bfull[:, qlo, :] - head)
        + (mom.bc[:, qhi, :] - mom.bc[:, qlo + 1, :])
        + tail
    )
    return np.where(same[None, :, None], within, across)


# --------------------------------------------------------------------------- #
# Batched dense linear algebra, written as k^3 elementwise passes
# --------------------------------------------------------------------------- #
def cholesky_batch(a: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of a batch of tiny symmetric matrices.

    Parameters
    ----------
    a : numpy.ndarray, shape (..., k, k)
        Symmetric matrices; only the lower triangle is read.

    Returns
    -------
    numpy.ndarray, shape (..., k, k)
        Lower-triangular ``L`` with ``L L' = a``. Batch members that are not
        numerically positive definite get ``nan`` on the failing diagonal, which
        propagates to the statistic and flags the window for QR escalation.

    Notes
    -----
    The loops run over ``k`` (a small compile-time-ish constant, 2-6 here) and
    every statement is a full-batch elementwise NumPy op, so the whole batch is
    factorised in ``k^3 / 6`` vectorised passes. This is deliberately *not*
    :func:`numpy.linalg.cholesky`: LAPACK's batched path pays per-matrix call
    overhead that dominates at this size -- measured 17x slower than this
    formulation for ``T = 5000``, ``p = 1``.
    """
    k = a.shape[-1]
    lower = np.zeros_like(a)
    for j in range(k):
        pivot = a[..., j, j].copy()
        for q in range(j):
            pivot -= lower[..., j, q] ** 2
        pivot = np.where(pivot > 0.0, pivot, np.nan)
        diag = np.sqrt(pivot)
        lower[..., j, j] = diag
        for i in range(j + 1, k):
            acc = a[..., i, j].copy()
            for q in range(j):
                acc -= lower[..., i, q] * lower[..., j, q]
            lower[..., i, j] = acc / diag
    return lower


def _forward_solve(lower: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``L x = rhs`` for lower-triangular ``L``, batched and elementwise."""
    k = lower.shape[-1]
    out = np.empty_like(rhs)
    for i in range(k):
        acc = rhs[..., i].copy()
        for q in range(i):
            acc -= lower[..., i, q] * out[..., q]
        out[..., i] = acc / lower[..., i, i]
    return out


def _back_solve(lower: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``L' x = rhs`` for lower-triangular ``L``, batched and elementwise."""
    k = lower.shape[-1]
    out = np.empty_like(rhs)
    for i in range(k - 1, -1, -1):
        acc = rhs[..., i].copy()
        for q in range(i + 1, k):
            acc -= lower[..., q, i] * out[..., q]
        out[..., i] = acc / lower[..., i, i]
    return out


# --------------------------------------------------------------------------- #
# Window statistics
# --------------------------------------------------------------------------- #
def window_adf(mom: Moments, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Right-tailed ADF ``t``-statistic for every ``(start, end)`` window.

    Parameters
    ----------
    mom : Moments
        Built by :func:`cumulative_moments`.
    starts, ends : numpy.ndarray
        Integer arrays of equal length giving **inclusive** window bounds in the
        index space of the original series, so the window is
        ``y[start], ..., y[end]`` and its length is ``end - start + 1``.

    Returns
    -------
    numpy.ndarray
        Shape ``(K,)`` for a 1-D input series, ``(N, K)`` for a panel. Windows
        with non-positive residual degrees of freedom, a singular Gram matrix or
        a numerically destroyed ``SSR`` come back as ``nan``.

    Notes
    -----
    Each window costs a handful of gathers plus an ``O(k^3)`` solve, independent
    of the window length.
    """
    stat, _ = window_adf_full(mom, starts, ends)
    return stat


def window_adf_full(
    mom: Moments, starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`window_adf` plus a boolean "recompute me by QR" flag per window.

    Returns
    -------
    stat : numpy.ndarray
        The ``t``-statistics, shape ``(K,)`` or ``(N, K)``.
    escalate : numpy.ndarray of bool
        Same shape. ``True`` where the statistic is non-finite, the Gram matrix
        failed to factorise, or ``SSR = s - beta'b`` fell at or below
        ``1e-10 * s`` -- i.e. where the normal equations lost the answer to
        cancellation and only a QR on the raw observations can recover it.
    """
    lo = np.asarray(starts, dtype=np.int64).ravel()
    hi = np.asarray(ends, dtype=np.int64).ravel() - mom.lag
    if lo.shape != hi.shape:
        raise ValueError("`starts` and `ends` must have the same length.")
    if lo.size and (lo.min() < 0 or hi.max() > mom.n_rows):
        raise ValueError(
            "window bounds fall outside the accumulated sample; "
            f"need 0 <= start and end <= {mom.n_obs - 1}."
        )
    nobs = (hi - lo).astype(np.float64)
    gram = _window_sums(mom, lo, hi)

    if mom.lag == 0 and mom.trend == "c":
        stat, escalate = _tstat_closed_form(gram, nobs)
    else:
        stat, escalate = _tstat_cholesky(mom, gram, nobs, lo.astype(np.float64))

    if not mom.batched:
        return stat[0], escalate[0]
    return stat, escalate


def _tstat_closed_form(
    gram: np.ndarray, nobs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``lag == 0`` ADF ``t``-statistic from five prefix sums, fully elementwise.

    With ``x = y_{t-1}``, ``d = dy_t`` and the centred sums
    ``Sxx_c = Sxx - Sx^2/n``, ``Sxd_c = Sxd - Sx Sd/n``,
    ``Sdd_c = Sdd - Sd^2/n``,

    ``beta = Sxd_c / Sxx_c``, ``SSR = Sdd_c - beta Sxd_c``,
    ``t = beta / sqrt(SSR / (n - 2) / Sxx_c)``.

    No linear algebra, no Python loop: this runs over the whole
    (end x start) batch at once.
    """
    # Channel layout for kb == 2: triu(2) is (0,0), (0,1), (1,1) then the two
    # cross-products then sum d^2.
    s_x = gram[..., 1]
    s_xx = gram[..., 2]
    s_d = gram[..., 3]
    s_xd = gram[..., 4]
    s_dd = gram[..., 5]

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_n = 1.0 / nobs
        sxx_c = s_xx - s_x * s_x * inv_n
        sxd_c = s_xd - s_x * s_d * inv_n
        sdd_c = s_dd - s_d * s_d * inv_n
        beta = sxd_c / sxx_c
        ssr = sdd_c - beta * sxd_c
        sigma2 = ssr / (nobs - 2.0)
        stat = beta / np.sqrt(sigma2 / sxx_c)

    valid = (nobs > 2.0) & (sxx_c > 0.0) & (ssr > _SSR_TOL * np.abs(s_dd))
    stat = np.where(valid, stat, np.nan)
    escalate = ~valid | ~np.isfinite(stat)
    return stat, escalate


def _tstat_cholesky(
    mom: Moments, gram: np.ndarray, nobs: np.ndarray, row_lo: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """General ``lag >= 0`` path: assemble ``A``, ``b``, ``s`` and solve.

    ``SSR = s - beta'b`` comes free from the forward solve because
    ``beta'b = b'A^{-1}b = z'z`` with ``L z = b``; a second forward solve
    against ``e_2`` gives ``(A^{-1})_{22}``.
    """
    kb, k = mom.kb, mom.k
    m_gram = mom.m_gram
    rows, cols = mom.iu
    batch = gram.shape[:-1]

    amat = np.zeros(batch + (k, k), dtype=np.float64)
    tri = gram[..., :m_gram]
    amat[..., rows, cols] = tri
    amat[..., cols, rows] = tri
    bvec = np.zeros(batch + (k,), dtype=np.float64)
    bvec[..., :kb] = gram[..., m_gram : m_gram + kb]
    s_dd = gram[..., m_gram + kb]

    if mom.trend == "ct":
        # The trend column is (r - r_lo) / n *inside the window*; recover its
        # moments from the raw ones. sum z_a is the (0, a) Gram entry.
        off = m_gram + kb + 1
        sum_rz = gram[..., off : off + kb]
        sum_rr = gram[..., off + kb]
        sum_rd = gram[..., off + kb + 1]
        sum_z = gram[..., :kb]
        lo = row_lo
        with np.errstate(divide="ignore", invalid="ignore"):
            trend_z = (sum_rz - lo[..., None] * sum_z) / nobs[..., None]
            trend_tt = (sum_rr - 2.0 * lo * sum_rz[..., 0] + lo * lo * nobs) / (
                nobs * nobs
            )
            trend_d = (sum_rd - lo * bvec[..., 0]) / nobs
        amat[..., k - 1, :kb] = trend_z
        amat[..., :kb, k - 1] = trend_z
        amat[..., k - 1, k - 1] = trend_tt
        bvec[..., k - 1] = trend_d

    with np.errstate(divide="ignore", invalid="ignore"):
        lower = cholesky_batch(amat)
        zsol = _forward_solve(lower, bvec)
        ssr = s_dd - np.sum(zsol * zsol, axis=-1)
        beta = _back_solve(lower, zsol)
        unit = np.zeros_like(bvec)
        unit[..., 1] = 1.0
        vsol = _forward_solve(lower, unit)
        ainv22 = np.sum(vsol * vsol, axis=-1)
        sigma2 = ssr / (nobs - float(k))
        stat = beta[..., 1] / np.sqrt(sigma2 * ainv22)

    valid = (
        (nobs > float(k))
        & (ssr > _SSR_TOL * np.abs(s_dd))
        & np.isfinite(ainv22)
        & (ainv22 > 0.0)
    )
    stat = np.where(valid, stat, np.nan)
    escalate = ~valid | ~np.isfinite(stat)
    return stat, escalate


# --------------------------------------------------------------------------- #
# QR reference / escalation path
# --------------------------------------------------------------------------- #
def window_adf_qr(
    y: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    lag: int = 0,
    trend: str = "c",
) -> np.ndarray:
    """ADF ``t``-statistic recomputed from raw observations by QR.

    Parameters
    ----------
    y : numpy.ndarray, shape (T,)
        The series (anchoring is optional; the intercept absorbs it).
    starts, ends : numpy.ndarray
        Inclusive window bounds in the index space of ``y``.
    lag : int, default=0
        Augmenting lags.
    trend : {"c", "ct"}, default="c"
        Deterministic terms.

    Returns
    -------
    numpy.ndarray, shape (K,)
        ``nan`` where the window is too short or the design is rank deficient.

    Notes
    -----
    ``O(n k^2)`` per window -- far more expensive than :func:`window_adf`, so
    this is for the handful of top candidates and for windows the prefix pass
    flags as numerically suspect. The payoff is backward stability in
    ``cond(X)`` rather than ``cond(X)^2``: at ``cond(X) = 1.7e8`` the normal
    equations returned ``+1.000e+00`` for a coefficient whose QR value is
    ``+2.8e-09``. On a strongly drifting series the prefix pass drifted
    ``5.9e-09`` from this path; one QR candidate per endpoint closed that to
    ``6.0e-12``.
    """
    arr = np.asarray(y, dtype=np.float64).ravel()
    lo = np.asarray(starts, dtype=np.int64).ravel()
    hi = np.asarray(ends, dtype=np.int64).ravel()
    if lo.shape != hi.shape:
        raise ValueError("`starts` and `ends` must have the same length.")
    if trend not in ("c", "ct"):
        raise ValueError(f"`trend` must be 'c' or 'ct', got {trend!r}.")
    lag = int(lag)
    kb = lag + 2
    k = kb + 1 if trend == "ct" else kb

    diffs = np.diff(arr)
    n_rows = arr.size - 1 - lag
    out = np.full(lo.size, np.nan, dtype=np.float64)
    eye = np.eye(k, dtype=np.float64)
    for w in range(lo.size):
        r_lo = int(lo[w])
        r_hi = int(hi[w]) - lag
        nobs = r_hi - r_lo
        if r_lo < 0 or r_hi > n_rows or nobs <= k:
            continue
        design = np.empty((nobs, k), dtype=np.float64)
        design[:, 0] = 1.0
        design[:, 1] = arr[r_lo + lag : r_hi + lag]
        for i in range(1, lag + 1):
            design[:, 1 + i] = diffs[r_lo + lag - i : r_hi + lag - i]
        if trend == "ct":
            design[:, k - 1] = np.arange(nobs, dtype=np.float64) / nobs
        target = diffs[r_lo + lag : r_hi + lag]

        qmat, rmat = np.linalg.qr(design)
        if not np.all(np.isfinite(rmat)) or np.min(np.abs(np.diag(rmat))) <= 0.0:
            continue
        beta = np.linalg.solve(rmat, (qmat.T @ target)[..., None])[..., 0]
        resid = target - design @ beta
        ssr = float(resid @ resid)
        if not np.isfinite(ssr) or ssr <= 0.0:
            continue
        # (X'X)^-1 = R^-1 R^-T, so its (2, 2) entry is the squared norm of the
        # second row of R^-1 -- no cross-product matrix is ever formed.
        rinv_row = np.linalg.solve(rmat, eye)[1, :]
        var = (ssr / (nobs - k)) * float(rinv_row @ rinv_row)
        if var <= 0.0:
            continue
        out[w] = float(beta[1]) / np.sqrt(var)
    return out
