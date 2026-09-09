"""Low-level numerical helpers shared by the catch22 feature functions.

Private to :mod:`panelary.catch22`; nothing here is part of the public
surface.  Kept dependency-free (numpy only) so the feature functions stay
importable on the bare ``numpy + polars`` core.
"""

from __future__ import annotations

import numpy as np


def _bspline_design(x: np.ndarray, knots: np.ndarray, k: int) -> np.ndarray:
    """B-spline design matrix by the Cox-de Boor recursion (pure NumPy).

    Returns an ``(x.size, len(knots) - k - 1)`` matrix whose column ``j`` is the
    degree-``k`` B-spline basis function ``B_{j,k}`` evaluated at ``x``.

    The degree-0 step uses half-open intervals ``[t_i, t_{i+1})`` so the basis
    forms a partition of unity, with one exception: the last *non-degenerate*
    interval is closed on the right, so ``x == knots[-1]`` is covered.  With a
    clamped knot vector the final intervals are zero-width repeats, so closing
    the last interval by index would leave the right endpoint evaluating to all
    zeros -- which is exactly the off-by-one that makes an otherwise-correct
    implementation disagree with SciPy by 1.0 in the final row.
    """
    x = np.asarray(x, dtype=np.float64)
    n_knots = knots.size
    spans = [i for i in range(n_knots - 1) if knots[i + 1] > knots[i]]
    basis = np.zeros((x.size, n_knots - 1))
    if not spans:
        return basis[:, : max(n_knots - k - 1, 0)]
    last = spans[-1]
    for i in spans:
        lo, hi = knots[i], knots[i + 1]
        inside = (x >= lo) & (x <= hi) if i == last else (x >= lo) & (x < hi)
        basis[inside, i] = 1.0

    for degree in range(1, k + 1):
        nxt = np.zeros((x.size, n_knots - degree - 1))
        for i in range(n_knots - degree - 1):
            den_left = knots[i + degree] - knots[i]
            den_right = knots[i + degree + 1] - knots[i + 1]
            col = np.zeros(x.size)
            if den_left > 0:
                col += (x - knots[i]) / den_left * basis[:, i]
            if den_right > 0:
                col += (knots[i + degree + 1] - x) / den_right * basis[:, i + 1]
            nxt[:, i] = col
        basis = nxt
    return basis


def _lsq_spline_fit(
    t: np.ndarray, y: np.ndarray, interior: np.ndarray, k: int = 3
) -> np.ndarray:
    """Least-squares spline of degree ``k`` with fixed interior knots, evaluated on ``t``.

    A NumPy reimplementation of ``scipy.interpolate.LSQUnivariateSpline(t, y,
    interior, k)(t)``: the same clamped knot vector, the same B-spline basis and
    the same least-squares problem, solved with :func:`numpy.linalg.lstsq`.
    Agrees with SciPy to ~1e-14 over n = 8..2048 on noise, random walks,
    seasonal and power-law series.

    This exists so :func:`PD_PeriodicityWang_th0_01` detrends identically with
    and without SciPy installed.  It previously fell back to a *zero* spline,
    which silently returned a different number on the bare numpy+polars core.
    """
    knots = np.concatenate(
        [
            np.repeat(t[0], k + 1),
            np.asarray(interior, dtype=np.float64),
            np.repeat(t[-1], k + 1),
        ]
    )
    design = _bspline_design(t, knots, k)
    if design.shape[1] == 0:
        return np.zeros_like(t)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return design @ coef


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _as_1d(x) -> np.ndarray:
    """Coerce input to a contiguous 1-D float64 array with NaNs dropped."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size and np.isnan(arr).any():
        arr = arr[~np.isnan(arr)]
    return arr


def _zscore(x: np.ndarray) -> np.ndarray:
    """Z-score a series using the sample standard deviation (``ddof=1``).

    Returns an all-zero array if the input has (near-)zero spread, matching the
    degenerate-input behaviour used throughout catch22.  Z-scoring is
    idempotent, so applying it inside every feature keeps the feature functions
    self-contained.
    """
    x = _as_1d(x)
    if x.size == 0:
        return x
    mean = x.mean()
    std = x.std(ddof=1) if x.size > 1 else 0.0
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(x)
    return (x - mean) / std


def _acf(y: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation function via FFT (Wiener-Khinchin).

    Returns ``acf`` with ``acf[0] == 1`` and ``acf[k]`` the biased
    autocorrelation estimate at lag ``k`` for ``k = 0 .. n-1``.
    """
    n = y.size
    y = y - y.mean()
    if n == 0:
        return np.array([1.0])
    nfft = int(2 ** np.ceil(np.log2(2 * n - 1))) if n > 1 else 1
    f = np.fft.fft(y, n=nfft)
    acov = np.fft.ifft(f * np.conjugate(f))[:n].real
    if acov[0] == 0:
        out = np.zeros(n)
        out[0] = 1.0
        return out
    return acov / acov[0]


def _first_zero_ac(y: np.ndarray) -> int:
    """First lag ``tau >= 1`` at which the ACF crosses (<= 0). Falls back to n."""
    acf = _acf(y)
    n = acf.size
    # Vectorised equivalent of `for tau in range(1, n): if acf[tau] <= 0: ...`
    crossings = np.flatnonzero(acf[1:] <= 0.0)
    return int(crossings[0]) + 1 if crossings.size else n


def _histcounts(y: np.ndarray, n_bins: int):
    """Equal-width histogram over ``[min, max]``; the maximum falls in the last
    bin (numpy's default). Returns ``(counts, edges)``."""
    counts, edges = np.histogram(y, bins=n_bins)
    return counts.astype(np.float64), edges


def _coarsegrain_quantile(y: np.ndarray, n_groups: int) -> np.ndarray:
    """Symbolise a series into ``n_groups`` equiprobable (quantile) groups.

    Returns integer labels in ``0 .. n_groups - 1``.
    """
    qs = np.quantile(y, np.arange(1, n_groups) / n_groups, method="linear")
    # np.searchsorted maps values below the first threshold -> 0, etc.
    labels = np.searchsorted(qs, y, side="right")
    return np.clip(labels, 0, n_groups - 1).astype(int)


def _pair_counts(symbols: np.ndarray, n_states: int) -> np.ndarray:
    """Counts of consecutive symbol pairs as an ``(n_states, n_states)`` matrix.

    ``out[a, b]`` is the number of positions ``i`` with ``symbols[i] == a`` and
    ``symbols[i + 1] == b``.  Uses :func:`numpy.bincount` on the flattened pair
    index, replacing an O(n) Python ``zip`` loop with a single C pass; the
    counts are integers, so the result is exact.
    """
    s = np.asarray(symbols, dtype=np.intp)
    if s.size < 2:
        return np.zeros((n_states, n_states), dtype=np.float64)
    flat = s[:-1] * n_states + s[1:]
    counts = np.bincount(flat, minlength=n_states * n_states)
    return counts.reshape(n_states, n_states).astype(np.float64)


def _num_bins_auto(y: np.ndarray) -> int:
    """Scott's normal-reference rule for the number of equal-width bins."""
    n = y.size
    std = y.std(ddof=1) if n > 1 else 0.0
    if std < 1e-3 or n < 2:
        return 0
    span = y.max() - y.min()
    return int(np.ceil(span / (3.5 * std / n ** (1.0 / 3.0))))


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of ``True`` values in a boolean array.

    Run-length encoded with :func:`numpy.diff` on a zero-padded copy, so the
    cost is O(n) in C rather than O(n) in Python (the previous scalar loop
    dominated ``SB_BinaryStats_*`` on long series).
    """
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not m.any():
        return 0
    padded = np.empty(m.size + 2, dtype=np.int8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = m
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int((ends - starts).max())


def _local_simple_residuals(y: np.ndarray, train_len: int) -> np.ndarray:
    """Residuals of a "local simple" mean forecaster.

    Point ``i`` (for ``i >= train_len``) is predicted by the mean of the
    preceding ``train_len`` points; the residual is ``y[i] - prediction``.
    """
    n = y.size
    if n <= train_len:
        return np.array([])
    # Prediction for target i is the mean of the previous `train_len` values.
    # A strided sliding-window view turns the per-target Python loop into one
    # C-level reduction; for the small windows catch22 uses (1 and 3) the
    # summation order is identical to `y[i - w : i].mean()`, so this is a
    # bit-for-bit drop-in (asserted in tests/test_perf_parity_catch22.py).
    windows = np.lib.stride_tricks.sliding_window_view(y[: n - 1], train_len)
    preds = windows.mean(axis=-1)
    return y[train_len:] - preds


def _line_sse(x: np.ndarray, y: np.ndarray) -> float:
    """Sum of squared residuals of an ordinary least-squares line fit."""
    if x.size < 2:
        return 0.0
    coef = np.polyfit(x, y, 1)
    resid = y - np.polyval(coef, x)
    return float(np.dot(resid, resid))
