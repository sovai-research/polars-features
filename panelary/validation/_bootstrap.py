"""Time-series bootstrap resampling schemes (pure NumPy, seeded, deterministic).

The resampling engine underneath cross-validation error bands, the Romano-Wolf
stepdown, Hansen's SPA test and the Model Confidence Set.

Implemented schemes
-------------------
* :func:`moving_block_bootstrap` — Kunsch (1989) non-overlapping-start moving
  blocks of fixed length.
* :func:`circular_block_bootstrap` — Politis & Romano (1992) circular blocks
  (wrap-around, so every observation has equal resampling probability).
* :func:`stationary_bootstrap` — Politis & Romano (1994) geometric block
  lengths; the resampled series is (conditionally) stationary.
* :func:`wild_bootstrap` — Wu (1986) / Liu (1988) multiplier bootstrap for
  heteroskedastic residuals (Rademacher, Mammen or Gaussian multipliers).
* :func:`sieve_bootstrap` — Buhlmann (1997) AR(p) sieve: fit an autoregression,
  IID-resample its residuals, regenerate.

Leak-safety
-----------
Every block scheme accepts ``boundaries``: the sorted start positions of the
disjoint *segments* the sample is made of (typically CV fold boundaries).
**A block never straddles a boundary** — blocks are drawn strictly inside one
segment, so a resample can never splice post-boundary information into a
pre-boundary stretch. This is guardrail #4 of the leak-safety contract.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "block_bootstrap_indices",
    "circular_block_bootstrap",
    "moving_block_bootstrap",
    "resolve_segments",
    "sieve_bootstrap",
    "stationary_bootstrap",
    "wild_bootstrap",
]


# --------------------------------------------------------------------------- #
# Segment handling (fold boundaries)
# --------------------------------------------------------------------------- #
def resolve_segments(
    n_obs: int, boundaries: Sequence[int] | np.ndarray | None
) -> list[tuple[int, int]]:
    """Turn ``boundaries`` into a list of half-open ``[start, stop)`` segments.

    Parameters
    ----------
    n_obs : int
        Length of the sample.
    boundaries : sequence of int, optional
        Positions at which a *new* segment starts (typically the first index of
        each CV fold). ``0`` and ``n_obs`` are implied and may be omitted.
        ``None`` (or an empty sequence) means one segment covering everything.

    Returns
    -------
    list of (int, int)
        Non-empty, disjoint, ordered segments covering ``range(n_obs)``.

    Raises
    ------
    ValueError
        If ``n_obs`` is not positive or a boundary is out of range.
    """
    if n_obs <= 0:
        raise ValueError(f"`n_obs` must be positive, got {n_obs}.")
    if boundaries is None:
        return [(0, n_obs)]
    cuts = sorted({int(b) for b in np.asarray(boundaries, dtype=np.int64).ravel()})
    for b in cuts:
        if not (0 <= b <= n_obs):
            raise ValueError(
                f"boundary {b} is outside [0, {n_obs}]; boundaries are positions "
                "into the sample."
            )
    edges = [0, *[b for b in cuts if 0 < b < n_obs], n_obs]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]


def _segment_choice(
    rng: np.random.Generator, segments: list[tuple[int, int]], size: int
) -> np.ndarray:
    """Draw ``size`` segment ids with probability proportional to segment length."""
    lengths = np.array([b - a for a, b in segments], dtype=float)
    return rng.choice(len(segments), size=size, p=lengths / lengths.sum())


# --------------------------------------------------------------------------- #
# Block index generation
# --------------------------------------------------------------------------- #
def block_bootstrap_indices(
    n_obs: int,
    *,
    block_length: int,
    n_boot: int = 1000,
    scheme: str = "moving",
    boundaries: Sequence[int] | None = None,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Generate ``(n_boot, n_obs)`` resampling indices for a block bootstrap.

    Parameters
    ----------
    n_obs : int
        Length of the original sample.
    block_length : int
        Fixed block length ``L`` for ``"moving"``/``"circular"``; the *mean*
        block length (``1 / p`` of the geometric law) for ``"stationary"``.
    n_boot : int, default=1000
        Number of bootstrap replications.
    scheme : {"moving", "circular", "stationary"}, default="moving"
        Block scheme. ``"moving"`` never wraps (blocks must fit inside the
        segment); ``"circular"`` wraps *within* the segment; ``"stationary"``
        uses geometric block lengths and wraps within the segment.
    boundaries : sequence of int, optional
        Segment start positions; blocks never straddle a boundary.
    seed : int | numpy.random.Generator, optional
        Seed or generator. Given the same seed the output is bit-identical.

    Returns
    -------
    ndarray of shape (n_boot, n_obs), dtype int64
        Row ``b`` holds the positions to gather for replication ``b``.

    Raises
    ------
    ValueError
        If ``block_length`` is not positive or ``scheme`` is unknown.
    """
    if block_length < 1:
        raise ValueError(f"`block_length` must be >= 1, got {block_length}.")
    if n_boot < 1:
        raise ValueError(f"`n_boot` must be >= 1, got {n_boot}.")
    if scheme not in {"moving", "circular", "stationary"}:
        raise ValueError(
            f"unknown `scheme` {scheme!r}; expected 'moving', 'circular' or "
            "'stationary'."
        )
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    segments = resolve_segments(n_obs, boundaries)

    out = np.empty((n_boot, n_obs), dtype=np.int64)
    for b in range(n_boot):
        out[b] = _one_block_draw(rng, segments, block_length, scheme)
    return out


def _one_block_draw(
    rng: np.random.Generator,
    segments: list[tuple[int, int]],
    block_length: int,
    scheme: str,
) -> np.ndarray:
    """Resample each segment *in place* (segment lengths are preserved)."""
    pieces: list[np.ndarray] = []
    for start, stop in segments:
        seg_len = stop - start
        pieces.append(start + _draw_within(rng, seg_len, block_length, scheme))
    return np.concatenate(pieces)


def _draw_within(
    rng: np.random.Generator, seg_len: int, block_length: int, scheme: str
) -> np.ndarray:
    """Return ``seg_len`` relative positions built from blocks inside a segment."""
    if seg_len <= 0:
        return np.empty(0, dtype=np.int64)
    length = min(block_length, seg_len)
    idx: list[np.ndarray] = []
    filled = 0
    while filled < seg_len:
        if scheme == "stationary":
            # Geometric block length with mean `block_length`, capped by the
            # segment so the block cannot straddle a boundary.
            p = 1.0 / float(block_length)
            this = int(min(rng.geometric(p) if p < 1.0 else 1, seg_len))
        else:
            this = length
        if scheme == "moving":
            # Non-wrapping: the start must leave room for a full block.
            start = int(rng.integers(0, seg_len - this + 1))
            block = np.arange(start, start + this, dtype=np.int64)
        else:
            start = int(rng.integers(0, seg_len))
            block = (start + np.arange(this, dtype=np.int64)) % seg_len
        idx.append(block)
        filled += this
    return np.concatenate(idx)[:seg_len]


def _resample(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Gather ``x`` along axis 0 for every row of ``indices``."""
    return x[indices]


# --------------------------------------------------------------------------- #
# Public bootstrap samplers
# --------------------------------------------------------------------------- #
def moving_block_bootstrap(
    x: np.ndarray,
    *,
    block_length: int,
    n_boot: int = 1000,
    boundaries: Sequence[int] | None = None,
    seed: int | np.random.Generator | None = None,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Moving-block bootstrap (Kunsch, 1989).

    Parameters
    ----------
    x : ndarray of shape (T,) or (T, K)
        Sample; resampling is along the first axis.
    block_length : int
        Block length ``L``. Choose ``L`` on the order of the autocorrelation
        length (a common rule of thumb is ``T ** (1/3)``).
    n_boot : int, default=1000
        Replications.
    boundaries : sequence of int, optional
        Fold-boundary positions; blocks never straddle them.
    seed : int | numpy.random.Generator, optional
        Seed for reproducibility.
    return_indices : bool, default=False
        Also return the ``(n_boot, T)`` index matrix.

    Returns
    -------
    ndarray of shape (n_boot, T[, K])
        The bootstrap samples, or ``(samples, indices)`` if ``return_indices``.
    """
    return _run(
        x,
        block_length=block_length,
        n_boot=n_boot,
        scheme="moving",
        boundaries=boundaries,
        seed=seed,
        return_indices=return_indices,
    )


def circular_block_bootstrap(
    x: np.ndarray,
    *,
    block_length: int,
    n_boot: int = 1000,
    boundaries: Sequence[int] | None = None,
    seed: int | np.random.Generator | None = None,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Circular-block bootstrap (Politis & Romano, 1992).

    Identical to :func:`moving_block_bootstrap` except blocks wrap around
    (*within their segment*), which removes the edge bias that under-samples the
    first and last observations.
    """
    return _run(
        x,
        block_length=block_length,
        n_boot=n_boot,
        scheme="circular",
        boundaries=boundaries,
        seed=seed,
        return_indices=return_indices,
    )


def stationary_bootstrap(
    x: np.ndarray,
    *,
    block_length: float,
    n_boot: int = 1000,
    boundaries: Sequence[int] | None = None,
    seed: int | np.random.Generator | None = None,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Stationary bootstrap (Politis & Romano, 1994).

    Block lengths are geometric with mean ``block_length`` (so the smoothing
    parameter is ``p = 1 / block_length``), which makes the resampled series
    stationary. This is the resampling scheme Hansen's SPA test and the Model
    Confidence Set are defined with.

    Parameters
    ----------
    x : ndarray of shape (T,) or (T, K)
        Sample.
    block_length : float
        *Mean* block length (``>= 1``).
    n_boot : int, default=1000
        Replications.
    boundaries : sequence of int, optional
        Fold-boundary positions; blocks never straddle them.
    seed : int | numpy.random.Generator, optional
        Seed.
    return_indices : bool, default=False
        Also return the index matrix.
    """
    return _run(
        x,
        block_length=int(max(1, round(float(block_length)))),
        n_boot=n_boot,
        scheme="stationary",
        boundaries=boundaries,
        seed=seed,
        return_indices=return_indices,
    )


def _run(
    x: np.ndarray,
    *,
    block_length: int,
    n_boot: int,
    scheme: str,
    boundaries: Sequence[int] | None,
    seed: int | np.random.Generator | None,
    return_indices: bool,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x)
    if arr.ndim not in (1, 2):
        raise ValueError(f"`x` must be 1-D or 2-D, got shape {arr.shape}.")
    idx = block_bootstrap_indices(
        arr.shape[0],
        block_length=block_length,
        n_boot=n_boot,
        scheme=scheme,
        boundaries=boundaries,
        seed=seed,
    )
    samples = _resample(arr, idx)
    return (samples, idx) if return_indices else samples


def wild_bootstrap(
    residuals: np.ndarray,
    *,
    n_boot: int = 1000,
    distribution: str = "rademacher",
    fitted: np.ndarray | None = None,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Wild (multiplier) bootstrap for heteroskedastic errors.

    Multiplies each residual by an IID mean-zero, unit-variance multiplier, so
    conditional heteroskedasticity is preserved observation by observation.

    Parameters
    ----------
    residuals : ndarray of shape (T,) or (T, K)
        Residuals to perturb.
    n_boot : int, default=1000
        Replications.
    distribution : {"rademacher", "mammen", "normal"}, default="rademacher"
        Multiplier law. ``"rademacher"`` is ``+-1`` with equal probability (best
        for symmetric errors); ``"mammen"`` is the two-point law of Mammen
        (1993), which additionally matches the third moment; ``"normal"`` is
        standard Gaussian.
    fitted : ndarray, optional
        If given, the returned samples are ``fitted + multiplier * residuals``
        (a bootstrap of the dependent variable). Otherwise only the perturbed
        residuals are returned.
    seed : int | numpy.random.Generator, optional
        Seed.

    Returns
    -------
    ndarray of shape (n_boot,) + residuals.shape
        The bootstrap replications.

    Raises
    ------
    ValueError
        If ``distribution`` is unknown or ``fitted`` does not broadcast.
    """
    e = np.asarray(residuals, dtype=float)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    shape = (n_boot, *e.shape)
    if distribution == "rademacher":
        v = rng.integers(0, 2, size=shape).astype(float) * 2.0 - 1.0
    elif distribution == "normal":
        v = rng.standard_normal(shape)
    elif distribution == "mammen":
        s5 = np.sqrt(5.0)
        p = (s5 + 1.0) / (2.0 * s5)  # P(v = -(sqrt5 - 1)/2)
        lo = -(s5 - 1.0) / 2.0
        hi = (s5 + 1.0) / 2.0
        v = np.where(rng.random(shape) < p, lo, hi)
    else:
        raise ValueError(
            f"unknown `distribution` {distribution!r}; expected 'rademacher', "
            "'mammen' or 'normal'."
        )
    out = v * e
    if fitted is not None:
        f = np.asarray(fitted, dtype=float)
        if f.shape != e.shape:
            raise ValueError(
                f"`fitted` shape {f.shape} must match `residuals` shape {e.shape}."
            )
        out = out + f
    return out


def _ar_ols(x: np.ndarray, order: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit an AR(``order``) by least squares; return (intercept, coefs, resid)."""
    n = x.shape[0]
    if order <= 0:
        return float(np.mean(x)), np.empty(0), x - float(np.mean(x))
    rows = n - order
    design = np.empty((rows, order + 1), dtype=float)
    design[:, 0] = 1.0
    for j in range(1, order + 1):
        design[:, j] = x[order - j : n - j]
    y = x[order:]
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    return float(beta[0]), beta[1:], resid


def sieve_bootstrap(
    x: np.ndarray,
    *,
    n_boot: int = 1000,
    order: int | None = None,
    max_order: int | None = None,
    burn_in: int = 100,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """AR-sieve bootstrap (Buhlmann, 1997).

    Fits an AR(``p``) by OLS (order chosen by AIC when ``order`` is ``None``),
    IID-resamples the centred residuals and regenerates series of the original
    length. Appropriate for linear, invertible processes; unlike the block
    schemes it produces smooth (not spliced) paths.

    Parameters
    ----------
    x : ndarray of shape (T,)
        Univariate series.
    n_boot : int, default=1000
        Replications.
    order : int, optional
        AR order. If ``None``, chosen by AIC over ``0..max_order``.
    max_order : int, optional
        Upper bound for AIC selection. Defaults to
        ``min(T // 4, ceil(10 * log10(T)))``.
    burn_in : int, default=100
        Discarded warm-up steps for each generated path.
    seed : int | numpy.random.Generator, optional
        Seed.

    Returns
    -------
    ndarray of shape (n_boot, T)
        The bootstrap replications.
    """
    arr = np.asarray(x, dtype=float).ravel()
    n = arr.shape[0]
    if n < 4:
        raise ValueError(f"need at least 4 observations, got {n}.")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    if order is None:
        hi = max_order or int(min(n // 4, np.ceil(10 * np.log10(n))))
        hi = max(1, int(hi))
        best, best_aic = 0, np.inf
        for p in range(hi + 1):
            _, _, res = _ar_ols(arr, p)
            sigma2 = float(np.mean(res**2))
            if sigma2 <= 0:
                continue
            aic = (n - p) * np.log(sigma2) + 2.0 * (p + 1)
            if aic < best_aic:
                best, best_aic = p, aic
        order = best

    const, coefs, resid = _ar_ols(arr, order)
    resid = resid - float(np.mean(resid))
    if resid.size == 0:
        resid = np.zeros(1)

    out = np.empty((n_boot, n), dtype=float)
    total = n + burn_in
    mean_level = const / (1.0 - float(np.sum(coefs))) if order > 0 else const
    if not np.isfinite(mean_level):
        mean_level = float(np.mean(arr))
    for b in range(n_boot):
        eps = resid[rng.integers(0, resid.size, size=total)]
        path = np.empty(total, dtype=float)
        path[:order] = mean_level
        for t in range(order, total):
            lagged = path[t - order : t][::-1] if order > 0 else np.empty(0)
            path[t] = const + float(np.dot(coefs, lagged)) + eps[t]
        out[b] = path[burn_in:]
    return out
