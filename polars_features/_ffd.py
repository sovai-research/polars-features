"""Dependency-free fractional-differencing kernel and causal expression builder.

This is a **leaf module**: it imports ONLY :mod:`numpy` and :mod:`polars`. It is
the single source of truth for the fixed-width-window fractional-differencing
("FFD") weights and the causal frac-diff expression, shared by

* the estimator layer -- :mod:`polars_features.transform.frac_diff`
  (:class:`~polars_features.transform.frac_diff.FracDiff`), and
* the Polars-native namespace layer -- :mod:`polars_features.namespaces.panel`
  (``pl.col(...).panel.frac_diff`` and the frame-level ``.panel.frac_diff``),

as well as the compatibility shims ``.ts.frac_diff``
(:mod:`polars_features.feature_extractors`) and
:func:`polars_features.preprocessing.fractional_diff`.

Because this module depends on nothing inside the package it can be imported by
the ``namespaces`` layer without violating that layer's import boundary (which
forbids importing :mod:`polars_features.core` / :mod:`polars_features.transform`),
while still giving every surface exactly **one** weight recursion and **one**
causal expression builder. See de Prado, *Advances in Financial Machine
Learning* (Wiley, 2018), Chapter 5.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = [
    "DEFAULT_THRESHOLD",
    "SAFETY_MAX_WIDTH",
    "estimate_ffd_order",
    "ffd_weights",
    "frac_diff_expr",
]

#: Canonical default weight-magnitude cutoff. Reconciles the previously divergent
#: defaults across surfaces (``None`` / ``10_000`` / ``1000``) to one value.
#:
#: The value was raised from the historical ``1e-5`` to ``5e-4`` to fix a
#: silent all-null footgun: at ``1e-5`` the kernel for a typical ``d`` is very
#: long (e.g. ``d=0.4`` -> width 1458), so on a 500-1000 row per-entity series
#: the leading-null warm-up covered *every* row and ``frac_diff`` returned 0
#: non-null values while still paying the full compute cost. At ``5e-4`` the
#: kernel is a few dozen to ~a hundred terms for common ``d`` (``d=0.4`` ->
#: width 90, ``d=0.5`` -> 69, ``d=0.9`` -> 17), so a default call on a 500-1000
#: row series returns *mostly* non-null values while still retaining meaningful
#: long memory. Callers that want a longer kernel pass a smaller ``threshold``
#: (or an explicit ``max_width``) exactly as before -- only the default moved.
DEFAULT_THRESHOLD: float = 5e-4

#: Hard safety cap on the kernel width used when ``max_width`` is ``None``, to
#: avoid pathological non-termination for tiny thresholds / near-integer ``d``.
SAFETY_MAX_WIDTH: int = 100_000


def ffd_weights(
    d: float, threshold: float = DEFAULT_THRESHOLD, max_width: int | None = None
) -> np.ndarray:
    """Compute fixed-width-window fractional-differencing weights.

    The weights follow the recurrence (de Prado 2018, Chapter 5)::

        w_0 = 1
        w_k = -w_{k-1} * (d - k + 1) / k

    Generation stops once ``|w_k| < threshold`` or ``max_width`` weights have
    been produced (when ``max_width`` is ``None`` a large :data:`SAFETY_MAX_WIDTH`
    cap guards against non-termination). The returned array is ordered from the
    *oldest* lag to the *current* observation, i.e. ``weights[-1]`` multiplies
    ``x_t`` and ``weights[0]`` multiplies ``x_{t-(width-1)}`` -- a drop-in kernel
    for a trailing weighted window.

    Divergence guard
    ----------------
    The kernel is absolutely summable only for ``d >= 0``: for ``d < 0`` (that
    is, fractional *integration*) the weights decay like ``k**(d - 1)``, which is
    not summable, so a truncated fixed-width filter is dominated by the
    truncation point and its output grows without bound as the window grows.
    Negative ``d`` is therefore rejected outright. The same failure mode can be
    reached from a legal ``d`` with an unreachably small ``threshold``: if the
    recursion has not fallen below ``threshold`` after :data:`SAFETY_MAX_WIDTH`
    terms and no explicit ``max_width`` was given, a :class:`ValueError` is
    raised rather than silently returning a 100 000-tap kernel that is longer
    than any realistic per-entity series (which would null out every row).
    Passing an explicit ``max_width`` is an informed opt-in to truncation and
    never raises.

    Parameters
    ----------
    d : float
        Differencing order, must be finite and ``>= 0``. ``0`` is the identity,
        ``1`` is the first difference; non-integer values give fractional
        differencing.
    threshold : float, default=:data:`DEFAULT_THRESHOLD`
        Magnitude below which trailing weights are dropped (controls window
        width). Must be positive.
    max_width : int, optional
        Hard cap on the window width, regardless of ``threshold``. The returned
        array never has more than ``max_width`` entries.

    Returns
    -------
    numpy.ndarray
        1-D array of weights, oldest-to-newest.

    Raises
    ------
    ValueError
        If ``threshold <= 0``, ``max_width < 1``, ``d`` is negative or
        non-finite, or the kernel fails to decay below ``threshold`` within
        :data:`SAFETY_MAX_WIDTH` terms while ``max_width`` is ``None``.
    """
    if threshold <= 0:
        raise ValueError(f"`threshold` must be positive, got {threshold!r}.")
    if max_width is not None and max_width < 1:
        raise ValueError(f"`max_width` must be >= 1, got {max_width!r}.")
    if not np.isfinite(d):
        raise ValueError(f"`d` must be finite, got {d!r}.")
    if d < 0:
        raise ValueError(
            f"`d` must be >= 0 for fixed-width fractional differencing, got {d!r}. "
            "Negative `d` is fractional *integration*: its binomial weights decay "
            "like k**(d-1) and are not summable, so a truncated fixed-width filter "
            "diverges (the output is dominated by the oldest retained lag and grows "
            "with the window). Difference first, or use a positive `d`."
        )

    cap = max_width if max_width is not None else SAFETY_MAX_WIDTH
    weights = [1.0]
    k = 1
    hit_cap = False
    while True:
        if len(weights) >= cap:
            # Check the cap BEFORE appending so `max_width` is an exact bound
            # (the old post-append check let `max_width=1` return 2 weights).
            hit_cap = True
            break
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    if hit_cap and max_width is None:
        raise ValueError(
            f"fractional-differencing weights for d={d!r} did not decay below "
            f"threshold={threshold!r} within the {SAFETY_MAX_WIDTH}-term safety cap "
            "(the kernel would be longer than any realistic per-entity series, so "
            "every row would be null). Raise `threshold` or pass an explicit "
            "`max_width` to opt into truncation."
        )
    # newest-first by construction (w_0 multiplies x_t); reverse to oldest-first
    return np.asarray(weights[::-1], dtype=np.float64)


def estimate_ffd_order(
    x: np.ndarray,
    *,
    method: str = "local_whittle",
    bandwidth_exponent: float = 0.5,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    """Estimate a data-driven fractional-differencing order ``d`` for ``x``.

    Thin, lazily-wired convenience so callers can make frac-diff *data driven*
    per entity/window instead of hard-coding one ``d`` for a whole panel: it
    estimates the fractional-integration order of ``x`` with a semiparametric
    long-memory estimator and clips it into ``[lower, upper]`` so the result is
    always a legal :func:`ffd_weights` order.

    The estimator lives in
    :mod:`polars_features.econ.features._longmemory` and is imported *inside*
    this function, so :mod:`polars_features._ffd` keeps its leaf-module property
    (module-level imports remain numpy + polars only) and the ``namespaces``
    layer can go on importing it without picking up the estimator layer.

    Parameters
    ----------
    x : numpy.ndarray
        1-D series, in time order. Only rows available to the caller (i.e. the
        *training* rows) should be passed: this is a fitted quantity.
    method : {"local_whittle", "gph"}, default="local_whittle"
        Semiparametric estimator of ``d``.
    bandwidth_exponent : float, default=0.5
        Number of periodogram ordinates used is ``n ** bandwidth_exponent``.
    lower, upper : float
        Clipping bounds for the returned order. The default ``[0, 1]`` matches
        the range over which the fixed-width FFD kernel is well behaved.

    Returns
    -------
    float
        The estimated, clipped differencing order.

    See Also
    --------
    polars_features.econ.features.estimate_fractional_order
    polars_features.econ.features.AutoFracDiff
    """
    from polars_features.econ.features._longmemory import estimate_fractional_order

    d_hat = estimate_fractional_order(
        x, method=method, bandwidth_exponent=bandwidth_exponent
    ).d
    if not np.isfinite(d_hat):
        return float(lower)
    return float(min(max(d_hat, lower), upper))


def frac_diff_expr(
    expr: pl.Expr,
    *,
    d: float | None = None,
    weights: np.ndarray | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    max_width: int | None = None,
) -> pl.Expr:
    """Build the causal fixed-width fractional-differencing expression.

    Computes the causal weighted trailing dot-product
    ``w[-1]*x[t] + w[-2]*x[t-1] + ... + w[0]*x[t-(width-1)]`` for every row that
    has a full trailing window, and emits ``null`` for the incomplete leading
    warm-up. The expression is **entity-agnostic**: grouping is the caller's
    responsibility via ``.over(entity)``.

    Provide either ``d`` (weights are built via :func:`ffd_weights`) or a
    precomputed ``weights`` array (oldest-to-newest, as returned by
    :func:`ffd_weights`).

    Implementation
    --------------
    The filter is applied as a **causal FIR convolution** of each series with
    the (reversed, newest-first) weight kernel via
    ``numpy.convolve(values, weights[::-1])[:n]``, executed inside a Polars
    ``map_batches`` UDF. Under ``.over(entity)`` Polars calls the UDF once per
    entity group with that group's values, so the convolution never crosses an
    entity boundary and stays strictly causal (row ``t`` uses only ``x[t]`` and
    earlier). This replaces the previous ``pl.sum_horizontal`` of ``width``
    lagged ``.shift()`` terms, which built an ``O(n * width)`` expression tree
    (thousands of shift nodes for a long kernel) and was ~65-144x slower.

    Warm-up / null policy
    ---------------------
    The first ``min(width - 1, n - 1)`` rows of each group are emitted as
    ``null`` (an incomplete trailing window). When the kernel fits inside the
    series (``width - 1 < n`` -- the normal case and every case where the old
    ``sum_horizontal`` path produced a valid non-null value) this is exactly
    ``width - 1`` leading nulls and the non-null values are bit-for-bit the same
    dot products as before. When the kernel is *longer* than the series
    (``width - 1 >= n``) the old path nulled **every** row; this path instead
    caps the warm-up at ``n - 1`` so at least the final, fullest-window row is a
    correctly-computed partial output rather than returning all-null.

    Interior ``null`` inputs are treated as ``0`` inside the convolution,
    matching the historical ``sum_horizontal`` zero-fill behaviour.
    """
    if weights is None:
        if d is None:
            raise ValueError("frac_diff_expr requires either `d` or `weights`.")
        weights = ffd_weights(d, threshold, max_width)
    weights = np.asarray(weights, dtype=np.float64)
    width = int(weights.shape[0])
    # ``weights`` is oldest-first (weights[-1] multiplies x_t). ``np.convolve``
    # with the newest-first kernel ``kernel`` gives, at output index ``t``,
    # ``sum_k x[t-k] * kernel[k]`` -- exactly the causal trailing dot product
    # ``sum_k x[t-k] * weights[width-1-k]``. Reverse once, up front.
    kernel = np.ascontiguousarray(weights[::-1])

    def _apply_ffd(s: pl.Series) -> pl.Series:
        n = s.len()
        if n == 0:
            return pl.Series(s.name, [], dtype=pl.Float64)
        # Interior/leading nulls -> 0 so they contribute nothing to the sum
        # (historical ``sum_horizontal`` semantics), and never propagate NaN.
        values = np.nan_to_num(
            s.cast(pl.Float64).to_numpy(), nan=0.0, posinf=np.inf, neginf=-np.inf
        )
        out = np.convolve(values, kernel)[:n]
        result = pl.Series(s.name, out, dtype=pl.Float64)
        n_null = min(width - 1, n - 1)
        if n_null > 0:
            # Emit real Polars nulls (NOT float NaN) for the incomplete warm-up,
            # matching the old ``when/then/otherwise(None)`` mask -- Polars treats
            # NaN as a valid float, so ``is_null()`` must still see these rows.
            result = result.scatter(np.arange(n_null), None)
        return result

    return expr.map_batches(_apply_ffd, return_dtype=pl.Float64)
