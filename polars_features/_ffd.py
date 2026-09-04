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
    "ffd_weights",
    "frac_diff_expr",
]

#: Canonical default weight-magnitude cutoff. Reconciles the previously divergent
#: defaults across surfaces (``None`` / ``10_000`` / ``1000``) to one value.
DEFAULT_THRESHOLD: float = 1e-5

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

    Parameters
    ----------
    d : float
        Differencing order. ``0`` is the identity, ``1`` is the first
        difference; non-integer values give fractional differencing.
    threshold : float, default=:data:`DEFAULT_THRESHOLD`
        Magnitude below which trailing weights are dropped (controls window
        width). Must be positive.
    max_width : int, optional
        Hard cap on the window width, regardless of ``threshold``.

    Returns
    -------
    numpy.ndarray
        1-D array of weights, oldest-to-newest.
    """
    if threshold <= 0:
        raise ValueError(f"`threshold` must be positive, got {threshold!r}.")
    if max_width is not None and max_width < 1:
        raise ValueError(f"`max_width` must be >= 1, got {max_width!r}.")

    cap = max_width if max_width is not None else SAFETY_MAX_WIDTH
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
        if len(weights) >= cap:
            break
    # newest-first by construction (w_0 multiplies x_t); reverse to oldest-first
    return np.asarray(weights[::-1], dtype=np.float64)


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
    ``w[-1]*x[t] + w[-2]*x[t-1] + ... + w[0]*x[t-(width-1)]`` and emits ``null``
    for the incomplete leading window (the first ``width - 1`` rows), matching
    rolling-window semantics. The expression is **entity-agnostic**: grouping is
    the caller's responsibility via ``.over(entity)``.

    Provide either ``d`` (weights are built via :func:`ffd_weights`) or a
    precomputed ``weights`` array (oldest-to-newest, as returned by
    :func:`ffd_weights`).

    Notes
    -----
    The sum is built with :func:`polars.sum_horizontal` (a flat, shallow
    expression tree) rather than a left-folded ``a + b + c + ...``. A long kernel
    (small ``threshold`` / small ``d``) can produce thousands of weights; a
    left-folded sum overflows the Rust evaluation stack. ``sum_horizontal`` also
    treats the nulls shifted in at the leading edge as ``0``; an explicit null
    mask on the first ``width - 1`` rows restores strict rolling-window / causal
    semantics (never zero-fill the incomplete window).
    """
    if weights is None:
        if d is None:
            raise ValueError("frac_diff_expr requires either `d` or `weights`.")
        weights = ffd_weights(d, threshold, max_width)
    weights = np.asarray(weights, dtype=np.float64)
    width = int(weights.shape[0])

    # ``weights`` is oldest-first: weights[j] multiplies x[t - (width-1-j)].
    terms = [float(w) * expr.shift((width - 1) - j) for j, w in enumerate(weights)]
    out = pl.sum_horizontal(terms)
    if width <= 1:
        return out
    row = pl.int_range(pl.len(), dtype=pl.Int64)
    return pl.when(row >= width - 1).then(out).otherwise(None)
