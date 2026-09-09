"""Per-cross-section OLS residual kernel (the single source of truth).

This is a **leaf module**: it imports ONLY :mod:`numpy` and :mod:`polars`. It
holds the one implementation of "regress a target on named factor columns
across one cross-section and keep the residual", shared by

* the Polars-native namespace layer -- :mod:`panelary.namespaces.xs`
  (``pl.col(...).xs.neutralize(...)`` and the frame-level ``.xs.neutralize``),
  and
* the estimator layer -- :mod:`panelary.transform.neutralize`
  (:class:`~panelary.transform.neutralize.Neutralize`).

Why it lives under ``namespaces/`` rather than ``transform/``
-------------------------------------------------------------
The ``namespaces`` package deliberately does **not** import
``panelary.core`` / ``panelary.transform``, so the expression layer stays
independently splittable (see the module docstring of
:mod:`panelary.namespaces.xs`). Putting the kernel in ``transform/`` would force
a ``namespaces -> transform`` edge and break that boundary; putting it here lets
the estimator layer depend *downwards* on the leaf, which is the direction the
layering already allows (cf. :mod:`panelary._internal._ffd`, the analogous frac-diff
leaf). Nothing in this module imports anything else from ``panelary``, so it
cannot participate in an import cycle.

Leakage contract
----------------
The kernel is deliberately **cross-section local**: it is handed the rows of a
*single* time slice and has no notion of ordering, history or state. Both call
sites feed it one ``time_col`` group at a time (``.over(time_col)`` for the
expression namespace, ``group_by(time_col)`` for the transformer), so no
information can cross dates. Keeping the arithmetic here — with no way to see
more than one cross-section — is what makes that guarantee checkable.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

__all__ = ["build_design", "cross_section_residuals"]


def build_design(
    df: pl.DataFrame, factors: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Build the one-hot expanded float64 design matrix for one cross-section.

    Numeric factor columns enter directly; every non-numeric column (String,
    Categorical, Enum, Boolean) is one-hot encoded **within this cross-section**
    only. The intercept, if any, is added by the caller.

    Parameters
    ----------
    df : polars.DataFrame
        The rows of a single cross-section. Must contain every name in
        ``factors``.
    factors : sequence of str
        Factor column names, in the order they should enter the design.

    Returns
    -------
    design : numpy.ndarray
        Shape ``(df.height, k)``, dtype ``float64``. Numeric factors first (in
        the order given), then the one-hot blocks.
    used : list of str
        The source column names, numeric first then categorical — the grouping
        that produced ``design``'s column blocks (not one name per column).
    """
    schema = df.schema
    numeric_cols = [f for f in factors if schema[f].is_numeric()]
    cat_cols = [f for f in factors if not schema[f].is_numeric()]

    parts: list[np.ndarray] = []
    if numeric_cols:
        parts.append(df.select(numeric_cols).to_numpy().astype(np.float64))
    if cat_cols:
        dummies = df.select(cat_cols).to_dummies(columns=cat_cols)
        parts.append(dummies.to_numpy().astype(np.float64))

    design = np.hstack(parts) if parts else np.empty((df.height, 0), dtype=np.float64)
    return design, numeric_cols + cat_cols


def cross_section_residuals(
    df: pl.DataFrame,
    target: str,
    factors: Sequence[str],
    *,
    add_intercept: bool,
) -> np.ndarray:
    """OLS residual of ``target`` on ``factors`` for one cross-section.

    Rows whose target or any factor is non-finite (null arrives as NaN through
    :meth:`polars.DataFrame.to_numpy`, and NaN/inf are treated the same way) are
    excluded from the fit and receive ``NaN`` — the caller turns that into a
    null. If the usable rows are fewer than the number of parameters the fit is
    rank-deficient in a way that would make the residual meaningless, so every
    row is left ``NaN``.

    Parameters
    ----------
    df : polars.DataFrame
        The rows of a single cross-section, containing ``target`` and every
        name in ``factors``.
    target : str
        Name of the column being neutralized.
    factors : sequence of str
        Factor column names.
    add_intercept : bool
        Include a constant column, which also de-means the residual.

    Returns
    -------
    numpy.ndarray
        Shape ``(df.height,)``, dtype ``float64``, ``NaN`` where undefined.
    """
    n = df.height
    y_full = df.get_column(target).to_numpy().astype(np.float64)

    design, _ = build_design(df, factors)
    if add_intercept:
        design = np.hstack([np.ones((n, 1), dtype=np.float64), design])

    residual = np.full(n, np.nan, dtype=np.float64)
    valid = np.isfinite(y_full)
    if design.shape[1] > 0:
        valid &= np.isfinite(design).all(axis=1)

    n_valid = int(valid.sum())
    # Need at least as many observations as parameters, else leave nulls.
    if n_valid == 0 or (design.shape[1] > 0 and n_valid < design.shape[1]):
        return residual

    x = design[valid]
    y = y_full[valid]
    if x.shape[1] == 0:  # no regressors at all -> residual is the raw value
        residual[valid] = y
        return residual

    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual[valid] = y - x @ coef
    return residual
