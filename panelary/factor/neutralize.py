"""Per-date feature-set orthogonalization (de-correlation).

This is **distinct** from the Numerai-style ``.xs.neutralize`` operator, which
residualizes *one* target on a set of named exposures. :func:`orthogonalize`
de-correlates a *whole feature set against itself* — it turns a block of
collinear characteristics into a set of mutually orthogonal (hence per-date
uncorrelated) columns, via Gram-Schmidt or a QR decomposition.

The decomposition needs a matrix, so it runs per date in numpy, **never** with
a global scaler: each date's cross-section is centered and orthogonalized on
its own, so no cross-date information leaks. Columns are centered before
decomposition, so orthogonal columns are also uncorrelated.

Groups are materialised as row-index lists (``group_by(...).agg(row_index)``)
and the matrix is gathered with numpy fancy-indexing from a single whole-frame
``to_numpy()``. That avoids ``GroupBy.map_groups`` -- which builds a DataFrame
per group, concatenates them and needs a join to get back to row order, and
which modern Polars refuses outright whenever the group keys are anything but
plain column-name strings.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

__all__ = ["orthogonalize"]

FrameT = pl.LazyFrame | pl.DataFrame

_VALID_METHODS = ("gram_schmidt", "qr")


def _as_list(x: str | Sequence[str]) -> list[str]:
    return [x] if isinstance(x, str) else list(x)


def _orthogonalize_matrix(x: np.ndarray, *, method: str) -> np.ndarray:
    """Return a matrix whose columns are mutually orthogonal (and mean-zero).

    Columns are centered first, so the returned columns are not only orthogonal
    but also uncorrelated. ``"qr"`` returns the (reduced) orthonormal ``Q``;
    ``"gram_schmidt"`` returns classical Gram-Schmidt residuals (each column is
    the part of the original feature not spanned by the earlier ones).
    """
    xc = x - np.nanmean(x, axis=0, keepdims=True)
    xc = np.nan_to_num(xc, nan=0.0)
    n, k = xc.shape
    if method == "qr":
        q, _ = np.linalg.qr(xc)
        # Reduced QR gives min(n, k) columns; pad if a rank-deficient short block.
        if q.shape[1] < k:
            q = np.hstack([q, np.zeros((n, k - q.shape[1]))])
        return q[:, :k]
    if method == "gram_schmidt":
        out = np.zeros((n, k), dtype=np.float64)
        for j in range(k):
            v = xc[:, j].copy()
            for i in range(j):
                qi = out[:, i]
                denom = float(qi @ qi)
                if denom > 1e-12:
                    v = v - (qi @ xc[:, j]) / denom * qi
            out[:, j] = v
        return out
    raise ValueError(
        f"orthogonalize: `method` must be one of {sorted(_VALID_METHODS)}, "
        f"got {method!r}."
    )


def orthogonalize(
    frame: FrameT,
    columns: str | Sequence[str],
    *,
    by: str | Sequence[str] | None = None,
    time: str,
    method: str = "gram_schmidt",
    suffix: str = "_orth",
) -> FrameT:
    """Per-date orthogonalize a feature set into uncorrelated columns.

    For each ``time`` group (optionally further split by ``by``), the selected
    ``columns`` are centered and decomposed so the outputs are mutually
    orthogonal within that group. Because the groups are per date, the operation
    is leak-safe: one date's basis never depends on another's.

    Parameters
    ----------
    frame : LazyFrame | DataFrame
        The panel. Returned as the same type (collected internally for the
        per-group numpy step).
    columns : str | Sequence[str]
        Feature column(s) to orthogonalize.
    by : str | Sequence[str] | None, keyword-only, default None
        Optional extra grouping key(s) (e.g. ``"sector"``); the group key
        becomes ``[time, *by]``. Still per-date, so still leak-safe.
    time : str, keyword-only
        Date/time cross-section key. **Required**.
    method : str, keyword-only, default "gram_schmidt"
        ``"gram_schmidt"`` or ``"qr"``.
    suffix : str, keyword-only, default "_orth"
        Output columns are written as ``f"{col}{suffix}"``.

    Returns
    -------
    LazyFrame | DataFrame
        The input frame with one orthogonalized column added per input column.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"orthogonalize: `method` must be one of {sorted(_VALID_METHODS)}, "
            f"got {method!r}."
        )
    cols = _as_list(columns)
    keys = [time, *(_as_list(by) if by is not None else [])]
    out_names = [f"{c}{suffix}" for c in cols]

    was_lazy = isinstance(frame, pl.LazyFrame)
    eager: pl.DataFrame = frame.collect() if isinstance(frame, pl.LazyFrame) else frame

    # One whole-frame conversion; every group is a numpy gather out of it. The
    # per-group index lists come straight from Polars, so the grouping stays in
    # Rust and only the linear algebra happens in Python.
    values = eager.select(cols).to_numpy().astype(np.float64)
    out = np.empty_like(values)

    row_idx = "__orth_row__"
    groups = (
        eager.with_row_index(row_idx)
        .group_by(keys, maintain_order=True)
        .agg(pl.col(row_idx))
        .get_column(row_idx)
    )
    for rows in groups:
        take = rows.to_numpy()
        # `take` holds one date's row positions (optionally x `by`), so the
        # basis this block is decomposed against is built from that date's
        # cross-section alone -- no other date's rows are ever gathered.
        out[take] = _orthogonalize_matrix(values[take], method=method)

    merged = eager.with_columns(
        [pl.Series(out_names[j], out[:, j]) for j in range(len(cols))]
    )
    return merged.lazy() if was_lazy else merged
