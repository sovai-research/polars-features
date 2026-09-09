"""The ``pl.col(...).catch22`` Polars expression namespace.

Importing this module registers the namespace as a side effect; the
registration is guarded so a re-import (or a second copy of the package on
the path) does not raise on double registration.
"""

from __future__ import annotations

import polars as pl

from ._catalogue import _compute, _resolve_names

# ---------------------------------------------------------------------------
# Optional, double-registration-guarded Polars namespace: ``expr.catch22``
# ---------------------------------------------------------------------------
if not hasattr(pl.Expr, "catch22"):

    @pl.api.register_expr_namespace("catch22")
    class _Catch22ExprNamespace:  # noqa: D101 - thin convenience wrapper
        def __init__(self, expr: pl.Expr) -> None:
            self._expr = expr

        def all(self, *, catch24: bool = False, alias: str = "catch22") -> pl.Expr:
            """Struct expression of all catch22 (or catch24) features.

            Use inside ``group_by(...).agg(...)``; see :func:`catch22_all_expr`.
            """
            names = _resolve_names("all", catch24)
            struct_dtype = pl.Struct([pl.Field(n, pl.Float64) for n in names])

            def _fn(s: pl.Series) -> pl.Series:
                return pl.Series([_compute(s.to_numpy(), names)], dtype=struct_dtype)

            return self._expr.map_batches(
                _fn, return_dtype=struct_dtype, returns_scalar=True
            ).alias(alias)
