"""Type stubs for the ``.panel`` Polars expression namespace.

These stubs give IDEs and static type checkers (mypy, Pyright) full knowledge of
the methods registered onto :class:`polars.Expr` via
``pl.api.register_expr_namespace("panel")``. Because Polars adds the namespace
dynamically at runtime, without these stubs ``pl.col(...).panel.frac_diff(...)``
would not autocomplete or type-check.

The package ships ``py.typed`` (PEP 561), so a checker that resolves
``polars_features.namespaces.panel`` will read this ``.pyi`` in preference to the
runtime module. Importing :class:`PanelExprNamespace` from the package is enough
for the stubs to be discovered.
"""

from __future__ import annotations

import polars as pl

__all__ = ["PanelExprNamespace", "register"]

class PanelExprNamespace:
    def __init__(self, expr: pl.Expr) -> None: ...
    def frac_diff(self, d: float, *, threshold: float = ...) -> pl.Expr: ...
    def zscore(self, window: int) -> pl.Expr: ...
    def rs_vol(self, window: int) -> pl.Expr: ...

def register() -> None: ...
