"""Type stubs for the ``.xs`` Polars expression namespace.

These stubs give IDEs and static type checkers (mypy, Pyright) full knowledge of
the methods registered onto :class:`polars.Expr` via
``pl.api.register_expr_namespace("xs")``. Because Polars adds the namespace
dynamically at runtime, without these stubs ``pl.col(...).xs.rank(...)`` would
not autocomplete or type-check.

The package ships ``py.typed`` (PEP 561), so a checker that resolves
``polars_features.namespaces.xs`` will read this ``.pyi`` in preference to the
runtime module. Importing :class:`XSExprNamespace` from the package is enough for
the stubs to be discovered.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

__all__ = [
    "XSExprNamespace",
    "XSLazyFrameNamespace",
    "XSDataFrameNamespace",
    "register",
]

_RankMethod = Literal["average", "min", "max", "dense", "ordinal", "random"]

class XSExprNamespace:
    def __init__(self, expr: pl.Expr) -> None: ...
    def rank(
        self,
        *,
        method: _RankMethod = ...,
        normalize: bool = ...,
    ) -> pl.Expr: ...
    def demean(self) -> pl.Expr: ...

class XSLazyFrameNamespace:
    def __init__(self, lf: pl.LazyFrame) -> None: ...
    def rank(
        self,
        column: str,
        *,
        over: str | None = ...,
        method: _RankMethod = ...,
        normalize: bool = ...,
        alias: str | None = ...,
    ) -> pl.LazyFrame: ...
    def demean(
        self,
        column: str,
        *,
        over: str | None = ...,
        alias: str | None = ...,
    ) -> pl.LazyFrame: ...

class XSDataFrameNamespace:
    def __init__(self, df: pl.DataFrame) -> None: ...
    def rank(
        self,
        column: str,
        *,
        over: str | None = ...,
        method: _RankMethod = ...,
        normalize: bool = ...,
        alias: str | None = ...,
    ) -> pl.DataFrame: ...
    def demean(
        self,
        column: str,
        *,
        over: str | None = ...,
        alias: str | None = ...,
    ) -> pl.DataFrame: ...

def register() -> None: ...
