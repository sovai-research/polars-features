"""Panelary Polars expression namespaces.

Importing this subpackage is a **side effect**: it registers Panelary's custom
Polars expression namespaces and populates the global feature registry. After

>>> import panelary.namespaces  # noqa: F401

the following accessors are available on any :class:`polars.Expr`:

* ``expr.panel`` — per-entity, causal time-series operators
  (:class:`~panelary.namespaces.panel.PanelExprNamespace`). Combine with
  ``.over(entity)``.
* ``expr.xs`` — cross-sectional operators
  (:class:`~panelary.namespaces.xs.XSExprNamespace`). Combine with
  ``.over(date)``.

Frame-level ergonomics are also registered on :class:`polars.LazyFrame` and
:class:`polars.DataFrame` so a bare frame can be operated on directly::

    lf.panel.frac_diff("ret", d=0.4, over="ticker", alias="ret_fd")
    df.xs.rank("ret", over="date", normalize=True)

These are provided by ``Panel{Lazy,Data}FrameNamespace`` and
``XS{Lazy,Data}FrameNamespace`` and build the SAME expressions as the expression
namespaces (shared ``_expr_*`` helpers — no duplication).

Registration is idempotent — re-importing this package will not raise the Polars
"namespace already registered" error.

Autocomplete & type-checking
----------------------------
Polars expression namespaces are added dynamically at runtime, which normally
defeats IDE autocomplete and static type checkers. Panelary ships ``.pyi`` stub
files (``panel.pyi``, ``xs.pyi``) that declare the namespace classes and their
typed method signatures. To make ``pl.col(...).panel.<method>`` resolve in your
editor / type checker, import the namespace classes from this package; tools
that honour ``py.typed`` (present at the package root) will pick up the stubs.
"""

from __future__ import annotations

from panelary.namespaces.panel import (
    PanelDataFrameNamespace,
    PanelExprNamespace,
    PanelLazyFrameNamespace,
)
from panelary.namespaces.xs import (
    XSDataFrameNamespace,
    XSExprNamespace,
    XSLazyFrameNamespace,
)

__all__ = [
    "PanelExprNamespace",
    "PanelLazyFrameNamespace",
    "PanelDataFrameNamespace",
    "XSExprNamespace",
    "XSLazyFrameNamespace",
    "XSDataFrameNamespace",
]
