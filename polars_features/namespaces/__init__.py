"""PanelKit Polars expression namespaces.

Importing this subpackage is a **side effect**: it registers PanelKit's custom
Polars expression namespaces and populates the global feature registry. After

>>> import polars_features.namespaces  # noqa: F401

the following accessors are available on any :class:`polars.Expr`:

* ``expr.panel`` — per-entity, causal time-series operators
  (:class:`~polars_features.namespaces.panel.PanelExprNamespace`). Combine with
  ``.over(entity)``.
* ``expr.xs`` — cross-sectional operators
  (:class:`~polars_features.namespaces.xs.XSExprNamespace`). Combine with
  ``.over(date)``.

Registration is idempotent — re-importing this package will not raise the Polars
"namespace already registered" error.

Autocomplete & type-checking
----------------------------
Polars expression namespaces are added dynamically at runtime, which normally
defeats IDE autocomplete and static type checkers. PanelKit ships ``.pyi`` stub
files (``panel.pyi``, ``xs.pyi``) that declare the namespace classes and their
typed method signatures. To make ``pl.col(...).panel.<method>`` resolve in your
editor / type checker, import the namespace classes from this package; tools
that honour ``py.typed`` (present at the package root) will pick up the stubs.
"""

from __future__ import annotations

from polars_features.namespaces.panel import PanelExprNamespace
from polars_features.namespaces.xs import XSExprNamespace

__all__ = ["PanelExprNamespace", "XSExprNamespace"]
