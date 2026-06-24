"""Cross-sectional operators exposed under the ``.xs`` expression namespace.

Cross-sectional (``xs``) operators compare entities **against each other at a
single point in time**. They return plain :class:`polars.Expr` objects with no
grouping of their own; the user composes them with ``.over(date)`` so each
calendar date forms one cross-section.

Cross-sectional operators are inherently **leakage-safe** in the time dimension
(they use only contemporaneous data, never the future) but are *not* panel-safe
in the per-entity sense — by design they mix information across entities within
the same date.

In addition to the expression namespace, this module registers **frame-level**
``.xs`` namespaces on :class:`polars.LazyFrame` and :class:`polars.DataFrame` so
users can operate on a bare frame::

    lf.xs.rank("ret", over="date", normalize=True)   # rank within each date
    df.xs.demean("ret", over="date")

For these frame-level cross-sectional ops the ``over`` cross-section key is
**required** — a cross-sectional operation is meaningless without it.

Architecture / import boundary
------------------------------
This module is part of the ``namespaces`` package — **Tier 1, the Polars-native
extension layer**. It imports ONLY from :mod:`polars`, the Rust plugin, and
:mod:`polars_features.registry`. It must NOT import from
``polars_features.core`` or ``polars_features.transform`` (the estimator layer),
so the expression layer stays independently splittable into a standalone
``polars-panel`` plugin later.

The expression-building logic is factored into module-level ``_expr_*`` helpers
so the expression namespace and both frame-level namespaces share **one**
implementation with no duplication.

Usage
-----
>>> import polars as pl
>>> import polars_features.namespaces  # registers the namespace (side-effect)
>>> df = pl.DataFrame(
...     {"date": [1, 1, 1], "entity": ["a", "b", "c"], "ret": [0.1, -0.2, 0.05]}
... )
>>> out = df.with_columns(
...     r=pl.col("ret").xs.rank(normalize=True).over("date")
... )  # doctest: +SKIP

Notes
-----
Importing this module is a side effect: it registers the ``"xs"`` namespace with
Polars (idempotently) and registers each operator as a
:class:`~polars_features.registry.FeatureSpec` in the global registry.
"""

from __future__ import annotations

from typing import TypeVar

import polars as pl

from polars_features.registry import FeatureSpec, registry

__all__ = [
    "XSExprNamespace",
    "XSLazyFrameNamespace",
    "XSDataFrameNamespace",
    "register",
]

_LICENSE = "Apache-2.0"
_SOURCE = "PanelKit"

#: A LazyFrame or DataFrame — both expose ``with_columns`` identically, so the
#: frame-level cross-sectional operators are generic over the two.
FrameT = TypeVar("FrameT", pl.LazyFrame, pl.DataFrame)

_VALID_RANK_METHODS = ("average", "min", "max", "dense", "ordinal", "random")


# ---------------------------------------------------------------------------
# Shared expression builders — the SINGLE source of truth for each operator's
# expression tree, called by both the expression namespace and the frame-level
# namespaces so there is no duplication.
# ---------------------------------------------------------------------------


def _expr_rank(
    expr: pl.Expr, *, method: str = "average", normalize: bool = False
) -> pl.Expr:
    """Build the cross-sectional rank expression.

    See :meth:`XSExprNamespace.rank` for the full description.
    """
    if method not in _VALID_RANK_METHODS:
        raise ValueError(
            f"xs.rank method must be one of {sorted(_VALID_RANK_METHODS)}, "
            f"got {method!r}."
        )
    ranked = expr.rank(method=method)  # type: ignore[arg-type]
    if normalize:
        # Count non-null observations in the cross-section.
        count = expr.is_not_null().sum()
        return ranked / (count + 1)
    return ranked


def _expr_demean(expr: pl.Expr) -> pl.Expr:
    """Build the cross-sectional demeaning expression (subtract group mean)."""
    return expr - expr.mean()


def _apply_xs(
    frame: FrameT,
    expr: pl.Expr,
    column: str,
    *,
    over: str | None,
    alias: str | None,
    op: str,
) -> FrameT:
    """Apply a cross-sectional ``expr`` to ``frame`` within each ``over`` group.

    Shared by the LazyFrame and DataFrame ``.xs`` namespaces. ``over`` (the
    cross-section key, e.g. ``date``) is **required**: cross-sectional ops are
    meaningless without a cross-section, so a missing key raises a clear error.
    The result is written to ``alias`` if given, else it replaces ``column``.
    """
    if over is None:
        raise ValueError(
            f"xs.{op} requires an `over` cross-section key (e.g. over='date'); "
            "cross-sectional operations are undefined without one."
        )
    expr = expr.over(over).alias(alias if alias is not None else column)
    return frame.with_columns(expr)


class XSExprNamespace:
    """Expression methods registered under the ``.xs`` namespace.

    Instances are created by Polars when you access ``expr.xs``; you do not
    construct this class directly. Every method returns a :class:`polars.Expr`
    and is intended to be combined with ``.over(date)``.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def rank(self, *, method: str = "average", normalize: bool = False) -> pl.Expr:
        """Cross-sectional rank of the value within its group.

        Ranks the input across the cross-section (e.g. all entities on one
        date). With ``normalize=True`` the ranks are rescaled to the open-ish
        interval ``(0, 1)`` via ``rank / (count + 1)``, giving a uniform-ish
        score that is robust to the size of the cross-section.

        Parameters
        ----------
        method : str, keyword-only, default "average"
            Tie-handling method forwarded to :meth:`polars.Expr.rank`. One of
            ``"average"``, ``"min"``, ``"max"``, ``"dense"``, ``"ordinal"``,
            ``"random"``.
        normalize : bool, keyword-only, default False
            If ``True``, divide ranks by ``count + 1`` so the output lies in
            ``(0, 1)``. Nulls are ignored in the count.

        Returns
        -------
        pl.Expr
            The cross-sectional rank. Combine with ``.over(date)``.
        """
        return _expr_rank(self._expr, method=method, normalize=normalize)

    def demean(self) -> pl.Expr:
        """Subtract the cross-sectional mean from each value.

        Centers the cross-section so it has zero mean (a market-neutralisation
        style operation when applied to returns ``.over(date)``). Nulls are
        ignored when computing the mean.

        Returns
        -------
        pl.Expr
            The demeaned series. Combine with ``.over(date)``.
        """
        return _expr_demean(self._expr)


class _XSFrameNamespace:
    """Shared frame-level ``.xs`` operators for LazyFrame and DataFrame.

    Each method names the target ``column`` first, takes a **required** ``over``
    cross-section key (e.g. ``date``), an optional ``alias`` (output column,
    defaults to replacing ``column``), and the same operator params as the
    expression namespace. The expression is built by the shared ``_expr_*``
    helpers, so nothing is duplicated with the expression namespace.

    LazyFrame and DataFrame share one implementation via :func:`_apply_xs`.
    """

    _frame: pl.LazyFrame | pl.DataFrame

    def rank(
        self,
        column: str,
        *,
        over: str | None = None,
        method: str = "average",
        normalize: bool = False,
        alias: str | None = None,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Cross-sectional rank of ``column`` within each ``over`` group.

        Parameters
        ----------
        column : str
            Name of the column to rank.
        over : str | None, keyword-only
            Cross-section key (e.g. ``"date"``). **Required** — raises
            :class:`ValueError` if omitted.
        method : str, keyword-only, default "average"
            Tie-handling method (``"average"``, ``"min"``, ``"max"``,
            ``"dense"``, ``"ordinal"``, ``"random"``).
        normalize : bool, keyword-only, default False
            If ``True``, rescale ranks to ``(0, 1)`` via ``rank / (count + 1)``.
        alias : str | None, keyword-only, default None
            Output column name. Defaults to replacing ``column`` in place.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The frame with the new/replaced column (same type as the input).
        """
        expr = _expr_rank(pl.col(column), method=method, normalize=normalize)
        return _apply_xs(self._frame, expr, column, over=over, alias=alias, op="rank")

    def demean(
        self,
        column: str,
        *,
        over: str | None = None,
        alias: str | None = None,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Subtract the cross-sectional mean of ``column`` within each ``over``.

        Parameters
        ----------
        column : str
            Name of the column to demean.
        over : str | None, keyword-only
            Cross-section key (e.g. ``"date"``). **Required** — raises
            :class:`ValueError` if omitted.
        alias : str | None, keyword-only, default None
            Output column name. Defaults to replacing ``column`` in place.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The frame with the new/replaced column.
        """
        expr = _expr_demean(pl.col(column))
        return _apply_xs(self._frame, expr, column, over=over, alias=alias, op="demean")


class XSLazyFrameNamespace(_XSFrameNamespace):
    """Frame-level ``.xs`` operators on :class:`polars.LazyFrame`.

    Accessed as ``lf.xs.<op>(...)``. Returns a new :class:`polars.LazyFrame`.
    """

    def __init__(self, lf: pl.LazyFrame) -> None:
        self._frame = lf


class XSDataFrameNamespace(_XSFrameNamespace):
    """Frame-level ``.xs`` operators on :class:`polars.DataFrame`.

    Accessed as ``df.xs.<op>(...)``. Returns a new :class:`polars.DataFrame`.
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._frame = df


def register() -> None:
    """Register the ``.xs`` namespace and its feature specs (idempotent).

    Registration is guarded so that re-importing this module does not raise the
    Polars "namespace already registered" error.
    """
    # Only register when absent to avoid the "overriding existing custom
    # namespace" UserWarning on re-import; the try/except guards the documented
    # "already registered" condition across Polars versions.
    if not _is_registered(pl.Expr, "xs"):
        try:
            pl.api.register_expr_namespace("xs")(XSExprNamespace)
        except Exception as exc:  # pragma: no cover - exact type varies by version
            if "already" not in str(exc).lower():
                raise

    # Frame-level ergonomics: ``lf.xs.<op>`` / ``df.xs.<op>``. Guarded the same
    # way to keep re-import a no-op and avoid the override UserWarning.
    if not _is_registered(pl.LazyFrame, "xs"):
        try:
            pl.api.register_lazyframe_namespace("xs")(XSLazyFrameNamespace)
        except Exception as exc:  # pragma: no cover - exact type varies by version
            if "already" not in str(exc).lower():
                raise
    if not _is_registered(pl.DataFrame, "xs"):
        try:
            pl.api.register_dataframe_namespace("xs")(XSDataFrameNamespace)
        except Exception as exc:  # pragma: no cover - exact type varies by version
            if "already" not in str(exc).lower():
                raise

    for spec in _SPECS:
        registry.register(spec)


def _is_registered(cls: type, name: str) -> bool:
    """Return ``True`` if a custom namespace ``name`` already exists on ``cls``."""
    return hasattr(cls, name)


_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="rank",
        namespace="xs",
        input_shape="series",
        output_shape="series",
        params={"method": str, "normalize": bool},
        tier="A",
        panel_safe=False,  # cross-sectional: deliberately mixes entities
        leakage_safe=True,  # uses only contemporaneous data
        source=_SOURCE,
        license=_LICENSE,
    ),
    FeatureSpec(
        name="demean",
        namespace="xs",
        input_shape="series",
        output_shape="series",
        params={},
        tier="A",
        panel_safe=False,
        leakage_safe=True,
        source=_SOURCE,
        license=_LICENSE,
    ),
)


register()
