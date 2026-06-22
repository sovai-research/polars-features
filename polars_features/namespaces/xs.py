"""Cross-sectional operators exposed under the ``.xs`` expression namespace.

Cross-sectional (``xs``) operators compare entities **against each other at a
single point in time**. They return plain :class:`polars.Expr` objects with no
grouping of their own; the user composes them with ``.over(date)`` so each
calendar date forms one cross-section.

Cross-sectional operators are inherently **leakage-safe** in the time dimension
(they use only contemporaneous data, never the future) but are *not* panel-safe
in the per-entity sense — by design they mix information across entities within
the same date.

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

import polars as pl

from polars_features.registry import FeatureSpec, registry

__all__ = ["XSExprNamespace", "register"]

_LICENSE = "Apache-2.0"
_SOURCE = "PanelKit"


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
        valid_methods = {"average", "min", "max", "dense", "ordinal", "random"}
        if method not in valid_methods:
            raise ValueError(
                f"xs.rank method must be one of {sorted(valid_methods)}, "
                f"got {method!r}."
            )
        ranked = self._expr.rank(method=method)  # type: ignore[arg-type]
        if normalize:
            # Count non-null observations in the cross-section.
            count = self._expr.is_not_null().sum()
            return ranked / (count + 1)
        return ranked

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
        return self._expr - self._expr.mean()


def register() -> None:
    """Register the ``.xs`` namespace and its feature specs (idempotent).

    Registration is guarded so that re-importing this module does not raise the
    Polars "namespace already registered" error.
    """
    # Only register when absent to avoid the "overriding existing custom
    # namespace" UserWarning on re-import; the try/except guards the documented
    # "already registered" condition across Polars versions.
    if not _is_registered("xs"):
        try:
            pl.api.register_expr_namespace("xs")(XSExprNamespace)
        except Exception as exc:  # pragma: no cover - exact type varies by version
            if "already" not in str(exc).lower():
                raise

    for spec in _SPECS:
        registry.register(spec)


def _is_registered(name: str) -> bool:
    """Return ``True`` if a custom expression namespace ``name`` already exists."""
    return hasattr(pl.Expr, name)


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
