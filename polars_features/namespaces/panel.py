"""Per-entity panel operators exposed under the ``.panel`` expression namespace.

The operators here are *time-series* transforms intended to be applied **within a
single entity** of a long-format ``(entity, time, *features)`` panel. They return
plain :class:`polars.Expr` objects and carry **no** grouping of their own — the
user composes them with ``.over(entity)`` so the same expression works on one
series or a million.

In addition to the expression namespace, this module registers **frame-level**
``.panel`` namespaces on :class:`polars.LazyFrame` and :class:`polars.DataFrame`
so users can operate on a bare frame without hand-composing
``pl.col(...).panel.<op>().over(entity)``::

    lf.panel.frac_diff("ret", d=0.4, over="ticker", alias="ret_fd")
    df.panel.zscore("ret", window=21, over="ticker")

All operators in this namespace are **causal** (leakage-safe): the value at time
``t`` depends only on observations at or before ``t``. Combined with
``.over(entity)`` they are also panel-safe (no cross-entity bleed).

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
...     {"entity": ["a", "a", "a"], "ret": [0.1, -0.2, 0.05]}
... )
>>> out = df.with_columns(
...     fd=pl.col("ret").panel.frac_diff(0.4).over("entity")
... )  # doctest: +SKIP

Notes
-----
Importing this module is a side effect: it registers the ``"panel"`` namespace
with Polars (idempotently) and registers each operator as a
:class:`~polars_features.registry.FeatureSpec` in the global registry.
"""

from __future__ import annotations

from typing import TypeVar

import polars as pl

from polars_features.registry import FeatureSpec, registry

__all__ = [
    "PanelExprNamespace",
    "PanelLazyFrameNamespace",
    "PanelDataFrameNamespace",
    "register",
]

#: A LazyFrame or DataFrame — both expose ``with_columns`` with identical
#: semantics, so the frame-level operators are generic over the two.
FrameT = TypeVar("FrameT", pl.LazyFrame, pl.DataFrame)

_LICENSE = "Apache-2.0"
_SOURCE = "PanelKit"


def _frac_diff_weights(d: float, threshold: float) -> list[float]:
    """Compute fixed-width fractional-differencing weights.

    Implements the standard expanding-window weight recursion for fractional
    differencing of order ``d`` (Hosking; popularised for finance by López de
    Prado), truncated when the absolute weight falls below ``threshold``. This
    produces a *fixed-width* causal kernel.

    Parameters
    ----------
    d : float
        Order of fractional differencing. ``d == 0`` returns the identity
        kernel ``[1.0]``; ``d == 1`` approximates a first difference.
    threshold : float
        Positive cutoff; weights with ``abs(w) < threshold`` (and all that
        follow) are dropped.

    Returns
    -------
    list[float]
        Weights ``w[0], w[1], ...`` ordered from the most recent lag (lag 0,
        always ``1.0``) to the most distant. The kernel length is the window.
    """
    if threshold <= 0:
        raise ValueError(
            f"frac_diff threshold must be strictly positive, got {threshold!r}."
        )
    weights: list[float] = [1.0]
    k = 1
    # Cap the kernel width to avoid pathological non-termination for tiny
    # thresholds / near-integer d.
    max_width = 10_000
    while k < max_width:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    return weights


# ---------------------------------------------------------------------------
# Shared expression builders.
#
# These module-level functions are the SINGLE source of truth for each
# operator's expression tree. The ``.panel`` expression namespace and both
# frame-level namespaces (LazyFrame/DataFrame) call them, so there is exactly
# one implementation per operator and no duplication.
# ---------------------------------------------------------------------------


def _expr_frac_diff(expr: pl.Expr, d: float, *, threshold: float = 1e-5) -> pl.Expr:
    """Build the fixed-width fractional-differencing expression (causal).

    See :meth:`PanelExprNamespace.frac_diff` for the full description.
    """
    weights = _frac_diff_weights(d, threshold)
    width = len(weights)
    # Causal weighted sum: w[0]*x[t] + w[1]*x[t-1] + ...
    #
    # Build a *flat* n-ary sum via ``pl.sum_horizontal`` rather than folding
    # the terms with Python ``+``. A long kernel (small ``threshold`` / small
    # ``d``) can produce thousands of weights; a left-folded ``a + b + c +
    # ...`` produces a deeply nested expression tree that overflows the
    # Rust evaluation stack. ``sum_horizontal`` keeps the tree shallow.
    terms = [w * expr.shift(lag) for lag, w in enumerate(weights)]
    out = pl.sum_horizontal(terms)
    if width <= 1:
        return out
    # ``sum_horizontal`` treats shifted-in nulls as 0, which would emit a
    # value before the kernel has full history. Mask the leading rows that
    # lack ``width`` observations to ``null`` so the output matches
    # rolling-window semantics and stays strictly causal.
    row = pl.int_range(pl.len(), dtype=pl.Int64)
    return pl.when(row >= width - 1).then(out).otherwise(None)


def _expr_zscore(expr: pl.Expr, window: int) -> pl.Expr:
    """Build the causal rolling z-score expression over a trailing ``window``."""
    if not isinstance(window, int) or window <= 0:
        raise ValueError(
            f"zscore window must be a positive integer, got {window!r}."
        )
    mean = expr.rolling_mean(window_size=window)
    std = expr.rolling_std(window_size=window)
    return (expr - mean) / std


def _expr_rs_vol(expr: pl.Expr, window: int) -> pl.Expr:
    """Build the trailing realized-volatility proxy expression over ``window``."""
    if not isinstance(window, int) or window <= 0:
        raise ValueError(
            f"rs_vol window must be a positive integer, got {window!r}."
        )
    return expr.rolling_std(window_size=window)


def _apply_over(
    frame: FrameT,
    expr: pl.Expr,
    column: str,
    *,
    over: str | None,
    alias: str | None,
) -> FrameT:
    """Apply a panel ``expr`` to ``frame``, optionally grouped by ``over``.

    Shared by the LazyFrame and DataFrame ``.panel`` namespaces. When ``over``
    is given the expression is evaluated per group via ``.over(over)`` (the
    panel-safe path); when ``None`` it is applied to the whole frame. The result
    is written to ``alias`` if provided, otherwise it replaces ``column``.
    """
    if over is not None:
        expr = expr.over(over)
    expr = expr.alias(alias if alias is not None else column)
    return frame.with_columns(expr)


class PanelExprNamespace:
    """Expression methods registered under the ``.panel`` namespace.

    Instances are created by Polars when you access ``expr.panel``; you do not
    construct this class directly. Every method returns a :class:`polars.Expr`.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def frac_diff(self, d: float, *, threshold: float = 1e-5) -> pl.Expr:
        """Fixed-width fractional differencing (causal).

        Applies a fractional-difference filter of order ``d`` using a truncated,
        fixed-width weight kernel. Fractional differencing removes the
        unit-root / trend while preserving more memory than an integer
        difference — useful for making financial series stationary without
        destroying predictive signal.

        The implementation is a causal convolution: the value at row ``t`` is a
        weighted sum of ``x[t], x[t-1], ...`` with the weights from
        :func:`_frac_diff_weights`. No future rows are used. The first
        ``len(weights) - 1`` rows are ``null`` (insufficient history), matching
        the semantics of a rolling window.

        Parameters
        ----------
        d : float
            Order of fractional differencing (typically ``0 < d < 1``).
        threshold : float, keyword-only, default 1e-5
            Weight-magnitude cutoff controlling the kernel width. Smaller values
            yield a longer kernel (more memory, more leading nulls).

        Returns
        -------
        pl.Expr
            The fractionally differenced series. Combine with ``.over(entity)``
            to apply per entity in a panel.

        Notes
        -----
        Because the kernel is fixed-width and applied via lagged shifts, the
        expression is fully causal and therefore leakage-safe.
        """
        return _expr_frac_diff(self._expr, d, threshold=threshold)

    def zscore(self, window: int) -> pl.Expr:
        """Causal rolling z-score over a trailing ``window``.

        Computes ``(x[t] - rolling_mean) / rolling_std`` using only the trailing
        ``window`` observations (inclusive of ``t``). Leading rows with fewer
        than ``window`` observations are ``null``.

        Parameters
        ----------
        window : int
            Trailing window length in rows; must be a positive integer.

        Returns
        -------
        pl.Expr
            The rolling z-score. Combine with ``.over(entity)`` for panels.
        """
        return _expr_zscore(self._expr, window)

    def rs_vol(self, window: int) -> pl.Expr:
        """Trailing realized volatility over ``window`` (single-series proxy).

        True Rogers-Satchell volatility is an OHLC estimator and requires four
        columns (open, high, low, close); a single :class:`polars.Expr` cannot
        provide those. To stay honest, this method computes a **causal,
        single-series proxy**: the trailing standard deviation of the input over
        ``window`` rows (treat the input as a return/price-change series).

        For the genuine Rogers-Satchell OHLC estimator, use a dedicated
        frame-shaped operator that takes all four price columns.

        Parameters
        ----------
        window : int
            Trailing window length in rows; must be a positive integer.

        Returns
        -------
        pl.Expr
            Rolling standard deviation (volatility proxy). Combine with
            ``.over(entity)`` for panels.
        """
        return _expr_rs_vol(self._expr, window)


class _PanelFrameNamespace:
    """Shared frame-level ``.panel`` operators for LazyFrame and DataFrame.

    Each method names the target ``column`` as its first argument, takes an
    ``over`` entity key (defaulting to ``None`` = no grouping, but grouping is
    strongly encouraged for panels), an optional ``alias`` (output column name;
    defaults to replacing ``column``), and the same operator params as the
    expression-namespace method. The expression itself is built by the shared
    ``_expr_*`` helpers, so there is no logic duplicated with the expression
    namespace.

    LazyFrame and DataFrame both expose ``with_columns`` with identical
    semantics, so a single implementation serves both via :func:`_apply_over`.
    """

    _frame: pl.LazyFrame | pl.DataFrame

    def frac_diff(
        self,
        column: str,
        *,
        d: float,
        over: str | None = None,
        alias: str | None = None,
        threshold: float = 1e-5,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Fixed-width fractional differencing of ``column`` (causal).

        Parameters
        ----------
        column : str
            Name of the column to fractionally difference.
        d : float, keyword-only
            Order of fractional differencing (typically ``0 < d < 1``).
        over : str | None, keyword-only, default None
            Entity key to group by (``.over(over)``). ``None`` applies the
            operator to the whole frame; pass the entity column for panels.
        alias : str | None, keyword-only, default None
            Output column name. Defaults to replacing ``column`` in place.
        threshold : float, keyword-only, default 1e-5
            Weight-magnitude cutoff controlling the kernel width.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The frame with the new/replaced column (same type as the input).
        """
        expr = _expr_frac_diff(pl.col(column), d, threshold=threshold)
        return _apply_over(self._frame, expr, column, over=over, alias=alias)

    def zscore(
        self,
        column: str,
        *,
        window: int,
        over: str | None = None,
        alias: str | None = None,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Causal rolling z-score of ``column`` over a trailing ``window``.

        Parameters
        ----------
        column : str
            Name of the column to z-score.
        window : int, keyword-only
            Trailing window length in rows; must be a positive integer.
        over : str | None, keyword-only, default None
            Entity key to group by. ``None`` applies to the whole frame.
        alias : str | None, keyword-only, default None
            Output column name. Defaults to replacing ``column`` in place.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The frame with the new/replaced column.
        """
        expr = _expr_zscore(pl.col(column), window)
        return _apply_over(self._frame, expr, column, over=over, alias=alias)

    def rs_vol(
        self,
        column: str,
        *,
        window: int,
        over: str | None = None,
        alias: str | None = None,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Trailing realized-volatility proxy of ``column`` over ``window``.

        Parameters
        ----------
        column : str
            Name of the column whose trailing volatility is computed.
        window : int, keyword-only
            Trailing window length in rows; must be a positive integer.
        over : str | None, keyword-only, default None
            Entity key to group by. ``None`` applies to the whole frame.
        alias : str | None, keyword-only, default None
            Output column name. Defaults to replacing ``column`` in place.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The frame with the new/replaced column.
        """
        expr = _expr_rs_vol(pl.col(column), window)
        return _apply_over(self._frame, expr, column, over=over, alias=alias)


class PanelLazyFrameNamespace(_PanelFrameNamespace):
    """Frame-level ``.panel`` operators on :class:`polars.LazyFrame`.

    Accessed as ``lf.panel.<op>(...)``. Returns a new :class:`polars.LazyFrame`.
    """

    def __init__(self, lf: pl.LazyFrame) -> None:
        self._frame = lf


class PanelDataFrameNamespace(_PanelFrameNamespace):
    """Frame-level ``.panel`` operators on :class:`polars.DataFrame`.

    Accessed as ``df.panel.<op>(...)``. Returns a new :class:`polars.DataFrame`.
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._frame = df


def register() -> None:
    """Register the ``.panel`` namespace and its feature specs (idempotent).

    Registration is guarded so that re-importing this module (a common
    occurrence under reloaders, test runners, or repeated imports) does not
    raise the Polars "namespace already registered" error.
    """
    # Polars exposes custom expression namespaces as attributes on ``pl.Expr``.
    # Re-registering an existing name triggers a noisy "overriding existing
    # custom namespace" UserWarning, so only register when absent — this makes
    # re-import a true no-op. The try/except is a belt-and-braces guard for the
    # documented "already registered" condition across Polars versions.
    if not _is_registered(pl.Expr, "panel"):
        try:
            pl.api.register_expr_namespace("panel")(PanelExprNamespace)
        except Exception as exc:  # pragma: no cover - exact type varies by version
            if "already" not in str(exc).lower():
                raise

    # Frame-level ergonomics: ``lf.panel.<op>`` / ``df.panel.<op>``. Guarded the
    # same way to keep re-import a no-op and avoid the override UserWarning.
    if not _is_registered(pl.LazyFrame, "panel"):
        try:
            pl.api.register_lazyframe_namespace("panel")(PanelLazyFrameNamespace)
        except Exception as exc:  # pragma: no cover - exact type varies by version
            if "already" not in str(exc).lower():
                raise
    if not _is_registered(pl.DataFrame, "panel"):
        try:
            pl.api.register_dataframe_namespace("panel")(PanelDataFrameNamespace)
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
        name="frac_diff",
        namespace="panel",
        input_shape="series",
        output_shape="series",
        params={"d": float, "threshold": float},
        tier="A",
        panel_safe=True,
        leakage_safe=True,
        source=_SOURCE,
        license=_LICENSE,
    ),
    FeatureSpec(
        name="zscore",
        namespace="panel",
        input_shape="series",
        output_shape="series",
        params={"window": int},
        tier="A",
        panel_safe=True,
        leakage_safe=True,
        source=_SOURCE,
        license=_LICENSE,
    ),
    FeatureSpec(
        name="rs_vol",
        namespace="panel",
        input_shape="series",
        output_shape="series",
        params={"window": int},
        tier="B",
        panel_safe=True,
        leakage_safe=True,
        source=_SOURCE,
        license=_LICENSE,
    ),
)


register()
