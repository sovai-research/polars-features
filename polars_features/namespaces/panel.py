"""Per-entity panel operators exposed under the ``.panel`` expression namespace.

The operators here are *time-series* transforms intended to be applied **within a
single entity** of a long-format ``(entity, time, *features)`` panel. They return
plain :class:`polars.Expr` objects and carry **no** grouping of their own — the
user composes them with ``.over(entity)`` so the same expression works on one
series or a million.

All operators in this namespace are **causal** (leakage-safe): the value at time
``t`` depends only on observations at or before ``t``. Combined with
``.over(entity)`` they are also panel-safe (no cross-entity bleed).

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

import polars as pl

from polars_features.registry import FeatureSpec, registry

__all__ = ["PanelExprNamespace", "register"]

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
        weights = _frac_diff_weights(d, threshold)
        width = len(weights)
        # Causal weighted sum: w[0]*x[t] + w[1]*x[t-1] + ...
        #
        # Build a *flat* n-ary sum via ``pl.sum_horizontal`` rather than folding
        # the terms with Python ``+``. A long kernel (small ``threshold`` / small
        # ``d``) can produce thousands of weights; a left-folded ``a + b + c +
        # ...`` produces a deeply nested expression tree that overflows the
        # Rust evaluation stack. ``sum_horizontal`` keeps the tree shallow.
        terms = [w * self._expr.shift(lag) for lag, w in enumerate(weights)]
        out = pl.sum_horizontal(terms)
        if width <= 1:
            return out
        # ``sum_horizontal`` treats shifted-in nulls as 0, which would emit a
        # value before the kernel has full history. Mask the leading rows that
        # lack ``width`` observations to ``null`` so the output matches
        # rolling-window semantics and stays strictly causal.
        row = pl.int_range(pl.len(), dtype=pl.Int64)
        return pl.when(row >= width - 1).then(out).otherwise(None)

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
        if not isinstance(window, int) or window <= 0:
            raise ValueError(
                f"zscore window must be a positive integer, got {window!r}."
            )
        mean = self._expr.rolling_mean(window_size=window)
        std = self._expr.rolling_std(window_size=window)
        return (self._expr - mean) / std

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
        if not isinstance(window, int) or window <= 0:
            raise ValueError(
                f"rs_vol window must be a positive integer, got {window!r}."
            )
        return self._expr.rolling_std(window_size=window)


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
    if not _is_registered("panel"):
        try:
            pl.api.register_expr_namespace("panel")(PanelExprNamespace)
        except Exception as exc:  # pragma: no cover - exact type varies by version
            if "already" not in str(exc).lower():
                raise

    for spec in _SPECS:
        registry.register(spec)


def _is_registered(name: str) -> bool:
    """Return ``True`` if a custom expression namespace ``name`` already exists."""
    return isinstance(getattr(pl.Expr, name, None), property) or hasattr(pl.Expr, name)


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
