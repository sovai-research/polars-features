"""Shared vocabulary for the point-in-time compiler and the leakage metric.

This module is the contract every other module in :mod:`panelary.leakage`
codes against. It holds no logic beyond :func:`node_kind`: the rule table
lives in ``_rules.py``, the walker in ``_compile.py``, the metric in
``_borrowed.py``.

The compiler works on the **serialised Polars expression tree**
(``Expr.meta.serialize(format="json")``), rewrites it as JSON, and reads it
back with ``pl.Expr.deserialize``. That round trip is what makes a rewrite
possible at all, and it is the one external dependency of this subpackage
that Polars does not promise to keep stable -- see
:data:`POLARS_TREE_FORMAT_TESTED`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "Classification",
    "CompileResult",
    "Context",
    "Finding",
    "LeakageRefused",
    "POLARS_TREE_FORMAT_TESTED",
    "Rule",
    "Verdict",
    "node_kind",
]

#: Polars minor versions whose serialised expression format this rule table has
#: been checked against. The format is NOT a stable public API, so the compiler
#: fails closed on anything it does not recognise and
#: ``tests/test_leakage_rules.py`` pins the shapes with golden trees.
POLARS_TREE_FORMAT_TESTED: tuple[str, ...] = ("1.44",)


class Classification(str, Enum):
    """What the rule table says about one node."""

    SAFE = "safe"
    """Output at ``t`` cannot depend on data after ``t``. Leave it alone."""

    REWRITE = "rewrite"
    """Leaks as written, but has an exact point-in-time equivalent."""

    REFUSE = "refuse"
    """Leaks with no causal equivalent, or is opaque. Fail closed."""


class Verdict(str, Enum):
    """The outcome for a whole expression."""

    SAFE = "safe"
    REWRITTEN = "rewritten"
    REFUSED = "refused"


class LeakageRefused(Exception):
    """Raised when an expression cannot be compiled to a point-in-time form.

    Carries the :class:`CompileResult` so callers can inspect every finding
    rather than just the message.
    """

    def __init__(self, message: str, result: CompileResult) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class Context:
    """Panel keys and options the rules need in order to decide.

    Parameters
    ----------
    time, entity : str, optional
        Panel key columns. ``time`` is required to repair an ``.over(entity)``
        that carries no ``order_by``: without it the compiler cannot know which
        order "within entity" means, and must refuse instead of guess.
    allow_approximate : bool, default=False
        Permit rewrites that are causal but not numerically identical to the
        leaky original (an expanding quantile, say). Off by default so that a
        rewrite never silently changes results.
    """

    time: str | None = None
    entity: str | None = None
    allow_approximate: bool = False


@dataclass(frozen=True)
class Finding:
    """One decision the compiler made, at one position in the tree."""

    kind: str
    """Qualified node kind, as returned by :func:`node_kind`."""

    classification: Classification
    reason: str
    """Why, in a sentence a user can act on."""

    path: tuple[str | int, ...] = ()
    """Position in the serialised tree, for pointing at the offending node."""

    rewrote_to: str | None = None
    """Short description of the replacement, when ``REWRITE``."""

    def __str__(self) -> str:  # pragma: no cover - display only
        where = ".".join(str(p) for p in self.path) or "<root>"
        tail = f" -> {self.rewrote_to}" if self.rewrote_to else ""
        return f"{where}: {self.kind} [{self.classification.value}] {self.reason}{tail}"


@dataclass(frozen=True)
class CompileResult:
    """The result of auditing or compiling one expression."""

    verdict: Verdict
    findings: tuple[Finding, ...] = ()
    expr: Any | None = None
    """The rewritten :class:`polars.Expr`; ``None`` when refused."""

    @property
    def refused(self) -> tuple[Finding, ...]:
        return tuple(
            f for f in self.findings if f.classification is Classification.REFUSE
        )

    @property
    def rewritten(self) -> tuple[Finding, ...]:
        return tuple(
            f for f in self.findings if f.classification is Classification.REWRITE
        )

    def raise_if_refused(self) -> Any:
        """Return the compiled expression, or raise :class:`LeakageRefused`."""
        if self.verdict is Verdict.REFUSED:
            detail = "\n  ".join(str(f) for f in self.refused)
            raise LeakageRefused(
                f"expression cannot be made point-in-time:\n  {detail}", self
            )
        return self.expr


@dataclass(frozen=True)
class Rule:
    """How to treat one kind of node.

    ``classify`` and ``rewrite`` both receive the node's *payload* (the value
    under the discriminator key) and the :class:`Context`. ``rewrite`` is only
    called when ``classify`` returned :data:`Classification.REWRITE`, and it
    returns a **complete replacement node**, not a payload -- that is, a
    single-key dict such as ``{"Function": {...}}``.

    Returning a node rather than a payload is deliberate: a rewrite is allowed
    to change the node's *kind*, which is what makes the valuable rewrites
    expressible at all. A whole-column ``Agg.Mean`` becomes an expanding
    ``BinaryExpr`` over ``cum_sum`` and a running count; no payload swap under
    the original discriminator could say that. The walker validates the return
    with :func:`node_kind` and refuses if it is not a node, so a rule that
    returns a bare payload fails loudly instead of producing a double-wrapped
    tree that Polars rejects with an opaque deserialisation error.
    """

    kind: str
    classify: Callable[[Any, Context], Classification]
    reason: str
    rewrite: Callable[[Any, Context], Any] | None = None
    rewrote_to: str | None = None
    children: tuple[str, ...] = field(default_factory=tuple)
    """Payload keys holding child nodes (a value or a list of values)."""


def node_kind(node: Any) -> str | None:
    """Qualified kind of a serialised expression node, or ``None``.

    Polars encodes a node as a single-key dict. Most kinds are the key itself
    (``Column``, ``Over``, ``BinaryExpr``), but ``Function`` and ``Agg`` carry
    the real identity one level down, so those are qualified:

    >>> node_kind({"Column": "x"})
    'Column'
    >>> node_kind({"Agg": {"Mean": {"Column": "x"}}})
    'Agg.Mean'
    >>> node_kind({"Function": {"input": [], "function": "Shift"}})
    'Function.Shift'
    >>> node_kind({"Function": {"input": [], "function": {"RollingExpr": {}}}})
    'Function.RollingExpr'
    >>> node_kind("not a node") is None
    True
    """
    if not isinstance(node, dict) or len(node) != 1:
        return None
    key = next(iter(node))
    payload = node[key]
    if key in ("Function", "AnonymousFunction"):
        fn = payload.get("function") if isinstance(payload, dict) else None
        if isinstance(fn, str):
            return f"{key}.{fn}"
        if isinstance(fn, dict) and len(fn) == 1:
            return f"{key}.{next(iter(fn))}"
        return key
    if key == "Agg" and isinstance(payload, dict) and len(payload) == 1:
        return f"Agg.{next(iter(payload))}"
    return key
