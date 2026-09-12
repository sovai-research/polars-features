"""The point-in-time compiler: walk a serialised Polars expression and fix it.

Two entry points, both operating on one :class:`polars.Expr`:

``audit``
    Never raises because of a leak. It returns a :class:`CompileResult` holding
    one :class:`Finding` per non-trivial decision, so a caller can report every
    problem in an expression rather than only the first.
``causalize``
    ``audit`` plus :meth:`CompileResult.raise_if_refused`, i.e. the strict form
    that hands back a point-in-time expression or explains why it cannot.

How it works
------------
``Expr.meta.serialize(format="json")`` renders the whole tree, and
``pl.Expr.deserialize`` reads a mutated one back. The walker therefore works on
plain JSON: it computes each node's kind with :func:`~panelary.leakage._types.node_kind`,
looks the kind up in ``_rules.RULES``, recurses **only** into the payload keys
the matching :class:`~panelary.leakage._types.Rule` names in ``children`` (so an
options dict can never be mistaken for a node), and applies rewrites bottom-up
so that a parent rule classifies a payload whose children are already causal.

Fail closed
-----------
The serialised format is not a stable Polars API, so every branch on which the
compiler cannot *prove* safety produces :data:`Classification.REFUSE`:

* a kind with no entry in the rule table;
* a value in a declared child slot that is not an expression node;
* a rule whose ``classify``/``rewrite`` raises, or returns a non-classification;
* a rule that classifies ``REWRITE`` but supplies no ``rewrite`` callable.

There is no path on which an unrecognised node is reported as ``SAFE``.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from ._types import (
    POLARS_TREE_FORMAT_TESTED,
    Classification,
    CompileResult,
    Context,
    Finding,
    Rule,
    Verdict,
    node_kind,
)

__all__ = ["audit", "causalize"]

#: Reported as the ``kind`` of a finding when a slot a rule declared to hold a
#: child expression holds something that is not an expression node at all.
NOT_A_NODE = "<not-an-expression-node>"


def _rule_table() -> Mapping[str, Rule]:
    """The live rule table, imported lazily.

    Indirection on purpose: it keeps the import out of module scope (so the
    walker can be exercised against a fixture table) and gives tests a single
    seam to monkeypatch.
    """
    from ._rules import RULES

    return RULES


def _format_error(action: str, exc: Exception) -> RuntimeError:
    return RuntimeError(
        f"could not {action} the Polars expression tree with polars "
        f"{pl.__version__}: {type(exc).__name__}: {exc}. The serialised "
        "expression format is not a stable Polars API; this compiler is "
        f"tested against POLARS_TREE_FORMAT_TESTED="
        f"{POLARS_TREE_FORMAT_TESTED}. Pin a tested Polars version or extend "
        "panelary.leakage._rules; the expression has NOT been made "
        "point-in-time."
    )


def _to_tree(expr: pl.Expr) -> Any:
    try:
        return json.loads(expr.meta.serialize(format="json"))
    except Exception as exc:  # pragma: no cover - depends on Polars internals
        raise _format_error("serialise", exc) from exc


def _from_tree(tree: Any) -> pl.Expr:
    try:
        return pl.Expr.deserialize(io.StringIO(json.dumps(tree)), format="json")
    except Exception as exc:
        raise _format_error("deserialise", exc) from exc


def _brief(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _refuse(
    kind: str, reason: str, path: tuple[str | int, ...], findings: list[Finding]
) -> None:
    findings.append(
        Finding(
            kind=kind,
            classification=Classification.REFUSE,
            reason=reason,
            path=path,
        )
    )


def _walk_slot(
    value: Any,
    rules: Mapping[str, Rule],
    ctx: Context,
    path: tuple[str | int, ...],
    findings: list[Finding],
) -> Any:
    """Walk one declared child slot, which holds a node or a list of nodes."""
    if value is None:
        return None
    if isinstance(value, list):
        out = list(value)
        for i, item in enumerate(out):
            # Heterogeneous lists are common (``Alias`` is ``[node, name]``,
            # ``Agg.Std`` is ``[node, ddof]``). Only real nodes are walked;
            # anything single-key that merely looks like one still reaches
            # ``_walk`` and is refused there, so this cannot open a hole.
            if node_kind(item) is not None:
                out[i] = _walk(item, rules, ctx, (*path, i), findings)
        return out
    return _walk(value, rules, ctx, path, findings)


def _walk_children(
    payload: Any,
    rule: Rule,
    rules: Mapping[str, Rule],
    ctx: Context,
    path: tuple[str | int, ...],
    findings: list[Finding],
) -> Any:
    """Return ``payload`` with every declared child slot walked (bottom-up)."""
    if not rule.children:
        return payload
    if isinstance(payload, Mapping):
        new_map = dict(payload)
        for key in rule.children:
            if key not in new_map:
                continue
            new_map[key] = _walk_slot(new_map[key], rules, ctx, (*path, key), findings)
        return new_map
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        new_seq = list(payload)
        for key in rule.children:
            if not (isinstance(key, str) and key.isdigit()):
                _refuse(
                    rule.kind,
                    f"rule declares child {key!r} but the payload is a sequence; "
                    "sequence payloads take positional child keys such as '0'",
                    path,
                    findings,
                )
                continue
            index = int(key)
            if index >= len(new_seq):
                _refuse(
                    rule.kind,
                    f"rule declares child at position {index} but the payload "
                    f"has only {len(new_seq)} element(s)",
                    path,
                    findings,
                )
                continue
            new_seq[index] = _walk_slot(
                new_seq[index], rules, ctx, (*path, index), findings
            )
        return new_seq
    _refuse(
        rule.kind,
        f"rule declares children {rule.children!r} but the payload is "
        f"{_brief(payload)}, which holds none",
        path,
        findings,
    )
    return payload


def _walk(
    node: Any,
    rules: Mapping[str, Rule],
    ctx: Context,
    path: tuple[str | int, ...],
    findings: list[Finding],
) -> Any:
    """Classify and possibly rewrite one node, children first."""
    kind = node_kind(node)
    if kind is None:
        _refuse(
            NOT_A_NODE,
            f"expected an expression node here, found {_brief(node)}",
            path,
            findings,
        )
        return node

    rule = rules.get(kind)
    if rule is None:
        # Invariant 1. Unknown means unproven, and unproven means refused.
        _refuse(
            kind,
            f"no rule for node kind {kind!r}, so it cannot be proven "
            "point-in-time (the compiler fails closed rather than guess)",
            path,
            findings,
        )
        return node

    key = next(iter(node))
    payload = _walk_children(node[key], rule, rules, ctx, path, findings)

    try:
        classification = rule.classify(payload, ctx)
    except Exception as exc:
        _refuse(
            kind,
            f"rule for {kind!r} raised while classifying: {type(exc).__name__}: {exc}",
            path,
            findings,
        )
        return {key: payload}

    if not isinstance(classification, Classification):
        _refuse(
            kind,
            f"rule for {kind!r} returned {_brief(classification)} instead of a "
            "Classification",
            path,
            findings,
        )
        return {key: payload}

    if classification is Classification.SAFE:
        return {key: payload}

    if classification is Classification.REFUSE:
        _refuse(kind, rule.reason, path, findings)
        return {key: payload}

    if rule.rewrite is None:
        _refuse(
            kind,
            f"rule for {kind!r} classified REWRITE but supplies no rewrite "
            f"({rule.reason})",
            path,
            findings,
        )
        return {key: payload}

    try:
        rewritten = rule.rewrite(payload, ctx)
    except Exception as exc:
        _refuse(
            kind,
            f"rule for {kind!r} raised while rewriting: {type(exc).__name__}: {exc}",
            path,
            findings,
        )
        return {key: payload}

    findings.append(
        Finding(
            kind=kind,
            classification=Classification.REWRITE,
            reason=rule.reason,
            path=path,
            rewrote_to=rule.rewrote_to,
        )
    )
    # `rewrite` returns a whole NODE, not a payload: a rewrite is allowed to
    # change the node's kind, which is what makes the valuable ones possible --
    # a whole-column `Agg.Mean` becomes an expanding `BinaryExpr`, and no
    # payload swap under the original discriminator could express that. Guard
    # the contract here rather than letting a malformed tree reach Polars,
    # where it surfaces as an opaque deserialisation error.
    if node_kind(rewritten) is None:
        _refuse(
            kind,
            f"rule for {kind!r} returned {rewritten!r}, which is not a node; "
            f"`Rule.rewrite` must return a complete node such as "
            f'{{"Column": "x"}}, not a bare payload',
            path,
            findings,
        )
        return {key: payload}
    return rewritten


def audit(
    expr: pl.Expr,
    *,
    time: str | None = None,
    entity: str | None = None,
    allow_approximate: bool = False,
) -> CompileResult:
    """Audit one expression for look-ahead, rewriting what can be rewritten.

    This never raises because an expression leaks: it reports. Every node the
    compiler could not prove safe produces a :class:`Finding` carrying the
    qualified node kind, the reason, and the ``path`` at which it sits, so a
    single call reports every problem rather than the first.

    Parameters
    ----------
    expr : polars.Expr
        The expression to audit.
    time, entity : str, optional
        Panel key columns. ``time`` is what lets the compiler repair an
        ``.over(entity)`` that carries no ``order_by``; without it such a node
        is refused rather than assumed sorted.
    allow_approximate : bool, default=False
        Permit rewrites that are causal but not numerically identical to the
        original.

    Returns
    -------
    CompileResult
        ``verdict`` is ``REFUSED`` if any finding refuses, else ``REWRITTEN``
        if any rewrote, else ``SAFE``. ``expr`` holds the compiled expression,
        and is ``None`` when refused. Only ``REWRITE`` and ``REFUSE``
        decisions are recorded; a node the table proves safe is silent.

    Raises
    ------
    TypeError
        If ``expr`` is not a :class:`polars.Expr`.
    RuntimeError
        If the serialised-tree round trip itself fails, which means this
        Polars version's format is not one the compiler was tested against.

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.leakage import audit
    >>> audit(pl.col("x").map_batches(lambda s: s)).verdict.value
    'refused'
    """
    if not isinstance(expr, pl.Expr):
        raise TypeError(f"audit() expects a polars.Expr, got {type(expr).__name__}")

    ctx = Context(time=time, entity=entity, allow_approximate=allow_approximate)
    findings: list[Finding] = []
    tree = _walk(_to_tree(expr), _rule_table(), ctx, (), findings)
    frozen = tuple(findings)

    if any(f.classification is Classification.REFUSE for f in frozen):
        return CompileResult(verdict=Verdict.REFUSED, findings=frozen, expr=None)

    verdict = (
        Verdict.REWRITTEN
        if any(f.classification is Classification.REWRITE for f in frozen)
        else Verdict.SAFE
    )
    # Deserialise even when nothing changed: it costs microseconds and keeps
    # the round trip on the tested path for every call, so format drift shows
    # up as a loud RuntimeError instead of a silently unrewritten expression.
    return CompileResult(verdict=verdict, findings=frozen, expr=_from_tree(tree))


def causalize(
    expr: pl.Expr,
    *,
    time: str | None = None,
    entity: str | None = None,
    allow_approximate: bool = False,
) -> pl.Expr:
    """Compile one expression to its point-in-time equivalent, or refuse.

    The strict form of :func:`audit`.

    Parameters
    ----------
    expr : polars.Expr
        The expression to compile.
    time, entity : str, optional
        Panel key columns; see :func:`audit`.
    allow_approximate : bool, default=False
        Permit rewrites that are causal but not numerically identical.

    Returns
    -------
    polars.Expr
        An expression whose value at ``t`` cannot depend on data after ``t``.

    Raises
    ------
    LeakageRefused
        If any node leaks with no causal equivalent. The exception carries the
        full :class:`CompileResult` on ``.result``.

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.leakage import causalize
    >>> expr = causalize(pl.col("x").shift(1))
    """
    result = audit(expr, time=time, entity=entity, allow_approximate=allow_approximate)
    compiled: pl.Expr = result.raise_if_refused()
    return compiled
