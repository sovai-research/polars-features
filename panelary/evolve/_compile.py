"""Population compiler: genomes -> one layered, deduplicated Polars plan.

This module is the performance- *and* correctness-critical part of
:mod:`panelary.evolve`. It turns a whole population of
:class:`~panelary.evolve._types.Genome` straight-line programs into a
**single** :class:`polars.LazyFrame` plan, and it exists because three measured
properties of Polars 1.44.1 make the obvious implementation either wrong or
slow.

Why the design looks like this
------------------------------

**1. Nested partition scopes are silently wrong (correctness).**
``pl.col("x").rolling_mean(2).over("sym").rank().over("date")`` returns
**0 of 8** non-null values on an 8-row panel, with no error and no warning.
Materialising the inner result first and ranking the *column* returns the
correct 6 of 8. Alpha101-style expressions alternate partitions constantly
(``rank(ts_argmax(...))``), so this is not an edge case.

The rule this module enforces is therefore absolute: **no operator expression
is ever nested inside another operator's** ``.over()``. Every node that needs a
partition (``ts`` -> ``.over(entity)``, ``xs`` -> ``.over(time)``) is
materialised into its own column in its own ``with_columns`` stage, and its
consumers reference that column by name. Partition-scope switches cannot occur
inside one expression because scoped expressions are never composed at all.

**2. Polars' CSE does not deduplicate shared subtrees across a population.**
300 genomes sharing one ``rolling_mean(20).over(entity)`` subtree: 0.895 s
inline versus 0.082 s hoisted-and-reused, a **10.9x** difference. So the
compiler canonicalises every gene to a hashable key
``(op_name, param, tuple(child_keys))`` -- with arguments sorted for
commutative operators -- and builds **one global DAG across the entire
population**. Identical semantics collapse to one node, which is computed once.

**3. Expression *planning* time is quadratic in expression depth**
(depth 1600 = 1.62 s before touching any data; polars#16224), while batched
shallow expressions are linear and cheap (2000 depth-3 expressions over 200k
rows = 0.57 s). So the plan is emitted as a handful of wide, shallow stages --
one ``with_columns`` per ``(layer, scope)`` group, batching every node of that
group into a single call -- never as one deep expression per genome.

Layering
--------
Each node gets ``layer = 1 + max(layer of its materialised dependencies)``,
with base columns at layer 0. Because a materialised node is *strictly* above
every node it reads, a stage never references a column created in the same
stage, and a scope switch is always a stage boundary (the "bump" is implicit:
every scoped node already starts a new layer).

Inlining
--------
Full materialisation would create one column per DAG node. At 200k rows a
float64 column is 1.6 MB, so a 3000-node DAG would carry ~5 GB of
intermediates. Row-wise (``elem``) nodes are the numerous ones and the cheap
ones (~0.28 ms/expr against 18.4 ms for a cross-sectional op), so an ``elem``
node that is referenced exactly once and is not a requested output is
**inlined** into its consumer instead of being materialised. Inlining an
``elem`` node is provably safe with respect to hazard (1): row-wise expressions
carry no ``.over()``, so no nesting is created, and a row-wise transform is
invariant to the window it is evaluated in. ``ts`` and ``xs`` nodes, and any
node with fan-out >= 2, are always materialised -- that is where the 10.9x
lives. Set ``inline_elem=False`` to materialise everything (used by the test
suite to prove the two paths agree).

Sortedness
----------
Rolling windows read a group's rows in **frame order**, so an unsorted panel
makes every ``ts`` operator silently wrong. ``validate`` receives no frame and
therefore cannot check this, so the choice made here is that
**the compiler sorts defensively**: ``compile_population`` sorts by
``(entity, time)`` unless ``sort=False``. A consequence worth stating plainly:
the returned frame is in ``(entity, time)`` order, which need not be the input
row order, so consumers must read their target column out of the *returned*
frame rather than zipping the result against an externally ordered array.

Determinism
-----------
Intermediate column names are ``__ev_<blake2b16>`` over the canonical key, so
they are stable across processes (Python's ``hash`` is salted for ``str`` and
is never used here) and identical semantics always yield the identical name.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import polars as pl

from ._types import (
    DIMENSIONLESS,
    MAX_DEPTH_DEFAULT,
    XS_COST_RATIO,
    EvalContext,
    Genome,
    Op,
)

__all__ = [
    "compile_population",
    "dag_stats",
    "active_nodes",
    "depth",
    "validate",
    "semantic_key",
    "relative_cost",
]

#: Prefix for compiler-owned intermediate columns.
NAME_PREFIX: Final[str] = "__ev_"

#: Bytes of blake2b digest used in an intermediate column name (16 hex chars).
_NAME_DIGEST_BYTES: Final[int] = 8

#: Deterministic stage order within a layer.
_SCOPE_ORDER: Final[tuple[str, ...]] = ("elem", "ts", "xs")

#: Hard ceiling on how deep an inlined ``elem`` chain may grow before the node
#: is materialised instead. Keeps Polars' quadratic planner off the cliff even
#: if a caller hands us genomes deeper than ``MAX_DEPTH_DEFAULT``.
_MAX_INLINE_DEPTH: Final[int] = 8

#: Default subsample size for :func:`semantic_key`.
_SEMANTIC_SAMPLE: Final[int] = 2048

#: Significant digits retained by :func:`semantic_key`.
_SEMANTIC_SIGDIGITS: Final[int] = 6

_OPS_CACHE: dict[str, Mapping[str, Op]] = {}
_BUILD_STYLE_CACHE: dict[str, str] = {}


# --------------------------------------------------------------------------
# operator registry access
# --------------------------------------------------------------------------
def _default_ops() -> Mapping[str, Op]:
    """Return the shared operator registry, imported lazily.

    Imported inside the function rather than at module scope so that this
    module stays importable (and benchmarkable) against an injected ``ops``
    mapping, matching Panelary's lazy-import convention.
    """
    cached = _OPS_CACHE.get("ops")
    if cached is None:
        from ._ops import OPS  # local import: see docstring

        cached = OPS
        _OPS_CACHE["ops"] = cached
    return cached


def _resolve_ops(ops: Mapping[str, Op] | None) -> Mapping[str, Op]:
    return _default_ops() if ops is None else ops


def _lookup(ops: Mapping[str, Op], name: str) -> Op:
    try:
        return ops[name]
    except KeyError:  # pragma: no cover - defensive
        raise ValueError(f"unknown operator {name!r}; not in the grammar") from None


def relative_cost(op: Op) -> int:
    """Relative evaluation cost of ``op``, floored at the cross-sectional ratio.

    ``Op.cost`` is authored per operator, but a cross-sectional operator is
    measured at ~20x a time-series one on this repo, so an ``xs`` operator is
    never costed below :data:`~panelary.evolve._types.XS_COST_RATIO`.
    """
    if op.kind == "xs":
        return max(int(op.cost), XS_COST_RATIO)
    return int(op.cost)


def _build_style(op: Op) -> str:
    """How ``op.build`` wants its parameter: ``"none"``, ``"pos"`` or ``"kw"``."""
    style = _BUILD_STYLE_CACHE.get(op.name)
    if style is not None:
        return style
    if not op.params:
        style = "none"
    else:
        style = "pos"
        try:
            sig = inspect.signature(op.build)
        except (TypeError, ValueError):  # pragma: no cover - builtins/C callables
            sig = None
        if sig is not None:
            for p in sig.parameters.values():
                if p.name == "param" and p.kind is inspect.Parameter.KEYWORD_ONLY:
                    style = "kw"
                    break
    _BUILD_STYLE_CACHE[op.name] = style
    return style


def _call_build(op: Op, children: Sequence[pl.Expr], param: Any) -> pl.Expr:
    style = _build_style(op)
    if style == "none":
        return op.build(*children)
    if style == "kw":
        return op.build(*children, param=param)
    return op.build(*children, param)


def _param_of(op: Op, param_ix: int) -> Any:
    if not op.params:
        return None
    if not 0 <= param_ix < len(op.params):
        raise ValueError(
            f"op {op.name!r}: param_ix {param_ix} out of range "
            f"for {len(op.params)} parameter(s)"
        )
    return op.params[param_ix]


# --------------------------------------------------------------------------
# canonical keys
# --------------------------------------------------------------------------
#: ``(op_name, param, child_keys)``. ``op_name`` is ``"$base"`` for a base
#: column, in which case ``param`` is the column name.
Key = tuple[str, Any, tuple[Any, ...]]


def _key_repr(key: Key) -> str:
    """Canonical, process-stable string encoding of a node key."""
    op, param, children = key
    inner = ",".join(_key_repr(c) for c in children)
    return f"{op}|{param!r}|({inner})"


def _node_name(key: Key) -> str:
    digest = hashlib.blake2b(
        _key_repr(key).encode("utf-8"), digest_size=_NAME_DIGEST_BYTES
    ).hexdigest()
    return f"{NAME_PREFIX}{digest}"


# --------------------------------------------------------------------------
# the DAG
# --------------------------------------------------------------------------
@dataclass(slots=True)
class _Node:
    """One deduplicated node of the population-wide DAG."""

    nid: int
    key: Key
    op: Op | None  # None <=> base column
    param: Any
    children: tuple[int, ...]
    scope: str  # "elem" | "ts" | "xs"
    base_col: str | None = None
    name: str = ""
    depth: int = 0
    fan_out: int = 0
    is_output: bool = False
    materialised: bool = True
    inline_depth: int = 0
    deps: frozenset[int] = frozenset()
    layer: int = 0

    @property
    def is_base(self) -> bool:
        return self.op is None


@dataclass(slots=True)
class _Dag:
    """The population-wide DAG plus its stage schedule."""

    nodes: list[_Node] = field(default_factory=list)
    index: dict[Key, int] = field(default_factory=dict)
    outputs: list[int] = field(default_factory=list)  # one node id per genome
    n_active_genes: int = 0
    #: ``(layer, scope) -> node ids``, in emission order.
    stages: list[tuple[int, str, list[int]]] = field(default_factory=list)

    def _intern(self, node: _Node) -> int:
        existing = self.index.get(node.key)
        if existing is not None:
            return existing
        node.nid = len(self.nodes)
        node.name = node.base_col if node.is_base else _node_name(node.key)
        self.nodes.append(node)
        self.index[node.key] = node.nid
        return node.nid

    def add_base(self, col: str) -> int:
        key: Key = ("$base", col, ())
        return self._intern(
            _Node(
                nid=-1,
                key=key,
                op=None,
                param=None,
                children=(),
                scope="elem",
                base_col=col,
                depth=0,
            )
        )

    def add_op(self, op: Op, param: Any, children: Sequence[int]) -> int:
        child_keys = [self.nodes[c].key for c in children]
        kids = list(children)
        if op.commutative and len(kids) > 1:
            order = sorted(range(len(kids)), key=lambda i: _key_repr(child_keys[i]))
            kids = [kids[i] for i in order]
            child_keys = [child_keys[i] for i in order]
        key: Key = (op.name, param, tuple(child_keys))
        node = _Node(
            nid=-1,
            key=key,
            op=op,
            param=param,
            children=tuple(kids),
            scope=op.kind,
            depth=1 + max((self.nodes[c].depth for c in kids), default=0),
        )
        return self._intern(node)


def _slot_units(genome: Genome, ctx: EvalContext, ops: Mapping[str, Op]) -> list[str]:
    units: list[str] = list(ctx.base_units[: genome.n_base])
    for gene in genome.genes:
        units.append(_lookup(ops, gene.op).out_unit)
    return units


def _output_gene(genome: Genome) -> int:
    """Index of the output gene.

    ``Genome.output()`` returns a **gene index** (its default is
    ``len(genes) - 1``), not a slot index; slot ``n_base + i`` is gene ``i``.
    """
    if not genome.genes:
        raise ValueError("genome has no genes; nothing to compile")
    out = genome.output()
    if not 0 <= out < len(genome.genes):
        raise ValueError(
            f"genome out_slot resolves to gene {out}, outside 0..{len(genome.genes) - 1}"
        )
    return out


def _build_dag(
    genomes: Sequence[Genome],
    ctx: EvalContext,
    ops: Mapping[str, Op],
    *,
    inline_elem: bool,
) -> _Dag:
    dag = _Dag()

    for genome in genomes:
        if genome.n_base > len(ctx.base_columns):
            raise ValueError(
                f"genome.n_base={genome.n_base} exceeds "
                f"{len(ctx.base_columns)} base columns in the context"
            )
        out_gene = _output_gene(genome)
        active = _active_gene_indices(genome, out_gene)
        dag.n_active_genes += len(active)

        # slot -> node id, filled in program order for active genes only.
        slot_node: dict[int, int] = {}
        for i in range(genome.n_base):
            slot_node[i] = dag.add_base(ctx.base_columns[i])
        for gi in sorted(active):
            gene = genome.genes[gi]
            op = _lookup(ops, gene.op)
            if len(gene.args) != op.arity:
                raise ValueError(
                    f"gene {gi} ({gene.op!r}): arity {op.arity} but "
                    f"{len(gene.args)} argument(s)"
                )
            child_ids = []
            for slot in gene.args:
                if slot not in slot_node:
                    raise ValueError(
                        f"gene {gi} ({gene.op!r}) references slot {slot}, "
                        "which is not a base column nor an earlier gene"
                    )
                child_ids.append(slot_node[slot])
            param = _param_of(op, gene.param_ix)
            slot_node[genome.n_base + gi] = dag.add_op(op, param, child_ids)

        out_id = slot_node[genome.n_base + out_gene]
        dag.outputs.append(out_id)
        dag.nodes[out_id].is_output = True

    _schedule(dag, inline_elem=inline_elem)
    return dag


def _schedule(dag: _Dag, *, inline_elem: bool) -> None:
    """Decide materialisation, assign ``(layer, scope)``, build the stages."""
    for node in dag.nodes:
        for child in node.children:
            dag.nodes[child].fan_out += 1

    for node in dag.nodes:
        if node.is_base:
            node.materialised = True
            node.layer = 0
            node.deps = frozenset({node.nid})
            node.inline_depth = 0
            continue

        deps: set[int] = set()
        inline_depth = 1
        for child in node.children:
            kid = dag.nodes[child]
            if kid.materialised:
                deps.add(child)
            else:
                deps |= kid.deps
                inline_depth = max(inline_depth, kid.inline_depth + 1)

        inlinable = (
            inline_elem
            and node.scope == "elem"
            and not node.is_output
            and node.fan_out <= 1
            and inline_depth <= _MAX_INLINE_DEPTH
        )
        node.inline_depth = inline_depth
        if inlinable:
            node.materialised = False
            node.deps = frozenset(deps)
            # Layer is informational for an inlined node; its consumers read
            # `deps` directly.
            node.layer = max((dag.nodes[d].layer for d in deps), default=0)
        else:
            node.materialised = True
            node.deps = frozenset({node.nid})
            node.layer = 1 + max((dag.nodes[d].layer for d in deps), default=0)

    buckets: dict[tuple[int, str], list[int]] = {}
    for node in dag.nodes:
        if node.is_base or not node.materialised:
            continue
        buckets.setdefault((node.layer, node.scope), []).append(node.nid)

    def stage_order(k: tuple[int, str]) -> tuple[int, int]:
        return (k[0], _SCOPE_ORDER.index(k[1]))

    dag.stages = [
        (layer, scope, buckets[(layer, scope)])
        for layer, scope in sorted(buckets, key=stage_order)
    ]


def _active_gene_indices(genome: Genome, out_gene: int) -> set[int]:
    active: set[int] = set()
    stack = [out_gene]
    while stack:
        gi = stack.pop()
        if gi in active:
            continue
        active.add(gi)
        for slot in genome.genes[gi].args:
            if slot >= genome.n_base:
                child = slot - genome.n_base
                if not 0 <= child < gi:
                    raise ValueError(
                        f"gene {gi} references slot {slot}; a gene may only read "
                        "base columns and strictly earlier genes"
                    )
                stack.append(child)
            elif slot < 0:
                raise ValueError(f"gene {gi} references negative slot {slot}")
    return active


# --------------------------------------------------------------------------
# expression emission
# --------------------------------------------------------------------------
def _base_expr_map(
    lf: pl.LazyFrame, ctx: EvalContext, needed: Iterable[str]
) -> dict[str, pl.Expr]:
    """Column reference per base column, upcast to Float64 where required."""
    schema = lf.collect_schema()
    missing = [c for c in (ctx.entity, ctx.time) if c not in schema]
    if missing:
        raise ValueError(f"frame is missing panel key column(s): {missing}")
    out: dict[str, pl.Expr] = {}
    for col in needed:
        if col not in schema:
            raise ValueError(f"frame is missing base column {col!r}")
        dtype = schema[col]
        if dtype == pl.Float64:
            out[col] = pl.col(col)
        elif dtype.is_numeric() or dtype == pl.Boolean:
            out[col] = pl.col(col).cast(pl.Float64)
        else:
            raise ValueError(
                f"base column {col!r} has non-numeric dtype {dtype}; "
                "feature columns must be numeric"
            )
    return out


def _node_expr(dag: _Dag, nid: int, base_expr: Mapping[str, pl.Expr]) -> pl.Expr:
    """Expression *body* of a node: children are refs, no ``.over()`` applied."""
    node = dag.nodes[nid]
    assert node.op is not None
    children = [_ref_expr(dag, c, base_expr) for c in node.children]
    return _call_build(node.op, children, node.param)


def _ref_expr(dag: _Dag, nid: int, base_expr: Mapping[str, pl.Expr]) -> pl.Expr:
    node = dag.nodes[nid]
    if node.is_base:
        assert node.base_col is not None
        return base_expr[node.base_col]
    if node.materialised:
        return pl.col(node.name)
    return _node_expr(dag, nid, base_expr)


def _scoped(expr: pl.Expr, scope: str, ctx: EvalContext) -> pl.Expr:
    if scope == "ts":
        return expr.over(ctx.entity)
    if scope == "xs":
        return expr.over(ctx.time)
    return expr


def compile_population(
    genomes: Sequence[Genome],
    ctx: EvalContext,
    lf: pl.LazyFrame,
    *,
    ops: Mapping[str, Op] | None = None,
    drop_intermediates: bool = True,
    keep: Sequence[str] = (),
    inline_elem: bool = True,
    sort: bool = True,
) -> tuple[pl.LazyFrame, list[str]]:
    """Compile a whole population into one layered, deduplicated lazy plan.

    Every genome is canonicalised into a population-wide DAG, so semantically
    identical subexpressions are computed exactly once, and the DAG is emitted
    as one ``with_columns`` per ``(layer, scope)`` group. A node that needs a
    partition (``ts`` -> ``.over(entity)``, ``xs`` -> ``.over(time)``) always
    lands in its own stage, which is what makes alternating partition scopes
    correct -- see the module docstring for the measurement.

    Parameters
    ----------
    genomes
        Population to compile. May contain duplicates.
    ctx
        Panel context: base columns and units, entity/time key names.
    lf
        Long-format panel. Must contain ``ctx.entity``, ``ctx.time`` and every
        base column referenced by the population.
    ops
        Operator registry. Defaults to :data:`panelary.evolve._ops.OPS`.
    drop_intermediates
        If True (default) the returned frame carries only the panel keys,
        ``keep``, and the population outputs. Intermediates are still computed;
        they are simply projected away at the end.
    keep
        Extra input columns (e.g. a forward-return target) to retain when
        ``drop_intermediates`` is True.
    inline_elem
        If True (default) a row-wise node used exactly once and not itself an
        output is folded into its consumer instead of getting its own column.
        This is safe by construction -- row-wise expressions carry no
        ``.over()`` -- and it is what keeps the intermediate frame narrow.
    sort
        If True (default) sort defensively by ``(entity, time)`` first. Rolling
        windows read a group in frame order, so an unsorted panel is silently
        wrong. The returned frame is then in ``(entity, time)`` order.

    Returns
    -------
    (lazyframe, names)
        ``names[i]`` is the output column of ``genomes[i]``. Two semantically
        identical genomes share one column, so ``names`` **may contain
        duplicates** -- that is the deduplication working. Index by name;
        do not ``select(names)`` without de-duplicating first.

    Raises
    ------
    ValueError
        On an unknown operator, an arity/slot/parameter violation, a missing or
        non-numeric column, or an empty population.
    """
    if not genomes:
        raise ValueError("cannot compile an empty population")
    registry = _resolve_ops(ops)
    dag = _build_dag(genomes, ctx, registry, inline_elem=inline_elem)

    needed_bases = {n.base_col for n in dag.nodes if n.is_base and n.base_col}
    base_expr = _base_expr_map(lf, ctx, sorted(needed_bases))

    out = lf.sort(ctx.entity, ctx.time, maintain_order=True) if sort else lf
    for _layer, scope, nids in dag.stages:
        exprs = [
            _scoped(_node_expr(dag, nid, base_expr), scope, ctx).alias(
                dag.nodes[nid].name
            )
            for nid in nids
        ]
        out = out.with_columns(exprs)

    names = [dag.nodes[nid].name for nid in dag.outputs]
    if drop_intermediates:
        wanted = list(dict.fromkeys([ctx.entity, ctx.time, *keep, *names]))
        out = out.select(wanted)
    return out, names


# --------------------------------------------------------------------------
# introspection
# --------------------------------------------------------------------------
def active_nodes(genome: Genome) -> frozenset[int]:
    """Indices of the genes reachable from the output gene.

    Returns *gene* indices (gene ``i`` occupies slot ``n_base + i``). Inactive
    genes are dead code: they cost nothing to evaluate and must not count
    towards complexity.
    """
    if not genome.genes:
        return frozenset()
    return frozenset(_active_gene_indices(genome, _output_gene(genome)))


def depth(genome: Genome) -> int:
    """Longest operator chain from the output back to a base column.

    A genome whose output reads base columns directly has depth 1; a base
    column on its own has depth 0.
    """
    if not genome.genes:
        return 0
    out_gene = _output_gene(genome)
    memo: dict[int, int] = {}

    def slot_depth(slot: int) -> int:
        if slot < genome.n_base:
            return 0
        gi = slot - genome.n_base
        cached = memo.get(gi)
        if cached is not None:
            return cached
        gene = genome.genes[gi]
        d = 1 + max((slot_depth(a) for a in gene.args), default=0)
        memo[gi] = d
        return d

    return slot_depth(genome.n_base + out_gene)


def _units_compatible(required: str, actual: str) -> bool:
    if required == "any" or actual == "any" or required == actual:
        return True
    return required in DIMENSIONLESS and actual in DIMENSIONLESS


def validate(
    genome: Genome,
    ctx: EvalContext,
    *,
    max_depth: int = MAX_DEPTH_DEFAULT,
    ops: Mapping[str, Op] | None = None,
) -> None:
    """Raise :class:`ValueError` unless ``genome`` is compilable under ``ctx``.

    Checks structure (arity, acyclicity via strictly-earlier slot references,
    parameter index bounds, output slot), dimensional typing of every operator
    argument, and the depth cap. Leak-safety needs no check here: an operator
    cannot enter the registry at all unless ``leakage_safe`` is True
    (:meth:`Op.__post_init__`) and ``.over()`` is applied only by this
    compiler.

    Panel sortedness is deliberately *not* checked here -- ``validate`` never
    sees a frame. :func:`compile_population` sorts defensively instead.
    """
    registry = _resolve_ops(ops)
    n_base = genome.n_base
    if n_base <= 0:
        raise ValueError("genome.n_base must be positive")
    if n_base > len(ctx.base_columns):
        raise ValueError(
            f"genome.n_base={n_base} exceeds {len(ctx.base_columns)} base columns"
        )
    if len(ctx.base_units) != len(ctx.base_columns):
        raise ValueError("EvalContext base_columns and base_units differ in length")
    if not genome.genes:
        raise ValueError("genome has no genes")

    out_gene = _output_gene(genome)
    units = _slot_units(genome, ctx, registry)

    for gi, gene in enumerate(genome.genes):
        op = _lookup(registry, gene.op)
        if len(gene.args) != op.arity:
            raise ValueError(
                f"gene {gi} ({gene.op!r}): arity {op.arity} but "
                f"{len(gene.args)} argument(s)"
            )
        _param_of(op, gene.param_ix)  # raises on an out-of-range param_ix
        limit = n_base + gi
        for pos, slot in enumerate(gene.args):
            if not 0 <= slot < limit:
                raise ValueError(
                    f"gene {gi} ({gene.op!r}) argument {pos} reads slot {slot}; "
                    f"must be in 0..{limit - 1} (base columns and earlier genes)"
                )
            if not _units_compatible(op.in_units[pos], units[slot]):
                raise ValueError(
                    f"gene {gi} ({gene.op!r}) argument {pos} expects unit "
                    f"{op.in_units[pos]!r} but slot {slot} carries {units[slot]!r}"
                )

    if genome.n_base > 0 and out_gene >= len(genome.genes):  # pragma: no cover
        raise ValueError("output gene out of range")

    d = depth(genome)
    if d > max_depth:
        raise ValueError(f"genome depth {d} exceeds max_depth {max_depth}")


def dag_stats(
    genomes: Sequence[Genome],
    ctx: EvalContext,
    *,
    ops: Mapping[str, Op] | None = None,
    inline_elem: bool = True,
) -> dict[str, int | float]:
    """Summarise the compiled DAG without touching any data.

    Useful for budgeting a generation before evaluating it: ``reuse_factor``
    is the multiple of work that deduplication saves, ``n_materialised`` is the
    number of intermediate columns the plan will carry, and ``est_cost`` is the
    relative compute in units of one row-wise operator
    (:func:`relative_cost`, which floors cross-sectional operators at
    ``XS_COST_RATIO``).
    """
    if not genomes:
        raise ValueError("cannot summarise an empty population")
    registry = _resolve_ops(ops)
    dag = _build_dag(genomes, ctx, registry, inline_elem=inline_elem)

    op_nodes = [n for n in dag.nodes if not n.is_base]
    n_base = len(dag.nodes) - len(op_nodes)
    n_nodes = len(op_nodes)
    n_mat = sum(1 for n in op_nodes if n.materialised)
    depths = [depth(g) for g in genomes]
    est_cost = sum(relative_cost(n.op) for n in op_nodes if n.op is not None)

    return {
        "n_genomes": len(genomes),
        "n_active_genes": dag.n_active_genes,
        "n_nodes": n_nodes,
        "n_base_nodes": n_base,
        "n_materialised": n_mat,
        "n_inlined": n_nodes - n_mat,
        "n_elem": sum(1 for n in op_nodes if n.scope == "elem"),
        "n_ts": sum(1 for n in op_nodes if n.scope == "ts"),
        "n_xs": sum(1 for n in op_nodes if n.scope == "xs"),
        "n_stages": len(dag.stages),
        "n_layers": max((n.layer for n in dag.nodes), default=0),
        "n_distinct_outputs": len(set(dag.outputs)),
        "dedup_ratio": n_nodes / max(dag.n_active_genes, 1),
        "reuse_factor": dag.n_active_genes / max(n_nodes, 1),
        "max_depth": max(depths, default=0),
        "mean_depth": float(np.mean(depths)) if depths else 0.0,
        "est_cost": est_cost,
    }


# --------------------------------------------------------------------------
# semantic deduplication
# --------------------------------------------------------------------------
def _round_significant(values: np.ndarray, sig: int) -> np.ndarray:
    """Round to ``sig`` significant digits; non-finite entries become 0.0."""
    out = np.zeros(values.shape, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    v = values[finite]
    nz = v != 0.0
    r = np.zeros_like(v)
    if nz.any():
        mag = np.floor(np.log10(np.abs(v[nz])))
        scale = np.power(10.0, sig - 1 - mag)
        r[nz] = np.round(v[nz] * scale) / scale
    out[finite] = r
    out[out == 0.0] = 0.0  # collapse -0.0 onto +0.0
    return out


def semantic_key(
    values: np.ndarray,
    *,
    seed: int = 0,
    n_sample: int = _SEMANTIC_SAMPLE,
    sig: int = _SEMANTIC_SIGDIGITS,
) -> int:
    """Hash of a feature's *behaviour*, for collapsing semantic duplicates.

    GP populations run 60-90% semantic duplicates -- genomes that look
    different and compute the same numbers (``x + x`` vs ``2 * x``,
    ``rank(rank(x))`` vs ``rank(x)``). Structural keys miss all of them, and a
    bitwise hash misses them too because floating-point paths differ in the
    last few ulps. So the values are subsampled on a fixed seeded index set,
    rounded to ``sig`` significant digits, and hashed together with their
    finite/non-finite mask (nulls and NaNs are part of a feature's behaviour).

    Two features collide iff they agree to ``sig`` significant digits on the
    sample *and* have the same null pattern there. Collisions are deliberate;
    the caller may keep one representative and skip the rest. Note the "on the
    sample" qualifier: with the default ``n_sample`` two features differing on
    a handful of rows out of millions can collide, which is the intended
    trade -- raise ``n_sample`` if a population needs finer resolution.

    Parameters
    ----------
    values
        Evaluated feature values. Flattened in C order; any dtype castable to
        float64.
    seed
        Seeds the subsample index set. Must be the same across a population for
        keys to be comparable.
    n_sample
        Rows sampled when ``values`` is longer than this. The sample is the
        same for every feature of the same length.
    sig
        Significant digits retained before hashing.

    Returns
    -------
    int
        Unsigned 64-bit key.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    n = v.size
    if n > n_sample:
        idx = np.sort(
            np.random.default_rng(seed).choice(n, size=n_sample, replace=False)
        )
        v = v[idx]
    finite = np.isfinite(v)
    rounded = _round_significant(v, sig)

    h = hashlib.blake2b(digest_size=8)
    h.update(b"panelary.evolve.semantic_key/1")
    h.update(np.array([n, seed, sig, v.size], dtype=np.int64).tobytes())
    h.update(np.packbits(finite).tobytes())
    h.update(rounded.tobytes())
    return int.from_bytes(h.digest(), "big")
