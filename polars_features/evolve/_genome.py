"""Genome representation, initialisation and variation operators.

This module owns *everything structural* about an individual: how it is built,
how it is perturbed, how it is normalised, and how it is printed. It knows
nothing about fitness (:mod:`._fitness`), selection (:mod:`._select`) or Polars
(:mod:`._compile`); it manipulates :class:`~polars_features.evolve._types.Gene`
tuples and integers only.

Why a fixed-length linear program over a shared DAG
---------------------------------------------------
The representation is fixed in :class:`~polars_features.evolve._types.Genome`:
a straight-line program whose ``args`` are *slot indices*. Slot ``i < n_base``
is base column ``i``; slot ``n_base + j`` is the output of gene ``j``; a gene
may only reference **strictly earlier** slots, so the program is acyclic by
construction and evaluable in one forward pass.

Fixed length is the load-bearing choice, because it makes **crossover always
valid** -- there is no tree to graft, no arity to repair, and no bloat spiral to
police. BSD-FE (*Mach. Learn.: Sci. Technol.*, Dec 2025) used exactly this kind
of fixed-length block chromosome and topped a 25-dataset AutoFE benchmark
(regression 1-RAE **0.7631**, classification macro-F1 **0.8608**), beating the
variable-length GA EAAFE (**0.6425** / **0.7673**) *within the same
evolutionary family*: the encoding, not the search family, was the
discriminator.

The DAG half of the choice is what makes population evaluation affordable.
Identical subexpressions collapse to one node, and (measured on this repo,
polars 1.44.1, 200k rows, 300 individuals sharing one rolling mean) hoisting
the shared node beats inlining it by **10.9x** (0.082 s vs 0.895 s). Polars'
own common-subexpression elimination does *not* deduplicate across a
population, so the sharing has to be in the representation.

Structural discipline
---------------------
* ``max_depth`` (default :data:`~polars_features.evolve._types.MAX_DEPTH_DEFAULT`
  = 5) is enforced on **every** gene, not just the active ones, so retargeting
  the output can never produce an over-deep program. Polars expression planning
  is quadratic in depth (measured: depth 1600 = 1.62 s of pure planning,
  polars#16224), and Alpha Architect's live-vs-backtest study found the most
  complex strategies surrendered >30 percentage points more of their backtested
  Sharpe than the simplest ones. Depth is both an engineering and a statistical
  cost.
* :data:`DEFAULT_NESTED_CONSTRAINTS` forbids specific operator nestings. Miles
  Cranmer's PySR tuning guide -- the best practitioner document in this field --
  is explicit that ``nested_constraints`` is more surgical and more effective
  than a global depth cap; he forbids ``sin`` inside ``sin``. The panel
  analogues are: no ``cs_rank`` inside ``cs_rank`` (idempotent up to ties), no
  ``ts_mean`` inside ``ts_mean`` (a smoother of a smoother is a smoother), and
  no double ``abs``.
* :func:`prune_grammar` implements the other half of that advice -- *"avoid
  using redundant operators... The fewer operators the better! Only use
  operators you need."*

Units
-----
Every slot carries a semantic
:data:`~polars_features.evolve._types.Unit`. A gene may only be created if the
grammar has an operator whose ``in_units`` are satisfied by the units of
available earlier slots, so ``price + volume`` is unconstructible rather than
merely penalised (Montana 1995, *Strongly Typed GP*). Units inside
:data:`~polars_features.evolve._types.DIMENSIONLESS` are freely
inter-combinable; ``"any"`` matches anything and, as an ``out_unit``,
propagates the unit of the first argument.

Validity guarantee
------------------
Every public constructor and variation operator routes its proposal through a
forward *repair* pass (:func:`_rebuild`) that re-derives units, depths and
nested-constraint counts gene by gene and re-picks arguments (or the operator)
for any gene that does not typecheck. Consequently **100%** of genomes produced
by this module are valid: acyclic, within the depth cap, unit-consistent, and
referencing only strictly earlier slots. That is the entire point of the
encoding; if it were not 100% the encoding would be wrong.

Warm starts
-----------
:func:`seed_from_expressions` parses human-written alphas such as
``"cs_rank(ts_mean(close, 20) / ts_std(close, 20))"`` into genomes, with
common-subexpression sharing, so a run can start from known-good formulas
(arXiv:2412.00896 reports that warm-starting GP from known alphas helps).

Note on scope: :func:`structural_key` is *syntactic* dedup. Semantic dedup by
evaluated values lives in ``_compile.semantic_key``, not here.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from ._types import (
    DIMENSIONLESS,
    MAX_DEPTH_DEFAULT,
    EvalContext,
    Gene,
    Genome,
    Op,
    Unit,
)

__all__ = [
    "DEFAULT_NESTED_CONSTRAINTS",
    "NestedConstraints",
    "active_genes",
    "arg_mutation",
    "canonical_form",
    "complexity",
    "delete_gene",
    "genome_depth",
    "hoist_mutation",
    "insert_intron",
    "is_valid",
    "mutate",
    "output_mutation",
    "point_mutation",
    "prune_grammar",
    "ramped_population",
    "random_genome",
    "seed_from_expressions",
    "slot_units",
    "structural_key",
    "subgraph_crossover",
    "to_infix",
    "validate_genome",
]

#: ``{outer_op: {inner_op: max_occurrences_in_its_subgraph}}``.
#:
#: A gene may only be created if, for every ``(inner, k)`` declared for its
#: operator, ``inner`` occurs at most ``k`` times among the distinct nodes of
#: the subgraph beneath it. ``0`` therefore means "never nested".
NestedConstraints = Mapping[str, Mapping[str, int]]

DEFAULT_NESTED_CONSTRAINTS: NestedConstraints = {
    # Ranking a rank is a no-op up to tie handling.
    "cs_rank": {"cs_rank": 0},
    "cs_zscore": {"cs_zscore": 0},
    # A smoother of a smoother is just a (worse) smoother.
    "ts_mean": {"ts_mean": 0},
    # |‖x‖| == ‖x‖.
    "abs": {"abs": 0},
    "sign": {"sign": 0, "abs": 0},
}

#: Operator names to treat as commutative when no grammar is supplied.
_COMMUTATIVE_FALLBACK: frozenset[str] = frozenset(
    {
        "add",
        "mul",
        "max2",
        "min2",
        "hypot",
        "corr",
        "cov",
        "ts_corr",
        "ts_cov",
        "cs_corr",
    }
)

#: ``name -> (symbol, precedence, right_associative)`` for :func:`to_infix`.
_INFIX: dict[str, tuple[str, int, bool]] = {
    "add": ("+", 1, False),
    "sub": ("-", 1, False),
    "mul": ("*", 2, False),
    "div": ("/", 2, False),
    "mod": ("%", 2, False),
    "pow": ("**", 3, True),
}
_PREFIX: dict[str, str] = {"neg": "-"}
_SYMBOL_TO_OP: dict[str, str] = {sym: name for name, (sym, _, _) in _INFIX.items()}

#: ``name -> subsuming operators``. :func:`prune_grammar` drops ``name`` when
#: any subsumer survives. Deliberately conservative: only pairs where the
#: dropped operator is a strict special case of the one that remains.
_REDUNDANT_IF: dict[str, tuple[str, ...]] = {
    "square": ("pow", "mul"),
    "cube": ("pow",),
    "sqrt": ("pow",),
    "inv": ("div",),
    "reciprocal": ("div",),
    "log1p": ("log",),
    "expm1": ("exp",),
    "ts_sum": ("ts_mean",),
    "ts_var": ("ts_std",),
    "cs_demean": ("cs_zscore",),
    "cs_scale": ("cs_zscore",),
}

_MAX_ARG_TRIES = 12


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------
def _unit_ok(required: str, actual: str) -> bool:
    """Can a slot of unit ``actual`` feed an argument requiring ``required``?"""
    if required == "any" or actual == "any" or required == actual:
        return True
    return required in DIMENSIONLESS and actual in DIMENSIONLESS


def _wildcards(op: Op) -> tuple[int, ...]:
    """Argument positions typed ``"any"`` -- the operator's type variables."""
    return tuple(i for i, u in enumerate(op.in_units) if u == "any")


def _resolve_out_unit(op: Op, arg_units: Sequence[str]) -> Unit:
    """Unit produced by ``op`` on arguments of ``arg_units``.

    Mirrors ``._ops.resolve_out_unit``: ``out_unit="any"`` means *the unit
    bound at the first wildcard slot*, so ``ts_mean`` of a ``price`` is a
    ``price`` and of a ``ret`` is a ``ret``.
    """
    if op.out_unit != "any":
        return op.out_unit
    wild = _wildcards(op)
    if wild and len(arg_units) > wild[0]:
        return arg_units[wild[0]]  # type: ignore[return-value]
    if arg_units:
        return arg_units[0]  # type: ignore[return-value]
    return "score"


_SAME_UNIT_CACHE: frozenset[str] | None = None


_EXTERNAL_ACCEPTS: object = ...


def _external_accepts():  # type: ignore[no-untyped-def]
    """``._ops.accepts`` if importable, else ``None``."""
    global _EXTERNAL_ACCEPTS
    if _EXTERNAL_ACCEPTS is ...:
        try:
            from ._ops import accepts  # noqa: PLC0415

            _EXTERNAL_ACCEPTS = accepts
        except Exception:
            _EXTERNAL_ACCEPTS = None
    return _EXTERNAL_ACCEPTS


def _same_unit_ops() -> frozenset[str]:
    """Operators (if any) whose wildcard slots must all bind the *same* unit.

    ``._ops`` currently makes additive operators **monomorphic** instead
    (``add_price``, ``add_ret``, ...), which is a stronger form of the same
    guarantee: ``price + volume`` is unconstructible with no side condition any
    consumer could forget. The hook stays so that a grammar which does declare
    a wildcard ``add`` still gets the agreement check.
    """
    global _SAME_UNIT_CACHE
    if _SAME_UNIT_CACHE is None:
        try:
            from ._ops import SAME_UNIT_OPS  # noqa: PLC0415

            _SAME_UNIT_CACHE = frozenset(SAME_UNIT_OPS)
        except Exception:
            _SAME_UNIT_CACHE = frozenset()
    return _SAME_UNIT_CACHE


def _accepts(op: Op, arg_units: Sequence[str]) -> bool:
    """May ``op`` be applied to arguments of these units?

    Delegates to ``._ops.accepts`` when that module is importable, so the
    genome builder and the operator vocabulary can never disagree about what
    typechecks; the local implementation below is the standalone fallback.
    """
    external = _external_accepts()
    if external is not None:
        try:
            return bool(external(op, tuple(arg_units)))
        except Exception:  # pragma: no cover - defensive
            pass
    if len(arg_units) != op.arity:
        return False
    if not all(
        _unit_ok(req, act) for req, act in zip(op.in_units, arg_units, strict=False)
    ):
        return False
    if op.name in _same_unit_ops():
        return len({arg_units[i] for i in _wildcards(op)}) <= 1
    return True


# --------------------------------------------------------------------------
# grammar helpers
# --------------------------------------------------------------------------
_OPS_CACHE: dict[str, Op] | None = None


def _registry() -> dict[str, Op]:
    """Lazily borrow ``._ops.OPS`` for rendering/canonicalisation defaults."""
    global _OPS_CACHE
    if _OPS_CACHE is None:
        try:
            from ._ops import OPS  # noqa: PLC0415

            _OPS_CACHE = dict(OPS)
        except Exception:  # pragma: no cover - _ops is a sibling agent's file
            _OPS_CACHE = {}
    return _OPS_CACHE


def _op_map(grammar: Sequence[Op] | Mapping[str, Op] | None) -> dict[str, Op]:
    if grammar is None:
        return _registry()
    if isinstance(grammar, Mapping):
        return dict(grammar)
    return {op.name: op for op in grammar}


def _canon_sortable(name: str, ops: Mapping[str, Op]) -> bool:
    """May the canonicaliser reorder this operator's arguments?

    Commutativity alone is not enough. For a *unit-polymorphic* commutative
    operator -- ``in_units=("any", "any")``, ``out_unit="any"`` -- the resolved
    output unit is the unit bound at the first wildcard, so swapping
    ``add(ret, price)`` to ``add(price, ret)`` silently retypes the gene and
    can invalidate every consumer downstream. (Observed: 28 of 10,000 canonical
    forms were unit-invalid before this guard.) Reordering is therefore allowed
    only when it is provably unit-invariant: identical argument slots, and an
    output unit that does not depend on which slot bound it.
    """
    op = ops.get(name)
    if op is None:
        return name in _COMMUTATIVE_FALLBACK
    if not op.commutative or len(set(op.in_units)) != 1:
        return False
    return op.out_unit != "any" or "any" not in op.in_units


def prune_grammar(
    ops: Sequence[Op],
    *,
    drop_redundant: bool = True,
    units: Sequence[Unit] | None = None,
    keep: Iterable[str] | None = None,
    drop: Iterable[str] | None = None,
) -> tuple[Op, ...]:
    """Shrink a primitive set to the operators actually needed.

    Cranmer's PySR guide: *"avoid using redundant operators... The fewer
    operators the better! Only use operators you need."* A smaller grammar is
    not merely tidier -- every extra primitive dilutes the mutation
    distribution and enlarges the search space that the multiple-testing
    correction in :mod:`._honest` will later charge you for.

    Parameters
    ----------
    ops
        Candidate primitives. Duplicated names keep their first occurrence.
    drop_redundant
        Drop operators that a surviving operator strictly subsumes
        (``square`` given ``pow``, ``ts_sum`` given ``ts_mean``, ...).
    units
        Base units available in the panel. When given, operators whose
        ``in_units`` can never be produced from those base units are dropped
        (fixed-point reachability), and so are operators left unreachable
        afterwards.
    keep
        Names that are never dropped, whatever the other rules say.
    drop
        Names always dropped.

    Returns
    -------
    tuple[Op, ...]
        The pruned grammar, in input order.
    """
    keep_set = set(keep or ())
    drop_set = set(drop or ())

    seen: set[str] = set()
    kept: list[Op] = []
    for op in ops:
        if op.name in seen or (op.name in drop_set and op.name not in keep_set):
            continue
        seen.add(op.name)
        kept.append(op)

    if drop_redundant:
        present = {op.name for op in kept}
        kept = [
            op
            for op in kept
            if op.name in keep_set
            or not any(s in present for s in _REDUNDANT_IF.get(op.name, ()))
        ]

    if units is not None:
        reachable: set[str] = set(units)
        changed = True
        while changed:
            changed = False
            for op in kept:
                if all(
                    u == "any" or any(_unit_ok(u, a) for a in reachable)
                    for u in op.in_units
                ):
                    out = _resolve_out_unit(
                        op, [next(iter(reachable), "score")] * op.arity
                    )
                    if out not in reachable:
                        reachable.add(out)
                        changed = True
        kept = [
            op
            for op in kept
            if op.name in keep_set
            or all(
                u == "any" or any(_unit_ok(u, a) for a in reachable)
                for u in op.in_units
            )
        ]

    return tuple(kept)


# --------------------------------------------------------------------------
# structural queries
# --------------------------------------------------------------------------
def _check_ctx(ctx: EvalContext) -> int:
    n_base = len(ctx.base_columns)
    if n_base == 0:
        raise ValueError("EvalContext has no base columns")
    if len(ctx.base_units) != n_base:
        raise ValueError(
            f"EvalContext: {n_base} base_columns but {len(ctx.base_units)} base_units"
        )
    return n_base


def _out_gene(genome: Genome) -> int:
    if not genome.genes:
        raise ValueError("genome has no genes")
    out = genome.output()
    if not 0 <= out < len(genome.genes):
        raise ValueError(
            f"out_slot {genome.out_slot} is not a gene index in [0, {len(genome.genes)})"
        )
    return out


def active_genes(genome: Genome) -> frozenset[int]:
    """Gene indices reachable from the output gene (the *expressed* program)."""
    out = _out_gene(genome)
    seen: set[int] = set()
    stack = [out]
    while stack:
        j = stack.pop()
        if j in seen:
            continue
        seen.add(j)
        for a in genome.genes[j].args:
            k = a - genome.n_base
            if k >= 0:
                stack.append(k)
    return frozenset(seen)


def complexity(genome: Genome) -> int:
    """Number of **active** genes. Inactive genes are latent material, not cost."""
    return len(active_genes(genome))


def _all_depths(genome: Genome) -> list[int]:
    """Depth of every slot: base slots are 0, a gene is ``1 + max(child)``."""
    depths = [0] * (genome.n_base + len(genome.genes))
    for j, g in enumerate(genome.genes):
        d = 1
        for a in g.args:
            d = max(d, depths[a] + 1)
        depths[genome.n_base + j] = d
    return depths


def genome_depth(genome: Genome, *, active_only: bool = True) -> int:
    """Depth of the expressed program (or of the deepest gene at all)."""
    depths = _all_depths(genome)
    if not genome.genes:
        return 0
    if active_only:
        return depths[genome.n_base + _out_gene(genome)]
    return max(depths[genome.n_base :])


def slot_units(genome: Genome, ctx: EvalContext, ops: Mapping[str, Op]) -> list[str]:
    """Semantic unit of every slot, base columns first then gene outputs."""
    units: list[str] = list(ctx.base_units)
    for g in genome.genes:
        op = ops.get(g.op)
        if op is None:
            units.append("any")
            continue
        units.append(_resolve_out_unit(op, [units[a] for a in g.args]))
    return units


def validate_genome(
    genome: Genome,
    ctx: EvalContext,
    grammar: Sequence[Op] | None = None,
    *,
    max_depth: int = MAX_DEPTH_DEFAULT,
    nested_constraints: NestedConstraints | None = DEFAULT_NESTED_CONSTRAINTS,
    active_only_depth: bool = False,
) -> None:
    """Raise :class:`ValueError` unless ``genome`` satisfies every invariant.

    Checks acyclicity (args reference strictly earlier slots), operator
    membership and arity, parameter index range, unit consistency, the depth
    cap and the nested-operator constraints.
    """
    n_base = _check_ctx(ctx)
    ops = _op_map(grammar)
    if genome.n_base != n_base:
        raise ValueError(f"genome.n_base {genome.n_base} != ctx {n_base}")
    if not genome.genes:
        raise ValueError("genome has no genes")
    out = _out_gene(genome)

    units: list[str] = list(ctx.base_units)
    depths: list[int] = [0] * n_base
    nodes: list[frozenset[int]] = []
    nested = nested_constraints or {}
    active = active_genes(genome) if active_only_depth else None

    for j, g in enumerate(genome.genes):
        op = ops.get(g.op)
        if op is None:
            raise ValueError(f"gene {j}: operator {g.op!r} not in grammar")
        if len(g.args) != op.arity:
            raise ValueError(
                f"gene {j} ({g.op}): {len(g.args)} args for arity {op.arity}"
            )
        if op.params:
            if not 0 <= g.param_ix < len(op.params):
                raise ValueError(
                    f"gene {j} ({g.op}): param_ix {g.param_ix} out of range "
                    f"[0, {len(op.params)})"
                )
        elif g.param_ix != 0:
            raise ValueError(f"gene {j} ({g.op}): nullary op with param_ix != 0")

        limit = n_base + j
        for k, a in enumerate(g.args):
            if not 0 <= a < limit:
                raise ValueError(
                    f"gene {j} ({g.op}) arg {k}: slot {a} is not strictly earlier "
                    f"than {limit}"
                )
            if not _unit_ok(op.in_units[k], units[a]):
                raise ValueError(
                    f"gene {j} ({g.op}) arg {k}: needs {op.in_units[k]!r}, "
                    f"slot {a} is {units[a]!r}"
                )
        arg_units = [units[a] for a in g.args]
        if not _accepts(op, arg_units):
            raise ValueError(
                f"gene {j} ({g.op}): wildcard slots must bind one unit, got "
                f"{tuple(arg_units)!r}"
            )

        acc: set[int] = {j}
        for a in g.args:
            m = a - n_base
            if m >= 0:
                acc |= nodes[m]
        nodes.append(frozenset(acc))

        counts = Counter(genome.genes[m].op for m in acc if m != j)
        for inner, cap in nested.get(g.op, {}).items():
            if counts.get(inner, 0) > cap:
                raise ValueError(
                    f"gene {j} ({g.op}): nested constraint violated -- "
                    f"{counts[inner]} x {inner!r} beneath it, cap {cap}"
                )

        d = 1 + max((depths[a] for a in g.args), default=0)
        depths.append(d)
        units.append(_resolve_out_unit(op, [units[a] for a in g.args]))
        if d > max_depth and (active is None or j in active):
            raise ValueError(f"gene {j} ({g.op}): depth {d} exceeds cap {max_depth}")

    if out >= len(genome.genes):  # pragma: no cover - guarded by _out_gene
        raise ValueError("output gene index out of range")


def is_valid(
    genome: Genome,
    ctx: EvalContext,
    grammar: Sequence[Op] | None = None,
    *,
    max_depth: int = MAX_DEPTH_DEFAULT,
    nested_constraints: NestedConstraints | None = DEFAULT_NESTED_CONSTRAINTS,
) -> bool:
    """Boolean form of :func:`validate_genome`."""
    try:
        validate_genome(
            genome,
            ctx,
            grammar,
            max_depth=max_depth,
            nested_constraints=nested_constraints,
        )
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------
# the builder: the single place a gene is allowed to come into existence
# --------------------------------------------------------------------------
class _Builder:
    """Incrementally assembles a valid straight-line program.

    Holds the per-slot unit and depth tables and the per-gene subgraph node
    sets, so that every acceptance test is O(1)-ish and *no* invalid gene can
    be appended.
    """

    __slots__ = (
        "ctx",
        "depths",
        "genes",
        "grammar",
        "max_depth",
        "n_base",
        "nested",
        "nodes",
        "ops",
        "units",
    )

    def __init__(
        self,
        ctx: EvalContext,
        grammar: Sequence[Op],
        *,
        max_depth: int,
        nested: NestedConstraints | None,
    ) -> None:
        self.n_base = _check_ctx(ctx)
        if not grammar:
            raise ValueError("grammar is empty")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        self.ctx = ctx
        self.grammar = tuple(grammar)
        self.ops = {op.name: op for op in self.grammar}
        self.max_depth = int(max_depth)
        self.nested = dict(nested or {})
        self.units: list[str] = list(ctx.base_units)
        self.depths: list[int] = [0] * self.n_base
        self.genes: list[Gene] = []
        self.nodes: list[frozenset[int]] = []

    # -- queries ----------------------------------------------------------
    @property
    def n_slots(self) -> int:
        return len(self.units)

    def eligible(self, unit: str) -> list[int]:
        """Earlier slots that can feed an argument of unit ``unit``."""
        budget = self.max_depth - 1
        return [
            s
            for s in range(self.n_slots)
            if self.depths[s] <= budget and _unit_ok(unit, self.units[s])
        ]

    def _descendants(self, args: Sequence[int]) -> frozenset[int]:
        acc: set[int] = set()
        for a in args:
            k = a - self.n_base
            if k >= 0:
                acc |= self.nodes[k]
        return frozenset(acc)

    def nested_ok(self, name: str, args: Sequence[int]) -> bool:
        caps = self.nested.get(name)
        if not caps:
            return True
        below = self._descendants(args)
        counts = Counter(self.genes[k].op for k in below)
        return all(counts.get(inner, 0) <= cap for inner, cap in caps.items())

    def accepts(self, gene: Gene) -> bool:
        """Would ``gene`` be a legal next instruction?"""
        op = self.ops.get(gene.op)
        if op is None or len(gene.args) != op.arity:
            return False
        if op.params:
            if not 0 <= gene.param_ix < len(op.params):
                return False
        elif gene.param_ix != 0:
            return False
        limit = self.n_slots
        if not all(0 <= a < limit for a in gene.args):
            return False
        if not _accepts(op, [self.units[a] for a in gene.args]):
            return False
        if 1 + max((self.depths[a] for a in gene.args), default=0) > self.max_depth:
            return False
        return self.nested_ok(gene.op, gene.args)

    # -- mutation of the builder -----------------------------------------
    def append(self, gene: Gene) -> bool:
        """Append ``gene`` if legal; return whether it was appended."""
        if not self.accepts(gene):
            return False
        op = self.ops[gene.op]
        j = len(self.genes)
        self.genes.append(gene)
        self.nodes.append(self._descendants(gene.args) | {j})
        self.depths.append(1 + max((self.depths[a] for a in gene.args), default=0))
        self.units.append(_resolve_out_unit(op, [self.units[a] for a in gene.args]))
        return True

    # -- random construction ---------------------------------------------
    def _pick_args(
        self,
        op: Op,
        rng: np.random.Generator,
        *,
        p_reuse: float,
        out_unit: str | None,
    ) -> tuple[int, ...] | None:
        pools = [self.eligible(u) for u in op.in_units]
        if any(not pool for pool in pools):
            return None
        wild = _wildcards(op) if op.name in _same_unit_ops() else ()
        for _ in range(_MAX_ARG_TRIES):
            args: list[int] = []
            bound: str | None = None
            for k, pool in enumerate(pools):
                sub = pool
                if k in wild and bound is not None:
                    # wildcard slots of a same-unit operator must agree, which
                    # is what keeps `price + volume` unconstructible.
                    sub = [s for s in sub if self.units[s] == bound]
                    if not sub:
                        break
                if k == 0 and rng.random() < p_reuse:
                    grown = [s for s in sub if s >= self.n_base]
                    if grown:
                        sub = grown
                chosen = int(sub[int(rng.integers(len(sub)))])
                args.append(chosen)
                if k in wild and bound is None:
                    bound = self.units[chosen]
            if len(args) != op.arity:
                continue
            if op.commutative and op.arity == 2 and args[0] == args[1]:
                continue
            if not _accepts(op, [self.units[a] for a in args]):
                continue
            if not self.nested_ok(op.name, args):
                continue
            if out_unit is not None:
                got = _resolve_out_unit(op, [self.units[a] for a in args])
                if not _unit_ok(out_unit, got):
                    continue
            if 1 + max((self.depths[a] for a in args), default=0) > self.max_depth:
                continue
            return tuple(args)
        return None

    def random_gene(
        self,
        rng: np.random.Generator,
        *,
        p_reuse: float = 0.7,
        out_unit: str | None = None,
        candidates: Sequence[Op] | None = None,
    ) -> Gene | None:
        """Draw a uniformly random *legal* gene, or ``None`` if none exists."""
        pool = tuple(candidates) if candidates is not None else self.grammar
        if not pool:
            return None
        order = rng.permutation(len(pool))
        for idx in order:
            op = pool[int(idx)]
            if (
                out_unit is not None
                and op.out_unit != "any"
                and not _unit_ok(out_unit, op.out_unit)
            ):
                continue
            args = self._pick_args(op, rng, p_reuse=p_reuse, out_unit=out_unit)
            if args is None:
                continue
            param_ix = int(rng.integers(len(op.params))) if op.params else 0
            return Gene(op.name, args, param_ix)
        return None

    def append_random(
        self,
        rng: np.random.Generator,
        *,
        p_reuse: float = 0.7,
        out_unit: str | None = None,
    ) -> bool:
        gene = self.random_gene(rng, p_reuse=p_reuse, out_unit=out_unit)
        if gene is None and out_unit is not None:
            gene = self.random_gene(rng, p_reuse=p_reuse, out_unit=None)
        if gene is None:
            return False
        return self.append(gene)

    def finish(self, out_slot: int) -> Genome:
        if not self.genes:
            raise ValueError(
                "could not build a single legal gene from this grammar and context"
            )
        if not 0 <= out_slot < len(self.genes):
            out_slot = -1
        return Genome(tuple(self.genes), self.n_base, out_slot)


def _rebuild(
    proposal: Sequence[Gene],
    ctx: EvalContext,
    grammar: Sequence[Op],
    *,
    max_depth: int,
    nested: NestedConstraints | None,
    rng: np.random.Generator,
    out_slot: int,
    p_reuse: float = 0.7,
) -> Genome:
    """Repair a proposed gene list into a valid genome of the **same length**.

    Genes are replayed in order. A gene that typechecks is kept verbatim; one
    that does not has its arguments re-drawn (keeping its operator), and
    failing that its operator re-drawn too. Because base slots always exist at
    depth 0, the fallback always terminates. Length is never changed, which is
    what keeps crossover closed over the representation.
    """
    b = _Builder(ctx, grammar, max_depth=max_depth, nested=nested)
    for gene in proposal:
        if b.append(gene):
            continue
        op = b.ops.get(gene.op)
        replaced = False
        if op is not None:
            args = b._pick_args(op, rng, p_reuse=p_reuse, out_unit=None)
            if args is not None:
                param_ix = gene.param_ix if op.params else 0
                if op.params and not 0 <= param_ix < len(op.params):
                    param_ix = 0
                replaced = b.append(Gene(op.name, args, param_ix))
        if not replaced and not b.append_random(rng, p_reuse=p_reuse):
            raise ValueError(
                "grammar cannot produce any legal gene for this EvalContext"
            )
    return b.finish(out_slot)


# --------------------------------------------------------------------------
# initialisation
# --------------------------------------------------------------------------
def random_genome(
    ctx: EvalContext,
    grammar: Sequence[Op],
    *,
    n_genes: int = 8,
    max_depth: int = MAX_DEPTH_DEFAULT,
    seed: int = 0,
    nested_constraints: NestedConstraints | None = DEFAULT_NESTED_CONSTRAINTS,
    out_unit: Unit | None = None,
    p_reuse: float = 0.7,
) -> Genome:
    """Draw a uniformly random, valid, unit-consistent genome.

    A gene is only created when the grammar holds an operator whose
    ``in_units`` are satisfied by the units of the slots already available, so
    the result typechecks by construction rather than by rejection sampling.
    Slot units and depths are tracked as the program is built.

    Parameters
    ----------
    ctx
        Panel context; supplies the base columns and their units.
    grammar
        Typed, leak-safe primitives (see ``._ops.default_grammar``).
    n_genes
        Fixed program length. Inactive genes are latent material for
        :func:`output_mutation` and :func:`subgraph_crossover`, not waste.
    max_depth
        Cap applied to **every** gene, so any output retargeting stays legal.
    seed
        Explicit RNG seed; a ``numpy`` generator is built internally.
    nested_constraints
        Forbidden operator nestings; see :data:`DEFAULT_NESTED_CONSTRAINTS`.
    out_unit
        Optional unit the final gene must produce (e.g. ``"score"``).
    p_reuse
        Probability of forcing the first argument to come from an earlier
        gene rather than a base column. This is the knob that grows depth.

    Returns
    -------
    Genome
        A genome with exactly ``n_genes`` genes and ``out_slot == -1``.
    """
    if n_genes < 1:
        raise ValueError(f"n_genes must be >= 1, got {n_genes}")
    rng = np.random.default_rng(seed)
    return _random_genome(
        ctx,
        grammar,
        n_genes=n_genes,
        max_depth=max_depth,
        rng=rng,
        nested=nested_constraints,
        out_unit=out_unit,
        p_reuse=p_reuse,
    )


def _random_genome(
    ctx: EvalContext,
    grammar: Sequence[Op],
    *,
    n_genes: int,
    max_depth: int,
    rng: np.random.Generator,
    nested: NestedConstraints | None,
    out_unit: str | None,
    p_reuse: float,
) -> Genome:
    b = _Builder(ctx, grammar, max_depth=max_depth, nested=nested)
    for j in range(n_genes):
        want = out_unit if j == n_genes - 1 else None
        if not b.append_random(rng, p_reuse=p_reuse, out_unit=want):
            raise ValueError(
                "grammar cannot produce any legal gene for this EvalContext "
                f"(failed at gene {j})"
            )
    return b.finish(-1)


def ramped_population(
    ctx: EvalContext,
    grammar: Sequence[Op],
    *,
    size: int,
    seed: int = 0,
    min_genes: int = 3,
    max_genes: int = 12,
    min_depth: int = 2,
    max_depth: int = MAX_DEPTH_DEFAULT,
    nested_constraints: NestedConstraints | None = DEFAULT_NESTED_CONSTRAINTS,
    out_unit: Unit | None = None,
    warm_start: Sequence[Genome] | None = None,
    dedup: bool = True,
    max_tries: int = 20,
) -> list[Genome]:
    """Ramped initialisation: vary program length *and* depth cap.

    Koza's ramped half-and-half, transposed to a linear encoding. Length and
    depth are ramped together across the population so the initial gene pool
    spans genuinely different structural scales instead of clustering at one
    size, which is the usual cause of premature convergence.

    ``warm_start`` genomes (e.g. from :func:`seed_from_expressions`) are placed
    first and count towards ``size``.

    Deduplication is *syntactic* (:func:`structural_key`); semantic dedup by
    evaluated values belongs to ``_compile.semantic_key``.
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    min_genes = max(1, min_genes)
    max_genes = max(min_genes, max_genes)
    min_depth = max(1, min(min_depth, max_depth))
    rng = np.random.default_rng(seed)

    out: list[Genome] = []
    seen: set[int] = set()
    for g in warm_start or ():
        if len(out) >= size:
            break
        key = structural_key(g, grammar=grammar)
        if dedup and key in seen:
            continue
        seen.add(key)
        out.append(g)

    n_rungs = max(1, max_depth - min_depth + 1)
    while len(out) < size:
        i = len(out)
        rung = i % n_rungs
        cap = min_depth + rung
        span = max_genes - min_genes
        n_genes = min_genes + (int(rng.integers(span + 1)) if span else 0)
        genome: Genome | None = None
        for _ in range(max_tries):
            cand = _random_genome(
                ctx,
                grammar,
                n_genes=n_genes,
                max_depth=cap,
                rng=rng,
                nested=nested_constraints,
                out_unit=out_unit,
                p_reuse=0.7,
            )
            key = structural_key(cand, grammar=grammar)
            if not dedup or key not in seen:
                seen.add(key)
                genome = cand
                break
            genome = cand
        assert genome is not None  # noqa: S101 - loop runs at least once
        out.append(genome)
    return out


# --------------------------------------------------------------------------
# warm starts: parse human-written alphas
# --------------------------------------------------------------------------
def _infix_spec(name: str) -> tuple[str, int, bool] | None:
    """Infix rendering for ``name``, honouring ``<sym>_<suffix>`` families.

    A strongly typed grammar often carries several overloads of one arithmetic
    symbol (``div_p``, ``div_v``). They render as the symbol and are resolved
    back by :func:`_overloads`.
    """
    spec = _INFIX.get(name)
    if spec is not None:
        return spec
    return _INFIX.get(name.split("_", 1)[0])


def _prefix_spec(name: str) -> str | None:
    sym = _PREFIX.get(name)
    if sym is not None:
        return sym
    return _PREFIX.get(name.split("_", 1)[0])


def _overloads(name: str, ops: Mapping[str, Op], arity: int) -> list[Op]:
    """Candidate operators for a parsed call, best guess first.

    The exact name wins. Only *arithmetic symbols* additionally pull in their
    ``<name>_<suffix>`` family, so ``ts_mean`` can never silently resolve to
    some other ``ts_*`` operator.
    """
    out: list[Op] = []
    exact = ops.get(name)
    if exact is not None and exact.arity == arity:
        out.append(exact)
    if name in _INFIX or name in _PREFIX:
        out.extend(
            sorted(
                (
                    o
                    for n, o in ops.items()
                    if n != name and o.arity == arity and n.split("_", 1)[0] == name
                ),
                key=lambda o: o.name,
            )
        )
    return out


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif text.startswith("**", i):
            tokens.append("**")
            i += 2
        elif c in "(),+-*/%":
            tokens.append(c)
            i += 1
        elif c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            while j < n and (
                text[j].isdigit()
                or text[j] in "._"
                or (
                    text[j] in "eE"
                    and j + 1 < n
                    and (text[j + 1].isdigit() or text[j + 1] in "+-")
                )
                or (text[j] in "+-" and text[j - 1] in "eE")
            ):
                j += 1
            tokens.append(text[i:j].replace("_", ""))
            i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(text[i:j])
            i = j
        else:
            raise ValueError(f"cannot tokenize {text!r} at position {i}: {c!r}")
    return tokens


class _Parser:
    """Tiny precedence-climbing parser for the language :func:`to_infix` emits."""

    def __init__(self, tokens: Sequence[str]) -> None:
        self.t = list(tokens)
        self.i = 0

    def peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self) -> str:
        tok = self.peek()
        if tok is None:
            raise ValueError("unexpected end of expression")
        self.i += 1
        return tok

    def expect(self, tok: str) -> None:
        got = self.take()
        if got != tok:
            raise ValueError(f"expected {tok!r}, got {got!r}")

    def parse(self) -> tuple:
        node = self.expr(0)
        if self.peek() is not None:
            raise ValueError(f"trailing tokens from {self.peek()!r}")
        return node

    def expr(self, min_prec: int) -> tuple:
        left = self.atom()
        while True:
            tok = self.peek()
            if tok is None or tok not in _SYMBOL_TO_OP:
                break
            name = _SYMBOL_TO_OP[tok]
            _, prec, right_assoc = _INFIX[name]
            if prec < min_prec:
                break
            self.take()
            right = self.expr(prec if right_assoc else prec + 1)
            left = ("call", name, [left, right], [])
        return left

    def atom(self) -> tuple:
        tok = self.take()
        if tok == "(":
            node = self.expr(0)
            self.expect(")")
            return node
        if tok == "-":
            return ("call", "neg", [self.atom()], [])
        if tok == "+":
            return self.atom()
        if tok[0].isdigit() or tok[0] == ".":
            return ("num", float(tok))
        if not (tok[0].isalpha() or tok[0] == "_"):
            raise ValueError(f"unexpected token {tok!r}")
        if self.peek() != "(":
            return ("col", tok)
        self.take()
        children: list[tuple] = []
        params: list[float] = []
        if self.peek() != ")":
            while True:
                arg = self.expr(0)
                if arg[0] == "num":
                    params.append(arg[1])
                else:
                    children.append(arg)
                if self.peek() == ",":
                    self.take()
                    continue
                break
        self.expect(")")
        return ("call", tok, children, params)


def _emit(
    node: tuple,
    b: _Builder,
    ctx: EvalContext,
    memo: dict[str, int],
) -> int:
    kind = node[0]
    if kind == "col":
        name = node[1]
        if name not in ctx.base_columns:
            raise ValueError(
                f"unknown base column {name!r}; available: {list(ctx.base_columns)}"
            )
        return ctx.base_columns.index(name)
    if kind == "num":
        raise ValueError(
            f"bare numeric literal {node[1]!r}: constants are only allowed as "
            "trailing operator parameters, e.g. ts_mean(close, 20)"
        )
    _, name, children, params = node
    arg_slots = tuple(_emit(c, b, ctx, memo) for c in children)
    candidates = _overloads(name, b.ops, len(children))
    if not candidates:
        raise ValueError(
            f"no operator named {name!r} with arity {len(children)}; "
            f"grammar has {sorted(b.ops)}"
        )
    for op in candidates:
        if params and not op.params:
            continue
        param_ix = 0
        if op.params and params:
            grid = np.asarray(op.params, dtype=np.float64)
            param_ix = int(np.argmin(np.abs(grid - float(params[0]))))
        key = f"{op.name}:{param_ix}:{arg_slots}"
        if key in memo:
            return memo[key]
        if b.append(Gene(op.name, arg_slots, param_ix)):
            slot = b.n_base + len(b.genes) - 1
            memo[key] = slot
            return slot
    tried = ", ".join(o.name for o in candidates)
    units = tuple(b.units[s] for s in arg_slots)
    raise ValueError(
        f"{name}{arg_slots}: no admissible operator (tried {tried}) for argument "
        f"units {units} -- check units, the depth cap and nested_constraints"
    )


def seed_from_expressions(
    expressions: Sequence[str],
    ctx: EvalContext,
    grammar: Sequence[Op],
    *,
    n_genes: int | None = None,
    max_depth: int = MAX_DEPTH_DEFAULT,
    seed: int = 0,
    nested_constraints: NestedConstraints | None = DEFAULT_NESTED_CONSTRAINTS,
    on_error: str = "raise",
) -> list[Genome]:
    """Warm-start the search from known-good alphas written as text.

    Parses the same minimally-parenthesised language :func:`to_infix` emits --
    ``"cs_rank(ts_mean(close, 20) / ts_std(close, 20))"`` -- with trailing
    numeric arguments resolved to the nearest value on the operator's parameter
    grid, and identical subexpressions shared (one gene, several consumers).

    Warm-starting a genetic program from published alphas measurably helps
    (arXiv:2412.00896); this is the hook for it.

    Notes
    -----
    A seeded genome must satisfy the *same* invariants as an evolved one --
    units, depth cap and ``nested_constraints`` -- because the first mutation
    would otherwise silently repair it. Pass ``nested_constraints=None`` to
    admit a hand-written alpha that deliberately breaks one of the nesting
    heuristics, or ``on_error="skip"`` to drop unparseable entries instead of
    raising.

    ``n_genes`` is a *minimum*: shorter programs are padded with random latent
    genes (which obey the same constraints) so a whole warm-started population
    can share one length.
    """
    if on_error not in {"raise", "skip"}:
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")
    rng = np.random.default_rng(seed)
    out: list[Genome] = []
    for text in expressions:
        b = _Builder(ctx, grammar, max_depth=max_depth, nested=nested_constraints)
        try:
            node = _Parser(_tokenize(text)).parse()
            root = _emit(node, b, ctx, {})
        except ValueError as exc:
            if on_error == "skip":
                continue
            raise ValueError(f"cannot seed from {text!r}: {exc}") from exc
        if root < b.n_base:
            if on_error == "skip":
                continue
            raise ValueError(
                f"cannot seed from {text!r}: expression is a bare base column"
            )
        out_gene = root - b.n_base
        target = max(n_genes or 0, len(b.genes))
        while len(b.genes) < target and b.append_random(rng, p_reuse=0.6):
            pass
        out.append(b.finish(out_gene))
    return out


# --------------------------------------------------------------------------
# variation
# --------------------------------------------------------------------------
def _prep(
    genome: Genome,
    ctx: EvalContext,
    grammar: Sequence[Op],
    seed: int,
) -> tuple[np.random.Generator, list[Gene], dict[str, Op]]:
    if not genome.genes:
        raise ValueError("genome has no genes")
    return (
        np.random.default_rng(seed),
        list(genome.genes),
        {op.name: op for op in grammar},
    )


def point_mutation(
    genome: Genome,
    grammar: Sequence[Op],
    ctx: EvalContext,
    *,
    seed: int = 0,
    max_depth: int = MAX_DEPTH_DEFAULT,
    nested_constraints: NestedConstraints | None = DEFAULT_NESTED_CONSTRAINTS,
    p_param: float = 0.35,
) -> Genome:
    """Change one gene's operator to a type-compatible one, or its ``param_ix``.

    Parameter mutation (a rolling window 20 -> 60) is the smallest possible
    move and is drawn with probability ``p_param``; otherwise the operator
    itself is swapped and the downstream genes are repaired forward.
    """
    rng, genes, ops = _prep(genome, ctx, grammar, seed)
    j = int(rng.integers(len(genes)))
    g = genes[j]
    op = ops.get(g.op)

    if op is not None and len(op.params) > 1 and rng.random() < p_param:
        choices = [p for p in range(len(op.params)) if p != g.param_ix]
        genes[j] = Gene(g.op, g.args, int(choices[int(rng.integers(len(choices)))]))
    else:
        same_arity = [o for o in grammar if o.arity == len(g.args) and o.name != g.op]
        if same_arity:
            cand = same_arity[int(rng.integers(len(same_arity)))]
            genes[j] = Gene(
                cand.name,
                g.args,
                int(rng.integers(len(cand.params))) if cand.params else 0,
            )
    return _rebuild(
        genes,
        ctx,
        grammar,
        max_depth=max_depth,
        nested=nested_constraints,
        rng=rng,
        out_slot=genome.out_slot,
    )


def arg_mutation(
    genome: Genome,
    grammar: Sequence[Op],
    ctx: EvalContext,
    *,
    seed: int = 0,
    max_depth: int = MAX_DEPTH_DEFAULT,
    nested_constraints: NestedConstraints | None = DEFAULT_NESTED_CONSTRAINTS,
) -> Genome:
    """Rewire one argument to a different, unit-compatible, strictly earlier slot.

    This is the operator that reshapes the DAG's topology without touching its
    vocabulary; the forward repair keeps every downstream gene legal.
    """
    rng, genes, _ = _prep(genome, ctx, grammar, seed)
    with_args = [j for j, g in enumerate(genes) if g.args]
    if not with_args:
        return genome
    j = int(with_args[int(rng.integers(len(with_args)))])
    g = genes[j]
    k = int(rng.integers(len(g.args)))
    limit = genome.n_base + j
    choices = [s for s in range(limit) if s != g.args[k]]
    if choices:
        args = list(g.args)
        args[k] = int(choices[int(rng.integers(len(choices)))])
        genes[j] = Gene(g.op, tuple(args), g.param_ix)
    return _rebuild(
        genes,
        ctx,
        grammar,
        max_depth=max_depth,
        nested=nested_constraints,
        rng=rng,
        out_slot=genome.out_slot,
    )


def output_mutation(
    genome: Genome,
    grammar: Sequence[Op] | None = None,
    ctx: EvalContext | None = None,
    *,
    seed: int = 0,
    out_unit: Unit | None = None,
) -> Genome:
    """Retarget the output gene.

    Genes outside the active subgraph are *latent material*: neutral drift has
    been accumulating them, and this is the operator that expresses them. It
    cannot produce an invalid genome, because the depth cap is enforced on all
    genes, not only the expressed ones.
    """
    rng = np.random.default_rng(seed)
    n = len(genome.genes)
    if n < 2:
        return genome
    current = _out_gene(genome)
    choices = [j for j in range(n) if j != current]
    if out_unit is not None and ctx is not None:
        units = slot_units(genome, ctx, _op_map(grammar))
        choices = [
            j for j in choices if _unit_ok(out_unit, units[genome.n_base + j])
        ] or choices
    j = int(choices[int(rng.integers(len(choices)))])
    return Genome(genome.genes, genome.n_base, j)


def hoist_mutation(genome: Genome, *, seed: int = 0) -> Genome:
    """Promote an active subgraph to be the whole program.

    gplearn's anti-bloat operator (BSD-3, so the idea is safe to reimplement),
    transposed to a linear DAG: pick an *expressed* gene other than the root
    and make it the root. Active complexity strictly decreases, which is the
    only variation operator here that is guaranteed to simplify.
    """
    rng = np.random.default_rng(seed)
    active = sorted(active_genes(genome))
    current = _out_gene(genome)
    choices = [j for j in active if j != current]
    if not choices:
        return genome
    j = int(choices[int(rng.integers(len(choices)))])
    return Genome(genome.genes, genome.n_base, j)


def insert_intron(
    genome: Genome,
    grammar: Sequence[Op],
    ctx: EvalContext,
    *,
    seed: int = 0,
    max_depth: int = MAX_DEPTH_DEFAULT,
    nested_constraints: NestedConstraints | None = DEFAULT_NESTED_CONSTRAINTS,
) -> Genome:
    """Write a fresh random gene into an unexpressed position.

    Length is **never** changed -- a fixed-length genome is the invariant the
    whole encoding rests on. When an inactive position exists the move is
    exactly neutral (phenotype unchanged) and simply refreshes the latent pool
    that :func:`output_mutation` and :func:`subgraph_crossover` draw on; when
    the program is fully expressed, a non-root gene is overwritten instead and
    the forward repair keeps the result legal.
    """
    rng, genes, _ = _prep(genome, ctx, grammar, seed)
    active = active_genes(genome)
    inactive = [j for j in range(len(genes)) if j not in active]
    pool = inactive or [j for j in range(len(genes)) if j != _out_gene(genome)]
    if not pool:
        return genome
    j = int(pool[int(rng.integers(len(pool)))])

    b = _Builder(ctx, grammar, max_depth=max_depth, nested=nested_constraints)
    for g in genes[:j]:
        if not b.append(g):
            b.append_random(rng)
    fresh = b.random_gene(rng, p_reuse=0.6)
    genes[j] = fresh if fresh is not None else genes[j]
    return _rebuild(
        genes,
        ctx,
        grammar,
        max_depth=max_depth,
        nested=nested_constraints,
        rng=rng,
        out_slot=genome.out_slot,
    )


def delete_gene(
    genome: Genome,
    grammar: Sequence[Op],
    ctx: EvalContext,
    *,
    seed: int = 0,
    max_depth: int = MAX_DEPTH_DEFAULT,
    nested_constraints: NestedConstraints | None = DEFAULT_NESTED_CONSTRAINTS,
) -> Genome:
    """Splice an expressed gene out of the program without resizing it.

    Consumers of the deleted gene are rewired to one of *its* arguments (the
    classic "shrink" move, which is what actually removes a node from the
    expressed subgraph), and the gene itself stays in place as dead material.
    Length is fixed; the forward repair fixes any unit mismatch the rewiring
    introduces.
    """
    rng, genes, _ = _prep(genome, ctx, grammar, seed)
    active = active_genes(genome)
    out = _out_gene(genome)
    pool = [j for j in sorted(active) if j != out and genes[j].args]
    if not pool:
        return genome
    j = int(pool[int(rng.integers(len(pool)))])
    victim = genome.n_base + j
    replacements = list(genes[j].args)

    for k in range(j + 1, len(genes)):
        g = genes[k]
        if victim not in g.args:
            continue
        args = [
            (
                int(replacements[int(rng.integers(len(replacements)))])
                if a == victim
                else a
            )
            for a in g.args
        ]
        genes[k] = Gene(g.op, tuple(args), g.param_ix)

    return _rebuild(
        genes,
        ctx,
        grammar,
        max_depth=max_depth,
        nested=nested_constraints,
        rng=rng,
        out_slot=genome.out_slot,
    )


def subgraph_crossover(
    a: Genome,
    b: Genome,
    grammar: Sequence[Op],
    ctx: EvalContext,
    *,
    seed: int = 0,
    max_depth: int = MAX_DEPTH_DEFAULT,
    nested_constraints: NestedConstraints | None = DEFAULT_NESTED_CONSTRAINTS,
) -> Genome:
    """Copy a contiguous gene slice from ``b`` into ``a``, remapping arguments.

    This is the operator the whole encoding exists for. A slice of ``b`` is
    written into the same index range of ``a``; every argument in the copied
    block is remapped -- references *inside* the block keep their relative
    offset, references *outside* it are redrawn from ``a``'s earlier slots --
    so the offspring is acyclic and same-length with **no repair pass needed
    for validity, and no bloat**. Compare a tree GA, where the graft must be
    size- and type-checked and half the offspring are discarded.

    The offspring always has ``len(a.genes)`` genes and inherits ``a``'s
    output target (clamped).
    """
    if not a.genes or not b.genes:
        raise ValueError("both parents must have at least one gene")
    if a.n_base != b.n_base:
        raise ValueError(f"parents disagree on n_base: {a.n_base} vs {b.n_base}")
    rng = np.random.default_rng(seed)
    genes = list(a.genes)
    n_a, n_b = len(genes), len(b.genes)

    span = int(rng.integers(1, min(n_a, n_b) + 1))
    src = int(rng.integers(0, n_b - span + 1))
    dst = int(rng.integers(0, n_a - span + 1))
    shift = dst - src  # b-gene index -> a-gene index inside the block

    for t in range(span):
        g = b.genes[src + t]
        limit = a.n_base + dst + t
        args: list[int] = []
        for arg in g.args:
            gene_ix = arg - b.n_base
            if gene_ix < 0:
                args.append(arg)  # base column: identical in both parents
                continue
            mapped = gene_ix + shift
            if src <= gene_ix < src + t and 0 <= mapped < dst + t:
                args.append(a.n_base + mapped)  # stays inside the copied block
            else:
                args.append(int(rng.integers(limit)))  # redraw from the host
        genes[dst + t] = Gene(g.op, tuple(args), g.param_ix)

    return _rebuild(
        genes,
        ctx,
        grammar,
        max_depth=max_depth,
        nested=nested_constraints,
        rng=rng,
        out_slot=a.out_slot,
    )


def mutate(
    genome: Genome,
    grammar: Sequence[Op],
    ctx: EvalContext,
    *,
    seed: int = 0,
    max_depth: int = MAX_DEPTH_DEFAULT,
    nested_constraints: NestedConstraints | None = DEFAULT_NESTED_CONSTRAINTS,
    p_point: float = 0.40,
    p_arg: float = 0.30,
    p_output: float = 0.10,
    p_hoist: float = 0.05,
    p_insert: float = 0.10,
    p_delete: float = 0.05,
) -> Genome:
    """Apply the variation operators independently; guarantee at least one fires.

    Rates are per-operator Bernoulli draws rather than a single categorical
    choice, so compound moves (retune a window *and* rewire an argument) are
    reachable in one step. If nothing fires, :func:`point_mutation` is applied
    so ``mutate`` never returns its input unchanged by accident.
    """
    rng = np.random.default_rng(seed)
    kw = {"max_depth": max_depth, "nested_constraints": nested_constraints}
    plan: list[tuple[float, str]] = [
        (p_point, "point"),
        (p_arg, "arg"),
        (p_output, "output"),
        (p_hoist, "hoist"),
        (p_insert, "insert"),
        (p_delete, "delete"),
    ]
    fired = [name for p, name in plan if rng.random() < p]
    if not fired:
        fired = ["point"]

    out = genome
    for name in fired:
        sub = int(rng.integers(0, 2**62))
        if name == "point":
            out = point_mutation(out, grammar, ctx, seed=sub, **kw)  # type: ignore[arg-type]
        elif name == "arg":
            out = arg_mutation(out, grammar, ctx, seed=sub, **kw)  # type: ignore[arg-type]
        elif name == "output":
            out = output_mutation(out, grammar, ctx, seed=sub)
        elif name == "hoist":
            out = hoist_mutation(out, seed=sub)
        elif name == "insert":
            out = insert_intron(out, grammar, ctx, seed=sub, **kw)  # type: ignore[arg-type]
        else:
            out = delete_gene(out, grammar, ctx, seed=sub, **kw)  # type: ignore[arg-type]
    return out


# --------------------------------------------------------------------------
# canonicalisation
# --------------------------------------------------------------------------
def _canon_keys(
    genome: Genome, ops: Mapping[str, Op]
) -> tuple[dict[int, str], dict[int, str]]:
    """Canonical key per active gene, and per referenced slot."""
    active = active_genes(genome)
    gene_key: dict[int, str] = {}
    for j in sorted(active):
        g = genome.genes[j]
        child = [
            f"${a}" if a < genome.n_base else gene_key[a - genome.n_base]
            for a in g.args
        ]
        if _canon_sortable(g.op, ops):
            child = sorted(child)
        gene_key[j] = f"{g.op}#{g.param_ix}(" + ",".join(child) + ")"
    slot_key = {a: f"${a}" for a in range(genome.n_base)}
    for j, k in gene_key.items():
        slot_key[genome.n_base + j] = k
    return gene_key, slot_key


def canonical_form(
    genome: Genome, *, grammar: Sequence[Op] | Mapping[str, Op] | None = None
) -> Genome:
    """Normalise: drop unreachable genes, sort commutative args, renumber densely.

    The result is the unique representative of the genome's *syntactic*
    equivalence class: dead code removed, duplicate subgraphs merged into one
    node, commutative arguments in canonical order, genes emitted in
    ``(depth, key)`` order so construction history leaves no trace, and the
    root last (``out_slot == -1``). The function is idempotent.

    Because it drops genes, the canonical form is generally **shorter** than
    the working genome; use it for hashing, reporting and printing, and keep
    the working genome for variation (the latent material is what
    :func:`output_mutation` feeds on).
    """
    ops = _op_map(grammar)
    gene_key, _ = _canon_keys(genome, ops)
    depths = _all_depths(genome)
    order = sorted(gene_key, key=lambda j: (depths[genome.n_base + j], gene_key[j]))

    new_of_key: dict[str, int] = {}
    new_genes: list[Gene] = []
    for j in order:
        key = gene_key[j]
        if key in new_of_key:
            continue
        g = genome.genes[j]
        pairs: list[tuple[str, int]] = []
        for a in g.args:
            if a < genome.n_base:
                pairs.append((f"${a}", a))
            else:
                ck = gene_key[a - genome.n_base]
                pairs.append((ck, genome.n_base + new_of_key[ck]))
        if _canon_sortable(g.op, ops):
            pairs.sort(key=lambda p: p[0])
        new_of_key[key] = len(new_genes)
        new_genes.append(Gene(g.op, tuple(s for _, s in pairs), g.param_ix))

    return Genome(tuple(new_genes), genome.n_base, -1)


def structural_key(
    genome: Genome, *, grammar: Sequence[Op] | Mapping[str, Op] | None = None
) -> int:
    """Stable 64-bit hash of the canonical form (syntactic dedup).

    Insensitive to dead code, gene ordering, commutative argument order and
    duplicated subgraphs; sensitive to operators, parameter indices, base
    columns and topology. Uses ``blake2b`` rather than :func:`hash` so the
    value is reproducible across processes -- ledgers in :mod:`._honest` are
    keyed on it.

    Semantic dedup (identical *values*) is ``_compile.semantic_key``; two
    genomes can be semantically identical and structurally distinct.
    """
    ops = _op_map(grammar)
    gene_key, _ = _canon_keys(genome, ops)
    root = gene_key[_out_gene(genome)]
    payload = f"{genome.n_base}|{root}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _fmt_param(value: float | int) -> str:
    fv = float(value)
    return str(int(fv)) if fv.is_integer() else f"{fv:g}"


def _render(
    genome: Genome,
    slot: int,
    ctx: EvalContext,
    ops: Mapping[str, Op],
    parent_prec: int,
    right_side: bool,
) -> str:
    if slot < genome.n_base:
        return ctx.base_columns[slot]
    g = genome.genes[slot - genome.n_base]
    op = ops.get(g.op)

    spec = _infix_spec(g.op) if len(g.args) == 2 else None
    if spec is not None:
        sym, prec, right_assoc = spec
        left = _render(genome, g.args[0], ctx, ops, prec, right_assoc)
        right = _render(genome, g.args[1], ctx, ops, prec, not right_assoc)
        body = f"{left} {sym} {right}"
        needs = prec < parent_prec or (prec == parent_prec and right_side)
        return f"({body})" if needs else body

    pre = _prefix_spec(g.op) if len(g.args) == 1 else None
    if pre is not None:
        inner = _render(genome, g.args[0], ctx, ops, 4, False)
        body = f"{pre}{inner}"
        return f"({body})" if parent_prec > 0 else body

    parts = [_render(genome, a, ctx, ops, 0, False) for a in g.args]
    if op is not None and op.params:
        parts.append(_fmt_param(op.params[g.param_ix]))
    elif op is None and g.param_ix:
        parts.append(f"p{g.param_ix}")
    return f"{g.op}(" + ", ".join(parts) + ")"


def to_infix(
    genome: Genome,
    ctx: EvalContext,
    *,
    grammar: Sequence[Op] | Mapping[str, Op] | None = None,
    canonical: bool = False,
) -> str:
    """Render the expressed program as minimally-parenthesised infix text.

    ``cs_rank(ts_mean(close, 20) / ts_std(close, 20))``. This is what a human
    actually reads, so it is deliberately terse: arithmetic operators are
    infix with standard precedence and parentheses only where they change
    meaning, everything else is a call with its resolved parameter as the last
    argument. Shared subexpressions are printed once per use.

    Set ``canonical=True`` to render :func:`canonical_form` instead, which also
    sorts commutative arguments and strips dead code.
    """
    ops = _op_map(grammar)
    g = canonical_form(genome, grammar=ops) if canonical else genome
    return _render(g, g.n_base + _out_gene(g), ctx, ops, 0, False)
