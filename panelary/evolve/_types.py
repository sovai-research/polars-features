"""Shared types and contracts for Panelary's evolutionary feature search.

This module is the *spine* of :mod:`panelary.evolve`. It defines the
vocabulary that every other module in the package codes against, and it has
**no third-party imports beyond numpy and polars** — matching Panelary's
light-core policy (see :mod:`panelary._deps`).

Design decisions encoded here, and the evidence behind them
-----------------------------------------------------------
The representation is a **linear straight-line program over a shared DAG**, not
a tree. Each :class:`Gene` addresses *earlier slots by index*, so:

* the genome is **fixed length** (crossover is always valid; no repair, no
  bloat spiral). This follows BSD-FE (Mach. Learn.: Sci. Technol., Dec 2025),
  whose fixed-length block chromosome beat the variable-length GA EAAFE by
  0.09-0.12 on the same 25-dataset benchmark;
* identical subexpressions **collapse to one node**, which is what makes
  population evaluation affordable. Measured on this repo (polars 1.44.1,
  200k rows, 300 individuals sharing one ``rolling_mean(20).over(entity)``):
  inline = 0.895 s vs hoisted-and-reused = 0.082 s, a **10.9x** difference.
  Polars' own CSE does *not* deduplicate these, so we must do it ourselves;
* depth is bounded by construction. This matters for a hard engineering reason
  as well as a statistical one: Polars expression *planning* time is quadratic
  in expression depth (measured here: depth 1600 = 1.62 s before touching any
  data; polars#16224, still open).

Operators are **strongly typed** (Montana 1995, *Strongly Typed Genetic
Programming*, Evol. Comput. 3(2):199-230) and carry a semantic unit, so that
``price + volume`` is *unconstructible* rather than merely penalised. Keijzer &
Babovic (GECCO 1999) and Durasevic et al. (arXiv:2004.12762) show dimensional
awareness measurably smooths the fitness landscape, so this buys search
efficiency, not just hygiene.

Leak-safety is a property of the *type system*, not of user discipline: an
operator may only enter the primitive set if ``leakage_safe`` is True, and the
grammar can express no forward-looking node. Across ~40 symbolic-regression and
GP libraries surveyed, **zero** have any lookahead concept; this is Panelary's
differentiator and it is enforced here at the representation level.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import polars as pl

__all__ = [
    "Unit",
    "OpKind",
    "Op",
    "Gene",
    "Genome",
    "EvalContext",
    "CaseMatrix",
    "FitnessResult",
    "Descriptors",
    "XS_COST_RATIO",
    "MAX_DEPTH_DEFAULT",
]

# --- semantic units -------------------------------------------------------
#
# A feature expression is dimensionally typed. Combining a `price` with a
# `volume` additively is meaningless; combining two `ret`s is fine. The unit
# lattice is deliberately small -- big type systems make the grammar sparse and
# the search space unreachable.
Unit = Literal[
    "price",  # levels in currency: close, vwap, high, ...
    "volume",  # share/contract counts
    "ret",  # dimensionless relative change
    "ratio",  # dimensionless quotient of like units
    "score",  # standardised / ranked / z-scored, dimensionless, ~O(1)
    "any",  # polymorphic: operator accepts whatever it is given
]

#: Units that are dimensionless and therefore freely inter-combinable.
DIMENSIONLESS: frozenset[str] = frozenset({"ret", "ratio", "score"})

OpKind = Literal[
    "elem",  # row-wise, no grouping needed
    "ts",  # per-entity time-series op; MUST be applied .over(entity)
    "xs",  # cross-sectional op; MUST be applied .over(time)
]

#: Measured on this machine (polars 1.44.1, 200k rows x 500 entities x 400 dates):
#: ``rank().over(time)`` costs ~18.4 ms/expr against ~0.85 ms/expr for
#: ``rolling_mean().over(entity)``. Cross-sectional operators are therefore
#: ~20x more expensive and the planner uses this to budget them.
XS_COST_RATIO: int = 20

#: Default cap on DAG depth. Kept low because (a) Polars planning time is
#: quadratic in depth, and (b) Alpha Architect's live-vs-backtest study found
#: the most complex strategies gave up >30 percentage points more of their
#: backtested Sharpe than the simplest ones.
MAX_DEPTH_DEFAULT: int = 5


@dataclass(frozen=True, slots=True)
class Op:
    """One typed, leak-safe primitive in the grammar.

    ``build`` receives the already-compiled child expressions and the resolved
    parameter value, and returns a bare :class:`polars.Expr`. It must **not**
    apply ``.over(...)`` itself -- the compiler owns grouping, because grouping
    is what it hoists and shares across the population.
    """

    name: str
    arity: int
    kind: OpKind
    in_units: tuple[Unit, ...]
    out_unit: Unit
    build: Callable[..., pl.Expr]
    #: Discrete parameter grid (e.g. rolling windows). Empty tuple = nullary.
    params: tuple[int | float, ...] = ()
    #: Relative evaluation cost, used by the budgeter. 1 = a row-wise op.
    cost: int = 1
    #: Only True operators may enter the primitive set. Enforced at build time.
    leakage_safe: bool = True
    #: Set True for ops where f(a, b) == f(b, a); lets the canonicaliser sort
    #: arguments and collapse more duplicates.
    commutative: bool = False
    #: Human-readable provenance for the registry/licence audit.
    source: str = "Panelary"

    def __post_init__(self) -> None:
        if len(self.in_units) != self.arity:
            raise ValueError(
                f"op {self.name!r}: arity {self.arity} but {len(self.in_units)} in_units"
            )
        if not self.leakage_safe:
            raise ValueError(
                f"op {self.name!r} is not leakage_safe and cannot enter the grammar"
            )


@dataclass(frozen=True, slots=True)
class Gene:
    """One instruction in a straight-line program.

    ``args`` are *slot indices*. Slot ``i < n_base`` denotes base column ``i``;
    slot ``n_base + j`` denotes the output of gene ``j``. Because a gene may
    only reference strictly earlier slots, the program is acyclic by
    construction and evaluable in a single forward pass.
    """

    op: str
    args: tuple[int, ...]
    param_ix: int = 0


@dataclass(frozen=True, slots=True)
class Genome:
    """A fixed-length straight-line program; the unit of selection.

    ``out_slot`` names the gene whose value is the genome's output; it is the
    last gene when negative. This lets mutation retarget the output without
    resizing the genome, which matters because genes that are not currently
    reachable are still latent material for later variation.

    .. note::
       Despite the name, ``out_slot`` and :meth:`output` are **gene indices**,
       not slot indices -- i.e. they index ``genes`` directly and are *not*
       offset by ``n_base`` the way :attr:`Gene.args` entries are. Both
       :mod:`._genome` and :mod:`._compile` validate
       ``0 <= output() < len(genes)`` on this basis.
    """

    genes: tuple[Gene, ...]
    n_base: int
    out_slot: int = -1

    def output(self) -> int:
        return len(self.genes) - 1 if self.out_slot < 0 else self.out_slot


@dataclass(frozen=True, slots=True)
class EvalContext:
    """Everything the compiler needs to turn genomes into Polars expressions."""

    base_columns: tuple[str, ...]
    base_units: tuple[Unit, ...]
    entity: str
    time: str
    #: Minimum non-null observations a rolling op needs before emitting a value.
    #: Never None -- an unbounded rolling window is a leak vector.
    min_periods: int = 1


@dataclass(frozen=True, slots=True)
class CaseMatrix:
    """Per-case errors/scores for lexicase selection.

    A "case" is a *(CV fold, time-bucket)* or *(CV fold, sector)* cell, never a
    single row -- a panel has millions of rows and the error matrix must stay
    small. Target 50-500 cases, which keeps a 500x300 float32 matrix at ~600 KB
    and makes selection free relative to fitness evaluation.

    This choice is load-bearing, not incidental: lexicase selects on individual
    cases rather than an aggregate, so a feature that works in 2008 and 2020 but
    not on average *survives*, where a pooled-IC tournament would kill it. That
    is precisely the regime-robustness property you want from an alpha.
    """

    #: (n_individuals, n_cases) float32, higher is better.
    scores: np.ndarray
    case_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FitnessResult:
    """The outcome of scoring one genome under leak-safe CV."""

    score: float  # aggregate (mean out-of-fold rank IC)
    per_case: np.ndarray  # (n_cases,) for lexicase
    complexity: int  # active node count
    turnover: float  # mean 1 - xs-rank-autocorr, in [0, 2]
    max_corr_to_library: float  # in [0, 1]
    coverage: float  # fraction of finite (entity, time) cells
    n_evals: int = 1  # trials consumed, for deflation


@dataclass(frozen=True, slots=True)
class Descriptors:
    """Behaviour descriptors locating a genome in the quality-diversity archive.

    Chosen so the archive spans *economically distinct* niches rather than
    syntactic ones. WorldQuant practice is explicit that "changing windows,
    weights, or neutralization rarely creates truly low-correlation alphas by
    itself", so decorrelation is made a first-class archive axis instead of a
    post-hoc filter.
    """

    turnover: float
    horizon: float
    max_corr: float
    complexity: int
    coverage: float

    def as_vector(self) -> tuple[float, ...]:
        return (
            self.turnover,
            self.horizon,
            self.max_corr,
            float(self.complexity),
            self.coverage,
        )


class Evaluator(Protocol):
    """Contract the search loop relies on; implemented in :mod:`._fitness`."""

    def evaluate(
        self, genomes: Sequence[Genome], /
    ) -> list[FitnessResult]:  # pragma: no cover - protocol
        ...
