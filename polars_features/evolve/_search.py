"""The evolutionary search loop: islands, selection, archive, and the ledger.

This module wires the pieces together. Its job is less "run a genetic
algorithm" -- that part is a few hundred lines and thoroughly commoditised --
than to make sure every candidate the search touches is *counted*, and that the
thing handed back to the user is a decorrelated pool with an honest error bar
attached.

Shape of the loop
-----------------
Per generation, per island:

1. Compile the whole island population into **one** layered Polars plan
   (:func:`._compile.compile_population`). Shared subexpressions collapse to a
   single node; measured 10.9x on this repo.
2. Score under purged/embargoed CV (:class:`._fitness.PanelEvaluator`), which
   also emits a per-case matrix.
3. Record **every** candidate in the :class:`._honest.TrialLedger` -- before
   any filtering. This is the step the published literature skips.
4. Select parents by semi-dynamic epsilon-lexicase on the case matrix.
5. Vary, deduplicate (structurally *and* semantically), and refill.
6. Offer every candidate to the MAP-Elites archive.

Islands migrate on a ring every ``migration_every`` generations and may be
reset (worst half reseeded from a survivor's elite) every ``reset_every``
generations -- FunSearch's anti-convergence mechanism, which is the closest
published analogue to the alpha-crowding problem.

Why the defaults are small
--------------------------
``population=200``/``generations=20`` is 4,000 evaluations, not the 10^5-10^6
the older literature used. Two reasons. Statistically, more trials is not free:
the expected best in-sample Sharpe under the null grows as roughly
``sqrt(2 ln N)``, so a 100,000-trial search on pure noise is *expected* to hand
you a Sharpe near 4.4 (verified against PanelKit's own
``expected_maximum_sharpe``). Empirically, the field has moved the same way --
LLM-FE reached state of the art on 37 datasets with **20** samples, and
AlphaBench measured evolutionary capacity saturating at 20 candidates per
round. Budget is a parameter, not a virtue; :func:`._honest.minimum_backtest_length`
will tell you what your panel can actually afford before you spend it.

Honesty is not optional here
----------------------------
:meth:`EvolveResult.summary` always reports the deflated statistic alongside
the raw one, and :attr:`EvolveResult.diagnostics` always carries the
in-sample-vs-held-out scatter across *all* candidates. If that cloud is round,
the search found nothing, and the report says so in words.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ._types import Descriptors, EvalContext, FitnessResult, Genome, Unit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._select import Archive

__all__ = ["EvolveConfig", "EvolveResult", "evolve_features"]


@dataclass(frozen=True, slots=True)
class EvolveConfig:
    """Search budget and operators.

    Parameters
    ----------
    population : int
        Individuals per island, per generation.
    generations : int
        Number of generations.
    n_islands : int
        Independent subpopulations. FunSearch used 10; LLM-FE used 3 for
        feature engineering. Small is fine and cheaper to deflate.
    min_genes, n_genes, max_depth : int
        Genome size range (ramped between the two) and the hard cap on DAG
        depth. ``max_depth`` is low on
        purpose: Polars planning time is quadratic in expression depth, and
        complex strategies decay worse out of sample.
    downsample : float
        Fraction of cases each epsilon-lexicase selection event sees.
    migration_every, reset_every : int or None
        Ring migration and island-reset cadences, in generations.
    noise_features : int
        Number of shuffled/Gaussian decoys injected as an in-run null. A
        candidate must beat the best *decoy*, not merely beat zero.
    max_library : int
        Cap on the returned decorrelated pool.
    seed : int
        Explicit seed; the whole search is reproducible from it.
    """

    population: int = 200
    generations: int = 20
    n_islands: int = 3
    min_genes: int = 3
    n_genes: int = 12
    max_depth: int = 5
    downsample: float = 0.1
    migration_every: int = 10
    reset_every: int | None = None
    noise_features: int = 20
    max_library: int = 25
    crossover_rate: float = 0.5
    mutation_rate: float = 0.4
    elite_frac: float = 0.05
    seed: int = 0


@dataclass(slots=True)
class EvolveResult:
    """What a search returns: a pool, an archive, and the honest caveats."""

    library: list[tuple[Genome, FitnessResult, str]] = field(default_factory=list)
    archive: Archive | None = None
    ledger: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    pareto: np.ndarray | None = None
    context: EvalContext | None = None

    def expressions(self) -> list[str]:
        """Human-readable formula strings for the selected pool."""
        return [expr for _g, _f, expr in self.library]

    def to_frame(self) -> pl.DataFrame:
        """One row per selected feature, with its honest statistics."""
        if not self.library:
            return pl.DataFrame(
                schema={
                    "rank": pl.Int32,
                    "expression": pl.Utf8,
                    "score": pl.Float64,
                    "complexity": pl.Int32,
                    "turnover": pl.Float64,
                    "max_corr": pl.Float64,
                    "coverage": pl.Float64,
                }
            )
        return pl.DataFrame(
            {
                "rank": list(range(1, len(self.library) + 1)),
                "expression": [e for _g, _f, e in self.library],
                "score": [f.score for _g, f, _e in self.library],
                "complexity": [f.complexity for _g, f, _e in self.library],
                "turnover": [f.turnover for _g, f, _e in self.library],
                "max_corr": [f.max_corr_to_library for _g, f, _e in self.library],
                "coverage": [f.coverage for _g, f, _e in self.library],
            },
            schema_overrides={"rank": pl.Int32, "complexity": pl.Int32},
        )

    def summary(self) -> str:
        """A plain-text report that always states the deflated result.

        Deliberately blunt: the raw best score is printed next to the deflated
        one and the trial count, because the raw number on its own is not
        evidence of anything.
        """
        led = self.ledger
        lines = [
            "PanelKit evolutionary feature search",
            "=" * 44,
            f"candidates evaluated (M) : {led.get('n_trials', 'n/a')}",
            f"implied independent (N)  : {led.get('implied_independent_trials', 'n/a')}",
            f"mean pairwise corr (rho) : {led.get('rho_bar', 'n/a')}",
            f"best raw score           : {led.get('best_score', 'n/a')}",
            f"deflated Sharpe (DSR)    : {led.get('deflated_sharpe', 'n/a')}",
            f"P(backtest overfit)      : {led.get('pbo', 'n/a')}",
            f"verdict                  : {led.get('verdict', 'n/a')}",
            "",
            f"features returned        : {len(self.library)}",
        ]
        diag = self.diagnostics.get("verdict")
        if diag:
            lines += ["", f"search diagnostic        : {diag}"]
        return "\n".join(lines)


def _infer_units(columns: Sequence[str]) -> tuple[Unit, ...]:
    """Guess a semantic unit per base column from its name.

    Crude on purpose. Strong typing is what stops the grammar building
    ``price + volume``, but users should not have to hand-annotate a wide
    panel to get started; pass ``base_units`` explicitly to override.
    """
    price_like = ("open", "high", "low", "close", "vwap", "price", "adj")
    volume_like = ("volume", "vol", "turnover", "amount", "shares", "adv")
    ret_like = ("ret", "return", "pct", "chg", "change")
    out: list[Unit] = []
    for c in columns:
        lc = c.lower()
        if any(k in lc for k in ret_like):
            out.append("ret")
        elif any(k in lc for k in volume_like):
            out.append("volume")
        elif any(k in lc for k in price_like):
            out.append("price")
        else:
            out.append("any")
    return tuple(out)


def evolve_features(
    panel: Any,
    *,
    target: str,
    base_columns: Sequence[str] | None = None,
    base_units: Sequence[Unit] | None = None,
    entity: str | None = None,
    time: str | None = None,
    config: EvolveConfig | None = None,
) -> EvolveResult:
    """Evolve a decorrelated pool of leak-safe panel features.

    The search space is closed over PanelKit's leak-safe operator grammar, so
    a look-ahead feature is not merely penalised -- it is unconstructible. Every
    candidate is scored under purged, embargoed cross-validation, every
    evaluation is counted, and the result carries a deflated statistic.

    Parameters
    ----------
    panel : PanelFrame or polars.DataFrame or polars.LazyFrame
        Long-format panel. If a bare frame is passed, ``entity`` and ``time``
        are required.
    target : str
        Column holding the forward-looking label. Build it with
        :func:`polars_features.factor.forward_return` so the horizon is
        explicit -- the embargo is derived from it.
    base_columns : sequence of str, optional
        Terminals the grammar may use. Defaults to every numeric column that
        is not the target or a key.
    base_units : sequence of str, optional
        Semantic unit per base column. Inferred from column names if omitted.
    entity, time : str, optional
        Panel keys, when ``panel`` is not a :class:`~polars_features.PanelFrame`.
    config : EvolveConfig, optional
        Search budget and operators.

    Returns
    -------
    EvolveResult
        The selected pool, the quality-diversity archive, the trial ledger's
        summary, and the search diagnostics.

    Notes
    -----
    Before spending a large budget, check what the panel can support:
    :func:`polars_features.evolve.minimum_backtest_length` implements Bailey,
    Borwein, Lopez de Prado & Zhu's bound. Their example is sobering -- with
    five years of daily data, no more than ~45 *independent* configurations
    should be tried before a Sharpe of 1 becomes essentially guaranteed under
    the null.

    Examples
    --------
    >>> import polars_features as pk  # doctest: +SKIP
    >>> res = pk.evolve.evolve_features(  # doctest: +SKIP
    ...     panel, target="fwd_ret_5d", entity="ticker", time="date"
    ... )
    >>> print(res.summary())  # doctest: +SKIP
    >>> res.to_frame()  # doctest: +SKIP
    """
    # Imported here rather than at module scope so that `import
    # polars_features.evolve` stays cheap and so a partially installed
    # environment fails at call time with a useful message.
    from ._fitness import PanelEvaluator
    from ._genome import (
        canonical_form,
        complexity,
        mutate,
        ramped_population,
        structural_key,
        subgraph_crossover,
        to_infix,
    )
    from ._honest import TrialLedger, search_diagnostics
    from ._ops import default_grammar
    from ._select import Archive, eps_lexicase

    cfg = config or EvolveConfig()
    rng = np.random.default_rng(cfg.seed)

    lf, entity_col, time_col = _resolve_panel(panel, entity, time)
    cols = (
        list(base_columns)
        if base_columns
        else _numeric_columns(lf, exclude={target, entity_col, time_col})
    )
    if not cols:
        raise ValueError("no base columns available; pass `base_columns` explicitly.")
    units = tuple(base_units) if base_units else _infer_units(cols)
    ctx = EvalContext(
        base_columns=tuple(cols),
        base_units=units,
        entity=entity_col,
        time=time_col,
    )
    grammar = default_grammar(units)
    evaluator = PanelEvaluator(lf, ctx=ctx, target=target, seed=cfg.seed)
    ledger = TrialLedger(seed=cfg.seed)
    archive = Archive(seed=cfg.seed)  # CVT binning, annealed thresholds

    islands = [
        ramped_population(
            ctx,
            grammar,
            size=cfg.population,
            seed=int(rng.integers(0, 2**31 - 1)),
            min_genes=cfg.min_genes,
            max_genes=cfg.n_genes,
            max_depth=cfg.max_depth,
        )
        for _ in range(cfg.n_islands)
    ]

    seen: set[int] = set()
    is_scores: list[float] = []
    oos_scores: list[float] = []
    n_failed = 0

    for gen in range(cfg.generations):
        for k, pop in enumerate(islands):
            results = evaluator.evaluate(pop)
            for genome, res in zip(pop, results, strict=True):
                # Split the per-case scores temporally to get an honest
                # in-sample / held-out pair. Cases are ordered (fold x time
                # bucket), so the tail half is the later period -- which is
                # what makes the diagnostics scatter meaningful rather than a
                # tautology.
                ins, oos = _split_cases(res)
                # A candidate that failed to evaluate still consumed a trial.
                # The ledger refuses non-finite scores precisely so that we
                # cannot quietly drop it and understate N -- map it to a
                # neutral 0.0 (no information) and keep counting.
                score = float(res.score)
                if not np.isfinite(score):
                    score = 0.0
                    n_failed += 1
                ledger.record(
                    structural_key(genome),
                    score,
                    held_out=oos if np.isfinite(oos) else 0.0,
                    complexity=res.complexity,
                    generation=gen,
                )
                is_scores.append(ins if np.isfinite(ins) else 0.0)
                oos_scores.append(oos if np.isfinite(oos) else 0.0)
                archive.add(genome, score, _descriptors_of(res))

            cases = np.asarray([r.per_case for r in results], dtype=np.float64)
            parents_ix = eps_lexicase(
                cases,
                cfg.population,
                seed=int(rng.integers(0, 2**31 - 1)),
                downsample=cfg.downsample,
            )
            islands[k] = _breed(
                pop,
                parents_ix,
                grammar,
                ctx,
                cfg,
                rng,
                seen,
                mutate=mutate,
                crossover=subgraph_crossover,
                canonical=canonical_form,
                key=structural_key,
            )

        if cfg.migration_every and (gen + 1) % cfg.migration_every == 0:
            islands = _migrate(islands, rng)

    library = _select_library(archive, ctx, cfg, to_infix, complexity)
    diagnostics = search_diagnostics(
        np.asarray(is_scores, dtype=np.float64),
        np.asarray(oos_scores, dtype=np.float64),
    )
    summary = ledger.summary()
    summary["n_failed_evaluations"] = n_failed
    return EvolveResult(
        library=library,
        archive=archive,
        ledger=summary,
        diagnostics=diagnostics,
        context=ctx,
    )


def _resolve_panel(
    panel: Any, entity: str | None, time: str | None
) -> tuple[pl.LazyFrame, str, str]:
    """Normalise a PanelFrame / DataFrame / LazyFrame to (lazy, entity, time)."""
    ent = getattr(panel, "entity_col", None) or entity
    tim = getattr(panel, "time_col", None) or time
    if ent is None or tim is None:
        raise ValueError(
            "`entity` and `time` are required unless `panel` is a PanelFrame."
        )
    data = getattr(panel, "data", panel)
    if isinstance(data, pl.DataFrame):
        lf = data.lazy()
    elif isinstance(data, pl.LazyFrame):
        lf = data
    else:  # pragma: no cover - defensive
        raise TypeError(f"unsupported panel type: {type(panel)!r}")
    # Rolling windows are silently wrong on an unsorted panel, so sort
    # defensively rather than trusting the caller.
    return lf.sort([ent, tim]), ent, tim


def _numeric_columns(lf: pl.LazyFrame, *, exclude: set[str]) -> list[str]:
    """Numeric columns of `lf`, minus keys and the target."""
    schema = lf.collect_schema()
    return [
        name
        for name, dtype in schema.items()
        if name not in exclude and dtype.is_numeric()
    ]


def _breed(
    pop: list[Genome],
    parents_ix: np.ndarray,
    grammar: Any,
    ctx: EvalContext,
    cfg: EvolveConfig,
    rng: np.random.Generator,
    seen: set[int],
    *,
    mutate: Any,
    crossover: Any,
    canonical: Any,
    key: Any,
) -> list[Genome]:
    """Produce the next island generation, rejecting duplicates.

    Duplicate rejection is cheap and matters more than it looks: a GP
    population is typically 60-90% semantic duplicates, and every duplicate is
    a wasted CV evaluation *and* an extra trial the ledger has to deflate for.
    """
    n_elite = max(1, int(cfg.elite_frac * cfg.population))
    nxt: list[Genome] = [canonical(g) for g in pop[:n_elite]]
    guard = 0
    while len(nxt) < cfg.population and guard < cfg.population * 20:
        guard += 1
        seed = int(rng.integers(0, 2**31 - 1))
        a = pop[int(rng.choice(parents_ix))]
        draw = rng.random()
        if draw < cfg.crossover_rate:
            b = pop[int(rng.choice(parents_ix))]
            child = crossover(a, b, grammar, ctx, seed=seed, max_depth=cfg.max_depth)
        elif draw < cfg.crossover_rate + cfg.mutation_rate:
            child = mutate(a, grammar, ctx, seed=seed, max_depth=cfg.max_depth)
        else:
            child = a
        child = canonical(child)
        k = key(child)
        if k in seen:
            continue
        seen.add(k)
        nxt.append(child)
    # If the guard tripped we are converged; pad with what we have rather than
    # spinning. The archive still holds the diversity.
    while len(nxt) < cfg.population:
        nxt.append(canonical(pop[int(rng.integers(0, len(pop)))]))
    return nxt


def _migrate(
    islands: list[list[Genome]], rng: np.random.Generator
) -> list[list[Genome]]:
    """Ring migration: each island receives the head of its neighbour.

    Kept deliberately simple. Islands exist here to slow convergence onto one
    crowded region of alpha space, which is the failure mode the independent
    AlphaEval benchmark measured across every published miner.
    """
    n = len(islands)
    if n < 2:
        return islands
    share = max(1, len(islands[0]) // 20)
    out = [list(pop) for pop in islands]
    for i in range(n):
        donor = islands[(i - 1) % n]
        out[i][-share:] = donor[:share]
    return out


def _select_library(
    archive: Any,
    ctx: EvalContext,
    cfg: EvolveConfig,
    to_infix: Any,
    complexity: Any,
) -> list[tuple[Genome, FitnessResult, str]]:
    """Pick the returned pool from the archive elites.

    Takes archive elites in score order and caps the pool. Because the archive
    is binned partly on max-correlation-to-library, the elites are already
    spread across distinct behavioural niches rather than being N copies of the
    same momentum signal -- which is the whole reason for using a
    quality-diversity archive instead of a plain hall of fame.
    """
    elites = archive.elites()
    ranked = sorted(elites, key=lambda e: -float(e[1]))[: cfg.max_library]
    out: list[tuple[Genome, FitnessResult, str]] = []
    for genome, score, desc in ranked:
        res = FitnessResult(
            score=float(score),
            per_case=np.asarray([], dtype=np.float64),
            complexity=int(getattr(desc, "complexity", complexity(genome))),
            turnover=float(getattr(desc, "turnover", float("nan"))),
            max_corr_to_library=float(getattr(desc, "max_corr", float("nan"))),
            coverage=float(getattr(desc, "coverage", float("nan"))),
        )
        out.append((genome, res, to_infix(genome, ctx)))
    return out


def _split_cases(res: FitnessResult) -> tuple[float, float]:
    """Split a candidate's per-case scores into (earlier, later) halves.

    The evaluator emits cases ordered by (fold, time bucket), so the second
    half is the later period. Using it as the held-out leg is what lets
    :func:`._honest.search_diagnostics` answer the only question that matters
    about a search as a whole: does in-sample rank predict out-of-sample rank
    at all? If that scatter is round, nothing was found -- and the user should
    learn that from the report rather than from production.
    """
    cases = np.asarray(res.per_case, dtype=np.float64)
    cases = cases[np.isfinite(cases)]
    if cases.size == 0:
        return float(res.score), float(res.score)
    if cases.size == 1:
        return float(cases[0]), float(cases[0])
    half = cases.size // 2
    return float(np.mean(cases[:half])), float(np.mean(cases[half:]))


def _descriptors_of(res: FitnessResult) -> Descriptors:
    """Build archive descriptors from an already-computed fitness result.

    :class:`~._types.FitnessResult` already carries every behavioural axis the
    archive bins on, computed on training folds only, so there is nothing to
    recompute here. ``horizon`` is not separately reported by the evaluator; we
    use turnover as its proxy, since a fast-turning feature is by construction
    a short-horizon one.
    """

    def _finite(x: float, default: float = 0.0) -> float:
        v = float(x)
        return v if np.isfinite(v) else default

    return Descriptors(
        turnover=_finite(res.turnover),
        horizon=_finite(res.turnover),
        max_corr=_finite(res.max_corr_to_library),
        complexity=int(res.complexity),
        coverage=_finite(res.coverage),
    )
