"""Selection, multi-objective sorting and the quality-diversity archive.

Pure NumPy. Nothing here touches Polars, and nothing here evaluates a genome --
this module only decides *who survives*, given a matrix of already-computed
scores.

Three mechanisms live here, each answering a different question:

``eps_lexicase`` / ``lexicase`` / ``tournament``
    *Who becomes a parent?* -- the inner-loop selection operator.
``fast_non_dominated_sort`` / ``crowding_distance`` / ``pareto_front``
    *What do we show the user at the end?* -- NSGA-II primitives used to
    present a Pareto front over (IC, -turnover, -complexity, -max_corr).
``Archive``
    *What do we keep?* -- a MAP-Elites archive with annealed acceptance
    thresholds, so the artefact of a run is a **decorrelated pool** of alphas
    rather than a single best formula.

Why lexicase, and why cases are cells and not rows
--------------------------------------------------
A "case" in :class:`~polars_features.evolve._types.CaseMatrix` is a
*(CV fold x time-bucket)* or *(CV fold x sector)* cell -- never a single panel
row. Lexicase selects on individual cases in a random order rather than on an
aggregate, so a feature that predicts well in one regime survives even if its
pooled IC is mediocre. That is exactly the regime-robust alpha you want, and
precisely what a pooled-IC tournament kills: the pooled mean is dominated by
the many ordinary cells, so the specialist never wins a tournament.

The measured consequence is in this module's own verification script: on a
synthetic population containing one specialist (excellent on 5% of cases, poor
on average), epsilon-lexicase selects it at roughly its "deserved" rate while
tournament selection selects it essentially never.

Why a quality-diversity archive at all
--------------------------------------
The independent AlphaEval benchmark (KDD 2026) found that **every** published
alpha-mining method scores *worse on diversity than randomly generated
formulas* (AlphaQCM 0.477 against a random-formula baseline of 0.981): the
mined pools are internally redundant by construction. Making
max-correlation-to-library an explicit archive axis attacks that directly --
two alphas with the same IC but different correlation profiles occupy different
cells and both survive.

References
----------
* La Cava, Helmuth, Spector & Moore (2019). "A Probabilistic and
  Multi-Objective Analysis of Lexicase and epsilon-Lexicase Selection."
  *Evolutionary Computation* 27(3):377-402. arXiv:1709.05394.
  **Algorithm 3** (semi-dynamic) is what is implemented here; the authors
  recommend it as the default.
* Geiger, Sobania & Rothlauf (2023). "Down-Sampled Epsilon-Lexicase Selection
  for Real-World Symbolic Regression Problems." arXiv:2302.04301.
* Deb, Pratap, Agarwal & Meyarivan (2002). "A Fast and Elitist Multiobjective
  Genetic Algorithm: NSGA-II." *IEEE Trans. Evol. Comput.* 6(2):182-197.
* Mouret & Clune (2015). "Illuminating search spaces by mapping elites."
  arXiv:1504.04909.
* Vassiliades, Chatzilygeroudis & Mouret (2018). "Using centroidal Voronoi
  tessellations to scale up the multi-dimensional archive of phenotypic elites
  algorithm." *IEEE Trans. Evol. Comput.* 22(4):623-630.
* Fontaine & Nikolaidis (2023). "Covariance Matrix Adaptation MAP-Annealing."
  GECCO '23. arXiv:2205.10752. (Only the annealed-threshold *archive rule* is
  taken; see :class:`Archive`.)
* Schmidt & Lipson (2011). "Age-Fitness Pareto Optimization." In *Genetic
  Programming Theory and Practice VIII*, 129-146.
* Luke & Panait (2002). "Lexicographic Parsimony Pressure." GECCO '02.

Clean-room note: all of the above are implemented from the published
descriptions. In particular DEAP's ``selAutomaticEpsilonLexicase`` is the
*dynamic* variant (epsilon recomputed on the shrinking pool) and is LGPL --
neither its behaviour nor its code is used here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

import numpy as np

from polars_features.evolve._types import (
    CaseMatrix,
    Descriptors,
    FitnessResult,
    Genome,
)

__all__ = [
    # selection
    "eps_lexicase",
    "eps_lexicase_with_diagnostics",
    "lexicase",
    "tournament",
    "LexicaseDiagnostics",
    # multi-objective
    "fast_non_dominated_sort",
    "crowding_distance",
    "pareto_front",
    "nsga2_survival",
    "objectives_from_results",
    "case_scores_from_results",
    # quality-diversity
    "Archive",
    "DEFAULT_DESCRIPTOR_BOUNDS",
    # ALPS
    "age_fitness_pareto_survival",
    "AFPO_ENABLED_BY_DEFAULT",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_scores(scores: np.ndarray | CaseMatrix) -> np.ndarray:
    """Coerce input to a float64 ``(n_pop, n_cases)`` *higher-is-better* array."""
    raw = scores.scores if isinstance(scores, CaseMatrix) else scores
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"scores must be 1-D or 2-D, got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"scores must be non-empty, got shape {arr.shape}")
    return arr


def _errors_and_epsilon(
    scores: np.ndarray, *, use_epsilon: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Convert higher-is-better scores to errors and compute the MAD epsilon.

    The lexicase literature is written for *error* (lower is better); PanelKit
    fitness is IC-like (higher is better). We negate once, here, so that the
    rest of the implementation reads exactly like the papers.

    ``epsilon_t = median(|e_t - median(e_t)|)`` -- the median absolute
    deviation over the **whole population**, Eqn. 2 of La Cava et al. MAD, not
    the standard deviation: MAD is robust to the handful of individuals with
    pathological errors that every GP population contains, and it is what makes
    epsilon a meaningful "within measurement noise" band rather than a
    reflection of the worst individual.
    """
    errors = -scores
    if use_epsilon:
        med = np.nanmedian(errors, axis=0)
        mad = np.nanmedian(np.abs(errors - med), axis=0)
        eps = np.nan_to_num(mad, nan=0.0, posinf=0.0, neginf=0.0)
        eps = np.maximum(eps, 0.0)
    else:
        eps = np.zeros(errors.shape[1], dtype=np.float64)
    # Non-finite errors (a genome that produced nulls on this case) are the
    # worst possible outcome, never the best. Do this *after* the MAD so a few
    # broken individuals cannot inflate epsilon.
    filled = np.where(np.isfinite(errors), errors, np.inf)
    return filled, eps


@dataclass(frozen=True, slots=True)
class LexicaseDiagnostics:
    """Monitoring output for a lexicase selection event batch.

    Down-sampling is a real trade: Geiger et al. (arXiv:2302.04301) measured it
    *reducing behavioural diversity* even while improving test error. These
    numbers are what you watch to decide whether to back the down-sample rate
    off. ``unique_fraction`` and ``selection_entropy`` are the diversity
    canaries; if either collapses toward zero, raise ``downsample``.
    """

    n_pop: int
    n_cases_total: int
    n_cases_used: int
    #: MAD epsilon per *used* case, in error units.
    epsilon: np.ndarray
    #: Mean number of cases consumed before the pool collapsed. Low values mean
    #: selection is effectively single-case (near-random); high values mean
    #: epsilon is too wide and selection is barely filtering.
    mean_cases_consumed: float
    #: Mean size of the surviving pool when selection stopped. 1.0 means every
    #: event resolved to a unique winner.
    mean_final_pool: float
    n_unique_selected: int
    #: ``n_unique_selected / min(n_select, n_pop)`` -- 1.0 means every
    #: individual that *could* have been a distinct parent was one.
    unique_fraction: float
    #: Shannon entropy of the selection-count distribution, normalised by
    #: ``log(n_pop)``. 1.0 = uniform over the population, 0.0 = one parent.
    selection_entropy: float


def _lexicase_events(
    scores: np.ndarray | CaseMatrix,
    n_select: int,
    *,
    seed: int,
    downsample: float,
    use_epsilon: bool,
) -> tuple[np.ndarray, LexicaseDiagnostics]:
    """Shared engine for :func:`eps_lexicase` and :func:`lexicase`."""
    arr = _as_scores(scores)
    n_pop, n_cases = arr.shape
    if n_select < 0:
        raise ValueError(f"n_select must be >= 0, got {n_select}")
    if not 0.0 < downsample <= 1.0:
        raise ValueError(f"downsample must be in (0, 1], got {downsample}")

    rng = np.random.default_rng(seed)

    # --- down-sampling -----------------------------------------------------
    # One random case subset per *call* (i.e. per generation), shared by every
    # selection event, exactly as in down-sampled lexicase. The per-event
    # randomness is the case *ordering*, not the case *subset*.
    n_used = max(1, int(round(downsample * n_cases)))
    if n_used >= n_cases:
        case_ix = np.arange(n_cases)
    else:
        case_ix = np.sort(rng.choice(n_cases, size=n_used, replace=False))

    sub = arr[:, case_ix]
    errors, eps = _errors_and_epsilon(sub, use_epsilon=use_epsilon)
    n_sub = case_ix.size

    selected = np.empty(n_select, dtype=np.int64)
    consumed = np.empty(n_select, dtype=np.float64)
    final_pool = np.empty(n_select, dtype=np.float64)
    all_ix = np.arange(n_pop, dtype=np.int64)

    for k in range(n_select):
        pool = all_ix
        order = rng.permutation(n_sub)
        used = 0
        for c in order:
            if pool.size <= 1:
                break
            col = errors[pool, c]
            # Semi-dynamic (Algorithm 3): the elite is the best in the
            # *current pool*, while epsilon was computed over the *whole
            # population* once per generation. The fully dynamic variant
            # recomputes epsilon on the pool too and is what DEAP ships; La
            # Cava et al. report semi-dynamic as the better default and it is
            # an order of magnitude cheaper.
            elite = col.min()
            pool = pool[col <= elite + eps[c]]
            used += 1
        winner = pool[0] if pool.size == 1 else pool[rng.integers(pool.size)]
        selected[k] = winner
        consumed[k] = used
        final_pool[k] = pool.size

    if n_select == 0:
        diag = LexicaseDiagnostics(
            n_pop=n_pop,
            n_cases_total=n_cases,
            n_cases_used=n_sub,
            epsilon=eps,
            mean_cases_consumed=0.0,
            mean_final_pool=0.0,
            n_unique_selected=0,
            unique_fraction=0.0,
            selection_entropy=0.0,
        )
        return selected, diag

    counts = np.bincount(selected, minlength=n_pop).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    ent = float(-(p * np.log(p)).sum())
    norm = float(np.log(n_pop)) if n_pop > 1 else 1.0
    diag = LexicaseDiagnostics(
        n_pop=n_pop,
        n_cases_total=n_cases,
        n_cases_used=n_sub,
        epsilon=eps,
        mean_cases_consumed=float(consumed.mean()),
        mean_final_pool=float(final_pool.mean()),
        n_unique_selected=int((counts > 0).sum()),
        unique_fraction=float((counts > 0).sum()) / float(min(n_select, n_pop)),
        selection_entropy=ent / norm if norm > 0 else 0.0,
    )
    return selected, diag


# --------------------------------------------------------------------------- #
# 1. lexicase family
# --------------------------------------------------------------------------- #
def eps_lexicase(
    scores: np.ndarray | CaseMatrix,
    n_select: int,
    *,
    seed: int,
    downsample: float = 0.1,
) -> np.ndarray:
    """Semi-dynamic epsilon-lexicase selection with down-sampling.

    Implements **Algorithm 3** of La Cava, Helmuth, Spector & Moore,
    "A Probabilistic and Multi-Objective Analysis of Lexicase and
    epsilon-Lexicase Selection", *Evolutionary Computation* 27(3), 2019
    (arXiv:1709.05394) -- the variant the authors recommend as the default.

    Each selection event:

    1. starts with the whole population as the pool;
    2. draws a **fresh random ordering of the cases** -- that shuffle *is* the
       mechanism, not an implementation detail: it is what makes a different
       specialist win each event and what gives lexicase its diversity;
    3. walks the cases, keeping only individuals within ``epsilon_t`` of the
       best error **in the current pool** (semi-dynamic);
    4. stops when one individual remains or the cases run out, breaking any
       remaining tie uniformly at random.

    ``epsilon_t`` is the median absolute deviation of case ``t``'s errors over
    the **whole population**, computed once per call (Eqn. 2). MAD, not the
    standard deviation.

    Parameters
    ----------
    scores : ndarray of shape (n_pop, n_cases), or CaseMatrix
        **Higher is better** -- PanelKit's convention, e.g. per-cell rank IC.
        The papers are written for error (lower is better); this function
        negates internally, so do not pre-negate. Non-finite entries are
        treated as the worst possible outcome and are excluded from the MAD.
        A 1-D array is read as a single case.
    n_select : int
        Number of selection events, i.e. number of parents to return. Sampling
        is with replacement, as in every lexicase implementation.
    seed : int
        Explicit seed; a ``numpy.random.Generator`` is constructed internally.
        Passing the same seed reproduces the selection exactly.
    downsample : float, default 0.1
        Fraction of cases used this generation, sampled once per call and
        shared by all selection events. Geiger, Sobania & Rothlauf
        (arXiv:2302.04301) found down-sampled epsilon-lexicase at ``s=0.1``
        beat standard epsilon-lexicase on 5 of 6 UCI regression benchmarks,
        improving median test MSE by up to ~87%. It also *measurably reduces
        behavioural diversity*, so use
        :func:`eps_lexicase_with_diagnostics` to monitor
        ``unique_fraction`` / ``selection_entropy`` and back the rate off if
        they collapse. ``1.0`` disables down-sampling.

    Returns
    -------
    ndarray of shape (n_select,), dtype int64
        Row indices into ``scores``, with repeats.

    Notes
    -----
    A "case" here is a *(CV fold x time-bucket)* or *(fold x sector)* cell, not
    a panel row. That is the point of the whole mechanism: a feature that
    predicts well in one regime survives even when its pooled IC is mediocre --
    exactly the alpha you want, and exactly what a pooled-IC tournament kills.

    See Also
    --------
    eps_lexicase_with_diagnostics : same selection, plus monitoring output.
    lexicase : the ``epsilon = 0`` ancestor, for ablation.
    tournament : the aggregate-fitness baseline, for ablation.
    """
    idx, _ = _lexicase_events(
        scores, n_select, seed=seed, downsample=downsample, use_epsilon=True
    )
    return idx


def eps_lexicase_with_diagnostics(
    scores: np.ndarray | CaseMatrix,
    n_select: int,
    *,
    seed: int,
    downsample: float = 0.1,
) -> tuple[np.ndarray, LexicaseDiagnostics]:
    """:func:`eps_lexicase`, additionally returning a
    :class:`LexicaseDiagnostics`.

    Identical selection for identical arguments -- the diagnostics are free.
    Use this in the search loop so that the down-sampling rate can be monitored
    (and backed off) rather than set and forgotten.
    """
    return _lexicase_events(
        scores, n_select, seed=seed, downsample=downsample, use_epsilon=True
    )


def lexicase(
    scores: np.ndarray | CaseMatrix,
    n_select: int,
    *,
    seed: int,
    downsample: float = 1.0,
) -> np.ndarray:
    """Standard (non-epsilon) lexicase selection -- Spector (2012).

    Identical to :func:`eps_lexicase` with ``epsilon = 0``: only individuals
    that *exactly* match the pool elite on a case survive it.

    Provided for **ablation**, not for production use on continuous fitness.
    With float scores, exact ties are vanishingly rare, so the pool almost
    always collapses to a single individual on the first case and selection
    degenerates to "pick a uniformly random case, take its argmax". That
    degeneration is precisely why La Cava et al. introduced the epsilon
    relaxation, and running both is how you demonstrate it on your own data.

    Parameters
    ----------
    scores, n_select, seed, downsample
        As :func:`eps_lexicase`; ``downsample`` defaults to ``1.0`` (no
        down-sampling) so this is a clean epsilon-only ablation.

    Returns
    -------
    ndarray of shape (n_select,), dtype int64
    """
    idx, _ = _lexicase_events(
        scores, n_select, seed=seed, downsample=downsample, use_epsilon=False
    )
    return idx


def tournament(
    scores: np.ndarray | CaseMatrix,
    n_select: int,
    *,
    seed: int,
    size: int = 7,
    complexity: np.ndarray | Sequence[int] | None = None,
    parsimony_size: int = 2,
    parsimony_prob: float = 0.7,
) -> np.ndarray:
    """Aggregate-fitness tournament selection, with an optional double
    tournament for parsimony pressure.

    Provided as the **ablation baseline** against which lexicase is judged.
    Fitness is the mean over cases (``nanmean``), which is exactly the pooled
    statistic lexicase refuses to compute -- a specialist that is excellent on
    5% of cases and poor elsewhere has a mediocre mean and will essentially
    never win a tournament. Measuring that gap on your own case matrix is the
    point of keeping this function.

    Parsimony uses the *double tournament* of Luke & Panait, "Lexicographic
    Parsimony Pressure" (GECCO '02): run ``parsimony_size`` independent fitness
    tournaments, then hold a second tournament among their winners in which the
    smaller individual wins with probability ``parsimony_prob``. This keeps
    size pressure from overwhelming fitness pressure the way a raw
    fitness-minus-lambda-times-size penalty does.

    Parameters
    ----------
    scores : ndarray of shape (n_pop, n_cases) or (n_pop,), or CaseMatrix
        Higher is better. Averaged over cases with ``nanmean``; an individual
        that is non-finite on every case gets ``-inf``.
    n_select : int
        Number of parents to return (with replacement).
    seed : int
        Explicit seed for the internal generator.
    size : int, default 7
        Tournament size. Larger = higher selection pressure.
    complexity : array-like of int, optional
        Per-individual active node count. When given, the double tournament is
        enabled; when ``None``, a plain fitness tournament is run.
    parsimony_size : int, default 2
        Number of fitness tournaments feeding the parsimony round.
    parsimony_prob : float, default 0.7
        Probability that the smaller individual wins the parsimony round.
        ``0.5`` disables size pressure; ``1.0`` makes it lexicographic.

    Returns
    -------
    ndarray of shape (n_select,), dtype int64
    """
    arr = _as_scores(scores)
    n_pop = arr.shape[0]
    if n_select < 0:
        raise ValueError(f"n_select must be >= 0, got {n_select}")
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    if parsimony_size < 1:
        raise ValueError(f"parsimony_size must be >= 1, got {parsimony_size}")
    if not 0.0 <= parsimony_prob <= 1.0:
        raise ValueError(f"parsimony_prob must be in [0, 1], got {parsimony_prob}")

    # Mean over cases, ignoring non-finite cells; an individual that is
    # non-finite everywhere gets -inf and can never win a heat.
    finite = np.isfinite(arr)
    n_finite = finite.sum(axis=1)
    totals = np.where(finite, arr, 0.0).sum(axis=1)
    fitness = np.where(n_finite > 0, totals / np.maximum(n_finite, 1), -np.inf)

    rng = np.random.default_rng(seed)

    if complexity is None:
        heats = rng.integers(0, n_pop, size=(n_select, size))
        best = heats[np.arange(n_select), np.argmax(fitness[heats], axis=1)]
        return best.astype(np.int64)

    size_vec = np.asarray(complexity, dtype=np.float64)
    if size_vec.shape != (n_pop,):
        raise ValueError(f"complexity must have shape ({n_pop},), got {size_vec.shape}")

    # `parsimony_size` independent fitness heats, then one parsimony heat.
    heats = rng.integers(0, n_pop, size=(n_select, parsimony_size, size))
    winners = np.take_along_axis(
        heats, np.argmax(fitness[heats], axis=2)[:, :, None], axis=2
    )[:, :, 0]
    smallest = winners[np.arange(n_select), np.argmin(size_vec[winners], axis=1)]
    random_pick = winners[
        np.arange(n_select), rng.integers(0, parsimony_size, size=n_select)
    ]
    take_small = rng.random(n_select) < parsimony_prob
    return np.where(take_small, smallest, random_pick).astype(np.int64)


# --------------------------------------------------------------------------- #
# 2. NSGA-II primitives
# --------------------------------------------------------------------------- #
def _clean_objectives(objectives: np.ndarray) -> np.ndarray:
    """float64 (n, m) with non-finite entries pushed to the worst value."""
    obj = np.asarray(objectives, dtype=np.float64)
    if obj.ndim == 1:
        obj = obj[:, None]
    if obj.ndim != 2:
        raise ValueError(f"objectives must be 1-D or 2-D, got shape {obj.shape}")
    # NaN would silently make a solution both non-dominating and non-dominated
    # (every comparison is False), landing it in front 0. Map it to -inf, i.e.
    # "worst possible on this objective", which is the honest reading.
    return np.where(np.isnan(obj), -np.inf, obj)


def _domination_counts(obj: np.ndarray, *, chunk: int = 512) -> np.ndarray:
    """Boolean ``(n, n)`` matrix where ``d[i, j]`` = "i dominates j".

    Built in row chunks: the naive one-shot broadcast allocates an
    ``(n, n, m)`` boolean temporary, which is the only part of NSGA-II that
    actually blows up on a few thousand individuals.
    """
    n = obj.shape[0]
    dom = np.zeros((n, n), dtype=bool)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = obj[start:stop, None, :]
        ge = (block >= obj[None, :, :]).all(axis=2)
        gt = (block > obj[None, :, :]).any(axis=2)
        dom[start:stop] = ge & gt
    return dom


def fast_non_dominated_sort(objectives: np.ndarray) -> list[np.ndarray]:
    """Sort individuals into Pareto fronts (Deb et al. 2002, NSGA-II).

    **Convention: every column is maximised.** ``i`` dominates ``j`` iff ``i``
    is no worse on all objectives and strictly better on at least one. Build
    the input with sign flips already applied -- the canonical PanelKit report
    front is ``(IC, -turnover, -complexity, -max_corr)``; see
    :func:`objectives_from_results`.

    Parameters
    ----------
    objectives : ndarray of shape (n, m)
        Higher is better in every column. ``NaN`` is treated as ``-inf``
        (worst) rather than as incomparable. A 1-D array is read as one
        objective.

    Returns
    -------
    list of ndarray
        ``fronts[0]`` is the non-dominated (Pareto) front, ``fronts[1]`` the
        front once ``fronts[0]`` is removed, and so on. Each element is an
        int64 array of ascending row indices; the concatenation of all fronts
        is a permutation of ``range(n)``.

    Examples
    --------
    >>> import numpy as np
    >>> obj = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    >>> [f.tolist() for f in fast_non_dominated_sort(obj)]
    [[0, 1], [2]]
    """
    obj = _clean_objectives(objectives)
    n = obj.shape[0]
    if n == 0:
        return []
    dom = _domination_counts(obj)
    # n_dominated[j] = how many individuals dominate j
    n_dominated = dom.sum(axis=0).astype(np.int64)

    fronts: list[np.ndarray] = []
    current = np.flatnonzero(n_dominated == 0).astype(np.int64)
    while current.size:
        fronts.append(current)
        # Peel: remove this front, decrement the counts of everyone it
        # dominated, and mark it as assigned so it cannot resurface.
        n_dominated -= dom[current].sum(axis=0).astype(np.int64)
        n_dominated[current] = -1
        current = np.flatnonzero(n_dominated == 0).astype(np.int64)
    return fronts


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance (Deb et al. 2002, Section III-B).

    The Manhattan perimeter of the cuboid spanned by a solution's two nearest
    neighbours in objective space, with each objective normalised by its range
    over the supplied set. Boundary solutions get ``inf`` so the extremes of a
    front are never discarded.

    Pass **one front at a time** -- the measure is only meaningful within a
    front, because normalisation is relative to the set you give it.

    Parameters
    ----------
    objectives : ndarray of shape (n, m)
        Higher is better per column (the value of the measure is unaffected by
        the sign convention, but keep it consistent with
        :func:`fast_non_dominated_sort`). ``NaN`` is treated as ``-inf``.

    Returns
    -------
    ndarray of shape (n,), dtype float64
        Aligned with the input rows. Objectives with zero range contribute
        nothing (rather than dividing by zero).

    Examples
    --------
    >>> import numpy as np
    >>> d = crowding_distance(np.array([[0.0], [0.5], [1.0]]))
    >>> bool(np.isinf(d[0])), bool(np.isinf(d[2])), float(d[1])
    (True, True, 1.0)
    """
    obj = _clean_objectives(objectives)
    n, m = obj.shape
    dist = np.zeros(n, dtype=np.float64)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        col = obj[:, j]
        order = np.argsort(col, kind="stable")
        lo = col[order[0]]
        hi = col[order[-1]]
        span = hi - lo
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        if not np.isfinite(span) or span <= 0.0:
            continue
        inner = order[1:-1]
        dist[inner] += (col[order[2:]] - col[order[:-2]]) / span
    return dist


def pareto_front(objectives: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated set (front 0).

    Same maximise-everything convention as :func:`fast_non_dominated_sort`.

    Parameters
    ----------
    objectives : ndarray of shape (n, m)

    Returns
    -------
    ndarray of int64, ascending. Empty when ``objectives`` has no rows.
    """
    fronts = fast_non_dominated_sort(objectives)
    return fronts[0] if fronts else np.empty(0, dtype=np.int64)


def nsga2_survival(
    objectives: np.ndarray, n_survivors: int, *, seed: int = 0
) -> np.ndarray:
    """Rank-then-crowding truncation, the NSGA-II environmental selection step.

    Fronts are admitted whole while they fit; the front that overflows is
    truncated by descending crowding distance, with ties broken by a seeded
    permutation so the result is deterministic but not order-biased.

    Parameters
    ----------
    objectives : ndarray of shape (n, m)
        Higher is better per column.
    n_survivors : int
    seed : int, default 0

    Returns
    -------
    ndarray of int64 of length ``min(n_survivors, n)``, best front first.
    """
    obj = _clean_objectives(objectives)
    n = obj.shape[0]
    n_survivors = int(min(max(n_survivors, 0), n))
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    filled = 0
    for front in fast_non_dominated_sort(obj):
        if filled + front.size <= n_survivors:
            out.append(front)
            filled += front.size
            if filled == n_survivors:
                break
            continue
        room = n_survivors - filled
        if room > 0:
            dist = crowding_distance(obj[front])
            jitter = rng.permutation(front.size)
            order = np.lexsort((jitter, -dist))
            out.append(front[order[:room]])
        break
    if not out:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(out).astype(np.int64)


def objectives_from_results(
    results: Sequence[FitnessResult],
) -> np.ndarray:
    """Stack :class:`FitnessResult` into the report objective matrix.

    Columns, all **maximised**: ``(score, -turnover, -complexity, -max_corr)``
    -- predictive power, tradability, simplicity, and decorrelation from the
    existing library. This is the front shown in the final report; it is
    deliberately *not* used to drive the inner loop, which selects with
    :func:`eps_lexicase`.

    Parameters
    ----------
    results : sequence of FitnessResult

    Returns
    -------
    ndarray of shape (len(results), 4), dtype float64
    """
    if not results:
        return np.zeros((0, 4), dtype=np.float64)
    return np.array(
        [
            (r.score, -r.turnover, -float(r.complexity), -r.max_corr_to_library)
            for r in results
        ],
        dtype=np.float64,
    )


def case_scores_from_results(results: Sequence[FitnessResult]) -> np.ndarray:
    """Stack ``FitnessResult.per_case`` into a ``(n_pop, n_cases)`` matrix.

    Raises
    ------
    ValueError
        If the per-case vectors are not all the same length -- a ragged case
        matrix means two individuals were scored on different folds, which
        would make lexicase comparisons meaningless.
    """
    if not results:
        return np.zeros((0, 0), dtype=np.float64)
    rows = [np.asarray(r.per_case, dtype=np.float64).ravel() for r in results]
    widths = {r.size for r in rows}
    if len(widths) != 1:
        raise ValueError(
            f"per_case vectors have inconsistent lengths: {sorted(widths)}"
        )
    return np.vstack(rows)


# --------------------------------------------------------------------------- #
# 3. MAP-Elites archive
# --------------------------------------------------------------------------- #
#: Bounds used to normalise :meth:`Descriptors.as_vector` into the unit cube,
#: in field order ``(turnover, horizon, max_corr, complexity, coverage)``.
#: ``turnover`` is ``1 - rank autocorrelation`` and lives in ``[0, 2]``;
#: ``horizon`` is in trading days; ``complexity`` is the active node count and
#: is bounded in practice by the depth cap. Values outside the bounds are
#: clipped, never rejected.
DEFAULT_DESCRIPTOR_BOUNDS: tuple[tuple[float, float], ...] = (
    (0.0, 2.0),
    (1.0, 60.0),
    (0.0, 1.0),
    (1.0, 64.0),
    (0.0, 1.0),
)


def _nearest_centroid(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Index of the nearest centroid for each row, computed in row chunks."""
    n = points.shape[0]
    out = np.empty(n, dtype=np.int64)
    c_sq = (centroids**2).sum(axis=1)
    step = max(1, int(2**22 // max(centroids.shape[0], 1)))
    for start in range(0, n, step):
        stop = min(start + step, n)
        block = points[start:stop]
        d2 = c_sq[None, :] - 2.0 * (block @ centroids.T)
        out[start:stop] = np.argmin(d2, axis=1)
    return out


def _kmeans_pp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding (Arthur & Vassilvitskii 2007), pure numpy."""
    n = x.shape[0]
    centroids = np.empty((k, x.shape[1]), dtype=np.float64)
    first = int(rng.integers(n))
    centroids[0] = x[first]
    closest = ((x - centroids[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = float(closest.sum())
        if total <= 0.0:
            centroids[i:] = x[rng.integers(0, n, size=k - i)]
            break
        probs = closest / total
        pick = int(rng.choice(n, p=probs))
        centroids[i] = x[pick]
        closest = np.minimum(closest, ((x - centroids[i]) ** 2).sum(axis=1))
    return centroids


@lru_cache(maxsize=8)
def _cvt_centroids(
    k: int, n_dims: int, seed: int, n_samples: int, n_iter: int
) -> np.ndarray:
    """Lloyd's algorithm on uniform samples of the unit cube -- a CVT.

    Pure numpy on purpose: importing scikit-learn for k-means would violate
    PanelKit's ``{numpy, polars}`` core (see ``AGENTS.md``). Cached because the
    tessellation depends only on ``(k, n_dims, seed, ...)``, never on the data,
    so every :class:`Archive` with the same parameters shares one -- which is
    also what makes cell ids comparable across runs.
    """
    rng = np.random.default_rng(seed)
    x = rng.random((n_samples, n_dims))
    centroids = _kmeans_pp_init(x, k, rng)
    for _ in range(n_iter):
        assign = _nearest_centroid(x, centroids)
        counts = np.bincount(assign, minlength=k).astype(np.float64)
        new = np.empty_like(centroids)
        for d in range(n_dims):
            new[:, d] = np.bincount(assign, weights=x[:, d], minlength=k)
        nonempty = counts > 0
        new[nonempty] /= counts[nonempty, None]
        # Re-seed empty cells onto the sample points furthest from any
        # centroid -- deterministic, and it keeps |centroids| == k.
        n_empty = int((~nonempty).sum())
        if n_empty:
            d2 = ((x - centroids[assign]) ** 2).sum(axis=1)
            far = np.argsort(-d2, kind="stable")[:n_empty]
            new[~nonempty] = x[far]
        shift = float(np.abs(new - centroids).max())
        centroids = new
        if shift < 1e-9:
            break
    # Sorting makes cell ids a stable function of the tessellation itself
    # rather than of k-means++ draw order.
    order = np.lexsort(tuple(centroids[:, d] for d in range(n_dims - 1, -1, -1)))
    out = np.ascontiguousarray(centroids[order])
    out.flags.writeable = False  # cached and shared; never mutate in place
    return out


@dataclass(slots=True)
class _Cell:
    """One archive niche: an incumbent elite plus its annealed threshold."""

    threshold: float
    genome: Genome | None = None
    fitness: float = -np.inf
    descriptors: Descriptors | None = None
    n_accepted: int = 0
    n_attempts: int = 0


@dataclass(slots=True)
class Archive:
    """MAP-Elites archive with CMA-MAE annealed acceptance thresholds.

    The artefact of an evolutionary alpha search should be a **decorrelated
    pool**, not one best formula. The independent AlphaEval benchmark
    (KDD 2026) found that every published alpha-mining method is *less diverse*
    than randomly generated formulas (AlphaQCM 0.477 against a random baseline
    of 0.981) -- their pools are internally redundant by construction. Making
    ``max_corr`` (correlation to the existing library) an archive axis attacks
    that directly: two features with equal IC but different correlation
    profiles fall in different cells and both survive.

    Binning
    -------
    ``binning="grid"``
        The original Mouret & Clune (arXiv:1504.04909) layout: an independent
        bin count per descriptor. Fine for 2-3 descriptors.
    ``binning="cvt"`` (default)
        A centroidal Voronoi tessellation with ``n_centroids`` cells
        (Vassiliades et al. 2018). **Essential here**: PanelKit uses 5
        descriptors, and a 5x10 grid is 100,000 cells that a realistic budget
        of a few thousand evaluations will never fill -- coverage would read as
        ~1% forever and the archive would degenerate into a list. A CVT
        decouples cell count from descriptor dimensionality; k in the 500-1000
        range is the useful window. The tessellation is computed once by pure
        numpy k-means over uniform samples of the unit cube and cached, so it
        is reproducible from ``seed`` alone and identical across runs.

    Annealed thresholds
    -------------------
    Each cell carries ``t_e``, initialised to ``min_f``. A candidate is
    accepted iff ``f > t_e``; on acceptance ``t_e <- (1 - alpha) * t_e +
    alpha * f`` (Fontaine & Nikolaidis, CMA-MAE, GECCO '23, arXiv:2205.10752;
    the paper uses ``alpha = 0.01``). With ``alpha = 1`` this collapses to
    vanilla MAP-Elites ("beat the incumbent"); with a small ``alpha`` the
    acceptance bar lags behind the incumbent, so a cell keeps returning
    improvement signal instead of going silent the moment it holds a good
    elite -- which is what stops the search from abandoning hard niches early.

    **Only the archive rule is taken from CMA-MAE, not the CMA part.** CMA-ES
    needs a continuous, fixed-dimension search vector; our genome is a discrete
    straight-line program, so there is no covariance matrix to adapt. The
    emitter side of that paper simply does not apply.

    The archive itself always stores the **best-ever** solution per cell, so
    :meth:`elites` never regresses even though the acceptance bar is
    deliberately loose.

    Parameters
    ----------
    binning : {"cvt", "grid"}, default "cvt"
    n_centroids : int, default 512
        Number of CVT cells. Ignored for ``binning="grid"``.
    bins : int or sequence of int, default 10
        Bins per descriptor for ``binning="grid"``. Ignored for CVT.
    n_dims : int, default 5
        Descriptor dimensionality; must match ``Descriptors.as_vector()``.
    bounds : sequence of (float, float), optional
        Descriptor bounds used for normalisation, defaulting to
        :data:`DEFAULT_DESCRIPTOR_BOUNDS`. Out-of-range values are clipped.
    min_f : float, default 0.0
        Initial threshold for every cell, i.e. the fitness floor below which
        nothing is archived. ``0.0`` is the natural floor for an IC-like score:
        "no better than random".
    alpha : float, default 0.01
        Threshold learning rate, per the CMA-MAE paper. ``1.0`` recovers
        vanilla MAP-Elites.
    seed : int, default 0
        Seeds the CVT. No other randomness exists in this class.
    n_samples, n_iter : int
        CVT k-means sampling budget and iteration cap.

    Examples
    --------
    >>> arc = Archive(binning="grid", bins=4, n_dims=5, min_f=0.0)
    >>> d = Descriptors(turnover=0.5, horizon=5.0, max_corr=0.1,
    ...                 complexity=8, coverage=0.9)
    >>> arc.add(None, 0.05, d)          # doctest: +SKIP
    True
    """

    binning: Literal["cvt", "grid"] = "cvt"
    n_centroids: int = 512
    bins: int | Sequence[int] = 10
    n_dims: int = 5
    bounds: Sequence[tuple[float, float]] | None = None
    min_f: float = 0.0
    alpha: float = 0.01
    seed: int = 0
    n_samples: int = 10_000
    n_iter: int = 30

    _lo: np.ndarray = field(init=False, repr=False)
    _hi: np.ndarray = field(init=False, repr=False)
    _bins: np.ndarray = field(init=False, repr=False)
    _centroids: np.ndarray | None = field(init=False, repr=False, default=None)
    _cells: dict[int, _Cell] = field(init=False, repr=False, default_factory=dict)
    _n_added: int = field(init=False, repr=False, default=0)
    _n_accepted: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        if self.n_dims < 1:
            raise ValueError(f"n_dims must be >= 1, got {self.n_dims}")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        bounds = self.bounds if self.bounds is not None else DEFAULT_DESCRIPTOR_BOUNDS
        arr = np.asarray(bounds, dtype=np.float64)
        if arr.shape != (self.n_dims, 2):
            raise ValueError(
                f"bounds must have shape ({self.n_dims}, 2), got {arr.shape}"
            )
        self._lo = arr[:, 0]
        self._hi = arr[:, 1]
        if not np.all(self._hi > self._lo):
            raise ValueError("every bound must satisfy hi > lo")

        if self.binning == "grid":
            b = np.broadcast_to(
                np.asarray(self.bins, dtype=np.int64), (self.n_dims,)
            ).copy()
            if np.any(b < 1):
                raise ValueError(f"bins must all be >= 1, got {self.bins!r}")
            self._bins = b
        elif self.binning == "cvt":
            if self.n_centroids < 1:
                raise ValueError(f"n_centroids must be >= 1, got {self.n_centroids}")
            if self.n_centroids > self.n_samples:
                raise ValueError(
                    "n_centroids must be <= n_samples "
                    f"({self.n_centroids} > {self.n_samples})"
                )
            self._bins = np.zeros(self.n_dims, dtype=np.int64)
            self._centroids = _cvt_centroids(
                int(self.n_centroids),
                int(self.n_dims),
                int(self.seed),
                int(self.n_samples),
                int(self.n_iter),
            )
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"unknown binning {self.binning!r}")

    # -- geometry ----------------------------------------------------------
    @property
    def n_cells(self) -> int:
        """Total number of cells, occupied or not."""
        if self.binning == "cvt":
            return int(self.n_centroids)
        return int(np.prod(self._bins))

    def _normalise(self, vector: Sequence[float]) -> np.ndarray:
        v = np.asarray(vector, dtype=np.float64).ravel()
        if v.size != self.n_dims:
            raise ValueError(
                f"descriptor vector has length {v.size}, expected {self.n_dims}"
            )
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        u = (v - self._lo) / (self._hi - self._lo)
        return np.clip(u, 0.0, 1.0)

    def cell_of(self, descriptors: Descriptors | Sequence[float]) -> int:
        """Cell id for a descriptor vector.

        Deterministic given ``seed``: the CVT is a cached, data-independent
        function of ``(n_centroids, n_dims, seed, n_samples, n_iter)``, so the
        same descriptor maps to the same cell in every run and across
        processes.
        """
        vec = (
            descriptors.as_vector()
            if isinstance(descriptors, Descriptors)
            else descriptors
        )
        u = self._normalise(vec)
        if self.binning == "cvt":
            assert self._centroids is not None  # narrowed by __post_init__
            return int(_nearest_centroid(u[None, :], self._centroids)[0])
        idx = np.minimum((u * self._bins).astype(np.int64), self._bins - 1)
        return int(np.ravel_multi_index(tuple(idx), tuple(self._bins)))

    # -- the MAP-Elites operator ------------------------------------------
    def add(
        self,
        genome: Genome | None,
        fitness: float,
        descriptors: Descriptors | Sequence[float],
    ) -> bool:
        """Offer a solution to the archive.

        Accepted iff ``fitness > t_e`` for its cell, where ``t_e`` starts at
        ``min_f`` and anneals upward on every acceptance. The stored elite is
        replaced only when the candidate also beats the incumbent's fitness, so
        the archive is monotone in quality while the acceptance bar stays
        deliberately loose.

        Parameters
        ----------
        genome : Genome or None
            ``None`` is permitted so the archive can be exercised without the
            genome machinery (tests, ablations).
        fitness : float
            Higher is better. Non-finite fitness is always rejected.
        descriptors : Descriptors or sequence of float

        Returns
        -------
        bool
            True if the candidate passed its cell's annealed threshold.
        """
        f = float(fitness)
        self._n_added += 1
        cell_id = self.cell_of(descriptors)
        cell = self._cells.get(cell_id)
        if cell is None:
            cell = _Cell(threshold=float(self.min_f))
            self._cells[cell_id] = cell
        cell.n_attempts += 1
        if not np.isfinite(f) or f <= cell.threshold:
            return False

        cell.threshold = (1.0 - self.alpha) * cell.threshold + self.alpha * f
        cell.n_accepted += 1
        self._n_accepted += 1
        if f > cell.fitness:
            cell.fitness = f
            cell.genome = genome
            if isinstance(descriptors, Descriptors):
                cell.descriptors = descriptors
            else:
                vals = np.asarray(descriptors, dtype=np.float64).ravel().tolist()
                cell.descriptors = Descriptors(
                    turnover=float(vals[0]),
                    horizon=float(vals[1]),
                    max_corr=float(vals[2]),
                    complexity=int(vals[3]),
                    coverage=float(vals[4]),
                )
        return True

    # -- readout -----------------------------------------------------------
    def elites(self) -> list[tuple[Genome, float, Descriptors]]:
        """Occupied cells as ``(genome, fitness, descriptors)``, best first."""
        out = [
            (c.genome, c.fitness, c.descriptors)
            for c in self._cells.values()
            if c.descriptors is not None
        ]
        out.sort(key=lambda t: -t[1])
        return out  # type: ignore[return-value]

    def coverage(self) -> float:
        """Fraction of cells that hold an elite, in ``[0, 1]``.

        Monotone non-decreasing over a run: cells are never emptied.
        """
        occupied = sum(1 for c in self._cells.values() if c.descriptors is not None)
        return occupied / float(self.n_cells)

    def qd_score(self, offset: float | None = None) -> float:
        """Quality-diversity score: ``sum(elite_fitness - offset)`` over cells.

        ``offset`` defaults to ``min_f``, which makes every term non-negative
        and the score monotone non-decreasing over a run -- the property that
        makes QD-score comparable between configurations. A run that improves
        one elite and a run that fills a new cell both move it, which is the
        whole point of the measure.
        """
        base = float(self.min_f) if offset is None else float(offset)
        return float(
            sum(
                max(c.fitness - base, 0.0)
                for c in self._cells.values()
                if c.descriptors is not None
            )
        )

    def best(self) -> tuple[Genome, float, Descriptors] | None:
        """Highest-fitness elite in the archive, or ``None`` if empty.

        Reported alongside :meth:`coverage` and :meth:`qd_score`, never on its
        own: a single best score from a large search is exactly the number that
        ``_honest.py`` exists to deflate.
        """
        best: tuple[Genome, float, Descriptors] | None = None
        for c in self._cells.values():
            if c.descriptors is None:
                continue
            if best is None or c.fitness > best[1]:
                best = (c.genome, c.fitness, c.descriptors)  # type: ignore[assignment]
        return best

    def to_records(self) -> list[dict[str, Any]]:
        """Occupied cells as plain dicts, ready for ``polars.DataFrame``.

        Keys: ``cell``, ``fitness``, ``threshold``, ``n_accepted``,
        ``n_attempts``, ``turnover``, ``horizon``, ``max_corr``,
        ``complexity``, ``coverage``, ``genome``.
        """
        records: list[dict[str, Any]] = []
        for cell_id, c in sorted(self._cells.items()):
            if c.descriptors is None:
                continue
            d = c.descriptors
            records.append(
                {
                    "cell": int(cell_id),
                    "fitness": float(c.fitness),
                    "threshold": float(c.threshold),
                    "n_accepted": int(c.n_accepted),
                    "n_attempts": int(c.n_attempts),
                    "turnover": float(d.turnover),
                    "horizon": float(d.horizon),
                    "max_corr": float(d.max_corr),
                    "complexity": int(d.complexity),
                    "coverage": float(d.coverage),
                    "genome": c.genome,
                }
            )
        records.sort(key=lambda r: -float(r["fitness"]))
        return records

    def thresholds(self) -> dict[int, float]:
        """Current annealed threshold per *touched* cell, for monitoring."""
        return {k: float(c.threshold) for k, c in self._cells.items()}

    @property
    def n_added(self) -> int:
        """Total candidates offered."""
        return self._n_added

    @property
    def n_accepted(self) -> int:
        """Total candidates that passed their cell threshold."""
        return self._n_accepted

    def __len__(self) -> int:
        return sum(1 for c in self._cells.values() if c.descriptors is not None)


# --------------------------------------------------------------------------- #
# 4. Age-fitness Pareto survival (ALPS-style) -- off by default
# --------------------------------------------------------------------------- #
#: AFPO is **not** enabled in the default search loop. Schmidt & Lipson
#: motivated it as a diversity mechanism, but La Cava, Helmuth, Spector & Moore
#: (arXiv:1709.05394) measured age-fitness Pareto optimisation producing *low*
#: semantic diversity -- the opposite of the original claim, and the opposite
#: of what lexicase gives us. It is shipped for ablation; ``_search.py`` should
#: consult this constant rather than turning it on silently.
AFPO_ENABLED_BY_DEFAULT: bool = False


def age_fitness_pareto_survival(
    fitness: np.ndarray | Sequence[float],
    ages: np.ndarray | Sequence[int],
    n_survivors: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Age-fitness Pareto survival (Schmidt & Lipson 2011, ALPS-style).

    Survival is Pareto-optimality over ``(fitness maximised, age minimised)``:
    an individual is evicted only if some other individual is both younger (or
    the same age) and fitter (or equally fit), with at least one strict. Young
    individuals are therefore protected from immediate competition with mature
    lineages, which is the mechanism that keeps a freshly injected random
    individual alive long enough to be worth injecting.

    The caller is responsible for the other half of the protocol, because
    genome construction lives in ``_genome.py``: **inject exactly one age-0
    random individual per generation**, and increment every survivor's age by
    one per generation. This function only decides who survives.

    Off by default -- see :data:`AFPO_ENABLED_BY_DEFAULT` for why.

    Parameters
    ----------
    fitness : array-like of float, shape (n,)
        Higher is better. Non-finite values are treated as ``-inf``.
    ages : array-like of int, shape (n,)
        Generations survived. Lower is better.
    n_survivors : int
        Population size to truncate to. When the Pareto layers do not divide
        evenly, the overflowing layer is truncated by crowding distance, so
        the extremes of the age-fitness trade-off are kept.
    seed : int, default 0
        Breaks crowding-distance ties reproducibly.

    Returns
    -------
    ndarray of int64, length ``min(n_survivors, n)``.
    """
    f = np.asarray(fitness, dtype=np.float64).ravel()
    a = np.asarray(ages, dtype=np.float64).ravel()
    if f.shape != a.shape:
        raise ValueError(f"fitness {f.shape} and ages {a.shape} must have equal length")
    # Both columns maximised: fitness as-is, age negated.
    objectives = np.column_stack([f, -a])
    return nsga2_survival(objectives, n_survivors, seed=seed)
