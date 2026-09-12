"""Borrowed accuracy: the leakage gap, decomposed exactly.

Every pipeline can be run twice: **permissively**, with each component fit once
on everything, and **point-in-time**, with each component refit per fold on the
training rows only. The gap between the two scores is the accuracy *borrowed*
from data the method will not have at prediction time::

    B = v(all permissive) - v(all point-in-time)

``B`` on its own is a single number and says nothing about *which* component
borrowed it. Ablating one component at a time does not answer that either: two
components can be individually harmless and leak badly together, and
one-at-a-time ablations attribute zero to both (see the interaction test in
``tests/test_leakage_borrowed.py``). The honest attribution is the **exact
Shapley value** of the cooperative game whose characteristic function is
``v(S)`` -- the score when exactly the components in ``S`` run permissively::

    phi_i = sum_{S subset of N\\{i}} |S|! (k-|S|-1)! / k! * ( v(S + i) - v(S) )

Shapley's efficiency axiom gives ``sum_i phi_i == B`` by construction, which is
what makes this a *decomposition* rather than a pile of ablations.

This module is deliberately numeric and side-effect free: it never fits a
model, never touches a frame, and never decides what "permissive" means. The
caller supplies ``evaluate(selection) -> float``; everything here is the
combinatorics. That is what makes it testable against the Shapley axioms, and
reusable for any notion of component, mode and score.

Notes
-----
Cost is ``2 ** k`` evaluations of ``evaluate``, memoised so each subset is
scored exactly once. This is exact -- there is no sampling -- and it is why
:func:`borrowed_accuracy` refuses more than ``max_components`` components
rather than quietly taking exponential time.

See Also
--------
panelary.testing.assert_no_lookahead : the yes/no form of the same experiment.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_MAX_COMPONENTS",
    "BorrowedAccuracyReport",
    "Component",
    "borrowed_accuracy",
    "resolve_modes",
]

#: Refusal threshold for :func:`borrowed_accuracy`. ``2 ** 12 == 4096``
#: evaluations is already a long afternoon of backtests; beyond it the caller
#: should group components, not wait.
DEFAULT_MAX_COMPONENTS: int = 12


@dataclass(frozen=True)
class Component:
    """One pipeline stage, runnable in two modes.

    The two callables are opaque to this module -- it never calls them. They
    exist so that a caller can keep the pair together and let
    :func:`resolve_modes` pick one per subset while building ``evaluate``.

    Parameters
    ----------
    name : str
        Identifier, unique within a run. It is the key in
        :attr:`BorrowedAccuracyReport.attribution`.
    permissive : callable
        The leaky mode: fit once, on everything.
    point_in_time : callable
        The honest mode: refit per fold, on training rows only.
    description : str, optional
        Free text for the report.

    Examples
    --------
    >>> scaler = Component("scaler", permissive=lambda: "all", point_in_time=lambda: "train")
    >>> scaler.mode(permissive=True)()
    'all'
    """

    name: str
    permissive: Callable[..., Any]
    point_in_time: Callable[..., Any]
    description: str = ""

    def mode(self, permissive: bool) -> Callable[..., Any]:
        """Return :attr:`permissive` if ``permissive`` else :attr:`point_in_time`.

        Parameters
        ----------
        permissive : bool
            Which mode to select.

        Returns
        -------
        callable
        """
        return self.permissive if permissive else self.point_in_time


def resolve_modes(
    components: Iterable[Component], selection: Iterable[str]
) -> dict[str, Callable[..., Any]]:
    """Map each component name to the callable its mode selects.

    Parameters
    ----------
    components : iterable of Component
        The pipeline stages.
    selection : iterable of str
        Names to run permissively; every other component runs point-in-time.

    Returns
    -------
    dict of str to callable
        One entry per component, in the order given.

    Raises
    ------
    ValueError
        If ``selection`` names something that is not a component.

    Examples
    --------
    >>> c = Component("a", permissive=lambda: 1, point_in_time=lambda: 0)
    >>> resolve_modes([c], {"a"})["a"]()
    1
    >>> resolve_modes([c], set())["a"]()
    0
    """
    chosen = frozenset(selection)
    comps = list(components)
    known = {c.name for c in comps}
    unknown = chosen - known
    if unknown:
        raise ValueError(
            f"selection names unknown component(s) {sorted(unknown)}; "
            f"known components are {sorted(known)}."
        )
    return {c.name: c.mode(c.name in chosen) for c in comps}


@dataclass(frozen=True)
class BorrowedAccuracyReport:
    """Borrowed accuracy and its exact Shapley decomposition.

    Attributes
    ----------
    total : float
        ``v(all permissive) - v(all point-in-time)``, sign-oriented so that a
        **positive** value always means accuracy was borrowed, whichever way
        the score points.
    attribution : dict of str to float
        Exact Shapley value per component. Sums to :attr:`total` (efficiency).
    n_evaluations : int
        Calls made to ``evaluate``: ``2 ** k``, each subset scored once.
    components : tuple of str
        Component names, in the order supplied.
    permissive_score, point_in_time_score : float
        The raw ``v(N)`` and ``v({})``, in the caller's own sign convention.
    higher_is_better : bool
        The convention the score was reported in.
    coalition_values : dict of frozenset to float
        The whole characteristic function ``v``, raw, for inspection. Keyed by
        the set of permissively-run component names.
    """

    total: float
    attribution: dict[str, float]
    n_evaluations: int
    components: tuple[str, ...]
    permissive_score: float
    point_in_time_score: float
    higher_is_better: bool = True
    coalition_values: dict[frozenset[str], float] = field(
        default_factory=dict, repr=False
    )

    @property
    def n_components(self) -> int:
        """Number of components, ``k``."""
        return len(self.components)

    @property
    def ranked(self) -> list[tuple[str, float]]:
        """``(name, phi)`` pairs, largest borrower first."""
        return sorted(self.attribution.items(), key=lambda kv: -kv[1])

    def share(self, name: str) -> float:
        """Fraction of :attr:`total` attributed to ``name``.

        Parameters
        ----------
        name : str
            A component name.

        Returns
        -------
        float
            ``phi_name / total``, or ``nan`` when ``total`` is zero.

        Raises
        ------
        KeyError
            If ``name`` is not a component.
        """
        phi = self.attribution[name]
        return phi / self.total if self.total != 0.0 else math.nan

    def __str__(self) -> str:
        direction = "higher is better" if self.higher_is_better else "lower is better"
        width = max((len(n) for n in self.components), default=1)
        width = max(width, len("point-in-time"))
        lines = [
            f"Borrowed accuracy: {self.total:+.6g}"
            f"  ({self.n_components} components, "
            f"{self.n_evaluations} evaluations, {direction})",
            f"  {'permissive':<{width}}  {self.permissive_score:+12.6g}",
            f"  {'point-in-time':<{width}}  {self.point_in_time_score:+12.6g}",
        ]
        if self.components:
            lines.append("  attribution (exact Shapley):")
            for name, phi in self.ranked:
                share = self.share(name)
                pct = "     -" if math.isnan(share) else f"{share:6.1%}"
                lines.append(f"    {name:<{width}}  {phi:+12.6g}  {pct}")
        return "\n".join(lines)


def _validated_names(components: Iterable[Component | str]) -> tuple[str, ...]:
    """Extract unique component names, preserving order."""
    names: list[str] = []
    for item in components:
        name = item.name if isinstance(item, Component) else item
        if not isinstance(name, str):
            raise TypeError(
                "each component must be a Component or a str name, got "
                f"{type(item).__name__!r}."
            )
        names.append(name)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"component names must be unique; repeated: {duplicates}. "
            "Shapley attribution is keyed by name, so a duplicate would be "
            "two players sharing one slot."
        )
    return tuple(names)


def _as_score(value: Any, selection: frozenset[str]) -> float:
    """Coerce one ``evaluate`` return to a finite float64."""
    if isinstance(value, (str, bytes)):
        # float("0.5") would succeed, and quietly turn a bug into a score.
        raise TypeError(
            f"evaluate({set(selection) or '{}'}) returned {value!r}, "
            "which is not a real number."
        )
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"evaluate({set(selection) or '{}'}) returned {value!r}, "
            "which is not a real number."
        ) from exc
    if not math.isfinite(score):
        raise ValueError(
            f"evaluate({set(selection) or '{}'}) returned {score!r}; "
            "borrowed accuracy needs a finite score for every subset."
        )
    return score


def borrowed_accuracy(
    components: Iterable[Component | str],
    evaluate: Callable[[frozenset[str]], float],
    *,
    higher_is_better: bool = True,
    max_components: int = DEFAULT_MAX_COMPONENTS,
) -> BorrowedAccuracyReport:
    r"""Measure borrowed accuracy and attribute it by exact Shapley value.

    ``evaluate(S)`` must return the score obtained when exactly the components
    named in ``S`` run **permissively** (fit once, on everything) and every
    other component runs **point-in-time** (refit per fold, on training rows
    only). Borrowed accuracy is then ``v(N) - v({})``, oriented by
    ``higher_is_better`` so a positive total always reads "borrowed".

    Attribution is the exact Shapley value

    .. math::

        \phi_i = \sum_{S \subseteq N \setminus \{i\}}
                 \frac{|S|!\,(k-|S|-1)!}{k!} \bigl( v(S \cup \{i\}) - v(S) \bigr)

    computed from every one of the ``2 ** k`` subsets. Nothing is sampled, so
    ``sum(attribution.values()) == total`` holds to floating-point tolerance --
    the efficiency axiom, and the property that makes this a decomposition
    rather than a set of ablations.

    Parameters
    ----------
    components : iterable of Component or str
        The pipeline stages, ``k`` of them. Only the names are used here; pass
        :class:`Component` objects when you want :func:`resolve_modes` to pick
        the modes for you while building ``evaluate``.
    evaluate : callable
        ``evaluate(selection: frozenset[str]) -> float``. Called exactly
        ``2 ** k`` times, once per subset, and memoised. It must be
        deterministic: the same subset must score the same, or the
        decomposition is meaningless.
    higher_is_better : bool, default=True
        Whether a larger score is a better one. When ``False`` (an error
        metric, say) every difference is negated, so ``total > 0`` still means
        the permissive run was flattered by leakage.
    max_components : int, default=12
        Refuse more components than this rather than spend ``2 ** k``
        evaluations. ``k = 12`` is 4096 backtests.

    Returns
    -------
    BorrowedAccuracyReport
        ``total``, the per-component ``attribution``, ``n_evaluations`` and the
        full characteristic function.

    Raises
    ------
    ValueError
        If ``components`` is empty, contains duplicate names, is longer than
        ``max_components``, or if ``evaluate`` returns a non-finite score.
    TypeError
        If a component is neither a :class:`Component` nor a ``str``, or if
        ``evaluate`` returns something that is not a real number.

    Notes
    -----
    **Cost:** ``2 ** k`` calls to ``evaluate`` (``k = len(components)``): 8 for
    3 components, 64 for 6, 1024 for 10. Each call is typically a full
    walk-forward backtest, so the evaluation dominates; the combinatorics here
    are free.

    Examples
    --------
    A game where one component leaks 0.1 and the other leaks nothing:

    >>> def evaluate(selection):
    ...     return 0.1 if "scaler" in selection else 0.0
    >>> report = borrowed_accuracy(["scaler", "imputer"], evaluate)
    >>> round(report.total, 12)
    0.1
    >>> {k: round(v, 12) for k, v in report.attribution.items()}
    {'scaler': 0.1, 'imputer': 0.0}
    >>> report.n_evaluations
    4

    A game where neither leaks alone but both leak together -- Shapley splits
    it evenly, where one-at-a-time ablation would attribute zero to each:

    >>> def joint(selection):
    ...     return 0.1 if {"a", "b"} <= selection else 0.0
    >>> {k: round(v, 12) for k, v in borrowed_accuracy(["a", "b"], joint).attribution.items()}
    {'a': 0.05, 'b': 0.05}
    """
    names = _validated_names(components)
    k = len(names)
    if k == 0:
        raise ValueError(
            "borrowed_accuracy needs at least one component; with none there "
            "is no pipeline to run two ways."
        )
    if k > max_components:
        raise ValueError(
            f"{k} components would need 2**{k} = {2**k} evaluations, above "
            f"max_components={max_components} (2**{max_components} = "
            f"{2**max_components}). Group components into coarser stages, or "
            "raise max_components deliberately and budget the time."
        )

    orient = 1.0 if higher_is_better else -1.0

    # --- evaluate every subset exactly once ------------------------------- #
    # Bitmask i of `names` <-> subset. Memoised on the frozenset so that a
    # caller passing the same selection twice cannot cost two backtests, and
    # so that `n_evaluations` counts real work rather than loop iterations.
    memo: dict[frozenset[str], float] = {}
    n_evaluations = 0

    def value(mask: int) -> float:
        nonlocal n_evaluations
        selection = frozenset(names[i] for i in range(k) if mask >> i & 1)
        if selection not in memo:
            memo[selection] = _as_score(evaluate(selection), selection)
            n_evaluations += 1
        return memo[selection]

    full = (1 << k) - 1
    values = [value(mask) for mask in range(1 << k)]

    # --- exact Shapley ---------------------------------------------------- #
    # weight[s] = s! (k - s - 1)! / k!, the probability that a uniformly random
    # permutation puts exactly the s members of S before player i.
    k_factorial = float(math.factorial(k))
    weight = [
        math.factorial(s) * math.factorial(k - s - 1) / k_factorial for s in range(k)
    ]

    attribution: dict[str, float] = {}
    for i, name in enumerate(names):
        bit = 1 << i
        phi = 0.0
        for mask in range(1 << k):
            if mask & bit:
                continue
            gain = orient * (values[mask | bit] - values[mask])
            phi += weight[bin(mask).count("1")] * gain
        attribution[name] = phi

    permissive_score = values[full]
    point_in_time_score = values[0]
    total = orient * (permissive_score - point_in_time_score)

    return BorrowedAccuracyReport(
        total=total,
        attribution=attribution,
        n_evaluations=n_evaluations,
        components=names,
        permissive_score=permissive_score,
        point_in_time_score=point_in_time_score,
        higher_is_better=higher_is_better,
        coalition_values=dict(memo),
    )
