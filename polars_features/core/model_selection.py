"""Leak-safe, finance-grade cross-validation for panels.

This module implements the cross-validation machinery from Marcos Lopez de
Prado, *Advances in Financial Machine Learning* (Wiley, 2018), adapted to
long-format Polars panels:

* :class:`PurgedKFold` — k-fold CV that **purges** training observations whose
  labels overlap the test window and applies an **embargo** after each test
  fold (de Prado, Ch. 7).
* :class:`CombinatorialPurgedCV` — the Combinatorial Purged Cross-Validation
  (CPCV) scheme that tests every combination of ``k`` of ``N`` groups, producing
  many backtest paths (de Prado, Ch. 12).
* :func:`expanding_window_split` / :func:`sliding_window_split` — walk-forward
  splitters, panel-aware re-exposures of
  :mod:`polars_features.cross_validation`.
* :func:`deflated_sharpe_ratio` — the Deflated Sharpe Ratio (Bailey & de Prado,
  2014) correcting for multiple testing, non-normality and sample length.
* :func:`probability_of_backtest_overfitting` — the PBO via Combinatorially
  Symmetric Cross-Validation (Bailey, Borwein, de Prado, Zhu, 2017).

Leakage model
-------------
Time alignment is along the **time axis**, shared across entities (a panel
backtest rebalances on common dates). Splitters therefore operate on the sorted
unique time index; purging and embargo are applied in time units, and the
resulting train/test time sets are mapped back to *all* entities, so per-entity
grouping is respected automatically (an entity contributes its rows for the
selected times only).

Each splitter is *event-based*: a "label" for an observation at time ``t`` is
assumed to span ``[t, t + horizon]`` (the prediction horizon). An observation is
**purged** from the training set if its label window overlaps any test label
window; an additional **embargo** of ``embargo`` time-steps after each test
block is removed from training to handle serial correlation.

All splitters yield ``(train, test)`` as :class:`PanelFrame` pairs by default,
or as integer time-index positions when ``return_indices=True``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from polars_features.core.panel_frame import PanelFrame, as_panel

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "PurgedKFold",
    "CombinatorialPurgedCV",
    "expanding_window_split",
    "sliding_window_split",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
]

# A fold is (train_panel, test_panel) or (train_time_idx, test_time_idx).
PanelFold = tuple[PanelFrame, PanelFrame]
IndexFold = tuple["NDArray[np.int64]", "NDArray[np.int64]"]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _unique_times(panel: PanelFrame) -> np.ndarray:
    """Return the sorted unique time values of ``panel`` as a numpy array."""
    return panel.time_index().to_numpy()


def _subset_by_times(panel: PanelFrame, times: Sequence) -> PanelFrame:
    """Return a PanelFrame containing only rows whose time is in ``times``.

    Stays lazy; uses ``is_in`` against the selected time values, so every entity
    contributes its rows for those times (panel grouping respected).
    """
    # `.implode()` gives the unambiguous set-membership form of `is_in`
    # (a plain same-dtype collection is deprecated in recent Polars).
    time_vals = pl.Series(values=list(times)).implode()
    lf = panel.lazy().filter(pl.col(panel.time_col).is_in(time_vals))
    return PanelFrame(lf, entity=panel.entity_col, time=panel.time_col, validate=False)


def _contiguous_blocks(positions: np.ndarray) -> list[tuple[int, int]]:
    """Split sorted positions into contiguous ``[start, end]`` index blocks."""
    if positions.size == 0:
        return []
    blocks: list[tuple[int, int]] = []
    start = prev = int(positions[0])
    for p in positions[1:]:
        p = int(p)
        if p == prev + 1:
            prev = p
        else:
            blocks.append((start, prev))
            start = prev = p
    blocks.append((start, prev))
    return blocks


def _purge_embargo_positions(
    n_times: int,
    test_positions: np.ndarray,
    horizon: int,
    embargo: int,
) -> np.ndarray:
    """Return the *train* time-index positions after purge + embargo.

    Parameters
    ----------
    n_times : int
        Number of unique time steps.
    test_positions : ndarray of int
        Positions (into the sorted unique-time index) belonging to the test set.
    horizon : int
        Label horizon in time-steps. An observation at position ``i`` has a label
        spanning positions ``[i, i + horizon]``. Training observations whose
        label window overlaps any test position are purged.
    embargo : int
        Number of time-steps after each contiguous test block to additionally
        remove from training.

    Returns
    -------
    ndarray of int
        Sorted training positions.

    Notes
    -----
    Purge logic (de Prado, 7.4.1): a train observation at position ``j`` overlaps
    the test set if ``[j, j + horizon]`` intersects ``[i, i + horizon]`` for some
    test position ``i``. Equivalently ``j`` is purged when it lies within
    ``horizon`` of any test position on either side. The embargo (7.4.3) removes
    a further ``embargo`` positions immediately *after* each test block.
    """
    test_set = {int(p) for p in test_positions}
    blocked = set(test_set)

    # Purge: any train position whose label window [j, j+horizon] overlaps a test
    # label window [i, i+horizon]. Overlap <=> |j - i| <= horizon for the closed
    # windows of equal length. We expand each test position by `horizon` on both
    # sides.
    for i in test_set:
        lo = max(0, i - horizon)
        hi = min(n_times - 1, i + horizon)
        for j in range(lo, hi + 1):
            blocked.add(j)

    # Embargo: remove `embargo` positions after each contiguous test block.
    if embargo > 0:
        for _start, end in _contiguous_blocks(np.array(sorted(test_set))):
            lo = end + 1
            hi = min(n_times - 1, end + embargo)
            for j in range(lo, hi + 1):
                blocked.add(j)

    train = np.array([p for p in range(n_times) if p not in blocked], dtype=np.int64)
    return train


# --------------------------------------------------------------------------- #
# PurgedKFold
# --------------------------------------------------------------------------- #
class PurgedKFold:
    """K-fold cross-validation with purging and embargo for panels.

    Splits the **sorted unique time index** into ``n_splits`` contiguous folds.
    Each fold is used once as the test set; the corresponding training set is the
    remaining times with (a) any observation whose label window overlaps the test
    window **purged**, and (b) an **embargo** of ``embargo`` time-steps after the
    test block removed.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be >= 2.
    horizon : int, default=0
        Label horizon in time-steps used for purging. ``0`` means each label is
        point-in-time (only the exact test times are purged from train).
    embargo : int, default=0
        Number of time-steps after each test block to embargo from training.
    return_indices : bool, default=False
        If True, :meth:`split` yields ``(train_positions, test_positions)`` as
        integer numpy arrays into the sorted unique time index. If False
        (default), it yields ``(train_panel, test_panel)`` as PanelFrames.

    Raises
    ------
    ValueError
        If ``n_splits < 2``, ``horizon < 0``, or ``embargo < 0``.

    References
    ----------
    Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*,
    Chapter 7 ("Cross-Validation in Finance").

    Examples
    --------
    >>> cv = PurgedKFold(n_splits=4, horizon=1, embargo=1)
    >>> for train, test in cv.split(panel):  # doctest: +SKIP
    ...     model.fit(train)
    ...     preds = model.predict(test)
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        horizon: int = 0,
        embargo: int = 0,
        return_indices: bool = False,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"`n_splits` must be >= 2, got {n_splits}.")
        if horizon < 0:
            raise ValueError(f"`horizon` must be >= 0, got {horizon}.")
        if embargo < 0:
            raise ValueError(f"`embargo` must be >= 0, got {embargo}.")
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo = embargo
        self.return_indices = return_indices

    def get_n_splits(self) -> int:
        """Return the number of folds (sklearn-compatible)."""
        return self.n_splits

    def _test_position_folds(self, n_times: int) -> list[np.ndarray]:
        """Partition ``range(n_times)`` into ``n_splits`` contiguous test folds."""
        if n_times < self.n_splits:
            raise ValueError(
                f"cannot make {self.n_splits} folds from only {n_times} unique "
                "time steps; reduce `n_splits` or supply more history."
            )
        return [
            np.array(part, dtype=np.int64)
            for part in np.array_split(
                np.arange(n_times, dtype=np.int64), self.n_splits
            )
        ]

    def split(
        self, panel: PanelFrame | pl.DataFrame | pl.LazyFrame
    ) -> Iterator[PanelFold | IndexFold]:
        """Generate purged, embargoed train/test folds.

        Parameters
        ----------
        panel : PanelFrame | polars.DataFrame | polars.LazyFrame
            Long-format panel. Frames are coerced via
            :func:`~polars_features.core.panel_frame.as_panel` (entity = col 0,
            time = col 1 by convention).

        Yields
        ------
        (train, test) : tuple
            PanelFrames, or integer position arrays if ``return_indices=True``.
        """
        pf = as_panel(panel)
        times = _unique_times(pf)
        n_times = times.shape[0]
        for test_pos in self._test_position_folds(n_times):
            train_pos = _purge_embargo_positions(
                n_times, test_pos, self.horizon, self.embargo
            )
            if self.return_indices:
                yield train_pos, test_pos
            else:
                yield (
                    _subset_by_times(pf, times[train_pos]),
                    _subset_by_times(pf, times[test_pos]),
                )


# --------------------------------------------------------------------------- #
# Combinatorial Purged Cross-Validation
# --------------------------------------------------------------------------- #
@dataclass
class _CPCVFold:
    """A single CPCV split: which groups are test, plus position arrays."""

    test_groups: tuple[int, ...]
    train_positions: NDArray[np.int64]
    test_positions: NDArray[np.int64]


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (CPCV) for panels.

    The sorted unique time index is partitioned into ``n_groups`` contiguous
    groups. Every combination of ``n_test_groups`` groups is used as the test
    set (so there are ``C(n_groups, n_test_groups)`` splits), with the remaining
    groups as training — purged and embargoed as in :class:`PurgedKFold`.

    Because each group appears in many test combinations, CPCV yields a number of
    distinct **backtest paths** rather than a single one, giving a distribution
    of out-of-sample performance instead of a point estimate.

    Parameters
    ----------
    n_groups : int, default=6
        Number of contiguous time groups ``N``. Must be >= 2.
    n_test_groups : int, default=2
        Number of groups per test set ``k`` (``1 <= k < N``).
    horizon : int, default=0
        Label horizon in time-steps used for purging.
    embargo : int, default=0
        Embargo in time-steps applied after each contiguous test block.
    return_indices : bool, default=False
        If True, yield integer position arrays; else PanelFrames.

    Attributes
    ----------
    n_splits : int
        ``C(n_groups, n_test_groups)``.
    n_paths : int
        Number of backtest paths, ``C(n_groups, n_test_groups) * n_test_groups
        / n_groups`` (de Prado, 12.4).

    Raises
    ------
    ValueError
        If parameters are out of range.

    References
    ----------
    Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*,
    Chapter 12 ("Backtesting through Cross-Validation").
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        *,
        horizon: int = 0,
        embargo: int = 0,
        return_indices: bool = False,
    ) -> None:
        if n_groups < 2:
            raise ValueError(f"`n_groups` must be >= 2, got {n_groups}.")
        if not (1 <= n_test_groups < n_groups):
            raise ValueError(
                f"`n_test_groups` must satisfy 1 <= k < n_groups "
                f"({n_groups}), got {n_test_groups}."
            )
        if horizon < 0:
            raise ValueError(f"`horizon` must be >= 0, got {horizon}.")
        if embargo < 0:
            raise ValueError(f"`embargo` must be >= 0, got {embargo}.")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.horizon = horizon
        self.embargo = embargo
        self.return_indices = return_indices

    @property
    def n_splits(self) -> int:
        """Number of train/test combinations, ``C(n_groups, n_test_groups)``."""
        return math.comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """Number of distinct backtest paths reconstructable from the splits.

        Equals ``C(N, k) * k / N`` (each of the ``N`` groups is tested in
        ``C(N-1, k-1)`` splits, and the paths recombine non-overlapping test
        groups).
        """
        return self.n_splits * self.n_test_groups // self.n_groups

    def get_n_splits(self) -> int:
        """Return :attr:`n_splits` (sklearn-compatible)."""
        return self.n_splits

    def _group_positions(self, n_times: int) -> list[np.ndarray]:
        """Partition ``range(n_times)`` into ``n_groups`` contiguous groups."""
        if n_times < self.n_groups:
            raise ValueError(
                f"cannot make {self.n_groups} groups from only {n_times} unique "
                "time steps; reduce `n_groups` or supply more history."
            )
        return [
            np.array(part, dtype=np.int64)
            for part in np.array_split(
                np.arange(n_times, dtype=np.int64), self.n_groups
            )
        ]

    def _iter_folds(self, n_times: int) -> Iterator[_CPCVFold]:
        groups = self._group_positions(n_times)
        for test_combo in itertools.combinations(
            range(self.n_groups), self.n_test_groups
        ):
            test_pos = np.sort(np.concatenate([groups[g] for g in test_combo])).astype(
                np.int64
            )
            train_pos = _purge_embargo_positions(
                n_times, test_pos, self.horizon, self.embargo
            )
            yield _CPCVFold(
                test_groups=test_combo,
                train_positions=train_pos,
                test_positions=test_pos,
            )

    def split(
        self, panel: PanelFrame | pl.DataFrame | pl.LazyFrame
    ) -> Iterator[PanelFold | IndexFold]:
        """Generate all CPCV train/test folds.

        Parameters
        ----------
        panel : PanelFrame | polars.DataFrame | polars.LazyFrame
            Long-format panel.

        Yields
        ------
        (train, test) : tuple
            PanelFrames, or integer position arrays if ``return_indices=True``.
            Folds are emitted in lexicographic order of the test-group
            combination; use :meth:`split_with_groups` to also obtain the test
            group ids for path reconstruction.
        """
        for train, test, _ in self.split_with_groups(panel):
            yield train, test

    def split_with_groups(
        self, panel: PanelFrame | pl.DataFrame | pl.LazyFrame
    ) -> Iterator[
        tuple[
            PanelFrame | NDArray[np.int64],
            PanelFrame | NDArray[np.int64],
            tuple[int, ...],
        ]
    ]:
        """Like :meth:`split` but also yields the test-group tuple per fold.

        Yields
        ------
        (train, test, test_groups) : tuple
            ``test_groups`` is the tuple of group indices used as test, enabling
            reconstruction of the :attr:`n_paths` backtest paths.
        """
        pf = as_panel(panel)
        times = _unique_times(pf)
        n_times = times.shape[0]
        for fold in self._iter_folds(n_times):
            if self.return_indices:
                yield fold.train_positions, fold.test_positions, fold.test_groups
            else:
                yield (
                    _subset_by_times(pf, times[fold.train_positions]),
                    _subset_by_times(pf, times[fold.test_positions]),
                    fold.test_groups,
                )

    def backtest_paths(self) -> list[list[tuple[int, int]]]:
        """Return the assignment of (split, group) to each backtest path.

        Each path is a list of ``(split_index, group_index)`` pairs covering all
        ``n_groups`` groups exactly once, in time order. Combine the test
        predictions of those (split, group) cells to obtain one full backtest
        path. There are :attr:`n_paths` such paths (de Prado, 12.4).

        Returns
        -------
        list of list of (int, int)
            ``paths[p][g]`` gives the ``(split_index, group_index)`` providing
            the test prediction for group ``g`` on path ``p``.
        """
        combos = list(itertools.combinations(range(self.n_groups), self.n_test_groups))
        # For each group, list the (split_index, position-within-combo) where it
        # is tested. Each group is tested in exactly C(N-1, k-1) splits == n_paths.
        per_group: list[list[int]] = [[] for _ in range(self.n_groups)]
        for s_idx, combo in enumerate(combos):
            for g in combo:
                per_group[g].append(s_idx)
        n_paths = self.n_paths
        paths: list[list[tuple[int, int]]] = []
        for p in range(n_paths):
            path: list[tuple[int, int]] = []
            for g in range(self.n_groups):
                split_idx = per_group[g][p]
                path.append((split_idx, g))
            paths.append(path)
        return paths


# --------------------------------------------------------------------------- #
# Walk-forward splitters (panel-aware re-exposure)
# --------------------------------------------------------------------------- #
def _window_panel_split(
    pf: PanelFrame,
    test_size: int,
    n_splits: int,
    step_size: int,
    window_size: int | None,
) -> list[PanelFold]:
    """Shared walk-forward split logic on the unique-time axis.

    Mirrors the slicing approach in
    :mod:`polars_features.cross_validation._window_split`, but slices the **shared
    unique-time index** (so all entities are aligned on the same train/test
    times) and returns PanelFrames.
    """
    times = _unique_times(pf)
    n_times = times.shape[0]

    backward_steps = np.arange(1, n_splits) * step_size + test_size
    cutoffs = np.flip(np.concatenate([np.array([test_size]), backward_steps]))

    folds: list[PanelFold] = []
    for i in range(n_splits):
        cutoff = int(cutoffs[i])
        test_start = n_times - cutoff
        test_end = test_start + test_size
        if test_start < 0:
            raise ValueError(
                f"split {i}: not enough history ({n_times} time steps) for "
                f"test_size={test_size}, n_splits={n_splits}, step_size={step_size}"
                + (f", window_size={window_size}" if window_size else "")
                + ". Reduce these parameters or supply more data."
            )
        test_pos = np.arange(test_start, min(test_end, n_times), dtype=np.int64)

        if window_size is not None:
            train_start = max(0, test_start - window_size)
            train_pos = np.arange(train_start, test_start, dtype=np.int64)
        else:
            train_pos = np.arange(0, test_start, dtype=np.int64)

        folds.append(
            (
                _subset_by_times(pf, times[train_pos]),
                _subset_by_times(pf, times[test_pos]),
            )
        )
    return folds


def expanding_window_split(
    test_size: int,
    n_splits: int = 5,
    step_size: int = 1,
):
    """Return a panel-aware expanding-window walk-forward splitter.

    Each split grows the training window from the start of history up to the test
    block; the test block is a fixed-size window that slides forward by
    ``step_size`` across splits. Splitting happens on the **shared unique-time
    index**, so all entities are aligned on the same train/test dates.

    For ``test_size=3, n_splits=5, step_size=1`` the folds are::

        | o o o x x x - - - - |
        | o o o o x x x - - - |
        | o o o o o x x x - - |
        | o o o o o o x x x - |
        | o o o o o o o x x x |

    Parameters
    ----------
    test_size : int
        Number of *time steps* per test block.
    n_splits : int, default=5
        Number of splits.
    step_size : int, default=1
        Forward step (in time steps) between consecutive test blocks.

    Returns
    -------
    callable
        ``split(panel) -> list[(train_panel, test_panel)]`` where ``panel`` is a
        :class:`PanelFrame` or a coercible frame.

    See Also
    --------
    polars_features.cross_validation.expanding_window_split : the row-based
        functime original this thinly wraps.
    """

    def split(panel: PanelFrame | pl.DataFrame | pl.LazyFrame) -> list[PanelFold]:
        pf = as_panel(panel)
        return _window_panel_split(pf, test_size, n_splits, step_size, None)

    return split


def sliding_window_split(
    test_size: int,
    n_splits: int = 5,
    step_size: int = 1,
    window_size: int = 10,
):
    """Return a panel-aware sliding-window walk-forward splitter.

    Like :func:`expanding_window_split` but the training window is a fixed length
    of ``window_size`` time steps that slides forward with the test block.

    For ``test_size=3, n_splits=5, step_size=1, window_size=5`` the folds are::

        | o o o o o x x x - - - - |
        | - o o o o o x x x - - - |
        | - - o o o o o x x x - - |
        | - - - o o o o o x x x - |
        | - - - - o o o o o x x x |

    Parameters
    ----------
    test_size : int
        Number of time steps per test block.
    n_splits : int, default=5
        Number of splits.
    step_size : int, default=1
        Forward step (in time steps) between consecutive test blocks.
    window_size : int, default=10
        Number of time steps in each (fixed-length) training window.

    Returns
    -------
    callable
        ``split(panel) -> list[(train_panel, test_panel)]``.
    """

    def split(panel: PanelFrame | pl.DataFrame | pl.LazyFrame) -> list[PanelFold]:
        pf = as_panel(panel)
        return _window_panel_split(pf, test_size, n_splits, step_size, window_size)

    return split


# --------------------------------------------------------------------------- #
# de Prado performance / overfitting metrics
# --------------------------------------------------------------------------- #
def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function (no SciPy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Accurate to ~1e-9 over the open interval (0, 1).
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"`p` must be in the open interval (0, 1), got {p}.")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_observations: int,
    sharpe_std: float | None = None,
    sharpe_variance_across_trials: float | None = None,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sharpe: float | None = None,
) -> float:
    """Deflated Sharpe Ratio (DSR): probability the strategy is truly skilled.

    Implements Bailey & Lopez de Prado (2014). The DSR is the probability that
    the observed (non-annualised) Sharpe ratio exceeds an *expected maximum
    Sharpe* due to multiple testing, while accounting for the non-normality
    (skewness, kurtosis) and length of the return series.

    Parameters
    ----------
    observed_sharpe : float
        The Sharpe ratio actually observed for the selected strategy, expressed
        per observation (same frequency as ``n_observations``; do **not**
        annualise).
    n_trials : int
        Number ``N`` of independent strategy configurations tried (the multiple
        testing count).
    n_observations : int
        Number ``T`` of return observations the Sharpe was estimated from.
    sharpe_std : float, optional
        Standard deviation of the Sharpe estimator. If omitted it is computed
        from ``n_observations``, ``skewness`` and ``kurtosis`` via the standard
        formula (Bailey & de Prado eq. for ``sigma(SR_hat)``).
    sharpe_variance_across_trials : float, optional
        Variance of the Sharpe ratios *across the N trials*. Used to estimate the
        expected maximum Sharpe under the null. If omitted, defaults to
        ``sharpe_std ** 2`` (a conservative fallback assuming trials are as noisy
        as the estimator).
    skewness : float, default=0.0
        Skewness of the return series.
    kurtosis : float, default=3.0
        Kurtosis of the return series (3.0 == normal).
    benchmark_sharpe : float, optional
        The expected maximum Sharpe under the null. If omitted it is computed
        from ``n_trials`` and ``sharpe_variance_across_trials`` using the
        expected-maximum-of-Gaussians approximation (eq. for ``SR_0``).

    Returns
    -------
    float
        The DSR in ``[0, 1]``. Values near 1 indicate the strategy is unlikely to
        be a false positive of the selection process.

    References
    ----------
    Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
    *Journal of Portfolio Management*, 40(5).

    Notes
    -----
    The estimator standard deviation is

    .. math::

        \\hat\\sigma(\\widehat{SR}) = \\sqrt{\\frac{1 - \\gamma_3\\,SR
        + \\frac{\\gamma_4 - 1}{4} SR^2}{T - 1}}

    where :math:`\\gamma_3` is skewness and :math:`\\gamma_4` is kurtosis. The
    expected maximum Sharpe across :math:`N` trials uses

    .. math::

        SR_0 = \\sqrt{V}\\left[(1-\\gamma)\\,Z^{-1}\\!\\left(1-\\tfrac1N\\right)
        + \\gamma\\,Z^{-1}\\!\\left(1-\\tfrac1N e^{-1}\\right)\\right]

    with :math:`V` the variance of Sharpes across trials and
    :math:`\\gamma\\approx 0.5772` the Euler–Mascheroni constant.
    """
    if n_observations < 2:
        raise ValueError("`n_observations` must be >= 2.")
    if n_trials < 1:
        raise ValueError("`n_trials` must be >= 1.")

    sr = float(observed_sharpe)

    if sharpe_std is None:
        var = (1.0 - skewness * sr + (kurtosis - 1.0) / 4.0 * sr * sr) / (
            n_observations - 1
        )
        if var <= 0:
            raise ValueError(
                "computed Sharpe-estimator variance is non-positive; check "
                "`skewness`/`kurtosis`/`observed_sharpe` inputs."
            )
        sharpe_std = math.sqrt(var)

    if benchmark_sharpe is None:
        v = (
            sharpe_variance_across_trials
            if sharpe_variance_across_trials is not None
            else sharpe_std * sharpe_std
        )
        if v < 0:
            raise ValueError("`sharpe_variance_across_trials` must be >= 0.")
        gamma = 0.5772156649015329  # Euler-Mascheroni
        if n_trials == 1:
            benchmark_sharpe = 0.0
        else:
            z1 = _norm_ppf(1.0 - 1.0 / n_trials)
            z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
            benchmark_sharpe = math.sqrt(v) * ((1.0 - gamma) * z1 + gamma * z2)

    dsr = _norm_cdf((sr - benchmark_sharpe) / sharpe_std)
    return float(dsr)


def probability_of_backtest_overfitting(
    performance_matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    n_partitions: int = 16,
    higher_is_better: bool = True,
) -> float:
    """Probability of Backtest Overfitting (PBO) via CSCV.

    Implements the Combinatorially Symmetric Cross-Validation estimator of
    Bailey, Borwein, Lopez de Prado & Zhu (2017). Given a matrix ``M`` of
    out-of-sample-style performance for ``S`` strategies over ``T`` time slices,
    PBO estimates the probability that the strategy selected as best in-sample
    underperforms the median strategy out-of-sample.

    Parameters
    ----------
    performance_matrix : ndarray of shape (T, S) or nested sequence
        ``T`` rows of per-period performance (e.g. returns) for ``S`` strategy
        configurations. Each column is one strategy's track record.
    n_partitions : int, default=16
        Even number ``S_partitions`` of contiguous, equal-size sub-matrices the
        rows are split into; CSCV evaluates every way of choosing half as the
        in-sample set (``C(n_partitions, n_partitions/2)`` combinations). Must be
        even and >= 2. Reduce if ``T`` is small.
    higher_is_better : bool, default=True
        Whether larger performance values are better (e.g. Sharpe, returns). If
        False (e.g. loss), the sign of the selection is flipped.

    Returns
    -------
    float
        PBO in ``[0, 1]``. Values near 0 indicate the in-sample-best strategy
        tends to remain good out-of-sample (low overfitting); values near or
        above 0.5 indicate selection is no better than chance.

    Raises
    ------
    ValueError
        If the matrix is degenerate (fewer than 2 strategies), ``n_partitions``
        is not a valid even number, or ``T`` is too small to partition.

    References
    ----------
    Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2017). "The
    Probability of Backtest Overfitting." *Journal of Computational Finance*,
    20(4).

    Notes
    -----
    Algorithm (CSCV):

    1. Split the ``T`` rows into ``S`` (``=n_partitions``) disjoint, equal
       sub-matrices.
    2. For each combination ``c`` of ``S/2`` sub-matrices forming the in-sample
       (IS) set, the complement is out-of-sample (OS).
    3. Find ``n*``, the strategy best on IS. Compute its OS rank as a fraction
       ``omega = rank / (n_strategies + 1)`` and the logit
       ``lambda = ln(omega / (1 - omega))``.
    4. PBO is the fraction of combinations with ``lambda <= 0`` (i.e. the IS-best
       strategy lands in the bottom half OS).
    """
    M = np.asarray(performance_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError(
            f"`performance_matrix` must be 2-D (T x S), got shape {M.shape}."
        )
    T, S = M.shape
    if S < 2:
        raise ValueError(f"need at least 2 strategies (columns), got {S}.")
    if n_partitions < 2 or n_partitions % 2 != 0:
        raise ValueError(
            f"`n_partitions` must be an even integer >= 2, got {n_partitions}."
        )
    if n_partitions > T:
        raise ValueError(
            f"need at least n_partitions={n_partitions} rows, got T={T}. "
            "Reduce `n_partitions` or supply a longer track record."
        )
    if not higher_is_better:
        M = -M

    # Trim rows so they divide evenly into n_partitions contiguous blocks.
    block = T // n_partitions
    block * n_partitions
    blocks = [M[i * block : (i + 1) * block, :] for i in range(n_partitions)]

    half = n_partitions // 2
    logits: list[float] = []
    for is_combo in itertools.combinations(range(n_partitions), half):
        is_set = set(is_combo)
        is_rows = np.vstack([blocks[i] for i in range(n_partitions) if i in is_set])
        os_rows = np.vstack([blocks[i] for i in range(n_partitions) if i not in is_set])

        # Performance per strategy = mean over the slice (proxy for Sharpe-like
        # aggregate; matches the de Prado reference using a chosen statistic).
        is_perf = is_rows.mean(axis=0)
        os_perf = os_rows.mean(axis=0)

        n_star = int(np.argmax(is_perf))
        # Rank of the IS-best strategy OS (1 = worst .. S = best).
        order = np.argsort(np.argsort(os_perf))  # 0-based ranks
        rank = int(order[n_star]) + 1
        omega = rank / (S + 1)
        omega = min(max(omega, 1e-12), 1 - 1e-12)
        logits.append(math.log(omega / (1.0 - omega)))

    logits_arr = np.asarray(logits)
    pbo = float(np.mean(logits_arr <= 0.0))
    return pbo
