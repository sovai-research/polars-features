from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import (
        Literal,
    )

    from panelary._internal._type_aliases import PolarsFrame

import numpy as np
import polars as pl

__all__ = [
    "train_test_split",
    "expanding_window_split",
    "sliding_window_split",
]


@overload
def train_test_split(
    test_size: int | float = ...,
    eager: Literal[True] = ...,
) -> Callable[[PolarsFrame], tuple[pl.DataFrame, pl.DataFrame]]: ...


@overload
def train_test_split(
    test_size: int | float = ...,
    eager: Literal[False] = ...,
) -> Callable[[PolarsFrame], tuple[pl.DataFrame, pl.DataFrame]]: ...


@overload
def train_test_split(
    test_size: int | float = ...,
    eager: bool = ...,
) -> Callable[
    [PolarsFrame],
    tuple[pl.DataFrame, pl.DataFrame] | tuple[pl.LazyFrame, pl.LazyFrame],
]: ...


def train_test_split(
    test_size: int | float = 0.25,
    eager: bool = False,
) -> Callable[
    [PolarsFrame],
    tuple[pl.DataFrame, pl.DataFrame] | tuple[pl.LazyFrame, pl.LazyFrame],
]:
    """Return a time-ordered train set and test set given `test_size`.

    Parameters
    ----------
    test_size : int | float, default=0.25
        Number or fraction of test samples.
    eager : bool, default=False
        If True, evaluate immediately and returns tuple of train-test `DataFrame`.

    Returns
    -------
    splitter : Union[EagerSplitter, LazySplitter]
        Function that takes a panel DataFrame, or LazyFrame, and returns:
        * A tuple of train / test LazyFrames, if `eager=False`.
        * A tuple of train / test DataFrames, if `eager=True`.
    """
    if isinstance(test_size, float):
        if test_size < 0 or test_size > 1:
            raise ValueError("`test_size` must be between 0 and 1")
    elif isinstance(test_size, int):
        if test_size < 0:
            raise ValueError("`test_size` must be greater than 0")
    else:
        raise TypeError("`test_size` must be int or float")

    def splitter(
        X: PolarsFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame] | tuple[pl.LazyFrame, pl.LazyFrame]:
        """Split the data into train and test sets."""

        return _splitter_train_test(
            X=X,
            test_size=test_size,
            eager=eager,
        )

    return splitter


@overload
def _splitter_train_test(
    X: PolarsFrame,
    test_size: int | float,
    eager: Literal[True] = ...,
) -> tuple[pl.DataFrame, pl.DataFrame]: ...


@overload
def _splitter_train_test(
    X: PolarsFrame,
    test_size: int | float,
    eager: Literal[False] = ...,
) -> tuple[pl.LazyFrame, pl.LazyFrame]: ...


@overload
def _splitter_train_test(
    X: PolarsFrame,
    test_size: int | float,
    eager: bool = ...,
) -> tuple[pl.DataFrame, pl.DataFrame] | tuple[pl.LazyFrame, pl.LazyFrame]: ...


def _splitter_train_test(
    X: PolarsFrame,
    test_size: int | float,
    eager: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame] | tuple[pl.LazyFrame, pl.LazyFrame]:
    if isinstance(X, pl.DataFrame):
        X = X.lazy()

    entity_col = X.columns[0]

    max_size = (
        X.group_by(entity_col).agg(pl.len()).select(pl.min("len")).collect().item()
    )

    if isinstance(test_size, int) and test_size > max_size:
        raise ValueError(
            "`test_size` must be less than the number of samples of the smallest entity"
        )

    train_length = (
        pl.len() - test_size
        if isinstance(test_size, int)
        else (pl.len() * (1 - test_size)).cast(int)
    )
    test_length = pl.len() - train_length

    train_split = (
        X.group_by(entity_col)
        .agg(pl.all().slice(offset=0, length=train_length))
        .explode(pl.all().exclude(entity_col))
    )
    test_split = (
        X.group_by(entity_col)
        .agg(pl.all().slice(offset=train_length, length=test_length))
        .explode(pl.all().exclude(entity_col))
    )
    if eager:
        train_split, test_split = pl.collect_all([train_split, test_split])
        return train_split, test_split
    return train_split, test_split


def expanding_window_split(
    test_size: int, n_splits: int = 5, step_size: int = 1, eager: bool = False
):
    """Return train/test splits using expanding window splitter.

    Split time series repeatedly into an growing training set and a fixed-size test set.
    For example, given `test_size = 3`, `n_splits = 5` and `step_size = 1`,
    the train `o`s and test `x`s folds can be visualized as:

    ```
    | o o o x x x - - - - |
    | o o o o x x x - - - |
    | o o o o o x x x - - |
    | o o o o o o x x x - |
    | o o o o o o o x x x |
    ```

    Parameters
    ----------
    test_size : int
        Number of test samples for each split.
    n_splits : int, default=5
        Number of splits.
    step_size : int, default=1
        Step size between windows.
    eager : bool, default=False
        If True return DataFrames. Otherwise, return LazyFrames.

    Returns
    -------
    splitter : Callable[pl.LazyFrame, Mapping[int, Tuple[pl.LazyFrame, pl.LazyFrame]]]
        Function that takes a panel LazyFrame and Dict of (train, test) splits, where
        the key represents the split number (1,2,...,n_splits) and the value is a tuple of LazyFrames.

    See Also
    --------
    panelary.core.model_selection.expanding_window_split : the panel-aware
        splitter of the same name. It follows the same fold schedule (both call
        :func:`_walk_forward_cutoffs`) but slices the shared unique-time index
        and returns ``PanelFrame`` folds, so every entity is tested on the same
        dates. This one slices each entity's own rows, so on a ragged panel
        entities are tested on different dates. ``pn.expanding_window_split``
        is the panel-aware one.
    """

    def split(X: pl.LazyFrame) -> pl.LazyFrame:
        splits = _window_split(X, test_size, n_splits, step_size)
        if eager:
            splits = {i: pl.collect_all(s) for i, s in splits.items()}
        return splits

    return split


def sliding_window_split(
    test_size: int,
    n_splits: int = 5,
    step_size: int = 1,
    window_size: int = 10,
    eager: bool = False,
):
    """Return train/test splits using sliding window splitter.
    Split time series repeatedly into a fixed-length training and test set.
    For example, given `test_size = 3`, `n_splits = 5`, `step_size = 1` and `window_size=5`
    the train `o`s and test `x`s folds can be visualized as:

    ```
    | o o o o o x x x - - - - |
    | - o o o o o x x x - - - |
    | - - o o o o o x x x - - |
    | - - - o o o o o x x x - |
    | - - - - o o o o o x x x |
    ```

    Parameters
    ----------
    test_size : int
        Number of test samples for each split.
    n_splits : int, default=5
        Number of splits.
    step_size : int, default=1
        Step size between windows.
    window_size: int, default=10
        Window size for training.
    eager : bool, default=False
        If True return DataFrames. Otherwise, return LazyFrames.

    Returns
    -------
    splitter : Callable[pl.LazyFrame, Mapping[int, Tuple[pl.LazyFrame, pl.LazyFrame]]]
        Function that takes a panel LazyFrame and Dict of (train, test) splits, where
        the key represents the split number (1,2,...,n_splits) and the value is a tuple of LazyFrames.

    Notes
    -----
    Splits whose history is shorter than ``window_size`` get a training window
    truncated at the start of that entity's history, never one that wraps around
    to the end of it.

    See Also
    --------
    panelary.core.model_selection.sliding_window_split : the panel-aware
        splitter of the same name. Same fold schedule (both call
        :func:`_walk_forward_cutoffs`), different axis: it slices the shared
        unique-time index and returns ``PanelFrame`` folds, whereas this one
        slices each entity's own rows. ``pn.sliding_window_split`` is the
        panel-aware one.
    """

    def split(X: pl.LazyFrame) -> pl.LazyFrame:
        splits = _window_split(X, test_size, n_splits, step_size, window_size)
        if eager:
            splits = {i: pl.collect_all(s) for i, s in splits.items()}
        return splits

    return split


def _walk_forward_cutoffs(
    test_size: int,
    n_splits: int,
    step_size: int,
) -> list[int]:
    """Return the walk-forward fold schedule, oldest test block first.

    Entry ``i`` is the distance from the **end** of the index to the first
    position of split ``i``'s test block, so on an index of length ``n`` that
    block occupies ``[n - cutoffs[i], n - cutoffs[i] + test_size)``. The last
    split always ends on the last observation (``cutoffs[-1] == test_size``) and
    each earlier split is a further ``step_size`` back.

    This is the whole of the walk-forward arithmetic and it is deliberately the
    only copy in the library: the panel-aware splitters in
    :mod:`panelary.core.model_selection` import this function rather than
    recomputing the schedule, so the two splitter families cannot drift on where
    folds fall. What differs between them is the **axis** they then slice, never
    the schedule — see :func:`_window_split`.

    The schedule is a function of the parameters alone; it never consults the
    length of the data, so it is prefix-invariant by construction.

    Parameters
    ----------
    test_size : int
        Number of observations in each test block.
    n_splits : int
        Number of splits.
    step_size : int
        Distance between the start of consecutive test blocks.

    Returns
    -------
    list of int
        ``n_splits`` cutoffs, descending (oldest fold first).
    """
    backward_steps = np.arange(1, n_splits) * step_size + test_size
    cutoffs = np.flip(np.concatenate([np.array([test_size]), backward_steps]))
    return [int(cutoff) for cutoff in cutoffs]


def _window_split(
    X: pl.LazyFrame,
    test_size: int,
    n_splits: int,
    step_size: int,
    window_size: int | None = None,
) -> Mapping[int, tuple[pl.LazyFrame, pl.LazyFrame]]:
    """Walk-forward splits along **each entity's own row count**.

    The fold schedule comes from :func:`_walk_forward_cutoffs`; the slicing is
    per entity, inside ``group_by(entity_col)``, so on a ragged panel every
    entity is split relative to its own last observation. That is the one thing
    that separates these splitters from the panel-aware ones in
    :mod:`panelary.core.model_selection`, which slice the shared unique-time
    index so that all entities are trained and tested on the same dates.
    """
    X = X.lazy()  # Defensive
    cutoffs = _walk_forward_cutoffs(test_size, n_splits, step_size)
    entity_col = X.columns[0]

    train_exprs: list[pl.Expr]
    # TODO: split in two functions?
    if window_size:
        # Sliding window CV. The training window ends where the test block
        # begins and is truncated at the start of history: a bare
        # `pl.len() - cutoff - window_size` offset goes *negative* when there is
        # less than `window_size` of history, and a negative Polars slice offset
        # counts back from the end of the group — which would train the earliest
        # folds on rows that come after their own test block.
        train_exprs = []
        for cutoff in cutoffs:
            test_start = pl.len().cast(pl.Int64) - cutoff
            train_start = pl.max_horizontal(test_start - window_size, pl.lit(0))
            train_length = pl.max_horizontal(test_start - train_start, pl.lit(0))
            train_exprs.append(pl.all().slice(train_start, train_length))
    else:
        # Expanding window CV
        train_exprs = [pl.all().slice(0, pl.len() - cutoff) for cutoff in cutoffs]

    test_exprs = [pl.all().slice(-cutoffs[i], test_size) for i in range(n_splits)]
    train_test_exprs = zip(train_exprs, test_exprs, strict=False)

    splits = {}
    for i, train_test_expr in enumerate(train_test_exprs):
        train_expr, test_expr = train_test_expr
        train_split = (
            X.group_by(entity_col).agg(train_expr).explode(pl.all().exclude(entity_col))
        )
        test_split = (
            X.group_by(entity_col).agg(test_expr).explode(pl.all().exclude(entity_col))
        )
        splits[i] = train_split, test_split
    return splits
