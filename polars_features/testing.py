"""Future-perturbation leak verifiers: the keystone of PanelKit's guarantee.

A transform is *leak-free* if its output at time ``t`` depends only on data at
times ``<= t`` (within each entity). PanelKit's whole "leak-safe" claim is only
worth as much as our ability to *check* it, and the cleanest, model-agnostic
check is a **future-perturbation experiment**:

1. Run the operation on a panel and record its output.
2. Corrupt every value strictly in the *future* (per entity / per the shared
   time axis), leaving the past untouched.
3. Run the operation again.
4. If the operation is leak-free, every output cell in the *past* must be
   **bit-identical** (within ``tol``) across the two runs. Any difference is a
   look-ahead: information from a perturbed future row flowed backwards.

This module exposes two assertions built on that mechanism:

* :func:`assert_no_lookahead` — split the shared time axis at a cut ``t`` and
  assert that perturbing ``time > t`` never changes any output at ``time <= t``.
* :func:`assert_no_train_test_leak` — perturb a *test* fold and assert that the
  *train*-fold outputs are unchanged (the CV-boundary version of the same idea).

Both accept ``op`` as either a :class:`polars.Expr` (applied via
``with_columns``) or a callable ``frame -> frame`` (the callable may take and
return a :class:`~polars_features.core.panel_frame.PanelFrame`,
:class:`polars.DataFrame`, or :class:`polars.LazyFrame`; the calling convention
is auto-detected). Failures name the first offending ``(column, entity, time)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from polars_features.core.panel_frame import PanelFrame, as_panel

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["assert_no_lookahead", "assert_no_train_test_leak"]


# --------------------------------------------------------------------------- #
# Coercion helpers
# --------------------------------------------------------------------------- #
def _as_pf(panel: Any, entity: str | None, time: str | None) -> PanelFrame:
    """Coerce ``panel`` into a :class:`PanelFrame` (keys of an existing one win)."""
    if isinstance(panel, PanelFrame):
        return panel
    return as_panel(panel, entity=entity, time=time)


def _to_df(obj: Any, *, method: str) -> pl.DataFrame:
    """Coerce an op result into an eager :class:`polars.DataFrame`."""
    if isinstance(obj, PanelFrame):
        return obj.collect()
    if isinstance(obj, pl.LazyFrame):
        return obj.collect()
    if isinstance(obj, pl.DataFrame):
        return obj
    raise TypeError(
        f"{method}: `op` returned {type(obj).__name__!r}; a leak check needs a "
        "PanelFrame, polars.DataFrame, or polars.LazyFrame back from `op`."
    )


def _make_apply(
    op: Any, pf: PanelFrame, df0: pl.DataFrame
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Return a ``DataFrame -> DataFrame`` runner for ``op``.

    For a :class:`polars.Expr`, the runner is ``df.with_columns(op)``. For a
    callable, the input calling-convention (bare DataFrame, PanelFrame, or
    LazyFrame) is detected once against ``df0`` and then reused, so both the
    baseline and perturbed runs are driven identically.
    """
    entity, time = pf.entity_col, pf.time_col

    if isinstance(op, pl.Expr):
        return lambda df: df.with_columns(op)

    if not callable(op):
        raise TypeError(
            "`op` must be a polars.Expr or a callable frame->frame, got "
            f"{type(op).__name__!r}."
        )

    wrappers: list[Callable[[pl.DataFrame], Any]] = [
        lambda d: d,  # bare DataFrame
        lambda d: PanelFrame(d, entity=entity, time=time, validate=False),
        lambda d: d.lazy(),  # LazyFrame
    ]
    chosen: Callable[[pl.DataFrame], Any] | None = None
    last_exc: Exception | None = None
    for wrap in wrappers:
        try:
            _to_df(op(wrap(df0)), method="assert_no_lookahead")
        except Exception as exc:  # noqa: BLE001 - probing calling conventions
            last_exc = exc
            continue
        chosen = wrap
        break
    if chosen is None:
        raise TypeError(
            "could not call `op`: it did not accept a polars DataFrame, "
            "PanelFrame, or LazyFrame (or did not return a frame). Last error: "
            f"{last_exc!r}"
        )

    def apply(df: pl.DataFrame) -> pl.DataFrame:
        return _to_df(op(chosen(df)), method="assert_no_lookahead")

    return apply


def _numeric_feature_cols(df: pl.DataFrame, entity: str, time: str) -> list[str]:
    """Numeric columns that are neither the entity nor the time key."""
    keys = {entity, time}
    return [
        name
        for name, dtype in df.schema.items()
        if name not in keys and dtype.is_numeric()
    ]


def _perturb(
    df: pl.DataFrame,
    cols: Sequence[str],
    mask: pl.Expr,
    *,
    seed: int = 0,
) -> pl.DataFrame:
    """Return a copy of ``df`` with a large perturbation added to ``cols``.

    Only rows where ``mask`` is True are altered; the entity/time keys and all
    unmasked rows are left byte-for-byte unchanged. The perturbation is a large
    (~1e6) random offset so any leak shows up far above ``tol``.
    """
    rng = np.random.default_rng(seed)
    out = df
    h = df.height
    for c in cols:
        noise = pl.Series(f"__noise_{c}__", rng.standard_normal(h) * 1.0e6 + 1.0e3)
        out = out.with_columns(noise).with_columns(
            pl.when(mask)
            .then(pl.col(c) + pl.col(f"__noise_{c}__"))
            .otherwise(pl.col(c))
            .alias(c)
        )
        out = out.drop(f"__noise_{c}__")
    return out


def _first_mismatch(
    base: pl.DataFrame,
    pert: pl.DataFrame,
    *,
    entity: str,
    time: str,
    tol: float,
) -> tuple[str, Any, Any] | None:
    """Return ``(column, entity_value, time_value)`` of the first differing cell.

    Both frames must already be filtered to the comparison region and sorted by
    ``(entity, time)``. Returns ``None`` if everything matches within ``tol``.
    """
    if base.height != pert.height:
        raise AssertionError(
            "leak check: the operation changed the number of rows in the "
            f"comparison region ({base.height} vs {pert.height}); the op must "
            "preserve the (entity, time) keys so past rows can be compared."
        )
    common = [c for c in base.columns if c in pert.columns]
    for c in common:
        bs = base[c]
        ps = pert[c]
        if bs.dtype.is_numeric() and ps.dtype.is_numeric():
            one_null = bs.is_null() ^ ps.is_null()
            big = (bs - ps).abs().fill_null(0.0) > tol
            mism = one_null | big
        else:
            mism = bs.ne_missing(ps)
        if bool(mism.any()):
            idx = int(mism.arg_true()[0])
            return c, base[entity][idx], base[time][idx]
    return None


def _assert_invariant(
    op: Any,
    pf: PanelFrame,
    *,
    perturb_mask: pl.Expr,
    compare_mask: pl.Expr,
    tol: float,
    kept_desc: str,
    changed_desc: str,
) -> None:
    """Core mechanism shared by both public assertions.

    Perturb the rows selected by ``perturb_mask``, re-run ``op``, and assert
    every output cell in the ``compare_mask`` region is unchanged within ``tol``.
    """
    entity, time = pf.entity_col, pf.time_col
    df = pf.collect()
    num_cols = _numeric_feature_cols(df, entity, time)
    if not num_cols:
        raise ValueError(
            "leak check: the panel has no numeric feature columns to perturb; "
            "provide a panel whose features are numeric so the future can be "
            "corrupted meaningfully."
        )

    apply = _make_apply(op, pf, df)
    base_out = apply(df)
    pert_df = _perturb(df, num_cols, perturb_mask)
    pert_out = apply(pert_df)

    for out, label in ((base_out, "baseline"), (pert_out, "perturbed")):
        cols = out.columns
        if entity not in cols or time not in cols:
            raise AssertionError(
                f"leak check: the {label} output dropped a key column "
                f"({entity!r}/{time!r}); the op must return a frame that still "
                "carries the (entity, time) keys so outputs can be aligned."
            )

    base_cmp = base_out.filter(compare_mask).sort([entity, time])
    pert_cmp = pert_out.filter(compare_mask).sort([entity, time])

    hit = _first_mismatch(base_cmp, pert_cmp, entity=entity, time=time, tol=tol)
    if hit is not None:
        col, ent_val, time_val = hit
        raise AssertionError(
            "LOOK-AHEAD LEAK DETECTED: perturbing the future "
            f"({changed_desc}) changed output column {col!r} at "
            f"{entity}={ent_val!r}, {time}={time_val!r}, which lies in the "
            f"protected region ({kept_desc}). A leak-free operation's output at "
            "a given time may depend only on data at that time or earlier "
            "(within the entity). Express the feature walk-forward, e.g. "
            f"`expr.shift(k).over({entity!r})` with k >= 0, or a trailing "
            "rolling window."
        )


# --------------------------------------------------------------------------- #
# Public assertions
# --------------------------------------------------------------------------- #
def assert_no_lookahead(
    op: Any,
    panel: Any,
    *,
    entity: str | None = None,
    time: str | None = None,
    tol: float = 1e-9,
    cut: Any = None,
) -> None:
    """Assert ``op`` never looks ahead on ``panel`` (future-perturbation test).

    Splits the shared, sorted unique-time axis at ``cut`` and verifies that
    corrupting every value at ``time > cut`` leaves every output value at
    ``time <= cut`` bit-identical (within ``tol``).

    Parameters
    ----------
    op : polars.Expr or callable
        The operation under test. A :class:`polars.Expr` is applied via
        ``frame.with_columns(op)``. A callable is invoked as ``op(frame)`` and
        may accept/return a :class:`PanelFrame`, :class:`polars.DataFrame`, or
        :class:`polars.LazyFrame` (auto-detected).
    panel : PanelFrame | polars.DataFrame | polars.LazyFrame
        A long-format panel. Bare frames are wrapped via
        :func:`~polars_features.core.panel_frame.as_panel`.
    entity, time : str, optional
        Panel keys, used only when ``panel`` is a bare frame.
    tol : float, default=1e-9
        Absolute tolerance for the "unchanged" comparison of numeric outputs.
    cut : optional
        A specific time value to split at (``time <= cut`` is protected). By
        default the median unique time is used so both sides are non-empty.

    Raises
    ------
    AssertionError
        If any protected (past) output cell changes when the future is
        perturbed — i.e. the operation leaks look-ahead information. The message
        names the first offending column, entity, and time.
    ValueError
        If the panel has fewer than two distinct times, or no numeric features.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_features.core.panel_frame import PanelFrame
    >>> df = pl.DataFrame(
    ...     {"e": ["a"] * 4, "t": [0, 1, 2, 3], "x": [1.0, 2.0, 3.0, 4.0]}
    ... )
    >>> panel = PanelFrame(df, entity="e", time="t")
    >>> assert_no_lookahead(pl.col("x").shift(1).over("e").alias("lag"), panel)
    >>> assert_no_lookahead(  # doctest: +IGNORE_EXCEPTION_DETAIL
    ...     pl.col("x").shift(-1).over("e").alias("lead"), panel
    ... )
    Traceback (most recent call last):
    AssertionError: LOOK-AHEAD LEAK DETECTED: ...
    """
    pf = _as_pf(panel, entity, time)
    times = pf.time_index().to_list()
    if len(times) < 2:
        raise ValueError(
            "assert_no_lookahead needs at least two distinct time steps to split "
            f"past from future, got {len(times)}."
        )
    if cut is None:
        mid = len(times) // 2
        cut = times[mid - 1]  # times[:mid] <= cut, times[mid:] > cut (both non-empty)
    else:
        if cut >= times[-1]:
            raise ValueError(
                f"`cut={cut!r}` leaves no future rows to perturb (max time is "
                f"{times[-1]!r}); choose a smaller cut."
            )
        if cut < times[0]:
            raise ValueError(
                f"`cut={cut!r}` leaves no past rows to protect (min time is "
                f"{times[0]!r}); choose a larger cut."
            )

    tcol = pf.time_col
    _assert_invariant(
        op,
        pf,
        perturb_mask=pl.col(tcol) > cut,
        compare_mask=pl.col(tcol) <= cut,
        tol=tol,
        kept_desc=f"{tcol} <= {cut!r}",
        changed_desc=f"{tcol} > {cut!r}",
    )


def _split_times(x: Any, pf: PanelFrame) -> list[Any]:
    """Extract the set of time values represented by one side of a split."""
    if isinstance(x, PanelFrame):
        return x.collect()[x.time_col].unique().to_list()
    if isinstance(x, pl.LazyFrame):
        return x.select(pf.time_col).collect()[pf.time_col].unique().to_list()
    if isinstance(x, pl.DataFrame):
        return x[pf.time_col].unique().to_list()
    if isinstance(x, pl.Series):
        return x.unique().to_list()
    # Fall back to an array-like of time values.
    return list(np.asarray(x).tolist())


def assert_no_train_test_leak(
    op: Any,
    panel: Any,
    split: tuple[Any, Any],
    *,
    entity: str | None = None,
    time: str | None = None,
    tol: float = 1e-9,
) -> None:
    """Assert perturbing the *test* fold leaves *train*-fold outputs unchanged.

    The cross-validation-boundary form of :func:`assert_no_lookahead`: given a
    ``(train, test)`` split, corrupt every value at the test-fold times and
    verify that no output at a train-fold time changes (within ``tol``). This
    catches transforms that let test-fold information bleed into train-fold
    features.

    Parameters
    ----------
    op : polars.Expr or callable
        The operation under test (see :func:`assert_no_lookahead`).
    panel : PanelFrame | polars.DataFrame | polars.LazyFrame
        The full panel the op runs on.
    split : (train, test)
        A pair whose two members identify the train and test rows. Each member
        may be a :class:`PanelFrame`, a polars frame, or an array-like of time
        values; only its set of time values is used.
    entity, time : str, optional
        Panel keys, used only when ``panel`` is a bare frame.
    tol : float, default=1e-9
        Absolute tolerance for the "unchanged" comparison.

    Raises
    ------
    AssertionError
        If any train-fold output changes when the test fold is perturbed.
    ValueError
        If the split is malformed or the panel has no numeric features.

    Notes
    -----
    For an *interior* test block (train rows exist on both sides of it), even a
    correct backward-looking feature computed on a post-test train row will
    legitimately depend on test-period values; that is precisely why purging
    exists. Use this assertion with walk-forward splits (train entirely before
    test) or with the purged train set to check the guarantee you actually rely
    on.
    """
    if not (isinstance(split, (tuple, list)) and len(split) == 2):
        raise ValueError(
            "`split` must be a (train, test) pair, got "
            f"{type(split).__name__!r} of length "
            f"{len(split) if hasattr(split, '__len__') else '?'}."
        )
    pf = _as_pf(panel, entity, time)
    train_times = _split_times(split[0], pf)
    test_times = _split_times(split[1], pf)
    if not test_times:
        raise ValueError("`split` test fold is empty; nothing to perturb.")
    if not train_times:
        raise ValueError("`split` train fold is empty; nothing to protect.")

    tcol = pf.time_col
    test_lit = pl.Series(values=list(test_times)).implode()
    train_lit = pl.Series(values=list(train_times)).implode()
    _assert_invariant(
        op,
        pf,
        perturb_mask=pl.col(tcol).is_in(test_lit),
        compare_mask=pl.col(tcol).is_in(train_lit),
        tol=tol,
        kept_desc="the train fold",
        changed_desc="the test fold",
    )
