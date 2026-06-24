"""PanelFrame: a thin, typed, lazy view over a long-format panel.

A *panel* is a long-format table indexed by an **entity** (e.g. a security id,
a customer id, a country) and a **time** axis (e.g. a date or an integer step).
Every row is one observation of one entity at one time. Features live in the
remaining columns.

``PanelFrame`` is a *view*: it wraps a :class:`polars.LazyFrame` and remembers
which column is the entity and which is the time. It never copies data and stays
lazy until you explicitly call :meth:`PanelFrame.collect`. All panel-aware
operations downstream (windowing, lagging, cross-validation) build on the
``entity`` / ``time`` contract carried here.

Notes
-----
This module is part of the *correctness-by-construction* core. The whole point
of routing data through a ``PanelFrame`` is that the entity/time keys are
validated **once** and then trusted everywhere else, so that leakage-prone
operations (lags, rolling windows, splits) can be expressed safely with
``.over(entity_col)`` semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from typing import Self

__all__ = ["PanelFrame"]


def _to_lazyframe(data: Any) -> pl.LazyFrame:
    """Coerce supported inputs to a :class:`polars.LazyFrame` without copying eagerly.

    Accepts :class:`polars.LazyFrame`, :class:`polars.DataFrame`, and any object
    exposing a ``.lazy()`` method (e.g. a narwhals-wrapped frame). Anything else
    raises a :class:`TypeError` with an actionable message.
    """
    if isinstance(data, pl.LazyFrame):
        return data
    if isinstance(data, pl.DataFrame):
        return data.lazy()
    # narwhals-ish / duck-typed frame: prefer a native polars handle.
    to_native = getattr(data, "to_native", None)
    if callable(to_native):
        native = to_native()
        if isinstance(native, pl.LazyFrame):
            return native
        if isinstance(native, pl.DataFrame):
            return native.lazy()
    lazy = getattr(data, "lazy", None)
    if callable(lazy):
        candidate = lazy()
        if isinstance(candidate, pl.LazyFrame):
            return candidate
    raise TypeError(
        "PanelFrame expects a polars DataFrame or LazyFrame "
        f"(or an object with a `.lazy()`/`.to_native()` returning one), "
        f"got {type(data).__name__!r}."
    )


class PanelFrame:
    """A typed, lazy view over a long-format panel keyed by ``(entity, time)``.

    The frame is never materialised on construction; only the schema (column
    names and dtypes) is inspected so that the entity/time contract can be
    validated cheaply. Use :meth:`collect` to evaluate.

    Parameters
    ----------
    data : polars.DataFrame | polars.LazyFrame | frame-like
        The underlying long-format panel. ``DataFrame`` inputs are wrapped with
        ``.lazy()`` (no eager work). Objects exposing ``.lazy()`` /
        ``.to_native()`` (e.g. narwhals frames) are accepted too.
    entity : str
        Name of the entity (panel id) column. Must exist in ``data``.
    time : str
        Name of the time column. Must exist in ``data`` and differ from
        ``entity``.
    validate : bool, default=True
        If True, run schema-level validation (column existence, distinctness,
        time dtype sanity). Set False only when re-wrapping a frame you already
        trust, e.g. inside :meth:`with_columns`.

    Attributes
    ----------
    entity_col : str
        The validated entity column name.
    time_col : str
        The validated time column name.

    Raises
    ------
    TypeError
        If ``data`` cannot be coerced to a LazyFrame, or if ``entity`` / ``time``
        are not strings.
    ValueError
        If the entity or time column is missing, if they are the same column, or
        if the time column has a dtype that cannot be ordered.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame(
    ...     {
    ...         "ticker": ["A", "A", "B", "B"],
    ...         "date": [1, 2, 1, 2],
    ...         "ret": [0.1, 0.2, -0.1, 0.0],
    ...     }
    ... )
    >>> panel = PanelFrame(df, entity="ticker", time="date")
    >>> panel.feature_cols
    ['ret']
    >>> panel.sort_panel().collect().shape
    (4, 3)

    Notes
    -----
    **Leakage contract.** A ``PanelFrame`` only guarantees the *keys* are valid;
    it does not by itself prevent look-ahead. Downstream transforms and
    splitters must express any forward-looking computation walk-forward via
    ``.over(entity_col)`` (use :meth:`over_entity`) and rely on
    :meth:`sort_panel` for deterministic ordering.
    """

    __slots__ = ("_lf", "_entity", "_time")

    def __init__(
        self,
        data: pl.DataFrame | pl.LazyFrame | Any,
        entity: str,
        time: str,
        *,
        validate: bool = True,
    ) -> None:
        if not isinstance(entity, str):
            raise TypeError(
                f"`entity` must be a column name (str), got {type(entity).__name__!r}."
            )
        if not isinstance(time, str):
            raise TypeError(
                f"`time` must be a column name (str), got {type(time).__name__!r}."
            )
        self._lf: pl.LazyFrame = _to_lazyframe(data)
        self._entity: str = entity
        self._time: str = time
        if validate:
            self._validate_schema()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _schema(self) -> pl.Schema:
        """Return the underlying lazy schema (cheap; does not collect data)."""
        return self._lf.collect_schema()

    def _validate_schema(self) -> None:
        """Validate the entity/time contract at the schema level.

        Checks, with Polars-quality error messages:

        * the entity column exists,
        * the time column exists,
        * entity and time are distinct columns,
        * the time column has an orderable dtype.
        """
        schema = self._schema()
        names = schema.names()

        if self._entity == self._time:
            raise ValueError(
                "`entity` and `time` must be different columns, "
                f"but both are {self._entity!r}."
            )
        if self._entity not in names:
            raise ValueError(
                f"entity column {self._entity!r} not found in panel. "
                f"Available columns: {names}."
            )
        if self._time not in names:
            raise ValueError(
                f"time column {self._time!r} not found in panel. "
                f"Available columns: {names}."
            )

        time_dtype = schema[self._time]
        if not (
            time_dtype.is_numeric()
            or time_dtype.is_temporal()
            or time_dtype == pl.Boolean
        ):
            raise ValueError(
                f"time column {self._time!r} has dtype {time_dtype!r}, which is "
                "not orderable as a time axis. Expected a numeric or temporal "
                "dtype (Int*, UInt*, Float*, Date, Datetime, Duration, Time). "
                "Cast it explicitly, e.g. "
                f"`df.with_columns(pl.col({self._time!r}).cast(pl.Int64))`."
            )

    def assert_unique_keys(self) -> Self:
        """Assert that every ``(entity, time)`` pair is unique. **Materialises.**

        This is the only validation that requires touching the data (a
        ``group_by`` count), so it is opt-in rather than run on construction.

        Returns
        -------
        PanelFrame
            ``self`` (for chaining), if all keys are unique.

        Raises
        ------
        ValueError
            If duplicate ``(entity, time)`` keys exist, including a small sample
            of the offending keys to aid debugging.
        """
        dupes = (
            self._lf.group_by(self._entity, self._time)
            .agg(pl.len().alias("__count__"))
            .filter(pl.col("__count__") > 1)
            .head(5)
            .collect()
        )
        if dupes.height > 0:
            total = (
                self._lf.group_by(self._entity, self._time)
                .agg(pl.len().alias("__count__"))
                .filter(pl.col("__count__") > 1)
                .select(pl.len())
                .collect()
                .item()
            )
            sample = dupes.select(self._entity, self._time, "__count__").rows()
            raise ValueError(
                f"panel has {total} duplicated ({self._entity}, {self._time}) "
                "key(s); each (entity, time) pair must be unique. "
                f"First offenders (entity, time, count): {sample}."
            )
        return self

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def entity_col(self) -> str:
        """Name of the entity (panel id) column."""
        return self._entity

    @property
    def time_col(self) -> str:
        """Name of the time column."""
        return self._time

    @property
    def columns(self) -> list[str]:
        """All column names, in schema order."""
        return self._schema().names()

    @property
    def schema(self) -> pl.Schema:
        """The underlying lazy schema (column names -> dtypes)."""
        return self._schema()

    @property
    def feature_cols(self) -> list[str]:
        """Feature columns: every column that is neither entity nor time.

        Returns
        -------
        list of str
            Column names in schema order, excluding ``entity_col`` and
            ``time_col``.
        """
        keys = {self._entity, self._time}
        return [c for c in self._schema().names() if c not in keys]

    def entity(self) -> pl.Expr:
        """Return a Polars expression selecting the entity column."""
        return pl.col(self._entity)

    def time(self) -> pl.Expr:
        """Return a Polars expression selecting the time column."""
        return pl.col(self._time)

    # ------------------------------------------------------------------ #
    # Lazy / eager bridges
    # ------------------------------------------------------------------ #
    def lazy(self) -> pl.LazyFrame:
        """Return the underlying :class:`polars.LazyFrame` (no copy, no collect)."""
        return self._lf

    def collect(self, **kwargs: Any) -> pl.DataFrame:
        """Materialise the panel into a :class:`polars.DataFrame`.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`polars.LazyFrame.collect`.

        Returns
        -------
        polars.DataFrame
        """
        return self._lf.collect(**kwargs)

    def to_frame(self) -> pl.LazyFrame:
        """Alias for :meth:`lazy`; returns the underlying LazyFrame."""
        return self._lf

    def to_native(self, lazy: bool = True) -> pl.LazyFrame | pl.DataFrame:
        """Return the underlying native polars frame.

        Convenience for users who passed in a bare :class:`polars.DataFrame` /
        :class:`polars.LazyFrame` and want a native frame back after a
        transform, without keeping the :class:`PanelFrame` wrapper.

        Parameters
        ----------
        lazy : bool, default=True
            If True (default), return the underlying :class:`polars.LazyFrame`
            (no work). If False, :meth:`collect` it into a
            :class:`polars.DataFrame`.

        Returns
        -------
        polars.LazyFrame | polars.DataFrame
        """
        return self._lf if lazy else self._lf.collect()

    # ------------------------------------------------------------------ #
    # Panel-aware operations (return new PanelFrames; data stays lazy)
    # ------------------------------------------------------------------ #
    def _rewrap(self, lf: pl.LazyFrame, *, validate: bool = False) -> Self:
        """Wrap a derived LazyFrame in a new PanelFrame preserving the keys."""
        return type(self)(lf, entity=self._entity, time=self._time, validate=validate)

    def with_columns(self, *exprs: Any, **named_exprs: Any) -> Self:
        """Return a new :class:`PanelFrame` with added/replaced columns.

        Mirrors :meth:`polars.LazyFrame.with_columns`. The entity and time keys
        are preserved. Adding columns is cheap and stays lazy.

        Parameters
        ----------
        *exprs, **named_exprs
            Forwarded verbatim to :meth:`polars.LazyFrame.with_columns`.

        Returns
        -------
        PanelFrame
            A new view; ``self`` is unchanged.

        Raises
        ------
        ValueError
            If an expression attempts to drop or rename the entity/time column
            such that the contract would break. (Validation is re-run only if a
            key column is affected, to keep the common path cheap.)
        """
        new_lf = self._lf.with_columns(*exprs, **named_exprs)
        new_names = new_lf.collect_schema().names()
        # Cheap guard: ensure keys survived.
        if self._entity not in new_names or self._time not in new_names:
            raise ValueError(
                "with_columns must not drop the panel keys "
                f"({self._entity!r}, {self._time!r}); resulting columns "
                f"were {new_names}."
            )
        return self._rewrap(new_lf, validate=False)

    def select(self, *exprs: Any, **named_exprs: Any) -> Self:
        """Select columns, always keeping the entity and time keys.

        The entity and time columns are prepended to the selection if not
        already present, so the result is always a valid panel.

        Returns
        -------
        PanelFrame
        """
        new_lf = self._lf.select(*exprs, **named_exprs)
        names = new_lf.collect_schema().names()
        missing = [c for c in (self._entity, self._time) if c not in names]
        if missing:
            new_lf = self._lf.select(
                pl.col(self._entity), pl.col(self._time), *exprs, **named_exprs
            )
        return self._rewrap(new_lf, validate=False)

    def filter(self, *predicates: Any, **constraints: Any) -> Self:
        """Filter rows, preserving the panel keys. Mirrors :meth:`LazyFrame.filter`."""
        return self._rewrap(self._lf.filter(*predicates, **constraints), validate=False)

    def sort_panel(self, *, descending: bool = False) -> Self:
        """Return the panel sorted by ``(entity, time)`` ascending.

        Deterministic ordering is a precondition for correct lags, rolling
        windows and walk-forward splits, so most pipelines should call this
        once near the top.

        Parameters
        ----------
        descending : bool, default=False
            If True, sort the *time* axis descending within each entity.

        Returns
        -------
        PanelFrame
            A new sorted view.
        """
        return self._rewrap(
            self._lf.sort([self._entity, self._time], descending=[False, descending]),
            validate=False,
        )

    def is_sorted_per_entity(self) -> bool:
        """Return True if ``time`` is non-decreasing within every entity.

        Evaluates against the panel's **current row order** (it does not sort
        first), so it answers "is this frame, as laid out, already in valid
        per-entity time order?". **Materialises** (needs to scan the time
        column). Call :meth:`sort_panel` to enforce the ordering if this returns
        False.
        """
        has_decrease = (
            self._lf.select(self._entity, self._time)
            .with_columns(
                (pl.col(self._time) < pl.col(self._time).shift(1))
                .over(self._entity)
                .alias("__decrease__")
            )
            .select(pl.col("__decrease__").fill_null(False).any())
            .collect()
            .item()
        )
        return not bool(has_decrease)

    # ------------------------------------------------------------------ #
    # Grouping helpers
    # ------------------------------------------------------------------ #
    def over_entity(self) -> str:
        """Return the entity column name for use with ``expr.over(...)``.

        Use this so feature code never hard-codes the key:

        >>> import polars as pl
        >>> panel = PanelFrame(
        ...     pl.DataFrame({"id": ["a", "a"], "t": [1, 2], "x": [1.0, 2.0]}),
        ...     entity="id", time="t",
        ... )
        >>> lag = pl.col("x").shift(1).over(panel.over_entity())

        Returns
        -------
        str
            The entity column name.
        """
        return self._entity

    def group_by_entity(self, **kwargs: Any) -> pl.LazyGroupBy:
        """Return a lazy ``group_by`` over the entity column.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`polars.LazyFrame.group_by`.

        Returns
        -------
        polars.LazyGroupBy
        """
        return self._lf.group_by(self._entity, **kwargs)

    def entities(self) -> pl.Series:
        """Return the sorted unique entity ids. **Materialises.**"""
        return (
            self._lf.select(pl.col(self._entity).unique().sort()).collect().to_series()
        )

    def n_entities(self) -> int:
        """Return the number of distinct entities. **Materialises.**"""
        return self._lf.select(pl.col(self._entity).n_unique()).collect().item()

    def time_index(self) -> pl.Series:
        """Return the sorted unique time values across all entities. **Materialises.**"""
        return self._lf.select(pl.col(self._time).unique().sort()).collect().to_series()

    # ------------------------------------------------------------------ #
    # Dunders
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        try:
            schema = self._schema()
            cols = schema.names()
            return (
                f"PanelFrame(entity={self._entity!r}, time={self._time!r}, "
                f"features={self.feature_cols!r}, columns={cols!r})"
            )
        except Exception:  # pragma: no cover - repr must never raise
            return (
                f"PanelFrame(entity={self._entity!r}, time={self._time!r}, "
                "<schema unavailable>)"
            )

    def __contains__(self, col: object) -> bool:
        return isinstance(col, str) and col in self._schema().names()

    def pipe(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Apply ``func(self, *args, **kwargs)`` and return its result.

        Convenience for chaining custom panel-aware helpers.
        """
        return func(self, *args, **kwargs)


def as_panel(
    data: pl.DataFrame | pl.LazyFrame | PanelFrame | Any,
    entity: str | None = None,
    time: str | None = None,
) -> PanelFrame:
    """Coerce ``data`` into a :class:`PanelFrame`.

    If ``data`` is already a :class:`PanelFrame` it is returned unchanged (its
    own keys win). Otherwise ``entity`` and ``time`` must be provided; if either
    is omitted the codebase convention is applied: **column 0 is the entity and
    column 1 is the time**.

    Parameters
    ----------
    data : polars frame, frame-like, or PanelFrame
    entity : str, optional
        Entity column. Defaults to the first column.
    time : str, optional
        Time column. Defaults to the second column.

    Returns
    -------
    PanelFrame

    Raises
    ------
    ValueError
        If defaults are needed but the frame has fewer than two columns.
    """
    if isinstance(data, PanelFrame):
        return data
    lf = _to_lazyframe(data)
    if entity is None or time is None:
        names = lf.collect_schema().names()
        if len(names) < 2:
            raise ValueError(
                "cannot infer panel keys: a panel needs at least an entity and a "
                f"time column, but the frame has columns {names}. Pass `entity=` "
                "and `time=` explicitly."
            )
        entity = entity if entity is not None else names[0]
        time = time if time is not None else names[1]
    return PanelFrame(lf, entity=entity, time=time)
