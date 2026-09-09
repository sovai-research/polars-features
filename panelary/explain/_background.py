"""``TimeAwareBackground`` -- the leak-safe reference set for panel attribution.

Every SHAP implementation needs a *background* (reference) distribution: the
thing the model's output is compared **against**. Every SHAP library treats it
casually -- ``shap.TreeExplainer(model, data=X)`` where ``X`` is the whole
dataset is the canonical example -- and that is precisely the dominant leakage
surface in time-series/panel attribution:

* pooling the *whole* dataset puts test-fold rows into a training-fold
  explanation;
* pooling the whole *training* fold puts rows from ``t' > t`` into the
  explanation of an observation at time ``t``;
* the tree-native "path-dependent" mode looks background-free but is not: its
  implicit reference is the training distribution baked into the trees' leaf
  cover counts, which leaks whenever the model itself saw future rows.

:class:`TimeAwareBackground` makes the background a first-class object with an
auditable contract:

**fold-bound**
    It is built from the panel handed to :meth:`fit` (the training fold) and
    carries no rows from anywhere else.
**past-only, per entity's own calendar**
    For an observation at ``(e, t)`` the admissible reference rows are
    ``{(e', t') : t' < t}``, optionally tightened to ``t' <= t - embargo`` using
    the same purge/embargo convention as
    :class:`~panelary.core.model_selection.PurgedKFold` (embargo counts
    *positions of the training calendar*, so it works for integer, date and
    datetime axes alike).
**deterministic**
    Sampling is seeded from ``(seed, n_admissible)``, so the reference set for a
    given ``t`` depends only on the frozen training fold -- never on which other
    rows happen to be in the frame being explained.

Modes
-----
``interventional`` (default)
    Marginal / "true to the model" (Janzing et al. 2020; Chen et al. 2020,
    arXiv:2006.16234). Requires a background. Breaks feature correlations, so
    credit goes only to features the model actually uses -- the right default
    for actionability, at the cost of evaluating the model off-manifold when the
    background is badly chosen (high-momentum + falling-price rows). Keeping the
    background *past-only and fold-bound* is what keeps it on-manifold.
``conditional``
    Observational / "true to the data". For tree ensembles the standard exact
    estimator of the conditional expectation is the trees' own path-dependent
    traversal, so this dispatches to the same native kernel as
    ``path_dependent`` -- **the difference is not the math, it is the reference
    Panelary binds and audits**: ``conditional`` still requires a fold-bound,
    past-only background, which is used for the reported expected value and for
    :meth:`~panelary.explain.TreeAttributor.check_efficiency`.
``path_dependent``
    The raw native call with no external reference at all. The implicit
    background is the training distribution frozen into the model. Allowed only
    with ``i_accept_path_dependent_background=True``; otherwise constructing the
    background raises.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame, as_panel
from panelary.explain._common import TimeAxis, resolve_features

if TYPE_CHECKING:
    from typing import Self

__all__ = ["TimeAwareBackground", "MODES", "POLICIES"]

#: Valid ``mode`` values (see the module docstring).
MODES = ("interventional", "conditional", "path_dependent")

#: Valid ``policy`` values.
#:
#: ``"past"``
#:     Strictly past rows of the training fold (the leak-safe default).
#: ``"cross_sectional"``
#:     The *same date's* cross-section from the training fold, falling back to
#:     the most recent strictly-past training date when the date is not in the
#:     fold. Never pools a future cross-section into a past date's background.
#: ``"fold"``
#:     The whole training fold, ignoring time. Fold-bound (no test rows) but
#:     *not* past-only: a row at ``t`` is explained against references from
#:     ``t' > t`` within the same fold. Warns unless acknowledged.
POLICIES = ("past", "cross_sectional", "fold")


class TimeAwareBackground:
    """A fold-bound, past-only, deterministic reference set for attribution.

    Parameters
    ----------
    mode : {"interventional", "conditional", "path_dependent"}, default="interventional"
        The value-function semantics. See the module docstring.
    policy : {"past", "cross_sectional", "fold"}, default="past"
        Which training rows are admissible as references for time ``t``.
    max_samples : int, default=200
        Cap on the number of reference rows per explained time. Sampling is
        deterministic given ``seed``.
    embargo : int, default=0
        Drop the ``embargo`` most recent admissible *unique training times*
        before sampling, mirroring the purge/embargo used by Panelary's purged
        cross-validators.
    seed : int, default=0
        Sampling seed. The per-time RNG is seeded from ``(seed, n_admissible)``
        so the reference set never depends on the frame being explained.
    min_samples : int, default=1
        Below this many admissible rows the background for that time is treated
        as empty.
    on_empty : {"null", "error"}, default="null"
        What to do for a time with no admissible past. ``"null"`` emits null
        attributions for those rows (with a single warning naming the count);
        ``"error"`` raises.
    i_accept_path_dependent_background : bool, default=False
        Required acknowledgement for ``mode="path_dependent"``.
    i_accept_within_fold_lookahead : bool, default=False
        Silences the warning for ``policy="fold"``.

    Attributes
    ----------
    features_ : list of str
        Feature columns frozen at :meth:`fit` time (attribution column order).
    n_train_rows_ : int
        Number of training rows retained as the reference pool.
    entity_col_, time_col_ : str
        The training panel's panel keys.

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.explain import TimeAwareBackground
    >>> train = pl.DataFrame(
    ...     {"id": ["a", "a", "b", "b"], "t": [1, 2, 1, 2], "x": [1.0, 2.0, 3.0, 4.0]}
    ... )
    >>> bg = TimeAwareBackground(max_samples=10).fit(train, entity="id", time="t")
    >>> bg.matrix_for(2).ravel().tolist()   # only t' < 2 rows
    [1.0, 3.0]
    >>> bg.matrix_for(1) is None            # nothing is strictly before t = 1
    True
    """

    def __init__(
        self,
        *,
        mode: str = "interventional",
        policy: str = "past",
        max_samples: int = 200,
        embargo: int = 0,
        seed: int = 0,
        min_samples: int = 1,
        on_empty: str = "null",
        i_accept_path_dependent_background: bool = False,
        i_accept_within_fold_lookahead: bool = False,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"`mode` must be one of {MODES}, got {mode!r}.")
        if policy not in POLICIES:
            raise ValueError(f"`policy` must be one of {POLICIES}, got {policy!r}.")
        if int(max_samples) < 1:
            raise ValueError(f"`max_samples` must be >= 1, got {max_samples!r}.")
        if int(embargo) < 0:
            raise ValueError(f"`embargo` must be >= 0, got {embargo!r}.")
        if int(min_samples) < 1:
            raise ValueError(f"`min_samples` must be >= 1, got {min_samples!r}.")
        if on_empty not in ("null", "error"):
            raise ValueError(f"`on_empty` must be 'null' or 'error', got {on_empty!r}.")
        if mode == "path_dependent" and not i_accept_path_dependent_background:
            raise ValueError(
                "mode='path_dependent' uses the model's own training "
                "distribution as an *implicit* background: the leaf cover counts "
                "baked into the trees. That reference is invisible, unauditable, "
                "and leaks whenever the model was trained on rows from the "
                "future of the observation being explained. Panelary will not "
                "select it silently.\n\n"
                "Either use mode='interventional' (default) with a fold-bound, "
                "past-only background, or acknowledge the implicit reference "
                "explicitly:\n\n"
                "    TimeAwareBackground(mode='path_dependent', "
                "i_accept_path_dependent_background=True)"
            )
        if mode == "path_dependent":
            warnings.warn(
                "TimeAwareBackground(mode='path_dependent'): attributions will be "
                "computed against the model's implicit training-distribution "
                "reference, which Panelary cannot audit for look-ahead.",
                UserWarning,
                stacklevel=2,
            )
        if policy == "fold" and not i_accept_within_fold_lookahead:
            warnings.warn(
                "TimeAwareBackground(policy='fold') pools the whole training "
                "fold, so a row at time t is explained against reference rows "
                "from t' > t. This is fold-bound but not past-only. Use "
                "policy='past' for the leak-safe default, or pass "
                "i_accept_within_fold_lookahead=True to silence this.",
                UserWarning,
                stacklevel=2,
            )

        self.mode = mode
        self.policy = policy
        self.max_samples = int(max_samples)
        self.embargo = int(embargo)
        self.seed = int(seed)
        self.min_samples = int(min_samples)
        self.on_empty = on_empty
        self.i_accept_path_dependent_background = bool(
            i_accept_path_dependent_background
        )
        self.i_accept_within_fold_lookahead = bool(i_accept_within_fold_lookahead)

        # learned state (frozen at fit time)
        self.features_: list[str] = []
        self.entity_col_: str | None = None
        self.time_col_: str | None = None
        self.n_train_rows_: int = 0
        self._matrix: np.ndarray | None = None
        self._axis: TimeAxis | None = None
        self._cache: dict[tuple[str, int], np.ndarray | None] = {}
        self._empty_warned = False

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(
        self,
        panel: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        features: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> Self:
        """Bind the background to a training fold.

        Only the rows of ``panel`` are ever used as references. Nothing about
        the frame later explained influences the reference set.
        """
        pf = as_panel(panel, entity, time)
        feats = resolve_features(
            pf, features, exclude=exclude, who="TimeAwareBackground.fit"
        )
        frame = pf.lazy().select([pf.time_col, *feats]).collect()
        self.features_ = feats
        self.entity_col_ = pf.entity_col
        self.time_col_ = pf.time_col
        self._matrix = (
            frame.select(feats).to_numpy().astype(float, copy=False)
            if frame.height
            else np.empty((0, len(feats)), dtype=float)
        )
        self.n_train_rows_ = int(frame.height)
        self._axis = TimeAxis(frame.get_column(pf.time_col))
        self._cache = {}
        self._empty_warned = False
        return self

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._axis is not None

    def _check_fitted(self) -> None:
        if self._axis is None or self._matrix is None:
            raise RuntimeError(
                "this TimeAwareBackground is not fitted yet; call "
                "`fit(train_panel)` before requesting reference rows."
            )

    # ------------------------------------------------------------------ #
    # Reference rows
    # ------------------------------------------------------------------ #
    def _admissible(self, t: Any) -> tuple[np.ndarray, int]:
        """Return ``(row_indices, cache_key)`` for the time ``t``."""
        assert self._axis is not None  # noqa: S101 - guarded by _check_fitted
        axis = self._axis
        if self.policy == "fold":
            return np.arange(self.n_train_rows_, dtype=np.int64), -1
        if self.policy == "cross_sectional":
            rows = axis.rows_at(t)
            if rows.size:
                pos = int(axis.unique.search_sorted(t, side="left"))
                return rows, pos
            pos = axis.latest_past_position(t)
            return axis.rows_at_position(pos), pos
        cut = axis.cutoff(t, embargo=self.embargo)
        return axis.rows_before(t, embargo=self.embargo), cut

    def _sample(self, rows: np.ndarray, key: int) -> np.ndarray:
        """Deterministically down-sample ``rows`` to ``max_samples``."""
        if rows.size <= self.max_samples:
            return rows
        # Seed from (seed, admissible-set identity) only -- never from the frame
        # being explained -- so the reference set is reproducible and invariant.
        rng = np.random.default_rng([self.seed, max(int(key), 0), int(rows.size)])
        picked = rng.choice(rows.size, size=self.max_samples, replace=False)
        return rows[np.sort(picked)]

    def rows_for(self, t: Any) -> np.ndarray:
        """Training row indices used as the reference set for time ``t``."""
        self._check_fitted()
        rows, key = self._admissible(t)
        if rows.size < self.min_samples:
            return np.empty(0, dtype=np.int64)
        return self._sample(rows, key)

    def matrix_for(self, t: Any) -> np.ndarray | None:
        """Reference feature matrix for time ``t``, or ``None`` if empty.

        Returns
        -------
        numpy.ndarray or None
            Shape ``(n_reference, n_features)`` with columns ordered as
            :attr:`features_`.
        """
        self._check_fitted()
        assert self._matrix is not None  # noqa: S101
        rows, key = self._admissible(t)
        cache_key = (self.policy, int(key))
        if self.policy != "fold" and cache_key in self._cache:
            return self._cache[cache_key]
        out: np.ndarray | None
        if rows.size < self.min_samples:
            out = None
        else:
            out = self._matrix[self._sample(rows, key)]
        if self.policy != "fold":
            self._cache[cache_key] = out
        return out

    def pooled_matrix(self) -> np.ndarray:
        """The whole retained training matrix (used by ``policy="fold"``)."""
        self._check_fitted()
        assert self._matrix is not None  # noqa: S101
        return self._matrix

    def handle_empty(self, n_rows: int) -> None:
        """Apply the ``on_empty`` policy for ``n_rows`` un-attributable rows."""
        if n_rows <= 0:
            return
        msg = (
            f"{n_rows} row(s) have no admissible past-only background under "
            f"policy={self.policy!r}, embargo={self.embargo} (typically the "
            "earliest times of the panel)."
        )
        if self.on_empty == "error":
            raise ValueError(
                msg + " Pass `on_empty='null'` to emit null attributions for "
                "them, lower `embargo`, or explain a panel that starts after "
                "the training fold."
            )
        if not self._empty_warned:
            self._empty_warned = True
            warnings.warn(
                msg + " Their attributions are null.", UserWarning, stacklevel=3
            )

    # ------------------------------------------------------------------ #
    # Auditability
    # ------------------------------------------------------------------ #
    def describe(self, times: Sequence[Any] | pl.Series) -> pl.DataFrame:
        """Audit table: reference-set size for each supplied time.

        Returns a frame with ``time``, ``n_admissible`` and ``n_reference``
        columns -- the evidence that no explanation saw the future.
        """
        self._check_fitted()
        values = list(times.to_list() if isinstance(times, pl.Series) else times)
        uniq = sorted(set(values), key=lambda v: (v is None, v))
        n_adm, n_ref = [], []
        for t in uniq:
            rows, _ = self._admissible(t)
            n_adm.append(int(rows.size))
            mat = self.matrix_for(t)
            n_ref.append(0 if mat is None else int(mat.shape[0]))
        return pl.DataFrame({"time": uniq, "n_admissible": n_adm, "n_reference": n_ref})

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = (
            f"fitted on {self.n_train_rows_} rows x {len(self.features_)} features"
            if self.is_fitted
            else "not fitted"
        )
        return (
            f"TimeAwareBackground(mode={self.mode!r}, policy={self.policy!r}, "
            f"max_samples={self.max_samples}, embargo={self.embargo}, "
            f"seed={self.seed}, {state})"
        )
