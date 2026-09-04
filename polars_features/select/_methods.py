"""Leak-safe feature-selection primitives for panels.

Three complementary selectors live here, all computed **only on the data passed
in** (the training fold), never on the whole sample:

* :func:`mrmr` -- minimum-Redundancy-Maximum-Relevance selection (Ding & Peng,
  2005). Greedily builds a feature set that is maximally correlated with the
  target yet minimally redundant with the already-selected features. Relevance
  is computed Polars-natively via :func:`polars.corr`.
* :func:`mda` -- Mean-Decrease-Accuracy importance (permutation importance)
  evaluated **through a purged cross-validation splitter** (e.g.
  :class:`~polars_features.core.model_selection.PurgedKFold`), so every
  importance number is an out-of-fold, leak-free estimate. The permutation is
  panel-aware: by default each feature is shuffled **within each entity**
  (``permute_within="entity"``) so the permuted values stay on that entity's
  manifold instead of being swapped across entities/time.
* :func:`mdi` -- Mean-Decrease-Impurity importance read from a fitted tree
  ensemble (in-sample; provided for completeness and speed).

A small :class:`MRMRSelector` wraps :func:`mrmr` as a
:class:`~polars_features.core.protocol.PanelTransformer` so feature selection can
be dropped into a :class:`~polars_features.core.pipeline.Pipeline` as a
``"select"`` step.

Notes
-----
**Leakage contract.** None of these functions looks at data outside the frame /
fold they are handed. :func:`mrmr` and :func:`mdi` are pure functions of their
input rows; :func:`mda` refits the estimator inside each purged fold and scores
on the held-out block, so the ranking never benefits from test-fold information.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from polars_features.core.panel_frame import PanelFrame, as_panel
from polars_features.core.protocol import PanelTransformer

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["mrmr", "mda", "mdi", "MRMRSelector"]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _resolve_target_and_features(
    panel: PanelFrame,
    y: str | None,
    features: Sequence[str] | None,
) -> tuple[str, list[str]]:
    """Resolve the target column and the numeric feature columns to consider."""
    feat_cols = panel.feature_cols
    if y is None:
        if not feat_cols:
            raise ValueError(
                "cannot infer a target: panel has no feature columns "
                f"(columns={panel.columns}). Pass `y=` explicitly."
            )
        target = feat_cols[-1]
    else:
        if y not in panel:
            raise ValueError(
                f"target column {y!r} not found in panel. "
                f"Available columns: {panel.columns}."
            )
        target = y

    schema = panel.schema
    if features is not None:
        feats = list(features)
        missing = [c for c in feats if c not in panel]
        if missing:
            raise ValueError(
                f"feature column(s) {missing} not found in panel. "
                f"Available columns: {panel.columns}."
            )
    else:
        feats = [c for c in feat_cols if c != target]
    numeric_feats = [c for c in feats if schema[c].is_numeric()]
    if not numeric_feats:
        raise ValueError(
            "no numeric feature columns available for selection "
            f"(considered {feats}). Encode/scale features to numeric first."
        )
    return target, numeric_feats


def _is_classifier(estimator: Any) -> bool:
    """Best-effort check whether ``estimator`` is a classifier."""
    try:
        from sklearn.base import is_classifier

        if is_classifier(estimator):
            return True
    except Exception:  # pragma: no cover - defensive
        pass
    return getattr(estimator, "_estimator_type", None) == "classifier"


def _clone_estimator(estimator: Any) -> Any:
    """Return an unfitted clone of ``estimator`` (falls back to the original)."""
    try:
        from sklearn.base import clone

        return clone(estimator)
    except Exception:  # pragma: no cover - non-sklearn estimator
        return estimator


# --------------------------------------------------------------------------- #
# mRMR
# --------------------------------------------------------------------------- #
def mrmr(
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    y: str,
    k: int,
    *,
    features: Sequence[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
) -> list[str]:
    """Select ``k`` features by minimum-Redundancy-Maximum-Relevance (mRMR).

    Relevance is the absolute Pearson correlation between each candidate feature
    and the target ``y``; redundancy is the mean absolute correlation between a
    candidate and the already-selected features. Features are chosen greedily to
    maximise the FCQ score ``relevance / redundancy`` (Ding & Peng, 2005).

    Everything is computed **only from the rows in** ``X`` -- there is no global
    fit, so calling ``mrmr`` on a training fold cannot leak test-fold structure.

    Parameters
    ----------
    X : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format panel containing the candidate features and the target. Bare
        frames are wrapped with :func:`~polars_features.core.panel_frame.as_panel`
        (entity = col 0, time = col 1 by convention) using ``entity`` / ``time``.
    y : str
        Name of the target column inside ``X``.
    k : int
        Number of features to select (``1 <= k <= n_features``).
    features : sequence of str, optional
        Restrict the candidate pool. Defaults to every numeric feature column
        that is neither the entity, the time, nor the target.
    entity, time : str, optional
        Panel keys, used only when ``X`` is a bare polars frame.

    Returns
    -------
    list of str
        The selected feature names, in selection order (most informative first).

    Raises
    ------
    ValueError
        If ``k`` is out of range or no numeric candidate features exist.

    References
    ----------
    Ding, C., & Peng, H. (2005). "Minimum redundancy feature selection from
    microarray gene expression data." *Journal of Bioinformatics and
    Computational Biology*, 3(2).
    """
    panel = as_panel(X, entity, time)
    target, feats = _resolve_target_and_features(panel, y, features)

    if not isinstance(k, int) or k < 1:
        raise ValueError(f"`k` must be a positive integer, got {k!r}.")
    if k > len(feats):
        raise ValueError(
            f"`k`={k} exceeds the number of candidate features ({len(feats)}: {feats})."
        )

    frame = panel.lazy().select([*feats, target]).collect()

    # Relevance: |corr(feature, target)|, computed Polars-natively.
    rel_row = frame.select(
        [pl.corr(pl.col(f), pl.col(target)).abs().alias(f) for f in feats]
    ).row(0)
    relevance = {
        f: (0.0 if v is None or np.isnan(v) else float(v))
        for f, v in zip(feats, rel_row, strict=True)
    }

    # Feature-feature correlation matrix (for redundancy). numpy is the pragmatic
    # choice for the full pairwise matrix; constant columns -> 0 correlation.
    feat_matrix = frame.select(feats).to_numpy()
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(feat_matrix, rowvar=False)
    corr = np.nan_to_num(np.atleast_2d(corr), nan=0.0)
    abs_corr = np.abs(corr)
    idx = {f: i for i, f in enumerate(feats)}

    selected: list[str] = []
    remaining = list(feats)

    # First pick: pure maximum relevance.
    first = max(remaining, key=lambda f: relevance[f])
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:
        best_f, best_score = None, -np.inf
        for f in remaining:
            redundancy = float(np.mean([abs_corr[idx[f], idx[s]] for s in selected]))
            # MID (mutual-information difference) criterion: relevance minus mean
            # redundancy. The difference form is used rather than the quotient
            # because the quotient is numerically unstable when features are
            # near-independent (redundancy -> 0 inflates the score by noise).
            score = relevance[f] - redundancy
            if score > best_score:
                best_f, best_score = f, score
        selected.append(best_f)  # type: ignore[arg-type]
        remaining.remove(best_f)  # type: ignore[arg-type]

    return selected


# --------------------------------------------------------------------------- #
# Mean-Decrease-Accuracy (permutation importance through purged CV)
# --------------------------------------------------------------------------- #
def _score(estimator: Any, X: NDArray[Any], y: NDArray[Any], is_clf: bool) -> float:
    """Return a higher-is-better score for ``estimator`` on ``(X, y)``."""
    preds = estimator.predict(X)
    if is_clf:
        from sklearn.metrics import accuracy_score

        return float(accuracy_score(y, preds))
    from sklearn.metrics import r2_score

    return float(r2_score(y, preds))


def _permute_within(
    values: NDArray[Any],
    groups: NDArray[Any] | None,
    rng: np.random.Generator,
) -> NDArray[Any]:
    """Permute ``values`` **within** each group defined by ``groups``.

    Shuffling a feature only within an entity's own rows (or within a single
    cross-section) keeps every permuted value on that group's manifold -- the
    permuted column still contains only values the group actually produced, so
    the importance measures information *within* the group rather than being
    inflated by cross-entity / cross-time swaps that create off-manifold rows.

    Parameters
    ----------
    values : numpy.ndarray
        1-D array of feature values to shuffle.
    groups : numpy.ndarray | None
        1-D array of group labels aligned with ``values`` (e.g. the entity id
        or the time key of each row). If ``None``, ``values`` is permuted
        globally (the classic, panel-unaware behaviour).
    rng : numpy.random.Generator
        Source of randomness (advanced in place, so repeats differ).

    Returns
    -------
    numpy.ndarray
        A new array; values are only ever moved between rows sharing a group.
    """
    if groups is None:
        return rng.permutation(values)
    out = values.copy()
    # `return_inverse` gives each row its group index; shuffle each group's
    # positions independently. Groups of size <= 1 are left untouched.
    _, inverse = np.unique(groups, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    for gi in range(int(inverse.max()) + 1 if inverse.size else 0):
        idx = np.flatnonzero(inverse == gi)
        if idx.size > 1:
            out[idx] = values[idx[rng.permutation(idx.size)]]
    return out


def mda(
    estimator: Any,
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    y: str,
    cv: Any,
    *,
    features: Sequence[str] | None = None,
    n_repeats: int = 1,
    random_state: int | None = 0,
    permute_within: str | None = "entity",
    entity: str | None = None,
    time: str | None = None,
) -> pl.DataFrame:
    """Mean-Decrease-Accuracy importance through a purged CV splitter.

    For each ``(train, test)`` fold produced by ``cv``, a fresh clone of
    ``estimator`` is fitted on the training block and scored on the held-out
    test block (accuracy for classifiers, :math:`R^2` for regressors). Each
    feature is then permuted (shuffled) in the test block and the drop in score
    recorded. Averaging the drops over folds gives the mean decrease in accuracy;
    the per-fold standard deviation quantifies its stability (de Prado, 2018,
    Ch. 8).

    Because ``cv`` is a *purged* splitter, the training and test blocks never
    overlap in label time, so the importances are leak-free.

    Parameters
    ----------
    estimator : object
        An sklearn-style estimator (``fit`` / ``predict``). It is cloned per
        fold, so the passed instance is left unfitted.
    X : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format panel with the features and the target.
    y : str
        Name of the target column inside ``X``.
    cv : object
        A splitter exposing ``split(panel) -> iterable of (train, test)`` that
        yields :class:`PanelFrame` folds, e.g.
        :class:`~polars_features.core.model_selection.PurgedKFold`.
    features : sequence of str, optional
        Restrict the evaluated features. Defaults to every numeric feature
        column that is not the target.
    n_repeats : int, default=1
        Number of permutation repeats per feature per fold (averaged).
    random_state : int, optional
        Seed for the permutation RNG.
    permute_within : {"entity", "time", None}, default="entity"
        How each feature column is shuffled in the test block.

        * ``"entity"`` (default, leak-safe / panel-correct) -- permute values
          only within each entity's own rows, so the shuffled column stays on
          that entity's manifold. This is the correct panel behaviour: it
          measures a feature's information *within* an entity rather than
          rewarding cross-entity level differences.
        * ``"time"`` -- permute values within each cross-section (all rows
          sharing a time value), useful when the relevant structure is
          cross-sectional.
        * ``None`` -- permute globally across the whole test block (the classic
          pooled behaviour). Retained for back-compat; it can inflate the
          importance of features that merely encode the entity or the date,
          because a global shuffle produces off-manifold rows.
    entity, time : str, optional
        Panel keys, used only when ``X`` is a bare polars frame.

    Returns
    -------
    polars.DataFrame
        Columns ``feature``, ``importance`` (mean decrease in score across
        folds) and ``importance_std``, sorted by ``importance`` descending.

    Raises
    ------
    ValueError
        If no fold produced a usable score (e.g. empty test blocks).

    References
    ----------
    Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*,
    Chapter 8 ("Feature Importance").
    """
    panel = as_panel(X, entity, time)
    target, feats = _resolve_target_and_features(panel, y, features)
    is_clf = _is_classifier(estimator)
    rng = np.random.default_rng(random_state)

    if permute_within not in ("entity", "time", None):
        raise ValueError(
            "mda: `permute_within` must be one of 'entity', 'time' or None, "
            f"got {permute_within!r}."
        )
    if permute_within == "entity":
        key_col: str | None = panel.entity_col
    elif permute_within == "time":
        key_col = panel.time_col
    else:
        key_col = None
    # Keep the grouping key on the collected test block so the permutation can
    # respect entity / cross-section boundaries. It is never fed to the model.
    te_cols = [*feats, target] if key_col is None else [*feats, target, key_col]

    decreases: dict[str, list[float]] = {f: [] for f in feats}
    n_folds_scored = 0

    for train, test in cv.split(panel):
        tr = train.lazy().select([*feats, target]).collect()
        te = test.lazy().select(te_cols).collect()
        if tr.height == 0 or te.height == 0:
            continue
        X_tr = tr.select(feats).to_numpy()
        y_tr = tr.get_column(target).to_numpy()
        X_te = te.select(feats).to_numpy()
        y_te = te.get_column(target).to_numpy()
        groups = None if key_col is None else te.get_column(key_col).to_numpy()
        # Need at least two rows / classes to produce a meaningful score.
        if is_clf and np.unique(y_tr).size < 2:
            continue

        est = _clone_estimator(estimator)
        est.fit(X_tr, y_tr)
        baseline = _score(est, X_te, y_te, is_clf)
        n_folds_scored += 1

        for j, f in enumerate(feats):
            drops = []
            for _ in range(n_repeats):
                X_perm = X_te.copy()
                X_perm[:, j] = _permute_within(X_perm[:, j], groups, rng)
                drops.append(baseline - _score(est, X_perm, y_te, is_clf))
            decreases[f].append(float(np.mean(drops)))

    if n_folds_scored == 0:
        raise ValueError(
            "mda: no fold produced a usable score (empty test blocks or a single "
            "class per training fold). Check the splitter and the panel size."
        )

    rows = []
    for f in feats:
        vals = np.asarray(decreases[f], dtype=float)
        rows.append(
            {
                "feature": f,
                "importance": float(vals.mean()) if vals.size else 0.0,
                "importance_std": float(vals.std(ddof=0)) if vals.size else 0.0,
            }
        )
    return pl.DataFrame(rows).sort("importance", descending=True)


# --------------------------------------------------------------------------- #
# Mean-Decrease-Impurity (from a fitted tree ensemble)
# --------------------------------------------------------------------------- #
def mdi(estimator: Any, features: Sequence[str]) -> pl.DataFrame:
    """Mean-Decrease-Impurity importance from a fitted tree ensemble.

    Reads ``estimator.feature_importances_`` (as produced by sklearn's random
    forests / gradient boosters and LightGBM) and pairs each value with its
    feature name. This is an *in-sample* importance -- fast, but not leak-free
    the way :func:`mda` is; prefer :func:`mda` when leak-safety matters.

    Parameters
    ----------
    estimator : object
        A **fitted** estimator exposing ``feature_importances_``.
    features : sequence of str
        Feature names, in the same order they were fed to ``estimator.fit``.

    Returns
    -------
    polars.DataFrame
        Columns ``feature`` and ``importance``, sorted by ``importance``
        descending.

    Raises
    ------
    AttributeError
        If ``estimator`` has no ``feature_importances_``.
    ValueError
        If the number of importances does not match ``len(features)``.
    """
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        raise AttributeError(
            f"{type(estimator).__name__} has no `feature_importances_`; fit a "
            "tree-based estimator first, or use `mda` for a model-agnostic "
            "importance."
        )
    importances = np.asarray(importances, dtype=float)
    feats = list(features)
    if importances.shape[0] != len(feats):
        raise ValueError(
            f"length mismatch: estimator reported {importances.shape[0]} "
            f"importances but {len(feats)} feature names were given."
        )
    return pl.DataFrame({"feature": feats, "importance": importances}).sort(
        "importance", descending=True
    )


# --------------------------------------------------------------------------- #
# Pipeline-friendly transformer
# --------------------------------------------------------------------------- #
class MRMRSelector(PanelTransformer):
    """Keep the ``k`` best features chosen by :func:`mrmr` (a ``"select"`` step).

    Fits by running :func:`mrmr` on the training panel and remembering the
    selected feature names; transforms by projecting any panel onto
    ``(entity, time)`` + the selected features (+ the target, if present, so a
    downstream estimator can still see it). Selection happens **only on the fit
    panel**, so the step is leak-safe inside cross-validation.

    Parameters
    ----------
    k : int
        Number of features to select.
    target : str, optional
        Target column used to score relevance. If omitted, the last feature
        column is used.
    features : sequence of str, optional
        Candidate pool (see :func:`mrmr`).
    keep_target : bool, default=True
        If True, retain the target column in the transformed panel so a following
        estimator can fit on it.
    entity, time : str, optional
        Default panel keys for bare polars frames.

    Attributes
    ----------
    panel_safe : bool
        Always ``True``.
    leakage_safe : bool
        Always ``True`` -- the feature set is fixed at fit time.
    selected_ : list of str
        The chosen feature names. Available after :meth:`fit`.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        k: int,
        *,
        target: str | None = None,
        features: Sequence[str] | None = None,
        keep_target: bool = True,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"`k` must be a positive integer, got {k!r}.")
        self.k = k
        self.target = target
        self.features = list(features) if features is not None else None
        self.keep_target = bool(keep_target)
        self.selected_: list[str] = []
        self.target_: str | None = None

    def _fit(self, panel: PanelFrame) -> None:
        target, _ = _resolve_target_and_features(panel, self.target, self.features)
        self.target_ = target
        self.selected_ = mrmr(
            panel,
            target,
            self.k,
            features=self.features,
        )

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        missing = [c for c in self.selected_ if c not in panel]
        if missing:
            raise ValueError(
                f"MRMRSelector.transform: selected column(s) {missing} not found "
                f"in panel. Available columns: {panel.columns}."
            )
        keep = list(self.selected_)
        if self.keep_target and self.target_ is not None and self.target_ in panel:
            keep.append(self.target_)
        return panel.select(*(pl.col(c) for c in keep))
