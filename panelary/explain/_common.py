"""Shared helpers for :mod:`panelary.explain`.

Nothing in this module imports a booster, ``shap``, ``shapiq`` or ``sklearn`` at
module scope: the whole ``explain`` subpackage must stay importable with only
``numpy`` + ``polars`` installed. Every heavy import is routed through
:func:`panelary._deps.require` inside the function that needs it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame

__all__ = [
    "BASE_VALUE_COL",
    "REFERENCE_EXPECTATION_COL",
    "RESERVED_COLS",
    "SHAP_PREFIX",
    "attribution_columns",
    "TimeAxis",
    "attach_shap_columns",
    "feature_matrix",
    "key_frame",
    "long_to_wide",
    "model_family",
    "raw_predict",
    "resolve_features",
    "shap_columns",
    "unwrap_model",
    "wide_to_long",
]

#: Default column prefix for attribution columns (``shap_<feature>``).
SHAP_PREFIX = "shap_"

#: Name of the emitted base-value column. Whatever engine produced the
#: attributions, this is the constant that makes ``base + sum(phi) == f(x)``.
BASE_VALUE_COL = "shap_base_value"

#: Audit column emitted in ``conditional`` mode: ``E[f]`` over the fold-bound,
#: past-only reference set. It is *reported*, never substituted for the engine's
#: own base value, so the efficiency axiom keeps holding exactly.
REFERENCE_EXPECTATION_COL = "shap_reference_expectation"

#: Emitted columns that share the ``shap_`` prefix but are **not** per-feature
#: attributions. Every consumer must exclude them, or a base value gets
#: aggregated as if it were a feature's credit.
RESERVED_COLS = (BASE_VALUE_COL, REFERENCE_EXPECTATION_COL)


def attribution_columns(
    df: pl.DataFrame, prefix: str = SHAP_PREFIX, *, exclude: Sequence[str] = ()
) -> list[str]:
    """The per-feature attribution columns of ``df`` (reserved columns excluded)."""
    skip = {*RESERVED_COLS, *exclude}
    return [c for c in df.columns if c.startswith(prefix) and c not in skip]


# --------------------------------------------------------------------------- #
# Feature / frame plumbing
# --------------------------------------------------------------------------- #
def resolve_features(
    panel: PanelFrame,
    features: Sequence[str] | None,
    *,
    exclude: Sequence[str] | None = None,
    who: str = "explain",
) -> list[str]:
    """Resolve the numeric feature columns used for attribution.

    Parameters
    ----------
    panel : PanelFrame
        The panel to resolve against.
    features : sequence of str, optional
        Explicit feature list. ``None`` uses every numeric feature column that
        is not in ``exclude``.
    exclude : sequence of str, optional
        Columns to drop from the automatic pool (e.g. the target).
    who : str
        Caller name, used in error messages.

    Returns
    -------
    list of str
    """
    drop = set(exclude or ())
    schema = panel.schema
    if features is not None:
        feats = list(features)
        missing = [c for c in feats if c not in panel]
        if missing:
            raise ValueError(
                f"{who}: feature column(s) {missing} not found in panel. "
                f"Available columns: {panel.columns}."
            )
    else:
        feats = [c for c in panel.feature_cols if c not in drop]
    if not feats:
        raise ValueError(
            f"{who}: no feature columns available for attribution "
            f"(columns={panel.columns}). Pass `features=` explicitly."
        )
    non_numeric = [c for c in feats if not schema[c].is_numeric()]
    if non_numeric:
        raise ValueError(
            f"{who}: feature column(s) {non_numeric} are not numeric and cannot "
            "be attributed. Encode them first or pass a numeric `features=` list."
        )
    return feats


def feature_matrix(panel: PanelFrame, feats: Sequence[str]) -> np.ndarray:
    """Collect ``feats`` into a float ``(n_rows, n_features)`` numpy matrix."""
    frame = panel.lazy().select(list(feats)).collect()
    return frame.to_numpy().astype(float, copy=False)


def key_frame(panel: PanelFrame) -> pl.DataFrame:
    """Collect just the ``(entity, time)`` key columns, in panel row order."""
    return panel.lazy().select([panel.entity_col, panel.time_col]).collect()


def shap_columns(feats: Sequence[str], prefix: str = SHAP_PREFIX) -> list[str]:
    """Return the attribution column names for ``feats``."""
    return [f"{prefix}{f}" for f in feats]


def attach_shap_columns(
    keys: pl.DataFrame,
    phi: np.ndarray,
    feats: Sequence[str],
    *,
    prefix: str = SHAP_PREFIX,
    base_values: np.ndarray | None = None,
    base_col: str = "shap_base_value",
) -> pl.DataFrame:
    """Build the wide attribution frame: keys + one ``shap_<feature>`` column."""
    feats = list(feats)
    if phi.shape[1] != len(feats):
        raise ValueError(
            f"attribution matrix has {phi.shape[1]} columns but {len(feats)} "
            "features were requested; this is an internal dispatch bug."
        )
    # `nan_to_null=True`: a row with no admissible past-only reference is
    # *un-attributable*, and must read as null rather than as a silent 0.0.
    cols = [
        pl.Series(
            name=f"{prefix}{f}",
            values=phi[:, j].astype(float, copy=False),
            nan_to_null=True,
        )
        for j, f in enumerate(feats)
    ]
    if base_values is not None:
        cols.append(
            pl.Series(
                name=base_col,
                values=np.asarray(base_values, dtype=float),
                nan_to_null=True,
            )
        )
    return keys.with_columns(cols)


def wide_to_long(
    df: pl.DataFrame,
    *,
    entity: str,
    time: str,
    prefix: str = SHAP_PREFIX,
    value_name: str = "shap_value",
    feature_name: str = "feature",
    extra_keep: Sequence[str] = (),
) -> pl.DataFrame:
    """Melt a wide ``shap_<feature>`` frame into a tidy long frame.

    Returns columns ``(entity, time, feature, shap_value)`` (plus ``extra_keep``),
    with the ``prefix`` stripped from the feature names.
    """
    keep = [entity, time, *extra_keep]
    value_cols = attribution_columns(df, prefix, exclude=keep)
    out = df.unpivot(
        index=keep,
        on=value_cols,
        variable_name=feature_name,
        value_name=value_name,
    )
    return out.with_columns(
        pl.col(feature_name).str.strip_prefix(prefix).alias(feature_name)
    )


def long_to_wide(
    df: pl.DataFrame,
    *,
    entity: str,
    time: str,
    prefix: str = SHAP_PREFIX,
    value_name: str = "shap_value",
    feature_name: str = "feature",
) -> pl.DataFrame:
    """Inverse of :func:`wide_to_long`."""
    out = df.pivot(
        on=feature_name,
        index=[entity, time],
        values=value_name,
        aggregate_function="sum",
    )
    renames = {c: f"{prefix}{c}" for c in out.columns if c not in (entity, time)}
    return out.rename(renames)


# --------------------------------------------------------------------------- #
# Time axis
# --------------------------------------------------------------------------- #
class TimeAxis:
    """A frozen, sorted view of a training panel's time axis.

    Used by :class:`~panelary.explain._background.TimeAwareBackground` to
    answer "which training rows are strictly in the past of time ``t``?" in a
    dtype-agnostic way (integers, dates and datetimes all work), without ever
    materialising a global time grid.

    Parameters
    ----------
    times : polars.Series
        The time value of **every** training row (not deduplicated).

    Attributes
    ----------
    unique : polars.Series
        Sorted unique training times.
    row_position : numpy.ndarray
        For each training row, the index of its time in :attr:`unique`.
    """

    __slots__ = ("unique", "row_position", "_order", "_sorted_positions")

    def __init__(self, times: pl.Series) -> None:
        self.unique: pl.Series = times.unique().sort()
        self.row_position: np.ndarray = (
            self.unique.search_sorted(times, side="left").to_numpy().astype(np.int64)
        )
        # Rows ordered by time position, so "everything strictly before t" is a
        # prefix of `self._order` and can be found with one binary search.
        self._order: np.ndarray = np.argsort(self.row_position, kind="stable")
        self._sorted_positions: np.ndarray = self.row_position[self._order]

    def n_unique(self) -> int:
        """Number of distinct training times."""
        return int(self.unique.len())

    def cutoff(self, t: Any, *, embargo: int = 0) -> int:
        """Number of *unique* training times admissible as background for ``t``.

        Admissible times are ``t' < t``; an ``embargo`` of ``k`` additionally
        drops the ``k`` most recent admissible unique times (the same
        purge/embargo convention Panelary's purged CV uses, expressed in
        positions of the training calendar).
        """
        k = int(self.unique.search_sorted(t, side="left"))
        return max(0, k - int(embargo))

    def rows_before(self, t: Any, *, embargo: int = 0) -> np.ndarray:
        """Training row indices whose time is admissible background for ``t``."""
        cut = self.cutoff(t, embargo=embargo)
        if cut <= 0:
            return np.empty(0, dtype=np.int64)
        n = int(np.searchsorted(self._sorted_positions, cut, side="left"))
        return self._order[:n]

    def rows_at(self, t: Any) -> np.ndarray:
        """Training row indices whose time equals ``t`` (empty if absent)."""
        lo = int(self.unique.search_sorted(t, side="left"))
        if lo >= self.n_unique() or self.unique[lo] != t:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(self.row_position == lo)

    def latest_past_position(self, t: Any) -> int:
        """Position of the most recent unique training time strictly before ``t``."""
        return int(self.unique.search_sorted(t, side="left")) - 1

    def rows_at_position(self, pos: int) -> np.ndarray:
        """Training row indices at unique-time position ``pos``."""
        if pos < 0 or pos >= self.n_unique():
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(self.row_position == pos)


# --------------------------------------------------------------------------- #
# Model adapter layer (reuse what `models.py` already wrapped)
# --------------------------------------------------------------------------- #
#: Duck-typing hook. Any object exposing this method is explained directly, with
#: no booster involved. Signature::
#:
#:     panelary_shap_values(X: np.ndarray, background: np.ndarray | None)
#:         -> tuple[np.ndarray, np.ndarray]   # (phi[n, f], base[n])
CUSTOM_HOOK = "panelary_shap_values"


def unwrap_model(model: Any) -> tuple[Any, list[str] | None]:
    """Unwrap a Panelary model wrapper into ``(native_estimator, features)``.

    Accepts either a fitted :mod:`panelary.models` wrapper (anything with
    a fitted ``estimator_`` and a resolved ``features_`` list -- that is the
    adapter layer ``models.py`` already builds around XGBoost / LightGBM /
    CatBoost / sklearn) or a bare fitted estimator, in which case the feature
    names must be supplied by the caller.

    Returns
    -------
    (estimator, features) : tuple
        ``features`` is ``None`` when the model carries no feature names.
    """
    if model is None:
        raise ValueError("`model` is required: pass a fitted Panelary estimator.")
    native = getattr(model, "estimator_", None)
    if native is not None:
        feats = getattr(model, "features_", None)
        return native, (list(feats) if feats else None)
    if hasattr(model, "estimator_") and getattr(model, "_fitted", True) is False:
        raise RuntimeError(
            f"{type(model).__name__} is not fitted; call `fit` before explaining it."
        )
    feats = getattr(model, "feature_names_in_", None)
    return model, ([str(f) for f in feats] if feats is not None else None)


def model_family(estimator: Any) -> str:
    """Classify a fitted estimator without importing any booster.

    Returns one of ``"custom"``, ``"xgboost"``, ``"lightgbm"``, ``"catboost"``,
    ``"sklearn"`` or ``"unknown"``. Detection is by ``__module__`` so that
    ``import panelary.explain`` never drags a booster into ``sys.modules``.
    """
    if hasattr(estimator, CUSTOM_HOOK):
        return "custom"
    top = type(estimator).__module__.split(".", 1)[0]
    if top in {"xgboost", "lightgbm", "catboost", "sklearn"}:
        return top
    return "unknown"


def raw_predict(estimator: Any, X: np.ndarray) -> np.ndarray:
    """Predict on the **raw margin / link scale** the SHAP values live on.

    SHAP values from every tree library are additive in the model's raw output
    space (log-odds for classifiers, the raw score for regressors), so the
    efficiency identity ``base + sum(phi) == raw_prediction`` must be checked
    there, not on probabilities.
    """
    family = model_family(estimator)
    if family == "lightgbm":
        return np.asarray(estimator.predict(X, raw_score=True), dtype=float).ravel()
    if family == "xgboost":
        booster_of = getattr(estimator, "get_booster", None)
        if booster_of is None:
            xgb = _require("xgboost", "raw prediction for attribution")
            return np.asarray(
                estimator.predict(xgb.DMatrix(X), output_margin=True), dtype=float
            ).ravel()
        return np.asarray(estimator.predict(X, output_margin=True), dtype=float).ravel()
    if family == "catboost":
        return np.asarray(
            estimator.predict(X, prediction_type="RawFormulaVal"), dtype=float
        ).ravel()
    out = estimator.predict(X)
    return np.asarray(out, dtype=float).ravel()


def _require(module: str, feature: str) -> Any:
    """Local alias so the heavy import stays inside the calling function."""
    from panelary._deps import require

    return require(module, feature=feature)
