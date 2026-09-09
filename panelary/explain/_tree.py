"""Native, exact TreeSHAP dispatch -- Panelary reimplements none of the math.

Exact TreeSHAP already ships **inside** the boosters Panelary wraps, and the
reference interventional engine ships inside ``shap``. This module is a
dispatcher, not an algorithm:

======================  ==========================================================
model family            engine
======================  ==========================================================
XGBoost                 ``Booster.predict(..., pred_contribs=True)`` (path-dependent);
                        ``shap.TreeExplainer(..., feature_perturbation=
                        "interventional")`` for the interventional value function
LightGBM                ``predict(..., pred_contrib=True)``; ``shap`` for interventional
CatBoost                ``get_feature_importance(type="ShapValues")``; natively
                        interventional via ``reference_data=`` +
                        ``shap_calc_type="Independent"``
scikit-learn ensembles  ``shap.TreeExplainer`` (both value functions)
custom                  any object exposing ``panelary_shap_values(X, background)``
======================  ==========================================================

Every one of those imports is lazy (:func:`panelary._internal._deps.require`), so
``import panelary.explain`` costs nothing but numpy + polars.

What Panelary *adds* is the argument the libraries leave to the user: the
reference set. See :class:`~panelary.explain.TimeAwareBackground`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from panelary._internal._deps import require
from panelary.core.panel_frame import PanelFrame, as_panel
from panelary.explain._background import TimeAwareBackground
from panelary.explain._common import (
    BASE_VALUE_COL,
    CUSTOM_HOOK,
    REFERENCE_EXPECTATION_COL,
    SHAP_PREFIX,
    attach_shap_columns,
    feature_matrix,
    model_family,
    raw_predict,
    resolve_features,
    unwrap_model,
    wide_to_long,
)

__all__ = [
    "tree_attributions",
    "check_efficiency",
    "BASE_VALUE_COL",
    "REFERENCE_EXPECTATION_COL",
]

_SHAP_EXTRA = "explain"


# --------------------------------------------------------------------------- #
# Native engine adapters
# --------------------------------------------------------------------------- #
def _split_bias(arr: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a native ``(n, n_features + 1)`` contribution block into (phi, base)."""
    if arr.shape[1] != n_features + 1:
        raise ValueError(
            f"native contributions have {arr.shape[1]} columns; expected "
            f"{n_features + 1} (one per feature plus the bias term). This "
            "usually means the model was trained on a different feature set "
            "than the one being explained."
        )
    return arr[:, :n_features], arr[:, n_features]


def _select_output(arr: Any, n_features: int, class_index: int | None) -> np.ndarray:
    """Reduce a possibly multi-output contribution object to a 2-D array."""
    if isinstance(arr, list):
        if class_index is None and len(arr) > 1:
            raise ValueError(
                f"the model produces {len(arr)} outputs (multiclass). Pass "
                "`class_index=` to choose which one to attribute."
            )
        return np.asarray(arr[class_index or 0], dtype=float)
    a = np.asarray(arr, dtype=float)
    if a.ndim == 3:
        if class_index is None and a.shape[2] > 1:
            raise ValueError(
                f"the model produces {a.shape[2]} outputs (multiclass). Pass "
                "`class_index=` to choose which one to attribute."
            )
        return a[:, :, class_index or 0]
    if a.ndim == 2 and a.shape[1] > n_features + 1:
        block = n_features + 1
        if a.shape[1] % block:
            raise ValueError(
                f"cannot interpret a contribution matrix of shape {a.shape} for "
                f"{n_features} features."
            )
        k = a.shape[1] // block
        if class_index is None and k > 1:
            raise ValueError(
                f"the model produces {k} outputs (multiclass). Pass "
                "`class_index=` to choose which one to attribute."
            )
        return a.reshape(a.shape[0], k, block)[:, class_index or 0, :]
    return a


def _native_path_dependent(
    native: Any, X: np.ndarray, feats: Sequence[str], class_index: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Path-dependent (tree-native) contributions, including the bias column."""
    family = model_family(native)
    n_features = len(feats)

    if family == "custom":
        phi, base = getattr(native, CUSTOM_HOOK)(X, None)
        return np.asarray(phi, dtype=float), np.asarray(base, dtype=float).ravel()

    if family == "lightgbm":
        require("lightgbm", feature="LightGBM attribution")
        raw = native.predict(X, pred_contrib=True)
        arr = _select_output(raw, n_features, class_index)
        return _split_bias(arr, n_features)

    if family == "xgboost":
        xgb = require("xgboost", feature="XGBoost attribution")
        booster = native.get_booster() if hasattr(native, "get_booster") else native
        raw = booster.predict(
            xgb.DMatrix(X, feature_names=list(feats)), pred_contribs=True
        )
        arr = _select_output(raw, n_features, class_index)
        return _split_bias(arr, n_features)

    if family == "catboost":
        cb = require("catboost", feature="CatBoost attribution")
        raw = native.get_feature_importance(type="ShapValues", data=cb.Pool(X))
        arr = _select_output(raw, n_features, class_index)
        return _split_bias(arr, n_features)

    return _shap_explainer(native, None, X, feats, class_index)


def _shap_explainer(
    native: Any,
    background: np.ndarray | None,
    X: np.ndarray,
    feats: Sequence[str],
    class_index: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Route through the reference ``shap`` TreeExplainer (exact, C++ kernel)."""
    shap = require(
        "shap", extra=_SHAP_EXTRA, feature="model-agnostic / interventional TreeSHAP"
    )
    if background is None:
        explainer = shap.TreeExplainer(
            native, feature_perturbation="tree_path_dependent"
        )
    else:
        explainer = shap.TreeExplainer(
            native, data=background, feature_perturbation="interventional"
        )
    values = _select_output(explainer.shap_values(X), len(feats), class_index)
    expected = explainer.expected_value
    if isinstance(expected, (list, tuple, np.ndarray)):
        expected = np.asarray(expected, dtype=float).ravel()
        base = np.full(X.shape[0], float(expected[class_index or 0]))
    else:
        base = np.full(X.shape[0], float(expected))
    return np.asarray(values, dtype=float), base


def _interventional(
    native: Any,
    background: np.ndarray,
    X: np.ndarray,
    feats: Sequence[str],
    class_index: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Marginal ("true to the model") contributions against ``background``."""
    family = model_family(native)

    if family == "custom":
        phi, base = getattr(native, CUSTOM_HOOK)(X, background)
        return np.asarray(phi, dtype=float), np.asarray(base, dtype=float).ravel()

    if family == "catboost":
        cb = require("catboost", feature="CatBoost interventional attribution")
        raw = native.get_feature_importance(
            type="ShapValues",
            data=cb.Pool(X),
            reference_data=cb.Pool(background),
            shap_calc_type="Independent",
        )
        arr = _select_output(raw, len(feats), class_index)
        return _split_bias(arr, len(feats))

    return _shap_explainer(native, background, X, feats, class_index)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def _resolve_explain_features(
    panel: PanelFrame,
    model_feats: Sequence[str] | None,
    features: Sequence[str] | None,
    background: TimeAwareBackground,
) -> list[str]:
    if features is not None:
        feats = list(features)
    elif model_feats:
        feats = list(model_feats)
    elif background.is_fitted and background.features_:
        feats = list(background.features_)
    else:
        feats = None
    feats = resolve_features(panel, feats, who="tree_attributions")
    if background.is_fitted and background.features_ != feats:
        raise ValueError(
            "the background was fitted on features "
            f"{background.features_} but the attribution asks for {feats}. The "
            "reference set must cover exactly the model's features, in the same "
            "order -- refit the background with `features=` matching the model."
        )
    return feats


def tree_attributions(
    model: Any,
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    background: TimeAwareBackground,
    features: Sequence[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
    output: str = "wide",
    prefix: str = SHAP_PREFIX,
    class_index: int | None = None,
    include_base: bool = True,
) -> pl.DataFrame:
    """Exact, native TreeSHAP for a fitted Panelary model, keyed by ``(entity, time)``.

    Parameters
    ----------
    model : object
        A **fitted** Panelary model wrapper (e.g.
        :class:`~panelary.models.PanelLGBMRegressor`) or a bare fitted
        booster / sklearn tree ensemble. Panelary wrappers carry their resolved
        feature list, so ``features`` can be omitted.
    X : PanelFrame | polars.DataFrame | polars.LazyFrame
        The panel to explain.
    background : TimeAwareBackground
        The reference set. Must be fitted on the **training fold** unless
        ``mode="path_dependent"``.
    features : sequence of str, optional
        Override the feature list (must match the background's).
    entity, time : str, optional
        Panel keys when ``X`` is a bare polars frame.
    output : {"wide", "long"}, default="wide"
        ``"wide"`` emits one ``shap_<feature>`` column per feature;
        ``"long"`` emits a tidy ``(entity, time, feature, shap_value)`` frame.
    prefix : str, default="shap_"
        Attribution column prefix.
    class_index : int, optional
        Which output to attribute for multiclass models.
    include_base : bool, default=True
        Emit the ``shap_base_value`` column (``E[f]`` over the reference set).

    Returns
    -------
    polars.DataFrame

    Notes
    -----
    Rows whose time has no admissible past-only reference (typically the first
    dates of a panel) get **null** attributions rather than a silently pooled
    background -- see ``TimeAwareBackground(on_empty=...)``.
    """
    if output not in ("wide", "long"):
        raise ValueError(f"`output` must be 'wide' or 'long', got {output!r}.")
    if not isinstance(background, TimeAwareBackground):
        raise TypeError(
            "`background` must be a TimeAwareBackground -- the fold-bound, "
            "past-only reference set that makes panel attribution leak-safe. "
            "Build one with "
            "`TimeAwareBackground().fit(train_panel, features=...)`."
        )
    if background.mode != "path_dependent" and not background.is_fitted:
        raise RuntimeError(
            f"the TimeAwareBackground (mode={background.mode!r}) is not fitted; "
            "call `background.fit(train_panel)` on the *training* fold first."
        )

    panel = as_panel(X, entity, time)
    native, model_feats = unwrap_model(model)
    feats = _resolve_explain_features(panel, model_feats, features, background)

    frame = panel.lazy().select([panel.entity_col, panel.time_col, *feats]).collect()
    mat = frame.select(feats).to_numpy().astype(float, copy=False)
    keys = frame.select([panel.entity_col, panel.time_col])
    n_rows = mat.shape[0]

    phi = np.full((n_rows, len(feats)), np.nan, dtype=float)
    base = np.full(n_rows, np.nan, dtype=float)
    reference_expectation: np.ndarray | None = None

    if background.mode in ("path_dependent", "conditional"):
        # Same exact tree kernel; the difference is what Panelary binds and
        # audits as the reference (see the _background module docstring).
        if n_rows:
            phi, base = _native_path_dependent(native, mat, feats, class_index)
        if background.mode == "conditional" and background.is_fitted and n_rows:
            # *Report* E[f] over the fold-bound, past-only reference set beside
            # the engine's own base value. Substituting it would silently break
            # the efficiency axiom, so it gets its own audit column.
            reference_expectation = _reference_expectation(
                native, background, frame, panel
            )
    else:
        groups = _group_rows_by_background(background, frame.get_column(panel.time_col))
        n_empty = 0
        for bg_mat, idx in groups:
            rows = np.asarray(idx, dtype=np.int64)
            if bg_mat is None:
                n_empty += rows.size
                continue
            g_phi, g_base = _interventional(
                native, bg_mat, mat[rows], feats, class_index
            )
            phi[rows] = g_phi
            base[rows] = g_base
        background.handle_empty(n_empty)

    out = attach_shap_columns(
        keys,
        phi,
        feats,
        prefix=prefix,
        base_values=base if include_base else None,
        base_col=BASE_VALUE_COL,
    )
    if reference_expectation is not None:
        out = out.with_columns(
            pl.Series(name=REFERENCE_EXPECTATION_COL, values=reference_expectation)
        )
    keep_cols = [
        c for c in (BASE_VALUE_COL, REFERENCE_EXPECTATION_COL) if c in out.columns
    ]
    if output == "long":
        return wide_to_long(
            out,
            entity=panel.entity_col,
            time=panel.time_col,
            prefix=prefix,
            extra_keep=keep_cols,
        )
    return out


def _group_rows_by_background(
    background: TimeAwareBackground, times: pl.Series
) -> list[tuple[np.ndarray | None, list[int]]]:
    """Bucket row positions by their (cached, identical) reference matrix.

    Rows sharing a reference set are explained in one native call. In the
    canonical walk-forward case -- explaining a test fold that starts after the
    training fold -- every row shares one reference set, so this is a single
    call.
    """
    buckets: dict[int, tuple[np.ndarray | None, list[int]]] = {}
    for i, t in enumerate(times.to_list()):
        mat = background.matrix_for(t)
        key = -1 if mat is None else id(mat)
        if key not in buckets:
            buckets[key] = (mat, [])
        buckets[key][1].append(i)
    return [(mat, idx) for mat, idx in buckets.values()]


def _reference_expectation(
    native: Any,
    background: TimeAwareBackground,
    frame: pl.DataFrame,
    panel: PanelFrame,
) -> np.ndarray:
    """``E[f]`` over each row's own fold-bound, past-only reference set."""
    out = np.full(frame.height, np.nan, dtype=float)
    for mat, idx in _group_rows_by_background(
        background, frame.get_column(panel.time_col)
    ):
        if mat is None or mat.shape[0] == 0:
            continue
        out[np.asarray(idx, dtype=np.int64)] = float(np.mean(raw_predict(native, mat)))
    return out


# --------------------------------------------------------------------------- #
# Efficiency (the local-accuracy axiom)
# --------------------------------------------------------------------------- #
def check_efficiency(
    model: Any,
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    background: TimeAwareBackground,
    features: Sequence[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
    class_index: int | None = None,
    tol: float = 1e-5,
    raise_on_fail: bool = False,
) -> pl.DataFrame:
    """Verify the Shapley efficiency axiom ``E[f] + sum_j phi_j == f(x)``.

    The identity holds on the model's **raw margin scale** (log-odds for
    classifiers), which is where every tree library's SHAP values live.

    Returns
    -------
    polars.DataFrame
        One row: ``n_rows``, ``n_checked``, ``max_abs_error``,
        ``mean_abs_error``, ``tol``, ``ok``.
    """
    panel = as_panel(X, entity, time)
    native, model_feats = unwrap_model(model)
    feats = _resolve_explain_features(panel, model_feats, features, background)
    attrs = tree_attributions(
        model,
        panel,
        background=background,
        features=feats,
        output="wide",
        class_index=class_index,
        include_base=True,
    )
    shap_cols = [f"{SHAP_PREFIX}{f}" for f in feats]
    total = (
        attrs.select(shap_cols).sum_horizontal().to_numpy()
        + attrs.get_column(BASE_VALUE_COL).to_numpy()
    )
    pred = raw_predict(native, feature_matrix(panel, feats))
    err = np.abs(total - pred)
    finite = np.isfinite(err)
    max_err = float(np.max(err[finite])) if finite.any() else float("nan")
    mean_err = float(np.mean(err[finite])) if finite.any() else float("nan")
    ok = bool(finite.any() and max_err <= tol)
    if raise_on_fail and not ok:
        raise AssertionError(
            f"SHAP efficiency check failed: max |E[f] + sum(phi) - f(x)| = "
            f"{max_err:.3g} > tol={tol:.3g} over {int(finite.sum())} checked "
            "rows. Either the attribution engine and the prediction disagree on "
            "the output scale (raw margin vs probability), or the background "
            "does not match the model's feature set."
        )
    return pl.DataFrame(
        {
            "n_rows": [int(err.size)],
            "n_checked": [int(finite.sum())],
            "max_abs_error": [max_err],
            "mean_abs_error": [mean_err],
            "tol": [float(tol)],
            "ok": [ok],
        }
    )
