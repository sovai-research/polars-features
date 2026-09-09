"""Any-order Shapley **interactions** via ``shapiq`` interop (optional extra).

This module is *interop only*: it does not implement TreeSHAP-IQ, KernelSHAP-IQ,
SVARM-IQ or Faith-Shap. ``shapiq`` (MIT, arXiv:2410.01649) is the maintained hub
for those, and reimplementing them would be exactly the mistake the ``explain``
subpackage exists to avoid. What Panelary adds is the same thing it adds to
first-order SHAP: a **fold-bound, past-only reference set** and a Polars-native,
``(entity, time)``-keyed result.

Requires the optional dependency::

    pip install 'panelary[explain]'

Vocabulary -- the "interactions" theme
--------------------------------------
Panelary uses one word, *interactions*, for higher-order structure on both sides
of a model, and deliberately keeps the two keyword names distinct so a call site
is never ambiguous:

* **model side (here):** ``max_order=k`` -- the maximum order of *feature*
  interaction Shapley values (``k=2`` is pairwise, ``k=3`` triples...).
* **data side (:mod:`panelary.reduce`):** ``order=3|4`` -- the cumulant
  order of higher-order factor analysis (co-skewness, co-kurtosis).

The flagship combination is running :func:`interaction_values` **on HFA-derived
factors**, which yields statements of the form "factor 2 x factor 5 synergy drove
this forecast".

Value-function mapping
----------------------
``shapiq``'s imputer knob is mirrored onto Panelary's
:class:`~panelary.explain.TimeAwareBackground` mode:

=====================  ===========================================================
Panelary ``mode``      ``shapiq`` engine
=====================  ===========================================================
``interventional``     ``TabularExplainer(imputer="marginal", data=background)``
                       -- the background is the fold-bound, past-only reference
``conditional``        ``TabularExplainer(imputer="conditional", data=background)``
``path_dependent``     ``TreeExplainer`` (TreeSHAP-IQ; implicit tree background)
=====================  ===========================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from panelary._deps import require
from panelary.core.panel_frame import PanelFrame, as_panel
from panelary.explain._background import TimeAwareBackground
from panelary.explain._common import (
    raw_predict,
    resolve_features,
    unwrap_model,
)

__all__ = ["interaction_values", "interaction_matrix"]

_EXTRA = "explain"


def _shapiq() -> Any:
    return require(
        "shapiq", extra=_EXTRA, feature="any-order Shapley interaction values"
    )


def _iv_items(iv: Any) -> list[tuple[tuple[int, ...], float]]:
    """Normalise a ``shapiq.InteractionValues`` into ``[(feature_idx, value)]``."""
    dict_values = getattr(iv, "dict_values", None)
    if dict_values:
        return [(tuple(k), float(v)) for k, v in dict_values.items()]
    lookup = getattr(iv, "interaction_lookup", None)
    values = getattr(iv, "values", None)
    if lookup is None or values is None:  # pragma: no cover - shapiq API drift
        raise TypeError(
            "could not read interaction values off the shapiq result "
            f"({type(iv).__name__}); this Panelary build expects `dict_values` "
            "or `interaction_lookup`/`values`."
        )
    arr = np.asarray(values, dtype=float).ravel()
    return [(tuple(k), float(arr[i])) for k, i in lookup.items()]


def interaction_values(
    model: Any,
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    background: TimeAwareBackground,
    max_order: int = 2,
    index: str = "k-SII",
    features: Sequence[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
    min_order: int = 1,
    max_rows: int | None = None,
    class_index: int | None = None,
    **explainer_kwargs: Any,
) -> pl.DataFrame:
    """Panel-keyed Shapley interaction values of order up to ``max_order``.

    Parameters
    ----------
    model : object
        A fitted Panelary model wrapper or bare estimator.
    X : panel
        Rows to explain. Interaction values are expensive; use ``max_rows``.
    background : TimeAwareBackground
        The fold-bound, past-only reference set. Its ``mode`` selects the
        ``shapiq`` engine (see the module docstring).
    max_order : int, default=2
        Maximum interaction order (``2`` = pairwise). Panelary uses
        ``max_order`` on the model side and ``order`` on the data side
        (:mod:`panelary.reduce`) -- see the module docstring.
    index : str, default="k-SII"
        ``shapiq`` interaction index (``"k-SII"``, ``"FSII"``, ``"SII"``, ...).
    min_order : int, default=1
        Lowest order to keep in the output.
    max_rows : int, optional
        Explain only the first ``max_rows`` rows (after panel sorting).
    class_index : int, optional
        Output index for multiclass models.
    **explainer_kwargs
        Forwarded to the ``shapiq`` explainer.

    Returns
    -------
    polars.DataFrame
        Tidy long frame: ``entity``, ``time``, ``order``, ``features``
        (``"|"``-joined feature names), ``interaction_value``.

    Raises
    ------
    ImportError
        If ``shapiq`` is not installed (``pip install
        'panelary[explain]'``).
    """
    if int(max_order) < 1:
        raise ValueError(f"`max_order` must be >= 1, got {max_order!r}.")
    if not isinstance(background, TimeAwareBackground):
        raise TypeError(
            "`background` must be a TimeAwareBackground so interaction values "
            "inherit the same fold-bound, past-only reference contract as "
            "first-order attributions."
        )
    if background.mode != "path_dependent" and not background.is_fitted:
        raise RuntimeError(
            "the TimeAwareBackground is not fitted; call `background.fit("
            "train_panel)` on the training fold first."
        )

    shapiq = _shapiq()
    panel = as_panel(X, entity, time)
    native, model_feats = unwrap_model(model)
    feats = resolve_features(
        panel,
        list(features) if features is not None else (model_feats or None),
        who="interaction_values",
    )
    frame = (
        panel.sort_panel()
        .lazy()
        .select([panel.entity_col, panel.time_col, *feats])
        .collect()
    )
    if max_rows is not None:
        frame = frame.head(int(max_rows))
    mat = frame.select(feats).to_numpy().astype(float, copy=False)
    ents = frame.get_column(panel.entity_col).to_list()
    times = frame.get_column(panel.time_col).to_list()

    def _predict(arr: np.ndarray) -> np.ndarray:
        return raw_predict(native, np.asarray(arr, dtype=float))

    rows_e: list[Any] = []
    rows_t: list[Any] = []
    rows_o: list[int] = []
    rows_f: list[str] = []
    rows_v: list[float] = []

    if background.mode == "path_dependent":
        explainer = shapiq.TreeExplainer(
            model=native,
            index=index,
            max_order=int(max_order),
            min_order=int(min_order),
            class_index=class_index,
            **explainer_kwargs,
        )
        for i in range(mat.shape[0]):
            for idx, val in _iv_items(explainer.explain(mat[i])):
                if not (min_order <= len(idx) <= max_order):
                    continue
                rows_e.append(ents[i])
                rows_t.append(times[i])
                rows_o.append(len(idx))
                rows_f.append("|".join(feats[j] for j in idx))
                rows_v.append(val)
    else:
        imputer = "marginal" if background.mode == "interventional" else "conditional"
        # One explainer per distinct reference set; the canonical walk-forward
        # case (a test fold after the training fold) yields exactly one.
        from panelary.explain._tree import _group_rows_by_background

        n_empty = 0
        for bg, idx_list in _group_rows_by_background(
            background, frame.get_column(panel.time_col)
        ):
            if bg is None or bg.shape[0] == 0:
                n_empty += len(idx_list)
                continue
            explainer = shapiq.TabularExplainer(
                model=_predict,
                data=bg,
                index=index,
                max_order=int(max_order),
                imputer=imputer,
                random_state=background.seed,
                **explainer_kwargs,
            )
            for i in idx_list:
                for tup, val in _iv_items(explainer.explain(mat[i])):
                    if not (min_order <= len(tup) <= max_order):
                        continue
                    rows_e.append(ents[i])
                    rows_t.append(times[i])
                    rows_o.append(len(tup))
                    rows_f.append("|".join(feats[j] for j in tup))
                    rows_v.append(val)
        background.handle_empty(n_empty)

    return pl.DataFrame(
        {
            panel.entity_col: rows_e,
            panel.time_col: rows_t,
            "order": rows_o,
            "features": rows_f,
            "interaction_value": rows_v,
        }
    )


def interaction_matrix(
    values: pl.DataFrame,
    *,
    features: Sequence[str] | None = None,
    order: int = 2,
    agg: str = "mean_abs",
    feature_col: str = "features",
    value_col: str = "interaction_value",
) -> pl.DataFrame:
    """Collapse a long interaction frame into a feature x feature matrix.

    Parameters
    ----------
    values : polars.DataFrame
        Output of :func:`interaction_values`.
    features : sequence of str, optional
        Row/column order of the matrix. Inferred from the data when omitted.
    order : int, default=2
        Which interaction order to lay out (only ``2`` forms a matrix).
    agg : {"mean_abs", "mean"}, default="mean_abs"
        How to pool across panel rows.

    Returns
    -------
    polars.DataFrame
        Square frame with a leading ``feature`` column.
    """
    if int(order) != 2:
        raise ValueError(
            f"`interaction_matrix` lays out pairwise interactions; got order="
            f"{order!r}. Filter `values` yourself for higher orders."
        )
    if agg not in ("mean_abs", "mean"):
        raise ValueError(f"`agg` must be 'mean_abs' or 'mean', got {agg!r}.")
    pairs = values.filter(pl.col("order") == 2)
    parts = pairs.get_column(feature_col).str.split("|").to_list()
    vals = pairs.get_column(value_col).to_numpy().astype(float, copy=False)
    if features is None:
        names = sorted({f for p in parts for f in p})
    else:
        names = list(features)
    pos = {f: i for i, f in enumerate(names)}
    acc = np.zeros((len(names), len(names)), dtype=float)
    cnt = np.zeros((len(names), len(names)), dtype=float)
    for p, v in zip(parts, vals, strict=False):
        if len(p) != 2 or p[0] not in pos or p[1] not in pos:
            continue
        a, b = pos[p[0]], pos[p[1]]
        x = abs(v) if agg == "mean_abs" else v
        acc[a, b] += x
        acc[b, a] += x
        cnt[a, b] += 1
        cnt[b, a] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mat = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    out = {"feature": names}
    for j, f in enumerate(names):
        out[f] = mat[:, j]
    return pl.DataFrame(out)
