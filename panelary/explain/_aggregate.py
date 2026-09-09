"""Panel-native aggregation of attributions: group SHAP and per-entity window SHAP.

Two things make this *panel* attribution rather than "SHAP with a ``.over()``":

**Group SHAP with an honest ``group_mode``.**
    Summing member attributions (``group_mode="additive"``) is *exact* for
    marginal (interventional) Shapley values by additivity of the value
    function, and it is the right antidote to the multicollinearity that makes
    per-feature financial attributions unstable. It is **not**, however, the
    value a feature *group* would receive if the group were held out jointly:
    that requires treating the group as a single coalition player, which is a
    different game. :func:`joint_group_shap` computes that one exactly by
    enumerating coalitions over the (few) groups, using the same fold-bound,
    past-only :class:`~panelary.explain.TimeAwareBackground` as the
    marginal imputer. Conflating the two is a real and common error, so Panelary
    makes you choose.

**Windows on each entity's own calendar.**
    :func:`window_shap` aggregates attributions over trailing per-entity
    windows -- never a global time grid. Entities with ragged calendars (late
    listings, halted names, different observation frequencies) keep their own
    history. Window aggregation also reduces the dependence between attributed
    units, which is why WindowSHAP-style summaries are more stable than
    per-cell SHAP.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from math import factorial
from typing import Any

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame, as_panel
from panelary.explain._background import TimeAwareBackground
from panelary.explain._common import (
    BASE_VALUE_COL,
    SHAP_PREFIX,
    attribution_columns,
    raw_predict,
    resolve_features,
    unwrap_model,
)

__all__ = [
    "GROUP_PREFIX",
    "group_shap",
    "joint_group_shap",
    "window_shap",
]

#: Column prefix for group attributions.
GROUP_PREFIX = "shap_group_"


# --------------------------------------------------------------------------- #
# Group definitions
# --------------------------------------------------------------------------- #
def _validate_groups(
    groups: Mapping[str, Sequence[str]], available: Sequence[str], who: str
) -> dict[str, list[str]]:
    if not groups:
        raise ValueError(
            f"{who}: `groups` must map at least one group name to features."
        )
    out: dict[str, list[str]] = {}
    seen: dict[str, str] = {}
    pool = set(available)
    for name, members in groups.items():
        members = list(members)
        if not members:
            raise ValueError(f"{who}: group {name!r} is empty.")
        missing = [m for m in members if m not in pool]
        if missing:
            raise ValueError(
                f"{who}: group {name!r} references feature(s) {missing} that are "
                f"not available. Available: {sorted(pool)}."
            )
        for m in members:
            if m in seen:
                raise ValueError(
                    f"{who}: feature {m!r} appears in both group {seen[m]!r} and "
                    f"{name!r}. Groups must be a partition (disjoint), otherwise "
                    "attributions are double-counted."
                )
            seen[m] = name
        out[name] = members
    return out


# --------------------------------------------------------------------------- #
# Additive group SHAP
# --------------------------------------------------------------------------- #
def group_shap(
    shap: pl.DataFrame,
    groups: Mapping[str, Sequence[str]],
    *,
    entity: str,
    time: str,
    prefix: str = SHAP_PREFIX,
    out_prefix: str = GROUP_PREFIX,
    keep_ungrouped: bool = True,
    ungrouped_name: str = "ungrouped",
    keep_base: bool = True,
) -> pl.DataFrame:
    """Sum attributions within a feature partition (**exact** by additivity).

    Parameters
    ----------
    shap : polars.DataFrame
        A *wide* attribution frame from
        :func:`~panelary.explain.tree_attributions` (keys +
        ``shap_<feature>`` columns).
    groups : mapping of str to sequence of str
        Group name -> member feature names. Must be disjoint.
    entity, time : str
        Panel key columns present in ``shap``.
    prefix : str, default="shap_"
        Prefix of the input attribution columns.
    out_prefix : str, default="shap_group_"
        Prefix of the emitted group columns.
    keep_ungrouped : bool, default=True
        Emit a residual group holding every attribution column not covered by
        ``groups``, so the group columns still sum to the row's total
        attribution (the efficiency axiom survives grouping).
    ungrouped_name : str, default="ungrouped"
        Name of that residual group.
    keep_base : bool, default=True
        Carry the ``shap_base_value`` column through if present.

    Returns
    -------
    polars.DataFrame
        ``(entity, time)`` plus one ``shap_group_<name>`` column per group.

    Notes
    -----
    This is exact for **marginal** attributions only, and it answers "how much
    of the prediction do these features explain *together, as a sum of
    individual credits*". For "what would the prediction lose if the whole group
    were held out jointly", use :func:`joint_group_shap`.
    """
    value_cols = attribution_columns(shap, prefix, exclude=(entity, time))
    feat_of = {c[len(prefix) :]: c for c in value_cols}
    resolved = _validate_groups(groups, list(feat_of), "group_shap")

    exprs = [
        pl.sum_horizontal([pl.col(feat_of[m]) for m in members]).alias(
            f"{out_prefix}{name}"
        )
        for name, members in resolved.items()
    ]
    covered = {m for members in resolved.values() for m in members}
    leftover = [feat_of[f] for f in feat_of if f not in covered]
    if keep_ungrouped and leftover:
        exprs.append(
            pl.sum_horizontal([pl.col(c) for c in leftover]).alias(
                f"{out_prefix}{ungrouped_name}"
            )
        )
    keep = [entity, time]
    if keep_base and BASE_VALUE_COL in shap.columns:
        keep.append(BASE_VALUE_COL)
    return shap.select([*[pl.col(c) for c in keep], *exprs])


# --------------------------------------------------------------------------- #
# Joint (coalition) group SHAP
# --------------------------------------------------------------------------- #
def _shapley_weights(n_players: int) -> np.ndarray:
    """``|S|! (n - |S| - 1)! / n!`` indexed by ``|S|``."""
    n = n_players
    return np.array(
        [factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)],
        dtype=float,
    )


def joint_group_shap(
    model: Any,
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    groups: Mapping[str, Sequence[str]],
    *,
    background: TimeAwareBackground,
    features: Sequence[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
    include_ungrouped: bool = True,
    ungrouped_name: str = "ungrouped",
    out_prefix: str = GROUP_PREFIX,
    max_groups: int = 12,
    max_evals: int = 20_000_000,
    chunk_rows: int = 256,
) -> pl.DataFrame:
    """Exact Shapley values with each **group as a single coalition player**.

    The value function is the marginal (interventional) one -- ``v(S) =
    E_background[f(x_S, b_{-S})]`` -- imputed from the same fold-bound,
    past-only :class:`~panelary.explain.TimeAwareBackground` as the
    per-feature attributions, so the leak-safety contract is unchanged.

    Because the player set is the (small) set of *groups*, the Shapley formula
    can be evaluated exactly by enumerating all ``2^G`` coalitions -- no
    sampling, no approximation, and no TreeSHAP reimplementation: the model's
    own ``predict`` is the only thing called.

    Parameters
    ----------
    model : object
        A fitted Panelary model wrapper or bare estimator.
    X : PanelFrame | polars.DataFrame | polars.LazyFrame
        The panel to explain.
    groups : mapping of str to sequence of str
        Disjoint feature groups. Each becomes one player.
    background : TimeAwareBackground
        Fitted, fold-bound reference set (the marginal imputer).
    include_ungrouped : bool, default=True
        Add the features not covered by ``groups`` as one extra player, so the
        group values satisfy efficiency (``sum_g phi_g = f(x) - E[f]``).
    max_groups : int, default=12
        Refuse to enumerate more than ``2 ** max_groups`` coalitions.
    max_evals : int, default=20_000_000
        Guard on ``n_rows * 2**G * n_background`` model evaluations.
    chunk_rows : int, default=256
        Rows per prediction batch (memory control).

    Returns
    -------
    polars.DataFrame
        ``(entity, time)``, ``shap_base_value`` and one
        ``shap_group_<name>`` column per player.
    """
    panel = as_panel(X, entity, time)
    native, model_feats = unwrap_model(model)
    feats = resolve_features(
        panel,
        list(features) if features is not None else (model_feats or None),
        who="joint_group_shap",
    )
    if background.is_fitted and background.features_ != feats:
        raise ValueError(
            f"the background covers features {background.features_} but the "
            f"attribution asks for {feats}; refit the background to match."
        )
    resolved = _validate_groups(groups, feats, "joint_group_shap")
    covered = {m for members in resolved.values() for m in members}
    leftover = [f for f in feats if f not in covered]
    if include_ungrouped and leftover:
        resolved[ungrouped_name] = leftover

    players = list(resolved)
    n_players = len(players)
    if n_players > max_groups:
        raise ValueError(
            f"joint_group_shap enumerates 2**{n_players} coalitions, above "
            f"`max_groups={max_groups}`. Coarsen the partition, or use "
            "`group_shap(..., group_mode='additive')`, which is exact for "
            "marginal values and costs nothing."
        )

    col_of = {f: j for j, f in enumerate(feats)}
    masks = np.zeros((n_players, len(feats)), dtype=bool)
    for p, name in enumerate(players):
        for m in resolved[name]:
            masks[p, col_of[m]] = True

    frame = panel.lazy().select([panel.entity_col, panel.time_col, *feats]).collect()
    mat = frame.select(feats).to_numpy().astype(float, copy=False)
    n_rows = mat.shape[0]
    phi = np.full((n_rows, n_players), np.nan, dtype=float)
    base = np.full(n_rows, np.nan, dtype=float)

    from panelary.explain._tree import _group_rows_by_background

    weights = _shapley_weights(n_players)
    coalitions = [
        frozenset(c)
        for size in range(n_players + 1)
        for c in combinations(range(n_players), size)
    ]

    n_empty = 0
    for bg, idx in _group_rows_by_background(
        background, frame.get_column(panel.time_col)
    ):
        rows = np.asarray(idx, dtype=np.int64)
        if bg is None or bg.shape[0] == 0:
            n_empty += rows.size
            continue
        n_bg = bg.shape[0]
        evals = rows.size * len(coalitions) * n_bg
        if evals > max_evals:
            raise ValueError(
                f"joint_group_shap would need {evals:,} model evaluations "
                f"({rows.size} rows x {len(coalitions)} coalitions x {n_bg} "
                f"background rows), above `max_evals={max_evals:,}`. Reduce "
                "`background.max_samples`, coarsen the partition, or explain "
                "fewer rows."
            )
        values = np.empty((rows.size, len(coalitions)), dtype=float)
        for c_i, coal in enumerate(coalitions):
            featmask = np.zeros(len(feats), dtype=bool)
            for p in coal:
                featmask |= masks[p]
            for start in range(0, rows.size, chunk_rows):
                chunk = rows[start : start + chunk_rows]
                xs = mat[chunk]  # (r, F)
                blended = np.where(
                    featmask[None, None, :], xs[:, None, :], bg[None, :, :]
                ).reshape(-1, len(feats))
                preds = raw_predict(native, blended).reshape(chunk.size, n_bg)
                values[start : start + chunk.size, c_i] = preds.mean(axis=1)

        index_of = {coal: i for i, coal in enumerate(coalitions)}
        empty_idx = index_of[frozenset()]
        base[rows] = values[:, empty_idx]
        for p in range(n_players):
            acc = np.zeros(rows.size, dtype=float)
            for coal in coalitions:
                if p in coal:
                    continue
                gain = values[:, index_of[coal | {p}]] - values[:, index_of[coal]]
                acc += weights[len(coal)] * gain
            phi[rows, p] = acc

    background.handle_empty(n_empty)

    keys = frame.select([panel.entity_col, panel.time_col])
    cols = [pl.Series(name=BASE_VALUE_COL, values=base)]
    cols += [
        pl.Series(name=f"{out_prefix}{name}", values=phi[:, p])
        for p, name in enumerate(players)
    ]
    return keys.with_columns(cols)


# --------------------------------------------------------------------------- #
# Per-entity window aggregation
# --------------------------------------------------------------------------- #
def window_shap(
    shap: pl.DataFrame,
    *,
    entity: str,
    time: str,
    window: int | str,
    prefix: str = SHAP_PREFIX,
    agg: str = "sum",
    min_periods: int | None = None,
    columns: Sequence[str] | None = None,
    out_suffix: str | None = None,
) -> pl.DataFrame:
    """Trailing per-entity window aggregation of attributions.

    Each entity is windowed on **its own calendar**: an ``int`` window counts
    that entity's own observations, and a duration string (``"30d"``, ``"3i"``)
    uses that entity's own time values. Rows of other entities never enter a
    window, and no global time grid is constructed, so ragged panels are handled
    correctly by construction.

    Parameters
    ----------
    shap : polars.DataFrame
        Wide attribution frame.
    entity, time : str
        Panel key columns.
    window : int or str
        Number of trailing observations, or a polars duration/period string.
    prefix : str, default="shap_"
        Prefix identifying the attribution columns to aggregate.
    agg : {"sum", "mean", "abs_mean"}, default="sum"
        ``"abs_mean"`` gives the classic "mean |SHAP|" importance profile.
    min_periods : int, optional
        Minimum observations in the window (integer windows only). Defaults to
        ``window``, so partially-filled leading windows are null rather than
        silently short.
    columns : sequence of str, optional
        Restrict to these attribution columns.
    out_suffix : str, optional
        Suffix for the emitted columns; defaults to ``f"_w{window}"``.

    Returns
    -------
    polars.DataFrame
        Sorted by ``(entity, time)``, keys plus the windowed columns.
    """
    if agg not in ("sum", "mean", "abs_mean"):
        raise ValueError(
            f"`agg` must be one of 'sum', 'mean', 'abs_mean', got {agg!r}."
        )
    value_cols = (
        list(columns)
        if columns is not None
        else attribution_columns(shap, prefix, exclude=(entity, time))
    )
    if not value_cols:
        raise ValueError(
            f"window_shap: no attribution columns found with prefix {prefix!r}."
        )
    missing = [c for c in value_cols if c not in shap.columns]
    if missing:
        raise ValueError(f"window_shap: column(s) {missing} not in the frame.")

    suffix = out_suffix if out_suffix is not None else f"_w{window}"
    df = shap.sort([entity, time])

    def _value(c: str) -> pl.Expr:
        col = pl.col(c)
        return col.abs() if agg == "abs_mean" else col

    reducer = "sum" if agg == "sum" else "mean"

    if isinstance(window, str):
        rolled = df.rolling(index_column=time, period=window, group_by=entity).agg(
            [getattr(_value(c), reducer)().alias(f"{c}{suffix}") for c in value_cols]
        )
        return rolled.select([entity, time, *[f"{c}{suffix}" for c in value_cols]])

    w = int(window)
    if w < 1:
        raise ValueError(f"`window` must be a positive integer, got {window!r}.")
    mp = w if min_periods is None else int(min_periods)
    roll = "rolling_sum" if agg == "sum" else "rolling_mean"
    exprs = [
        getattr(_value(c), roll)(window_size=w, min_samples=mp)
        .over(entity)
        .alias(f"{c}{suffix}")
        for c in value_cols
    ]
    return df.select([pl.col(entity), pl.col(time), *exprs])
