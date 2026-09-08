"""Attribution stability: separate reference-induced oscillation from regime drift.

An attribution that moves is not automatically a signal. Two very different
things make a feature's SHAP value wander over a panel:

1. **Reference-induced oscillation.** The background is a *sample*. Redraw it
   (a different seed, a slightly different past window) and the attributions
   move -- especially for collinear financial features, where credit is shared
   almost arbitrarily among near-duplicates. This is noise in the explanation,
   not in the model.
2. **Genuine regime drift.** The model really is leaning on a different feature
   than it did last year.

Reporting a "feature importance changed" story without separating the two is
how attribution narratives get overstated. This module measures both on the
same footing and reports the ratio:

* :func:`background_sensitivity` -- re-draws the reference set under several
  seeds (same fold, same past-only policy, so nothing leaks) and measures the
  spread of the resulting attributions.
* :func:`attribution_drift` -- measures the spread of the cross-sectional mean
  attribution *across time*.
* :func:`attribution_stability` -- puts them side by side and returns a verdict
  per feature.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from polars_features.core.panel_frame import PanelFrame, as_panel
from polars_features.explain._common import SHAP_PREFIX, attribution_columns

__all__ = [
    "attribution_drift",
    "attribution_stability",
    "background_sensitivity",
]


def _attr_columns(df: pl.DataFrame, prefix: str, entity: str, time: str) -> list[str]:
    cols = attribution_columns(df, prefix, exclude=(entity, time))
    if not cols:
        raise ValueError(
            f"no attribution columns found with prefix {prefix!r}; got {df.columns}."
        )
    return cols


def background_sensitivity(
    attributor: Any,
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    seeds: Sequence[int] = (0, 1, 2, 3),
    entity: str | None = None,
    time: str | None = None,
) -> pl.DataFrame:
    """Spread of attributions across independent draws of the reference set.

    The attributor is re-fitted on **its own fit panel** with a different
    background seed for each draw, so every draw is still fold-bound and
    past-only: this measures reference *sampling* noise, never leakage.

    Parameters
    ----------
    attributor : TreeAttributor
        A fitted attributor (its fit panel is reused).
    X : panel
        The rows to explain.
    seeds : sequence of int, default=(0, 1, 2, 3)
        Background seeds to draw.

    Returns
    -------
    polars.DataFrame
        ``feature``, ``reference_sd`` (mean over rows of the per-row standard
        deviation across seeds) and ``reference_sd_of_mean`` (standard deviation
        across seeds of the feature's mean attribution).
    """
    from copy import copy

    if not getattr(attributor, "is_fitted", False):
        raise RuntimeError(
            "background_sensitivity needs a fitted TreeAttributor (its fit panel "
            "defines the fold the reference set is bound to)."
        )
    fit_panel = getattr(attributor, "_fit_panel", None)
    if fit_panel is None:  # pragma: no cover - defensive
        raise RuntimeError("the attributor did not record its fit panel.")
    panel = as_panel(X, entity, time)
    seeds = list(seeds)
    if len(seeds) < 2:
        raise ValueError("`seeds` must contain at least two distinct seeds.")
    bg_mode = getattr(getattr(attributor, "background_", None), "mode", None)
    if bg_mode is not None and bg_mode != "interventional":
        warnings.warn(
            f"background_sensitivity on a {bg_mode!r} attributor: the "
            "path-dependent tree kernel ignores the external reference set, so "
            "the measured reference sensitivity is structurally zero and the "
            "resulting drift verdicts are not meaningful. Re-run with "
            "mode='interventional' to diagnose reference-induced oscillation.",
            UserWarning,
            stacklevel=2,
        )

    stacks: list[np.ndarray] = []
    feats: list[str] = list(attributor.features_)
    for s in seeds:
        clone = copy(attributor)
        clone.seed = int(s)
        if hasattr(clone.background, "seed"):
            bg = clone._new_background()
            bg.seed = int(s)
            clone.background = bg
        clone._fitted = False
        clone.fit(fit_panel)
        wide = clone.attributions(panel)
        cols = [f"{attributor.prefix}{f}" for f in feats]
        stacks.append(wide.select(cols).to_numpy().astype(float, copy=False))

    cube = np.stack(stacks, axis=0)  # (n_seeds, n_rows, n_features)
    with np.errstate(invalid="ignore"):
        per_row_sd = np.nanstd(cube, axis=0, ddof=1)
        reference_sd = np.nanmean(per_row_sd, axis=0)
        mean_per_seed = np.nanmean(cube, axis=1)
        sd_of_mean = np.nanstd(mean_per_seed, axis=0, ddof=1)
    return pl.DataFrame(
        {
            "feature": feats,
            "reference_sd": np.nan_to_num(reference_sd, nan=0.0),
            "reference_sd_of_mean": np.nan_to_num(sd_of_mean, nan=0.0),
        }
    )


def attribution_drift(
    shap: pl.DataFrame,
    *,
    entity: str,
    time: str,
    prefix: str = SHAP_PREFIX,
) -> pl.DataFrame:
    """Spread of the cross-sectional mean attribution **across time**.

    Parameters
    ----------
    shap : polars.DataFrame
        Wide attribution frame keyed by ``(entity, time)``.
    entity, time : str
        Panel key columns.
    prefix : str, default="shap_"
        Attribution column prefix.

    Returns
    -------
    polars.DataFrame
        ``feature``, ``mean_abs_attribution``, ``temporal_sd`` and
        ``first_last_shift`` (mean attribution in the last third of the sample
        minus the first third -- a crude but readable regime indicator).
    """
    cols = _attr_columns(shap, prefix, entity, time)
    per_time = (
        shap.group_by(time).agg([pl.col(c).mean().alias(c) for c in cols]).sort(time)
    )
    values = per_time.select(cols).to_numpy().astype(float, copy=False)
    n = values.shape[0]
    cut = max(1, n // 3)
    with np.errstate(invalid="ignore"):
        temporal_sd = (
            np.nanstd(values, axis=0, ddof=1) if n > 1 else np.zeros(len(cols))
        )
        shift = np.nanmean(values[-cut:], axis=0) - np.nanmean(values[:cut], axis=0)
        mean_abs = np.nanmean(
            np.abs(shap.select(cols).to_numpy().astype(float, copy=False)), axis=0
        )
    return pl.DataFrame(
        {
            "feature": [c[len(prefix) :] for c in cols],
            "mean_abs_attribution": np.nan_to_num(mean_abs, nan=0.0),
            "temporal_sd": np.nan_to_num(temporal_sd, nan=0.0),
            "first_last_shift": np.nan_to_num(shift, nan=0.0),
        }
    )


def attribution_stability(
    attributor: Any,
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    seeds: Sequence[int] = (0, 1, 2, 3),
    threshold: float = 2.0,
    entity: str | None = None,
    time: str | None = None,
) -> pl.DataFrame:
    """Per-feature stability report: reference noise vs regime drift.

    Parameters
    ----------
    attributor : TreeAttributor
        A fitted attributor.
    X : panel
        Rows to explain.
    seeds : sequence of int
        Background seeds used to estimate reference-induced oscillation.
    threshold : float, default=2.0
        ``drift_ratio = temporal_sd / reference_sd`` above which the movement is
        called regime drift rather than reference noise.

    Returns
    -------
    polars.DataFrame
        ``feature``, ``mean_abs_attribution``, ``reference_sd``, ``temporal_sd``,
        ``drift_ratio``, ``verdict``.

    Notes
    -----
    ``verdict`` is one of ``"stable"`` (the feature barely moves at all),
    ``"regime-drift"`` (moves across time far more than across reference draws)
    and ``"reference-noise"`` (moves no more than redrawing the background
    does -- do **not** tell a story about it).

    The diagnostic is only meaningful for ``mode="interventional"``. The
    path-dependent tree kernel used by ``conditional`` / ``path_dependent``
    ignores the external reference set, so ``reference_sd`` is structurally
    zero, ``drift_ratio`` is infinite, and every feature reads as drift;
    :func:`background_sensitivity` warns in that case.
    """
    panel = as_panel(X, entity, time)
    ref = background_sensitivity(attributor, panel, seeds=seeds)
    wide = attributor.attributions(panel)
    drift = attribution_drift(
        wide, entity=panel.entity_col, time=panel.time_col, prefix=attributor.prefix
    )
    joined = drift.join(ref, on="feature", how="left")
    eps = 1e-12
    scale = joined.get_column("mean_abs_attribution").to_numpy()
    temporal = joined.get_column("temporal_sd").to_numpy()
    reference = joined.get_column("reference_sd").to_numpy()
    tiny = temporal <= 1e-8 * np.maximum(np.nanmax(scale), eps)
    # A zero reference spread means the value function ignores the reference set
    # (the path-dependent kernel) -- report an honest infinity rather than a
    # number manufactured by an epsilon.
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(
            tiny,
            0.0,
            np.where(reference > eps, temporal / np.maximum(reference, eps), np.inf),
        )
    verdict = np.where(
        tiny,
        "stable",
        np.where(ratio >= float(threshold), "regime-drift", "reference-noise"),
    )
    return joined.with_columns(
        [
            pl.Series(name="drift_ratio", values=ratio),
            pl.Series(name="verdict", values=verdict),
        ]
    ).select(
        [
            "feature",
            "mean_abs_attribution",
            "reference_sd",
            "temporal_sd",
            "drift_ratio",
            "verdict",
        ]
    )
