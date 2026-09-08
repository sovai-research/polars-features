"""Information Coefficient (IC) — per-date signal/forward-return correlation.

The IC is the field-default measure of signal quality: the cross-sectional
correlation between a signal at ``t`` and its forward return, computed **per
date** and never pooled. :func:`ic` returns the per-date IC time series;
:func:`ic_summary` reduces it to the standard five-number summary (mean IC, its
volatility, the information ratio ICIR, a t-statistic, and the hit rate).

Leak-safety is inherited: the forward-return column must have been produced by
:func:`polars_features.factor.forward_return` (a backward shift), and every
correlation is grouped ``.over(time)`` so no cross-date information mixes.
"""

from __future__ import annotations

import math

import polars as pl

__all__ = ["ic", "ic_summary"]

FrameT = pl.LazyFrame | pl.DataFrame

_VALID_METHODS = ("spearman", "pearson")


def ic(
    frame: FrameT,
    *,
    signal: str,
    forward_return: str,
    time: str,
    method: str = "spearman",
) -> pl.DataFrame:
    """Per-date information coefficient of ``signal`` vs ``forward_return``.

    Groups the panel by ``time`` and computes, within each date's cross-section,
    the correlation between the signal and the forward return. ``"spearman"``
    (default) is the rank-IC; ``"pearson"`` is the linear IC. Rows with a null in
    either column (e.g. the horizon-edge nulls from
    :func:`~polars_features.factor.forward_return`) are dropped before
    correlating.

    Parameters
    ----------
    frame : LazyFrame | DataFrame
        A frame already carrying an aligned forward-return column.
    signal, forward_return, time : str, keyword-only
        Column names for the signal, the (leak-safe) forward return, and the
        date/time cross-section key.
    method : str, keyword-only, default "spearman"
        ``"spearman"`` (rank-IC) or ``"pearson"``.

    Returns
    -------
    pl.DataFrame
        Two columns, ``[time, "ic"]``, one row per date, sorted by ``time``.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"ic: `method` must be one of {sorted(_VALID_METHODS)}, got {method!r}."
        )
    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    return (
        lf.drop_nulls(subset=[signal, forward_return])
        .group_by(time)
        .agg(
            pl.corr(signal, forward_return, method=method).alias("ic")  # type: ignore[arg-type]
        )
        .sort(time)
        .collect()
    )


def ic_summary(ic_frame: pl.DataFrame, *, ic_col: str = "ic") -> dict[str, float]:
    """Reduce a per-date IC series to the standard summary statistics.

    Parameters
    ----------
    ic_frame : pl.DataFrame
        The per-date IC frame produced by :func:`ic`.
    ic_col : str, keyword-only, default "ic"
        Name of the IC column in ``ic_frame``.

    Returns
    -------
    dict[str, float]
        ``mean_ic`` (average IC), ``ic_std`` (sample std, ddof=1),
        ``icir`` (``mean_ic / ic_std`` — the information ratio),
        ``t_stat`` (``icir * sqrt(T)``), ``hit_rate`` (fraction of dates with a
        positive IC), and ``n_periods`` (number of non-null dates ``T``).
    """
    series = ic_frame.get_column(ic_col).drop_nulls()
    n = series.len()
    if n == 0:
        return {
            "mean_ic": float("nan"),
            "ic_std": float("nan"),
            "icir": float("nan"),
            "t_stat": float("nan"),
            "hit_rate": float("nan"),
            "n_periods": 0.0,
        }
    mean_ic = float(series.mean())
    ic_std = float(series.std(ddof=1)) if n > 1 else float("nan")
    icir = mean_ic / ic_std if ic_std and math.isfinite(ic_std) else float("nan")
    t_stat = icir * math.sqrt(n) if math.isfinite(icir) else float("nan")
    hit_rate = float((series > 0).sum()) / n
    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "icir": icir,
        "t_stat": t_stat,
        "hit_rate": hit_rate,
        "n_periods": float(n),
    }
