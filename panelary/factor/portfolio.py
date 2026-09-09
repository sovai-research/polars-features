"""Portfolio sorts — per-date quantile buckets and the long-short spread.

The characteristic sort is the canonical factor backtest: each date, bucket the
cross-section into ``q`` equal-count quantiles of the signal, then track the
forward return of each bucket. The top-minus-bottom (long-short) spread is the
factor's raw premium, and a monotone increase in mean return across buckets is
the sign of a well-behaved signal.

Every bucket cut is computed **per date** (reusing the leak-safe
:func:`~panelary.namespaces.xs._expr_quantile_bin` under ``.over(time)``),
so a later date's cross-section can never move an earlier date's buckets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from panelary.namespaces.xs import _expr_quantile_bin

__all__ = ["portfolio_sort", "SortResult"]

FrameT = pl.LazyFrame | pl.DataFrame


@dataclass(frozen=True)
class SortResult:
    """Result of a :func:`portfolio_sort`.

    Attributes
    ----------
    long_short : pl.DataFrame
        Per-date long-short spread, columns ``[time, "spread"]``.
    bucket_means : pl.DataFrame
        Equal-weight-across-dates mean forward return per bucket, columns
        ``["bucket", "mean_ret"]`` (bucket ``0`` = lowest signal).
    mean_spread : float
        Time-series mean of the long-short spread.
    t_stat : float
        ``mean_spread / (std / sqrt(T))`` of the spread series (ddof=1).
    monotonicity : float
        Correlation between bucket index and per-bucket mean return in
        ``[-1, 1]``; ``+1`` for a perfectly increasing sort.
    n_periods : int
        Number of dates with a defined spread.
    q : int
        Number of buckets used.
    """

    long_short: pl.DataFrame
    bucket_means: pl.DataFrame
    mean_spread: float
    t_stat: float
    monotonicity: float
    n_periods: int
    q: int

    def summary(self) -> dict[str, float]:
        """Return the scalar summary as a plain dict."""
        return {
            "mean_spread": self.mean_spread,
            "t_stat": self.t_stat,
            "monotonicity": self.monotonicity,
            "n_periods": float(self.n_periods),
            "q": float(self.q),
        }


def portfolio_sort(
    frame: FrameT,
    *,
    signal: str,
    forward_return: str,
    time: str,
    q: int = 5,
) -> SortResult:
    """Sort the cross-section into ``q`` signal buckets and measure the spread.

    Parameters
    ----------
    frame : LazyFrame | DataFrame
        A frame carrying an aligned forward-return column.
    signal, forward_return, time : str, keyword-only
        Column names for the signal, the (leak-safe) forward return, and the
        date/time cross-section key.
    q : int, keyword-only, default 5
        Number of equal-count quantile buckets (integer ``>= 2``).

    Returns
    -------
    SortResult
        The per-date long-short spread, per-bucket mean returns, and the scalar
        summary (mean spread, t-stat, monotonicity).
    """
    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    bucket = _expr_quantile_bin(pl.col(signal), q).over(time).alias("__bucket__")
    per_bucket = (
        lf.drop_nulls(subset=[signal, forward_return])
        .with_columns(bucket)
        .group_by([time, "__bucket__"])
        .agg(pl.col(forward_return).mean().alias("__mean__"))
        .collect()
    )

    # Per-date long-short spread: top bucket (q-1) minus bottom bucket (0).
    top = per_bucket.filter(pl.col("__bucket__") == q - 1).select(
        time, pl.col("__mean__").alias("__top__")
    )
    bot = per_bucket.filter(pl.col("__bucket__") == 0).select(
        time, pl.col("__mean__").alias("__bot__")
    )
    long_short = (
        top.join(bot, on=time, how="inner")
        .with_columns((pl.col("__top__") - pl.col("__bot__")).alias("spread"))
        .select(time, "spread")
        .sort(time)
    )

    spread = long_short.get_column("spread").drop_nulls()
    n = spread.len()
    mean_spread = float(spread.mean()) if n else float("nan")
    if n > 1:
        std = float(spread.std(ddof=1))
        t_stat = mean_spread / (std / math.sqrt(n)) if std else float("nan")
    else:
        t_stat = float("nan")

    # Equal-weight-across-dates mean return per bucket (average of per-date means).
    bucket_means = (
        per_bucket.group_by("__bucket__")
        .agg(pl.col("__mean__").mean().alias("mean_ret"))
        .sort("__bucket__")
        .rename({"__bucket__": "bucket"})
    )

    # Monotonicity: correlation between bucket index and mean return.
    idx = bucket_means.get_column("bucket").to_numpy().astype(float)
    rets = bucket_means.get_column("mean_ret").to_numpy().astype(float)
    if idx.size > 1 and np.std(idx) > 0 and np.std(rets) > 0:
        monotonicity = float(np.corrcoef(idx, rets)[0, 1])
    else:
        monotonicity = float("nan")

    return SortResult(
        long_short=long_short,
        bucket_means=bucket_means,
        mean_spread=mean_spread,
        t_stat=t_stat,
        monotonicity=monotonicity,
        n_periods=n,
        q=q,
    )
