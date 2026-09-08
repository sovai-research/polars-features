"""Microstructure liquidity features: Amihud illiquidity and the Roll spread.

Both are trailing rolling statistics computed per entity with native Polars
rolling expressions, so they are cheap and causal: row ``t`` uses only rows
``t - window + 1 .. t`` of the same entity.

* **Amihud (2002) illiquidity** -- ``mean_t(|r| / dollar_volume)``, the average
  price impact per unit of traded value. Usually reported scaled by ``1e6``.
* **Roll (1984) effective spread** -- ``2 * sqrt(-Cov(dp_t, dp_{t-1}))`` when
  that autocovariance is negative (bid-ask bounce); undefined otherwise, which
  is reported as null or, optionally, floored at zero.
* **Amivest liquidity** -- ``mean_t(volume / |r|)``, the reciprocal-flavoured
  companion to Amihud.
* **Turnover** -- ``mean_t(volume / shares_outstanding)``.

References
----------
Amihud (2002), *Journal of Financial Markets*; Roll (1984), *Journal of
Finance*; Goyenko, Holden & Trzcinka (2009) for the low-frequency-proxy survey.
"""

from __future__ import annotations

import polars as pl

from polars_features.econ.features._common import sorted_panel

__all__ = [
    "amihud_illiquidity",
    "roll_spread",
    "amivest_liquidity",
    "liquidity_features",
]

#: Conventional Amihud scaling (returns per million units of traded value).
AMIHUD_SCALE: float = 1e6


def amihud_illiquidity(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    returns: str,
    dollar_volume: str,
    window: int = 21,
    min_periods: int | None = None,
    scale: float = AMIHUD_SCALE,
    alias: str | None = None,
) -> pl.DataFrame:
    """Trailing Amihud illiquidity ratio per entity.

    ``ILLIQ_t = scale * mean(|r_s| / dollar_volume_s)`` over the trailing
    ``window``. Rows with non-positive traded value contribute nothing (they are
    treated as missing) rather than producing an infinite ratio.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel.
    entity, time : str
        Panel keys.
    returns : str
        Period return column.
    dollar_volume : str
        Traded value (price times volume) column.
    window : int, default=21
        Trailing window length.
    min_periods : int, optional
        Minimum observations in the window (defaults to ``window``).
    scale : float, default=1e6
        Multiplicative scaling of the raw ratio.
    alias : str, optional
        Output column name (defaults to ``"amihud_{window}"``).

    Returns
    -------
    polars.DataFrame
        ``df`` sorted by ``(entity, time)`` with the feature column appended.
    """
    frame = sorted_panel(df, entity, time)
    mp = window if min_periods is None else int(min_periods)
    name = alias or f"amihud_{window}"
    ratio = (
        pl.when(pl.col(dollar_volume) > 0)
        .then(pl.col(returns).abs() / pl.col(dollar_volume))
        .otherwise(None)
    )
    return frame.with_columns(
        (scale * ratio.rolling_mean(window, min_samples=mp).over(entity)).alias(name)
    )


def roll_spread(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    price: str | None = None,
    price_change: str | None = None,
    window: int = 21,
    min_periods: int | None = None,
    clip_positive: bool = False,
    alias: str | None = None,
) -> pl.DataFrame:
    """Trailing Roll (1984) effective-spread estimator per entity.

    ``S_t = 2 * sqrt(-Cov(dp_s, dp_{s-1}))`` over the trailing window, where
    ``dp`` is the price change. The estimator is only defined when the
    autocovariance is negative (the bid-ask-bounce case); positive
    autocovariance yields null, or ``0.0`` with ``clip_positive=True``.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel.
    entity, time : str
        Panel keys.
    price : str, optional
        Price column, differenced internally. Mutually exclusive with
        ``price_change``.
    price_change : str, optional
        Pre-computed price change column.
    window : int, default=21
        Trailing window length.
    min_periods : int, optional
        Minimum observations (defaults to ``window``).
    clip_positive : bool, default=False
        Emit ``0.0`` instead of null where the autocovariance is non-negative.
    alias : str, optional
        Output column name (defaults to ``"roll_spread_{window}"``).

    Returns
    -------
    polars.DataFrame
    """
    if (price is None) == (price_change is None):
        raise ValueError("pass exactly one of `price` or `price_change`.")
    frame = sorted_panel(df, entity, time)
    mp = window if min_periods is None else int(min_periods)
    name = alias or f"roll_spread_{window}"
    dp = (
        pl.col(price).diff().over(entity) if price is not None else pl.col(price_change)
    )
    dp_lag = dp.shift(1).over(entity)
    cov = (
        (dp * dp_lag).rolling_mean(window, min_samples=mp)
        - dp.rolling_mean(window, min_samples=mp)
        * dp_lag.rolling_mean(window, min_samples=mp)
    ).over(entity)
    spread = 2.0 * (-cov).sqrt()
    fallback = pl.lit(0.0) if clip_positive else pl.lit(None, dtype=pl.Float64)
    return frame.with_columns(
        pl.when(cov < 0).then(spread).otherwise(fallback).alias(name)
    )


def amivest_liquidity(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    returns: str,
    volume: str,
    window: int = 21,
    min_periods: int | None = None,
    alias: str | None = None,
) -> pl.DataFrame:
    """Trailing Amivest liquidity ratio ``mean(volume / |r|)`` per entity.

    Rows with a zero return are treated as missing (the ratio is undefined).
    """
    frame = sorted_panel(df, entity, time)
    mp = window if min_periods is None else int(min_periods)
    name = alias or f"amivest_{window}"
    ratio = (
        pl.when(pl.col(returns).abs() > 0)
        .then(pl.col(volume) / pl.col(returns).abs())
        .otherwise(None)
    )
    return frame.with_columns(
        ratio.rolling_mean(window, min_samples=mp).over(entity).alias(name)
    )


def liquidity_features(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    returns: str,
    price: str | None = None,
    dollar_volume: str | None = None,
    volume: str | None = None,
    shares_outstanding: str | None = None,
    windows: tuple[int, ...] = (21, 63),
    min_periods: int | None = None,
) -> pl.DataFrame:
    """Compute the whole liquidity block over one or more trailing windows.

    Emits whichever measures the supplied columns allow: Amihud (needs
    ``dollar_volume``), Roll (needs ``price``), Amivest (needs ``volume``) and
    turnover (needs ``volume`` and ``shares_outstanding``), for each window in
    ``windows``.

    Returns
    -------
    polars.DataFrame
        ``df`` sorted by ``(entity, time)`` with the feature columns appended.
    """
    out = sorted_panel(df, entity, time)
    for window in windows:
        mp = window if min_periods is None else int(min_periods)
        if dollar_volume is not None:
            out = amihud_illiquidity(
                out,
                entity=entity,
                time=time,
                returns=returns,
                dollar_volume=dollar_volume,
                window=window,
                min_periods=mp,
            )
        if price is not None:
            out = roll_spread(
                out,
                entity=entity,
                time=time,
                price=price,
                window=window,
                min_periods=mp,
            )
        if volume is not None:
            out = amivest_liquidity(
                out,
                entity=entity,
                time=time,
                returns=returns,
                volume=volume,
                window=window,
                min_periods=mp,
            )
            if shares_outstanding is not None:
                out = out.with_columns(
                    (
                        pl.when(pl.col(shares_outstanding) > 0)
                        .then(pl.col(volume) / pl.col(shares_outstanding))
                        .otherwise(None)
                    )
                    .rolling_mean(window, min_samples=mp)
                    .over(entity)
                    .alias(f"turnover_{window}")
                )
        # Realized volatility of returns is the natural companion control.
        out = out.with_columns(
            pl.col(returns)
            .rolling_std(window, min_samples=mp)
            .over(entity)
            .alias(f"ret_vol_{window}")
        )
    return out
