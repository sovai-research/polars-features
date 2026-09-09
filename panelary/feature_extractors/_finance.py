"""Finance-flavoured summary statistics over a price/return series."""

from __future__ import annotations

import numpy as np
import polars as pl

from panelary._internal import _numpy_stats


def realized_volatility(x: pl.Series) -> float:
    """
    Realized Volatility

    Computes the square root of the sum of squared log returns.

    Parameters
    ----------
    x : pl.Series
        Price series.

    Returns
    -------
    float
        Realized volatility (non-annualized).
    """
    log_returns = np.diff(np.log(x.to_numpy()))
    return np.sqrt(np.sum(log_returns**2))


def return_skew(x: pl.Series) -> float:
    """
    Skewness of simple (pct_change) Returns

    Parameters
    ----------
    x : pl.Series
        Price series.

    Returns
    -------
    float
        Skewness of the simple (pct_change) returns.
    """
    # Use simple percentage returns to stay consistent with the
    # ``.ts.return_skew`` namespace method (``pct_change().skew()``).
    prices = x.to_numpy()
    returns = np.diff(prices) / prices[:-1]
    return _numpy_stats.skew(returns)


def return_kurtosis(x: pl.Series) -> float:
    """
    Kurtosis of simple (pct_change) Returns

    Parameters
    ----------
    x : pl.Series
        Price series.

    Returns
    -------
    float
        Kurtosis of the simple (pct_change) returns.
    """
    # Use simple percentage returns to stay consistent with the
    # ``.ts.return_kurtosis`` namespace method (``pct_change().kurtosis()``).
    prices = x.to_numpy()
    returns = np.diff(prices) / prices[:-1]
    return _numpy_stats.kurtosis(returns)


def num_direction_changes(x: pl.Series) -> int:
    """
    Number of Direction Changes

    Counts how often the price trend switches direction.

    Parameters
    ----------
    x : pl.Series
        Price series.

    Returns
    -------
    int
        Number of sign changes in first-order price differences.
    """
    returns = np.diff(x.to_numpy())
    return np.sum(np.diff(np.sign(returns)) != 0)


def max_drawdown(x: pl.Series) -> float:
    """
    Maximum Drawdown

    Computes the largest peak-to-trough drop in the time series as a *ratio*
    (a non-positive number), matching the ``.ts.max_drawdown`` namespace method:

        ``(x / cummax(x)).min() - 1``

    For example a series that peaks at 100 and troughs at 75 has a max drawdown
    of ``75 / 100 - 1 == -0.25``.

    Parameters
    ----------
    x : pl.Series
        Price series.

    Returns
    -------
    float
        Maximum drawdown as a ratio in ``[-1, 0]`` (0.0 if never in drawdown).
    """
    prices = x.to_numpy()
    cumulative_max = np.maximum.accumulate(prices)
    return float(np.min(prices / cumulative_max) - 1.0)


def signed_mci(
    trade_price: pl.Expr,
    bid_price: pl.Expr,
    ask_price: pl.Expr,
    side: pl.Expr,
) -> pl.Expr:
    """
    Compute signed Marginal Cost of Immediacy (MCI) reflecting cost paid by aggressive orders:

    Signed MCI = side * (trade_price - mid_price)

    where mid_price = (bid_price + ask_price) / 2
    and side = +1 for buyer-initiated trades, -1 for seller-initiated trades.

    Parameters
    ----------
    trade_price : Expr
        The actual trade price.
    bid_price : Expr
        Best bid price at trade time.
    ask_price : Expr
        Best ask price at trade time.
    side : Expr
        Trade side indicator (+1 buy, -1 sell).

    Returns
    -------
    Expr : expression producing signed MCI values
    """
    mid_price = (bid_price + ask_price) / 2
    return side * (trade_price - mid_price)
