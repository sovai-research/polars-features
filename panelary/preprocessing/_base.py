"""Shared plumbing for the ``panelary.preprocessing`` transformers.

Holds the two names every other module in the package needs: the numeric-column
selector and the look-ahead warning class. Deliberately dependency-free beyond
polars, so importing it costs nothing.
"""

from __future__ import annotations

import polars.selectors as cs


def PL_NUMERIC_COLS(*exclude):
    return cs.numeric() - cs.by_name(exclude)


class LeakageWarning(UserWarning):
    """Warning that an operation reads the future (look-ahead / target leakage).

    Emitted by leak-prone code paths that are easy to use unsafely inside a
    backtest or cross-validation split -- for example :func:`impute` with the
    ``"bfill"`` or ``"interpolate"`` methods, which fill a missing value using
    later (future) observations. Such fills pass silently through CV yet inflate
    out-of-sample performance. Prefer a strictly point-in-time, leak-safe imputer
    (e.g. ``impute("cafe")`` / :func:`cafe_impute`, or forward-fill) unless you
    have deliberately opted in via ``allow_leaky=True``.
    """


#: Imputation methods that read the future and therefore leak inside a CV split.
_LEAKY_IMPUTE_METHODS: frozenset[str] = frozenset({"bfill", "interpolate"})
