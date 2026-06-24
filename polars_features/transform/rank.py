"""Cross-sectional ranking transformer.

:class:`CrossSectionalRank` ranks a column **across entities within each date**
(``expr.rank().over(time_col)``). This is the panel-native
``.xs.rank().over("date")`` capability exposed as a
:class:`~polars_features.core.protocol.PanelTransformer`.

Because the rank of each row is computed using only values observed at *the
same timestamp*, the transform is leak-safe by construction: no past or future
date participates, and there is no fitted state to carry across a train/test
boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import polars as pl

from polars_features.core.panel_frame import PanelFrame
from polars_features.core.protocol import PanelTransformer

if TYPE_CHECKING:
    pass

__all__ = ["CrossSectionalRank"]

_NORMALIZE = ("none", "uniform", "gaussian")
NormalizeMode = Literal["none", "uniform", "gaussian"]

# Standard-normal inverse-CDF approximation (Acklam) constant guard.
_EPS = 1e-12


def _as_column_list(columns: str | Sequence[str] | None) -> list[str] | None:
    if columns is None:
        return None
    if isinstance(columns, str):
        return [columns]
    cols = list(columns)
    if not cols:
        raise ValueError(
            "`columns` was an empty sequence; pass None to rank all features."
        )
    return cols


def _check_columns_exist(
    panel: PanelFrame, columns: Sequence[str], *, where: str
) -> None:
    available = set(panel.columns)
    missing = [c for c in columns if c not in available]
    if missing:
        raise ValueError(
            f"{where}: column(s) {missing} not found in panel. "
            f"Available columns: {panel.columns}."
        )


def _gaussian_inverse_cdf(p: pl.Expr) -> pl.Expr:
    """Approximate the inverse standard-normal CDF (probit) for ``p`` in (0, 1).

    Uses the Beasley-Springer / Moro rational approximation, expressed as pure
    Polars arithmetic so no Python callback is needed. Accuracy is ~1e-9 in the
    central region, which is more than enough for a Gaussian-rank feature.
    """
    # clamp away from 0/1 to avoid +/- inf
    p = (
        pl.when(p < _EPS)
        .then(pl.lit(_EPS))
        .otherwise(pl.when(p > 1 - _EPS).then(pl.lit(1 - _EPS)).otherwise(p))
    )

    # Coefficients (Acklam's algorithm).
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    # central region
    q_c = p - 0.5
    r_c = q_c * q_c
    num_c = (
        ((((a[0] * r_c + a[1]) * r_c + a[2]) * r_c + a[3]) * r_c + a[4]) * r_c + a[5]
    ) * q_c
    den_c = ((((b[0] * r_c + b[1]) * r_c + b[2]) * r_c + b[3]) * r_c + b[4]) * r_c + 1.0
    central = num_c / den_c

    # lower tail
    q_l = (-2.0 * p.log()).sqrt()
    lower = (
        ((((c[0] * q_l + c[1]) * q_l + c[2]) * q_l + c[3]) * q_l + c[4]) * q_l + c[5]
    ) / ((((d[0] * q_l + d[1]) * q_l + d[2]) * q_l + d[3]) * q_l + 1.0)

    # upper tail
    q_u = (-2.0 * (1.0 - p).log()).sqrt()
    upper = -(
        ((((c[0] * q_u + c[1]) * q_u + c[2]) * q_u + c[3]) * q_u + c[4]) * q_u + c[5]
    ) / ((((d[0] * q_u + d[1]) * q_u + d[2]) * q_u + d[3]) * q_u + 1.0)

    return (
        pl.when(p < p_low).then(lower).when(p > p_high).then(upper).otherwise(central)
    )


class CrossSectionalRank(PanelTransformer):
    """Rank column(s) across entities within each timestamp.

    For each date, values are ranked over all entities observed at that date
    (``expr.rank().over(time_col)``), then optionally normalized.

    Parameters
    ----------
    columns : str or sequence of str, optional
        Columns to rank. ``None`` (default) ranks every feature column.
    method : {"average", "min", "max", "dense", "ordinal", "random"}, default="average"
        Tie-breaking rule, forwarded to :meth:`polars.Expr.rank`.
    descending : bool, default=False
        If True, the largest value gets rank 1.
    normalize : {"none", "uniform", "gaussian"}, default="uniform"
        Post-rank transform applied per date.

        * ``"none"`` : raw integer/average ranks.
        * ``"uniform"`` : map ranks to ``(0, 1)`` via ``(rank - 0.5) / n`` where
          ``n`` is the cross-section size that date (a leak-free quantile proxy).
        * ``"gaussian"`` : feed the uniform scores through the inverse normal CDF
          to obtain an approximately standard-normal "Gaussian rank".
    suffix : str, default="_rank"
        Suffix for the output columns. Set to ``""`` to overwrite in place.

    Attributes
    ----------
    panel_safe : bool
        Always ``True``.
    leakage_safe : bool
        Always ``True`` -- each rank uses only its own same-date cross-section.

    Notes
    -----
    **Leakage guarantee.** Ranking is partitioned by ``time_col`` only; no
    cross-date information enters any row. :meth:`fit` is a stateless no-op kept
    for protocol symmetry.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_features.core import PanelFrame
    >>> df = pl.DataFrame(
    ...     {"id": ["a", "b", "c", "a", "b", "c"],
    ...      "t": [1, 1, 1, 2, 2, 2],
    ...      "x": [3.0, 1.0, 2.0, 9.0, 7.0, 8.0]}
    ... )
    >>> panel = PanelFrame(df, entity="id", time="t")
    >>> out = CrossSectionalRank(normalize="uniform").fit_transform(panel)
    >>> out.collect().shape
    (6, 4)
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        columns: str | Sequence[str] | None = None,
        *,
        method: str = "average",
        descending: bool = False,
        normalize: NormalizeMode = "uniform",
        suffix: str = "_rank",
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if normalize not in _NORMALIZE:
            raise ValueError(
                f"`normalize` must be one of {_NORMALIZE}, got {normalize!r}."
            )
        valid_methods = ("average", "min", "max", "dense", "ordinal", "random")
        if method not in valid_methods:
            raise ValueError(
                f"`method` must be one of {valid_methods}, got {method!r}."
            )
        self.columns = _as_column_list(columns)
        self.method = method
        self.descending = bool(descending)
        self.normalize: NormalizeMode = normalize
        self.suffix = suffix
        self._ranked_cols_: list[str] = []

    def _resolve_columns(self, panel: PanelFrame) -> list[str]:
        schema = panel.schema
        if self.columns is None:
            cols = [c for c in panel.feature_cols if schema[c].is_numeric()]
            if not cols:
                raise ValueError(
                    "CrossSectionalRank: panel has no numeric feature columns to "
                    f"rank (columns={panel.columns}). Pass `columns=` explicitly."
                )
            return cols
        _check_columns_exist(panel, self.columns, where="CrossSectionalRank.fit")
        bad = [c for c in self.columns if not schema[c].is_numeric()]
        if bad:
            raise ValueError(
                f"CrossSectionalRank.fit: column(s) {bad} are not numeric and "
                "cannot be ranked. Pass only numeric columns via `columns=`."
            )
        return self.columns

    def _fit(self, panel: PanelFrame) -> None:
        self._ranked_cols_ = self._resolve_columns(panel)

    def _rank_expr(self, col: str, time_col: str) -> pl.Expr:
        rank = (
            pl.col(col)
            .rank(method=self.method, descending=self.descending)
            .over(time_col)
        )
        if self.normalize == "none":
            return rank
        # n = non-null count in this date's cross-section
        n = pl.col(col).count().over(time_col)
        uniform = (rank - 0.5) / n
        if self.normalize == "uniform":
            return uniform
        return _gaussian_inverse_cdf(uniform)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        _check_columns_exist(
            panel, self._ranked_cols_, where="CrossSectionalRank.transform"
        )
        time_col = panel.time_col
        out_exprs = [
            self._rank_expr(col, time_col).alias(f"{col}{self.suffix}")
            for col in self._ranked_cols_
        ]
        lf = panel.lazy().with_columns(out_exprs)
        return PanelFrame(
            lf, entity=panel.entity_col, time=panel.time_col, validate=False
        )
