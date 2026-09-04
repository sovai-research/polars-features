"""Statistical (PCA) factors from a returns panel.

:class:`StatisticalFactors` extracts ``k`` PCA statistical factors from a returns
panel. Internally the long panel is pivoted to a wide *time x entity* returns
matrix; the rotation (loadings :math:`\\beta`, one weight per entity per factor)
is fit on the **training dates only** and sign-fixed, and any date's
cross-section is projected onto those fixed loadings to yield that date's factor
values. The date-level factors are broadcast back onto every ``(entity, time)``
row, so they drop in as ordinary panel features (market/statistical factors
attached to each asset).

This is the leak-safe, panel-level analogue of SovAI's per-ticker rolling PCA
(``regime_change_pca.py``): loadings come from train only, and the arbitrary
PC-sign flip that made the upstream series noisy is cured by deterministic
sign-fixing.

A ``scope="global"`` variant is offered explicitly for research / full-sample
factor estimation; it flips ``leakage_safe`` to ``False`` so the core
:meth:`~polars_features.core.protocol.PanelTransformer._check_leakage` gate
refuses it across a train/test boundary unless the caller refits per fold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl

from polars_features.core.panel_frame import PanelFrame
from polars_features.core.protocol import PanelTransformer
from polars_features.reduce._base import _sign_of_max_abs

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["StatisticalFactors"]


class StatisticalFactors(PanelTransformer):
    """Leak-safe PCA statistical factors extracted from a returns panel.

    Parameters
    ----------
    returns : str, optional
        Name of the returns column to factorise. If omitted, the last feature
        column (in schema order) is used.
    k : int, optional
        Number of factors to extract. When ``None`` (default) the count is
        resolved from ``explained_variance`` on the training matrix.
    explained_variance : float, default 0.95
        Variance target used to auto-resolve ``k`` (ignored when ``k`` is given).
    scope : {"train", "global"}, default "train"
        ``"train"`` (default) fits loadings on the ``fit`` dates only and is
        ``leakage_safe``. ``"global"`` marks the instance ``leakage_safe = False``
        (intended for full-sample research fits); the core leakage gate then
        refuses it across a train/test boundary.
    sign_fix : bool, default True
        Deterministically fix each factor's sign (largest-magnitude loading
        forced positive) so loadings and factor series are stable across refits.
    prefix : str, default "factor"
        Prefix for the emitted factor columns (``factor_1 .. factor_k``).
    keep_features : bool, default False
        If False (default) the output is the ``(entity, time)`` keys plus the
        factor columns. If True, the factor columns are appended to the panel.
    random_state : int, default 42
        Seed forwarded to :class:`sklearn.decomposition.PCA`.
    entity, time : str, optional
        Default panel keys for bare polars frames.

    Attributes
    ----------
    panel_safe : bool
        Always ``True``.
    leakage_safe : bool
        ``True`` for ``scope="train"``, ``False`` for ``scope="global"``.
    loadings_ : numpy.ndarray
        Sign-fixed loadings ``(k, n_entities)`` (factor betas per entity).
    entity_names_ : list of str
        Entity ids (as strings) indexing the columns of :attr:`loadings_`.
    factor_names_ : list of str
        The emitted factor column names.
    explained_variance_ratio_ : numpy.ndarray
        Per-factor explained-variance ratio.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        *,
        returns: str | None = None,
        k: int | None = None,
        explained_variance: float = 0.95,
        scope: Literal["train", "global"] = "train",
        sign_fix: bool = True,
        prefix: str = "factor",
        keep_features: bool = False,
        random_state: int = 42,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if k is not None and (not isinstance(k, int) or k < 1):
            raise ValueError(f"`k` must be a positive integer or None, got {k!r}.")
        if not 0.0 < explained_variance <= 1.0:
            raise ValueError(
                f"`explained_variance` must be in (0, 1], got {explained_variance!r}."
            )
        if scope not in ("train", "global"):
            raise ValueError(f"`scope` must be 'train' or 'global', got {scope!r}.")
        self.returns = returns
        self.k = k
        self.explained_variance = explained_variance
        self.scope = scope
        # Per-instance override of the class default: the global variant is not
        # leak-safe and must be refused by _check_leakage across a boundary.
        self.leakage_safe = scope == "train"
        self.sign_fix = bool(sign_fix)
        self.prefix = prefix
        self.keep_features = bool(keep_features)
        self.random_state = random_state
        # learned state
        self.returns_: str | None = None
        self.pca_: Any | None = None
        self.loadings_: NDArray[Any] | None = None
        self.entity_names_: list[str] = []
        self.factor_names_: list[str] = []
        self.entity_means_: NDArray[Any] | None = None
        self.sign_flip_: NDArray[Any] | None = None
        self.explained_variance_ratio_: NDArray[Any] | None = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _resolve_returns(self, panel: PanelFrame) -> str:
        if self.returns is not None:
            if self.returns not in panel:
                raise ValueError(
                    f"{type(self).__name__}: returns column {self.returns!r} not "
                    f"found in panel. Available columns: {panel.columns}."
                )
            return self.returns
        feats = panel.feature_cols
        if not feats:
            raise ValueError(
                f"{type(self).__name__}: panel has no feature columns to use as "
                "returns. Pass `returns=` explicitly."
            )
        return feats[-1]

    def _wide_matrix(
        self, panel: PanelFrame, returns: str, entity_names: list[str] | None
    ) -> tuple[NDArray[Any], list[str], pl.Series]:
        """Pivot the panel to a (time x entity) returns matrix.

        Returns the matrix (float, with NaN for missing cells), the entity-column
        order (as strings), and the time index series.
        """
        sub = (
            panel.lazy()
            .select(
                pl.col(panel.time_col).alias("__t"),
                pl.col(panel.entity_col).cast(pl.Utf8).alias("__e"),
                pl.col(returns).cast(pl.Float64).alias("__r"),
            )
            .collect()
        )
        wide = sub.pivot(on="__e", index="__t", values="__r").sort("__t")
        if entity_names is None:
            entity_names = sorted(sub.get_column("__e").unique().to_list())
        present = set(wide.columns)
        add = [e for e in entity_names if e not in present]
        if add:
            wide = wide.with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(e) for e in add]
            )
        times = wide.get_column("__t")
        W = wide.select(entity_names).to_numpy().astype(np.float64)
        return W, entity_names, times

    def _resolve_k(self, W: NDArray[Any]) -> int:
        from sklearn.decomposition import PCA

        n_dates, n_entities = W.shape
        cap = max(1, min(n_dates, n_entities))
        if self.k is not None:
            return min(int(self.k), cap)
        temp = PCA(n_components=cap, random_state=self.random_state)
        temp.fit(W)
        ratio = np.cumsum(np.asarray(temp.explained_variance_ratio_))
        k = int(np.argmax(ratio >= self.explained_variance)) + 1
        return max(1, min(k, cap))

    # ------------------------------------------------------------------ #
    # PanelTransformer hooks
    # ------------------------------------------------------------------ #
    def _fit(self, panel: PanelFrame) -> None:
        from sklearn.decomposition import PCA

        returns = self._resolve_returns(panel)
        W, entity_names, _ = self._wide_matrix(panel, returns, None)

        means = np.nanmean(W, axis=0)
        means = np.where(np.isnan(means), 0.0, means)
        fill = np.where(np.isnan(W), means, W)

        k = self._resolve_k(fill)
        pca = PCA(n_components=k, random_state=self.random_state)
        pca.fit(fill)

        comps = np.asarray(pca.components_)
        if self.sign_fix:
            sign = np.array(
                [_sign_of_max_abs(comps[i]) for i in range(k)], dtype=np.float64
            )
        else:
            sign = np.ones(k, dtype=np.float64)

        self.returns_ = returns
        self.pca_ = pca
        self.entity_names_ = entity_names
        self.entity_means_ = means
        self.sign_flip_ = sign
        self.loadings_ = comps * sign[:, None]
        self.factor_names_ = [f"{self.prefix}_{i}" for i in range(1, k + 1)]
        self.explained_variance_ratio_ = np.asarray(pca.explained_variance_ratio_)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        if self.pca_ is None:  # pragma: no cover - guarded by _check_fitted
            raise RuntimeError(f"{type(self).__name__} is not fitted.")
        if self.returns_ not in panel:
            raise ValueError(
                f"{type(self).__name__}.transform: returns column {self.returns_!r} "
                f"not found in panel. Available columns: {panel.columns}."
            )
        W, _, times = self._wide_matrix(panel, self.returns_, self.entity_names_)
        fill = np.where(np.isnan(W), self.entity_means_, W)
        factors = np.asarray(self.pca_.transform(fill)) * self.sign_flip_

        fac_df = pl.DataFrame(
            {panel.time_col: times.rename(panel.time_col)}
        ).with_columns(
            [
                pl.Series(name=name, values=factors[:, i])
                for i, name in enumerate(self.factor_names_)
            ]
        )

        full = panel.collect()
        joined = full.join(fac_df, on=panel.time_col, how="left")
        keys = [panel.entity_col, panel.time_col]
        if self.keep_features:
            out = joined
        else:
            out = joined.select([*keys, *self.factor_names_])
        return PanelFrame(
            out, entity=panel.entity_col, time=panel.time_col, validate=False
        )

    # ------------------------------------------------------------------ #
    # Leakage contract
    # ------------------------------------------------------------------ #
    def _check_leakage(self, train: PanelFrame, test: PanelFrame | None = None) -> None:
        # First the default gate: refuse a leakage_safe=False (global) fit across
        # a train/test boundary.
        super()._check_leakage(train, test)
        if test is None or not self.leakage_safe:
            return
        # Point-in-time invariant: every test date must post-date all train dates,
        # so factor loadings fit on train cannot have seen the test cross-sections.
        tmax = train.lazy().select(pl.col(train.time_col).max()).collect().item()
        tmin = test.lazy().select(pl.col(test.time_col).min()).collect().item()
        if tmin is not None and tmax is not None and tmin <= tmax:
            raise RuntimeError(
                f"{type(self).__name__}: test dates must all be strictly greater "
                f"than the maximum training date ({tmax!r}), but the earliest test "
                f"date is {tmin!r}. Refit the factors per fold so their loadings "
                "never see the test cross-sections."
            )
