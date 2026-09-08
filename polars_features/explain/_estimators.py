"""``TreeAttributor`` -- attribution as a leak-safe ``PanelTransformer`` step.

The point of wrapping attribution in the
:class:`~polars_features.core.protocol.PanelTransformer` contract is that the
*reference set* becomes fold-bound automatically: ``fit(train)`` builds the
:class:`~polars_features.explain.TimeAwareBackground` from the training rows and
freezes it; ``transform(test)`` computes attributions against that frozen,
past-only reference and learns nothing. Drop it into a
:class:`~polars_features.core.pipeline.Pipeline` behind a purged splitter and
the explanations inherit the split's leak-safety instead of quietly undoing it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import polars as pl

from polars_features.core.panel_frame import PanelFrame
from polars_features.core.protocol import PanelTransformer
from polars_features.explain._aggregate import (
    GROUP_PREFIX,
    group_shap,
    joint_group_shap,
    window_shap,
)
from polars_features.explain._background import TimeAwareBackground
from polars_features.explain._common import SHAP_PREFIX, resolve_features
from polars_features.explain._tree import check_efficiency, tree_attributions

if TYPE_CHECKING:
    pass

__all__ = ["TreeAttributor"]


class TreeAttributor(PanelTransformer):
    """Leak-safe, panel-keyed feature attribution for a fitted tree model.

    Parameters
    ----------
    model : object
        A **fitted** PanelKit model wrapper (e.g.
        :class:`~polars_features.models.PanelLGBMRegressor`), a bare fitted
        booster / sklearn tree ensemble, or any object exposing
        ``panelkit_shap_values(X, background)``. The attributor never trains
        anything.
    features : sequence of str, optional
        Feature columns to attribute. Defaults to the model's own resolved
        feature list when available, else every numeric feature column.
    mode : {"interventional", "conditional", "path_dependent"}, default="interventional"
        Value-function semantics; see
        :class:`~polars_features.explain.TimeAwareBackground`.
    background : str or TimeAwareBackground, default="past"
        Either a policy name (``"past"``, ``"cross_sectional"``, ``"fold"``) or a
        pre-configured :class:`TimeAwareBackground`, whose **configuration** is
        reused but which is always **refitted on the fit panel** so the reference
        set can never carry rows from another fold.
    max_samples, embargo, seed, min_samples, on_empty
        Forwarded to :class:`TimeAwareBackground` when ``background`` is a policy
        string.
    output : {"wide", "long"}, default="wide"
        ``"wide"`` appends ``shap_<feature>`` columns; ``"long"`` returns a tidy
        ``(entity, time, feature, shap_value)`` frame.
    keep : {"all", "shap"}, default="all"
        For ``output="wide"``: keep the incoming panel columns alongside the
        attributions, or emit only the keys plus attributions.
    prefix : str, default="shap_"
        Attribution column prefix.
    class_index : int, optional
        Which output to attribute for multiclass models.
    include_base : bool, default=True
        Emit the ``shap_base_value`` column.
    entity, time : str, optional
        Default panel keys for bare polars frames.

    Attributes
    ----------
    background_ : TimeAwareBackground
        The frozen, fold-bound reference set (available after :meth:`fit`).
    features_ : list of str
        The attributed features, in attribution-column order.

    Examples
    --------
    >>> from polars_features.explain import TreeAttributor          # doctest: +SKIP
    >>> attr = TreeAttributor(fitted_model, embargo=1).fit(train)   # doctest: +SKIP
    >>> attr.transform(test).collect()                              # doctest: +SKIP

    Notes
    -----
    ``panel_safe`` / ``leakage_safe`` are both ``True``: no attribution for a row
    at ``(e, t)`` depends on any row at ``t' >= t``, on any other entity present
    in the frame being explained, or on any fold other than the fit fold. That
    invariant is the subject of ``tests/test_explain_leakage.py``.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        model: Any,
        *,
        features: Sequence[str] | None = None,
        mode: str = "interventional",
        background: str | TimeAwareBackground = "past",
        max_samples: int = 200,
        embargo: int = 0,
        seed: int = 0,
        min_samples: int = 1,
        on_empty: str = "null",
        output: str = "wide",
        keep: str = "all",
        prefix: str = SHAP_PREFIX,
        class_index: int | None = None,
        include_base: bool = True,
        i_accept_path_dependent_background: bool = False,
        i_accept_within_fold_lookahead: bool = False,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if output not in ("wide", "long"):
            raise ValueError(f"`output` must be 'wide' or 'long', got {output!r}.")
        if keep not in ("all", "shap"):
            raise ValueError(f"`keep` must be 'all' or 'shap', got {keep!r}.")
        self.model = model
        self.features = list(features) if features is not None else None
        self.mode = mode
        self.background = background
        self.max_samples = max_samples
        self.embargo = embargo
        self.seed = seed
        self.min_samples = min_samples
        self.on_empty = on_empty
        self.output = output
        self.keep = keep
        self.prefix = prefix
        self.class_index = class_index
        self.include_base = include_base
        self.i_accept_path_dependent_background = i_accept_path_dependent_background
        self.i_accept_within_fold_lookahead = i_accept_within_fold_lookahead
        # learned state
        self.background_: TimeAwareBackground | None = None
        self.features_: list[str] = []

    # ------------------------------------------------------------------ #
    # Background construction
    # ------------------------------------------------------------------ #
    def _new_background(self) -> TimeAwareBackground:
        """Build a fresh, unfitted background from the configuration."""
        if isinstance(self.background, TimeAwareBackground):
            src = self.background
            return TimeAwareBackground(
                mode=src.mode,
                policy=src.policy,
                max_samples=src.max_samples,
                embargo=src.embargo,
                seed=src.seed,
                min_samples=src.min_samples,
                on_empty=src.on_empty,
                i_accept_path_dependent_background=(
                    src.i_accept_path_dependent_background
                ),
                i_accept_within_fold_lookahead=src.i_accept_within_fold_lookahead,
            )
        return TimeAwareBackground(
            mode=self.mode,
            policy=str(self.background),
            max_samples=self.max_samples,
            embargo=self.embargo,
            seed=self.seed,
            min_samples=self.min_samples,
            on_empty=self.on_empty,
            i_accept_path_dependent_background=(
                self.i_accept_path_dependent_background
            ),
            i_accept_within_fold_lookahead=self.i_accept_within_fold_lookahead,
        )

    # ------------------------------------------------------------------ #
    # PanelTransformer hooks
    # ------------------------------------------------------------------ #
    def _fit(self, panel: PanelFrame) -> None:
        from polars_features.explain._common import unwrap_model

        _, model_feats = unwrap_model(self.model)
        wanted = self.features if self.features is not None else (model_feats or None)
        self.features_ = resolve_features(panel, wanted, who="TreeAttributor.fit")
        bg = self._new_background()
        # Path-dependent needs no rows, but we still bind it to the fit fold so
        # `describe()` and the efficiency check have a reference to report.
        bg.fit(panel, features=self.features_)
        self.background_ = bg

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        out = self._attributions(panel)
        if self.output == "wide" and self.keep == "all":
            base = panel.lazy().collect()
            new_cols = [c for c in out.columns if c not in base.columns]
            out = base.hstack(out.select(new_cols))
        return PanelFrame(
            out, entity=panel.entity_col, time=panel.time_col, validate=False
        )

    def _attributions(self, panel: PanelFrame) -> pl.DataFrame:
        if self.background_ is None:  # pragma: no cover - guarded by _check_fitted
            raise RuntimeError("TreeAttributor is not fitted.")
        return tree_attributions(
            self.model,
            panel,
            background=self.background_,
            features=self.features_,
            output=self.output,
            prefix=self.prefix,
            class_index=self.class_index,
            include_base=self.include_base,
        )

    # ------------------------------------------------------------------ #
    # Convenience API
    # ------------------------------------------------------------------ #
    def attributions(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> pl.DataFrame:
        """Return the attributions as a plain :class:`polars.DataFrame`."""
        panel = self._as_panel(X, method="attributions", entity=entity, time=time)
        self._check_fitted("attributions")
        return self._attributions(panel)

    def check_efficiency(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        tol: float = 1e-5,
        raise_on_fail: bool = False,
        entity: str | None = None,
        time: str | None = None,
    ) -> pl.DataFrame:
        """Verify ``E[f] + sum_j phi_j == f(x)`` on the model's raw output scale.

        The Shapley efficiency (local-accuracy) axiom is the cheapest end-to-end
        proof that the attribution engine, the background and the model agree.
        Returns a one-row report; pass ``raise_on_fail=True`` to assert.
        """
        panel = self._as_panel(X, method="check_efficiency", entity=entity, time=time)
        self._check_fitted("check_efficiency")
        assert self.background_ is not None  # noqa: S101
        return check_efficiency(
            self.model,
            panel,
            background=self.background_,
            features=self.features_,
            class_index=self.class_index,
            tol=tol,
            raise_on_fail=raise_on_fail,
        )

    def group_attributions(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        groups: Mapping[str, Sequence[str]],
        *,
        group_mode: str = "additive",
        keep_ungrouped: bool = True,
        ungrouped_name: str = "ungrouped",
        out_prefix: str = GROUP_PREFIX,
        entity: str | None = None,
        time: str | None = None,
        **joint_kwargs: Any,
    ) -> pl.DataFrame:
        """Attributions aggregated over a feature partition.

        Parameters
        ----------
        groups : mapping of str to sequence of str
            Disjoint feature groups (factor buckets: momentum / value / size...).
        group_mode : {"additive", "joint"}, default="additive"
            ``"additive"`` sums member attributions -- exact for marginal values,
            free (a post-aggregation, not a recompute). ``"joint"`` re-solves the
            Shapley game with each **group as one coalition player**, which is a
            different (and more expensive) quantity: what the prediction loses
            when the whole group is held out together. They coincide only when
            the groups do not interact.
        """
        if group_mode not in ("additive", "joint"):
            raise ValueError(
                f"`group_mode` must be 'additive' or 'joint', got {group_mode!r}."
            )
        panel = self._as_panel(X, method="group_attributions", entity=entity, time=time)
        self._check_fitted("group_attributions")
        assert self.background_ is not None  # noqa: S101
        if group_mode == "joint":
            return joint_group_shap(
                self.model,
                panel,
                groups,
                background=self.background_,
                features=self.features_,
                include_ungrouped=keep_ungrouped,
                ungrouped_name=ungrouped_name,
                out_prefix=out_prefix,
                **joint_kwargs,
            )
        wide = tree_attributions(
            self.model,
            panel,
            background=self.background_,
            features=self.features_,
            output="wide",
            prefix=self.prefix,
            class_index=self.class_index,
            include_base=True,
        )
        return group_shap(
            wide,
            groups,
            entity=panel.entity_col,
            time=panel.time_col,
            prefix=self.prefix,
            out_prefix=out_prefix,
            keep_ungrouped=keep_ungrouped,
            ungrouped_name=ungrouped_name,
        )

    def window_attributions(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        window: int | str,
        agg: str = "sum",
        min_periods: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> pl.DataFrame:
        """Trailing per-entity window aggregation of this model's attributions."""
        panel = self._as_panel(
            X, method="window_attributions", entity=entity, time=time
        )
        self._check_fitted("window_attributions")
        assert self.background_ is not None  # noqa: S101
        wide = tree_attributions(
            self.model,
            panel,
            background=self.background_,
            features=self.features_,
            output="wide",
            prefix=self.prefix,
            class_index=self.class_index,
            include_base=False,
        )
        return window_shap(
            wide,
            entity=panel.entity_col,
            time=panel.time_col,
            window=window,
            prefix=self.prefix,
            agg=agg,
            min_periods=min_periods,
        )

    def background_report(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> pl.DataFrame:
        """Audit table: how many reference rows each explained time actually saw."""
        panel = self._as_panel(X, method="background_report", entity=entity, time=time)
        self._check_fitted("background_report")
        assert self.background_ is not None  # noqa: S101
        times = panel.lazy().select(panel.time_col).collect().get_column(panel.time_col)
        return self.background_.describe(times)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self._fitted else "not fitted"
        return (
            f"TreeAttributor(model={type(self.model).__name__}, mode={self.mode!r}, "
            f"background={self.background!r}, output={self.output!r}, {state})"
        )
