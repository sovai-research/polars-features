"""Leak-safe, panel-aware feature attribution (SHAP) for Panelary.

Panelary does **not** reimplement SHAP. Exact TreeSHAP already ships inside the
models you train (XGBoost ``pred_contribs``, LightGBM ``pred_contrib``, CatBoost
``ShapValues``), and ``shapiq`` is the maintained hub for any-order interactions.
What no library ships -- and what this subpackage is -- is **leak-safe,
panel-aware** attribution:

    Every SHAP library treats the background/reference set casually, and that is
    precisely the dominant leakage surface. Panelary makes the background a
    fold-bound, past-only, deterministic, auditable object.

Public API
----------
The reference set (the moat):

* :class:`TimeAwareBackground` -- fold-bound, past-only (``t' < t``, optional
  embargo), deterministically sampled reference sets, with an explicit
  ``interventional`` / ``conditional`` / ``path_dependent`` mode knob and a
  cross-sectional policy for per-date attribution.

Attribution:

* :class:`TreeAttributor` -- a :class:`~panelary.core.protocol.PanelTransformer`
  that dispatches to the model's own **native exact** TreeSHAP and returns
  ``shap_<feature>`` columns keyed by ``(entity, time)`` (or a tidy long frame).
* :func:`tree_attributions` -- the functional core.
* :func:`check_efficiency` -- the local-accuracy axiom, ``E[f] + sum phi = f(x)``.

Panel-native aggregation and diagnostics:

* :func:`group_shap` (additive, exact) and :func:`joint_group_shap`
  (groups as coalition players) -- the ``group_mode`` distinction made explicit.
* :func:`window_shap` -- trailing windows on each entity's **own** calendar.
* :func:`attribution_stability`, :func:`background_sensitivity`,
  :func:`attribution_drift` -- reference-induced oscillation vs regime drift.

Interactions (optional ``explain`` extra, ``shapiq`` interop only):

* :func:`interaction_values`, :func:`interaction_matrix` -- any-order Shapley
  interactions with ``max_order=k``. See :mod:`panelary.explain._interactions`
  for how this pairs with the data-side ``order=`` vocabulary in
  :mod:`panelary.reduce`.

Examples
--------
>>> from panelary.explain import TreeAttributor            # doctest: +SKIP
>>> attr = TreeAttributor(fitted_model, embargo=1).fit(train)     # doctest: +SKIP
>>> shap = attr.attributions(test)                                # doctest: +SKIP
>>> attr.check_efficiency(test, raise_on_fail=True)               # doctest: +SKIP

Notes
-----
Nothing here is imported at ``import panelary`` cost beyond numpy and
polars: every booster, ``shap`` and ``shapiq`` import is lazy and routed through
:func:`panelary._deps.require`.

Deferred (tracked, not shipped): a native Arrow/Polars TreeSHAP kernel and the
sparse Mobius/Fourier (SPEX) engine. Both were Rust-plugin work, and Panelary
dropped its Rust extension in 0.4.0.
"""

from __future__ import annotations

from panelary.explain._aggregate import (
    GROUP_PREFIX,
    group_shap,
    joint_group_shap,
    window_shap,
)
from panelary.explain._background import MODES, POLICIES, TimeAwareBackground
from panelary.explain._common import SHAP_PREFIX, long_to_wide, wide_to_long
from panelary.explain._estimators import TreeAttributor
from panelary.explain._interactions import interaction_matrix, interaction_values
from panelary.explain._stability import (
    attribution_drift,
    attribution_stability,
    background_sensitivity,
)
from panelary.explain._tree import (
    BASE_VALUE_COL,
    REFERENCE_EXPECTATION_COL,
    check_efficiency,
    tree_attributions,
)

__all__ = [
    # background (the moat)
    "TimeAwareBackground",
    "MODES",
    "POLICIES",
    # attribution
    "TreeAttributor",
    "tree_attributions",
    "check_efficiency",
    # aggregation
    "group_shap",
    "joint_group_shap",
    "window_shap",
    # stability
    "attribution_stability",
    "background_sensitivity",
    "attribution_drift",
    # interactions (optional `explain` extra)
    "interaction_values",
    "interaction_matrix",
    # helpers / constants
    "wide_to_long",
    "long_to_wide",
    "SHAP_PREFIX",
    "GROUP_PREFIX",
    "BASE_VALUE_COL",
    "REFERENCE_EXPECTATION_COL",
]
