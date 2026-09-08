"""The headline contract: attributions cannot see the future.

These tests are **unconditional** -- they must run in a bare
``numpy + polars + pytest`` environment. Attribution is therefore driven through
a tiny deterministic stub model that implements the documented
``panelkit_shap_values(X, background)`` hook with *exact* closed-form Shapley
values, so the assertions are about PanelKit's leak-safety contract and nothing
else. Booster-specific behaviour lives in ``test_explain_tree.py``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_features.core.model_selection import PurgedKFold
from polars_features.core.panel_frame import PanelFrame
from polars_features.core.protocol import PanelTransformer
from polars_features.explain import (
    BASE_VALUE_COL,
    TimeAwareBackground,
    TreeAttributor,
    tree_attributions,
)

FEATURES = ["x0", "x1", "x2"]


class LinearStub:
    """A deterministic additive model with closed-form exact Shapley values.

    ``f(x) = w . x``. For an additive model the exact interventional Shapley
    value of feature ``j`` against a background ``B`` is
    ``phi_j = w_j (x_j - mean_B(x_j))`` and ``E[f] = w . mean_B``, so efficiency
    holds identically. With no background (the ``path_dependent`` analogue) the
    implicit reference is whatever was baked in at "training" time.
    """

    def __init__(self, weights, baked_reference=None):
        self.w = np.asarray(weights, dtype=float)
        self.baked_reference = (
            np.zeros_like(self.w)
            if baked_reference is None
            else np.asarray(baked_reference, dtype=float)
        )

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w

    def panelkit_shap_values(self, X, background):
        X = np.asarray(X, dtype=float)
        mu = (
            self.baked_reference
            if background is None
            else np.asarray(background, dtype=float).mean(axis=0)
        )
        phi = self.w[None, :] * (X - mu[None, :])
        base = np.full(X.shape[0], float(self.w @ mu))
        return phi, base


def make_panel(n_entities=4, n_times=20, seed=0, start_time=0):
    rng = np.random.default_rng(seed)
    rows = {"id": [], "t": [], "x0": [], "x1": [], "x2": []}
    for e in range(n_entities):
        for k in range(n_times):
            rows["id"].append(f"e{e}")
            rows["t"].append(start_time + k)
            rows["x0"].append(float(rng.normal()))
            rows["x1"].append(float(rng.normal(loc=e)))
            rows["x2"].append(float(rng.normal(scale=2.0)))
    return pl.DataFrame(rows)


@pytest.fixture
def model():
    return LinearStub([1.5, -2.0, 0.25], baked_reference=[0.1, 0.2, 0.3])


@pytest.fixture
def train():
    return make_panel(seed=1, n_times=20, start_time=0)


@pytest.fixture
def test_fold():
    return make_panel(seed=2, n_times=8, start_time=100)


def _fit(model, train, **kw):
    kw.setdefault("features", FEATURES)
    return TreeAttributor(model, entity="id", time="t", **kw).fit(train)


def _row(df, ent, t):
    return df.filter((pl.col("id") == ent) & (pl.col("t") == t))


def _shap_of(df, ent, t):
    r = _row(df, ent, t)
    return r.select([f"shap_{f}" for f in FEATURES]).to_numpy().ravel()


# --------------------------------------------------------------------------- #
# Contract declarations
# --------------------------------------------------------------------------- #
def test_attributor_declares_the_contract():
    assert issubclass(TreeAttributor, PanelTransformer)
    assert TreeAttributor.panel_safe is True
    assert TreeAttributor.leakage_safe is True


def test_transform_before_fit_raises(model, train):
    attr = TreeAttributor(model, features=FEATURES, entity="id", time="t")
    with pytest.raises(RuntimeError, match="not fitted"):
        attr.transform(train)


# --------------------------------------------------------------------------- #
# The headline invariants
# --------------------------------------------------------------------------- #
def test_attribution_is_invariant_to_future_rows_in_the_transform_set(
    model, train, test_fold
):
    attr = _fit(model, train)
    full = attr.attributions(test_fold)
    past_only = attr.attributions(test_fold.filter(pl.col("t") <= 103))
    for t in (100, 101, 102, 103):
        np.testing.assert_allclose(
            _shap_of(full, "e1", t), _shap_of(past_only, "e1", t)
        )


def test_attribution_is_invariant_to_other_entities_in_the_transform_set(
    model, train, test_fold
):
    attr = _fit(model, train)
    full = attr.attributions(test_fold)
    one = attr.attributions(test_fold.filter(pl.col("id") == "e2"))
    for t in (100, 104, 107):
        np.testing.assert_allclose(_shap_of(full, "e2", t), _shap_of(one, "e2", t))


def test_adding_future_training_rows_does_not_change_a_past_attribution(model, train):
    """The background is past-only, so future *training* rows are inadmissible."""
    explain_at = train.filter(pl.col("t") == 10)
    future = make_panel(seed=99, n_times=10, start_time=50)
    grown = pl.concat([train, future])

    a = _fit(model, train).attributions(explain_at)
    b = _fit(model, grown).attributions(explain_at)

    cols = [f"shap_{f}" for f in FEATURES] + [BASE_VALUE_COL]
    np.testing.assert_allclose(
        a.select(cols).to_numpy(), b.select(cols).to_numpy(), atol=1e-12
    )


def test_adding_future_training_rows_does_change_a_fold_policy_attribution(
    model, train
):
    """Control: the leak we are protecting against is real, not hypothetical."""
    explain_at = train.filter(pl.col("t") == 10)
    grown = pl.concat([train, make_panel(seed=99, n_times=10, start_time=50)])
    kw = {
        "policy": "fold",
        "i_accept_within_fold_lookahead": True,
        "max_samples": 10_000,
    }
    a = _fit(model, train, background=TimeAwareBackground(**kw)).attributions(
        explain_at
    )
    b = _fit(model, grown, background=TimeAwareBackground(**kw)).attributions(
        explain_at
    )
    assert not np.allclose(
        a.get_column(BASE_VALUE_COL).to_numpy(),
        b.get_column(BASE_VALUE_COL).to_numpy(),
    )


def test_reference_rows_are_strictly_in_the_past(train, test_fold):
    bg = TimeAwareBackground(max_samples=10_000).fit(
        train, features=FEATURES, entity="id", time="t"
    )
    train_times = np.asarray(train.get_column("t").to_list())
    for t in [0, 1, 5, 19, 100]:
        rows = bg.rows_for(t)
        if rows.size:
            assert train_times[rows].max() < t


def test_embargo_drops_the_most_recent_reference_times(train):
    plain = TimeAwareBackground(embargo=0, max_samples=10_000).fit(
        train, features=FEATURES, entity="id", time="t"
    )
    embargoed = TimeAwareBackground(embargo=3, max_samples=10_000).fit(
        train, features=FEATURES, entity="id", time="t"
    )
    train_times = np.asarray(train.get_column("t").to_list())
    assert train_times[plain.rows_for(10)].max() == 9
    assert train_times[embargoed.rows_for(10)].max() == 6
    assert embargoed.rows_for(10).size < plain.rows_for(10).size


def test_no_admissible_past_yields_null_attributions(model, train):
    attr = _fit(model, train, on_empty="null")
    with pytest.warns(UserWarning, match="no admissible past-only background"):
        out = attr.attributions(train.filter(pl.col("t") == 0))
    assert out.get_column("shap_x0").is_null().all()


def test_on_empty_error_raises(model, train):
    attr = _fit(model, train, on_empty="error")
    with pytest.raises(ValueError, match="no admissible past-only background"):
        attr.attributions(train.filter(pl.col("t") == 0))


# --------------------------------------------------------------------------- #
# Mode governance
# --------------------------------------------------------------------------- #
def test_path_dependent_requires_explicit_acknowledgement():
    with pytest.raises(ValueError, match="i_accept_path_dependent_background"):
        TimeAwareBackground(mode="path_dependent")


def test_path_dependent_warns_even_when_acknowledged():
    with pytest.warns(UserWarning, match="implicit training-distribution"):
        TimeAwareBackground(
            mode="path_dependent", i_accept_path_dependent_background=True
        )


def test_path_dependent_uses_the_models_baked_in_reference(model, train, test_fold):
    with pytest.warns(UserWarning):
        bg = TimeAwareBackground(
            mode="path_dependent", i_accept_path_dependent_background=True
        )
    attr = _fit(model, train, background=bg)
    out = attr.attributions(test_fold)
    expected = float(model.w @ model.baked_reference)
    np.testing.assert_allclose(
        out.get_column(BASE_VALUE_COL).to_numpy(), expected, atol=1e-12
    )


def test_fold_policy_warns_about_within_fold_lookahead():
    with pytest.warns(UserWarning, match="not past-only"):
        TimeAwareBackground(policy="fold")


def test_conditional_reports_the_reference_expectation(model, train, test_fold):
    attr = _fit(model, train, mode="conditional")
    out = attr.attributions(test_fold)
    assert "shap_reference_expectation" in out.columns
    # Reported, never substituted: efficiency still holds against the base value.
    total = out.select([f"shap_{f}" for f in FEATURES]).sum_horizontal().to_numpy()
    total = total + out.get_column(BASE_VALUE_COL).to_numpy()
    pred = model.predict(test_fold.select(FEATURES).to_numpy())
    np.testing.assert_allclose(total, pred, atol=1e-10)


def test_cross_sectional_policy_never_pools_a_future_cross_section(train):
    bg = TimeAwareBackground(policy="cross_sectional", max_samples=10_000).fit(
        train, features=FEATURES, entity="id", time="t"
    )
    train_times = np.asarray(train.get_column("t").to_list())
    assert set(train_times[bg.rows_for(7)].tolist()) == {7}
    # A time absent from the fold falls back to the latest *past* cross-section.
    assert set(train_times[bg.rows_for(500)].tolist()) == {19}


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_reference_sampling_is_deterministic_and_frame_independent(train):
    kw = {"features": FEATURES, "entity": "id", "time": "t"}
    a = TimeAwareBackground(max_samples=7, seed=3).fit(train, **kw)
    b = TimeAwareBackground(max_samples=7, seed=3).fit(train, **kw)
    c = TimeAwareBackground(max_samples=7, seed=4).fit(train, **kw)
    np.testing.assert_array_equal(a.rows_for(12), b.rows_for(12))
    assert not np.array_equal(a.rows_for(12), c.rows_for(12))
    assert a.rows_for(12).size == 7


def test_background_describe_is_monotone_in_time(train):
    bg = TimeAwareBackground(max_samples=10_000).fit(
        train, features=FEATURES, entity="id", time="t"
    )
    rep = bg.describe(pl.Series("t", list(range(20))))
    n = rep.get_column("n_admissible").to_numpy()
    assert n[0] == 0
    assert np.all(np.diff(n) >= 0)


# --------------------------------------------------------------------------- #
# Composability inside purged CV
# --------------------------------------------------------------------------- #
def test_purged_cv_composition_keeps_every_reference_in_the_past(model):
    panel = PanelFrame(make_panel(n_entities=3, n_times=30, seed=7), "id", "t")
    cv = PurgedKFold(n_splits=4, horizon=1, embargo=1)
    n_folds = 0
    for tr, te in cv.split(panel):
        attr = _fit(model, tr)
        out = attr.attributions(te)
        assert out.height == te.collect().height
        train_times = np.asarray(tr.collect().get_column("t").to_list())
        for t in te.collect().get_column("t").unique().to_list():
            rows = attr.background_.rows_for(t)
            if rows.size:
                assert train_times[rows].max() < t
        n_folds += 1
    assert n_folds == 4


def test_check_leakage_does_not_refuse_the_attributor(model, train, test_fold):
    attr = _fit(model, train)
    attr._check_leakage(PanelFrame(train, "id", "t"), PanelFrame(test_fold, "id", "t"))


# --------------------------------------------------------------------------- #
# Functional core guards
# --------------------------------------------------------------------------- #
def test_tree_attributions_requires_a_time_aware_background(model, train):
    with pytest.raises(TypeError, match="TimeAwareBackground"):
        tree_attributions(model, train, background=None, entity="id", time="t")


def test_tree_attributions_requires_a_fitted_background(model, train):
    with pytest.raises(RuntimeError, match="not fitted"):
        tree_attributions(
            model,
            train,
            background=TimeAwareBackground(),
            features=FEATURES,
            entity="id",
            time="t",
        )


def test_background_feature_mismatch_is_rejected(model, train):
    bg = TimeAwareBackground().fit(train, features=["x0", "x1"], entity="id", time="t")
    with pytest.raises(ValueError, match="must cover exactly"):
        tree_attributions(
            model,
            train,
            background=bg,
            features=FEATURES,
            entity="id",
            time="t",
        )
