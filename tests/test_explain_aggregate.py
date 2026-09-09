"""Panel-native aggregation: group SHAP (additive vs joint) and window SHAP."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.explain import (
    TimeAwareBackground,
    TreeAttributor,
    group_shap,
    joint_group_shap,
    tree_attributions,
    window_shap,
)

FEATURES = ["mom_1", "mom_2", "val_1", "size_1"]
GROUPS = {"momentum": ["mom_1", "mom_2"], "value": ["val_1"]}


class LinearStub:
    """Purely additive model: exact Shapley values in closed form."""

    def __init__(self, weights):
        self.w = np.asarray(weights, dtype=float)

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w

    def panelary_shap_values(self, X, background):
        X = np.asarray(X, dtype=float)
        mu = (
            np.zeros_like(self.w)
            if background is None
            else np.asarray(background, dtype=float).mean(axis=0)
        )
        return self.w[None, :] * (X - mu[None, :]), np.full(
            X.shape[0], float(self.w @ mu)
        )


class InteractingStub(LinearStub):
    """f(x) = w.x + c * x0 * x2 -- groups genuinely interact."""

    def __init__(self, weights, c=3.0):
        super().__init__(weights)
        self.c = float(c)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.c * X[:, 0] * X[:, 2]


def make_panel(n_entities=3, n_times=20, seed=0, start_time=0):
    rng = np.random.default_rng(seed)
    n = n_entities * n_times
    return pl.DataFrame(
        {
            "id": [f"e{e}" for e in range(n_entities) for _ in range(n_times)],
            "t": [start_time + k for _ in range(n_entities) for k in range(n_times)],
            **{f: rng.normal(size=n) for f in FEATURES},
        }
    )


@pytest.fixture
def train():
    return make_panel(seed=1)


@pytest.fixture
def future():
    return make_panel(seed=2, n_times=5, start_time=500)


def _bg(train, **kw):
    kw.setdefault("max_samples", 40)
    return TimeAwareBackground(**kw).fit(
        train, features=FEATURES, entity="id", time="t"
    )


def _wide(model, train, future):
    return tree_attributions(
        model,
        future,
        background=_bg(train),
        features=FEATURES,
        entity="id",
        time="t",
    )


# --------------------------------------------------------------------------- #
# Additive group SHAP
# --------------------------------------------------------------------------- #
def test_additive_group_shap_is_the_exact_member_sum(train, future):
    wide = _wide(LinearStub([1.0, -2.0, 0.5, 3.0]), train, future)
    grouped = group_shap(wide, GROUPS, entity="id", time="t")
    np.testing.assert_allclose(
        grouped.get_column("shap_group_momentum").to_numpy(),
        wide.get_column("shap_mom_1").to_numpy()
        + wide.get_column("shap_mom_2").to_numpy(),
    )
    np.testing.assert_allclose(
        grouped.get_column("shap_group_value").to_numpy(),
        wide.get_column("shap_val_1").to_numpy(),
    )


def test_group_shap_preserves_efficiency_via_the_residual_group(train, future):
    model = LinearStub([1.0, -2.0, 0.5, 3.0])
    wide = _wide(model, train, future)
    grouped = group_shap(wide, GROUPS, entity="id", time="t", keep_ungrouped=True)
    group_cols = [c for c in grouped.columns if c.startswith("shap_group_")]
    assert "shap_group_ungrouped" in group_cols
    total = (
        grouped.select(group_cols).sum_horizontal().to_numpy()
        + grouped.get_column("shap_base_value").to_numpy()
    )
    np.testing.assert_allclose(
        total, model.predict(future.select(FEATURES).to_numpy()), atol=1e-10
    )


def test_group_shap_rejects_overlapping_groups(train, future):
    wide = _wide(LinearStub([1.0, 1.0, 1.0, 1.0]), train, future)
    with pytest.raises(ValueError, match="double-counted"):
        group_shap(
            wide,
            {"a": ["mom_1", "val_1"], "b": ["val_1"]},
            entity="id",
            time="t",
        )


def test_group_shap_rejects_unknown_features(train, future):
    wide = _wide(LinearStub([1.0, 1.0, 1.0, 1.0]), train, future)
    with pytest.raises(ValueError, match="not available"):
        group_shap(wide, {"a": ["nope"]}, entity="id", time="t")


# --------------------------------------------------------------------------- #
# Joint (coalition) group SHAP
# --------------------------------------------------------------------------- #
def test_joint_group_shap_matches_additive_for_an_additive_model(train, future):
    """With no interactions the two group_modes coincide -- as the theory says."""
    model = LinearStub([1.0, -2.0, 0.5, 3.0])
    bg = _bg(train)
    additive = group_shap(
        tree_attributions(
            model, future, background=bg, features=FEATURES, entity="id", time="t"
        ),
        GROUPS,
        entity="id",
        time="t",
    )
    joint = joint_group_shap(
        model,
        future,
        GROUPS,
        background=bg,
        features=FEATURES,
        entity="id",
        time="t",
    )
    for name in ("momentum", "value", "ungrouped"):
        np.testing.assert_allclose(
            joint.get_column(f"shap_group_{name}").to_numpy(),
            additive.get_column(f"shap_group_{name}").to_numpy(),
            atol=1e-9,
        )


def test_joint_group_shap_differs_when_groups_interact(train, future):
    """The distinction `group_mode` exposes is real, not cosmetic."""
    model = InteractingStub([1.0, -2.0, 0.5, 3.0], c=4.0)
    bg = _bg(train)
    additive = group_shap(
        tree_attributions(
            model, future, background=bg, features=FEATURES, entity="id", time="t"
        ),
        GROUPS,
        entity="id",
        time="t",
    )
    joint = joint_group_shap(
        model,
        future,
        GROUPS,
        background=bg,
        features=FEATURES,
        entity="id",
        time="t",
    )
    assert not np.allclose(
        joint.get_column("shap_group_momentum").to_numpy(),
        additive.get_column("shap_group_momentum").to_numpy(),
        atol=1e-6,
    )


def test_joint_group_shap_satisfies_efficiency(train, future):
    model = InteractingStub([1.0, -2.0, 0.5, 3.0], c=4.0)
    joint = joint_group_shap(
        model,
        future,
        GROUPS,
        background=_bg(train),
        features=FEATURES,
        entity="id",
        time="t",
    )
    cols = [c for c in joint.columns if c.startswith("shap_group_")]
    total = (
        joint.select(cols).sum_horizontal().to_numpy()
        + joint.get_column("shap_base_value").to_numpy()
    )
    np.testing.assert_allclose(
        total, model.predict(future.select(FEATURES).to_numpy()), atol=1e-9
    )


def test_joint_group_shap_refuses_too_many_players(train, future):
    with pytest.raises(ValueError, match="max_groups"):
        joint_group_shap(
            LinearStub([1.0, 1.0, 1.0, 1.0]),
            future,
            {f: [f] for f in FEATURES},
            background=_bg(train),
            features=FEATURES,
            entity="id",
            time="t",
            max_groups=2,
        )


def test_joint_group_shap_uses_the_past_only_background(train):
    """Joint group values inherit the same reference contract as first-order SHAP."""
    model = LinearStub([1.0, -2.0, 0.5, 3.0])
    early = make_panel(seed=3, n_times=3, start_time=5)
    grown = pl.concat([train, make_panel(seed=9, n_times=5, start_time=900)])
    kw = {"features": FEATURES, "entity": "id", "time": "t"}
    a = joint_group_shap(
        model, early, GROUPS, background=_bg(train, max_samples=10_000), **kw
    )
    b = joint_group_shap(
        model, early, GROUPS, background=_bg(grown, max_samples=10_000), **kw
    )
    np.testing.assert_allclose(
        a.get_column("shap_group_momentum").to_numpy(),
        b.get_column("shap_group_momentum").to_numpy(),
        atol=1e-12,
    )


def test_group_mode_dispatch_on_the_transformer(train, future):
    attr = TreeAttributor(
        LinearStub([1.0, -2.0, 0.5, 3.0]),
        features=FEATURES,
        entity="id",
        time="t",
        max_samples=40,
    ).fit(train)
    additive = attr.group_attributions(future, GROUPS, group_mode="additive")
    joint = attr.group_attributions(future, GROUPS, group_mode="joint")
    np.testing.assert_allclose(
        additive.get_column("shap_group_value").to_numpy(),
        joint.get_column("shap_group_value").to_numpy(),
        atol=1e-9,
    )
    with pytest.raises(ValueError, match="group_mode"):
        attr.group_attributions(future, GROUPS, group_mode="nonsense")


# --------------------------------------------------------------------------- #
# Window SHAP on each entity's own calendar
# --------------------------------------------------------------------------- #
def _ragged_shap():
    """Entity 'a' observes every step; 'b' starts late and skips steps."""
    return pl.DataFrame(
        {
            "id": ["a"] * 5 + ["b"] * 3,
            "t": [1, 2, 3, 4, 5, 10, 20, 30],
            "shap_f": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0, 300.0],
        }
    )


def test_window_shap_never_mixes_entities():
    out = window_shap(_ragged_shap(), entity="id", time="t", window=2, agg="sum")
    got = out.sort(["id", "t"]).get_column("shap_f_w2").to_list()
    # First observation of *each* entity is null (window not yet full).
    assert got[0] is None
    assert got[5] is None
    assert got[1:5] == [3.0, 5.0, 7.0, 9.0]
    assert got[6:] == [300.0, 500.0]


def test_window_shap_respects_each_entitys_own_calendar():
    """A duration window uses the entity's own time values, not a global grid."""
    out = window_shap(_ragged_shap(), entity="id", time="t", window="3i")
    got = out.sort(["id", "t"]).get_column("shap_f_w3i").to_list()
    # 'b' observes at t = 10, 20, 30, so a 3-step period never spans two rows.
    assert got[5:] == [100.0, 200.0, 300.0]
    # 'a' observes every step, so its 3-step period accumulates.
    assert got[:5] == [1.0, 3.0, 6.0, 9.0, 12.0]


def test_window_shap_abs_mean_profile():
    df = pl.DataFrame(
        {"id": ["a"] * 4, "t": [1, 2, 3, 4], "shap_f": [1.0, -3.0, 5.0, -7.0]}
    )
    out = window_shap(df, entity="id", time="t", window=2, agg="abs_mean")
    assert out.get_column("shap_f_w2").to_list() == [None, 2.0, 4.0, 6.0]


def test_window_shap_validates_arguments():
    df = _ragged_shap()
    with pytest.raises(ValueError, match="agg"):
        window_shap(df, entity="id", time="t", window=2, agg="nope")
    with pytest.raises(ValueError, match="no attribution columns"):
        window_shap(df.rename({"shap_f": "f"}), entity="id", time="t", window=2)


def test_window_attributions_from_the_transformer(train, future):
    attr = TreeAttributor(
        LinearStub([1.0, -2.0, 0.5, 3.0]),
        features=FEATURES,
        entity="id",
        time="t",
        max_samples=40,
    ).fit(train)
    out = attr.window_attributions(future, window=2, agg="mean")
    assert out.height == future.height
    assert "shap_mom_1_w2" in out.columns
