"""Native TreeSHAP dispatch: exactness, output shape, and booster parity.

Booster-dependent tests are guarded with ``pytest.importorskip`` -- the boosters
are optional extras. The engine-independent behaviour (output shapes, long/wide
round-trip, efficiency plumbing) is exercised through the deterministic stub so
it runs everywhere.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_features.explain import (
    BASE_VALUE_COL,
    TimeAwareBackground,
    TreeAttributor,
    check_efficiency,
    long_to_wide,
    tree_attributions,
    wide_to_long,
)

FEATURES = ["x0", "x1", "x2"]


class LinearStub:
    """Additive model with exact closed-form interventional Shapley values."""

    def __init__(self, weights):
        self.w = np.asarray(weights, dtype=float)

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w

    def panelkit_shap_values(self, X, background):
        X = np.asarray(X, dtype=float)
        mu = (
            np.zeros_like(self.w)
            if background is None
            else np.asarray(background, dtype=float).mean(axis=0)
        )
        return self.w[None, :] * (X - mu[None, :]), np.full(
            X.shape[0], float(self.w @ mu)
        )


def make_panel(n_entities=4, n_times=25, seed=0, start_time=0, with_target=False):
    rng = np.random.default_rng(seed)
    n = n_entities * n_times
    df = pl.DataFrame(
        {
            "id": [f"e{e}" for e in range(n_entities) for _ in range(n_times)],
            "t": [start_time + k for _ in range(n_entities) for k in range(n_times)],
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    if with_target:
        df = df.with_columns(
            (
                2.0 * pl.col("x0")
                - 1.0 * pl.col("x1")
                + 0.5 * pl.col("x0") * pl.col("x2")
            ).alias("y")
        )
    return df


@pytest.fixture
def train():
    return make_panel(seed=1, with_target=True)


@pytest.fixture
def future():
    return make_panel(seed=2, n_times=6, start_time=1000, with_target=True)


def _bg(train, **kw):
    kw.setdefault("max_samples", 300)
    return TimeAwareBackground(**kw).fit(
        train, features=FEATURES, entity="id", time="t"
    )


# --------------------------------------------------------------------------- #
# Engine-independent output contract
# --------------------------------------------------------------------------- #
def test_wide_output_shape_and_column_names(train, future):
    out = tree_attributions(
        LinearStub([1.0, 2.0, 3.0]),
        future,
        background=_bg(train),
        features=FEATURES,
        entity="id",
        time="t",
    )
    assert out.height == future.height
    assert out.columns == ["id", "t", "shap_x0", "shap_x1", "shap_x2", BASE_VALUE_COL]


def test_long_output_is_tidy_and_round_trips(train, future):
    kw = {"background": _bg(train), "features": FEATURES, "entity": "id", "time": "t"}
    model = LinearStub([1.0, 2.0, 3.0])
    wide = tree_attributions(model, future, output="wide", **kw)
    long = tree_attributions(model, future, output="long", **kw)
    assert long.columns == ["id", "t", BASE_VALUE_COL, "feature", "shap_value"]
    assert long.height == future.height * len(FEATURES)
    assert set(long.get_column("feature").unique().to_list()) == set(FEATURES)
    back = long_to_wide(long.drop(BASE_VALUE_COL), entity="id", time="t")
    merged = wide.join(back, on=["id", "t"], suffix="_rt")
    for f in FEATURES:
        np.testing.assert_allclose(
            merged.get_column(f"shap_{f}").to_numpy(),
            merged.get_column(f"shap_{f}_rt").to_numpy(),
        )


def test_wide_to_long_strips_the_prefix(train, future):
    wide = tree_attributions(
        LinearStub([1.0, 1.0, 1.0]),
        future,
        background=_bg(train),
        features=FEATURES,
        entity="id",
        time="t",
        include_base=False,
    )
    long = wide_to_long(wide, entity="id", time="t")
    assert sorted(long.get_column("feature").unique().to_list()) == FEATURES


def test_efficiency_holds_for_the_stub(train, future):
    model = LinearStub([1.5, -0.5, 2.0])
    rep = check_efficiency(
        model,
        future,
        background=_bg(train),
        features=FEATURES,
        entity="id",
        time="t",
        raise_on_fail=True,
    )
    assert rep.get_column("ok").item() is True
    assert rep.get_column("n_checked").item() == future.height


def test_efficiency_failure_is_reported_not_silent(train, future):
    class Broken(LinearStub):
        def panelkit_shap_values(self, X, background):
            phi, base = super().panelkit_shap_values(X, background)
            return phi * 0.5, base

    rep = check_efficiency(
        Broken([1.0, 1.0, 1.0]),
        future,
        background=_bg(train),
        features=FEATURES,
        entity="id",
        time="t",
    )
    assert rep.get_column("ok").item() is False
    with pytest.raises(AssertionError, match="efficiency check failed"):
        check_efficiency(
            Broken([1.0, 1.0, 1.0]),
            future,
            background=_bg(train),
            features=FEATURES,
            entity="id",
            time="t",
            raise_on_fail=True,
        )


def test_transformer_keep_all_appends_columns(train, future):
    attr = TreeAttributor(
        LinearStub([1.0, 1.0, 1.0]),
        features=FEATURES,
        keep="all",
        entity="id",
        time="t",
    ).fit(train)
    out = attr.transform(future).collect()
    assert set(future.columns).issubset(out.columns)
    assert "shap_x1" in out.columns


def test_transformer_keep_shap_only(train, future):
    attr = TreeAttributor(
        LinearStub([1.0, 1.0, 1.0]),
        features=FEATURES,
        keep="shap",
        entity="id",
        time="t",
    ).fit(train)
    out = attr.transform(future).collect()
    assert out.columns == ["id", "t", "shap_x0", "shap_x1", "shap_x2", BASE_VALUE_COL]


def test_background_report_is_auditable(train, future):
    attr = TreeAttributor(
        LinearStub([1.0, 1.0, 1.0]), features=FEATURES, entity="id", time="t"
    ).fit(train)
    rep = attr.background_report(future)
    assert rep.columns == ["time", "n_admissible", "n_reference"]
    assert (rep.get_column("n_reference") > 0).all()


# --------------------------------------------------------------------------- #
# LightGBM (native `pred_contrib`)
# --------------------------------------------------------------------------- #
def test_lightgbm_path_dependent_is_exact(train, future):
    lgb = pytest.importorskip("lightgbm")
    est = lgb.LGBMRegressor(n_estimators=25, num_leaves=7, verbose=-1)
    est.fit(train.select(FEATURES).to_numpy(), train.get_column("y").to_numpy())

    with pytest.warns(UserWarning):
        bg = TimeAwareBackground(
            mode="path_dependent", i_accept_path_dependent_background=True
        )
    out = tree_attributions(
        est, future, background=bg, features=FEATURES, entity="id", time="t"
    )
    total = (
        out.select([f"shap_{f}" for f in FEATURES]).sum_horizontal().to_numpy()
        + out.get_column(BASE_VALUE_COL).to_numpy()
    )
    raw = est.predict(future.select(FEATURES).to_numpy(), raw_score=True)
    np.testing.assert_allclose(total, raw, atol=1e-8)


def test_lightgbm_through_the_panelkit_wrapper(train, future):
    pytest.importorskip("lightgbm")
    from polars_features.models import PanelLGBMRegressor

    model = PanelLGBMRegressor(
        target="y",
        features=FEATURES,
        entity="id",
        time="t",
        n_estimators=25,
        num_leaves=7,
        verbose=-1,
    ).fit(train)

    with pytest.warns(UserWarning):
        bg = TimeAwareBackground(
            mode="path_dependent", i_accept_path_dependent_background=True
        )
    attr = TreeAttributor(model, background=bg, entity="id", time="t").fit(train)
    # The wrapper carries its own resolved feature list -- no `features=` needed.
    assert attr.features_ == FEATURES
    rep = attr.check_efficiency(future, tol=1e-8, raise_on_fail=True)
    assert rep.get_column("ok").item() is True


def test_lightgbm_conditional_reports_a_past_only_reference(train, future):
    pytest.importorskip("lightgbm")
    from polars_features.models import PanelLGBMRegressor

    model = PanelLGBMRegressor(
        target="y",
        features=FEATURES,
        entity="id",
        time="t",
        n_estimators=15,
        num_leaves=5,
        verbose=-1,
    ).fit(train)
    attr = TreeAttributor(model, mode="conditional", entity="id", time="t").fit(train)
    out = attr.attributions(future)
    assert "shap_reference_expectation" in out.columns
    assert out.get_column("shap_reference_expectation").is_not_null().all()
    attr.check_efficiency(future, tol=1e-8, raise_on_fail=True)


def test_lightgbm_multiclass_requires_a_class_index(train, future):
    lgb = pytest.importorskip("lightgbm")
    y = (train.get_column("y").to_numpy() > 0).astype(int) + (
        train.get_column("x1").to_numpy() > 0
    ).astype(int)
    est = lgb.LGBMClassifier(n_estimators=10, num_leaves=5, verbose=-1)
    est.fit(train.select(FEATURES).to_numpy(), y)

    with pytest.warns(UserWarning):
        bg = TimeAwareBackground(
            mode="path_dependent", i_accept_path_dependent_background=True
        )
    kw = {"background": bg, "features": FEATURES, "entity": "id", "time": "t"}
    with pytest.raises(ValueError, match="multiclass"):
        tree_attributions(est, future, **kw)
    out = tree_attributions(est, future, class_index=1, **kw)
    assert out.height == future.height


# --------------------------------------------------------------------------- #
# Interventional engine (`shap`, optional)
# --------------------------------------------------------------------------- #
def test_interventional_against_a_past_only_background_is_exact(train, future):
    pytest.importorskip("lightgbm")
    pytest.importorskip("shap")
    import lightgbm as lgb

    est = lgb.LGBMRegressor(n_estimators=20, num_leaves=5, verbose=-1)
    est.fit(train.select(FEATURES).to_numpy(), train.get_column("y").to_numpy())
    rep = check_efficiency(
        est,
        future,
        background=_bg(train, max_samples=64),
        features=FEATURES,
        entity="id",
        time="t",
        tol=1e-6,
    )
    assert rep.get_column("ok").item() is True


def test_interventional_without_shap_is_an_actionable_error(train, future):
    pytest.importorskip("lightgbm")
    if _have("shap"):
        pytest.skip("`shap` is installed; the missing-dependency path is moot")
    import lightgbm as lgb

    est = lgb.LGBMRegressor(n_estimators=5, verbose=-1)
    est.fit(train.select(FEATURES).to_numpy(), train.get_column("y").to_numpy())
    with pytest.raises(ImportError, match=r"polars-features\[explain\]"):
        tree_attributions(
            est,
            future,
            background=_bg(train),
            features=FEATURES,
            entity="id",
            time="t",
        )


def _have(module):
    from polars_features._deps import have

    return have(module)
