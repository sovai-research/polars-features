"""Tests for leak-safe feature selection in :mod:`polars_features.select`."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from polars_features.core import PanelFrame
from polars_features.core.model_selection import PurgedKFold
from polars_features.select import MRMRSelector, mda, mdi, mrmr


def _synthetic_panel(
    n_entities: int = 8,
    n_periods: int = 40,
    seed: int = 0,
) -> pl.DataFrame:
    """Long panel where ``y = 3*f0 - 2*f2 + noise`` -- f0 and f2 are informative."""
    rng = np.random.default_rng(seed)
    n = n_entities * n_periods
    ids = np.repeat([f"e{i}" for i in range(n_entities)], n_periods)
    times = np.tile(np.arange(n_periods), n_entities)
    feats = {f"f{j}": rng.standard_normal(n) for j in range(6)}
    y = 3.0 * feats["f0"] - 2.0 * feats["f2"] + 0.1 * rng.standard_normal(n)
    return pl.DataFrame({"id": ids, "t": times, **feats, "y": y})


def test_mrmr_recovers_informative_features():
    df = _synthetic_panel(seed=0)
    panel = PanelFrame(df, entity="id", time="t")
    selected = mrmr(panel, "y", k=2)
    assert set(selected) == {"f0", "f2"}


def test_mrmr_accepts_bare_frame_and_respects_k():
    df = _synthetic_panel(seed=1)
    selected = mrmr(df, "y", k=3, entity="id", time="t")
    assert len(selected) == 3
    # The two informative features must be among the top-3.
    assert {"f0", "f2"}.issubset(set(selected))


def test_mrmr_first_pick_is_most_relevant():
    df = _synthetic_panel(seed=2)
    selected = mrmr(df, "y", k=1, entity="id", time="t")
    # f0 has the larger coefficient (3 vs -2) -> highest |correlation|.
    assert selected == ["f0"]


def test_mda_ranks_informative_features_highest():
    df = _synthetic_panel(seed=3)
    panel = PanelFrame(df, entity="id", time="t")
    cv = PurgedKFold(n_splits=4, horizon=1, embargo=1)
    imp = mda(LinearRegression(), panel, "y", cv, random_state=0)
    assert imp.columns == ["feature", "importance", "importance_std"]
    assert imp.height == 6
    top2 = set(imp.head(2).get_column("feature").to_list())
    assert top2 == {"f0", "f2"}
    # Informative features have strictly positive mean decrease.
    inf_imp = imp.filter(pl.col("feature").is_in(["f0", "f2"]))
    assert inf_imp.get_column("importance").min() > 0.0


def test_mda_runs_through_purged_kfold_without_error():
    df = _synthetic_panel(seed=4)
    cv = PurgedKFold(n_splits=5, horizon=2, embargo=1)
    imp = mda(
        LinearRegression(),
        df,
        "y",
        cv,
        n_repeats=2,
        entity="id",
        time="t",
    )
    assert imp.height == 6
    assert not imp.get_column("importance").is_null().any()


def test_mdi_from_fitted_tree():
    df = _synthetic_panel(seed=5)
    feats = [f"f{j}" for j in range(6)]
    X = df.select(feats).to_numpy()
    y = df.get_column("y").to_numpy()
    rf = RandomForestRegressor(n_estimators=50, random_state=0).fit(X, y)
    imp = mdi(rf, feats)
    assert imp.columns == ["feature", "importance"]
    top2 = set(imp.head(2).get_column("feature").to_list())
    assert top2 == {"f0", "f2"}


def test_mrmr_selector_as_pipeline_step():
    df = _synthetic_panel(seed=6)
    panel = PanelFrame(df, entity="id", time="t")
    selector = MRMRSelector(k=2, target="y")
    out = selector.fit_transform(panel)
    assert selector.leakage_safe is True
    assert set(selector.selected_) == {"f0", "f2"}
    collected = out.collect()
    # Keeps entity, time, the 2 selected features, and the target.
    assert set(collected.columns) == {"id", "t", "f0", "f2", "y"}


def test_mrmr_selector_drops_target_when_requested():
    df = _synthetic_panel(seed=7)
    panel = PanelFrame(df, entity="id", time="t")
    selector = MRMRSelector(k=2, target="y", keep_target=False)
    out = selector.fit_transform(panel).collect()
    assert "y" not in out.columns
    assert set(out.columns) == {"id", "t", "f0", "f2"}
