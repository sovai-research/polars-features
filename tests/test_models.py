"""Tests for concrete panel estimators in :mod:`panelary.models`."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from panelary.core import Pipeline
from panelary.models import (
    PanelLGBMRegressor,
    PanelSklearnClassifier,
    PanelSklearnRegressor,
)
from panelary.transform.scaling import TimeSeriesScaler


def _synthetic_panel(
    n_entities: int = 8,
    n_periods: int = 30,
    seed: int = 0,
) -> pl.DataFrame:
    """Return a long panel where ``y = 3*f0 - 2*f2 + noise`` (f0, f2 informative)."""
    rng = np.random.default_rng(seed)
    n = n_entities * n_periods
    ids = np.repeat([f"e{i}" for i in range(n_entities)], n_periods)
    times = np.tile(np.arange(n_periods), n_entities)
    feats = {f"f{j}": rng.standard_normal(n) for j in range(6)}
    y = 3.0 * feats["f0"] - 2.0 * feats["f2"] + 0.1 * rng.standard_normal(n)
    data = {"id": ids, "t": times, **feats, "y": y}
    return pl.DataFrame(data)


def _train_test(df: pl.DataFrame, cutoff: int = 20):
    train = df.filter(pl.col("t") < cutoff)
    test = df.filter(pl.col("t") >= cutoff)
    return train, test


def test_regressor_fits_and_predicts_with_low_error():
    df = _synthetic_panel()
    train, test = _train_test(df)
    model = PanelSklearnRegressor(LinearRegression(), target="y")
    model.fit(train, entity="id", time="t")
    preds = model.predict(test, entity="id", time="t").collect()

    # Predictions align to (entity, time): same rows, keys preserved, one pred col.
    assert preds.height == test.height
    assert preds.columns == ["id", "t", "prediction"]
    joined = preds.join(test.select("id", "t", "y"), on=["id", "t"])
    assert joined.height == test.height

    y_true = joined.get_column("y").to_numpy()
    y_hat = joined.get_column("prediction").to_numpy()
    ss_res = float(np.sum((y_true - y_hat) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.95


def test_predictions_align_to_entity_time_order():
    df = _synthetic_panel(seed=1)
    model = PanelSklearnRegressor(LinearRegression(), target="y").fit(
        df, entity="id", time="t"
    )
    # Predict on a shuffled frame; keys must still round-trip exactly.
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=3)
    preds = model.predict(shuffled, entity="id", time="t").collect()
    assert preds.height == shuffled.height
    assert set(zip(preds["id"], preds["t"])) == set(zip(shuffled["id"], shuffled["t"]))


def test_default_estimator_is_dependency_free():
    df = _synthetic_panel(seed=2)
    train, test = _train_test(df)
    # No estimator supplied -> sklearn HistGradientBoostingRegressor.
    model = PanelSklearnRegressor(target="y").fit(train, entity="id", time="t")
    from sklearn.ensemble import HistGradientBoostingRegressor

    assert isinstance(model.estimator_, HistGradientBoostingRegressor)
    preds = model.predict(test, entity="id", time="t").collect()
    assert preds.height == test.height


def test_sample_weight_passthrough():
    base = _synthetic_panel(seed=4)
    df = base.with_columns(
        pl.Series("w", np.abs(np.random.default_rng(5).standard_normal(base.height)))
    )
    # weight is a column; y stays the target, w is excluded from features.
    model = PanelSklearnRegressor(
        LinearRegression(), target="y", sample_weight="w"
    ).fit(df, entity="id", time="t")
    assert "w" not in model.features_
    assert model.features_ == [f"f{j}" for j in range(6)]
    preds = model.predict(df, entity="id", time="t").collect()
    assert preds.height == df.height


def test_explicit_features_subset():
    df = _synthetic_panel(seed=6)
    model = PanelSklearnRegressor(
        LinearRegression(), target="y", features=["f0", "f2"]
    ).fit(df, entity="id", time="t")
    assert model.features_ == ["f0", "f2"]
    preds = model.predict(df, entity="id", time="t").collect()
    y_true = df.get_column("y").to_numpy()
    y_hat = preds.join(df.select("id", "t", "y"), on=["id", "t"]).get_column(
        "prediction"
    )
    # Only the informative features -> still near-perfect fit.
    assert np.corrcoef(y_true, y_hat.to_numpy())[0, 1] > 0.98


def test_classifier_fits_and_predicts():
    df = _synthetic_panel(seed=7).with_columns(
        (pl.col("y") > pl.col("y").median()).cast(pl.Int64).alias("label")
    )
    train, test = _train_test(df)
    model = PanelSklearnClassifier(target="label", features=[f"f{j}" for j in range(6)])
    model.fit(train, entity="id", time="t")
    preds = model.predict(test, entity="id", time="t").collect()
    assert preds.columns == ["id", "t", "prediction"]
    assert preds.height == test.height
    # predict_proba emits one column per class.
    proba = model.predict_proba(test, entity="id", time="t").collect()
    assert proba.height == test.height
    assert len([c for c in proba.columns if "proba" in c]) == 2


def test_pipeline_termination():
    df = _synthetic_panel(seed=8)
    train, test = _train_test(df)
    pipe = Pipeline(
        [
            ("scale", TimeSeriesScaler(columns=[f"f{j}" for j in range(6)])),
            (
                "model",
                PanelSklearnRegressor(LinearRegression(), target="y"),
            ),
        ]
    )
    pipe.fit(train, entity="id", time="t")
    assert pipe.leakage_safe is True
    preds = pipe.predict(test, entity="id", time="t").collect()
    assert preds.height == test.height
    assert "prediction" in preds.columns


def test_lgbm_regressor_constructs_and_predicts():
    # LightGBM may or may not be installed; either the real booster or the
    # sklearn fallback must fit/predict without error. Pass no backend-specific
    # hyper-parameters so the fallback constructor accepts the same call.
    df = _synthetic_panel(seed=9)
    train, test = _train_test(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PanelLGBMRegressor(target="y")
    model.fit(train, entity="id", time="t")
    preds = model.predict(test, entity="id", time="t").collect()
    assert preds.height == test.height
