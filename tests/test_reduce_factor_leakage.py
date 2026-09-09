"""The leak-safety contract for the factor-extraction family.

Every extractor in :mod:`panelary.reduce` claims ``panel_safe = True`` and
``leakage_safe = True``. This file is what those claims cash out to, applied
uniformly to all four methods:

1. **Fit on train only.** Loadings, standardisation statistics and NaN fill
   values come from the ``fit`` panel and nothing else -- fitting on train+test
   gives *different* loadings, which is the proof that the frozen ones never saw
   the test fold.
2. **Row-local transform.** A row's factors depend only on that row and the
   frozen parameters, never on which other rows share its fold.
3. **Sign stability.** A row's factors are identical whether it was in the fit
   set or arrives later through ``transform``.
4. **Determinism.** Refitting the same estimator on the same rows is
   bit-identical.
5. **Composability.** The extractors survive a :class:`Pipeline` across a
   train/test boundary and a walk-forward split without tripping the leakage
   gate.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.core.panel_frame import PanelFrame
from panelary.core.pipeline import Pipeline
from panelary.cross_validation import expanding_window_split
from panelary.reduce import (
    HFAFactors,
    ICAFactors,
    PCAFactors,
    RobustPCAFactors,
)

ALL_EXTRACTORS = [PCAFactors, HFAFactors, ICAFactors, RobustPCAFactors]
IDS = [cls.__name__ for cls in ALL_EXTRACTORS]


def _panel_df(
    n_entities: int = 8, n_periods: int = 60, n_feat: int = 10, seed: int = 0
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_entities * n_periods
    latent = rng.standard_exponential((n, 3)) - 1.0
    loadings = rng.standard_normal((3, n_feat))
    feats = latent @ loadings + 0.5 * rng.standard_normal((n, n_feat))
    # Test rows drift, so a transform that re-learns statistics is detectable.
    drift = np.repeat(np.linspace(0.0, 2.0, n_periods)[None, :], n_entities, axis=0)
    feats = feats + drift.reshape(-1, 1)
    data: dict[str, object] = {
        "id": np.repeat([f"e{i}" for i in range(n_entities)], n_periods),
        "t": np.tile(np.arange(n_periods), n_entities),
    }
    for j in range(n_feat):
        data[f"f{j}"] = feats[:, j]
    return pl.DataFrame(data)


def _split(df: pl.DataFrame, cut: int = 40):
    return df.filter(pl.col("t") < cut), df.filter(pl.col("t") >= cut)


def _make(cls, **kwargs):
    return cls(2, entity="id", time="t", **kwargs)


def _factors(frame: pl.DataFrame) -> np.ndarray:
    return frame.select(["factor_1", "factor_2"]).to_numpy()


# --------------------------------------------------------------------------- #
# 1. fit-on-train only
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", ALL_EXTRACTORS, ids=IDS)
def test_loadings_and_stats_come_from_the_fit_panel_only(cls):
    df = _panel_df()
    train, _test = _split(df)

    on_train = _make(cls).fit(train)
    on_full = _make(cls).fit(df)

    # Train-only standardisation statistics, not full-sample ones.
    train_mean = train.select([f"f{j}" for j in range(10)]).to_numpy().mean(axis=0)
    assert np.allclose(on_train.mean_, train_mean)
    assert not np.allclose(on_train.mean_, on_full.mean_)
    # ...and therefore different loadings from a full-sample fit.
    assert not np.allclose(on_train.loadings_, on_full.loadings_, atol=1e-6)


@pytest.mark.parametrize("cls", ALL_EXTRACTORS, ids=IDS)
def test_transform_does_not_refit(cls):
    df = _panel_df()
    train, test = _split(df)
    ext = _make(cls).fit(train)

    loadings_before = ext.loadings_.copy()
    mean_before = ext.mean_.copy()
    std_before = ext.std_.copy()
    ext.transform(test)
    assert np.array_equal(ext.loadings_, loadings_before)
    assert np.array_equal(ext.mean_, mean_before)
    assert np.array_equal(ext.std_, std_before)


@pytest.mark.parametrize("cls", ALL_EXTRACTORS, ids=IDS)
def test_transform_matches_the_explicit_frozen_linear_map(cls):
    """`F = ((X - train_mean) / train_std) @ loadings` -- nothing else."""
    df = _panel_df()
    train, test = _split(df)
    ext = _make(cls).fit(train)

    X = test.select(ext.feature_names_in_).to_numpy()
    expected = ((X - ext.mean_) / ext.std_) @ ext.loadings_
    got = _factors(ext.transform(test).collect())
    assert np.allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("cls", ALL_EXTRACTORS, ids=IDS)
def test_nan_fill_uses_train_column_means(cls):
    df = _panel_df()
    train, test = _split(df)
    ext = _make(cls).fit(train)

    # Blank out a whole column in the test fold: it must be filled with the
    # TRAIN mean (a self-computed mean would be undefined / different).
    holed = test.with_columns(pl.lit(None, dtype=pl.Float64).alias("f0"))
    got = _factors(ext.transform(holed).collect())

    X = holed.select(ext.feature_names_in_).to_numpy()
    X[:, 0] = ext.column_mean_[0]
    expected = ((X - ext.mean_) / ext.std_) @ ext.loadings_
    assert np.allclose(got, expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# 2. row-local transform (test-fold composition is irrelevant)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", ALL_EXTRACTORS, ids=IDS)
def test_factors_do_not_depend_on_test_fold_composition(cls):
    df = _panel_df()
    train, test = _split(df)
    ext = _make(cls).fit(train)

    full = ext.transform(test).collect()
    half = ext.transform(test.filter(pl.col("t") < 50)).collect()
    shared = full.filter(pl.col("t") < 50)
    assert np.allclose(_factors(shared), _factors(half), atol=1e-12)

    # Even a single row on its own gets the same answer.
    one = ext.transform(test.head(1)).collect()
    assert np.allclose(_factors(one), _factors(full)[:1], atol=1e-12)


# --------------------------------------------------------------------------- #
# 3. sign stability across the fit/transform boundary
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", ALL_EXTRACTORS, ids=IDS)
def test_rows_seen_at_fit_time_get_the_same_factors_later(cls):
    df = _panel_df()
    train, _test = _split(df)
    ext = _make(cls).fit(train)

    in_fit = _factors(ext.transform(train).collect())
    # The same rows, arriving later inside a bigger panel that also holds
    # unseen future rows, must come out identical (values *and* signs).
    later = _factors(ext.transform(df).collect().filter(pl.col("t") < 40))
    assert np.allclose(in_fit, later, atol=1e-12)


@pytest.mark.parametrize("cls", ALL_EXTRACTORS, ids=IDS)
def test_loading_signs_obey_the_convention(cls):
    df = _panel_df()
    ext = _make(cls).fit(df)
    for j in range(ext.n_factors_):
        col = ext.loadings_[:, j]
        assert col[np.argmax(np.abs(col))] > 0, f"factor {j + 1} sign not fixed"


@pytest.mark.parametrize("cls", ALL_EXTRACTORS, ids=IDS)
def test_signs_survive_a_row_shuffle(cls):
    df = _panel_df()
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=11)
    a = _make(cls).fit(df)
    b = _make(cls).fit(shuffled)
    assert np.allclose(a.loadings_, b.loadings_, atol=1e-6)


# --------------------------------------------------------------------------- #
# 4. determinism
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", ALL_EXTRACTORS, ids=IDS)
def test_refit_is_bit_identical(cls):
    df = _panel_df()
    a = _make(cls).fit(df)
    b = _make(cls).fit(df)
    assert np.array_equal(a.loadings_, b.loadings_)
    assert np.array_equal(
        _factors(a.transform(df).collect()), _factors(b.transform(df).collect())
    )


# --------------------------------------------------------------------------- #
# 5. composability: Pipeline + walk-forward CV
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", ALL_EXTRACTORS, ids=IDS)
def test_pipeline_transforms_the_test_fold_without_tripping_the_gate(cls):
    df = _panel_df()
    train, test = _split(df)
    pipe = Pipeline([("factors", _make(cls))], entity="id", time="t")
    pipe.fit(train)
    out = pipe.transform(test)
    assert isinstance(out, PanelFrame)
    frame = out.collect()
    assert {"factor_1", "factor_2"}.issubset(frame.columns)
    assert frame.height == test.height


@pytest.mark.parametrize("cls", ALL_EXTRACTORS, ids=IDS)
def test_walk_forward_folds_each_get_their_own_frozen_loadings(cls):
    df = _panel_df(n_entities=4, n_periods=40, n_feat=8)
    splits = expanding_window_split(test_size=5, n_splits=3, step_size=5)(df.lazy())

    seen = []
    for _i, (train_lf, test_lf) in splits.items():
        train = train_lf.collect()
        test = test_lf.collect()
        if train.height < 40 or test.height == 0:
            continue
        ext = _make(cls).fit(train)
        out = ext.transform(test).collect()
        assert out.height == test.height
        # The test-fold projection uses only frozen train statistics.
        X = test.select(ext.feature_names_in_).to_numpy()
        expected = ((X - ext.mean_) / ext.std_) @ ext.loadings_
        assert np.allclose(_factors(out), expected, atol=1e-12)
        seen.append(ext.loadings_.copy())

    assert len(seen) >= 2
    # Expanding folds see different training data -> different loadings.
    assert not np.allclose(seen[0], seen[-1], atol=1e-9)


@pytest.mark.parametrize("cls", ALL_EXTRACTORS, ids=IDS)
def test_declares_the_contract_attributes(cls):
    assert cls.panel_safe is True
    assert cls.leakage_safe is True
