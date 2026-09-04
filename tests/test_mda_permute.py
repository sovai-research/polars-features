"""Tests for panel-aware permutation in :func:`polars_features.select.mda`.

These cover the ``permute_within`` correctness fix:

* :func:`_permute_within` keeps each group's value multiset intact (within-entity
  and within-time), while a global shuffle does not respect group boundaries.
* :func:`mda(..., permute_within="entity")` returns the expected tidy frame and
  runs through a :class:`PurgedKFold`.
* a feature that is a pure per-entity constant gets ~0 within-entity importance
  (the regression test for the cross-entity-shuffle bug).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from polars_features.core import PanelFrame
from polars_features.core.model_selection import PurgedKFold
from polars_features.select import mda
from polars_features.select._methods import _permute_within


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
    feats = {f"f{j}": rng.standard_normal(n) for j in range(4)}
    y = 3.0 * feats["f0"] - 2.0 * feats["f2"] + 0.1 * rng.standard_normal(n)
    return pl.DataFrame({"id": ids, "t": times, **feats, "y": y})


# --------------------------------------------------------------------------- #
# (a) _permute_within preserves per-group multisets; global shuffle does not
# --------------------------------------------------------------------------- #
def test_permute_within_preserves_per_group_multiset():
    rng = np.random.default_rng(0)
    groups = np.repeat([0, 1, 2], 5)
    values = np.arange(15, dtype=float)

    permuted = _permute_within(values, groups, rng)

    # Global multiset is preserved (it is a permutation, not a resample).
    assert sorted(permuted.tolist()) == sorted(values.tolist())
    # Per-group sorted values are unchanged: nothing crossed a group boundary.
    for g in np.unique(groups):
        mask = groups == g
        assert sorted(permuted[mask].tolist()) == sorted(values[mask].tolist())


def test_permute_within_time_groups():
    rng = np.random.default_rng(1)
    # Interleaved time key: rows for the same date are non-contiguous.
    time_key = np.tile([10, 20, 30], 6)
    values = np.arange(18, dtype=float)

    permuted = _permute_within(values, time_key, rng)

    for t in np.unique(time_key):
        mask = time_key == t
        assert sorted(permuted[mask].tolist()) == sorted(values[mask].tolist())


def test_global_shuffle_breaks_group_multiset():
    rng = np.random.default_rng(0)
    groups = np.repeat([0, 1, 2], 20)
    # Each group's values live in a disjoint band so a cross-group swap is
    # detectable: group g holds values in [100*g, 100*g + 20).
    values = np.concatenate(
        [np.arange(20) + 100 * g for g in range(3)]
    ).astype(float)

    global_shuffled = _permute_within(values, None, rng)

    # A global permutation reorders the whole vector, so at least one group's
    # sorted values change (values leak across the disjoint bands).
    changed = any(
        sorted(global_shuffled[groups == g].tolist())
        != sorted(values[groups == g].tolist())
        for g in np.unique(groups)
    )
    assert changed


def test_permute_within_determinism():
    groups = np.repeat([0, 1, 2, 3], 6)
    values = np.arange(24, dtype=float)

    a = _permute_within(values, groups, np.random.default_rng(42))
    b = _permute_within(values, groups, np.random.default_rng(42))
    c = _permute_within(values, groups, np.random.default_rng(7))

    assert np.array_equal(a, b)  # same seed -> identical
    assert not np.array_equal(a, c)  # different seed -> different (w.h.p.)


# --------------------------------------------------------------------------- #
# (b) mda(..., permute_within="entity") shape + runs under PurgedKFold
# --------------------------------------------------------------------------- #
def test_mda_permute_within_entity_shape_and_purged_kfold():
    df = _synthetic_panel(seed=3)
    panel = PanelFrame(df, entity="id", time="t")
    cv = PurgedKFold(n_splits=4, horizon=1, embargo=1)

    imp = mda(LinearRegression(), panel, "y", cv, permute_within="entity")

    assert imp.columns == ["feature", "importance", "importance_std"]
    assert imp.height == 4
    top2 = set(imp.head(2).get_column("feature").to_list())
    assert top2 == {"f0", "f2"}
    assert not imp.get_column("importance").is_null().any()


def test_mda_default_is_within_entity():
    """The default path must be the leak-safe within-entity permutation."""
    df = _synthetic_panel(seed=4)
    panel = PanelFrame(df, entity="id", time="t")
    cv = PurgedKFold(n_splits=4, horizon=1, embargo=1)

    default_imp = mda(LinearRegression(), panel, "y", cv, random_state=0)
    entity_imp = mda(
        LinearRegression(), panel, "y", cv, random_state=0, permute_within="entity"
    )

    assert default_imp.sort("feature").get_column("importance").to_list() == (
        entity_imp.sort("feature").get_column("importance").to_list()
    )


def test_mda_permute_within_time_runs():
    df = _synthetic_panel(seed=5)
    cv = PurgedKFold(n_splits=4, horizon=1, embargo=1)
    imp = mda(
        LinearRegression(),
        df,
        "y",
        cv,
        permute_within="time",
        entity="id",
        time="t",
    )
    assert imp.height == 4
    assert not imp.get_column("importance").is_null().any()


def test_mda_rejects_bad_permute_within():
    df = _synthetic_panel(seed=6)
    cv = PurgedKFold(n_splits=3, horizon=1, embargo=1)
    try:
        mda(LinearRegression(), df, "y", cv, permute_within="nope", entity="id", time="t")
    except ValueError as exc:
        assert "permute_within" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("mda accepted an invalid permute_within value")


# --------------------------------------------------------------------------- #
# (c) a pure per-entity constant feature gets ~0 within-entity importance
# --------------------------------------------------------------------------- #
def test_per_entity_constant_feature_has_zero_within_entity_importance():
    """A feature that is constant within each entity carries no *within-entity*
    information, so shuffling it within entities is a no-op and its importance
    must be ~0. A naive global shuffle, by contrast, would inflate it."""
    rng = np.random.default_rng(11)
    n_entities, n_periods = 8, 40
    n = n_entities * n_periods
    ids = np.repeat([f"e{i}" for i in range(n_entities)], n_periods)
    times = np.tile(np.arange(n_periods), n_entities)
    entity_level = np.repeat(rng.standard_normal(n_entities), n_periods)

    f_informative = rng.standard_normal(n)
    y = 3.0 * f_informative + 2.0 * entity_level + 0.1 * rng.standard_normal(n)
    df = pl.DataFrame(
        {
            "id": ids,
            "t": times,
            "f_info": f_informative,
            "f_entity_const": entity_level,  # varies across entities, constant within
            "y": y,
        }
    )
    panel = PanelFrame(df, entity="id", time="t")
    cv = PurgedKFold(n_splits=4, horizon=1, embargo=1)

    imp = mda(LinearRegression(), panel, "y", cv, permute_within="entity")
    const_imp = (
        imp.filter(pl.col("feature") == "f_entity_const")
        .get_column("importance")
        .item()
    )
    info_imp = imp.filter(pl.col("feature") == "f_info").get_column("importance").item()

    # Within-entity shuffle leaves the constant column untouched -> ~0 importance.
    assert abs(const_imp) < 1e-9
    # The genuinely informative feature is clearly important.
    assert info_imp > 0.1
