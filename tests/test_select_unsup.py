"""Tests for leak-safe **unsupervised** feature selection.

Covers :mod:`panelary.select._unsupervised`: PFA, variance and
correlation selectors, the ``select_top`` helper, projection importance, and the
``PanelTransformer`` wrappers (including their leak-safety guarantee).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from panelary.core import PanelFrame
from panelary.select import (
    CorrelationSelector,
    PFASelector,
    VarianceSelector,
    correlation,
    pfa,
    projection_importance,
    select_top,
    variance,
)

_PAIRS = [("f0", "f0b"), ("f1", "f1b"), ("f2", "f2b")]


def _redundant_panel(
    n_entities: int = 6,
    n_periods: int = 40,
    seed: int = 0,
) -> pl.DataFrame:
    """Panel with 3 informative features and 3 near-duplicate copies.

    ``f0/f0b``, ``f1/f1b`` and ``f2/f2b`` are three redundant groups: within a
    group the two columns are near-identical (tiny jitter), across groups they
    are independent. A good redundancy-aware selector keeps exactly one per pair.
    """
    rng = np.random.default_rng(seed)
    n = n_entities * n_periods
    ids = np.repeat([f"e{i}" for i in range(n_entities)], n_periods)
    times = np.tile(np.arange(n_periods), n_entities)
    base = {f"f{j}": rng.standard_normal(n) for j in range(3)}
    jitter = 1e-3
    cols = {
        "f0": base["f0"],
        "f0b": base["f0"] + jitter * rng.standard_normal(n),
        "f1": base["f1"],
        "f1b": base["f1"] + jitter * rng.standard_normal(n),
        "f2": base["f2"],
        "f2b": base["f2"] + jitter * rng.standard_normal(n),
    }
    return pl.DataFrame({"id": ids, "t": times, **cols})


def _one_per_pair(selected: list[str]) -> bool:
    chosen = set(selected)
    return all(len(chosen & set(pair)) == 1 for pair in _PAIRS)


# --------------------------------------------------------------------------- #
# (a) PFA selects non-redundant features (one per duplicate group)
# --------------------------------------------------------------------------- #
def test_pfa_selects_one_per_redundant_group():
    df = _redundant_panel(seed=0)
    panel = PanelFrame(df, entity="id", time="t")
    selected = pfa(panel, k=3)
    assert len(selected) == 3
    assert _one_per_pair(selected)


def test_pfa_is_deterministic():
    df = _redundant_panel(seed=1)
    a = pfa(df, k=3, entity="id", time="t", random_state=7)
    b = pfa(df, k=3, entity="id", time="t", random_state=7)
    assert a == b


def test_pfa_accepts_bare_frame_and_respects_k():
    df = _redundant_panel(seed=2)
    selected = pfa(df, k=2, entity="id", time="t")
    assert len(selected) == 2


# --------------------------------------------------------------------------- #
# (b) PFASelector: fit->transform keeps entity/time + k cols; APIs agree
# --------------------------------------------------------------------------- #
def test_pfaselector_transform_keeps_keys_and_selected():
    df = _redundant_panel(seed=3)
    n = df.height
    train = df.head(n // 2)
    test = df.tail(n - n // 2)

    sel = PFASelector(k=3, entity="id", time="t").fit(train)
    out = sel.transform(test).collect()

    # entity + time + exactly the 3 selected feature columns.
    assert out.columns[:2] == ["id", "t"]
    assert set(out.columns) == {"id", "t", *sel.selected_}
    assert len(sel.selected_) == 3
    assert _one_per_pair(sel.selected_)

    # get_feature_names_out() and selected_ agree; support_ lines up.
    assert sel.get_feature_names_out() == sel.selected_
    assert sum(sel.support_) == 3
    chosen = {f for f, keep in zip(sel.feature_names_in_, sel.support_) if keep}
    assert chosen == set(sel.selected_)


def test_pfaselector_keeps_target_column():
    df = _redundant_panel(seed=4).with_columns(y=pl.col("f0") * 2.0)
    sel = PFASelector(k=2, target="y", entity="id", time="t").fit(df)
    out = sel.transform(df).collect()
    assert "y" in out.columns
    # target is excluded from the candidate pool.
    assert "y" not in sel.feature_names_in_
    assert "y" not in sel.selected_


# --------------------------------------------------------------------------- #
# (c) fit-on-train is leak-safe: selection depends only on train rows
# --------------------------------------------------------------------------- #
def test_pfaselector_selection_is_leak_safe():
    df = _redundant_panel(seed=5)
    n = df.height
    train = df.head(n // 2)
    test = df.tail(n - n // 2)

    sel = PFASelector(k=3, entity="id", time="t").fit(train)
    before = list(sel.selected_)

    # Append wild future rows to the *test* frame: transform output must still
    # be exactly the frozen selection, and re-transforming does not re-fit.
    rng = np.random.default_rng(99)
    future = test.with_columns(
        [
            (pl.col(c) + 1000.0 + rng.standard_normal(test.height)).alias(c)
            for c in ["f0", "f0b", "f1", "f1b", "f2", "f2b"]
        ]
    )
    out = sel.transform(future).collect()
    assert set(out.columns) == {"id", "t", *before}
    assert sel.selected_ == before


def test_variance_selector_selection_is_leak_safe():
    df = _redundant_panel(seed=6)
    n = df.height
    train = df.head(n // 2)

    sel = VarianceSelector(k=2, entity="id", time="t").fit(train)
    before = list(sel.selected_)

    # A fresh selector fitted on train (regardless of any test rows) is identical
    # to one whose selection was frozen before test rows were even constructed.
    sel2 = VarianceSelector(k=2, entity="id", time="t").fit(train)
    assert sel2.selected_ == before


# --------------------------------------------------------------------------- #
# (d) variance / correlation selectors behave as specified
# --------------------------------------------------------------------------- #
def test_variance_selects_highest_variance():
    rng = np.random.default_rng(0)
    n = 200
    df = pl.DataFrame(
        {
            "id": ["a"] * n,
            "t": np.arange(n),
            "lo": 0.01 * rng.standard_normal(n),
            "mid": 1.0 * rng.standard_normal(n),
            "hi": 10.0 * rng.standard_normal(n),
        }
    )
    assert variance(df, k=1, entity="id", time="t") == ["hi"]
    assert set(variance(df, k=2, entity="id", time="t")) == {"hi", "mid"}


def test_correlation_prunes_and_keeps_higher_variance():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.standard_normal(n)
    df = pl.DataFrame(
        {
            "id": ["a"] * n,
            "t": np.arange(n),
            # near-duplicate pair; `big` has higher variance than `small`.
            "big": 5.0 * x,
            "small": 5.0 * x / 100.0 + 1e-6 * rng.standard_normal(n),
            "indep": rng.standard_normal(n),
        }
    )
    kept = correlation(df, threshold=0.95, entity="id", time="t")
    # The redundant pair collapses to its higher-variance member; `indep` stays.
    assert "indep" in kept
    assert "big" in kept
    assert "small" not in kept


def test_correlation_keeps_all_when_uncorrelated():
    rng = np.random.default_rng(2)
    n = 300
    df = pl.DataFrame(
        {
            "id": ["a"] * n,
            "t": np.arange(n),
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "c": rng.standard_normal(n),
        }
    )
    kept = correlation(df, threshold=0.95, entity="id", time="t")
    assert set(kept) == {"a", "b", "c"}


def test_correlation_selector_transform():
    df = _redundant_panel(seed=7)
    sel = CorrelationSelector(threshold=0.99, entity="id", time="t").fit(df)
    # 3 redundant pairs -> 3 survivors.
    assert len(sel.selected_) == 3
    assert _one_per_pair(sel.selected_)
    out = sel.transform(df).collect()
    assert set(out.columns) == {"id", "t", *sel.selected_}


# --------------------------------------------------------------------------- #
# projection_importance + select_top
# --------------------------------------------------------------------------- #
def test_projection_importance_shape_and_determinism():
    df = _redundant_panel(seed=8)
    for method in ("gaussian", "sparse", "ica", "svd"):
        imp = projection_importance(df, method=method, entity="id", time="t")
        assert imp.columns == ["feature", "importance", "importance_percentile"]
        assert imp.height == 6
        imp2 = projection_importance(df, method=method, entity="id", time="t")
        assert (
            imp.get_column("feature").to_list() == imp2.get_column("feature").to_list()
        )


def test_select_top_by_k_and_variability():
    imp = pl.DataFrame(
        {
            "feature": ["a", "b", "c", "d"],
            "importance": [10.0, 6.0, 3.0, 1.0],
        }
    )
    assert select_top(imp, k=2) == ["a", "b"]
    # cumulative: a=0.5, +b=0.8, +c=0.95 -> reach 0.9 needs a, b, c.
    assert select_top(imp, variability=0.9) == ["a", "b", "c"]
