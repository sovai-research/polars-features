"""Tests for the leak-safe panel clustering package (`polars_features.cluster`)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_features.cluster import (
    CrossSectionalClusterer,
    KShapeClusterer,
    k_from_n_entities,
    ncc,
    sbd,
)
from polars_features.cluster._kshape import KShapeCore
from polars_features.core.panel_frame import PanelFrame


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def _shape_groups(n_per=8, length=64, seed=0):
    """Three distinct base shapes, each with random phase shifts + noise.

    Groups differ by *shape* (frequency), not merely by phase, so a
    shift-invariant clusterer must still separate them.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, length)
    bases = [np.sin(t), np.sin(2 * t), np.sign(np.sin(t))]  # freq1, freq2, square
    series = []
    labels = []
    for g, base in enumerate(bases):
        for _ in range(n_per):
            shift = int(rng.integers(0, length))
            s = np.roll(base, shift) + rng.normal(0, 0.05, size=length)
            series.append(s)
            labels.append(g)
    X = np.array(series)
    return X, np.array(labels)


def _panel_from_tensor(X, labels=None, times=None, entity="id", time="t"):
    """Long PanelFrame from a (m, length) array of series."""
    m, length = X.shape
    times = np.arange(length) if times is None else np.asarray(times)
    rows_id = np.repeat([f"e{i}" for i in range(m)], length)
    rows_t = np.tile(times, m)
    rows_v = X.reshape(-1)
    df = pl.DataFrame({entity: rows_id, time: rows_t, "val": rows_v})
    return PanelFrame(df, entity=entity, time=time)


def _purity(true_labels, pred_labels):
    total = len(true_labels)
    hits = 0
    for c in np.unique(pred_labels):
        members = true_labels[pred_labels == c]
        if len(members):
            counts = np.bincount(members)
            hits += counts.max()
    return hits / total


# --------------------------------------------------------------------------- #
# (a) k-Shape recovers the three synthetic clusters
# --------------------------------------------------------------------------- #
def test_kshape_recovers_three_clusters():
    X, true = _shape_groups(n_per=8, length=64, seed=1)
    core = KShapeCore(3, centroid_init="zero", max_iter=100, seed=42).fit(X)
    pred = core.labels_
    purity = _purity(true, pred)
    assert purity >= 0.8, f"purity {purity} too low"

    # Try the sklearn adjusted-rand as a stronger check when available.
    try:
        from sklearn.metrics import adjusted_rand_score

        ari = adjusted_rand_score(true, pred)
        assert ari >= 0.5, f"ARI {ari} too low"
    except ImportError:  # pragma: no cover - sklearn is a dependency
        pass


# --------------------------------------------------------------------------- #
# (b) SBD is zero on identical series and shift-invariant
# --------------------------------------------------------------------------- #
def test_sbd_zero_and_shift_invariant():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 4 * np.pi, 64)
    x = np.sin(t) + rng.normal(0, 0.01, size=64)

    assert sbd(x, x) == pytest.approx(0.0, abs=1e-8)

    # SBD uses linear (zero-padded) cross-correlation, so it is invariant to a
    # *linear* shift: the same pattern placed at different offsets in a
    # zero-padded window has SBD ~ 0.
    base = np.sin(np.linspace(0, 2 * np.pi, 30))
    for lead in (0, 8, 20):
        p = np.concatenate([base, np.zeros(30)])
        q = np.concatenate([np.zeros(lead), base, np.zeros(30 - lead)])
        assert sbd(p, q) < 0.05

    # ncc peak of a series with itself is 1.
    assert ncc(x, x).max() == pytest.approx(1.0, abs=1e-8)


# --------------------------------------------------------------------------- #
# (c) KShapeClusterer fit/predict emits columns + is deterministic
# --------------------------------------------------------------------------- #
def test_kshape_clusterer_fit_predict_deterministic():
    X, _ = _shape_groups(n_per=6, length=48, seed=7)
    panel = _panel_from_tensor(X)

    clf = KShapeClusterer("val", n_clusters=3, window="expanding", seed=42)
    out1 = clf.fit(panel).predict(panel).collect()

    # distance columns + label present
    dist_cols = [c for c in out1.columns if c.startswith("ksh_dist_")]
    assert len(dist_cols) == 3
    assert "ksh_label" in out1.columns
    # keyed to (entity, time), same number of rows as input
    assert out1.height == panel.collect().height
    # later rows produce non-null labels
    assert out1["ksh_label"].null_count() < out1.height

    # Determinism across two seeded runs.
    clf2 = KShapeClusterer("val", n_clusters=3, window="expanding", seed=42)
    out2 = clf2.fit(panel).predict(panel).collect()
    assert out1.equals(out2)


def test_kshape_clusterer_static_and_distances_emit():
    X, _ = _shape_groups(n_per=5, length=40, seed=11)
    panel = _panel_from_tensor(X)
    clf = KShapeClusterer(
        "val", n_clusters=3, window="static", emit="distances", seed=1
    )
    out = clf.fit(panel).predict(panel).collect()
    assert "ksh_label" not in out.columns
    assert len([c for c in out.columns if c.startswith("ksh_dist_")]) == 3


# --------------------------------------------------------------------------- #
# (d) No look-ahead: appending future rows never changes a past assignment
# --------------------------------------------------------------------------- #
def test_kshape_no_lookahead_causal():
    X, _ = _shape_groups(n_per=6, length=60, seed=5)
    length = X.shape[1]
    cut = 40

    # Train centroids on an early window (train fold), predict expanding.
    train_panel = _panel_from_tensor(X[:, :cut], times=np.arange(cut))
    clf = KShapeClusterer("val", n_clusters=3, window="expanding", seed=42).fit(
        train_panel
    )

    # Predict on the first `cut` times, then on the full series.
    short = _panel_from_tensor(X[:, :cut], times=np.arange(cut))
    full = _panel_from_tensor(X, times=np.arange(length))

    out_short = clf.predict(short).collect().sort(["id", "t"])
    out_full = clf.predict(full).collect().sort(["id", "t"])

    # Align on shared (id, t) and compare labels + distances.
    joined = out_short.join(out_full, on=["id", "t"], how="inner", suffix="_full")
    assert joined.height == out_short.height
    assert (
        joined["ksh_label"].fill_null(-1) == joined["ksh_label_full"].fill_null(-1)
    ).all()
    for i in range(3):
        a = joined[f"ksh_dist_{i}"].fill_null(-1.0).to_numpy()
        b = joined[f"ksh_dist_{i}_full"].fill_null(-1.0).to_numpy()
        assert np.allclose(a, b, atol=1e-9)


# --------------------------------------------------------------------------- #
# Cross-sectional clusterer
# --------------------------------------------------------------------------- #
def test_cross_sectional_fit_predict():
    rng = np.random.default_rng(0)
    # Two well-separated blobs of entities, observed over several dates.
    rows = []
    for e in range(20):
        center = (0.0, 0.0) if e < 10 else (10.0, 10.0)
        for t in range(5):
            rows.append(
                {
                    "id": f"e{e}",
                    "t": t,
                    "f1": center[0] + rng.normal(0, 0.3),
                    "f2": center[1] + rng.normal(0, 0.3),
                }
            )
    df = pl.DataFrame(rows)
    panel = PanelFrame(df, entity="id", time="t")

    clf = CrossSectionalClusterer(["f1", "f2"], n_clusters=2, emit="both", seed=0)
    out = clf.fit(panel).predict(panel).collect()
    assert "xcl_label" in out.columns
    assert out.height == df.height
    # Two distinct clusters recovered; each entity's rows share a label.
    assert out["xcl_label"].n_unique() == 2

    # Determinism.
    out2 = (
        CrossSectionalClusterer(["f1", "f2"], n_clusters=2, emit="both", seed=0)
        .fit(panel)
        .predict(panel)
        .collect()
    )
    assert out.equals(out2)


def test_cross_sectional_scaler_fit_on_train_only():
    # Scaler learned on train must not change when unseen test rows are added.
    rng = np.random.default_rng(1)
    train = pl.DataFrame(
        {
            "id": [f"e{i}" for i in range(10) for _ in range(3)],
            "t": [t for _ in range(10) for t in range(3)],
            "f1": rng.normal(0, 1, 30),
        }
    )
    panel = PanelFrame(train, entity="id", time="t")
    clf = CrossSectionalClusterer("f1", n_clusters=2, seed=0).fit(panel)
    mean_before = clf.scaler_.mean_.copy()

    # Predicting on a very different test batch must not refit the scaler.
    test = pl.DataFrame({"id": ["z", "z"], "t": [0, 1], "f1": [100.0, -100.0]})
    clf.predict(PanelFrame(test, entity="id", time="t")).collect()
    assert np.array_equal(clf.scaler_.mean_, mean_before)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def test_k_from_n_entities_monotone_capped():
    ks = [k_from_n_entities(n) for n in range(1, 5000, 50)]
    assert ks[0] == 3
    assert all(1 <= k <= 12 for k in ks)
    assert all(b >= a for a, b in zip(ks, ks[1:]))  # non-decreasing
    assert k_from_n_entities(10) == 3
    assert k_from_n_entities(100000) == 12


def test_exports():
    import polars_features.cluster as c

    for name in (
        "KShapeClusterer",
        "CrossSectionalClusterer",
        "sbd",
        "ncc",
        "k_from_n_entities",
    ):
        assert name in c.__all__
