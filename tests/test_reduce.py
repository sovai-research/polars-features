"""Tests for the leak-safe dimensionality-reduction subpackage.

Covers, per the design brief:

(a) ``PanelPCA(n_components=2).fit(train).transform(test)`` equals a reference
    sklearn PCA fit on train-only then applied to test, and *differs* from a
    full-sample fit -- proving leak-safety;
(b) component signs are stable across two fits with shuffled row order;
(c) ``CrossSectionalPCA`` produces per-date components (and each date's scores
    depend only on same-date rows);
(d) ``StatisticalFactors`` returns K factor columns + loadings, is invariant to
    appended future test dates, and the ``global`` variant is refused by the
    leakage gate.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from polars_features.core import PanelFrame
from polars_features.reduce import (
    CrossSectionalPCA,
    PanelPCA,
    StatisticalFactors,
    reduce_features,
)
from polars_features.reduce._base import _sign_of_max_abs


def _panel(n_entities: int = 10, n_periods: int = 24, n_feat: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = n_entities * n_periods
    ids = np.repeat([f"e{i}" for i in range(n_entities)], n_periods)
    times = np.tile(np.arange(n_periods), n_entities)
    # Correlated features so PCA has real structure.
    latent = rng.standard_normal((n, 2))
    loadings = rng.standard_normal((2, n_feat))
    feats = latent @ loadings + 0.3 * rng.standard_normal((n, n_feat))
    data = {"id": ids, "t": times}
    for j in range(n_feat):
        data[f"f{j}"] = feats[:, j]
    return pl.DataFrame(data)


def _split(df: pl.DataFrame, cutoff: int = 16):
    return df.filter(pl.col("t") < cutoff), df.filter(pl.col("t") >= cutoff)


def _feat_matrix(df: pl.DataFrame, cols) -> np.ndarray:
    return df.select(cols).to_numpy().astype(np.float64)


# --------------------------------------------------------------------------- #
# (a) leak-safe parity with a train-only reference + divergence from full-sample
# --------------------------------------------------------------------------- #
def test_panelpca_matches_train_only_reference_and_differs_from_full_sample():
    df = _panel(seed=1)
    train, test = _split(df)
    feat_cols = [f"f{j}" for j in range(5)]

    pca = PanelPCA(n_components=2).fit(train, entity="id", time="t")
    out = pca.transform(test, entity="id", time="t").collect()
    got = out.select("pc_1", "pc_2").to_numpy()

    # Reference: StandardScaler + PCA fit on TRAIN ONLY, same deterministic
    # sign-fix, applied to test.
    Xtr = _feat_matrix(train, feat_cols)
    Xte = _feat_matrix(test, feat_cols)
    scaler = StandardScaler().fit(Xtr)
    ref = PCA(n_components=2, random_state=42).fit(scaler.transform(Xtr))
    sign = np.array([_sign_of_max_abs(ref.components_[i]) for i in range(2)])
    ref_scores = ref.transform(scaler.transform(Xte)) * sign

    assert out.columns == ["id", "t", "pc_1", "pc_2"]
    assert np.allclose(got, ref_scores, atol=1e-8)

    # Full-sample fit gives a different rotation -> different test scores. This
    # is exactly the leak the transformer avoids.
    Xfull = _feat_matrix(df, feat_cols)
    scaler_f = StandardScaler().fit(Xfull)
    pca_f = PCA(n_components=2, random_state=42).fit(scaler_f.transform(Xfull))
    sign_f = np.array([_sign_of_max_abs(pca_f.components_[i]) for i in range(2)])
    full_scores = pca_f.transform(scaler_f.transform(Xte)) * sign_f
    assert not np.allclose(got, full_scores, atol=1e-4)


def test_panelpca_transform_is_row_local():
    # A test row's scores must not change when other test rows are dropped.
    df = _panel(seed=2)
    train, test = _split(df)
    pca = PanelPCA(n_components=2).fit(train, entity="id", time="t")

    full = pca.transform(test, entity="id", time="t").collect()
    subset_ids = test.filter(pl.col("id").is_in(["e0", "e1", "e2"]))
    sub = pca.transform(subset_ids, entity="id", time="t").collect()

    j = full.join(sub, on=["id", "t"], suffix="_s")
    assert j.height == sub.height
    assert np.allclose(
        j.select("pc_1", "pc_2").to_numpy(),
        j.select("pc_1_s", "pc_2_s").to_numpy(),
        atol=1e-10,
    )


# --------------------------------------------------------------------------- #
# (b) sign stability across shuffled-row refits
# --------------------------------------------------------------------------- #
def test_panelpca_signs_stable_under_row_shuffle():
    df = _panel(seed=3)
    train, test = _split(df)

    a = PanelPCA(n_components=3).fit(train, entity="id", time="t")
    shuffled = train.sample(fraction=1.0, shuffle=True, seed=99)
    b = PanelPCA(n_components=3).fit(shuffled, entity="id", time="t")

    # Sign-fixed loadings are identical regardless of fit-row order.
    assert np.allclose(a.components_, b.components_, atol=1e-8)
    assert np.array_equal(a.sign_flip_, b.sign_flip_)

    # And the projected test scores agree row-for-row.
    oa = a.transform(test, entity="id", time="t").collect()
    ob = b.transform(test, entity="id", time="t").collect()
    j = oa.join(ob, on=["id", "t"], suffix="_b")
    assert np.allclose(
        j.select("pc_1", "pc_2", "pc_3").to_numpy(),
        j.select("pc_1_b", "pc_2_b", "pc_3_b").to_numpy(),
        atol=1e-8,
    )


def test_reduce_features_convenience_runs():
    df = _panel(seed=4)
    out = reduce_features(df, method="pca", n_components=2, entity="id", time="t")
    assert isinstance(out, PanelFrame)
    cols = out.collect().columns
    assert cols == ["id", "t", "pc_1", "pc_2"]


# --------------------------------------------------------------------------- #
# (c) cross-sectional per-date components
# --------------------------------------------------------------------------- #
def test_cross_sectional_pca_is_per_date():
    df = _panel(n_entities=12, n_periods=8, seed=5)
    xs = CrossSectionalPCA(n_components=2)
    out = xs.fit_transform(df, entity="id", time="t").collect()

    assert out.columns == ["id", "t", "cspc_1", "cspc_2"]
    assert out.height == df.height
    # Every date should carry finite scores (>=2 entities per date here).
    assert out.select(pl.col("cspc_1").is_finite().all()).item()

    # Per-date locality: perturbing one date must not move another date's scores.
    scores_t3 = out.filter(pl.col("t") == 3).sort("id").select("cspc_1", "cspc_2")
    perturbed = df.with_columns(
        pl.when(pl.col("t") == 5)
        .then(pl.col("f0") + 100.0)
        .otherwise(pl.col("f0"))
        .alias("f0")
    )
    out2 = (
        CrossSectionalPCA(n_components=2)
        .fit_transform(perturbed, entity="id", time="t")
        .collect()
    )
    scores_t3_b = out2.filter(pl.col("t") == 3).sort("id").select("cspc_1", "cspc_2")
    assert np.allclose(scores_t3.to_numpy(), scores_t3_b.to_numpy(), atol=1e-10)


# --------------------------------------------------------------------------- #
# (d) statistical factors: shape, loadings, invariance, gated global fit
# --------------------------------------------------------------------------- #
def _returns_panel(n_entities: int = 8, n_periods: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_dates = n_periods
    # Common factor structure across entities.
    factors = rng.standard_normal((n_dates, 2))
    betas = rng.standard_normal((n_entities, 2))
    rows = []
    for e in range(n_entities):
        r = factors @ betas[e] + 0.1 * rng.standard_normal(n_dates)
        for t in range(n_dates):
            rows.append({"id": f"e{e}", "t": t, "ret": float(r[t])})
    return pl.DataFrame(rows)


def test_statistical_factors_returns_k_columns_and_loadings():
    df = _returns_panel(seed=7)
    sf = StatisticalFactors(returns="ret", k=2)
    out = sf.fit_transform(df, entity="id", time="t").collect()

    assert out.columns == ["id", "t", "factor_1", "factor_2"]
    assert out.height == df.height
    assert sf.loadings_.shape == (2, 8)
    assert len(sf.entity_names_) == 8
    assert sf.explained_variance_ratio_.shape == (2,)


def test_statistical_factors_invariant_to_appended_future_dates():
    df = _returns_panel(seed=8)
    train = df.filter(pl.col("t") < 20)
    test = df.filter(pl.col("t") >= 20)

    sf = StatisticalFactors(returns="ret", k=2).fit(train, entity="id", time="t")

    # A test date's factor uses only train loadings + that date's cross-section,
    # so appending later test dates must not change an earlier one.
    early = test.filter(pl.col("t") <= 24)
    full = sf.transform(test, entity="id", time="t").collect()
    part = sf.transform(early, entity="id", time="t").collect()
    j = full.join(part, on=["id", "t"], suffix="_p")
    assert j.height == part.height
    assert np.allclose(
        j.select("factor_1", "factor_2").to_numpy(),
        j.select("factor_1_p", "factor_2_p").to_numpy(),
        atol=1e-10,
    )


def test_statistical_factors_global_variant_refused_by_leakage_gate():
    df = _returns_panel(seed=9)
    train = df.filter(pl.col("t") < 20)
    test = df.filter(pl.col("t") >= 20)
    train_p = PanelFrame(train, entity="id", time="t")
    test_p = PanelFrame(test, entity="id", time="t")

    sf = StatisticalFactors(returns="ret", k=2, scope="global").fit(train_p)
    assert sf.leakage_safe is False
    with pytest.raises(RuntimeError, match="leakage_safe = False"):
        sf._check_leakage(train_p, test_p)

    # The leak-safe default is accepted across a clean, forward-only boundary.
    sf_ok = StatisticalFactors(returns="ret", k=2).fit(train_p)
    sf_ok._check_leakage(train_p, test_p)  # no raise
