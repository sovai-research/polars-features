"""``shapiq`` interop (v3): any-order Shapley interactions, panel-keyed.

``shapiq`` is an **optional** dependency (``pip install
'panelary[explain]'``). The interop tests are guarded with
``pytest.importorskip``; the argument contract, the missing-dependency message
and the pure-Polars matrix helper are tested unconditionally.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary._internal._deps import have
from panelary.explain import (
    TimeAwareBackground,
    interaction_matrix,
    interaction_values,
)

FEATURES = ["x0", "x1", "x2"]


class LinearStub:
    def __init__(self, weights, c=0.0):
        self.w = np.asarray(weights, dtype=float)
        self.c = float(c)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.c * X[:, 0] * X[:, 1]

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


def make_panel(n_entities=3, n_times=12, seed=0, start_time=0):
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
    return make_panel(seed=2, n_times=2, start_time=500)


def _bg(train, **kw):
    kw.setdefault("max_samples", 20)
    return TimeAwareBackground(**kw).fit(
        train, features=FEATURES, entity="id", time="t"
    )


# --------------------------------------------------------------------------- #
# Argument contract (no shapiq needed)
# --------------------------------------------------------------------------- #
def test_max_order_must_be_at_least_one(train, future):
    with pytest.raises(ValueError, match="max_order"):
        interaction_values(
            LinearStub([1.0, 1.0, 1.0]),
            future,
            background=_bg(train),
            max_order=0,
            entity="id",
            time="t",
        )


def test_background_must_be_time_aware(train, future):
    with pytest.raises(TypeError, match="TimeAwareBackground"):
        interaction_values(
            LinearStub([1.0, 1.0, 1.0]),
            future,
            background=None,
            entity="id",
            time="t",
        )


def test_background_must_be_fitted(train, future):
    with pytest.raises(RuntimeError, match="not fitted"):
        interaction_values(
            LinearStub([1.0, 1.0, 1.0]),
            future,
            background=TimeAwareBackground(),
            entity="id",
            time="t",
        )


@pytest.mark.skipif(have("shapiq"), reason="`shapiq` is installed")
def test_missing_shapiq_is_an_actionable_error(train, future):
    with pytest.raises(ImportError, match=r"panelary\[explain\]"):
        interaction_values(
            LinearStub([1.0, 1.0, 1.0]),
            future,
            background=_bg(train),
            features=FEATURES,
            entity="id",
            time="t",
        )


# --------------------------------------------------------------------------- #
# The pure-Polars matrix helper (no shapiq needed)
# --------------------------------------------------------------------------- #
def _fake_long():
    return pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b"],
            "t": [1, 1, 1, 2, 2, 2],
            "order": [1, 2, 2, 1, 2, 2],
            "features": ["x0", "x0|x1", "x1|x2", "x0", "x0|x1", "x1|x2"],
            "interaction_value": [5.0, 2.0, -4.0, 5.0, 4.0, -8.0],
        }
    )


def test_interaction_matrix_is_symmetric_and_pools_pairs():
    mat = interaction_matrix(_fake_long(), features=FEATURES)
    assert mat.get_column("feature").to_list() == FEATURES
    arr = mat.select(FEATURES).to_numpy()
    np.testing.assert_allclose(arr, arr.T)
    # mean |value| over the two rows: |2| and |4| -> 3
    assert arr[0, 1] == pytest.approx(3.0)
    assert arr[1, 2] == pytest.approx(6.0)
    assert arr[0, 2] == pytest.approx(0.0)
    # first-order rows are ignored by the pairwise layout
    assert arr[0, 0] == pytest.approx(0.0)


def test_interaction_matrix_signed_mode_keeps_cancellation():
    mat = interaction_matrix(_fake_long(), features=FEATURES, agg="mean")
    arr = mat.select(FEATURES).to_numpy()
    assert arr[1, 2] == pytest.approx(-6.0)


def test_interaction_matrix_rejects_non_pairwise_orders():
    with pytest.raises(ValueError, match="pairwise"):
        interaction_matrix(_fake_long(), order=3)
    with pytest.raises(ValueError, match="agg"):
        interaction_matrix(_fake_long(), agg="nope")


# --------------------------------------------------------------------------- #
# Real shapiq interop
# --------------------------------------------------------------------------- #
def test_shapiq_interop_returns_panel_keyed_k_sii(train, future):
    pytest.importorskip("shapiq")
    out = interaction_values(
        LinearStub([1.0, -2.0, 0.5], c=3.0),
        future,
        background=_bg(train),
        features=FEATURES,
        entity="id",
        time="t",
        max_order=2,
        max_rows=2,
    )
    assert {"id", "t", "order", "features", "interaction_value"} <= set(out.columns)
    assert set(out.get_column("order").unique().to_list()) <= {1, 2}
    assert out.height > 0


def test_shapiq_pairwise_values_feed_the_matrix_helper(train, future):
    pytest.importorskip("shapiq")
    model = LinearStub([1.0, -2.0, 0.5], c=3.0)
    out = interaction_values(
        model,
        future,
        background=_bg(train, max_samples=32),
        features=FEATURES,
        entity="id",
        time="t",
        max_order=2,
        max_rows=2,
    )
    pairs = out.filter(pl.col("order") == 2)
    assert pairs.height > 0
    mat = interaction_matrix(out, features=FEATURES)
    arr = mat.select(FEATURES).to_numpy()
    np.testing.assert_allclose(arr, arr.T)
    # The only interaction in the data-generating process is x0 x x1.
    assert arr[0, 1] >= arr[0, 2]
