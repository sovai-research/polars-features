"""Attribution stability: reference-induced oscillation vs genuine regime drift."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_features.explain import (
    TreeAttributor,
    attribution_drift,
    attribution_stability,
    background_sensitivity,
)

FEATURES = ["stable", "drifting", "constant"]


class LinearStub:
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


def _panel(n_entities, times, seed, drift=None):
    rng = np.random.default_rng(seed)
    ids, ts, stable, drifting, constant = [], [], [], [], []
    for k, t in enumerate(times):
        shift = 0.0 if drift is None else drift * k / max(1, len(times) - 1)
        for e in range(n_entities):
            ids.append(f"e{e}")
            ts.append(t)
            stable.append(float(rng.normal()))
            drifting.append(float(rng.normal() + shift))
            constant.append(1.0)
    return pl.DataFrame(
        {
            "id": ids,
            "t": ts,
            "stable": stable,
            "drifting": drifting,
            "constant": constant,
        }
    )


@pytest.fixture
def train():
    return _panel(20, list(range(20)), seed=1)


@pytest.fixture
def future():
    # A genuine regime shift lives only in `drifting`, on the explained rows.
    return _panel(20, list(range(100, 120)), seed=2, drift=10.0)


def _attr(train, **kw):
    kw.setdefault("max_samples", 10)
    kw.setdefault("seed", 0)
    return TreeAttributor(
        LinearStub([1.0, 1.0, 1.0]),
        features=FEATURES,
        entity="id",
        time="t",
        **kw,
    ).fit(train)


# --------------------------------------------------------------------------- #
# Reference sensitivity
# --------------------------------------------------------------------------- #
def test_background_sensitivity_reports_one_row_per_feature(train, future):
    rep = background_sensitivity(_attr(train), future, seeds=(0, 1, 2, 3, 4, 5))
    assert rep.columns == ["feature", "reference_sd", "reference_sd_of_mean"]
    assert rep.get_column("feature").to_list() == FEATURES
    assert (rep.get_column("reference_sd") >= 0).all()


def test_reference_noise_is_positive_for_a_subsampled_background(train, future):
    rep = background_sensitivity(
        _attr(train, max_samples=8), future, seeds=(0, 1, 2, 3, 4, 5)
    )
    assert rep.filter(pl.col("feature") == "stable")["reference_sd"].item() > 0


def test_reference_noise_vanishes_when_the_background_is_not_subsampled(train, future):
    rep = background_sensitivity(
        _attr(train, max_samples=10_000), future, seeds=(0, 1, 2, 3)
    )
    np.testing.assert_allclose(rep.get_column("reference_sd").to_numpy(), 0.0)


def test_background_sensitivity_needs_a_fitted_attributor(train, future):
    attr = TreeAttributor(
        LinearStub([1.0, 1.0, 1.0]), features=FEATURES, entity="id", time="t"
    )
    with pytest.raises(RuntimeError, match="fitted"):
        background_sensitivity(attr, future)


def test_background_sensitivity_needs_at_least_two_seeds(train, future):
    with pytest.raises(ValueError, match="at least two"):
        background_sensitivity(_attr(train), future, seeds=(0,))


# --------------------------------------------------------------------------- #
# Temporal drift
# --------------------------------------------------------------------------- #
def test_attribution_drift_isolates_the_shifted_feature(train, future):
    wide = _attr(train).attributions(future)
    drift = attribution_drift(wide, entity="id", time="t")
    assert drift.columns == [
        "feature",
        "mean_abs_attribution",
        "temporal_sd",
        "first_last_shift",
    ]
    by = {r["feature"]: r for r in drift.to_dicts()}
    assert by["drifting"]["temporal_sd"] > 5 * by["stable"]["temporal_sd"]
    assert by["drifting"]["first_last_shift"] > 5.0
    assert by["constant"]["temporal_sd"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# The combined verdict
# --------------------------------------------------------------------------- #
def test_attribution_stability_separates_drift_from_reference_noise(train, future):
    rep = attribution_stability(
        _attr(train, max_samples=8), future, seeds=(0, 1, 2, 3, 4, 5), threshold=2.0
    )
    assert rep.columns == [
        "feature",
        "mean_abs_attribution",
        "reference_sd",
        "temporal_sd",
        "drift_ratio",
        "verdict",
    ]
    verdicts = {r["feature"]: r["verdict"] for r in rep.to_dicts()}
    assert verdicts["drifting"] == "regime-drift"
    assert verdicts["stable"] == "reference-noise"
    assert verdicts["constant"] == "stable"


def test_drift_ratio_is_scale_free(train, future):
    """Doubling the model weights must not change the diagnosis."""
    a = attribution_stability(
        _attr(train, max_samples=8), future, seeds=(0, 1, 2, 3, 4, 5)
    )
    big = TreeAttributor(
        LinearStub([10.0, 10.0, 10.0]),
        features=FEATURES,
        entity="id",
        time="t",
        max_samples=8,
    ).fit(train)
    b = attribution_stability(big, future, seeds=(0, 1, 2, 3, 4, 5))
    np.testing.assert_allclose(
        a.get_column("drift_ratio").to_numpy(),
        b.get_column("drift_ratio").to_numpy(),
        rtol=1e-9,
    )
    assert a.get_column("verdict").to_list() == b.get_column("verdict").to_list()


def test_conditional_mode_warns_that_reference_noise_is_structurally_zero(
    train, future
):
    """The path-dependent kernel ignores the reference set -- say so, loudly."""
    attr = _attr(train, mode="conditional")
    with pytest.warns(UserWarning, match="structurally zero"):
        rep = background_sensitivity(attr, future, seeds=(0, 1, 2))
    np.testing.assert_allclose(
        rep.get_column("reference_sd").to_numpy(), 0.0, atol=1e-12
    )


def test_zero_reference_noise_reports_an_infinite_ratio(train, future):
    attr = _attr(train, mode="conditional")
    with pytest.warns(UserWarning):
        rep = attribution_stability(attr, future, seeds=(0, 1, 2))
    ratios = {r["feature"]: r["drift_ratio"] for r in rep.to_dicts()}
    assert np.isinf(ratios["drifting"])
    assert ratios["constant"] == 0.0
