"""Tests for Diebold-Yilmaz connectedness (``econ._connectedness``).

The acceptance criteria are the **structural identities** of the spillover
table, which any correct generalised-FEVD implementation must satisfy exactly:

* every row of the normalised FEVD table sums to 100%;
* total connectedness lies in ``[0, 100]``;
* independent series produce exactly zero spillovers;
* net directional connectedness sums to zero across entities, and the net
  pairwise matrix is antisymmetric.

Together these pin the implementation down without needing the GPL R reference.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_features.econ import (
    ConnectednessFeatures,
    connectedness,
    generalized_fevd,
    ma_coefficients,
    rolling_connectedness,
    var_ols,
)
from polars_features.econ._connectedness import connectedness_from_var

# --------------------------------------------------------------------------- #
# Simulation helpers
# --------------------------------------------------------------------------- #
_A_CONNECTED = np.array(
    [
        [
            [0.40, 0.30, 0.00, 0.00],
            [0.00, 0.40, 0.20, 0.00],
            [0.10, 0.00, 0.40, 0.00],
            [0.00, 0.00, 0.30, 0.40],
        ]
    ]
)


def _simulate_var(
    coefs: np.ndarray, n: int, *, seed: int = 0, sigma=None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p, k, _ = coefs.shape
    chol = np.eye(k) if sigma is None else np.linalg.cholesky(sigma)
    y = np.zeros((n + 50, k))
    for t in range(p, n + 50):
        acc = np.zeros(k)
        for i in range(p):
            acc += coefs[i] @ y[t - i - 1]
        y[t] = acc + chol @ rng.normal(size=k)
    return y[50:]


def _panel_from_matrix(mat: np.ndarray) -> pl.DataFrame:
    n_times, k = mat.shape
    return pl.DataFrame(
        {
            "id": np.repeat(np.arange(k), n_times),
            "t": np.tile(np.arange(n_times), k),
            "v": mat.T.reshape(-1),
        }
    )


# --------------------------------------------------------------------------- #
# VAR / MA backend
# --------------------------------------------------------------------------- #
def test_var_ols_recovers_known_coefficients():
    y = _simulate_var(_A_CONNECTED, 4000, seed=1)
    coefs, sigma, resid = var_ols(y, lags=1)
    np.testing.assert_allclose(coefs[0], _A_CONNECTED[0], atol=0.05)
    np.testing.assert_allclose(sigma, np.eye(4), atol=0.08)
    assert resid.shape == (y.shape[0] - 1, 4)


def test_ma_coefficients_follow_the_wold_recursion():
    a = np.array([[[0.5, 0.2], [0.0, 0.3]]])
    psi = ma_coefficients(a, 4)
    np.testing.assert_allclose(psi[0], np.eye(2))
    np.testing.assert_allclose(psi[1], a[0])
    np.testing.assert_allclose(psi[2], a[0] @ a[0])
    np.testing.assert_allclose(psi[3], a[0] @ a[0] @ a[0])


def test_var_ols_rejects_impossible_configurations():
    with pytest.raises(ValueError, match="`lags` must be >= 1"):
        var_ols(np.zeros((10, 2)), lags=0)
    with pytest.raises(ValueError, match="usable observations"):
        var_ols(np.zeros((4, 3)), lags=1)
    with pytest.raises(ValueError, match=r"\(T, K\) matrix"):
        var_ols(np.zeros(10), lags=1)


# --------------------------------------------------------------------------- #
# Structural identities
# --------------------------------------------------------------------------- #
def test_fevd_rows_sum_to_one():
    coefs, sigma, _ = var_ols(_simulate_var(_A_CONNECTED, 600, seed=2), lags=1)
    table = generalized_fevd(coefs, sigma, horizon=10)
    np.testing.assert_allclose(table.sum(axis=1), np.ones(4), atol=1e-12)
    assert (table >= 0).all()


def test_spillover_table_rows_sum_to_100_percent():
    res = connectedness(_simulate_var(_A_CONNECTED, 600, seed=3), lags=1, horizon=10)
    np.testing.assert_allclose(res.table.sum(axis=1), np.full(4, 100.0), atol=1e-10)


@pytest.mark.parametrize("horizon", [1, 5, 10, 25])
def test_total_connectedness_is_a_percentage(horizon):
    res = connectedness(
        _simulate_var(_A_CONNECTED, 600, seed=4), lags=1, horizon=horizon
    )
    assert 0.0 <= res.total <= 100.0


def test_independent_series_have_exactly_zero_spillover():
    """A diagonal VAR with a diagonal covariance must give a diagonal table."""
    k = 4
    coefs = np.zeros((1, k, k))
    np.fill_diagonal(coefs[0], 0.5)
    sigma = np.diag([1.0, 2.0, 0.5, 1.5])
    res = connectedness_from_var(coefs, sigma, horizon=10)
    off_diagonal = res.table - np.diag(np.diag(res.table))
    assert np.abs(off_diagonal).max() == 0.0
    assert res.total == 0.0
    np.testing.assert_allclose(res.to_others, np.zeros(k))
    np.testing.assert_allclose(res.from_others, np.zeros(k))


def test_estimated_independent_series_have_near_zero_spillover():
    rng = np.random.default_rng(5)
    y = rng.normal(size=(3000, 4))
    res = connectedness(y, lags=1, horizon=10)
    assert res.total < 3.0  # only estimation noise


def test_net_connectedness_sums_to_zero_and_pairwise_is_antisymmetric():
    res = connectedness(_simulate_var(_A_CONNECTED, 800, seed=6), lags=1, horizon=10)
    assert abs(res.net.sum()) < 1e-10
    np.testing.assert_allclose(res.net_pairwise, -res.net_pairwise.T, atol=1e-12)
    # Net = TO - FROM, and the net pairwise row sums reproduce it up to sign.
    np.testing.assert_allclose(res.net, res.to_others - res.from_others, atol=1e-12)


def test_a_pure_transmitter_has_positive_net_connectedness():
    """Series 0 drives series 1 and 2 but nothing drives series 0."""
    coefs = np.array([[[0.5, 0.0, 0.0], [0.6, 0.2, 0.0], [0.6, 0.0, 0.2]]])
    res = connectedness_from_var(coefs, np.eye(3), horizon=12)
    assert res.net[0] > 0
    assert res.net[1] < 0 and res.net[2] < 0
    assert res.from_others[0] == pytest.approx(0.0, abs=1e-10)


def test_result_frames():
    res = connectedness(
        _simulate_var(_A_CONNECTED, 400, seed=7),
        names=["a", "b", "c", "d"],
        lags=1,
        horizon=8,
    )
    tidy = res.to_frame()
    assert tidy.columns == ["entity", "to_others", "from_others", "net"]
    assert tidy["entity"].to_list() == ["a", "b", "c", "d"]
    table = res.spillover_table()
    assert table.columns == ["from\\to", "a", "b", "c", "d"]


# --------------------------------------------------------------------------- #
# Rolling / panel interface
# --------------------------------------------------------------------------- #
def test_rolling_connectedness_is_backward_looking():
    mat = _simulate_var(_A_CONNECTED, 260, seed=8)
    panel = _panel_from_matrix(mat)
    full = rolling_connectedness(
        panel, value="v", window=120, lags=1, horizon=8, entity="id", time="t"
    )
    truncated = rolling_connectedness(
        panel.filter(pl.col("t") < 200),
        value="v",
        window=120,
        lags=1,
        horizon=8,
        entity="id",
        time="t",
    )
    common = full.filter(pl.col("t") < 200).sort("id", "t")
    np.testing.assert_allclose(
        common["dy_net"].to_numpy(),
        truncated.sort("id", "t")["dy_net"].to_numpy(),
        atol=1e-10,
    )


def test_rolling_connectedness_emits_one_row_per_entity_and_date():
    mat = _simulate_var(_A_CONNECTED, 200, seed=9)
    feats = rolling_connectedness(
        _panel_from_matrix(mat),
        value="v",
        window=100,
        lags=1,
        horizon=6,
        entity="id",
        time="t",
    )
    assert feats.columns == ["id", "t", "dy_to", "dy_from", "dy_net", "dy_total"]
    n_dates = feats["t"].n_unique()
    assert feats.height == n_dates * 4
    assert n_dates == 200 - 100 + 1
    assert ((feats["dy_total"] >= 0) & (feats["dy_total"] <= 100)).all()
    # Net still sums to zero within each date.
    per_date = feats.group_by("t").agg(pl.col("dy_net").sum().alias("s"))
    assert np.abs(per_date["s"].to_numpy()).max() < 1e-8


def test_rolling_connectedness_rejects_a_too_short_window():
    panel = _panel_from_matrix(_simulate_var(_A_CONNECTED, 100, seed=10))
    with pytest.raises(ValueError, match="too short"):
        rolling_connectedness(panel, value="v", window=5, lags=1, entity="id", time="t")


def test_connectedness_features_transformer():
    mat = _simulate_var(_A_CONNECTED, 200, seed=11)
    panel = _panel_from_matrix(mat)
    tr = ConnectednessFeatures(value="v", window=100, lags=1, horizon=6).fit(
        panel, entity="id", time="t"
    )
    assert ConnectednessFeatures.panel_safe is True
    assert ConnectednessFeatures.leakage_safe is True
    out = tr.transform(panel).collect()
    assert set(tr.output_names).issubset(out.columns)
    assert out.height == panel.height
    # Warm-up dates have no network yet.
    warm = out.filter(pl.col("t") < 99)
    assert warm["dy_net"].null_count() == warm.height
    assert out.filter(pl.col("t") >= 99)["dy_net"].null_count() == 0


def test_connectedness_features_rejects_a_missing_column():
    panel = _panel_from_matrix(_simulate_var(_A_CONNECTED, 150, seed=12))
    tr = ConnectednessFeatures(value="missing", window=100)
    with pytest.raises(ValueError, match="not found"):
        tr.fit(panel, entity="id", time="t")
