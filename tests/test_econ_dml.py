"""Tests for double machine learning and PDS-LASSO (``econ._dml``).

Acceptance: the cross-fitted DML confidence interval must be **valid on a
simulation** -- roughly nominal coverage of the true effect in a partially linear
model with a high-dimensional, sparse confounder. The tests also pin down the
leak-safety contract: folds come from Panelary's purged/embargoed splitter, and
the fitted transformer freezes its nuisance models.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.econ import (
    DoubleMLTransformer,
    dml_partial_linear,
    lasso_penalty,
    post_double_selection,
    post_lasso,
    rigorous_lasso,
)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def _pl_model(
    n: int = 800,
    p: int = 20,
    *,
    theta: float = 1.0,
    seed: int = 0,
    b: tuple[float, ...] = (1.5, 1.2, 1.0),
    g: tuple[float, ...] = (1.0, -1.0, 0.8),
) -> pl.DataFrame:
    """Partially linear model ``y = theta d + X b + u``, ``d = X g + v``.

    The same three controls confound both equations, so omitting them biases
    ``theta`` -- which is exactly what double selection / cross-fitting must fix.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[: len(b)] = b
    gamma = np.zeros(p)
    gamma[: len(g)] = g
    d = x @ gamma + rng.normal(size=n)
    y = theta * d + x @ beta + rng.normal(size=n)
    return pl.DataFrame(
        {
            "id": np.zeros(n, dtype=np.int64),
            "t": np.arange(n),
            "y": y,
            "d": d,
            **{f"x{j}": x[:, j] for j in range(p)},
        }
    )


# --------------------------------------------------------------------------- #
# Rigorous LASSO
# --------------------------------------------------------------------------- #
def test_lasso_penalty_scales_with_n_and_p():
    small = lasso_penalty(100, 10)
    more_controls = lasso_penalty(100, 1000)
    bigger_sample = lasso_penalty(10_000, 10)
    assert more_controls > small  # more candidates -> stronger penalty
    assert bigger_sample < small  # more data -> weaker penalty
    with pytest.raises(ValueError, match="must be positive"):
        lasso_penalty(0, 5)


def test_rigorous_lasso_recovers_a_sparse_support():
    rng = np.random.default_rng(1)
    n, p = 400, 60
    x = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:4] = [3.0, -2.5, 2.0, -1.5]
    y = x @ beta + 0.5 * rng.normal(size=n)
    fit = rigorous_lasso(x, y)
    assert set(range(4)).issubset(set(fit.selected.tolist()))
    assert fit.selected.size < 15  # sparse, not a kitchen sink
    # Shrinkage: the LASSO coefficients are pulled towards zero...
    assert np.all(np.abs(fit.coef[:4]) < np.abs(beta[:4]))
    # ...and post-LASSO undoes it.
    pred = post_lasso(x, y)
    np.testing.assert_allclose(pred(x), y, atol=3.0)


def test_rigorous_lasso_shrinks_pure_noise_to_zero():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 40))
    y = rng.normal(size=300)
    fit = rigorous_lasso(x, y)
    assert fit.selected.size <= 2


def test_lasso_validates_shapes():
    with pytest.raises(ValueError, match="rows"):
        rigorous_lasso(np.zeros((10, 3)), np.zeros(9))


def test_post_lasso_with_empty_support_predicts_the_mean():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 30))
    y = rng.normal(size=200) * 1e-6
    pred = post_lasso(x, y)
    out = pred(x)
    assert np.allclose(out, out[0])


# --------------------------------------------------------------------------- #
# Post-double selection
# --------------------------------------------------------------------------- #
def test_post_double_selection_recovers_the_effect():
    res = post_double_selection(_pl_model(seed=4), y="y", d="d", entity="id", time="t")
    assert abs(res.theta - 1.0) < 0.1
    assert res.conf_int[0] < 1.0 < res.conf_int[1]
    assert set(res.selected) >= {"x0", "x1", "x2"}
    assert res.summary().columns == [
        "term",
        "estimate",
        "std_error",
        "t_stat",
        "p_value",
        "ci_lower",
        "ci_upper",
        "n_selected",
    ]


def test_post_double_selection_beats_the_naive_short_regression():
    """Omitting the confounders biases the effect; double selection does not."""
    df = _pl_model(seed=5)
    y = df["y"].to_numpy()
    d = df["d"].to_numpy()
    naive = float(np.polyfit(d, y, 1)[0])
    res = post_double_selection(df, y="y", d="d", entity="id", time="t")
    assert abs(naive - 1.0) > 0.3
    assert abs(res.theta - 1.0) < abs(naive - 1.0)


def test_post_double_selection_takes_the_union_of_both_selections():
    res = post_double_selection(_pl_model(seed=6), y="y", d="d", entity="id", time="t")
    union = set(res.selected_y) | set(res.selected_d)
    assert set(res.selected) == union


def test_post_double_selection_with_no_controls():
    df = _pl_model(n=300, p=3, seed=7).drop("x0", "x1", "x2")
    res = post_double_selection(df, y="y", d="d", controls=[], entity="id", time="t")
    assert res.selected == []
    assert np.isfinite(res.theta)


# --------------------------------------------------------------------------- #
# Cross-fitted DML
# --------------------------------------------------------------------------- #
def test_dml_recovers_the_effect_and_reports_residual_features():
    df = _pl_model(seed=8)
    res = dml_partial_linear(df, y="y", d="d", entity="id", time="t", n_folds=5)
    assert abs(res.theta - 1.0) < 0.12
    assert res.conf_int[0] < 1.0 < res.conf_int[1]
    assert res.n_folds == 5
    assert res.fold_thetas.size == 5
    assert res.y_resid.shape == res.d_resid.shape == (df.height,)
    assert np.isfinite(res.y_resid).all()
    assert res.summary().height == 1


def test_dml_confidence_interval_covers_at_roughly_the_nominal_rate():
    """The M6 acceptance test: valid inference on a simulated DGP."""
    covered = 0
    thetas = []
    reps = 60
    for seed in range(reps):
        res = dml_partial_linear(
            _pl_model(seed=1000 + seed), y="y", d="d", entity="id", time="t"
        )
        thetas.append(res.theta)
        covered += res.conf_int[0] <= 1.0 <= res.conf_int[1]
    coverage = covered / reps
    assert coverage >= 0.85, f"DML coverage collapsed to {coverage:.2f}"
    assert abs(float(np.mean(thetas)) - 1.0) < 0.05


@pytest.mark.parametrize("learner", ["post-lasso", "lasso", "ridge", "ols"])
def test_dml_learners_all_run(learner):
    res = dml_partial_linear(
        _pl_model(n=400, p=10, seed=9),
        y="y",
        d="d",
        entity="id",
        time="t",
        learner=learner,
    )
    assert np.isfinite(res.theta)
    assert res.learner == learner


def test_dml_accepts_a_custom_callable_learner():
    def constant_learner(x, y):
        mean = float(y.mean())
        return lambda z: np.full(z.shape[0], mean)

    res = dml_partial_linear(
        _pl_model(n=300, p=5, seed=10),
        y="y",
        d="d",
        entity="id",
        time="t",
        learner=constant_learner,
    )
    assert np.isfinite(res.theta)


def test_dml_rejects_unknown_learners_and_fold_counts():
    df = _pl_model(n=200, p=5, seed=11)
    with pytest.raises(ValueError, match="unknown learner"):
        dml_partial_linear(df, y="y", d="d", entity="id", time="t", learner="magic")
    with pytest.raises(ValueError, match="`n_folds` must be >= 2"):
        dml_partial_linear(df, y="y", d="d", entity="id", time="t", n_folds=1)


# --------------------------------------------------------------------------- #
# Leak safety: purge / embargo
# --------------------------------------------------------------------------- #
def test_purging_and_embargo_shrink_the_scored_sample():
    """Purging an overlapping horizon must remove training rows, not test rows."""
    df = _pl_model(n=600, p=10, seed=12)
    plain = dml_partial_linear(df, y="y", d="d", entity="id", time="t", n_folds=5)
    purged = dml_partial_linear(
        df,
        y="y",
        d="d",
        entity="id",
        time="t",
        n_folds=5,
        horizon=10,
        embargo=10,
    )
    assert purged.horizon == 10 and purged.embargo == 10
    # Every row is still scored out of fold; only the *training* rows shrink.
    assert purged.n_obs == plain.n_obs == df.height
    assert abs(purged.theta - 1.0) < 0.2


def test_out_of_fold_residuals_never_come_from_an_in_fold_model():
    """A degenerate learner that memorises its training rows must leave the
    out-of-fold residuals non-zero -- proof the folds really are held out."""
    df = _pl_model(n=300, p=4, seed=13)

    def memoriser(x, y):
        keys = {tuple(np.round(row, 10)): val for row, val in zip(x, y, strict=True)}

        def _predict(z):
            return np.array(
                [keys.get(tuple(np.round(row, 10)), 0.0) for row in z], dtype=float
            )

        return _predict

    res = dml_partial_linear(df, y="y", d="d", entity="id", time="t", learner=memoriser)
    # If any fold had seen its own test rows the residual would be exactly zero.
    assert np.abs(res.d_resid).min() > 0.0


def test_dml_raises_when_the_treatment_is_perfectly_explained():
    df = _pl_model(n=300, p=5, seed=14)
    df = df.with_columns((pl.col("x0") * 2.0).alias("d"))
    with pytest.raises(ValueError, match="not identified|no variation"):
        dml_partial_linear(df, y="y", d="d", entity="id", time="t", learner="ols")


# --------------------------------------------------------------------------- #
# Transformer
# --------------------------------------------------------------------------- #
def test_transformer_freezes_nuisance_models_and_theta():
    df = _pl_model(n=600, p=10, seed=15)
    train = df.filter(pl.col("t") < 300)
    tr = DoubleMLTransformer(y="y", d="d").fit(train, entity="id", time="t")
    assert DoubleMLTransformer.panel_safe is True
    assert DoubleMLTransformer.leakage_safe is True
    theta = tr.theta_
    out_train = tr.transform(train).collect().sort("t")
    out_full = tr.transform(df).collect().sort("t")
    assert tr.theta_ == theta  # transform never refits
    assert set(tr.output_names).issubset(out_train.columns)
    np.testing.assert_allclose(
        out_train["dml_d_resid"].to_numpy(),
        out_full.filter(pl.col("t") < 300)["dml_d_resid"].to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        out_train["dml_effect"].to_numpy(),
        theta * out_train["dml_d_resid"].to_numpy(),
        atol=1e-12,
    )


def test_transformer_works_without_the_target_column():
    df = _pl_model(n=400, p=6, seed=16)
    tr = DoubleMLTransformer(y="y", d="d").fit(df, entity="id", time="t")
    out = tr.transform(df.drop("y")).collect()
    assert "dml_d_resid" in out.columns
    assert "dml_y_resid" not in out.columns


def test_transformer_requires_fitting_and_its_controls():
    tr = DoubleMLTransformer(y="y", d="d")
    df = _pl_model(n=200, p=4, seed=17)
    with pytest.raises(RuntimeError, match="not fitted"):
        tr.transform(df, entity="id", time="t")
    tr.fit(df, entity="id", time="t")
    with pytest.raises(ValueError, match="are missing"):
        tr.transform(df.drop("x0"), entity="id", time="t")
