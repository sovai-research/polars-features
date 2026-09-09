"""Tests for the ICA sibling of the factor family.

(a) FastICA separates independent non-Gaussian sources from a linear mixture --
    each recovered component matches exactly one true source, and it does so
    better than PCA, which can only decorrelate;
(b) the estimator honours the shared API (shapes, names, `keep=`, determinism);
(c) ``scikit-learn`` is imported **lazily** -- importing
    :mod:`panelary.reduce` must not pull it in.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from panelary.reduce import ICAFactors, ica_factors
from panelary.reduce._estimators import pca_factors

pytest.importorskip("sklearn", reason="ICA needs the optional 'ml' extra")


def _mixed_sources(seed: int = 0, n: int = 2000, p: int = 6):
    """Two independent, strongly non-Gaussian sources, linearly mixed."""
    rng = np.random.default_rng(seed)
    s1 = rng.standard_exponential(n) - 1.0  # skewed
    s2 = np.sign(rng.standard_normal(n)) * rng.uniform(0.5, 1.5, n)  # bimodal
    S = np.column_stack([s1 / s1.std(), s2 / s2.std()])
    A = rng.standard_normal((2, p))
    X = S @ A + 0.05 * rng.standard_normal((n, p))
    return X, S


def _match_score(F: np.ndarray, S: np.ndarray) -> float:
    """Best one-to-one |correlation| matching between recovered and true sources.

    With only two components the permutation search is a two-way comparison.
    """
    corr = np.abs(np.corrcoef(F.T, S.T)[: F.shape[1], F.shape[1] :])
    straight = corr[0, 0] + corr[1, 1]
    swapped = corr[0, 1] + corr[1, 0]
    return float(max(straight, swapped) / 2.0)


def _panel_frame(X: np.ndarray, n_entities: int = 10) -> pl.DataFrame:
    n, p = X.shape
    per = n // n_entities
    data = {
        "id": np.repeat([f"e{i}" for i in range(n_entities)], per),
        "t": np.tile(np.arange(per), n_entities),
    }
    for j in range(p):
        data[f"x{j}"] = X[: n_entities * per, j]
    return pl.DataFrame(data)


# --------------------------------------------------------------------------- #
# (a) separation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(3))
def test_ica_recovers_the_independent_sources(seed):
    X, S = _mixed_sources(seed)
    F, _u, _extra = ica_factors(X, 2, random_state=0)
    assert _match_score(F, S) > 0.95


@pytest.mark.parametrize("seed", range(3))
def test_ica_separates_better_than_pca(seed):
    X, S = _mixed_sources(seed)
    F_ica, _u, _ = ica_factors(X, 2, random_state=0)
    F_pca, _u2, _ = pca_factors(X, 2)
    assert _match_score(F_ica, S) > _match_score(F_pca, S)


# --------------------------------------------------------------------------- #
# (b) shared API
# --------------------------------------------------------------------------- #
def test_estimator_shapes_names_and_keep():
    X, _S = _mixed_sources(0, n=1000)
    df = _panel_frame(X)
    ext = ICAFactors(2, entity="id", time="t").fit(df)
    assert ext.loadings_.shape == (6, 2)
    assert ext.get_feature_names_out() == ["factor_1", "factor_2"]

    appended = ext.transform(df).collect()
    assert set(df.columns).issubset(appended.columns)

    only = (
        ICAFactors(2, keep="factors", entity="id", time="t").fit_transform(df).collect()
    )
    assert only.columns == ["id", "t", "factor_1", "factor_2"]


def test_estimator_is_deterministic_given_a_seed():
    X, _S = _mixed_sources(0, n=1000)
    df = _panel_frame(X)
    a = ICAFactors(2, random_state=7, entity="id", time="t").fit(df)
    b = ICAFactors(2, random_state=7, entity="id", time="t").fit(df)
    assert np.array_equal(a.loadings_, b.loadings_)


def test_auto_n_factors_resolves_via_bai_ng():
    X, _S = _mixed_sources(0, n=1500, p=30)
    ext = ICAFactors(max_factors=3, entity="id", time="t").fit(_panel_frame(X))
    assert 1 <= ext.n_factors_ <= 3


def test_declares_the_leakage_contract():
    assert ICAFactors.panel_safe is True
    assert ICAFactors.leakage_safe is True


def test_functional_core_validates_r():
    X, _S = _mixed_sources(0, n=400)
    with pytest.raises(ValueError, match="positive integer"):
        ica_factors(X, 0)


# --------------------------------------------------------------------------- #
# (c) sklearn stays lazy
# --------------------------------------------------------------------------- #
def test_importing_reduce_does_not_import_sklearn():
    code = (
        "import sys; "
        "import panelary.reduce as r; "
        "assert 'sklearn' not in sys.modules, 'sklearn was imported eagerly'; "
        "assert 'scipy' not in sys.modules, 'scipy was imported eagerly'; "
        "assert hasattr(r, 'ICAFactors')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
