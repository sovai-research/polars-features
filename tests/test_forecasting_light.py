"""Lightweight import-hygiene tests for the forecasting subpackage.

These verify that ``import polars_features.forecasting`` does not force the
optional ``flaml`` / ``tqdm`` dependencies, that the progress shim iterates
identically with or without ``tqdm``, and that an ``auto_*`` forecaster can
still build its FLAML search space when ``flaml`` is installed.

Deliberately import-level only -- the slow, end-to-end ``tests/test_forecasting.py``
is not exercised here.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_forecasting_import_does_not_pull_flaml_or_tqdm():
    """A fresh interpreter importing the subpackage must not load flaml/tqdm."""
    code = (
        "import polars_features.forecasting, sys; "
        "bad = [m for m in sys.modules if m.split('.')[0] in {'flaml', 'tqdm'}]; "
        "assert not bad, bad; "
        "print('forecasting import clean')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "forecasting import clean" in proc.stdout


def test_progress_shim_iterates_transparently():
    from polars_features._progress import progress, trange

    assert list(progress([1, 2, 3])) == [1, 2, 3]
    assert list(progress(iter("ab"), desc="anything", total=2)) == ["a", "b"]
    assert list(progress([])) == []
    # trange mirrors range(...) semantics
    assert list(trange(4)) == [0, 1, 2, 3]
    assert list(trange(1, 4)) == [1, 2, 3]
    assert list(trange(1, 7, 2)) == [1, 3, 5]


def test_auto_forecaster_builds_search_space_when_flaml_present():
    pytest.importorskip("flaml")

    from polars_features.forecasting.automl import (
        auto_elastic_net,
        auto_knn,
        auto_lasso,
        auto_ridge,
    )

    for cls, expected_keys in (
        (auto_lasso, {"alpha", "fit_intercept"}),
        (auto_ridge, {"alpha", "fit_intercept"}),
        (auto_elastic_net, {"alpha", "l1_ratio", "fit_intercept"}),
        (auto_knn, {"leaf_size"}),
    ):
        space = cls(freq="1d").default_search_space
        assert space is not None
        assert set(space) == expected_keys
