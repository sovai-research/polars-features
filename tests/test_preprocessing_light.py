"""Light-import regression guards for ``polars_features.preprocessing``.

These tests protect the work that removed scikit-learn, scipy and cloudpickle
from this module's *import-time* cost. They assert that:

* the module carries no module-top ``import sklearn/scipy/cloudpickle`` (those
  are now lazy, imported inside the transforms that need them);
* importing the module does not eagerly pull ``cloudpickle`` into ``sys.modules``;
* stdlib ``pickle`` is a behavioural drop-in for the previously-used
  ``cloudpickle`` when (de)serialising the fitted regressors used by
  ``deseasonalize_fourier``;
* the scipy-backed (boxcox / yeojohnson) and sklearn-free (detrend) transforms
  still round-trip exactly.
"""

from __future__ import annotations

import ast
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import polars_features.preprocessing as pp
from polars_features.preprocessing import boxcox, detrend, yeojohnson

_MODULE_PATH = Path(pp.__file__)
_HEAVY = {"sklearn", "scipy", "cloudpickle"}


def _panel(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = 48
    rows = []
    for ent in ("A", "B"):
        base = rng.uniform(1.0, 5.0)
        for t in range(n):
            val = base + 2.0 * np.sin(2 * np.pi * t / 12) + 0.5 * t + rng.normal(0, 0.1)
            rows.append({"entity": ent, "time": t, "y": float(abs(val) + 1.0)})
    return pl.DataFrame(rows)


def test_no_module_top_heavy_imports() -> None:
    """No top-level ``import sklearn/scipy/cloudpickle`` in preprocessing.py."""
    tree = ast.parse(_MODULE_PATH.read_text())
    offenders = []
    for node in tree.body:  # module-level statements only
        if isinstance(node, ast.Import):
            offenders += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            offenders.append(node.module.split(".")[0])
    assert not (set(offenders) & _HEAVY), sorted(set(offenders) & _HEAVY)


def test_import_does_not_pull_cloudpickle() -> None:
    """Importing the module must not eagerly import cloudpickle."""
    code = (
        "import sys, polars_features.preprocessing; "
        "assert 'cloudpickle' not in sys.modules, 'cloudpickle eagerly imported'; "
        "print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_pickle_is_dropin_for_regressors() -> None:
    """stdlib pickle round-trips the fitted regressors identically."""
    sklearn_lm = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=40)
    for cls in (sklearn_lm.LinearRegression, sklearn_lm.TheilSenRegressor):
        reg = cls().fit(X=X, y=y)
        expected = reg.predict(X=X)
        restored = pickle.loads(pickle.dumps(reg))
        assert np.allclose(restored.predict(X=X), expected)


def test_boxcox_roundtrip() -> None:
    pytest.importorskip("scipy")
    df = _panel()
    key = ["entity", "time"]
    t = boxcox()
    fwd = df.lazy().pipe(t).collect().sort(key)
    inv = fwd.lazy().pipe(t.invert).collect().sort(key)
    assert np.allclose(
        inv.get_column("y").to_numpy(),
        df.sort(key).get_column("y").to_numpy(),
    )


def test_yeojohnson_roundtrip() -> None:
    pytest.importorskip("scipy")
    df = _panel()
    key = ["entity", "time"]
    t = yeojohnson()
    fwd = df.lazy().pipe(t).collect().sort(key)
    inv = fwd.lazy().pipe(t.invert).collect().sort(key)
    assert np.allclose(
        inv.get_column("y").to_numpy(),
        df.sort(key).get_column("y").to_numpy(),
    )


def test_detrend_roundtrip() -> None:
    df = _panel()
    key = ["entity", "time"]
    t = detrend(freq="1i", method="linear")
    fwd = df.lazy().pipe(t).collect().sort(key)
    inv = fwd.lazy().pipe(t.invert).collect().sort(key)
    assert np.allclose(
        inv.get_column("y").to_numpy(),
        df.sort(key).get_column("y").to_numpy(),
    )
