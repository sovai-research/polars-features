"""Import-hygiene guard for the light 0.4.0 core.

PanelKit's mandatory footprint is ``numpy + polars``.  Everything heavier is an
optional extra that must be imported *lazily inside the function that uses it*
(via :func:`polars_features._deps.require`).  These tests are the CI ratchet
that makes a regression loud:

* ``import polars_features`` must not pull any heavy optional dependency into
  ``sys.modules``;
* neither may the eagerly-imported feature modules (``feature_extractors``,
  ``catch22``, ``reduce``, ``cluster``, ``select``);
* the cold import must stay inside a wall-clock budget.

Every check runs in a **fresh subprocess** so the answer is not polluted by
whatever the pytest session itself has already imported (pytest's own conftest
imports pandas and scipy, for instance).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

#: Heavy third-party packages that must never be imported as a side effect of
#: ``import polars_features``.  Keep in sync with the playbook's Wave-0 probe.
FORBIDDEN_MODULES = (
    "scipy",
    "sklearn",
    "pandas",
    "flaml",
    "tqdm",
    "numba",
    "holidays",
    "umap",
)

#: Cold-import wall-clock budget in milliseconds.  Measured locally at ~78 ms on
#: CPython 3.13 (polars itself is ~70 ms of that), so 300 ms leaves ~4x headroom
#: for slower/cold CI runners while still catching a real regression (e.g. an
#: accidental eager ``import sklearn``, which costs >500 ms on its own).
#: Override with ``PANELKIT_IMPORT_BUDGET_MS`` for unusually slow machines.
IMPORT_BUDGET_MS = float(os.environ.get("PANELKIT_IMPORT_BUDGET_MS", "300"))

#: Number of fresh subprocesses to time; the *minimum* is used so a single
#: scheduler hiccup on a shared CI runner cannot fail the build.
_TIMING_REPEATS = 3


_PROBE = r"""
import json, sys, time

t0 = time.perf_counter()
import {module}
elapsed_ms = (time.perf_counter() - t0) * 1000.0

tops = {{name.split(".", 1)[0] for name in sys.modules}}
print(json.dumps({{"elapsed_ms": elapsed_ms, "tops": sorted(tops)}}))
"""


def _probe(module: str) -> dict:
    """Import ``module`` in a fresh interpreter; return timing + loaded top-levels."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module)],
        capture_output=True,
        text=True,
        check=False,
        # -S / -E would change import semantics; run a normal interpreter but
        # from a directory-independent cwd so the repo root is on sys.path the
        # same way an installed package would be.
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert proc.returncode == 0, (
        f"probe subprocess failed for {module!r}:\n{proc.stdout}\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# 1. No heavy dependency is imported as a side effect
# --------------------------------------------------------------------------- #
def test_import_polars_features_is_clean():
    """``import polars_features`` loads no heavy optional dependency."""
    result = _probe("polars_features")
    leaked = sorted(set(result["tops"]) & set(FORBIDDEN_MODULES))
    assert not leaked, (
        f"`import polars_features` eagerly imported {leaked}. Optional deps must "
        "be imported lazily inside the function that uses them, via "
        "`polars_features._deps.require(...)`."
    )


@pytest.mark.parametrize(
    "module",
    [
        "polars_features.feature_extractors",
        "polars_features.catch22",
        "polars_features.reduce",
        "polars_features.cluster",
        "polars_features.select",
        "polars_features._numpy_stats",
        "polars_features._deps",
    ],
)
def test_feature_modules_are_clean(module):
    """Individually importing a feature module stays scipy/sklearn/pandas free."""
    result = _probe(module)
    leaked = sorted(set(result["tops"]) & set(FORBIDDEN_MODULES))
    assert not leaked, (
        f"`import {module}` eagerly imported {leaked}; route it through "
        "`polars_features._deps.require(...)` inside the using function."
    )


_DEPS_STANDALONE_PROBE = r"""
import importlib.util, json, sys, pathlib

path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("_pk_deps_standalone", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

tops = {name.split(".", 1)[0] for name in sys.modules}
print(json.dumps(sorted(tops)))
"""


def test_deps_module_is_stdlib_only():
    """``_deps`` must not pull anything third-party -- it is the bootstrap gate.

    Loaded straight off disk (not as ``polars_features._deps``) so the parent
    package's own numpy/polars imports do not mask a regression here.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    deps_path = os.path.join(repo, "polars_features", "_deps.py")
    proc = subprocess.run(
        [sys.executable, "-c", _DEPS_STANDALONE_PROBE, deps_path],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    tops = set(json.loads(proc.stdout.strip().splitlines()[-1]))
    third_party = tops & ({"numpy", "polars"} | set(FORBIDDEN_MODULES))
    assert not third_party, (
        "polars_features/_deps.py must have no third-party imports (it is the "
        f"lazy-import gate itself); found {sorted(third_party)}."
    )


# --------------------------------------------------------------------------- #
# 2. Cold-import time budget
# --------------------------------------------------------------------------- #
def test_import_time_within_budget():
    """Cold ``import polars_features`` stays under the wall-clock budget."""
    timings = [_probe("polars_features")["elapsed_ms"] for _ in range(_TIMING_REPEATS)]
    best = min(timings)
    assert best <= IMPORT_BUDGET_MS, (
        f"`import polars_features` took {best:.0f} ms (best of {_TIMING_REPEATS}), "
        f"over the {IMPORT_BUDGET_MS:.0f} ms budget. Timings: "
        f"{[round(t) for t in timings]}. Either an eager heavy import crept in or "
        "module-level work needs deferring; raise PANELKIT_IMPORT_BUDGET_MS only "
        "if the machine itself is slow."
    )


def test_import_overhead_over_polars_is_small():
    """PanelKit's own import cost, net of polars, stays modest.

    A machine-independent companion to the absolute budget: polars alone
    dominates the import, so the *delta* is the part PanelKit controls.
    """
    polars_ms = min(_probe("polars")["elapsed_ms"] for _ in range(_TIMING_REPEATS))
    panelkit_ms = min(
        _probe("polars_features")["elapsed_ms"] for _ in range(_TIMING_REPEATS)
    )
    overhead = panelkit_ms - polars_ms
    # Locally ~8 ms; 150 ms is a generous ratchet that still catches an eager
    # sklearn/scipy import or an expensive module-level registry build.
    assert overhead <= 150.0, (
        f"PanelKit adds {overhead:.0f} ms on top of polars "
        f"({panelkit_ms:.0f} ms vs {polars_ms:.0f} ms) -- too much module-level work."
    )
