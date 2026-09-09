"""Import-hygiene guard for the light 0.4.0 core.

Panelary's mandatory footprint is ``numpy + polars``.  Everything heavier is an
optional extra that must be imported *lazily inside the function that uses it*
(via :func:`panelary._internal._deps.require`).  These tests are the CI ratchet
that makes a regression loud:

* ``import panelary`` must not pull any heavy optional dependency into
  ``sys.modules``;
* neither may the eagerly-imported feature modules (``feature_extractors``,
  ``catch22``, ``reduce``, ``cluster``, ``select``);
* the cold import must stay inside a wall-clock budget, scaled by the
  machine's own speed so the guard measures Panelary and not the runner.

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
#: ``import panelary``.  Keep in sync with the playbook's Wave-0 probe.
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

#: Cold-import wall-clock budget in milliseconds, *on a reference-speed machine*.
#: Measured locally at ~78 ms on CPython 3.13 (polars itself is ~70 ms of that),
#: so 300 ms leaves ~4x headroom while still catching a real regression (e.g. an
#: accidental eager ``import sklearn``, which costs >500 ms on its own).
#: Override with ``PANELARY_IMPORT_BUDGET_MS`` for unusually slow machines.
IMPORT_BUDGET_MS = float(os.environ.get("PANELARY_IMPORT_BUDGET_MS", "300"))

#: What ``import polars`` costs on the machine ``IMPORT_BUDGET_MS`` was
#: calibrated against.  ``test_import_time_within_budget`` measures polars in the
#: same run and scales the budget by how much slower this machine is, so the
#: assertion stays about *Panelary's module-level work* rather than about the
#: runner's disk and CPU contention.  See that test for why this keeps its teeth.
POLARS_REFERENCE_MS = 70.0

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
def test_import_panelary_is_clean():
    """``import panelary`` loads no heavy optional dependency."""
    result = _probe("panelary")
    leaked = sorted(set(result["tops"]) & set(FORBIDDEN_MODULES))
    assert not leaked, (
        f"`import panelary` eagerly imported {leaked}. Optional deps must "
        "be imported lazily inside the function that uses them, via "
        "`panelary._internal._deps.require(...)`."
    )


@pytest.mark.parametrize(
    "module",
    [
        "panelary.feature_extractors",
        "panelary.catch22",
        "panelary.reduce",
        "panelary.cluster",
        "panelary.select",
        "panelary._internal._numpy_stats",
        "panelary._internal._deps",
    ],
)
def test_feature_modules_are_clean(module):
    """Individually importing a feature module stays scipy/sklearn/pandas free."""
    result = _probe(module)
    leaked = sorted(set(result["tops"]) & set(FORBIDDEN_MODULES))
    assert not leaked, (
        f"`import {module}` eagerly imported {leaked}; route it through "
        "`panelary._internal._deps.require(...)` inside the using function."
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

    Loaded straight off disk (not as ``panelary._internal._deps``) so the parent
    package's own numpy/polars imports do not mask a regression here.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    deps_path = os.path.join(repo, "panelary", "_internal", "_deps.py")
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
        "panelary/_internal/_deps.py must have no third-party imports (it is the "
        f"lazy-import gate itself); found {sorted(third_party)}."
    )


# --------------------------------------------------------------------------- #
# 2. Cold-import time budget
# --------------------------------------------------------------------------- #
def _best_import_ms(module: str, repeats: int = _TIMING_REPEATS) -> float:
    """Best-of-``repeats`` import time for ``module``, after a warm-up run.

    Two things are deliberately excluded from the number this returns, because
    neither is the module-level work the budget exists to police:

    * **Bytecode compilation.**  On a fresh CI runner the first import of a
      package compiles every ``.py`` to ``.pyc``; that showed up as a 13.5 s
      outlier in one run.  The discarded warm-up run pays that cost (and warms
      the page cache for polars' large shared object) before timing starts.
    * **Scheduler noise.**  Taking the minimum, not the mean, means one stalled
      run cannot fail the build.
    """
    _probe(module)  # warm-up: compile bytecode, populate the page cache
    return min(_probe(module)["elapsed_ms"] for _ in range(repeats))


def test_import_time_within_budget():
    """Cold ``import panelary`` stays under the wall-clock budget.

    The budget is scaled by how slow *this machine* is, measured in the same run
    by importing polars -- a fixed external package Panelary cannot influence.
    Without that, the test measures the runner as much as the library: shared CI
    runners have been seen taking 5x the local time for the identical import,
    which is a fact about the runner, not a regression.

    This keeps the guard's teeth because the two failure modes it exists to
    catch do **not** move the polars yardstick:

    * an eager ``import sklearn``/``scipy`` adds its several hundred ms to
      Panelary only, so the ratio to polars blows out and the assertion fires;
    * expensive module-level work (a registry built at import, a table
      materialised) likewise lands entirely on the Panelary side.

    Only a uniformly slow *machine* moves both numbers together, and that is
    precisely the case the scaling is meant to forgive.
    """
    polars_ms = _best_import_ms("polars")
    panelary_ms = _best_import_ms("panelary")
    slowdown = max(1.0, polars_ms / POLARS_REFERENCE_MS)
    budget = IMPORT_BUDGET_MS * slowdown
    assert panelary_ms <= budget, (
        f"`import panelary` took {panelary_ms:.0f} ms (best of "
        f"{_TIMING_REPEATS}, after a warm-up), over the {budget:.0f} ms budget "
        f"({IMPORT_BUDGET_MS:.0f} ms x {slowdown:.1f} for a machine on which "
        f"`import polars` alone takes {polars_ms:.0f} ms vs the "
        f"{POLARS_REFERENCE_MS:.0f} ms reference). Either an eager heavy import "
        "crept in or module-level work needs deferring; raise "
        "PANELARY_IMPORT_BUDGET_MS only if the machine itself is slow."
    )


def test_import_overhead_over_polars_is_small():
    """Panelary's own import cost, net of polars, stays modest.

    A machine-independent companion to the absolute budget: polars alone
    dominates the import, so the *delta* is the part Panelary controls.
    """
    polars_ms = _best_import_ms("polars")
    panelary_ms = _best_import_ms("panelary")
    overhead = panelary_ms - polars_ms
    # Locally ~8 ms; 150 ms is a generous ratchet that still catches an eager
    # sklearn/scipy import or an expensive module-level registry build.
    assert overhead <= 150.0, (
        f"Panelary adds {overhead:.0f} ms on top of polars "
        f"({panelary_ms:.0f} ms vs {polars_ms:.0f} ms) -- too much module-level work."
    )
