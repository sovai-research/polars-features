"""``panelary.leakage`` is a first-class public subpackage — and stays cheap.

Three obligations are asserted here, in order of how quietly each would break:

1. **Reachability.** ``pn.leakage`` resolves and is advertised in
   ``pn.__all__``. ``panelary.detect`` shipped in 0.4.0 unreachable because
   nobody wired it into ``panelary/__init__.py``; ``tests/test_public_api.py``
   turned that class of gap into a build failure, and this file is the
   subpackage-specific half of the same guard.
2. **The re-exports are real.** Every name in ``panelary.leakage.__all__``
   resolves on the package, so a rename in ``_compile.py`` / ``_borrowed.py`` /
   ``_types.py`` cannot leave a dangling export behind.
3. **The import stays free.** The subpackage is ``numpy + polars`` only, which
   is the entire reason it is eager rather than in ``_LAZY_SUBMODULES``. The
   assertions mirror ``tests/test_import_hygiene.py``: a fresh subprocess, and
   no heavy optional dependency in ``sys.modules`` afterwards.
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

import panelary as pn
from panelary import leakage

#: Optional dependencies that must not appear as a side effect of importing
#: either ``panelary`` or ``panelary.leakage``. A subset of
#: ``tests/test_import_hygiene.py::FORBIDDEN_MODULES``, plus ``plotly``, which
#: the light-core guarantee also covers.
FORBIDDEN_MODULES = ("scipy", "sklearn", "pandas", "plotly", "numba", "statsmodels")

_LEAKAGE_ROOT = pathlib.Path(leakage.__file__).parent
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

_PROBE = r"""
import json, sys
import {module}
print(json.dumps(sorted({{name.split(".", 1)[0] for name in sys.modules}})))
"""


def _top_level_modules_after_importing(module: str) -> set[str]:
    """Import ``module`` in a fresh interpreter; return the top-levels it loaded."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module)],
        capture_output=True,
        text=True,
        check=False,
        cwd=os.fspath(_REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"probe subprocess failed for {module!r}:\n{proc.stdout}\n{proc.stderr}"
    )
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


# --------------------------------------------------------------------------- #
# 1. Reachable and advertised
# --------------------------------------------------------------------------- #
def test_leakage_is_reachable() -> None:
    """``pn.leakage`` is the module, not something shadowing it."""
    assert hasattr(pn, "leakage")
    assert pn.leakage is importlib.import_module("panelary.leakage")


def test_leakage_is_advertised() -> None:
    """It is in ``__all__``, so ``from panelary import *`` brings it along."""
    assert "leakage" in pn.__all__


def test_leakage_is_eager_not_lazy() -> None:
    """It costs ~0 ms and pulls no extra, so it must not be deferred."""
    assert "leakage" not in pn._LAZY_SUBMODULES


def test_leakage_does_not_shadow_a_verb() -> None:
    """The name was chosen to avoid the collision `causal` would have caused."""
    assert callable(pn.causal), "the `causal` verb must still be the verb"
    assert "leakage" not in {
        "bubbles",
        "causal",
        "cluster",
        "features",
        "impute",
        "reduce",
        "regression",
        "select",
    }


# --------------------------------------------------------------------------- #
# 2. The re-exports resolve
# --------------------------------------------------------------------------- #
def test_all_is_sorted_and_unique() -> None:
    """``__all__`` is a sorted set, like every other subpackage's."""
    names = list(leakage.__all__)
    assert names == sorted(names), "panelary.leakage.__all__ should be sorted"
    assert len(names) == len(set(names)), "duplicate name in panelary.leakage.__all__"


@pytest.mark.parametrize("name", sorted(leakage.__all__))
def test_exported_name_resolves(name: str) -> None:
    """Every advertised name is actually there."""
    assert hasattr(leakage, name), (
        f"panelary.leakage.__all__ advertises {name!r}, which does not resolve. "
        "Either the implementation module renamed it or __init__.py invented it."
    )


def test_headline_symbols_are_exported() -> None:
    """The package's reason to exist is reachable without a deep import."""
    for name in ("audit", "causalize", "borrowed_accuracy", "LeakageRefused"):
        assert name in leakage.__all__, f"{name!r} missing from panelary.leakage"


def test_star_import_matches_all() -> None:
    """``from panelary.leakage import *`` yields exactly ``__all__``."""
    namespace: dict[str, object] = {}
    exec("from panelary.leakage import *", namespace)  # noqa: S102
    exported = {k for k in namespace if not k.startswith("__")}
    assert exported == set(leakage.__all__)


# --------------------------------------------------------------------------- #
# 3. Import cost is unchanged
# --------------------------------------------------------------------------- #
def test_import_panelary_pulls_no_optional_dependency() -> None:
    """Wiring `leakage` in eagerly must not cost the light core anything."""
    leaked = sorted(
        _top_level_modules_after_importing("panelary") & set(FORBIDDEN_MODULES)
    )
    assert not leaked, (
        f"`import panelary` now eagerly imports {leaked}. `panelary.leakage` is "
        "numpy + polars only; route anything heavier through "
        "`panelary._internal._deps.require(...)` inside the using function."
    )


def test_import_leakage_alone_pulls_no_optional_dependency() -> None:
    """The subpackage is clean on its own, not merely clean in context."""
    leaked = sorted(
        _top_level_modules_after_importing("panelary.leakage") & set(FORBIDDEN_MODULES)
    )
    assert not leaked, f"`import panelary.leakage` eagerly imported {leaked}"


def test_leakage_imports_no_optional_dependency_at_module_level() -> None:
    """Static half of the same guarantee: no top-level `import scipy` etc."""
    offenders = []
    for path in sorted(_LEAKAGE_ROOT.glob("*.py")):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if line[:1].isspace() or not line.startswith(("import ", "from ")):
                continue  # indented -> inside a function body, which is the rule
            for dep in FORBIDDEN_MODULES:
                if line.startswith((f"import {dep}", f"from {dep}")):
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "module-level optional import in panelary/leakage: " + "; ".join(offenders)
    )


# --------------------------------------------------------------------------- #
# 4. The filesystem-derived public surface is exactly the package
# --------------------------------------------------------------------------- #
def test_only_the_package_itself_is_public() -> None:
    """Implementation modules are underscore-prefixed, per the layout contract.

    ``tests/test_public_api.py`` derives the public surface from the
    filesystem, so a public-named file here would create a public API
    obligation nobody decided to take on.
    """
    public_named = [
        p.name
        for p in sorted(_LEAKAGE_ROOT.glob("*.py"))
        if p.name != "__init__.py" and not p.name.startswith("_")
    ]
    assert not public_named, (
        f"panelary/leakage/ has public-named modules {public_named}; the layout "
        "contract wants implementation behind `_*.py` and re-exported from "
        "__init__.py."
    )


def test_satisfies_the_filesystem_derived_public_api_check() -> None:
    """`leakage` passes `tests/test_public_api.py`'s rule, not just this file."""
    assert (_LEAKAGE_ROOT / "__init__.py").exists()
    assert not _LEAKAGE_ROOT.name.startswith(("_", "."))
    # A non-underscore directory with an __init__.py is REQUIRED public API:
    # reachable, and advertised in __all__ or declared lazy.
    assert hasattr(pn, "leakage")
    assert "leakage" in pn.__all__ or "leakage" in pn._LAZY_SUBMODULES
