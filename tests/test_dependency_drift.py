"""Dependency-drift guards for the light 0.4.0 core.

Three invariants, all read *dynamically* from the two sources of truth
(``pyproject.toml`` and :mod:`polars_features._deps`) so the tests keep working
while either side legitimately grows:

1. ``[project.dependencies]`` is exactly ``{numpy, polars}``.  Any new hard
   dependency has to be a deliberate, reviewed change to this test too.
2. Every module in ``_deps._MODULE_TO_EXTRA`` points at an extra that actually
   exists in ``[project.optional-dependencies]`` -- otherwise the
   ``pip install 'polars-features[<extra>]'`` hint that :func:`_deps.require`
   raises would send users to a non-existent extra.
3. Every ``require("...")`` call site in the package resolves to a declared
   extra, so a newly-added lazy dependency cannot ship without its extra.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys

import pytest

from polars_features import _deps

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10 fallback
    tomllib = pytest.importorskip("tomli")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_ROOT = REPO_ROOT / "polars_features"

#: The complete, intended mandatory footprint.  Changing this set is a product
#: decision (it is the whole point of the 0.4.0 "light core"), so the test is
#: written to make that change explicit and reviewable rather than silent.
EXPECTED_HARD_DEPS = {"numpy", "polars"}

#: Entries in ``_MODULE_TO_EXTRA`` that point at an extra which no longer
#: exists, kept only because the mapping has not been pruned yet.  Each one is
#: allowed **only while the module is genuinely unused in the package**; the
#: companion test below enforces that, so an entry cannot rot into a real
#: broken install hint.
#:
#: Currently empty, and it should stay that way: ``narwhals`` -> ``interop``
#: was quarantined here until the dangling row was deleted from
#: ``polars_features/_deps.py`` (the ``interop`` extra went away in 0.4.0 and
#: nothing imports narwhals).  Prefer deleting a stale row over adding it here.
KNOWN_STALE_MODULES: set[str] = set()

# Strip a PEP 508 requirement down to its distribution name.
_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def _load_pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _dist_name(requirement: str) -> str:
    """``"polars>=1.0.0"`` -> ``"polars"`` (normalised, extras stripped)."""
    match = _REQ_NAME.match(requirement)
    assert match, f"unparseable requirement {requirement!r}"
    return match.group(1).lower().replace("_", "-")


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return _load_pyproject()


@pytest.fixture(scope="module")
def declared_extras(pyproject) -> set[str]:
    return set(pyproject["project"].get("optional-dependencies", {}))


# --------------------------------------------------------------------------- #
# 1. The mandatory footprint
# --------------------------------------------------------------------------- #
def test_hard_dependencies_are_numpy_and_polars_only(pyproject):
    declared = {_dist_name(r) for r in pyproject["project"]["dependencies"]}
    assert declared == EXPECTED_HARD_DEPS, (
        "[project.dependencies] must stay exactly {numpy, polars} -- the light "
        f"core is the product promise. Found {sorted(declared)}. Anything else "
        "belongs in an optional extra, imported lazily via "
        "`polars_features._deps.require(...)`."
    )


def test_build_backend_is_pure_python(pyproject):
    """No compiled extension: the wheel must stay universal (py3-none-any)."""
    build = pyproject["build-system"]
    assert build["build-backend"] == "hatchling.build"
    assert not (REPO_ROOT / "Cargo.toml").exists(), (
        "A Cargo.toml reappeared; 0.4.0 is a pure-Python distribution."
    )


# --------------------------------------------------------------------------- #
# 2. _MODULE_TO_EXTRA points at real extras
# --------------------------------------------------------------------------- #
def test_module_to_extra_targets_declared_extras(declared_extras):
    """Every ``require()`` install hint resolves to a real extra."""
    broken = {
        module: extra
        for module, extra in _deps._MODULE_TO_EXTRA.items()
        if extra not in declared_extras and module not in KNOWN_STALE_MODULES
    }
    assert not broken, (
        "polars_features._deps._MODULE_TO_EXTRA maps modules to extras that are "
        f"not declared in [project.optional-dependencies]: {broken}. Declared "
        f"extras: {sorted(declared_extras)}. Either add the extra to "
        "pyproject.toml or fix the mapping -- otherwise `require()` tells users "
        "to run a `pip install` that cannot work."
    )


def test_known_stale_mappings_are_genuinely_unused(declared_extras):
    """The quarantine list may only hold modules nothing actually imports.

    Guards the escape hatch above: the moment somebody starts using one of these
    modules, this test fails and the extra has to be declared for real.
    """
    sources = "\n".join(
        p.read_text(encoding="utf-8") for p in PACKAGE_ROOT.rglob("*.py")
    )
    for module in sorted(KNOWN_STALE_MODULES):
        extra = _deps._MODULE_TO_EXTRA.get(module)
        if extra in declared_extras:
            continue  # someone declared it properly; nothing to police
        used = re.search(rf"\brequire\(\s*[\"']{re.escape(module)}\b", sources)
        assert not used, (
            f"{module!r} is quarantined in KNOWN_STALE_MODULES because its extra "
            f"{extra!r} does not exist, but the package now calls "
            f"require({module!r}). Declare the extra in pyproject.toml and drop "
            "it from KNOWN_STALE_MODULES."
        )
        imported = re.search(
            rf"^\s*(?:import|from)\s+{re.escape(module)}\b", sources, re.M
        )
        assert not imported, (
            f"{module!r} is quarantined but is imported directly somewhere in "
            "polars_features/; route it through require() and declare its extra."
        )


# --------------------------------------------------------------------------- #
# 3. Every require() call site is covered
# --------------------------------------------------------------------------- #
def _require_call_modules() -> dict[str, list[str]]:
    """Map top-level module name -> the files that ``require()`` it."""
    found: dict[str, list[str]] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:  # a sibling agent mid-edit; nothing to assert here
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if name != "require" or not node.args:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                top = arg.value.split(".", 1)[0]
                found.setdefault(top, []).append(str(path.relative_to(REPO_ROOT)))
    return found


def test_every_require_call_maps_to_a_declared_extra(declared_extras):
    call_sites = _require_call_modules()
    assert call_sites, "no require() call sites found -- has the AST scan broken?"

    problems: dict[str, str] = {}
    for module, files in call_sites.items():
        extra = _deps._MODULE_TO_EXTRA.get(module)
        if extra is None:
            problems[module] = f"not in _MODULE_TO_EXTRA (used in {files})"
        elif extra not in declared_extras:
            problems[module] = f"maps to undeclared extra {extra!r} (used in {files})"
    assert not problems, (
        "lazy dependencies without a working install hint: "
        f"{problems}. Add the module to `_deps._MODULE_TO_EXTRA` and declare the "
        "matching extra in [project.optional-dependencies]."
    )


def test_recommended_and_all_extras_reference_real_extras(pyproject, declared_extras):
    """The bundle extras (`recommended`, `all`) must not name a dead extra."""
    optional = pyproject["project"]["optional-dependencies"]
    bundle_re = re.compile(r"polars[_-]features\[([^\]]+)\]")
    for bundle in ("recommended", "all"):
        for requirement in optional.get(bundle, []):
            match = bundle_re.search(requirement)
            if not match:
                continue
            referenced = {part.strip() for part in match.group(1).split(",")}
            missing = sorted(referenced - declared_extras)
            assert not missing, (
                f"extra {bundle!r} references undeclared extras {missing}."
            )
