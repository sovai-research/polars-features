"""The shipped ``.pyi`` stubs must match the runtime namespaces.

The package ships ``py.typed`` (PEP 561), so ``panelary/namespaces/*.pyi`` is
user-visible API: a checker reads the stub in preference to the runtime module.
When the two drift, a documented, working call fails to type-check for every
user while the test suite stays green.

That is exactly what happened: ``xs.pyi`` declared only ``rank`` and ``demean``
of the seven registered ``.xs`` methods, so ``pl.col("ret").xs.zscore()`` --
documented in AGENTS.md and carrying a FeatureSpec in the registry -- was a
type error. It also declared ``column: str`` where the runtime takes
``columns: str | Sequence[str]``. These tests make that drift a build failure.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import polars as pl
import pytest

import panelary  # noqa: F401  -- registers the namespaces as an import side effect

_STUB_DIR = pathlib.Path(panelary.__file__).parent / "namespaces"

#: (stub file, stub class, object exposing the runtime namespace).
_CASES = [
    ("xs.pyi", "XSExprNamespace", pl.col("x").xs),
    ("xs.pyi", "XSLazyFrameNamespace", pl.LazyFrame({"a": [1]}).xs),
    ("xs.pyi", "XSDataFrameNamespace", pl.DataFrame({"a": [1]}).xs),
    ("panel.pyi", "PanelExprNamespace", pl.col("x").panel),
]


def _stub_methods(stub: str, cls_name: str) -> dict[str, list[str]]:
    """Map method name -> parameter names, for one class in a stub file."""
    tree = ast.parse((_STUB_DIR / stub).read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            return {
                fn.name: [a.arg for a in fn.args.args + fn.args.kwonlyargs]
                for fn in node.body
                if isinstance(fn, ast.FunctionDef) and not fn.name.startswith("_")
            }
    raise AssertionError(f"{stub} declares no class {cls_name!r}")


def _runtime_methods(ns: object) -> dict[str, list[str]]:
    return {
        name: [p for p in inspect.signature(fn).parameters if p != "self"]
        for name, fn in inspect.getmembers(type(ns), inspect.isfunction)
        if not name.startswith("_")
    }


@pytest.mark.parametrize(
    ("stub", "cls_name", "ns"), _CASES, ids=[f"{s}:{c}" for s, c, _ in _CASES]
)
def test_stub_declares_every_runtime_method(
    stub: str, cls_name: str, ns: object
) -> None:
    missing = sorted(set(_runtime_methods(ns)) - set(_stub_methods(stub, cls_name)))
    assert not missing, (
        f"{stub}:{cls_name} is missing {missing}. The stub ships to users, so a "
        f"method absent here is a type error at every call site. Add it."
    )


@pytest.mark.parametrize(
    ("stub", "cls_name", "ns"), _CASES, ids=[f"{s}:{c}" for s, c, _ in _CASES]
)
def test_stub_declares_no_phantom_method(stub: str, cls_name: str, ns: object) -> None:
    extra = sorted(
        set(_stub_methods(stub, cls_name)) - set(_runtime_methods(ns)) - {"__init__"}
    )
    assert not extra, (
        f"{stub}:{cls_name} declares {extra}, which do not exist at runtime."
    )


@pytest.mark.parametrize(
    ("stub", "cls_name", "ns"), _CASES, ids=[f"{s}:{c}" for s, c, _ in _CASES]
)
def test_stub_parameter_names_match(stub: str, cls_name: str, ns: object) -> None:
    stub_m, runtime_m = _stub_methods(stub, cls_name), _runtime_methods(ns)
    mismatches = {
        name: (stub_m[name], params)
        for name, params in runtime_m.items()
        if name in stub_m and [p for p in stub_m[name] if p != "self"] != params
    }
    assert not mismatches, (
        "stub/runtime parameter mismatch (stub, runtime): "
        + "; ".join(f"{n}: {s} != {r}" for n, (s, r) in sorted(mismatches.items()))
    )
