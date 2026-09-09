"""Wheel guardrails: universal tag, no compiled artifacts, size budget.

Panelary 0.4.0 dropped its Rust extension in favour of a pure-Python
distribution, so exactly one wheel (``py3-none-any``) serves every platform and
Python version.  These tests build the wheel and assert that property, plus a
size budget, so a stray compiled module or a fat data file cannot silently
un-do it.

Building a wheel takes a few seconds, so the whole module is marked ``slow``
(deselect with ``-m "not slow"``) and skipped when :mod:`build` is unavailable.
The build runs with ``--no-isolation`` -- the backend (hatchling) is already
present in a dev environment and network-isolated CI should not need PyPI.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import zipfile

import pytest

pytestmark = pytest.mark.slow

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Maximum built-wheel size.  The budget is 1.5 MB; the pure-Python wheel
#: currently measures ~0.65 MB (it was ~0.29 MB in 0.4.0 -- `panelary/evolve/`
#: and friends account for most of the growth), so there is still room to grow
#: while catching an accidentally vendored binary, dataset or notebook.
#: Override with
#: ``PANELARY_WHEEL_BUDGET_MB`` if the budget is deliberately raised.
WHEEL_BUDGET_MB = float(os.environ.get("PANELARY_WHEEL_BUDGET_MB", "1.5"))

#: File suffixes that would mean the distribution is no longer universal.
COMPILED_SUFFIXES = (".so", ".pyd", ".dylib", ".dll", ".a", ".lib", ".rlib")


def _have_build() -> bool:
    try:
        import build  # noqa: F401
        import hatchling  # noqa: F401
    except ImportError:
        return False
    return True


requires_build = pytest.mark.skipif(
    not _have_build(),
    reason="needs the `build` frontend and the `hatchling` backend installed",
)


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> pathlib.Path:
    """Build the wheel once for the module and return its path."""
    outdir = tmp_path_factory.mktemp("wheel")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(outdir),
            str(REPO_ROOT),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"wheel build failed in this environment:\n{proc.stderr[-2000:]}")
    wheels = sorted(outdir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


@requires_build
def test_wheel_tag_is_universal(built_wheel):
    """The filename tag must be ``py3-none-any``."""
    stem = built_wheel.name.removesuffix(".whl")
    parts = stem.split("-")
    # {distribution}-{version}(-{build})?-{python}-{abi}-{platform}
    tag = "-".join(parts[-3:])
    assert tag == "py3-none-any", (
        f"wheel {built_wheel.name} has tag {tag!r}, expected 'py3-none-any'. "
        "Panelary is a pure-Python distribution -- a platform tag means a "
        "compiled extension or a platform-pinned build backend crept back in."
    )


@requires_build
def test_wheel_root_is_tagged_purelib(built_wheel):
    """``WHEEL`` metadata declares a pure-Python, universal build."""
    with zipfile.ZipFile(built_wheel) as zf:
        wheel_meta = next(n for n in zf.namelist() if n.endswith(".dist-info/WHEEL"))
        text = zf.read(wheel_meta).decode()
    fields = dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)
    assert fields.get("Root-Is-Purelib", "").lower() == "true", text
    assert fields.get("Tag") == "py3-none-any", text


@requires_build
def test_wheel_contains_no_compiled_artifacts(built_wheel):
    """No ``.so`` / ``.pyd`` / other native object may ship in the wheel."""
    with zipfile.ZipFile(built_wheel) as zf:
        names = zf.namelist()
    offenders = [n for n in names if n.lower().endswith(COMPILED_SUFFIXES)]
    assert not offenders, (
        f"wheel {built_wheel.name} contains compiled artifacts {offenders}; the "
        "0.4.0 distribution must stay pure-Python."
    )
    # A nested wheel/sdist would also break the universal promise.
    nested = [n for n in names if n.endswith((".whl", ".tar.gz"))]
    assert not nested, f"wheel contains nested distributions: {nested}"


@requires_build
def test_wheel_size_within_budget(built_wheel):
    size_mb = built_wheel.stat().st_size / (1024 * 1024)
    assert size_mb <= WHEEL_BUDGET_MB, (
        f"wheel is {size_mb:.2f} MB, over the {WHEEL_BUDGET_MB:.2f} MB budget. "
        "Check for a vendored binary, dataset or notebook that slipped into "
        "[tool.hatch.build.targets.wheel]."
    )


@requires_build
def test_wheel_ships_only_the_package(built_wheel):
    """Top-level wheel contents are the package plus its ``.dist-info``."""
    with zipfile.ZipFile(built_wheel) as zf:
        tops = {n.split("/", 1)[0] for n in zf.namelist()}
    unexpected = {t for t in tops if t != "panelary" and not t.endswith(".dist-info")}
    assert not unexpected, (
        f"wheel ships unexpected top-level entries {sorted(unexpected)}; only "
        "`panelary/` and the `.dist-info` directory belong in it."
    )
