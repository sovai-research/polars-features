"""CI guardrail: every ``--ignore`` path in CI config points at a real file.

``pytest --ignore=<path>`` accepts a path that does not exist **silently** --
no warning, no error, exit code 0.  That makes the option a quiet trap for the
two hand-maintained ignore lists in this repo:

* ``.github/workflows/ci.yml`` -- the *bare core* matrix job skips ~15 test
  modules, each of which imports an optional dependency at module scope.  If
  one of those files is renamed and the ignore is not updated, the ignore stops
  applying and the module runs in an environment that deliberately has no
  optional dependencies installed, where it fails on the import.
* ``Makefile`` -- the ``test`` target skips ``tests/test_forecasting.py``.

So the failure mode is: rename a test file, everything is green locally, and a
job that was skipping it starts running it in the one environment where it
cannot pass.  This module closes that gap by asserting that every path named
after ``--ignore`` exists on disk.

The scan is a deliberately dumb plain-text regex over the two files rather than
a YAML/Make parse: ``pyyaml`` is not a declared dependency, and the interesting
property (does this literal path exist?) is textual anyway.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The files carrying hand-maintained ignore lists, relative to the repo root.
CONFIG_FILES = (
    ".github/workflows/ci.yml",
    "Makefile",
)

#: Lower bound on the number of ``--ignore`` paths the scan must find.  Without
#: it, a change of option spelling (``--ignore path`` -> ``--deselect``, a
#: rewritten job) would make this test vacuously pass instead of failing loudly.
#: There are 17 occurrences today (15 in the bare-core job + one per full-suite
#: invocation); the floor is set below that so routine edits do not trip it.
MIN_IGNORE_PATHS = 12

#: ``--ignore=tests/foo.py`` and the space-separated ``--ignore tests/foo.py``.
#: The value stops at whitespace, so a trailing shell line-continuation ``\``
#: is never part of it.
_IGNORE_RE = re.compile(r"--ignore(?:=|\s+)([^\s\\'\"]+)")


def _scan(relative_path: str) -> list[tuple[int, str]]:
    """Return ``(line number, ignored path)`` pairs found in one config file.

    Full-line comments are skipped so that a path mentioned in prose (``# we
    used to skip tests/foo.py``) cannot fail the test.
    """
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    found: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        for match in _IGNORE_RE.finditer(line):
            value = match.group(1)
            if value.startswith("-"):  # `--ignore --other-flag`: not a path
                continue
            found.append((lineno, value))
    return found


@pytest.fixture(scope="module")
def ignored_paths() -> dict[str, list[tuple[int, str]]]:
    return {name: _scan(name) for name in CONFIG_FILES}


@pytest.mark.parametrize("relative_path", CONFIG_FILES)
def test_config_file_exists(relative_path):
    """Guards the scan itself: a renamed config would silently check nothing."""
    assert (REPO_ROOT / relative_path).is_file(), (
        f"{relative_path} is missing; tests/test_ci_guardrails.py scans it for "
        "`--ignore` paths and cannot do its job without it."
    )


def test_scan_finds_the_ignore_lists(ignored_paths):
    total = sum(len(entries) for entries in ignored_paths.values())
    assert total >= MIN_IGNORE_PATHS, (
        f"found only {total} `--ignore` paths across {list(CONFIG_FILES)}, "
        f"expected at least {MIN_IGNORE_PATHS}. Either the ignore lists shrank "
        "a lot (great -- lower MIN_IGNORE_PATHS) or the option is now spelled "
        "differently and this guard has stopped guarding anything."
    )
    # The Makefile `test:` target skips the heavy forecasting module; if that
    # ever goes away the constant above should be revisited too.
    assert ignored_paths["Makefile"], (
        "no `--ignore` found in the Makefile; if the `test` target no longer "
        "skips anything, drop the Makefile from CONFIG_FILES."
    )


@pytest.mark.parametrize("relative_path", CONFIG_FILES)
def test_every_ignored_path_exists(relative_path, ignored_paths):
    """Every ``--ignore=<path>`` must resolve to a file (or directory) on disk.

    ``pytest`` does not complain about a stale one, so this is the only thing
    standing between a test-file rename and a job that quietly stops skipping.
    """
    missing = [
        f"{relative_path}:{lineno}: --ignore={value}"
        for lineno, value in ignored_paths[relative_path]
        if not (REPO_ROOT / value).exists()
    ]
    assert not missing, (
        "`pytest --ignore=<path>` silently accepts a path that does not exist, "
        "so these ignores no longer skip anything:\n  "
        + "\n  ".join(missing)
        + "\nUpdate the ignore list to the file's new name, or delete the entry "
        "if the test is gone."
    )
