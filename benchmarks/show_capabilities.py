"""Print which optional Panelary features are available in this environment.

Reads :data:`panelary._internal._deps._MODULE_TO_EXTRA` -- the single source of
truth for "which extra provides which import" -- and probes each module.  Used
by the CI extras matrix so the job log itself is evidence that ``[]`` really is
bare and ``[recommended]`` / ``[all]`` really do light their features up.

Run:  python benchmarks/show_capabilities.py
Exit code is always 0; this is a report, not a gate.
"""

from __future__ import annotations

import sys

from panelary._internal._deps import _MODULE_TO_EXTRA, have


def main() -> int:
    rows = sorted(_MODULE_TO_EXTRA.items())
    width = max(len(module) for module, _ in rows)
    available = 0
    print(f"{'module'.ljust(width)}  extra          status")
    print(f"{'-' * width}  -------------  ---------")
    for module, extra in rows:
        ok = have(module)
        available += ok
        print(
            f"{module.ljust(width)}  {('[' + extra + ']').ljust(13)}  "
            f"{'available' if ok else '-'}"
        )
    print(f"\n{available}/{len(rows)} optional modules available")
    print(f"python: {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
