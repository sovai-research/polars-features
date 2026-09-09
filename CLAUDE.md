# CLAUDE.md

**See [AGENTS.md](./AGENTS.md).** It is the single source of truth for agent
instructions in this repository: setup, build/lint/type-check/test commands,
the `panel_safe` / `leakage_safe` correctness contract, naming and dependency
conventions, and an explicit "what NOT to do" list.

This file exists only so that Claude Code picks up that context; it is
deliberately a pointer rather than a fork, so the two never drift apart. Add
new instructions to `AGENTS.md`, not here.

Quick reference (details and caveats in `AGENTS.md`):

```bash
uv venv && uv pip install -e ".[dev,recommended]"   # setup
ruff check . && ruff format --check .               # lint  (~0.1s)
mypy                                                # types (blocking in CI)
pytest tests/test_<area>.py -q                      # iterate on one file
pytest -q -m "not slow"                             # full gate (>15 min)
```

The one thing to internalise before editing: **no lookahead, ever.**
Within-entity work goes `.over(entity_col)` in time order; cross-sectional work
goes `.over(time_col)`; anything with a `fit` is fit per fold on training data
only, never globally and never on the test fold.
