# `panelary/_internal/`

Private helper layer. **Not public API**, not documented, not covered by
`tests/test_public_api.py` — that test derives the public surface from the
filesystem and skips underscore-prefixed names, which is the whole reason this
directory can exist without adding an API obligation.

The full rules live in `plans/todo/layout-build-contract.md`. Four of them bite
the moment you open a file in here:

1. **This package is a leaf.** Nothing in `_internal/` may import from a public
   subpackage. There is one violation today — `_ranges.py:9` imports
   `panelary.offsets._strip_freq_alias` — and it should not gain company.
2. **`__init__.py` re-exports nothing, deliberately.** Import the submodule
   directly (`from panelary._internal._numpy_stats import skew`) so the package
   costs zero at `import panelary` time. Do not add a convenience re-export.
3. **`_deps.py` must stay at this exact path and stay stdlib-only.**
   `tests/test_import_hygiene.py:140-160` loads
   `panelary/_internal/_deps.py` straight off disk by that literal string and
   asserts it has no third-party import. It is the bootstrap gate for every
   optional dependency; it cannot depend on one.
4. **No optional dependency at module scope, anywhere in here.** Route it
   through `panelary._internal._deps.require` inside the using function.
   `tests/test_import_hygiene.py:105-124` probes `_numpy_stats` and `_deps` as
   standalone imports and fails on a leaked scipy/sklearn/pandas.

`_verbs.py` is the exception to "helper": it is the implementation of the eight
top-level golden-path verbs. It imports polars, numpy and every subpackage
*inside* function bodies, which is what lets `panelary/__init__.py:294-306` wire
it up for ~0 ms. Keep the imports deferred.
