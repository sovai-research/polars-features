# PanelKit continuous-improvement playbook

A sequenced set of **ready-to-paste prompts** for future improvement rounds on
`polars-features` (PanelKit), plus **TODO experiments** and a **recheck loop** that
re-examines whatever was already improved to see if it can go further.

How to use it: run the waves in order. Each wave is a single prompt you can paste
to an agent orchestrator ("spawn N agents…"). After every wave, run the **Recheck
& measure** prompt (Wave 0, repeatable) to confirm no regression and quantify the
new headroom. Prefer parallel agents with **disjoint file ownership**; every change
must keep exact output parity (assert against a baseline captured *before* the change)
and stay leak-safe (`assert_no_lookahead` / fit-on-train-only).

Baselines to keep current (regenerate before each wave):
- Import time: `python3 -c "import time,importlib,sys; t=time.perf_counter(); importlib.import_module('polars_features'); print(int((time.perf_counter()-t)*1000),'ms'); print(sorted({m.split('.')[0] for m in sys.modules} & {'scipy','sklearn','pandas','flaml','tqdm','numba','holidays','umap'}))"`
- Speed harness: `benchmarks/bench_vs_pandas.py` and the deps audit harness `scratchpad/deps/d7_speed.md`.
- Test floor: `pytest tests/ --ignore=tests/test_forecasting.py --ignore=tests/test_benchmarks.py -q` (expected all green).
- Install weight: build the wheel (`python3 -m build`) and record wheel size + `pip install` cold time in a fresh venv for `[]`, `[recommended]`, `[all]`.

---

## Wave 0 — Recheck & measure (run after EVERY wave; repeatable)

> Spawn 3 parallel read-only investigator agents to VERIFY and QUANTIFY the current
> state of PanelKit (`polars-features`) after the latest changes — do not modify code.
> (1) **No-regression**: run the full test suite (minus the 40-min forecasting suite)
> and report pass/fail; run `ruff check .`; confirm `import polars_features` still
> pulls only numpy+polars (no scipy/sklearn/pandas/flaml/tqdm/numba/holidays in
> sys.modules) and report the import time in ms. (2) **Headroom**: re-run the speed
> harness (`benchmarks/bench_vs_pandas.py` + a synthetic 0.5M/2.5M-row panel over
> `.ts`/`.panel`/`.xs`/`extract_features`/`reduce`/`cluster`/`factor`) and rank the
> current top-10 slowest public operations with measured times. (3) **Could-it-be-
> better**: for each of the last round's changes (list them), state whether a further
> improvement is now visible (algorithmic, vectorization, dep removal) with evidence.
> Deliver a short table: metric | previous | now | target | next lever. Flag any
> regression as P0.

---

## Wave 1 — Dependency structure, deepened

> Spawn 6 parallel investigator agents (read-only; measure with synthetic data) to
> push PanelKit's dependency structure further toward a minimal, predictable core.
> Dimensions: (1) **numpy-only reducers/selectors** — audit `reduce/` and `select/`
> for the class-B sklearn uses that were left in the `ml` extra (StandardScaler, PCA,
> TruncatedSVD, GaussianRandomProjection, LinearRegression, Ridge) and design exact
> numpy replacements with parity notes, so `reduce`/`select` run on numpy+polars and
> `ml` shrinks to only the genuinely-hard learners (HistGradientBoosting, KNN,
> Lasso/ElasticNet, KernelPCA, NMF, FastICA, AgglomerativeClustering). (2) **scipy
> holdouts** — the two lazy holdouts (`number_cwt_peaks` → find_peaks_cwt; KDTree
> entropy counts) — design numpy/polars replacements or a better algorithm so the
> `scipy` extra can be dropped entirely; the KDTree entropy is O(n²) — propose an
> O(n log n) or Polars-native neighbor count. (3) **lazy submodule loading** — if
> import time regressed above ~90ms, design a PEP-562 `__getattr__` for `pk.models`/
> `reduce`/`cluster`/`factor`/`select` with a `.pyi`/TYPE_CHECKING block. (4) **numpy
> surface** — can any numpy use be expressed as a Polars expression instead (removing
> eager materialization), shrinking the numpy dependency footprint / round-trips?
> (5) **transitive audit** — build the full transitive dependency tree for `[]`,
> `[recommended]`, `[all]`; flag heavy or duplicated transitive deps and any dep that
> a single small function forces. (6) **extras UX** — verify every `require()` hint
> maps to a real extra; add a `pk.show_versions()` / capability probe that reports
> which optional features are available. Each agent: ranked, buildable recommendations
> with effort + parity/perf evidence. No code changes.

## Wave 2 — Speed, deepened

> Spawn 6 parallel agents (measure-first, then implement with parity) to attack the
> remaining speed hotspots surfaced by Wave 0's ranking. Candidates to investigate and,
> where a clean win exists, implement with disjoint file ownership + parity tests:
> (1) **CUSUM numba path** — verify the optional numba `fast` path matches pure-Python
> exactly and quantify the speedup; consider a fully-vectorized Polars/numpy scan.
> (2) **catch22 remaining features** — after the DFA batching, profile the next
> slowest catch22 features (periodicity, outlier-timing, spline) and batch/vectorize.
> (3) **`extract_features` planning** — confirm all requested features compile into a
> single lazy Polars plan with no per-feature `.collect()`; add an execution-plan test.
> (4) **`reduce`/`cluster`** — replace per-date/per-entity Python-object loops
> (CrossSectionalPCA's 500 sklearn objects; KMeans per group) with batched numpy SVD /
> vectorized assignment. (5) **rolling/grouped ops** — find any O(n·w) rolling feature
> still built from summed `.shift()` terms (like the old frac-diff) and convert to
> incremental/convolution O(n). (6) **round-trip hunt** — grep for `.collect()` inside
> library code and lazy→eager→lazy round-trips; remove where safe. Deliver before/after
> timings at 0.5M and 2.5M rows for every change.

## Wave 3 — Efficient replacers (correctness-preserving swaps)

> Spawn parallel agents to REPLACE remaining heavy or slow implementations with lean,
> exact equivalents, each behind a parity harness. Targets: (1) any residual scipy call
> → numpy (extend `_numpy_stats`); (2) sklearn class-B → numpy (from Wave 1's design),
> shrinking the `ml` extra; (3) pandas usage in `benchmarks/`/tests that could be
> polars (keep pandas as a *test-only* dep, never runtime); (4) pure-Python inner loops
> that dominate a profile → numpy-vectorized or optional-numba (`fast` extra), never a
> new hard dep. Rule: capture the current output as a golden baseline BEFORE the swap,
> assert byte/atol-1e-9 parity after, and add the golden test to the suite. Report a
> table: replaced | old dep/impl | new impl | parity error | speedup | dep removed?

## Wave 4 — Packaging, CI, and guardrails

> Spawn agents to harden the light-package gains so they can't silently regress.
> (1) Add a pytest **import-hygiene guard**: a test asserting `import polars_features`
> loads none of {scipy, sklearn, pandas, flaml, tqdm, numba, holidays, umap} and that
> import time is under a threshold (e.g. 150ms) — fails CI on regression. (2) Add a
> **wheel-is-universal** check (built wheel tag is `py3-none-any`, contains no `.so`).
> (3) Add a **dependency-drift test**: `[project.dependencies]` stays == {numpy, polars};
> every `require()` module maps to a declared extra. (4) Set up a CI matrix that
> installs `[]`, `[recommended]`, and `[all]` and runs the suite in each, proving the
> bare core works and the extras' features light up. (5) Add a wheel-size budget check.
> Implement the tests; keep them fast.

---

## TODO experiments (hypotheses to test, not yet decided)

- **Drop numpy from the hottest paths?** Measure whether pure-Polars expressions can
  replace numpy in the most-used extractors so numpy becomes near-optional for the
  core feature path (numpy would remain for the ML/reduce extras). Likely NOT worth it
  — but quantify.
- **numba as the universal accelerator.** Prototype numba-jitted versions of the top-5
  sequential/loop kernels (cusum, sample/approx entropy, DFA, streak features); measure
  vs numpy and vs the old Rust. If numba matches Rust while keeping a pure-Python wheel,
  standardize on the `fast` extra pattern.
- **polars-native FFT?** Check whether a Polars plugin-free spectral path (or a cached
  numpy rfft plan reused across groups) beats per-group numpy.fft; measure on 2.5M rows.
- **Streaming/lazy end-to-end.** Build the golden-path pipeline fully lazy and measure
  peak memory + wall-clock vs the eager path on 5M+ rows; find where a `.collect()` is
  forced.
- **KDTree entropy O(n²) → O(n log n).** Prototype a sorted-window / Polars neighbor
  count for sample/approx entropy and measure accuracy vs speed.
- **Wheel/install weight.** Measure cold `pip install` time + on-disk size for `[]` vs
  `[recommended]` vs the pre-0.4.0 batteries-included install; publish the numbers in docs.
- **Import-time micro-budget.** Attribute the remaining ~86ms import (polars floor vs
  namespace registration vs registry build) and see if registration can be deferred
  without breaking `.panel`/`.xs`/`.ts` first-use.

## The recheck discipline (paste at the end of any wave)

> Before declaring this wave done: (a) re-run Wave 0's Recheck & measure and paste the
> metric table (import ms, top-10 op timings, test pass count, ruff, wheel size);
> (b) for every change made this wave, explicitly answer "can this be improved even
> more?" with a yes/no + the next lever; (c) confirm no new hard dependency was added
> and `[project.dependencies]` is still exactly {numpy, polars}; (d) confirm parity
> baselines were asserted for every numeric change; (e) update CHANGELOG + this
> playbook (tick done items, add newly-discovered levers). Only then commit in logical
> chunks and push.
