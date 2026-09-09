"""Shared fixtures, naive references and import guards for the ``detect`` suite.

This module deliberately carries **no test functions**. It exists so that the
four ``tests/test_detect_*.py`` files share one set of

* guarded imports of the (concurrently authored) ``panelary.detect``
  submodules,
* deterministic data generators (``numpy.random.default_rng`` only -- the
  legacy global RNG is banned repo-wide because it makes "run it twice" tests
  meaningless), and
* **independently written, deliberately slow and obvious** reference
  implementations. The references never call into ``panelary.detect``;
  they are the ground truth the vectorised prefix-sum kernels are measured
  against.

Missing-module policy
---------------------
While Agents A-D are still landing their files the suite must *collect*
cleanly, so a missing submodule turns into a ``pytest.skip``. A skip is loud in
the report and is never mistaken for a pass. Setting
``PANELARY_DETECT_STRICT=1`` promotes every such skip to a hard failure, which
is what CI should do once the module is complete.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

import numpy as np
import polars as pl
import pytest

# --------------------------------------------------------------------------- #
# Guarded imports
# --------------------------------------------------------------------------- #
#: Promote "module not written yet" skips to failures.
STRICT = os.environ.get("PANELARY_DETECT_STRICT", "").lower() in {"1", "true", "yes"}


def _try_import(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except Exception:  # ImportError, or a syntax error in a half-landed file
        return None


moments = _try_import("panelary.detect._moments")
bsadf = _try_import("panelary.detect._bsadf")
critvals = _try_import("panelary.detect._critvals")
monitors = _try_import("panelary.detect._monitors")
panel = _try_import("panelary.detect._panel")

#: The package ``__init__`` is the orchestrator's file. Until it lands,
#: ``panelary.detect`` still imports -- as an implicit *namespace*
#: package, whose ``__file__`` is ``None`` and whose contents are empty. Treating
#: that as "present" would make every package-level test pass vacuously, so it
#: only counts once there is a real module behind it.
_pkg = _try_import("panelary.detect")
detect_pkg = _pkg if getattr(_pkg, "__file__", None) else None

#: Every submodule, for the import-hygiene probe and the API-wide name scan.
SUBMODULES = ("_moments", "_bsadf", "_critvals", "_monitors", "_panel")

_MODULES = {
    "_moments": moments,
    "_bsadf": bsadf,
    "_critvals": critvals,
    "_monitors": monitors,
    "_panel": panel,
    "detect": detect_pkg,
}


def require(*names: str) -> None:
    """Skip (or fail, under ``PANELARY_DETECT_STRICT``) if a module is absent."""
    missing = [n for n in names if _MODULES.get(n) is None]
    if not missing:
        return
    msg = (
        "panelary.detect: not importable (missing, broken, or -- for "
        f"'detect' -- still only an implicit namespace package): {missing}"
    )
    if STRICT:
        pytest.fail(msg + " -- PANELARY_DETECT_STRICT is set")
    pytest.skip(msg)


def require_attr(module: ModuleType | None, attr: str, modname: str):
    """Return ``module.attr``; skip/fail with a contract-quoting message."""
    require(modname)
    fn = getattr(module, attr, None)
    if fn is None:
        msg = f"`{modname}.{attr}` is missing; the build contract requires it"
        if STRICT:
            pytest.fail(msg)
        pytest.skip(msg)
    return fn


# --------------------------------------------------------------------------- #
# Deterministic data
# --------------------------------------------------------------------------- #
#: Window/length constants shared by the leak-safety parametrisations. They are
#: *literals*, never functions of the sample size -- a window rule of the form
#: ``min_window = f(T)`` is exactly the leak invariant 1 exists to catch.
MIN_W = 30
N_LONG = 200


def random_walk(n: int = N_LONG, *, seed: int = 0, sigma: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n) * sigma)


def bubble_series(n: int = N_LONG, *, seed: int = 1) -> np.ndarray:
    """Random walk with an explosive AR(1) episode in the middle."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    y = np.zeros(n)
    lo, hi = int(0.45 * n), int(0.70 * n)
    for t in range(1, n):
        rho = 1.04 if lo <= t < hi else 1.0
        y[t] = rho * y[t - 1] + eps[t]
    return y


def panel_matrix(
    n_entities: int = 8, n_time: int = 150, *, seed: int = 3
) -> np.ndarray:
    """An ``(N, T)`` panel of correlated random walks."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(n_time)
    eps = rng.standard_normal((n_entities, n_time))
    return np.cumsum(0.6 * common[None, :] + eps, axis=1)


# --------------------------------------------------------------------------- #
# Naive references -- slow, obvious, and independent of the module under test
# --------------------------------------------------------------------------- #
def naive_adf(y: np.ndarray, lag: int = 0, trend: str = "c") -> float:
    """ADF t-statistic on ``beta`` from a per-window OLS fit, via ``lstsq``.

    Regression (PSY specification, intercept only by default)::

        dy[t] = alpha + beta * y[t-1] + sum_i gamma_i * dy[t-i] + e[t]

    Written with an explicit Python loop and ``np.linalg.lstsq`` so it shares
    no code path -- and no algebra -- with the O(1) prefix-sum kernel.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    p = int(lag)
    if n < p + 3:
        return np.nan
    dy = np.diff(y)  # dy[i] == y[i+1] - y[i]

    rows: list[list[float]] = []
    rhs: list[float] = []
    for t in range(p + 1, n):
        cols = [1.0, float(y[t - 1])]
        if trend == "ct":
            cols.append(float(t))
        cols.extend(float(dy[t - 1 - i]) for i in range(1, p + 1))
        rows.append(cols)
        rhs.append(float(dy[t - 1]))

    x = np.asarray(rows, dtype=np.float64)
    yy = np.asarray(rhs, dtype=np.float64)
    nobs, k = x.shape
    if nobs <= k:
        return np.nan
    beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
    resid = yy - x @ beta
    s2 = float(resid @ resid) / (nobs - k)
    xtx_inv = np.linalg.pinv(x.T @ x)
    var = s2 * xtx_inv[1, 1]
    if not np.isfinite(var) or var <= 0.0:
        return np.nan
    return float(beta[1] / np.sqrt(var))


def naive_bsadf(
    y: np.ndarray, *, min_window: int, lag: int = 0, trend: str = "c"
) -> np.ndarray:
    """Exhaustive backward sup-ADF: ``max_s ADF(y[s : e+1])`` for every ``e``."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    out = np.full(n, np.nan)
    for e in range(n):
        best = -np.inf
        for s in range(0, e - min_window + 2):
            v = naive_adf(y[s : e + 1], lag, trend)
            if np.isfinite(v) and v > best:
                best = v
        if np.isfinite(best):
            out[e] = best
    return out


def naive_page_cusum(z: np.ndarray) -> np.ndarray:
    """Textbook recursion ``g_t = max(0, g_{t-1} + z_t)``, one Python loop."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty(z.size, dtype=np.float64)
    g = 0.0
    for i, v in enumerate(z):
        g = max(0.0, g + float(v))
        out[i] = g
    return out


def naive_focus(z: np.ndarray) -> np.ndarray:
    """Brute-force FOCuS: ``max_{0<=i<=k} (S[k+1]-S[i])**2 / (2*(k+1-i))``."""
    z = np.asarray(z, dtype=np.float64)
    s = np.concatenate([[0.0], np.cumsum(z)])
    n = z.size
    out = np.empty(n, dtype=np.float64)
    for k in range(n):
        t = k + 1
        i = np.arange(0, t)
        d = s[t] - s[i]
        out[k] = float(np.max(d * d / (2.0 * (t - i))))
    return out


def naive_focus_one_sided(z: np.ndarray) -> np.ndarray:
    """One-sided (upward-shift only) FOCuS, used only for failure diagnostics."""
    z = np.asarray(z, dtype=np.float64)
    s = np.concatenate([[0.0], np.cumsum(z)])
    n = z.size
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        t = k + 1
        i = np.arange(0, t)
        d = s[t] - s[i]
        d = np.where(d > 0.0, d, 0.0)
        out[k] = float(np.max(d * d / (2.0 * (t - i))))
    return out


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #
def bit_identical(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact equality with ``NaN == NaN``; no tolerance at all."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b, equal_nan=True))


def max_abs_dev(a: np.ndarray, b: np.ndarray) -> float:
    """Largest absolute deviation, treating matched NaNs as agreement."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return np.inf  # a NaN mask mismatch is an infinite disagreement
    diff = np.abs(a - b)
    diff[both_nan] = 0.0
    return float(np.nanmax(diff)) if diff.size else 0.0


# --------------------------------------------------------------------------- #
# The registry of public entry points exercised by the generic leak invariants
# --------------------------------------------------------------------------- #
def _page_cusum_case(y: np.ndarray) -> np.ndarray:
    fn = monitors.page_cusum_expr
    df = pl.DataFrame({"z": np.asarray(y, dtype=np.float64)})
    return df.select(fn(pl.col("z")).alias("g")).get_column("g").to_numpy()


def resolve_shiryaev_kwargs(z: np.ndarray) -> tuple[dict, np.ndarray]:
    """Work out how ``_monitors.shiryaev_roberts`` wants its ``llr`` argument.

    The contract fixes the recursion (log domain, ``logaddexp``) but not the
    shape of ``llr``. Try, in order: a callable, a per-observation array, a
    scalar shift. Returns the accepted kwargs plus the reference LLR sequence
    the recursion must consume.
    """
    fn = monitors.shiryaev_roberts
    delta = 1.0
    ref = delta * np.asarray(z, dtype=np.float64) - 0.5 * delta**2

    def _llr_callable(x):
        return delta * np.asarray(x, dtype=np.float64) - 0.5 * delta**2

    for kwargs in ({"llr": _llr_callable}, {"llr": ref}, {"llr": delta}):
        try:
            out = fn(np.asarray(z, dtype=np.float64), **kwargs)
        except Exception:
            continue
        if isinstance(out, np.ndarray) and out.shape == np.shape(z):
            return kwargs, ref
    pytest.fail(
        "`_monitors.shiryaev_roberts(z, llr=...)` accepted none of the three "
        "documented `llr` forms (callable / per-observation array / scalar "
        "shift) while returning an array shaped like `z`."
    )


def build_cases() -> dict[str, callable]:
    """Every 1-D public entry point, as ``f(y) -> ndarray`` aligned with ``y``.

    Built lazily from whichever modules have landed, so the generic invariants
    automatically cover new entry points as Agents A-D push their files.
    """
    cases: dict[str, callable] = {}

    if bsadf is not None and hasattr(bsadf, "bsadf_sequence"):
        fn = bsadf.bsadf_sequence
        for lag in (0, 1, 2):
            cases[f"bsadf_exhaustive_lag{lag}"] = lambda y, _l=lag: fn(
                y, min_window=MIN_W, lag=_l, grid=None
            )
            cases[f"bsadf_grid32_lag{lag}"] = lambda y, _l=lag: fn(
                y, min_window=MIN_W, lag=_l, grid=32
            )
        cases["bsadf_refine_top3"] = lambda y: fn(
            y, min_window=MIN_W, lag=0, grid=32, refine_top_k=3
        )

    if monitors is not None:
        if hasattr(monitors, "page_cusum_expr"):
            cases["page_cusum"] = _page_cusum_case
        if hasattr(monitors, "focus"):
            cases["focus"] = lambda y: monitors.focus(np.asarray(y, dtype=np.float64))
        if hasattr(monitors, "hb_cusum"):
            cases["hb_cusum_stat"] = lambda y: monitors.hb_cusum(y, r0=MIN_W)[0]
            cases["hb_cusum_bound"] = lambda y: monitors.hb_cusum(y, r0=MIN_W)[1]
        if hasattr(monitors, "end_of_sample_S"):
            cases["end_of_sample_S"] = lambda y: monitors.end_of_sample_S(y, m=10)

    return cases


#: Entry points whose output must be invariant to an additive level shift.
#: ``cumulative_moments`` anchors ``y -> y - y[0]``, and an ADF regression with
#: an intercept is level-invariant in exact arithmetic, so any drift here is
#: pure floating-point conditioning loss in the prefix sums.
LEVEL_INVARIANT_PREFIXES = ("bsadf_",)


def case_names() -> list[str]:
    return sorted(build_cases())


def run_case(name: str, *, seed: int = 0, n: int = N_LONG) -> np.ndarray:
    """Evaluate one registered case on the canonical series (subprocess entry)."""
    return np.asarray(build_cases()[name](random_walk(n, seed=seed)), dtype=np.float64)
