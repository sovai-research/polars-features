"""The golden-path verbs: `pn.impute`, `pn.select`, ... must exist and work.

Panelary's top-level surface is eight verbs (``panelary/_verbs.py``). This file
is the contract for that surface, and it is deliberately paranoid about three
things:

1. **Every verb is exercised against a real frame.** The README once documented
   ``pn.col``, ``pn.transform.winsorize`` and ``pn.models.lgbm_classifier``,
   none of which existed. A verb that imports but has never been called is the
   same failure with better packaging, so each one here is invoked on a
   synthetic panel and its return type checked.
2. **The eight are uniform.** Same first parameter, same ``method`` keyword,
   same ``entity`` / ``time`` tail. That uniformity *is* the API; a drifting
   signature is a defect, not a detail.
3. **Only eight.** ``pn.signatures`` and ``pn.anomaly`` were considered and
   deliberately not shipped, because nothing implements them. The test that
   they raise ``AttributeError`` is what keeps that decision from being quietly
   reversed by a well-meaning autocomplete.
4. **Three of them collide with a subpackage.** ``select``, ``cluster`` and
   ``reduce`` are verbs *and* module names, so they are delegating wrappers
   rather than plain functions, and all four access patterns
   (call, ``pn.select.mrmr``, ``from panelary.select import mrmr``,
   ``import panelary.select as s``) are pinned here. The other five must stay
   plain functions -- no machinery where there is no conflict.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import pathlib

import numpy as np
import polars as pl
import pytest

import panelary as pn

#: The complete public verb list. Adding a name here without an implementation
#: is exactly the failure mode this file exists to prevent.
VERBS = (
    "bubbles",
    "causal",
    "cluster",
    "features",
    "impute",
    "reduce",
    "regression",
    "select",
)

#: Names that were considered and refused: no implementation stands behind
#: them, so they must not resolve.
NOT_SHIPPED = ("signatures", "anomaly")

#: The three verbs whose name is also a subpackage name, with a public symbol
#: of that subpackage to probe. Only these get the delegating wrapper.
COLLIDING = (
    ("select", "mrmr"),
    ("cluster", "KShapeClusterer"),
    ("reduce", "PanelPCA"),
)

#: The five that shadow nothing and stay plain functions. No machinery without
#: a collision to justify it.
PLAIN = ("bubbles", "causal", "features", "impute", "regression")

_N_ENTITIES = 6
_N_PERIODS = 60


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    """A small, well-behaved balanced panel: keys, two factors, a price, a target."""
    rng = np.random.default_rng(0)
    n = _N_ENTITIES * _N_PERIODS
    factor = rng.normal(size=n)
    return pl.DataFrame(
        {
            "ticker": np.repeat([f"E{i}" for i in range(_N_ENTITIES)], _N_PERIODS),
            "date": np.tile(np.arange(_N_PERIODS), _N_ENTITIES),
            "a": factor + rng.normal(size=n) * 0.1,
            "b": 2.0 * factor + rng.normal(size=n) * 0.1,
            "px": np.abs(np.cumsum(rng.normal(size=n))) + 10.0,
            "ret": 0.5 * factor + rng.normal(size=n) * 0.1,
        }
    )


@pytest.fixture(scope="module")
def gappy(panel: pl.DataFrame) -> pl.DataFrame:
    """The same panel with holes punched in ``a``, including a leading one."""
    return panel.with_columns(
        pl.when(pl.col("date") % 7 == 3).then(None).otherwise(pl.col("a")).alias("a")
    )


# --------------------------------------------------------------------------- #
# 1. The surface exists
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("verb", VERBS)
def test_verb_is_importable_and_callable(verb: str) -> None:
    """``pn.<verb>`` resolves to a callable defined in `panelary._internal._verbs`."""
    assert hasattr(pn, verb), f"pn.{verb} does not resolve"
    fn = getattr(pn, verb)
    assert callable(fn), f"pn.{verb} is not callable"
    assert fn.__module__ == "panelary._internal._verbs"


@pytest.mark.parametrize("verb", VERBS)
def test_verb_is_advertised(verb: str) -> None:
    """The verb is in ``__all__`` (so ``from panelary import *`` finds it)."""
    assert verb in pn.__all__
    assert verb in dir(pn)


def test_all_has_no_duplicates() -> None:
    """`select` / `cluster` / `reduce` are both verb and subpackage names."""
    assert len(pn.__all__) == len(set(pn.__all__)), (
        f"duplicated names in __all__: "
        f"{sorted({n for n in pn.__all__ if pn.__all__.count(n) > 1})}"
    )


@pytest.mark.parametrize("verb", VERBS)
def test_verb_is_documented(verb: str) -> None:
    """Every verb carries a full numpy-style docstring, leak-safety included."""
    doc = inspect.getdoc(getattr(pn, verb))
    assert doc, f"pn.{verb} has no docstring"
    for section in ("Parameters", "Returns", "Raises", "Examples", "Leak-safety"):
        assert section in doc, f"pn.{verb} docstring is missing a {section!r} section"


@pytest.mark.parametrize("verb", VERBS)
def test_verb_signature_is_uniform(verb: str) -> None:
    """All eight read as one API: same first argument, same keyword spine."""
    params = inspect.signature(getattr(pn, verb)).parameters
    names = list(params)

    assert names[0] == "data"
    assert params["data"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["data"].default is inspect.Parameter.empty

    assert names[1] == "method", f"pn.{verb}: `method` must be the first keyword"
    assert params["method"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["method"].default is not inspect.Parameter.empty

    for key in ("entity", "time"):
        assert params[key].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[key].default is None

    # `**kwargs` is the escape hatch to the underlying implementation.
    assert params[names[-1]].kind is inspect.Parameter.VAR_KEYWORD


# --------------------------------------------------------------------------- #
# 2. Only eight: the names we refused must stay refused
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", NOT_SHIPPED)
def test_unimplemented_verbs_are_not_shipped(name: str) -> None:
    """`pn.signatures` / `pn.anomaly` have no implementation, so they must not resolve.

    Shipping a name that does nothing is worse than shipping no name: it is
    discoverable, it is documented by autocomplete, and it fails at call time
    instead of import time.
    """
    with pytest.raises(AttributeError):
        getattr(pn, name)
    assert name not in pn.__all__


# --------------------------------------------------------------------------- #
# 3. Each verb actually runs
# --------------------------------------------------------------------------- #
def test_impute_runs(gappy: pl.DataFrame) -> None:
    out = pn.impute(gappy)
    assert isinstance(out, pl.DataFrame)
    assert out.columns == gappy.columns
    assert out.height == gappy.height
    # Every interior gap is filled; only a leading gap can survive a forward fill.
    assert out["a"].null_count() < gappy["a"].null_count()


def test_impute_preserves_the_container(gappy: pl.DataFrame) -> None:
    """PanelFrame in, PanelFrame out; lazy in, lazy out; eager in, eager out."""
    assert isinstance(pn.impute(gappy), pl.DataFrame)
    assert isinstance(pn.impute(gappy.lazy()), pl.LazyFrame)
    assert isinstance(pn.impute(pn.as_panel(gappy)), pn.PanelFrame)


def test_select_runs(panel: pl.DataFrame) -> None:
    chosen = pn.select(panel, target="ret", k=2)
    assert isinstance(chosen, list)
    assert len(chosen) == 2
    assert set(chosen) <= {"a", "b", "px"}


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("variance", {"k": 2}),
        ("correlation", {"threshold": 0.9}),
        ("pfa", {"k": 2}),
    ],
)
def test_select_unsupervised_methods_run(
    panel: pl.DataFrame, method: str, kwargs: dict
) -> None:
    if method == "pfa":
        pytest.importorskip("sklearn")
    chosen = pn.select(panel, method=method, **kwargs)
    assert isinstance(chosen, list)
    assert chosen
    assert set(chosen) <= set(panel.columns)


def test_features_runs(panel: pl.DataFrame) -> None:
    out = pn.features(panel, method=["absolute_energy"], columns="px")
    assert isinstance(out, pl.DataFrame)
    assert out.height == _N_ENTITIES
    assert "absolute_energy" in out.columns


def test_features_is_lazy_in_lazy_out(panel: pl.DataFrame) -> None:
    out = pn.features(panel.lazy(), method=["absolute_energy"], columns="px")
    assert isinstance(out, pl.LazyFrame)
    assert out.collect().height == _N_ENTITIES


def test_bubbles_runs(panel: pl.DataFrame) -> None:
    out = pn.bubbles(panel, columns="px", min_window=20)
    assert isinstance(out, pl.DataFrame)
    assert out.height == panel.height
    assert "px__bsadf" in out.columns
    assert out["px__bsadf"].is_finite().any()


def test_bubbles_hb_cusum_emits_statistic_and_boundary(panel: pl.DataFrame) -> None:
    out = pn.bubbles(panel, method="hb_cusum", columns="px", min_window=20)
    assert {"px__hb_cusum", "px__hb_cusum_boundary"} <= set(out.columns)


def test_cluster_runs(panel: pl.DataFrame) -> None:
    out = pn.cluster(panel, columns=["a", "b"], n_clusters=2)
    assert isinstance(out, pn.PanelFrame)
    frame = out.collect()
    assert frame.height == panel.height
    assert "ksh_label" in frame.columns


def test_cluster_cross_sectional_runs(panel: pl.DataFrame) -> None:
    pytest.importorskip("sklearn")
    out = pn.cluster(panel, method="cross_sectional", columns=["a", "b"], n_clusters=2)
    assert isinstance(out, pn.PanelFrame)
    assert "xcl_label" in out.collect().columns


def test_regression_runs(panel: pl.DataFrame) -> None:
    res = pn.regression(panel, y="ret", x=["a"], absorb=["ticker"])
    assert hasattr(res, "params")
    # ret = 0.5 * factor + noise, and `a` is the factor plus small noise.
    assert float(res.params[0]) == pytest.approx(0.5, abs=0.1)


@pytest.mark.parametrize("method", ["fama_macbeth", "mean_group", "cce_mg"])
def test_regression_alternative_methods_run(panel: pl.DataFrame, method: str) -> None:
    res = pn.regression(panel, method=method, y="ret", x=["a"])
    assert res is not None


def test_causal_runs(panel: pl.DataFrame) -> None:
    res = pn.causal(panel, y="ret", treatment="a", controls=["b"])
    assert hasattr(res, "theta")
    assert np.isfinite(float(res.theta))


def test_causal_pds_runs(panel: pl.DataFrame) -> None:
    res = pn.causal(panel, method="pds", y="ret", treatment="a", controls=["b"])
    assert np.isfinite(float(res.theta))


def test_reduce_runs(panel: pl.DataFrame) -> None:
    pytest.importorskip("sklearn")
    out = pn.reduce(panel, columns=["a", "b"], n_components=1)
    assert isinstance(out, pn.PanelFrame)
    frame = out.collect()
    assert frame.height == panel.height
    assert any(c.startswith("pc_") for c in frame.columns)


@pytest.mark.parametrize("method", ["pca_factors", "hfa", "robust_pca"])
def test_reduce_factor_extractors_run(panel: pl.DataFrame, method: str) -> None:
    out = pn.reduce(panel, method=method, columns=["a", "b", "px"], n_components=1)
    assert "factor_1" in out.collect().columns


# --------------------------------------------------------------------------- #
# 4. Leak-safety behaviour promised by the docstrings
# --------------------------------------------------------------------------- #
def test_impute_default_never_reads_the_future() -> None:
    """The default (`ffill`) leaves a *leading* gap unfilled -- as it must.

    A leading null is the one hole a point-in-time imputer cannot plug: there is
    no past to carry forward. Any method that fills it (bfill, interpolate,
    entity mean) has read a future row, which is precisely the leak.
    """
    df = pl.DataFrame(
        {
            "ticker": ["A", "A", "A"],
            "date": [1, 2, 3],
            "px": [None, 2.0, None],
        }
    )
    out = pn.impute(df)
    assert out["px"].to_list() == [None, 2.0, 2.0]


def test_impute_leaky_methods_warn() -> None:
    """`bfill` reads later rows; the warning is the feature."""
    from panelary.preprocessing import LeakageWarning

    df = pl.DataFrame({"ticker": ["A", "A"], "date": [1, 2], "px": [None, 2.0]})
    with pytest.warns(LeakageWarning):
        pn.impute(df, method="bfill")


def test_bubbles_requires_an_absolute_min_window(panel: pl.DataFrame) -> None:
    """`min_window` has no default on purpose: a sample-dependent one leaks."""
    with pytest.raises(ValueError, match="min_window"):
        pn.bubbles(panel, columns="px")


def test_bubbles_is_prefix_invariant(panel: pl.DataFrame) -> None:
    """Truncating the future must not change an already-published statistic.

    This is the property that makes the verb safe to call inside a walk-forward
    loop, and the one a careless facade (mis-sorted rows, a full-sample
    normalisation) would silently destroy even though the kernel underneath is
    correct.
    """
    cut = _N_PERIODS - 10
    full = pn.bubbles(panel, columns="px", min_window=20)
    short = pn.bubbles(panel.filter(pl.col("date") < cut), columns="px", min_window=20)

    joined = short.join(
        full.select("ticker", "date", full_stat=pl.col("px__bsadf")),
        on=["ticker", "date"],
        how="inner",
    )
    assert joined.height == _N_ENTITIES * cut
    np.testing.assert_allclose(
        joined["px__bsadf"].to_numpy(), joined["full_stat"].to_numpy()
    )


@pytest.mark.parametrize(
    ("verb", "kwargs"),
    [
        ("impute", {}),
        ("select", {"target": "ret", "k": 1}),
        ("features", {"method": ["absolute_energy"], "columns": "px"}),
        ("bubbles", {"columns": "px", "min_window": 20}),
        ("cluster", {"columns": ["a", "b"], "n_clusters": 2}),
        ("regression", {"y": "ret", "x": ["a"], "absorb": ["ticker"]}),
        ("causal", {"y": "ret", "treatment": "a", "controls": ["b"]}),
        ("reduce", {"columns": ["a", "b"], "n_components": 1}),
    ],
)
def test_every_verb_accepts_a_panelframe(
    panel: pl.DataFrame, verb: str, kwargs: dict
) -> None:
    """A PanelFrame carries its own keys, so no verb should need `entity=`/`time=`."""
    if verb == "reduce":
        pytest.importorskip("sklearn")
    assert getattr(pn, verb)(pn.as_panel(panel), **kwargs) is not None


# --------------------------------------------------------------------------- #
# 4b. `select` / `cluster` / `reduce` are verbs AND doors to their subpackage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("verb", PLAIN)
def test_non_colliding_verbs_are_plain_functions(verb: str) -> None:
    """Five of the eight shadow nothing, so they carry no wrapper machinery."""
    assert inspect.isfunction(getattr(pn, verb)), (
        f"pn.{verb} shadows no subpackage; it should be a plain function"
    )


@pytest.mark.parametrize(("verb", "symbol"), COLLIDING)
def test_colliding_verb_is_callable(verb: str, symbol: str) -> None:
    """Pattern 1: `pn.select(df, ...)` -- it is still, first of all, a verb."""
    obj = getattr(pn, verb)
    assert callable(obj)
    assert not inspect.isfunction(obj), (
        f"pn.{verb} collides with panelary.{verb}; a plain function here loses "
        "the module's contents"
    )


@pytest.mark.parametrize(("verb", "symbol"), COLLIDING)
def test_colliding_verb_delegates_attributes_to_its_subpackage(
    verb: str, symbol: str
) -> None:
    """Pattern 2: `pn.select.mrmr` resolves to the module's own object."""
    obj = getattr(pn, verb)
    module = importlib.import_module(f"panelary.{verb}")
    assert getattr(obj, symbol) is getattr(module, symbol)
    assert obj.__all__ == module.__all__
    assert symbol in dir(obj)


@pytest.mark.parametrize(("verb", "symbol"), COLLIDING)
def test_colliding_verb_leaves_from_import_untouched(verb: str, symbol: str) -> None:
    """Pattern 3: `from panelary.select import mrmr` goes via sys.modules."""
    module = importlib.import_module(f"panelary.{verb}")
    assert hasattr(module, symbol)
    assert module.__name__ == f"panelary.{verb}"


@pytest.mark.parametrize(("verb", "symbol"), COLLIDING)
def test_colliding_verb_survives_import_as_alias(verb: str, symbol: str) -> None:
    """Pattern 4: `import panelary.cluster as c` binds the wrapper, which behaves.

    ``import a.b as c`` resolves through ``getattr(a, "b")``, so the alias picks
    up the verb rather than the module. Delegation is what keeps that working.
    """
    import panelary.cluster as c
    import panelary.reduce as r
    import panelary.select as s

    alias = {"select": s, "cluster": c, "reduce": r}[verb]
    assert hasattr(alias, symbol)
    assert hasattr(alias, "__all__")
    assert callable(alias)


@pytest.mark.parametrize(("verb", "symbol"), COLLIDING)
def test_colliding_verb_keeps_the_verbs_signature_and_docstring(
    verb: str, symbol: str
) -> None:
    """The wrapper must look like the verb, not like its own class."""
    obj = getattr(pn, verb)
    params = inspect.signature(obj).parameters
    assert list(params)[0] == "data"
    assert "method" in params
    assert obj.__name__ == verb
    doc = inspect.getdoc(obj)
    assert doc is not None
    assert "Leak-safety" in doc
    assert doc == inspect.getdoc(obj.__wrapped__)


@pytest.mark.parametrize(("verb", "symbol"), COLLIDING)
def test_colliding_verb_reports_a_real_miss_clearly(verb: str, symbol: str) -> None:
    """A genuine typo must name both places we looked, not 'function has no ...'."""
    with pytest.raises(AttributeError, match=f"panelary.{verb}"):
        getattr(pn, verb).no_such_name


# --------------------------------------------------------------------------- #
# 4c. One version number, two files that must agree
# --------------------------------------------------------------------------- #
_PYPROJECT = pathlib.Path(pn.__file__).resolve().parent.parent / "pyproject.toml"


def _declared_version() -> str | None:
    """The ``[project] version`` from the repo's pyproject, if we are in a checkout."""
    if not _PYPROJECT.is_file():
        return None
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10 has no tomllib
        return None
    return str(tomllib.loads(_PYPROJECT.read_text())["project"]["version"])


def test_version_matches_pyproject() -> None:
    """`panelary.__version__` and pyproject's version are two copies of one fact.

    They silently desynced once already (module 0.4.0 against pyproject 0.5.0),
    because nothing compared them.
    """
    declared = _declared_version()
    if declared is None:
        pytest.skip("not running from a source checkout with a readable pyproject")
    assert pn.__version__ == declared, (
        f"panelary/__init__.py says {pn.__version__}, pyproject.toml says "
        f"{declared}. Update both, or make the version dynamic."
    )


def test_version_matches_the_installed_distribution() -> None:
    """The number users see in `pip show` is the number the module reports."""
    installed = importlib.metadata.version("panelary")
    declared = _declared_version()
    if declared is not None and installed != declared:
        pytest.skip(
            f"this environment's install metadata is stale (distribution says "
            f"{installed}, pyproject says {declared}); re-run "
            f"`uv pip install -e .` to refresh it"
        )
    assert pn.__version__ == installed


# --------------------------------------------------------------------------- #
# 5. Bad input fails loudly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("verb", "kwargs"),
    [
        ("impute", {}),
        ("select", {}),
        ("bubbles", {"min_window": 20, "columns": "px"}),
        ("cluster", {}),
        ("reduce", {}),
        ("regression", {"y": "ret", "x": ["a"]}),
        ("causal", {"y": "ret", "treatment": "a"}),
    ],
)
def test_unknown_method_raises(panel: pl.DataFrame, verb: str, kwargs: dict) -> None:
    with pytest.raises(ValueError, match="unknown method"):
        getattr(pn, verb)(panel, method="not_a_method", **kwargs)


def test_select_mrmr_requires_a_target(panel: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="target"):
        pn.select(panel, method="mrmr", k=2)


def test_regression_hdfe_requires_absorb(panel: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="absorb"):
        pn.regression(panel, y="ret", x=["a"])


# --------------------------------------------------------------------------- #
# 6. The verbs shadow three subpackage names; the modules must stay reachable
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("module", "symbol"),
    [
        ("panelary.select", "mrmr"),
        ("panelary.cluster", "KShapeClusterer"),
        ("panelary.reduce", "PanelPCA"),
    ],
)
def test_shadowed_subpackages_are_still_importable(module: str, symbol: str) -> None:
    """`pn.select` is the verb, but `from panelary.select import mrmr` still works.

    Module resolution goes through ``sys.modules``, not through the parent
    package's attributes, so importing the submodule by name is unaffected by
    the verb sitting on ``panelary.select``.
    """
    assert hasattr(importlib.import_module(module), symbol)
