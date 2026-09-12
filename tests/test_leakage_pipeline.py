"""``Pipeline.audit`` / ``Pipeline.causalize`` — the pipeline face of the compiler.

A :class:`~panelary.core.pipeline.Pipeline` holds ``(name, PanelTransformer)``
tuples and nothing else, so these tests exercise the two halves of the audit
separately:

* **attribute-based** — an ordinary step is judged by the ``panel_safe`` /
  ``leakage_safe`` declaration that every concrete ``PanelTransformer`` is
  required to make. These tests need nothing from :mod:`panelary.leakage`.
* **expression-based** — a step that offers its expressions through
  ``leakage_exprs()`` is judged by the point-in-time compiler. These tests are
  skipped while :mod:`panelary.leakage._rules` is unbuilt.
"""

from __future__ import annotations

import polars as pl
import pytest

from panelary.core.panel_frame import PanelFrame
from panelary.core.pipeline import Pipeline
from panelary.core.protocol import PanelTransformer
from panelary.testing import assert_no_lookahead


# --------------------------------------------------------------------------- #
# Is the point-in-time compiler actually usable yet?
# --------------------------------------------------------------------------- #
# `panelary.leakage` imports today, but `audit()` reaches for
# `panelary.leakage._rules`, which is still being written. Probe a real call
# rather than the import so the skip reason names the true cause.
def _compiler_status() -> tuple[bool, str]:
    try:
        import panelary.leakage as leakage

        leakage.audit(pl.col("x").rolling_mean(3, center=True), time="t", entity="e")
    except Exception as exc:  # pragma: no cover - depends on build order
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


_COMPILER_OK, _COMPILER_WHY = _compiler_status()
requires_compiler = pytest.mark.skipif(
    not _COMPILER_OK,
    reason=(
        f"panelary.leakage's point-in-time compiler is not usable yet: {_COMPILER_WHY}"
    ),
)


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class _Passthrough(PanelTransformer):
    """A step that declares itself fully safe and does nothing."""

    panel_safe = True
    leakage_safe = True

    def _fit(self, panel: PanelFrame) -> None:
        return None

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        return panel


class _NotLeakageSafe(_Passthrough):
    """A step that admits it may look ahead."""

    panel_safe = True
    leakage_safe = False


class _NotPanelSafe(_Passthrough):
    """A step that admits it may mix entities."""

    panel_safe = False
    leakage_safe = True


class _ExprStep(PanelTransformer):
    """A step whose leakage surface is a bag of Polars expressions.

    Implements the optional ``leakage_exprs`` / ``with_leakage_exprs`` hook, so
    ``Pipeline.audit`` hands its expressions to the compiler. It declares
    ``leakage_safe = False`` as a conservative class-level default; offering the
    expressions is what makes the compiler, not the declaration, authoritative.
    """

    panel_safe = True
    leakage_safe = False

    def __init__(self, exprs: dict[str, pl.Expr]) -> None:
        super().__init__()
        self._exprs = dict(exprs)

    def leakage_exprs(self) -> dict[str, pl.Expr]:
        return dict(self._exprs)

    def with_leakage_exprs(self, exprs: dict[str, pl.Expr]) -> _ExprStep:
        return type(self)(exprs)

    def _fit(self, panel: PanelFrame) -> None:
        return None

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        return panel.with_columns(*[e.alias(name) for name, e in self._exprs.items()])


class _UnrepairableExprStep(_ExprStep):
    """Offers expressions but cannot be rebuilt from rewritten ones."""

    with_leakage_exprs = None  # type: ignore[assignment]


@pytest.fixture
def panel() -> PanelFrame:
    n = 8
    df = pl.DataFrame(
        {
            "e": ["a"] * n + ["b"] * n,
            "t": list(range(n)) * 2,
            "x": [float(i) for i in range(n)] + [float(2 * i + 1) for i in range(n)],
        }
    )
    return PanelFrame(df, entity="e", time="t")


def _leaky_expr() -> pl.Expr:
    """A centred rolling mean: it reads the future, and has an exact repair."""
    return pl.col("x").rolling_mean(3, center=True).over("e", order_by="t")


def _safe_expr() -> pl.Expr:
    return pl.col("x").rolling_mean(3).over("e", order_by="t")


# --------------------------------------------------------------------------- #
# Attribute-based audit (no compiler needed)
# --------------------------------------------------------------------------- #
def test_all_safe_steps_audit_clean() -> None:
    pipe = Pipeline([("a", _Passthrough()), ("b", _Passthrough())])
    report = pipe.audit()

    assert report.verdict == "safe"
    assert report.ok
    assert report.refused == ()
    assert report.rewritten == ()
    assert [s.name for s in report.steps] == ["a", "b"]
    assert all(s.verdict == "safe" and s.reasons == () for s in report.steps)


def test_leakage_unsafe_step_is_reported() -> None:
    pipe = Pipeline([("ok", _Passthrough()), ("bad", _NotLeakageSafe())])
    report = pipe.audit()

    assert report.verdict == "refused"
    assert not report.ok
    assert [s.name for s in report.refused] == ["bad"]

    bad = report.steps[1]
    assert bad.step == "_NotLeakageSafe"
    assert bad.leakage_safe is False
    assert bad.panel_safe is True
    assert bad.results == ()  # nothing was handed to the compiler
    assert any("leakage_safe = False" in r for r in bad.reasons)


def test_panel_unsafe_step_is_reported() -> None:
    report = Pipeline([("bad", _NotPanelSafe())]).audit()

    assert report.verdict == "refused"
    bad = report.steps[0]
    assert bad.panel_safe is False
    assert any("panel_safe = False" in r for r in bad.reasons)


def test_nested_pipeline_is_audited_recursively() -> None:
    inner = Pipeline([("bad", _NotLeakageSafe())])
    outer = Pipeline([("ok", _Passthrough()), ("inner", inner)])
    report = outer.audit()

    assert report.verdict == "refused"
    nested = report.steps[1].nested
    assert nested is not None
    assert [s.name for s in nested.refused] == ["bad"]


def test_causalize_refuses_an_unrepairable_step() -> None:
    from panelary.leakage import LeakageRefused

    pipe = Pipeline([("bad", _NotLeakageSafe())])
    with pytest.raises(LeakageRefused) as excinfo:
        pipe.causalize()

    assert "bad" in str(excinfo.value)
    assert excinfo.value.result.refused  # the findings name the step
    assert pipe.steps[0][1].leakage_safe is False  # original untouched


def test_causalize_returns_a_new_object_and_leaves_the_original_alone() -> None:
    step_a, step_b = _Passthrough(), _Passthrough()
    pipe = Pipeline([("a", step_a), ("b", step_b)], entity="e", time="t")
    before = list(pipe.steps)

    out = pipe.causalize()

    assert isinstance(out, Pipeline)
    assert out is not pipe
    assert out.steps is not pipe.steps
    assert pipe.steps == before  # original step list unchanged
    assert [n for n, _ in out.steps] == ["a", "b"]
    # Nothing needed rewriting, so unchanged steps are carried by reference.
    assert out[0] is step_a
    assert out[1] is step_b
    assert out._entity == "e" and out._time == "t"


def test_audit_is_additive_and_does_not_disturb_fit() -> None:
    """Auditing must not touch fitted state (it is a pure read)."""
    pipe = Pipeline([("a", _Passthrough())])
    assert pipe.audit().verdict == "safe"
    assert pipe.is_fitted is False


# --------------------------------------------------------------------------- #
# Expression-based audit (needs the compiler)
# --------------------------------------------------------------------------- #
@requires_compiler
def test_safe_expression_step_audits_clean() -> None:
    pipe = Pipeline([("f", _ExprStep({"f": _safe_expr()}))], entity="e", time="t")
    report = pipe.audit()

    assert report.verdict == "safe"
    step = report.steps[0]
    assert len(step.results) == 1
    # The declared attribute says False; offering the expressions makes the
    # compiler authoritative, and it is reported alongside for transparency.
    assert step.leakage_safe is False


@requires_compiler
def test_leaky_expression_step_is_marked_rewritten() -> None:
    pipe = Pipeline([("f", _ExprStep({"f": _leaky_expr()}))], entity="e", time="t")
    report = pipe.audit()

    assert report.verdict == "rewritten"
    assert report.ok
    assert [s.name for s in report.rewritten] == ["f"]
    assert report.steps[0].reasons  # says which expression and why


@requires_compiler
def test_expression_step_without_a_rebuild_hook_is_refused() -> None:
    pipe = Pipeline(
        [("f", _UnrepairableExprStep({"f": _leaky_expr()}))], entity="e", time="t"
    )
    report = pipe.audit()

    assert report.verdict == "refused"
    assert any("with_leakage_exprs" in r for r in report.steps[0].reasons)


@requires_compiler
def test_causalize_rewrites_the_leaky_expression_step(panel: PanelFrame) -> None:
    leaky_step = _ExprStep({"f": _leaky_expr()})
    pipe = Pipeline([("f", leaky_step)], entity="e", time="t")

    out = pipe.causalize()

    assert out is not pipe
    assert out[0] is not leaky_step
    # The original still holds the original expression.
    assert pipe[0] is leaky_step
    assert str(leaky_step.leakage_exprs()["f"]) == str(_leaky_expr())
    assert out.audit().verdict == "safe"


@requires_compiler
def test_causalize_end_to_end_removes_the_lookahead(panel: PanelFrame) -> None:
    """The original leaks under the future-perturbation test; the result does not."""
    pipe = Pipeline([("f", _ExprStep({"f": _leaky_expr()}))], entity="e", time="t")

    with pytest.raises(AssertionError, match="LOOK-AHEAD LEAK DETECTED"):
        assert_no_lookahead(lambda frame: pipe.fit_transform(frame), panel)

    safe_pipe = pipe.causalize()
    assert_no_lookahead(lambda frame: safe_pipe.fit_transform(frame), panel)


@requires_compiler
def test_causalize_is_idempotent() -> None:
    pipe = Pipeline([("f", _ExprStep({"f": _leaky_expr()}))], entity="e", time="t")
    once = pipe.causalize()
    twice = once.causalize()

    assert twice.audit().verdict == "safe"
    assert str(twice[0].leakage_exprs()["f"]) == str(once[0].leakage_exprs()["f"])
