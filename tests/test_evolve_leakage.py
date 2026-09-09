"""Leakage and correctness regressions for `panelary.evolve`.

These are the tests that would fail under a deliberately leaky implementation,
per the contract in `AGENTS.md`. Three distinct hazards are covered:

1. **Structural leak-safety** -- every operator the grammar can emit is
   registered ``leakage_safe``, so a look-ahead feature is unconstructible
   rather than merely discouraged.
2. **Prefix invariance** -- ``f(x[:T])[t] == f(x[:T+k])[t]``. This is the
   executable form of "no lookahead, ever", and it is what
   ``evolve.assert_causal`` checks.
3. **The nested-``over`` silent-null hazard** -- on polars 1.44.1,
   ``x.rolling_mean(2).over(entity).rank().over(time)`` evaluates to *all
   nulls, with no error*. The compiler must stage partition-scope switches
   into separate passes. This is a silent-corruption bug, so it gets a
   dedicated regression.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

evolve = pytest.importorskip("panelary.evolve")


def _panel(n_entities: int = 20, n_times: int = 60, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    ents = np.repeat(np.arange(n_entities), n_times)
    times = np.tile(np.arange(n_times), n_entities)
    n = ents.size
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pl.DataFrame(
        {
            "ticker": ents.astype(np.int64),
            "date": times.astype(np.int64),
            "close": close,
            "volume": rng.lognormal(10.0, 0.4, n),
            "fwd_ret": rng.standard_normal(n) * 0.01,
        }
    ).sort(["ticker", "date"])


class TestStructuralLeakSafety:
    def test_every_operator_is_leakage_safe(self) -> None:
        """The grammar must not contain a single non-causal primitive."""
        for name, op in evolve.OPS.items():
            assert op.leakage_safe, f"{name} is not leakage_safe"

    def test_operators_do_not_apply_over_themselves(self) -> None:
        """`Op.build` must return a bare expression.

        Grouping belongs to the compiler: it is what gets hoisted and shared
        across the population, and what must be staged to avoid the
        nested-``over`` hazard. An operator that applies its own ``.over()``
        would silently defeat both.
        """
        for name, op in evolve.OPS.items():
            if op.arity != 1 or op.kind == "elem":
                continue
            param = op.params[0] if op.params else None
            expr = (
                op.build(pl.col("close"), param)
                if param is not None
                else op.build(pl.col("close"))
            )
            rendered = str(expr)
            assert ".over(" not in rendered, f"{name} applies its own .over()"


class TestUnitTyping:
    """Dimensional typing must make meaningless features unconstructible.

    Montana (1995) strongly-typed GP, extended to physical units by Keijzer &
    Babovic (1999). The payoff is not tidiness: Durasevic et al.
    (arXiv:2004.12762) show dimensional awareness measurably *smooths* the
    fitness landscape, so this buys search efficiency too.

    Regression for a real defect -- the first implementation declared the whole
    additive family ``("any", "any")``, which silently allowed ``close +
    volume`` while the documentation claimed otherwise.
    """

    @staticmethod
    def _ctx():
        return evolve.EvalContext(
            base_columns=("close", "volume"),
            base_units=("price", "volume"),
            entity="ticker",
            time="date",
        )

    @staticmethod
    def _accepts(name: str) -> bool:
        from panelary.evolve._compile import validate
        from panelary.evolve._types import Gene, Genome

        try:
            validate(
                Genome(genes=(Gene(op=name, args=(0, 1), param_ix=0),), n_base=2),
                TestUnitTyping._ctx(),
                max_depth=5,
            )
        except Exception:
            return False
        return True

    def test_additive_ops_reject_incommensurable_units(self) -> None:
        """price (+) volume must be rejected, not merely penalised."""
        additive = [
            n
            for n, op in evolve.OPS.items()
            if op.arity == 2
            and any(n.startswith(p) for p in ("add", "sub", "max", "min"))
        ]
        assert additive, "no additive operators found"
        for name in additive:
            assert not self._accepts(name), f"{name} accepted (price, volume)"

    def test_ratio_and_correlation_stay_permissive(self) -> None:
        """A quotient or correlation of unlike units is well defined.

        ``ts_corr(close, volume, d)`` is an Alpha101 staple; forbidding it
        would remove a whole family of real alphas for no gain.
        """
        for name in ("div", "ts_corr"):
            if name in evolve.OPS:
                assert self._accepts(name), f"{name} should accept (price, volume)"

    def test_unannotated_panel_gets_a_usable_grammar(self) -> None:
        """`any` stays polymorphic, so users need not annotate to get started."""
        grammar = evolve.default_grammar(("any", "any"))
        assert len(grammar) == len(evolve.OPS)


class TestPrefixInvariance:
    @pytest.mark.parametrize("cut", [30, 45])
    def test_time_series_operators_are_prefix_invariant(self, cut: int) -> None:
        """f(x[:T])[t] must equal f(x[:T+k])[t] for every t <= T."""
        full = _panel(6, 60)
        head = full.filter(pl.col("date") < cut)
        ts_ops = [op for op in evolve.OPS.values() if op.kind == "ts" and op.arity == 1]
        assert ts_ops, "no unary ts operators registered"
        for op in ts_ops:
            param = op.params[0] if op.params else None
            expr = (
                op.build(pl.col("close"), param)
                if param is not None
                else op.build(pl.col("close"))
            )
            fa = full.with_columns(expr.over("ticker").alias("v")).filter(
                pl.col("date") < cut
            )["v"]
            fb = head.with_columns(expr.over("ticker").alias("v"))["v"]
            va = np.asarray(fa.to_numpy(), dtype=float)
            vb = np.asarray(fb.to_numpy(), dtype=float)
            mask = np.isfinite(va) & np.isfinite(vb)
            assert np.allclose(va[mask], vb[mask], atol=1e-9), (
                f"{op.name} is not prefix invariant"
            )


class TestNestedOverHazard:
    def test_raw_polars_nested_over_is_all_null(self) -> None:
        """Document the upstream hazard this module exists to work around.

        If this test ever *fails*, polars has fixed the bug and the compiler's
        staging requirement can be revisited.
        """
        df = _panel(5, 12)
        nested = df.with_columns(
            pl.col("close")
            .rolling_mean(2)
            .over("ticker")
            .rank()
            .over("date")
            .alias("n")
        )
        assert nested["n"].is_not_null().sum() == 0, (
            "polars nested-over hazard appears fixed upstream; revisit _compile staging"
        )

    def test_compiler_stages_partition_switches(self) -> None:
        """The compiler must produce a non-null result for cs_rank(ts_mean(x))."""
        pytest.importorskip("panelary.evolve._compile")
        df = _panel(8, 30)
        ctx = evolve.EvalContext(
            base_columns=("close", "volume"),
            base_units=("price", "volume"),
            entity="ticker",
            time="date",
        )
        genome = _cs_rank_of_ts_mean(ctx)
        if genome is None:
            pytest.skip("grammar lacks cs_rank/ts_mean")
        lf, names = evolve.compile_population([genome], ctx, df.lazy())
        out = lf.collect()[names[0]]
        assert out.is_not_null().sum() > 0, (
            "compiler produced an all-null feature: the nested-over hazard is not staged"
        )


def _cs_rank_of_ts_mean(ctx):
    """Build cs_rank(ts_mean(close, 5)) directly, or None if unavailable."""
    from panelary.evolve._types import Gene, Genome

    if "ts_mean" not in evolve.OPS or "cs_rank" not in evolve.OPS:
        return None
    ts_mean = evolve.OPS["ts_mean"]
    try:
        param_ix = list(ts_mean.params).index(5)
    except ValueError:
        param_ix = 0
    n_base = len(ctx.base_columns)
    return Genome(
        genes=(
            Gene(op="ts_mean", args=(0,), param_ix=param_ix),
            Gene(op="cs_rank", args=(n_base,), param_ix=0),
        ),
        n_base=n_base,
    )
