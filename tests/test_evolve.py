"""Behavioural tests for `panelary.evolve`.

The tests that matter here are not "does the search run" but "does it tell the
truth". A feature miner that cannot distinguish a planted signal from noise --
or worse, reports signal in noise -- is an expensive random number generator.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

evolve = pytest.importorskip("panelary.evolve")


def _panel(*, seed: int, planted: bool, n_entities: int = 60, n_times: int = 400):
    rng = np.random.default_rng(seed)
    ents = np.repeat(np.arange(n_entities), n_times)
    times = np.tile(np.arange(n_times), n_entities)
    n = ents.size
    close = np.empty(n)
    for e in range(n_entities):
        sl = slice(e * n_times, (e + 1) * n_times)
        close[sl] = 100.0 + np.cumsum(rng.standard_normal(n_times) * 0.5)
    df = pl.DataFrame(
        {
            "ticker": ents.astype(np.int64),
            "date": times.astype(np.int64),
            "close": close,
            "volume": rng.lognormal(10.0, 0.4, n),
        }
    ).sort(["ticker", "date"])
    mom = np.nan_to_num(
        df.select(pl.col("close").pct_change(5).over("ticker"))["close"].to_numpy()
    )
    noise = rng.standard_normal(n) * 0.02
    y = (-0.5 * mom + noise) if planted else noise
    return df.with_columns(pl.Series("fwd_ret", y))


def _run(planted: bool):
    cfg = evolve.EvolveConfig(
        population=60, generations=4, n_islands=2, max_library=10, seed=1
    )
    return evolve.evolve_features(
        _panel(seed=0, planted=planted),
        target="fwd_ret",
        entity="ticker",
        time="date",
        config=cfg,
    )


@pytest.mark.slow
class TestSignalVersusNoise:
    """The load-bearing behaviour: the report must not claim signal in noise."""

    def test_planted_signal_is_found_and_generalises(self) -> None:
        res = _run(planted=True)
        assert res.library, "no features returned"
        assert res.diagnostics["holdout_mean_ic"] > 0.0
        assert "NO SIGNAL" not in res.diagnostics["verdict"]

    def test_pure_noise_is_reported_as_noise(self) -> None:
        """A search over noise must say so, however good the best score looks."""
        res = _run(planted=False)
        assert res.diagnostics["holdout_mean_ic"] < res.ledger["best_score"], (
            "held-out performance should collapse relative to the in-search best"
        )
        assert res.ledger["verdict"] != "PASS"

    def test_holdout_is_time_disjoint_from_the_search(self) -> None:
        """The generalisation check must use dates the search never saw.

        Correlating two halves of the same cross-validated score measures
        consistency, not generalisation, and reports "signal" on pure noise --
        this asserts we did not regress to that.
        """
        from panelary.evolve._search import _time_split

        df = _panel(seed=0, planted=True, n_entities=5, n_times=100)
        train, held = _time_split(df.lazy(), "date", 0.2)
        assert held is not None
        tmax = train.select(pl.col("date").max()).collect().item()
        hmin = held.select(pl.col("date").min()).collect().item()
        assert hmin > tmax, "holdout overlaps the search window"


class TestReportingContract:
    def test_summary_reports_the_deflated_number(self) -> None:
        """The raw best score must never appear without its deflation."""
        res = evolve.EvolveResult(
            ledger={
                "n_trials": 1000.0,
                "best_score": 0.9,
                "deflated_sharpe": 0.01,
                "pbo": 0.6,
                "verdict": "FAIL",
            }
        )
        text = res.summary()
        for token in ("deflated", "verdict", "candidates evaluated"):
            assert token in text.lower()

    def test_empty_result_frame_has_a_schema(self) -> None:
        assert evolve.EvolveResult().to_frame().height == 0


class TestTrialAccounting:
    def test_every_candidate_is_counted(self) -> None:
        """Search-count amnesia is the defining failure of this literature.

        `M` must equal population x generations x islands, including duplicates
        and failed evaluations -- dropping them understates N and under-deflates.
        """
        cfg = evolve.EvolveConfig(
            population=20, generations=2, n_islands=2, max_library=5, seed=3
        )
        res = evolve.evolve_features(
            _panel(seed=1, planted=True, n_entities=20, n_times=120),
            target="fwd_ret",
            entity="ticker",
            time="date",
            config=cfg,
        )
        assert res.ledger["n_trials"] == pytest.approx(20 * 2 * 2)
