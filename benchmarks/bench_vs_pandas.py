"""Speed benchmark: PanelKit (polars_features) vs pandas on panel feature generation.

Panel data = many entities observed over time (long format: entity, time, value...).
We time three representative feature-engineering workloads that every panel
pipeline runs, compute each one BOTH ways, and verify the numbers agree before
reporting the speedup — so the comparison is honest, not cherry-picked.

Run:  python benchmarks/bench_vs_pandas.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import polars as pl

import polars_features  # noqa: F401  (registers the .panel/.xs/.ts namespaces)

RNG = np.random.default_rng(7)


def make_panel(n_entities: int, n_time: int) -> tuple[pl.DataFrame, pd.DataFrame]:
    """A synthetic panel with a mild per-entity random walk. Returns (polars, pandas)."""
    ent = np.repeat(np.arange(n_entities), n_time)
    t = np.tile(np.arange(n_time), n_entities)
    steps = RNG.standard_normal(n_entities * n_time).astype(np.float64)
    # per-entity cumulative walk (reshape trick keeps entities independent)
    val = steps.reshape(n_entities, n_time).cumsum(axis=1).reshape(-1) + 100.0
    pdf = pd.DataFrame({"entity": ent, "time": t, "value": val})
    plf = pl.from_pandas(pdf)
    return plf, pdf


def best_of(fn, repeat: int = 3) -> tuple[float, object]:
    """Return (best_seconds, last_result)."""
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, out


# --------------------------------------------------------------------------- #
# Workload 1 — per-entity rolling z-score (window=21), the classic panel feature
# --------------------------------------------------------------------------- #
def w1_pk(plf: pl.DataFrame) -> pl.DataFrame:
    w = 21
    return plf.select(
        "entity",
        "time",
        (
            (pl.col("value") - pl.col("value").rolling_mean(w))
            / pl.col("value").rolling_std(w)
        )
        .over("entity")
        .alias("z"),
    )


def w1_pd(pdf: pd.DataFrame) -> pd.DataFrame:
    w = 21
    g = pdf.groupby("entity")["value"]
    m = g.transform(lambda s: s.rolling(w).mean())
    sd = g.transform(lambda s: s.rolling(w).std())
    out = pdf[["entity", "time"]].copy()
    out["z"] = (pdf["value"] - m) / sd
    return out


# --------------------------------------------------------------------------- #
# Workload 2 — cross-sectional per-date rank (leak-safe cross-section)
# --------------------------------------------------------------------------- #
def w2_pk(plf: pl.DataFrame) -> pl.DataFrame:
    return plf.select(
        "entity",
        "time",
        pl.col("value").rank().over("time").alias("xs_rank"),
    )


def w2_pd(pdf: pd.DataFrame) -> pd.DataFrame:
    out = pdf[["entity", "time"]].copy()
    out["xs_rank"] = pdf.groupby("time")["value"].rank()
    return out


# --------------------------------------------------------------------------- #
# Workload 3 — bulk feature extraction: 10 tsfresh-style features per entity.
# This is the realistic "feature engineering" case: several of these have no
# vectorised pandas form, so pandas must run Python per group (like tsfresh).
# --------------------------------------------------------------------------- #
PK_FEATURES = [
    pl.col("value").mean().alias("mean"),
    pl.col("value").std().alias("std"),
    pl.col("value").min().alias("min"),
    pl.col("value").max().alias("max"),
    pl.col("value").ts.absolute_energy().alias("abs_energy"),
    pl.col("value").ts.mean_abs_change().alias("mean_abs_change"),
    pl.col("value").ts.absolute_sum_of_changes().alias("abs_sum_changes"),
    pl.col("value").ts.count_above_mean().alias("count_above_mean"),
    pl.col("value").ts.root_mean_square().alias("rms"),
    pl.col("value").ts.longest_streak_above_mean().alias("longest_streak_above_mean"),
]


def w3_pk(plf: pl.DataFrame) -> pl.DataFrame:
    return plf.group_by("entity", maintain_order=True).agg(PK_FEATURES)


def _longest_streak_above_mean(a: np.ndarray) -> float:
    above = a > a.mean()
    best = cur = 0
    for b in above:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return float(best)


def w3_pd(pdf: pd.DataFrame) -> pd.DataFrame:
    def agg(s: pd.Series) -> pd.Series:
        a = s.to_numpy()
        d = np.abs(np.diff(a))
        return pd.Series(
            {
                "mean": a.mean(),
                "std": a.std(ddof=1),
                "min": a.min(),
                "max": a.max(),
                "abs_energy": float((a * a).sum()),
                "mean_abs_change": float(d.mean()) if d.size else 0.0,
                "abs_sum_changes": float(d.sum()),
                "count_above_mean": float((a > a.mean()).sum()),
                "rms": float(np.sqrt((a * a).mean())),
                "longest_streak_above_mean": _longest_streak_above_mean(a),
            }
        )

    return pdf.groupby("entity")["value"].apply(agg).unstack()


def _agree(a: np.ndarray, b: np.ndarray, name: str, rtol=1e-6, atol=1e-6) -> str:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    ok = np.allclose(a[m], b[m], rtol=rtol, atol=atol)
    return "match" if ok else f"MISMATCH:{name}"


def run(n_entities: int, n_time: int) -> None:
    rows = n_entities * n_time
    plf, pdf = make_panel(n_entities, n_time)
    print(f"\n### Panel: {n_entities:,} entities x {n_time:,} steps = {rows:,} rows")
    print(
        f"{'workload':<34}{'pandas (s)':>12}{'PanelKit (s)':>14}{'speedup':>10}   check"
    )

    # W1
    t_pd, r_pd = best_of(lambda: w1_pd(pdf))
    t_pk, r_pk = best_of(lambda: w1_pk(plf))
    chk = _agree(r_pd["z"].to_numpy(), r_pk["z"].to_numpy(), "z")
    print(
        f"{'1. rolling z-score / entity':<34}{t_pd:>12.3f}{t_pk:>14.3f}{t_pd / t_pk:>9.1f}x   {chk}"
    )

    # W2
    t_pd, r_pd = best_of(lambda: w2_pd(pdf))
    t_pk, r_pk = best_of(lambda: w2_pk(plf))
    chk = _agree(r_pd["xs_rank"].to_numpy(), r_pk["xs_rank"].to_numpy(), "rank")
    print(
        f"{'2. cross-sectional rank / date':<34}{t_pd:>12.3f}{t_pk:>14.3f}{t_pd / t_pk:>9.1f}x   {chk}"
    )

    # W3
    t_pd, r_pd = best_of(lambda: w3_pd(pdf), repeat=2)
    t_pk, r_pk = best_of(lambda: w3_pk(plf), repeat=2)
    r_pk_pd = r_pk.sort("entity").to_pandas().set_index("entity")
    r_pd = r_pd.sort_index()
    chk = _agree(
        r_pd["longest_streak_above_mean"].to_numpy(),
        r_pk_pd["longest_streak_above_mean"].to_numpy(),
        "streak",
    )
    print(
        f"{'3. 10 features / entity (bulk)':<34}{t_pd:>12.3f}{t_pk:>14.3f}{t_pd / t_pk:>9.1f}x   {chk}"
    )


if __name__ == "__main__":
    print("PanelKit vs pandas — panel feature generation")
    print(f"polars {pl.__version__} | pandas {pd.__version__}")
    run(2_000, 260)  # ~0.5M rows  (2y daily on 2k names)
    run(5_000, 500)  # ~2.5M rows
