"""Hotspot profiler for Panelary's public operations on a synthetic panel.

Where ``bench_vs_pandas.py`` answers "are we faster than pandas?", this script
answers "where does *our own* time go?" -- the measurement the improvement
playbook's Wave 0 asks for before any optimisation round.  It builds a synthetic
long-format panel (entity, time, features) at a chosen size and times the
golden-path operations, printing a ranked table.

Only ``numpy`` and ``polars`` are required.  Workloads that need an optional
extra (scikit-learn for ``reduce``/``cluster``) are reported as ``skipped`` when
the extra is absent, so this runs unchanged in the bare-core CI variant.

Run::

    python benchmarks/bench_hotspots.py                 # 0.5M and 2.5M rows
    python benchmarks/bench_hotspots.py --rows 200000   # one smaller size
    python benchmarks/bench_hotspots.py --only catch22
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np
import polars as pl

import panelary  # noqa: F401  (registers the .panel/.xs/.ts namespaces)
from panelary._deps import have

RNG = np.random.default_rng(7)

#: Sizes from the playbook's speed harness: ~0.5M and ~2.5M rows.
DEFAULT_SIZES = ((2_000, 260), (5_000, 500))


def make_panel(n_entities: int, n_time: int, n_features: int = 6) -> pl.DataFrame:
    """Long-format panel with per-entity random walks plus iid feature columns."""
    n = n_entities * n_time
    steps = RNG.standard_normal(n)
    value = steps.reshape(n_entities, n_time).cumsum(axis=1).reshape(-1) + 100.0
    data = {
        "entity": np.repeat(np.arange(n_entities), n_time),
        "time": np.tile(np.arange(n_time), n_entities),
        "value": value,
    }
    for j in range(n_features):
        data[f"f{j}"] = RNG.standard_normal(n)
    return pl.DataFrame(data)


def best_of(fn: Callable[[], object], repeat: int = 3) -> float:
    """Best wall-clock seconds over ``repeat`` runs (one warm-up first)."""
    fn()
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


# --------------------------------------------------------------------------- #
# Workloads
# --------------------------------------------------------------------------- #
def _rolling_zscore(panel: pl.DataFrame):
    return panel.select(
        "entity",
        "time",
        (
            (pl.col("value") - pl.col("value").rolling_mean(21))
            / pl.col("value").rolling_std(21)
        )
        .over("entity")
        .alias("z"),
    )


def _cross_sectional_rank(panel: pl.DataFrame):
    return panel.select(
        "entity", "time", pl.col("value").rank().over("time").alias("r")
    )


def _ts_features(panel: pl.DataFrame):
    return panel.group_by("entity", maintain_order=True).agg(
        pl.col("value").ts.absolute_energy().alias("abs_energy"),
        pl.col("value").ts.mean_abs_change().alias("mean_abs_change"),
        pl.col("value").ts.count_above_mean().alias("count_above_mean"),
        pl.col("value").ts.root_mean_square().alias("rms"),
        pl.col("value").ts.longest_streak_above_mean().alias("streak"),
    )


def _catch22_per_entity(panel: pl.DataFrame, n_entities: int):
    from panelary.catch22 import catch22_all

    # catch22 is O(entities) Python calls by construction; profile a capped
    # sample so the whole harness stays interactive at 2.5M rows.
    sample = min(n_entities, 200)
    values = (
        panel.filter(pl.col("entity") < sample)
        .group_by("entity")
        .agg(pl.col("value"))
        .get_column("value")
    )
    return [catch22_all(v.to_numpy()) for v in values]


def _entropy_features(panel: pl.DataFrame):
    from panelary import feature_extractors as fe

    series = panel.filter(pl.col("entity") < 20).group_by("entity").agg(pl.col("value"))
    out = []
    for v in series.get_column("value"):
        s = v.explode() if v.dtype == pl.List else v
        out.append(fe.sample_entropy(pl.Series("v", s.to_numpy()), 0.2, 2))
    return out


def _cross_sectional_pca(panel: pl.DataFrame):
    from panelary.reduce.xs import CrossSectionalPCA

    cols = [c for c in panel.columns if c.startswith("f")]
    reducer = CrossSectionalPCA(
        n_components=3, columns=cols, entity="entity", time="time"
    )
    return reducer.fit(panel).transform(panel).collect()


def _cross_sectional_cluster(panel: pl.DataFrame):
    from panelary.cluster import CrossSectionalClusterer

    cols = [c for c in panel.columns if c.startswith("f")]
    model = CrossSectionalClusterer(cols, n_clusters=6, entity="entity", time="time")
    return model.fit(panel).predict(panel).collect()


def _kshape(panel: pl.DataFrame, n_entities: int):
    from panelary.cluster._kshape import KShapeCore

    sample = min(n_entities, 300)
    width = int(panel.get_column("time").max()) + 1
    matrix = (
        panel.filter(pl.col("entity") < sample)
        .sort("entity", "time")
        .get_column("value")
        .to_numpy()
        .reshape(sample, width)
    )
    return KShapeCore(4, seed=1, max_iter=5).fit(matrix[:, : min(width, 256)])


WORKLOADS: dict[str, tuple[str, Callable, tuple[str, ...]]] = {
    "rolling": (".ts rolling z-score per entity", _rolling_zscore, ()),
    "xs_rank": (".xs cross-sectional rank per date", _cross_sectional_rank, ()),
    "ts_features": (".ts bulk feature aggregation", _ts_features, ()),
    "catch22": ("catch22_all (<=200 entities)", _catch22_per_entity, ()),
    "entropy": ("sample_entropy (20 entities)", _entropy_features, ()),
    "reduce_xs": ("reduce.CrossSectionalPCA", _cross_sectional_pca, ("sklearn",)),
    "cluster_xs": (
        "cluster.CrossSectionalClusterer",
        _cross_sectional_cluster,
        ("sklearn",),
    ),
    "kshape": ("cluster k-Shape fit (5 iters)", _kshape, ()),
}

#: Workloads whose callable also takes the entity count.
_NEEDS_N_ENTITIES = {"catch22", "kshape"}


def run(n_entities: int, n_time: int, only: str | None) -> None:
    panel = make_panel(n_entities, n_time)
    rows = panel.height
    print(f"\n### Panel: {n_entities:,} entities x {n_time:,} steps = {rows:,} rows")
    print(f"{'operation':<40}{'seconds':>10}{'us / 1k rows':>16}")
    print("-" * 66)

    results: list[tuple[str, float]] = []
    for key, (label, fn, needs) in WORKLOADS.items():
        if only and only != key:
            continue
        missing = [module for module in needs if not have(module)]
        if missing:
            print(f"{label:<40}{'skipped':>10}   (needs {', '.join(missing)})")
            continue
        call = (
            (lambda fn=fn: fn(panel, n_entities))
            if key in _NEEDS_N_ENTITIES
            else (lambda fn=fn: fn(panel))
        )
        seconds = best_of(call, repeat=2)
        results.append((label, seconds))
        print(f"{label:<40}{seconds:>10.3f}{seconds / rows * 1e9:>16.1f}")

    if results:
        print("-" * 66)
        for label, seconds in sorted(results, key=lambda r: -r[1])[:5]:
            print(f"  slowest: {label:<38}{seconds:>8.3f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="approximate row count for a single run (default: 0.5M and 2.5M)",
    )
    parser.add_argument(
        "--only",
        choices=sorted(WORKLOADS),
        default=None,
        help="profile a single workload",
    )
    args = parser.parse_args()

    print("Panelary hotspot profile")
    print(f"polars {pl.__version__} | numpy {np.__version__}")

    if args.rows is not None:
        n_time = 260
        n_entities = max(2, args.rows // n_time)
        run(n_entities, n_time, args.only)
    else:
        for n_entities, n_time in DEFAULT_SIZES:
            run(n_entities, n_time, args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
