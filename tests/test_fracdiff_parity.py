"""Parity + causality tests for the consolidated fractional-differencing surfaces.

After consolidation there is exactly ONE weight recursion
(:func:`polars_features._ffd.ffd_weights`) and ONE causal expression builder
(:func:`polars_features._ffd.frac_diff_expr`). Every public frac-diff surface

* the ``.panel.frac_diff`` expression namespace,
* the frame-level ``.panel.frac_diff`` namespace,
* the ``.ts.frac_diff`` expression (feature_extractors shim),
* :class:`polars_features.transform.frac_diff.FracDiff`, and
* :func:`polars_features.preprocessing.fractional_diff`

must therefore produce numerically identical output on the same panel, with
nulls in identical positions, and must be strictly causal per entity.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_features.feature_extractors  # noqa: F401  (registers .ts namespace)
import polars_features.namespaces  # noqa: F401  (registers .panel namespace)
from polars_features._ffd import ffd_weights, frac_diff_expr
from polars_features.core import PanelFrame
from polars_features.preprocessing import fractional_diff
from polars_features.transform.frac_diff import FracDiff

# (d, threshold) combos whose kernel width fits inside the panel below so each
# surface produces a non-trivial (non-all-null) tail to compare.
CASES = [
    (0.4, 1e-4),
    (0.5, 1e-4),
    (0.9, 1e-4),
    (0.9, 1e-5),
    (1.0, 1e-4),
]

N_TIME = 500
ENTITIES = ("A", "B")


def _make_panel(n_time: int = N_TIME) -> pl.DataFrame:
    rng = np.random.default_rng(20260904)
    frames = []
    for i, ent in enumerate(ENTITIES):
        # Distinct random-walk price path per entity.
        steps = rng.standard_normal(n_time) * 0.5 + 0.01 * (i + 1)
        px = 100.0 + np.cumsum(steps)
        frames.append(
            pl.DataFrame(
                {
                    "entity": [ent] * n_time,
                    "time": np.arange(n_time, dtype=np.int64),
                    "px": px,
                }
            )
        )
    return pl.concat(frames).sort("entity", "time")


def _panel_expr_out(df: pl.DataFrame, d: float, thr: float) -> np.ndarray:
    out = df.with_columns(
        pl.col("px").panel.frac_diff(d, threshold=thr).over("entity").alias("fd")
    )
    return out.get_column("fd").to_numpy()


def _panel_frame_out(df: pl.DataFrame, d: float, thr: float) -> np.ndarray:
    out = df.panel.frac_diff("px", d=d, over="entity", threshold=thr, alias="fd")
    return out.get_column("fd").to_numpy()


def _ts_expr_out(df: pl.DataFrame, d: float, thr: float) -> np.ndarray:
    out = df.with_columns(
        pl.col("px").ts.frac_diff(d, min_weight=thr).over("entity").alias("fd")
    )
    return out.get_column("fd").to_numpy()


def _fracdiff_class_out(df: pl.DataFrame, d: float, thr: float) -> np.ndarray:
    panel = PanelFrame(df, entity="entity", time="time").sort_panel()
    out = FracDiff(columns=["px"], d=d, threshold=thr, suffix="_fd").fit_transform(
        panel
    )
    return out.lazy().collect().get_column("px_fd").to_numpy()


def _fractional_diff_out(df: pl.DataFrame, d: float, thr: float) -> np.ndarray:
    # ``fractional_diff`` replaces the numeric column in place.
    res = df.lazy().pipe(fractional_diff(d=d, min_weight=thr)).collect()
    return res.get_column("px").to_numpy()


ALL_SURFACES = {
    "panel_expr": _panel_expr_out,
    "panel_frame": _panel_frame_out,
    "ts_expr": _ts_expr_out,
    "FracDiff": _fracdiff_class_out,
    "fractional_diff": _fractional_diff_out,
}


@pytest.mark.parametrize(("d", "thr"), CASES)
def test_all_surfaces_numerically_identical(d: float, thr: float) -> None:
    df = _make_panel()
    outputs = {name: fn(df, d, thr) for name, fn in ALL_SURFACES.items()}

    reference = outputs["panel_expr"]
    # Guard against a vacuous (all-null) comparison: the chosen cases must leave
    # a non-null tail.
    assert np.isfinite(reference).any(), f"case d={d}, thr={thr} produced all nulls"

    for name, arr in outputs.items():
        assert arr.shape == reference.shape, name
        # Nulls become NaN in the numpy view; require identical null positions.
        np.testing.assert_array_equal(
            np.isnan(arr), np.isnan(reference), err_msg=f"null mask mismatch: {name}"
        )
        np.testing.assert_allclose(
            arr,
            reference,
            atol=1e-9,
            rtol=0.0,
            equal_nan=True,
            err_msg=f"value mismatch: {name} vs panel_expr (d={d}, thr={thr})",
        )


@pytest.mark.parametrize("d", [0.4, 0.9, 1.0])
def test_per_entity_causality_future_rows_do_not_change_past(d: float) -> None:
    """A value at time ``t`` is unchanged when future rows are appended."""
    thr = 1e-3
    short = _make_panel(n_time=200)

    # Append 80 future rows to entity "A" only.
    rng = np.random.default_rng(7)
    extra_px = 100.0 + np.cumsum(rng.standard_normal(80) * 0.5)
    extra = pl.DataFrame(
        {
            "entity": ["A"] * 80,
            "time": np.arange(200, 280, dtype=np.int64),
            "px": extra_px,
        }
    )
    extended = pl.concat([short, extra]).sort("entity", "time")

    for fn in ALL_SURFACES.values():
        base = fn(short, d, thr)
        ext = fn(extended, d, thr)

        # Rows of entity A in the SHORT frame line up with the first 200 A-rows
        # of the extended frame (both sorted entity, time; A sorts first).
        base_a = base[:200]
        ext_a = ext[:200]
        np.testing.assert_allclose(base_a, ext_a, atol=1e-9, rtol=0.0, equal_nan=True)


def test_weight_recursion_reference_values() -> None:
    # d=0 is the identity kernel.
    np.testing.assert_allclose(ffd_weights(0.0, 1e-5), np.array([1.0]))
    # d=1 is the first difference: oldest-to-newest [-1, 1].
    np.testing.assert_allclose(ffd_weights(1.0, 1e-5), np.array([-1.0, 1.0]))


def test_frac_diff_expr_requires_d_or_weights() -> None:
    with pytest.raises(ValueError, match="either `d` or `weights`"):
        frac_diff_expr(pl.col("x"))


def test_no_entity_boundary_bleed() -> None:
    """Entity B's leading rows must not consume entity A's tail (d=1 first diff)."""
    df = pl.DataFrame(
        {
            "entity": ["A", "A", "A", "B", "B", "B"],
            "time": [0, 1, 2, 0, 1, 2],
            "px": [1.0, 2.0, 4.0, 10.0, 13.0, 17.0],
        }
    ).sort("entity", "time")
    out = _panel_expr_out(df, d=1.0, thr=1e-3)
    # width == 2 => first row of each entity is null; B[0] must be null, NOT
    # 10.0 - 4.0 (which would be a cross-entity leak).
    assert np.isnan(out[0])  # A[0]
    assert np.isnan(out[3])  # B[0]
    # First differences within each entity.
    np.testing.assert_allclose(out[1:3], [1.0, 2.0], atol=1e-9)
    np.testing.assert_allclose(out[4:6], [3.0, 4.0], atol=1e-9)
