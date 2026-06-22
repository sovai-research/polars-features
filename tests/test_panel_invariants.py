"""Property-based and golden tests for PanelFrame invariants.

These exercise the *correctness-by-construction* core in
``polars_features.core.panel_frame`` only. They require no compiled Rust
extension (pure-Python Phase-1 core).
"""

from __future__ import annotations

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polars_features.core.panel_frame import PanelFrame, as_panel

# --------------------------------------------------------------------------- #
# Hypothesis strategies for small synthetic panels
# --------------------------------------------------------------------------- #
ENTITIES = ["A", "B", "C"]


@st.composite
def panel_rows(draw):
    """Draw a small panel as a list of (entity, time, value) rows.

    Each entity may appear with a different (possibly unsorted, possibly
    sparse) set of integer times. Keys are kept unique per (entity, time).
    """
    n = draw(st.integers(min_value=1, max_value=24))
    rows = []
    seen = set()
    for _ in range(n):
        ent = draw(st.sampled_from(ENTITIES))
        t = draw(st.integers(min_value=0, max_value=15))
        if (ent, t) in seen:
            continue
        seen.add((ent, t))
        val = draw(st.floats(min_value=-100, max_value=100, allow_nan=False))
        rows.append((ent, t, val))
    # Guarantee at least one row.
    if not rows:
        rows.append(("A", 0, 0.0))
    return rows


def _df_from_rows(rows):
    return pl.DataFrame(
        {
            "entity": [r[0] for r in rows],
            "time": [r[1] for r in rows],
            "value": [r[2] for r in rows],
        }
    )


# --------------------------------------------------------------------------- #
# Construction / schema validation
# --------------------------------------------------------------------------- #
def test_construction_rejects_missing_entity():
    df = pl.DataFrame({"time": [1, 2], "value": [0.1, 0.2]})
    with pytest.raises(ValueError) as exc:
        PanelFrame(df, entity="entity", time="time")
    assert "entity" in str(exc.value)


def test_construction_rejects_missing_time():
    df = pl.DataFrame({"entity": ["A", "B"], "value": [0.1, 0.2]})
    with pytest.raises(ValueError) as exc:
        PanelFrame(df, entity="entity", time="time")
    assert "time" in str(exc.value)


def test_construction_rejects_same_entity_and_time():
    df = pl.DataFrame({"x": ["A"], "value": [0.1]})
    with pytest.raises(ValueError) as exc:
        PanelFrame(df, entity="x", time="x")
    assert "different columns" in str(exc.value)


def test_construction_rejects_non_orderable_time():
    df = pl.DataFrame({"entity": ["A"], "time": ["not-a-time"], "value": [0.1]})
    with pytest.raises(ValueError) as exc:
        PanelFrame(df, entity="entity", time="time")
    assert "orderable" in str(exc.value)


def test_feature_cols_excludes_keys():
    df = pl.DataFrame({"entity": ["A"], "time": [1], "f1": [0.1], "f2": [0.2]})
    pf = PanelFrame(df, entity="entity", time="time")
    assert set(pf.feature_cols) == {"f1", "f2"}
    assert "entity" not in pf.feature_cols
    assert "time" not in pf.feature_cols


def test_as_panel_infers_keys_from_first_two_columns():
    df = pl.DataFrame({"entity": ["A"], "time": [1], "value": [0.1]})
    pf = as_panel(df)
    assert pf.entity_col == "entity"
    assert pf.time_col == "time"


def test_as_panel_returns_panelframe_unchanged():
    df = pl.DataFrame({"entity": ["A"], "time": [1], "value": [0.1]})
    pf = PanelFrame(df, entity="entity", time="time")
    assert as_panel(pf) is pf


# --------------------------------------------------------------------------- #
# sort_panel: per-entity time-monotonic ordering
# --------------------------------------------------------------------------- #
@settings(max_examples=50, deadline=None)
@given(rows=panel_rows())
def test_sort_panel_is_time_monotonic_per_entity(rows):
    df = _df_from_rows(rows)
    pf = PanelFrame(df, entity="entity", time="time")
    sorted_pf = pf.sort_panel()
    # is_sorted_per_entity inspects current row order; after sort it must hold.
    assert sorted_pf.is_sorted_per_entity()

    # Explicitly verify monotonicity per entity on the materialised frame.
    out = sorted_pf.collect()
    for ent in out["entity"].unique().to_list():
        times = out.filter(pl.col("entity") == ent)["time"].to_list()
        assert times == sorted(times)


@settings(max_examples=50, deadline=None)
@given(rows=panel_rows())
def test_sort_panel_descending_time_within_entity(rows):
    df = _df_from_rows(rows)
    pf = PanelFrame(df, entity="entity", time="time")
    out = pf.sort_panel(descending=True).collect()
    for ent in out["entity"].unique().to_list():
        times = out.filter(pl.col("entity") == ent)["time"].to_list()
        assert times == sorted(times, reverse=True)


def test_is_sorted_per_entity_detects_unsorted():
    # B is out of time order within its entity.
    df = pl.DataFrame(
        {
            "entity": ["A", "A", "B", "B"],
            "time": [1, 2, 5, 3],
            "value": [0.0, 0.0, 0.0, 0.0],
        }
    )
    pf = PanelFrame(df, entity="entity", time="time")
    assert not pf.is_sorted_per_entity()
    assert pf.sort_panel().is_sorted_per_entity()


# --------------------------------------------------------------------------- #
# assert_unique_keys
# --------------------------------------------------------------------------- #
def test_assert_unique_keys_passes_for_unique():
    df = pl.DataFrame(
        {"entity": ["A", "A", "B"], "time": [1, 2, 1], "value": [0.0, 0.0, 0.0]}
    )
    pf = PanelFrame(df, entity="entity", time="time")
    assert pf.assert_unique_keys() is pf


def test_assert_unique_keys_raises_on_duplicates():
    df = pl.DataFrame({"entity": ["A", "A"], "time": [1, 1], "value": [0.0, 1.0]})
    pf = PanelFrame(df, entity="entity", time="time")
    with pytest.raises(ValueError) as exc:
        pf.assert_unique_keys()
    assert "duplicated" in str(exc.value)


@settings(max_examples=50, deadline=None)
@given(rows=panel_rows())
def test_assert_unique_keys_property(rows):
    # `rows` is constructed with unique (entity, time) keys -> must pass.
    df = _df_from_rows(rows)
    pf = PanelFrame(df, entity="entity", time="time")
    assert pf.assert_unique_keys() is pf

    # Duplicating any single row must trigger the raise.
    dup = pl.concat([df, df.head(1)])
    pf_dup = PanelFrame(dup, entity="entity", time="time")
    with pytest.raises(ValueError):
        pf_dup.assert_unique_keys()


# --------------------------------------------------------------------------- #
# with_columns: preserves keys + laziness
# --------------------------------------------------------------------------- #
def test_with_columns_preserves_keys_and_laziness():
    df = pl.DataFrame({"entity": ["A"], "time": [1], "value": [2.0]})
    pf = PanelFrame(df, entity="entity", time="time")
    out = pf.with_columns((pl.col("value") * 2).alias("doubled"))
    assert isinstance(out, PanelFrame)
    assert out.entity_col == "entity"
    assert out.time_col == "time"
    assert "doubled" in out.columns
    # Stays lazy: the underlying handle is a LazyFrame, not materialised.
    assert isinstance(out.lazy(), pl.LazyFrame)


def test_with_columns_rejects_dropping_a_key():
    # Re-wrap a frame missing the 'time' key (validate=False trusts the caller),
    # then any with_columns must trip the cheap key-survival guard.
    df = pl.DataFrame({"entity": ["A"], "time": [1], "value": [2.0]})
    pf_broken = PanelFrame(
        df.lazy().drop("time"), entity="entity", time="time", validate=False
    )
    with pytest.raises(ValueError) as exc:
        pf_broken.with_columns(pl.col("value").alias("v2"))
    assert "must not drop the panel keys" in str(exc.value)


# --------------------------------------------------------------------------- #
# over_entity: grouping never mixes values across entities
# --------------------------------------------------------------------------- #
@st.composite
def dense_panel(draw):
    """Draw a panel where every entity has a contiguous dense time axis.

    This makes per-entity cumulative/shift comparisons well-defined regardless
    of row order.
    """
    ents = draw(
        st.lists(st.sampled_from(ENTITIES), min_size=1, max_size=3, unique=True)
    )
    lengths = {e: draw(st.integers(min_value=1, max_value=8)) for e in ents}
    rows = []
    for e in ents:
        for t in range(lengths[e]):
            val = draw(st.floats(min_value=-10, max_value=10, allow_nan=False))
            rows.append((e, t, val))
    return rows


@settings(max_examples=50, deadline=None)
@given(rows=dense_panel(), seed=st.integers(min_value=0, max_value=10_000))
def test_over_entity_cumsum_invariant_to_row_order(rows, seed):
    """A per-entity cumulative sum computed via .over(entity) must be invariant
    to the input row order (after sorting the panel), and must never mix values
    across entities."""
    df = _df_from_rows(rows)
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)

    def cumsum_per_entity(frame):
        pf = PanelFrame(frame, entity="entity", time="time").sort_panel()
        out = pf.with_columns(
            pl.col("value").cum_sum().over(pf.over_entity()).alias("cs")
        )
        return out.collect().sort(["entity", "time"])

    ref = cumsum_per_entity(df)
    got = cumsum_per_entity(shuffled)
    assert ref.equals(got)

    # Cross-check: the cumulative sum for each entity equals an independent
    # per-entity computation (i.e. no leakage of values across entities).
    for ent in ref["entity"].unique().to_list():
        sub = ref.filter(pl.col("entity") == ent).sort("time")
        expected = sub["value"].cum_sum().to_list()
        assert sub["cs"].to_list() == pytest.approx(expected)


@settings(max_examples=50, deadline=None)
@given(rows=dense_panel(), seed=st.integers(min_value=0, max_value=10_000))
def test_over_entity_shift_does_not_mix_entities(rows, seed):
    """A per-entity lag (shift(1).over(entity)) must produce a null for each
    entity's first observation and never borrow a value from another entity."""
    df = _df_from_rows(rows)
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)

    pf = PanelFrame(shuffled, entity="entity", time="time").sort_panel()
    out = pf.with_columns(
        pl.col("value").shift(1).over(pf.over_entity()).alias("lag")
    ).collect()

    for ent in out["entity"].unique().to_list():
        sub = out.filter(pl.col("entity") == ent).sort("time")
        lags = sub["lag"].to_list()
        vals = sub["value"].to_list()
        # First lag of each entity is null (no borrowing across entity bounds).
        assert lags[0] is None
        # Remaining lags equal the entity's own previous value.
        for i in range(1, len(vals)):
            assert lags[i] == pytest.approx(vals[i - 1])
