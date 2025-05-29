from __future__ import annotations

import numpy as np
import polars as pl



def test__cusum_filter():
    vals = list(np.random.default_rng(seed=0).normal(0.0, 0.1, 150))
    vals_2 = list(np.random.default_rng(seed=0).normal(0.2, 0.2, 50))
    data = vals + vals_2

    drift = 0.05
    threshold = 1.0
    warmup_period = 50

    df = pl.DataFrame({"data": data}).with_row_index(name="idx")

    df = df.with_columns((pl.col("data") - drift).alias("drifted"))

    df = df.with_columns(pl.col("drifted").cum_sum().alias("cusum_raw"))

    df = df.with_columns([
        (pl.col("cusum_raw") > threshold).cast(pl.Int8).alias("cusum_event"),
        (pl.col("idx") >= warmup_period).alias("after_warmup")
    ])

    changepoint = (
        df.filter(pl.col("cusum_event") == 1)
          .filter(pl.col("after_warmup"))
          .select("idx")
          .get_column("idx")
          .min()
    )

    print("Detected changepoint at index:", changepoint)
    assert (changepoint is not None) and (changepoint >= 150)
