"""Conformal prediction intervals for panel forecasts.

Two complementary layers live here:

* :func:`enbpi` / :func:`conformalize` -- frame-level ensemble batch
  prediction intervals, consumed by the forecasters.
* The array-level time-series conformal suite -- split conformal, ACI,
  conformal PID, NexCP and CQR -- which restores validity when
  exchangeability fails.  See :mod:`panelary.conformal._intervals`.

Every estimator here consumes calibration scores that must come from a
**purged and embargoed** train/calibration split -- use
:func:`conformal_calibration_split`.
"""

from __future__ import annotations

from panelary.conformal._enbpi import conformalize, enbpi
from panelary.conformal._intervals import (
    AdaptiveConformalResult,
    _rolling_scores,
    adaptive_conformal_intervals,
    conformal_calibration_split,
    conformal_pid_intervals,
    conformal_quantile,
    conformalized_quantile_regression,
    cqr_scores,
    nexcp_intervals,
    nexcp_quantile,
    split_conformal_interval,
)

__all__ = [
    "AdaptiveConformalResult",
    "adaptive_conformal_intervals",
    "conformal_calibration_split",
    "conformal_pid_intervals",
    "conformal_quantile",
    "conformalize",
    "conformalized_quantile_regression",
    "cqr_scores",
    "enbpi",
    "nexcp_intervals",
    "nexcp_quantile",
    "split_conformal_interval",
]
