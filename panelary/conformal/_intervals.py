"""Time-series conformal prediction.

Classical split conformal is only valid under *exchangeability*, which a time
series violates: distribution shift, trends and volatility clustering all make
yesterday's calibration scores a biased sample for today. The estimators below
restore validity in three complementary ways:

* :func:`adaptive_conformal_intervals` (ACI, Gibbs & Candes 2021) and
  :func:`conformal_pid_intervals` (Angelopoulos, Candes & Tibshirani 2023)
  *track* the miscoverage online, giving long-run coverage under arbitrary
  drift without any exchangeability assumption.
* :func:`nexcp_quantile` / :func:`nexcp_intervals` (Barber, Candes, Ramdas &
  Tibshirani 2023) down-weight stale calibration points, trading a small,
  explicitly bounded coverage gap for robustness to non-exchangeability.
* :func:`conformalized_quantile_regression` (Romano, Patterson & Candes 2019)
  conformalises a *quantile* model, so interval width adapts to
  heteroskedasticity instead of being a constant band.

Every one of them consumes calibration scores that must come from a
**purged and embargoed** train/calibration split — use
:func:`conformal_calibration_split`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def conformal_calibration_split(
    n_times: int,
    *,
    calibration_size: int | float = 0.25,
    horizon: int = 0,
    embargo: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Purged, embargoed train/calibration split for conformal prediction.

    Conformal coverage guarantees hold only if the calibration scores come from
    a model that never saw the calibration rows. On a panel that means the split
    must be **temporal**, with the boundary purged by the label horizon and
    embargoed for serial correlation — a random split silently destroys the
    guarantee. Delegates to
    :func:`panelary.validation.purged_calibration_split`.

    Parameters
    ----------
    n_times : int
        Number of unique time steps in the panel.
    calibration_size : int | float, default=0.25
        Number of calibration steps, or a fraction of ``n_times``.
    horizon : int, default=0
        Label horizon in time steps.
    embargo : int, default=0
        Embargo in time steps.

    Returns
    -------
    (train, calibration) : tuple of numpy.ndarray
        Sorted, disjoint position arrays with ``max(train) < min(calibration)``.
    """
    from panelary.validation._cv import purged_calibration_split

    return purged_calibration_split(
        n_times,
        calibration_size=calibration_size,
        horizon=horizon,
        embargo=embargo,
    )


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile of calibration scores.

    Returns the ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile, the
    correction that makes split conformal intervals attain at least ``1 - alpha``
    marginal coverage for a finite calibration set (Vovk et al.; Lei et al.
    2018). Returns ``inf`` when ``n`` is too small for the level to be
    achievable, which is the correct (infinitely wide, honestly uninformative)
    answer.

    Parameters
    ----------
    scores : ndarray of shape (n,)
        Non-negative nonconformity scores from the calibration block.
    alpha : float
        Target miscoverage in ``(0, 1)``.

    Returns
    -------
    float
        The conformal quantile.

    Examples
    --------
    >>> import numpy as np
    >>> conformal_quantile(np.arange(1.0, 101.0), 0.1)
    91.0
    """
    s = np.asarray(scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        raise ValueError("`scores` contains no finite values.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"`alpha` must be in (0, 1), got {alpha}.")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.partition(s, k - 1)[k - 1])


def split_conformal_interval(
    calibration_scores: np.ndarray,
    point_predictions: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric split-conformal interval around point predictions.

    The exchangeability-assuming baseline: a *constant* radius
    :func:`conformal_quantile` added either side of the point forecast. Included
    so the adaptive methods can be measured against it — under drift its
    conditional coverage degrades badly while the adaptive methods hold.

    Parameters
    ----------
    calibration_scores : ndarray of shape (n,)
        Absolute residuals (or any nonconformity score) from the purged
        calibration block.
    point_predictions : ndarray of shape (m,)
        Point forecasts to wrap.
    alpha : float, default=0.1
        Target miscoverage.

    Returns
    -------
    (lower, upper) : tuple of numpy.ndarray
    """
    radius = conformal_quantile(calibration_scores, alpha)
    pred = np.asarray(point_predictions, dtype=float)
    return pred - radius, pred + radius


@dataclass(frozen=True)
class AdaptiveConformalResult:
    """Output of an online (adaptive) conformal procedure.

    Attributes
    ----------
    radius : numpy.ndarray of shape (T,)
        The interval half-width used at each step, computed from information
        available strictly *before* that step.
    alpha_t : numpy.ndarray of shape (T,)
        The internal (adapted) miscoverage level at each step. For
        :func:`conformal_pid_intervals` this is ``nan`` — PID tracks the
        radius directly rather than a level.
    covered : numpy.ndarray of shape (T,) of bool
        Whether the realised score fell inside the interval.
    coverage : float
        Empirical coverage over the evaluated steps.
    target_coverage : float
        The nominal ``1 - alpha``.
    n_warmup : int
        Number of leading steps excluded from :attr:`coverage` while the
        calibration set filled up.
    """

    radius: np.ndarray
    alpha_t: np.ndarray
    covered: np.ndarray
    coverage: float
    target_coverage: float
    n_warmup: int


def _rolling_scores(scores: np.ndarray, t: int, window: int | None) -> np.ndarray:
    lo = 0 if window is None else max(0, t - window)
    return scores[lo:t]


def adaptive_conformal_intervals(
    scores: np.ndarray,
    *,
    alpha: float = 0.1,
    gamma: float = 0.01,
    window: int | None = None,
    warmup: int | None = None,
) -> AdaptiveConformalResult:
    """Adaptive Conformal Inference (ACI) of Gibbs & Candes (2021).

    ACI keeps an internal level ``alpha_t`` and updates it after every
    observation by

    .. math:: \\alpha_{t+1} = \\alpha_t + \\gamma\\,(\\alpha - \\mathrm{err}_t)

    where ``err_t = 1{score_t > q_t}``. Intervals widen when coverage is being
    missed and narrow when it is too generous, so the **long-run** coverage
    converges to ``1 - alpha`` *regardless of distribution shift* — no
    exchangeability, no stationarity, not even a stochastic model of the data is
    required. This is the property naive split conformal loses on a drifting
    time series.

    Parameters
    ----------
    scores : ndarray of shape (T,)
        Nonconformity scores in **time order** (e.g. ``abs(y - y_hat)``), one per
        step, produced out-of-sample.
    alpha : float, default=0.1
        Target miscoverage.
    gamma : float, default=0.01
        Adaptation step size. Larger reacts faster to shift at the cost of a
        noisier interval width.
    window : int, optional
        Rolling calibration window (number of past scores used for the
        empirical quantile). ``None`` uses all history.
    warmup : int, optional
        Steps excluded from the reported coverage while history accumulates.
        Defaults to ``min(20, T // 5)``.

    Returns
    -------
    AdaptiveConformalResult

    Raises
    ------
    ValueError
        If ``scores`` is empty or the hyper-parameters are out of range.

    References
    ----------
    Gibbs, I., & Candes, E. (2021). "Adaptive Conformal Inference Under
    Distribution Shift." *NeurIPS 2021*.
    """
    s = np.asarray(scores, dtype=float).ravel()
    n = s.size
    if n == 0:
        raise ValueError("`scores` must not be empty.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"`alpha` must be in (0, 1), got {alpha}.")
    if gamma <= 0:
        raise ValueError(f"`gamma` must be positive, got {gamma}.")
    if warmup is None:
        warmup = min(20, n // 5)

    radius = np.empty(n, dtype=float)
    alpha_t = np.empty(n, dtype=float)
    covered = np.zeros(n, dtype=bool)
    a = float(alpha)
    for t in range(n):
        alpha_t[t] = a
        hist = _rolling_scores(s, t, window)
        if hist.size == 0 or a >= 1.0:
            q = float("inf") if a < 1.0 else 0.0
        elif a <= 0.0:
            q = float("inf")
        else:
            q = conformal_quantile(hist, min(max(a, 1e-9), 1 - 1e-9))
        radius[t] = q
        err = 0.0 if s[t] <= q else 1.0
        covered[t] = s[t] <= q
        a = a + gamma * (alpha - err)
        a = float(min(max(a, -1.0), 2.0))  # keep the recursion bounded
    evaluated = covered[warmup:]
    coverage = float(evaluated.mean()) if evaluated.size else float("nan")
    return AdaptiveConformalResult(
        radius=radius,
        alpha_t=alpha_t,
        covered=covered,
        coverage=coverage,
        target_coverage=1.0 - alpha,
        n_warmup=int(warmup),
    )


def conformal_pid_intervals(
    scores: np.ndarray,
    *,
    alpha: float = 0.1,
    k_p: float = 0.1,
    k_i: float = 0.01,
    k_d: float = 0.0,
    integral_clip: float = 100.0,
    scorecast: np.ndarray | None = None,
    q_init: float | None = None,
    warmup: int | None = None,
) -> AdaptiveConformalResult:
    """Conformal PID control of the radius (Angelopoulos, Candes & Tibshirani).

    Where ACI adapts the *level*, conformal PID adapts the *radius* directly with
    a proportional-integral-derivative controller driven by the coverage error
    ``e_t = err_t - alpha``:

    .. math:: q_{t+1} = k_p e_t + k_i \\sum_{i \\le t} e_i
              + k_d (e_t - e_{t-1}) + \\hat g_{t+1}

    applied as an increment to ``q_t``. The integral term is the quantile
    tracker of Gibbs-Candes (so ``k_p = k_d = 0`` recovers plain quantile
    tracking); the proportional term reacts faster to a sudden break; the
    optional **scorecaster** ``\\hat g`` injects a forecast of the score itself,
    which is what lets the method anticipate seasonal volatility instead of only
    reacting to it.

    Parameters
    ----------
    scores : ndarray of shape (T,)
        Nonconformity scores in time order.
    alpha : float, default=0.1
        Target miscoverage.
    k_p, k_i, k_d : float
        Proportional, integral and derivative gains. Scale with the units of the
        score; the defaults suit scores of order 1.
    integral_clip : float, default=100.0
        Saturation applied to the accumulated error, preventing integral
        wind-up after a long outage.
    scorecast : ndarray of shape (T,), optional
        A leak-free forecast of the score at each step, added to the radius.
        Must be produced from information available before the step.
    q_init : float, optional
        Initial radius. Defaults to the empirical ``1 - alpha`` quantile of the
        first ``min(20, T)`` scores, which are then treated as warm-up.
    warmup : int, optional
        Steps excluded from the reported coverage. Defaults to
        ``min(20, T // 5)``.

    Returns
    -------
    AdaptiveConformalResult
        ``alpha_t`` is ``nan`` (PID tracks the radius, not a level).

    References
    ----------
    Angelopoulos, A. N., Candes, E. J., & Tibshirani, R. J. (2023). "Conformal
    PID Control for Time Series Prediction." *NeurIPS 2023*.
    """
    s = np.asarray(scores, dtype=float).ravel()
    n = s.size
    if n == 0:
        raise ValueError("`scores` must not be empty.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"`alpha` must be in (0, 1), got {alpha}.")
    if integral_clip <= 0:
        raise ValueError(f"`integral_clip` must be positive, got {integral_clip}.")
    if scorecast is not None:
        g = np.asarray(scorecast, dtype=float).ravel()
        if g.size != n:
            raise ValueError(
                f"`scorecast` has length {g.size} but `scores` has length {n}."
            )
    else:
        g = np.zeros(n, dtype=float)
    if warmup is None:
        warmup = min(20, n // 5)
    if q_init is None:
        head = s[: min(20, n)]
        q_init = float(np.quantile(head, 1.0 - alpha))

    radius = np.empty(n, dtype=float)
    covered = np.zeros(n, dtype=bool)
    q = float(q_init)
    integral = 0.0
    prev_err = 0.0
    for t in range(n):
        radius[t] = q + g[t]
        covered[t] = s[t] <= radius[t]
        e = (0.0 if covered[t] else 1.0) - alpha
        integral = float(np.clip(integral + e, -integral_clip, integral_clip))
        q = q + k_p * e + k_i * integral + k_d * (e - prev_err)
        q = max(q, 0.0)
        prev_err = e
    evaluated = covered[warmup:]
    coverage = float(evaluated.mean()) if evaluated.size else float("nan")
    return AdaptiveConformalResult(
        radius=radius,
        alpha_t=np.full(n, np.nan),
        covered=covered,
        coverage=coverage,
        target_coverage=1.0 - alpha,
        n_warmup=int(warmup),
    )


def nexcp_quantile(
    scores: np.ndarray, alpha: float = 0.1, *, rho: float = 0.99
) -> float:
    """Weighted conformal quantile for non-exchangeable data (NexCP).

    Barber, Candes, Ramdas & Tibshirani (2023) show that split conformal remains
    valid up to an explicit *coverage gap* equal to the total-variation distance
    between the data and its exchangeable idealisation, and that geometric
    weights ``w_i = rho^{n-i}`` (recent points weighted most) minimise that gap
    for a slowly drifting series. The quantile is taken over the calibration
    scores with normalised weights, reserving mass ``1 / (sum(w) + 1)`` for the
    unseen test point — the weighted analogue of the ``(n + 1)`` correction.

    Parameters
    ----------
    scores : ndarray of shape (n,)
        Calibration scores in **time order** (oldest first).
    alpha : float, default=0.1
        Target miscoverage.
    rho : float, default=0.99
        Geometric decay in ``(0, 1]``. ``rho = 1`` recovers unweighted split
        conformal; smaller values forget faster.

    Returns
    -------
    float
        The weighted conformal quantile (``inf`` if the level is unachievable
        with the given weights).

    Raises
    ------
    ValueError
        If ``rho`` or ``alpha`` are out of range.
    """
    s = np.asarray(scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        raise ValueError("`scores` contains no finite values.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"`alpha` must be in (0, 1), got {alpha}.")
    if not (0.0 < rho <= 1.0):
        raise ValueError(f"`rho` must be in (0, 1], got {rho}.")
    w = rho ** np.arange(n - 1, -1, -1, dtype=float)
    total = float(w.sum()) + 1.0  # +1 is the point mass at +inf (the test point)
    order = np.argsort(s)
    cumulative = np.cumsum(w[order]) / total
    hit = np.flatnonzero(cumulative >= 1.0 - alpha)
    if hit.size == 0:
        return float("inf")
    return float(s[order[int(hit[0])]])


def nexcp_intervals(
    scores: np.ndarray,
    *,
    alpha: float = 0.1,
    rho: float = 0.99,
    window: int | None = None,
    warmup: int | None = None,
) -> AdaptiveConformalResult:
    """Rolling NexCP radii over a score stream.

    At each step the radius is :func:`nexcp_quantile` of the past scores only,
    so the sequence is causal by construction and can be compared directly with
    :func:`adaptive_conformal_intervals`.

    Parameters
    ----------
    scores : ndarray of shape (T,)
        Nonconformity scores in time order.
    alpha : float, default=0.1
        Target miscoverage.
    rho : float, default=0.99
        Geometric weight decay.
    window : int, optional
        Rolling history length. ``None`` uses all past scores.
    warmup : int, optional
        Steps excluded from the reported coverage. Defaults to
        ``min(20, T // 5)``.

    Returns
    -------
    AdaptiveConformalResult
        ``alpha_t`` is constant at ``alpha`` (NexCP fixes the level and reweights
        the calibration set instead).
    """
    s = np.asarray(scores, dtype=float).ravel()
    n = s.size
    if n == 0:
        raise ValueError("`scores` must not be empty.")
    if warmup is None:
        warmup = min(20, n // 5)
    radius = np.empty(n, dtype=float)
    covered = np.zeros(n, dtype=bool)
    for t in range(n):
        hist = _rolling_scores(s, t, window)
        radius[t] = (
            float("inf") if hist.size == 0 else nexcp_quantile(hist, alpha, rho=rho)
        )
        covered[t] = s[t] <= radius[t]
    evaluated = covered[warmup:]
    return AdaptiveConformalResult(
        radius=radius,
        alpha_t=np.full(n, float(alpha)),
        covered=covered,
        coverage=float(evaluated.mean()) if evaluated.size else float("nan"),
        target_coverage=1.0 - alpha,
        n_warmup=int(warmup),
    )


def cqr_scores(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """CQR nonconformity scores ``max(lower - y, y - upper)``.

    The signed distance outside a predicted quantile interval: negative when the
    realisation is comfortably inside, positive when it falls outside. Feeding
    these to :func:`conformal_quantile` yields the additive correction that
    restores exact coverage.

    Parameters
    ----------
    y_true : ndarray of shape (n,)
        Realised values.
    lower, upper : ndarray of shape (n,)
        Predicted lower and upper conditional quantiles.

    Returns
    -------
    ndarray of shape (n,)
        The CQR scores.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError(
            f"`y_true`, `lower` and `upper` must have the same shape, got "
            f"{y.shape}, {lo.shape}, {hi.shape}."
        )
    return np.maximum(lo - y, y - hi)


def conformalized_quantile_regression(
    y_calibration: np.ndarray,
    lower_calibration: np.ndarray,
    upper_calibration: np.ndarray,
    lower_test: np.ndarray,
    upper_test: np.ndarray,
    *,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Conformalized Quantile Regression (Romano, Patterson & Candes, 2019).

    Takes any quantile model's ``[lower, upper]`` band and inflates (or shrinks)
    it by a single conformal offset so that coverage is guaranteed at
    ``1 - alpha`` while the band keeps the **heteroskedastic shape** the quantile
    model learned — unlike a constant-radius split-conformal interval, a CQR
    interval is narrow in calm periods and wide in turbulent ones.

    Parameters
    ----------
    y_calibration : ndarray of shape (n,)
        Realised values on the purged calibration block.
    lower_calibration, upper_calibration : ndarray of shape (n,)
        The quantile model's predictions on the calibration block.
    lower_test, upper_test : ndarray of shape (m,)
        The quantile model's predictions on the test points.
    alpha : float, default=0.1
        Target miscoverage.

    Returns
    -------
    (lower, upper) : tuple of numpy.ndarray
        The conformalised interval for the test points.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.standard_normal(200)
    >>> band_lo, band_hi = np.full(200, -0.5), np.full(200, 0.5)
    >>> lo, hi = conformalized_quantile_regression(
    ...     y, band_lo, band_hi, band_lo[:3], band_hi[:3], alpha=0.1
    ... )
    >>> bool(np.all(hi - lo > 1.0))  # the too-narrow band is inflated
    True
    """
    offset = conformal_quantile(
        cqr_scores(y_calibration, lower_calibration, upper_calibration), alpha
    )
    lo = np.asarray(lower_test, dtype=float) - offset
    hi = np.asarray(upper_test, dtype=float) + offset
    return lo, hi
