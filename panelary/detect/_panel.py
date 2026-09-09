"""Cross-sectional (panel) layer for the explosive-behaviour detectors.

This module turns a matrix of *per-entity* BSADF sequences -- the ``(N, T)``
output of :func:`panelary.detect._bsadf.bsadf_panel` -- into panel
quantities that are usable as features, plus the two pieces of machinery that
make such a panel statistically meaningful at all: a factor **residualiser**
and a cross-sectionally-dependent **sieve bootstrap** for the panel null.

Why not just average
--------------------
The published panel statistic (Pavlidis et al. 2016) is the cross-sectional
*mean* of the per-entity BSADF sequences. Under the alternative each bubbling
entity's BSADF diverges roughly exponentially in the time since inception, so
the mean is a sum of a few explosive terms and ``N - k`` bounded ones: it is
dominated by the extremes and its level carries no information about *how many*
entities are involved.

Measured on a 500-name synthetic panel (``T = 400``, ``r0 = 0.10``, AR(1)
bubble with ``delta = 1.05`` over ``t in [250, 330)``; entity threshold = the
family-wise 95% sup-BSADF critical value, 2.32):

========  ===========================  ==========================
fraction  mean BSADF before -> during  breadth before -> during
========  ===========================  ==========================
0.00      -0.67 -> -0.63               0.000 -> 0.002
0.05      -0.67 -> +1.32               0.000 -> 0.050
0.20      -0.67 -> +5.97               0.000 -> 0.198
========  ===========================  ==========================

:func:`breadth` recovers the true participation fraction almost exactly. The
decisive experiment is to hold the fraction fixed and change only the bubble's
*intensity*: at ``delta = 1.08`` the mean explodes to ``+15.06`` (5% bubbling)
and ``+51.99`` (20% bubbling) while breadth barely moves, to ``0.052`` and
``0.202``. The mean measures how violent the worst few entities are, on no
fixed scale; breadth measures how many entities are involved, on a bounded and
directly interpretable one. The mean is therefore exposed only as
:func:`panel_mean`, a **winsorised intensity** measure, documented as
inferior.

:func:`breadth` is the binary-indicator (``EXU_{i,t}``) construct of
Martinez-Garcia & Grossman (2020): each entity contributes 0/1 according to
whether its own date-``t`` critical value is exceeded, and the panel reading is
the cross-sectional share of exceedances.

Why residualise
---------------
Cross-sectional dependence, not ``N``, is the binding constraint. On a pure-null
panel with equicorrelated shocks ``eps_it = sqrt(rho) f_t + sqrt(1 - rho) u_it``
(``N = 200``, ``T = 300``, ``r0 = 0.10``, 60 replications, entity threshold
calibrated at ``rho = 0``), the 95th percentile of the sup-over-``t`` breadth is

=======  ==========================  =========================
``rho``  95% sup-t breadth under H0  after ``residualise(k=1)``
=======  ==========================  =========================
0.0      0.010                       0.010
0.3      0.035                       0.010
0.6      0.096                       0.010
0.9      0.251                       0.010
=======  ==========================  =========================

A threshold calibrated for independent entities over-rejects by 25x at
``rho = 0.9``: a quarter of bubble-free panels flag. Averaging over ``N`` buys
nothing once a common factor is present -- the effective cross-section size is
roughly ``1 / rho``. :func:`residualise` strips ``n_factors`` principal
components out of *returns* with strictly backward-looking, piecewise-constant
betas, and restored the independent-panel null exactly at every ``rho`` tested.

Why a row-resampling bootstrap
------------------------------
The limiting distribution of a mean-type panel unit-root statistic is **not**
invariant to cross-sectional error dependence (Chang 2004), so simulating each
entity independently is invalid however many replications you run.
:func:`sieve_bootstrap_cv` therefore resamples **whole rows** (whole dates) of
the restricted-ADF residual matrix, which is the entire mechanism by which the
contemporaneous cross-sectional covariance survives into the bootstrap world.
Measured on a panel with ``rho = 0.7``: the mean off-diagonal residual
correlation is ``0.658`` in the data and ``0.660`` under the row bootstrap,
against ``0.000`` when each entity's residuals are resampled independently. The
resulting thresholds cut the empirical size of the sup-breadth test at
``rho = 0.9`` from ``0.230`` (independent calibration) to ``0.090``.

What this module deliberately does **not** emit
-----------------------------------------------
No full-sample GSADF, and no episode start / end / duration membership. Those
are *ex post* labels: you only know an episode existed once it terminated, so
attaching them to a row at date ``t`` back-dates future information into the
feature matrix. Everything here is a quantity that is knowable at ``t``.

References
----------
Pavlidis, E., Yusupova, A., Paya, I., Peel, D., Martinez-Garcia, E., Mack, A. &
Grossman, V. (2016). Episodes of exuberance in housing markets: in search of
the smoking gun. *Journal of Real Estate Finance and Economics*, 53(4),
419-449. DOI 10.1007/s11146-015-9531-2.

Martinez-Garcia, E. & Grossman, V. (2020). Explosive dynamics in house prices?
An exploration of financial market spillovers in housing markets around the
world. *Journal of International Money and Finance*, 101, 102103.

Chang, Y. (2004). Bootstrap unit root tests in panels with cross-sectional
dependency. *Journal of Econometrics*, 120(2), 263-293.

Phillips, P. C. B., Shi, S. & Yu, J. (2015). Testing for multiple bubbles:
historical episodes of exuberance and collapse in the S&P 500.
*International Economic Review*, 56(4), 1043-1078.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary.econ._common import pinv_sym, winsorize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = [
    "breadth",
    "panel_mean",
    "cross_sectional_rank",
    "residualise",
    "sieve_bootstrap_cv",
    "panel_features",
]


# --------------------------------------------------------------------------- #
# Known-bug guards -- documented constants, referenced by the tests.
# --------------------------------------------------------------------------- #
#: Panel sieve-bootstrap critical values published by the reference R
#: implementation (90 / 95 / 99%). **Do not calibrate against these.** That
#: implementation's panel loop *assigns* rather than accumulates the per-entity
#: statistic inside the entity loop and then divides by ``N``, so what it returns
#: is the **last entity's** statistic scaled by ``1 / N`` -- not a panel average.
#: The bug shrinks the reported critical values by roughly a factor of ``N``.
R_PANEL_CV_BUGGY: tuple[float, float, float] = (0.105, 0.122, 0.155)

#: Independently computed panel critical values for a comparable design, i.e.
#: what the R numbers above *should* have been (90 / 95 / 99%). An order of
#: magnitude larger.
R_PANEL_CV_INDEPENDENT: tuple[float, float, float] = (1.04, 1.24, 1.79)

#: Sanity anchor for reviewers. Under a **cross-sectionally independent** panel
#: null, correct critical values for the *mean* panel BSADF statistic sit near
#: zero or below, because averaging ``N`` entity sequences concentrates on the
#: mean of the per-entity BSADF (measured: ``-0.741``) rather than on a supremum
#: (the univariate GSADF 95% at ``T = 250``, ``r0 = 0.10`` is ``2.225``).
#: Measured panel-mean bootstrap cv at ``rho = 0`` (``N = 60``, ``T = 250``,
#: ``lag = 1``, ``nboot = 150``): sup-over-``t`` 90/95/99% =
#: ``-0.267 / -0.237 / -0.168``; per-endpoint at the last date =
#: ``-0.353 / -0.323 / -0.268``. Every one of them is negative.
#:
#: Anyone sanity-checking a correct panel cv against the familiar univariate
#: ~2.2 will conclude the code is broken. It is not. ``tests/test_detect_panel``
#: asserts the independent-null panel-mean cv is at or below this bound.
#:
#: Under cross-sectional dependence the mean statistic inherits the common
#: factor and this bound no longer applies: the same design gives ``+0.95`` at
#: ``rho = 0.5`` and ``+2.06`` at ``rho = 0.9``. That is a further argument for
#: :func:`residualise`, not a licence to compare against 2.2.
PANEL_MEAN_CV_UPPER_BOUND: float = 0.5


def _as_2d(stat: NDArray[Any], name: str) -> NDArray[Any]:
    """Validate and float64-upcast an ``(N, T)`` statistic matrix."""
    arr = np.asarray(stat, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"`{name}` must be 1-D or 2-D (N, T), got shape {arr.shape}.")
    return arr


def _broadcast_cv(cv: Any, shape: tuple[int, int]) -> NDArray[Any]:
    """Broadcast a scalar / ``(T,)`` / ``(N, T)`` critical value to ``(N, T)``."""
    arr = np.asarray(cv, dtype=np.float64)
    if arr.ndim == 0:
        return np.broadcast_to(arr, shape)
    if arr.ndim == 1:
        if arr.shape[0] != shape[1]:
            raise ValueError(
                f"`cv` has length {arr.shape[0]} but the statistic has "
                f"{shape[1]} dates; a 1-D `cv` must be indexed by date."
            )
        return np.broadcast_to(arr[None, :], shape)
    if arr.shape != shape:
        raise ValueError(f"`cv` shape {arr.shape} does not match stat shape {shape}.")
    return arr


# --------------------------------------------------------------------------- #
# 1. Aggregators
# --------------------------------------------------------------------------- #
def breadth(
    stat: NDArray[Any],
    cv: NDArray[Any] | float,
    *,
    min_count: int = 1,
) -> NDArray[Any]:
    """Cross-sectional **share of entities exceeding their date-``t`` critical value**.

    This is the ``EXU_{i,t}`` binary-indicator panel construct of
    Martinez-Garcia & Grossman (2020): entity ``i`` contributes
    ``1{stat[i, t] > cv[t]}`` and the panel reading at ``t`` is the mean of those
    indicators over the entities that *have* a statistic at ``t``.

    Prefer this to :func:`panel_mean`. Because each indicator is bounded in
    ``[0, 1]``, no single explosive entity can move the aggregate by more than
    ``1 / N``, and the resulting number is on a fixed, interpretable scale --
    it estimates the fraction of the cross-section currently exhibiting
    explosive behaviour. The cross-sectional mean of the raw BSADF sequences has
    neither property: under the alternative the per-entity statistic diverges
    roughly exponentially in time-since-inception, so the mean is dominated by a
    handful of extremes and its level is uninformative about participation.

    Parameters
    ----------
    stat : ndarray, shape (N, T) or (T,)
        Per-entity statistic sequences, e.g. the output of
        :func:`panelary.detect._bsadf.bsadf_panel`. Non-finite entries
        (entities without enough history at ``t``) are treated as missing.
    cv : ndarray or float
        Date-``t`` critical values. A scalar (one constant threshold), a ``(T,)``
        vector -- the usual case, e.g. ``mc_table(...)[j]`` -- or a full
        ``(N, T)`` matrix when entities carry entity-specific thresholds. A
        ``(T,)`` ``cv`` must be indexed by *date position*, never by the caller's
        own sample size.
    min_count : int, default 1
        Minimum number of entities with a finite statistic required for a date to
        report a value. Dates with fewer return ``nan``.

    Returns
    -------
    ndarray, shape (T,)
        Fraction in ``[0, 1]``; ``nan`` at dates with fewer than ``min_count``
        observed entities.

    Notes
    -----
    Missing entities are excluded from the denominator, not imputed as
    non-exceeding. Imputing zeros would make breadth mechanically drift upward
    as entities accumulate enough history to produce a statistic.

    The comparison is strict (``>``), matching the convention that a statistic
    exactly at its critical value does not reject.

    **Choosing ``cv`` sets the scale of the reading.** A per-endpoint
    ``mc_table(...)`` row at the 95% level gives each entity a 5% per-date size,
    so breadth sits around ``0.05`` under the null and the signal is the *excess*
    over that floor. A family-wise threshold instead -- the 95% quantile of the
    per-entity sup-over-``t`` BSADF, i.e.
    :func:`panelary.detect._critvals.training_max_cv` or a sup-quantile of
    the Monte-Carlo paths -- pins the null floor near zero (measured ``0.000``
    over 70 pre-bubble dates on a 500-name panel) and makes the level of breadth
    read directly as a participation fraction. The tables in this module's
    docstring use the family-wise threshold for that reason.

    Examples
    --------
    >>> import numpy as np
    >>> stat = np.array([[0.0, 3.0], [0.0, 0.0], [np.nan, 4.0]])
    >>> breadth(stat, np.array([1.0, 1.0]))
    array([0.        , 0.66666667])
    """
    s = _as_2d(stat, "stat")
    thr = _broadcast_cv(cv, s.shape)
    observed = np.isfinite(s) & np.isfinite(thr)
    n_obs = observed.sum(axis=0)
    exceed = np.where(observed, s > thr, False).sum(axis=0)
    out = np.full(s.shape[1], np.nan, dtype=np.float64)
    ok = n_obs >= max(1, int(min_count))
    out[ok] = exceed[ok] / n_obs[ok]
    return out


def panel_mean(
    stat: NDArray[Any],
    *,
    limit: float = 0.05,
    min_count: int = 1,
) -> NDArray[Any]:
    """Winsorised cross-sectional **mean** statistic -- an intensity proxy.

    .. warning::
       This is the *published* panel statistic (Pavlidis et al. 2016) and it is
       **inferior to** :func:`breadth` for detection. Use it only as a secondary
       intensity reading, never as the primary flag.

       Under the alternative each bubbling entity's BSADF diverges roughly
       exponentially in the time since inception, so the cross-sectional mean is
       driven by a few extreme entities. On a 500-name panel the mean moved from
       ``-0.67`` to ``+1.32`` when 5% of names bubbled and from ``-0.67`` to
       ``+5.97`` when 20% did. Worse, holding the fraction fixed and raising the
       bubble's autoregressive root from ``1.05`` to ``1.08`` sent the same two
       readings to ``+15.06`` and ``+51.99``: the statistic responds to the
       *intensity* of a handful of entities far more than to how many are
       involved. :func:`breadth` over all four of those experiments read
       ``0.050 / 0.198 / 0.052 / 0.202`` against true fractions of
       ``0.05 / 0.20 / 0.05 / 0.20``.

       The mean's null critical values are also counter-intuitively near zero or
       negative (see :data:`PANEL_MEAN_CV_UPPER_BOUND`), because averaging
       concentrates around the mean per-entity BSADF (``-0.741``) rather than a
       supremum (``2.225``).

    Winsorising at ``limit`` per date caps the influence of the extremes and
    makes the series at least comparable across dates; it does not fix the
    scale problem.

    Parameters
    ----------
    stat : ndarray, shape (N, T) or (T,)
        Per-entity statistic sequences. Non-finite entries are excluded.
    limit : float, default 0.05
        Symmetric winsorisation fraction applied **within each date's
        cross-section**, so it never uses information from other dates. Pass
        ``0.0`` for the raw published mean.
    min_count : int, default 1
        Minimum number of finite entities for a date to report a value.

    Returns
    -------
    ndarray, shape (T,)
        Winsorised cross-sectional mean; ``nan`` where too few entities.
    """
    s = _as_2d(stat, "stat")
    if not 0.0 <= limit < 0.5:
        raise ValueError(f"`limit` must be in [0, 0.5), got {limit}.")
    n_time = s.shape[1]
    out = np.full(n_time, np.nan, dtype=np.float64)
    floor = max(1, int(min_count))
    for t in range(n_time):
        col = s[:, t]
        col = col[np.isfinite(col)]
        if col.size < floor:
            continue
        out[t] = float(np.mean(winsorize(col, limit) if limit > 0.0 else col))
    return out


# --------------------------------------------------------------------------- #
# 2. Cross-sectional rank
# --------------------------------------------------------------------------- #
def cross_sectional_rank(stat: NDArray[Any]) -> NDArray[Any]:
    """Per-date rank of each entity within the cross-section, scaled to ``[0, 1]``.

    Ranking within a date is free (no calibration, no critical values, no
    simulation) and it neutralises the common factor almost entirely: a shock
    ``f_t`` that shifts every entity's statistic by the same amount leaves the
    within-date ordering untouched, so the rank is invariant to any strictly
    monotone common component. That is exactly the nuisance that inflates
    :func:`breadth` under cross-sectional dependence, which makes the rank a
    useful complement to -- not a substitute for -- a residualised level.

    Ties receive their average rank, so the transform is deterministic and
    independent of entity ordering.

    Parameters
    ----------
    stat : ndarray, shape (N, T) or (T,)
        Per-entity statistic sequences. Non-finite entries mark entities without
        enough history at that date.

    Returns
    -------
    ndarray, shape (N, T)
        Rank in ``[0, 1]``: ``0`` for the (uniquely) lowest observed entity at
        that date and ``1`` for the (uniquely) highest; tied extremes share the
        average of their positions and so fall inside the open interval.
        Entities that were non-finite on input stay
        ``nan`` on output. A date with a single observed entity yields ``0.5``
        (the midpoint), and a date with none yields all ``nan``.

    Notes
    -----
    Missing entities are **excluded from the rank denominator**, not imputed.
    Imputing (say, with the cross-sectional median) would hand a synthetic rank
    to entities that have no statistic yet, and would shift every real entity's
    rank as short-history entities enter the panel.

    The rank is computed strictly within a date, so it uses no information from
    any other date and is trivially prefix-invariant.

    Examples
    --------
    >>> import numpy as np
    >>> cross_sectional_rank(np.array([[1.0], [2.0], [3.0], [np.nan]]))
    array([[0. ],
           [0.5],
           [1. ],
           [nan]])
    """
    s = _as_2d(stat, "stat")
    n_ent, n_time = s.shape
    # Work date-major: rows are dates, columns are entities.
    x = s.T
    missing = ~np.isfinite(x)
    n_obs = (~missing).sum(axis=1).astype(np.float64)

    # Sort missing entries to the end so they never affect the observed ranks.
    filled = np.where(missing, np.inf, x)
    order = np.argsort(filled, axis=1, kind="stable")
    srt = np.take_along_axis(filled, order, axis=1)

    pos = np.broadcast_to(np.arange(n_ent, dtype=np.float64), (n_time, n_ent))
    is_start = np.ones((n_time, n_ent), dtype=bool)
    if n_ent > 1:
        is_start[:, 1:] = srt[:, 1:] != srt[:, :-1]
    is_end = np.ones((n_time, n_ent), dtype=bool)
    if n_ent > 1:
        is_end[:, :-1] = is_start[:, 1:]
    # Within a tie run the positions are contiguous, so the average rank is
    # simply the midpoint of the run's first and last position.
    run_start = np.maximum.accumulate(np.where(is_start, pos, -np.inf), axis=1)
    run_end = np.flip(
        np.minimum.accumulate(np.flip(np.where(is_end, pos, np.inf), axis=1), axis=1),
        axis=1,
    )
    avg_sorted = 0.5 * (run_start + run_end)

    ranks = np.empty_like(avg_sorted)
    np.put_along_axis(ranks, order, avg_sorted, axis=1)

    denom = np.where(n_obs > 1.0, n_obs - 1.0, 1.0)[:, None]
    out = ranks / denom
    out = np.where(n_obs[:, None] > 1.0, out, 0.5)
    out[missing] = np.nan
    return np.ascontiguousarray(out.T)


# --------------------------------------------------------------------------- #
# 3. Backward-looking factor residualiser
# --------------------------------------------------------------------------- #
def _sign_of_max_abs(vec: NDArray[Any]) -> float:
    """Sign (+1 / -1) of the largest-magnitude entry; exact zero maps to +1.

    Mirrors :func:`panelary.reduce._base._sign_of_max_abs` so that
    component orientation is fixed by the same deterministic convention used by
    every other reducer in Panelary: components stay stable across refits and
    across entity reordering.
    """
    if vec.size == 0:  # pragma: no cover - defensive
        return 1.0
    idx = int(np.argmax(np.abs(vec)))
    s = float(np.sign(vec[idx]))
    return s if s != 0.0 else 1.0


def _fit_loadings(
    block: NDArray[Any], n_factors: int, min_obs: int
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Fit train-window mean, scale and sign-fixed loadings on one ``(N, L)`` block.

    Returns ``(mu, sigma, loadings)`` with ``loadings`` of shape ``(N, k)`` and
    zero rows for entities with fewer than ``min_obs`` finite observations.
    """
    n_ent, n_len = block.shape
    finite = np.isfinite(block)
    counts = finite.sum(axis=1)
    safe = np.where(counts > 0, counts, 1)
    mu = np.where(finite, block, 0.0).sum(axis=1) / safe
    dev = np.where(finite, block - mu[:, None], 0.0)
    var = (dev * dev).sum(axis=1) / safe
    sigma = np.sqrt(var)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1.0)
    mu = np.where(counts > 0, mu, 0.0)

    usable = counts >= min_obs
    loadings = np.zeros((n_ent, max(int(n_factors), 0)), dtype=np.float64)
    k = int(n_factors)
    if k <= 0 or not usable.any():
        return mu, sigma, loadings

    # Standardised, mean-imputed training matrix (entity x time).
    z = (dev / sigma[:, None]) * usable[:, None]
    k = min(k, int(usable.sum()), max(n_len - 1, 0))
    if k <= 0:
        return mu, sigma, loadings[:, :0]

    # We only need the top-k right singular vectors of the (time x entity)
    # matrix, i.e. the leading eigenvectors of the (entity x entity) Gram. When
    # there are fewer entities than periods -- the usual case once `min_periods`
    # has elapsed -- `eigh` on the small Gram is far cheaper than a full SVD.
    if n_ent <= n_len:
        gram = z @ z.T
        gram = 0.5 * (gram + gram.T)
        _w, vec = np.linalg.eigh(gram)
        comps = np.ascontiguousarray(vec[:, ::-1][:, :k].T)  # (k, N)
    else:
        _u, _s, vt = np.linalg.svd(z.T, full_matrices=False)
        comps = vt[:k]  # (k, N), orthonormal rows in entity space
    sign = np.array([_sign_of_max_abs(comps[j]) for j in range(k)], dtype=np.float64)
    comps = comps * sign[:, None]
    out = np.zeros((n_ent, k), dtype=np.float64)
    out[usable] = comps[:, usable].T
    return mu, sigma, out


def residualise(
    returns: NDArray[Any],
    *,
    n_factors: int,
    min_periods: int,
    refit_every: int = 21,
    window: int | None = None,
    min_obs: int | None = None,
) -> NDArray[Any]:
    """Strip ``n_factors`` common components out of a ``(N, T)`` return panel.

    Cross-sectional dependence -- not the number of entities -- is what breaks a
    panel explosiveness test. With equicorrelated shocks
    ``eps_it = sqrt(rho) f_t + sqrt(1 - rho) u_it`` on a pure-null panel
    (``N = 200``, ``T = 300``, 60 replications), the 95th percentile of the
    sup-over-``t`` :func:`breadth` was measured at ``0.010`` for ``rho = 0``,
    ``0.035`` for ``rho = 0.3``, ``0.096`` for ``rho = 0.6`` and ``0.251`` for
    ``rho = 0.9`` -- a 25-fold over-rejection against a threshold calibrated for
    independent entities, because the effective cross-section size collapses to
    roughly ``1 / rho``. Passing the same returns through
    ``residualise(n_factors=1)`` first drove the figure back to ``0.010`` at
    *every* ``rho``. Removing the common components is what makes the panel null
    tractable.

    Betas are estimated by principal components (numpy SVD only -- no sklearn)
    on a strictly backward-looking training block, **frozen**, and reused until
    the next scheduled refit. The factor realisation at each date is recovered
    from the *contemporaneous* cross-section by projecting that date's returns
    onto the frozen loadings, so nothing from after ``t`` enters the residual
    at ``t``.

    Parameters
    ----------
    returns : ndarray, shape (N, T)
        Panel of returns (not levels). Non-finite entries are treated as
        missing: they are mean-imputed inside the training block and returned as
        ``nan`` in the output.
    n_factors : int
        Number of principal components to remove. ``0`` removes only the
        per-entity training-window mean. The effective count is capped at the
        number of usable entities and at ``block_length - 1``.
    min_periods : int
        Length of history required before any residual is produced, and the
        index of the first refit. Output columns ``0 .. min_periods - 1`` are
        ``nan``.
    refit_every : int, default 21
        Refit cadence in periods. Betas are re-estimated at
        ``t = min_periods, min_periods + refit_every, ...`` and held
        **piecewise-constant** in between.
    window : int, optional
        Training-block length. ``None`` (default) uses an expanding block
        anchored at index 0; an integer uses a rolling block of that many
        trailing periods.
    min_obs : int, optional
        Minimum finite observations an entity needs inside the training block to
        take part in the factor extraction and to receive a non-zero beta.
        Defaults to ``max(2, min_periods // 4)``. Entities below the threshold
        get a zero beta, i.e. their residual is their own demeaned return.

    Returns
    -------
    ndarray, shape (N, T)
        Idiosyncratic residuals in the units of ``returns``. ``nan`` before
        ``min_periods`` and wherever the input was non-finite.

    Notes
    -----
    **Why the refit schedule matters (replay stability).** If the betas were
    re-estimated every bar, the loading used at date ``t`` would change as soon
    as one more observation arrived, so the residual series you would reconstruct
    tomorrow would differ from the one you stored today -- at *every* historical
    date, not just the last. Any model trained on the stored residuals would then
    be evaluated on a series it never saw. Freezing the betas between refits at a
    cadence anchored to index ``0`` makes the construction replay-stable: the
    residual at ``t`` depends only on ``t``, ``min_periods`` and ``refit_every``,
    never on how much data has been appended, so
    ``residualise(r[:, :T])[:, t] == residualise(r[:, :T + k])[:, t]`` exactly.
    Re-estimating every bar reintroduces look-ahead one level up, where the leak
    detector is not looking.

    **Sign convention.** Components are oriented by
    :func:`panelary.reduce._base._sign_of_max_abs`: the largest-magnitude
    loading of each component is forced positive. This is the same convention
    every reducer in :mod:`panelary.reduce` uses, so component identity
    (and hence any diagnostic built on the loadings) is stable across refits.

    **Entity isolation.** This routine mixes entities *within* a date by
    construction -- that is the point. It is time-leak-safe but not
    entity-isolated, so it is a panel-level preprocessing step rather than a
    per-entity ``over(entity)`` expression.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> f = rng.standard_normal(400)
    >>> r = 0.9 * f[None, :] + 0.1 * rng.standard_normal((30, 400))
    >>> e = residualise(r, n_factors=1, min_periods=100)
    >>> bool(np.nanstd(e[:, 100:]) < 0.2 * np.nanstd(r[:, 100:]))
    True
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim == 1:
        r = r[None, :]
    if r.ndim != 2:
        raise ValueError(f"`returns` must be 2-D (N, T), got shape {r.shape}.")
    n_ent, n_time = r.shape
    if int(n_factors) < 0:
        raise ValueError(f"`n_factors` must be >= 0, got {n_factors}.")
    if int(min_periods) < 2:
        raise ValueError(f"`min_periods` must be >= 2, got {min_periods}.")
    if int(refit_every) < 1:
        raise ValueError(f"`refit_every` must be >= 1, got {refit_every}.")
    if window is not None and int(window) < int(min_periods):
        raise ValueError(
            f"`window` ({window}) must be >= `min_periods` ({min_periods})."
        )
    min_periods = int(min_periods)
    refit_every = int(refit_every)
    floor_obs = int(min_obs) if min_obs is not None else max(2, min_periods // 4)

    out = np.full((n_ent, n_time), np.nan, dtype=np.float64)
    if n_time <= min_periods:
        return out

    # Refit dates are anchored at index 0 and depend only on min_periods /
    # refit_every -- never on len(returns). This is the prefix-invariance hook.
    for start in range(min_periods, n_time, refit_every):
        lo = 0 if window is None else max(0, start - int(window))
        mu, sigma, loadings = _fit_loadings(r[:, lo:start], int(n_factors), floor_obs)
        stop = min(start + refit_every, n_time)
        seg = r[:, start:stop]
        z = (seg - mu[:, None]) / sigma[:, None]
        finite = np.isfinite(z)
        k = loadings.shape[1]
        if k == 0:
            out[:, start:stop] = np.where(finite, z * sigma[:, None], np.nan)
            continue
        # The factor realisation at each date is recovered from that date's own
        # cross-section, one matrix-vector product per date. Keeping it a matvec
        # (rather than one GEMM over the whole segment) is deliberate: a GEMM's
        # floating-point accumulation order can depend on the block width, and a
        # truncated final segment would then differ in the last bit from the same
        # date computed in a longer replay. The matvec is bit-identical.
        resid = np.full_like(z, np.nan)
        gram_cache: dict[bytes, NDArray[Any]] = {}
        for j in range(seg.shape[1]):
            mask = finite[:, j]
            if not mask.any():
                continue
            all_obs = bool(mask.all())
            key = np.packbits(mask).tobytes()
            sub_w = loadings if all_obs else loadings[mask]
            obs = z[:, j] if all_obs else z[mask, j]
            ginv = gram_cache.get(key)
            if ginv is None:
                ginv = pinv_sym(sub_w.T @ sub_w)
                gram_cache[key] = ginv
            # Contemporaneous factor realisation given the frozen loadings.
            fac = ginv @ (sub_w.T @ obs)
            fitted = loadings @ fac
            if all_obs:
                resid[:, j] = z[:, j] - fitted
            else:
                resid[mask, j] = z[mask, j] - fitted[mask]
        out[:, start:stop] = resid * sigma[:, None]
    return out


# --------------------------------------------------------------------------- #
# 4. Cross-sectionally dependent sieve bootstrap
# --------------------------------------------------------------------------- #
def _restricted_adf_fit(
    y: NDArray[Any], lag: int
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Fit the **restricted** (H0-imposed) ADF regression entity by entity.

    The restricted model drops the ``y_{t-1}`` level term, which is exactly what
    imposes the unit root::

        dy_t = c + sum_{j=1..lag} phi_j dy_{t-j} + e_t

    Parameters
    ----------
    y : ndarray, shape (N, T)
        Levels.
    lag : int
        Number of lagged first differences.

    Returns
    -------
    const : ndarray, shape (N,)
    phi : ndarray, shape (N, lag)
    resid : ndarray, shape (T - 1 - lag, N)
        Date-major so that a bootstrap draw can resample whole **rows**.
    dy0 : ndarray, shape (N, lag)
        The first ``lag`` observed first differences, used to initialise the
        bootstrap recursion deterministically.
    """
    n_ent, n_time = y.shape
    dy = np.diff(y, axis=1)  # (N, T-1)
    n_eff = dy.shape[1] - lag
    if n_eff < lag + 2:
        raise ValueError(
            f"series too short for the restricted ADF fit: T={n_time}, lag={lag}."
        )
    target = dy[:, lag:]  # (N, n_eff)
    design = np.empty((n_ent, n_eff, lag + 1), dtype=np.float64)
    design[:, :, 0] = 1.0
    for j in range(1, lag + 1):
        design[:, :, j] = dy[:, lag - j : lag - j + n_eff]
    gram = design.transpose(0, 2, 1) @ design  # (N, k, k)
    rhs = design.transpose(0, 2, 1) @ target[:, :, None]  # (N, k, 1)
    # NumPy 2.0 silently mis-solves solve(A, b) when b's leading dim equals the
    # matrix dimension; always pass an explicit trailing axis and squeeze it.
    coef = np.linalg.solve(gram, rhs)[..., 0]  # (N, k)
    fitted = (design @ coef[:, :, None])[..., 0]
    resid = target - fitted  # (N, n_eff)
    return (
        np.ascontiguousarray(coef[:, 0]),
        np.ascontiguousarray(coef[:, 1:]),
        np.ascontiguousarray(resid.T),
        np.ascontiguousarray(dy[:, :lag]),
    )


def _regenerate(
    const: NDArray[Any],
    phi: NDArray[Any],
    innov: NDArray[Any],
    dy0: NDArray[Any],
    y0: NDArray[Any],
) -> NDArray[Any]:
    """Run the fitted AR recursion forward and cumulate to levels."""
    n_ent = const.shape[0]
    lag = phi.shape[1]
    n_eff = innov.shape[0]
    n_diff = lag + n_eff
    dy = np.empty((n_ent, n_diff), dtype=np.float64)
    if lag:
        dy[:, :lag] = dy0
    if lag == 0:
        dy[:, :] = const[:, None] + innov.T
    else:
        for t in range(lag, n_diff):
            acc = const.copy()
            for j in range(1, lag + 1):
                acc += phi[:, j - 1] * dy[:, t - j]
            dy[:, t] = acc + innov[t - lag]
    out = np.empty((n_ent, n_diff + 1), dtype=np.float64)
    out[:, 0] = y0
    np.cumsum(dy, axis=1, out=out[:, 1:])
    out[:, 1:] += y0[:, None]
    return out


def sieve_bootstrap_cv(
    y: NDArray[Any],
    *,
    min_window: int,
    lag: int,
    nboot: int,
    seed: int,
    levels: Sequence[float] = (0.9, 0.95, 0.99),
    statistic: str = "breadth",
    entity_cv: NDArray[Any] | None = None,
    entity_level: float = 0.95,
    entity_nrep: int = 1000,
    winsor_limit: float = 0.05,
    sup: bool = False,
    include_drift: bool = True,
    **bsadf_kwargs: Any,
) -> NDArray[Any]:
    """Critical values for a **panel** statistic under cross-sectional dependence.

    The limiting distribution of a mean-type panel unit-root statistic is *not*
    invariant to cross-sectional error dependence (Chang 2004), so simulating
    entities independently -- however many replications you run -- produces
    critical values for the wrong null. This routine instead bootstraps the
    observed panel with its dependence intact:

    1. impose ``H0`` and fit the **restricted** ADF regression entity by entity
       (``dy_t = c + sum_j phi_j dy_{t-j} + e_t``, with the ``y_{t-1}`` level
       term dropped), keeping the intercept, the lag coefficients and the
       residuals;
    2. stack the residuals into a ``T* x N`` matrix, **dates on the rows**;
    3. resample **whole rows with replacement**. This step, and only this step,
       is what preserves the contemporaneous cross-sectional covariance: a row is
       one date's shock across all entities, so drawing rows keeps
       ``corr(e_i, e_j)`` while destroying serial dependence (which the fitted AR
       recursion then puts back);
    4. regenerate the first differences through the fitted AR recursion and
       cumulate to levels;
    5. compute the panel statistic on the bootstrap panel.

    Repeat ``nboot`` times and take quantiles.

    .. note::
       This is a **research / calibration utility**, not a per-row feature. It
       costs ``nboot`` full BSADF panel evaluations and its output is a table of
       thresholds, computed once, that :func:`breadth` and :func:`panel_features`
       then consume.

    Parameters
    ----------
    y : ndarray, shape (N, T)
        Observed panel **in levels** (e.g. log prices, or cumulated
        :func:`residualise` output). Residualise first if the panel has a strong
        common factor: the bootstrap reproduces whatever dependence is present,
        so a heavily correlated panel simply yields very wide thresholds.
    min_window : int
        Minimum BSADF window, forwarded to
        :func:`panelary.detect._bsadf.bsadf_panel`.
    lag : int
        ADF lag order, used both in the restricted fit and in the BSADF.
    nboot : int
        Number of bootstrap replications.
    seed : int
        Explicit RNG seed. There is no unseeded randomness anywhere in this
        module.
    levels : sequence of float, default (0.9, 0.95, 0.99)
        Quantile levels of the bootstrap panel-statistic distribution.
    statistic : {"breadth", "mean"}, default "breadth"
        Which panel aggregate to calibrate. ``"breadth"`` calibrates
        :func:`breadth`; ``"mean"`` calibrates :func:`panel_mean` and is provided
        for comparability with the published statistic only.
    entity_cv : ndarray, optional
        Length-``T`` per-date critical values for the *individual* BSADF, used
        to form the exceedance indicators when ``statistic="breadth"``. When
        ``None`` these are taken from
        :func:`panelary.detect._critvals.mc_table` at ``entity_level``.
        Splitting the problem this way is deliberate and correct: cross-sectional
        dependence changes the *joint* law of the panel, which is what the row
        bootstrap targets, but it leaves each entity's *marginal* BSADF law
        unchanged, so the marginal thresholds may come from the standard
        independent Monte Carlo.
    entity_level : float, default 0.95
        Level at which the internal ``mc_table`` entity thresholds are taken.
        Note these are *per-endpoint* thresholds, so the null breadth floor sits
        near ``1 - entity_level``; pass a family-wise ``entity_cv`` explicitly if
        you want breadth to read as a participation fraction (see
        :func:`breadth`). Measured on the same panel at ``rho = 0.3``: the 95%
        sup-breadth cv is ``0.403`` with the per-endpoint threshold and ``0.033``
        with a family-wise one.
    entity_nrep : int, default 1000
        Replications for that internal ``mc_table`` call.
    winsor_limit : float, default 0.05
        Winsorisation passed to :func:`panel_mean` when ``statistic="mean"``, so
        the calibrated null matches the statistic you actually compute.
    sup : bool, default False
        If ``True``, reduce each replication to its supremum over
        ``t >= min_window`` before taking quantiles, giving a single
        family-wise-error threshold per level instead of a per-date curve.
    include_drift : bool, default True
        Whether the fitted intercept enters the regeneration recursion. ``True``
        follows the fitted restricted model (a random walk *with* the estimated
        drift). Set ``False`` to impose the strict driftless null that the
        classical PSY tables correspond to.
    **bsadf_kwargs
        Forwarded to ``bsadf_panel`` (e.g. ``grid``), so the bootstrap uses the
        *identical* estimator configuration as the application.

    Notes
    -----
    Measured size, sup-over-``t`` breadth at the 95% level, ``N = 60``,
    ``T = 250``, ``r0 = 0.10``, ``rho = 0.9``, 200 fresh null panels: the
    bootstrap threshold (``0.259``) rejected ``9.0%`` of the time; a threshold
    taken from an independent-entity calibration (``0.05``) rejected ``23.0%``.
    The bootstrap does not fully restore nominal size -- it conditions on one
    realised panel's residual structure -- but it removes most of the distortion.
    Combining it with :func:`residualise` is better than either alone.

    The row resampling itself is exact: on a ``rho = 0.7`` panel the mean
    off-diagonal residual correlation was ``0.658`` in the data and ``0.660``
    across 200 bootstrap draws, versus ``0.000`` when each entity's residual
    column was resampled independently.

    Returns
    -------
    ndarray
        Shape ``(len(levels), T)`` when ``sup=False`` -- per-endpoint critical
        values, ``nan`` before ``min_window``. Shape ``(len(levels),)`` when
        ``sup=True``.

    Warns
    -----
    Two traps, both real, both present in packages you may be tempted to
    cross-check against:

    * **The reference R implementation's panel sieve bootstrap never accumulates
        across entities.** It assigns the per-entity statistic inside the entity
        loop rather than adding to a running total, then divides by ``N``, so it
        returns the *last* entity's statistic over ``N``. Its published panel
        critical values (:data:`R_PANEL_CV_BUGGY`, ``0.105 / 0.122 / 0.155``) are
        an order of magnitude below independently computed ones for a comparable
        design (:data:`R_PANEL_CV_INDEPENDENT`, ``1.04 / 1.24 / 1.79``). **Do not
        calibrate against them.**
    * **Correct panel critical values for the mean statistic are near zero or
        negative.** Averaging ``N`` entity sequences concentrates the aggregate
        around the mean ADF t-statistic (about ``-0.4``) rather than around a
        supremum (about ``2.2``). A reviewer who sanity-checks a correct panel cv
        against the familiar univariate GSADF value of ~2.2 will conclude, wrongly,
        that the code is broken. See :data:`PANEL_MEAN_CV_UPPER_BOUND`.

    References
    ----------
    Chang, Y. (2004). *Journal of Econometrics*, 120(2), 263-293.
    Phillips, Shi & Yu (2015). *International Economic Review*, 56(4).
    Pavlidis et al. (2016). *JREFE*, 53(4). DOI 10.1007/s11146-015-9531-2.
    """
    from panelary.detect._bsadf import bsadf_panel

    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"`y` must be 2-D (N, T), got shape {arr.shape}.")
    if statistic not in {"breadth", "mean"}:
        raise ValueError(f"`statistic` must be 'breadth' or 'mean', got {statistic!r}.")
    if int(nboot) < 1:
        raise ValueError(f"`nboot` must be >= 1, got {nboot}.")
    lvl = np.asarray(levels, dtype=np.float64)
    if lvl.size == 0 or np.any((lvl <= 0.0) | (lvl >= 1.0)):
        raise ValueError(f"`levels` must all lie in (0, 1), got {levels!r}.")
    n_ent, n_time = arr.shape
    lag = int(lag)

    const, phi, resid, dy0 = _restricted_adf_fit(arr, lag)
    if not include_drift:
        const = np.zeros_like(const)
    y0 = arr[:, 0].copy()
    n_rows = resid.shape[0]

    if statistic == "breadth":
        if entity_cv is None:
            from panelary.detect._critvals import mc_table

            entity_cv = mc_table(
                min_window=int(min_window),
                lag=lag,
                t_max=n_time,
                nrep=int(entity_nrep),
                seed=int(seed) + 1,
                levels=(float(entity_level),),
                **{k: v for k, v in bsadf_kwargs.items() if k == "grid"},
            )[0]
        entity_cv = np.asarray(entity_cv, dtype=np.float64).ravel()
        if entity_cv.shape[0] != n_time:
            raise ValueError(
                f"`entity_cv` has length {entity_cv.shape[0]} but `y` has "
                f"{n_time} dates."
            )

    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(nboot), n_time), dtype=np.float64)
    for b in range(int(nboot)):
        rows = rng.integers(0, n_rows, size=n_rows)
        innov = resid[rows]  # whole ROWS -> cross-sectional covariance preserved
        y_star = _regenerate(const, phi, innov, dy0, y0)
        stat = bsadf_panel(y_star, min_window=int(min_window), lag=lag, **bsadf_kwargs)
        if statistic == "breadth":
            draws[b] = breadth(stat, entity_cv)
        else:
            draws[b] = panel_mean(stat, limit=float(winsor_limit))

    if sup:
        tail = draws[:, int(min_window) :]
        per_rep = np.where(
            np.isfinite(tail).any(axis=1), np.nanmax(tail, axis=1), np.nan
        )
        per_rep = per_rep[np.isfinite(per_rep)]
        if per_rep.size == 0:  # pragma: no cover - degenerate input
            return np.full(lvl.shape[0], np.nan)
        return np.quantile(per_rep, lvl)

    out = np.full((lvl.shape[0], n_time), np.nan, dtype=np.float64)
    ok = np.isfinite(draws).any(axis=0)
    if ok.any():
        out[:, ok] = np.nanquantile(draws[:, ok], lvl, axis=0)
    return out


# --------------------------------------------------------------------------- #
# 5. Tidy feature frame
# --------------------------------------------------------------------------- #
def panel_features(
    stat: NDArray[Any],
    cv: NDArray[Any] | float,
    *,
    entities: Sequence[Any] | NDArray[Any] | None = None,
    dates: Sequence[Any] | NDArray[Any] | None = None,
    entity_col: str = "entity",
    time_col: str = "date",
    prefix: str = "bsadf",
    winsor_limit: float = 0.05,
    min_count: int = 1,
) -> pl.DataFrame:
    """Assemble the panel detection features into one tidy long Polars frame.

    Every column is knowable at its own row's date. In particular there is
    **no** full-sample GSADF column and **no** episode start / end / duration or
    membership flag: an episode is only identifiable once it has terminated, so
    labelling row ``t`` with "this row was inside episode 3, which ran from
    ``t0`` to ``t1``" back-dates information from ``t1 > t`` into the feature
    matrix. Use ``<prefix>_run`` -- the length of the exceedance run *ending* at
    ``t`` -- as the leak-safe substitute for episode duration.

    Parameters
    ----------
    stat : ndarray, shape (N, T)
        Per-entity BSADF sequences.
    cv : ndarray or float
        Date-``t`` critical values: scalar, ``(T,)`` or ``(N, T)``.
    entities : sequence, optional
        Entity labels, length ``N``. Defaults to ``0 .. N - 1``.
    dates : sequence, optional
        Date labels, length ``T``. Defaults to ``0 .. T - 1``.
    entity_col, time_col : str
        Names for the two key columns.
    prefix : str, default "bsadf"
        Prefix for the emitted feature columns.
    winsor_limit : float, default 0.05
        Winsorisation for the secondary ``panel_mean`` column.
    min_count : int, default 1
        Minimum observed entities for a date to report ``breadth`` /
        ``panel_mean``.

    Returns
    -------
    polars.DataFrame
        ``N * T`` rows sorted by ``(entity, date)`` with columns

        ``<entity_col>``, ``<time_col>``
            Panel keys.
        ``<prefix>``
            The entity statistic at ``t``.
        ``<prefix>_cv``
            Its date-``t`` critical value.
        ``<prefix>_excess``
            ``stat - cv``. Signed distance from the threshold; a continuous,
            better-behaved regressor than the raw statistic.
        ``<prefix>_flag``
            Boolean ``stat > cv`` (null where either is missing).
        ``<prefix>_run``
            Number of consecutive exceedances *ending* at ``t`` (``0`` when not
            flagged). Resets on a missing observation.
        ``<prefix>_rank``
            :func:`cross_sectional_rank` of the statistic within the date.
        ``breadth``
            Per-date :func:`breadth`, broadcast to every row of that date.
        ``panel_mean``
            Per-date winsorised :func:`panel_mean`, broadcast likewise. Kept as a
            secondary intensity reading only -- see that function's warning.

    Examples
    --------
    >>> import numpy as np
    >>> s = np.array([[0.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    >>> f = panel_features(s, 1.0)
    >>> f["bsadf_run"].to_list()
    [0, 1, 2, 0, 0, 0]
    """
    s = _as_2d(stat, "stat")
    n_ent, n_time = s.shape
    thr = np.array(_broadcast_cv(cv, s.shape), dtype=np.float64, copy=True)

    if entities is None:
        ent_labels: NDArray[Any] = np.arange(n_ent)
    else:
        ent_labels = np.asarray(entities)
        if ent_labels.shape[0] != n_ent:
            raise ValueError(
                f"`entities` has length {ent_labels.shape[0]}, expected {n_ent}."
            )
    if dates is None:
        date_labels: NDArray[Any] = np.arange(n_time)
    else:
        date_labels = np.asarray(dates)
        if date_labels.shape[0] != n_time:
            raise ValueError(
                f"`dates` has length {date_labels.shape[0]}, expected {n_time}."
            )

    observed = np.isfinite(s) & np.isfinite(thr)
    flag = observed & (s > thr)
    excess = np.where(observed, s - thr, np.nan)

    # Consecutive-exceedance run ending at t: t minus the last index <= t that
    # was not a (finite) exceedance. Backward-looking by construction.
    idx = np.broadcast_to(np.arange(n_time, dtype=np.int64), (n_ent, n_time))
    last_off = np.maximum.accumulate(np.where(flag, -1, idx), axis=1)
    run = np.where(flag, idx - last_off, 0).astype(np.int32)

    rank = cross_sectional_rank(s)
    br = breadth(s, thr, min_count=min_count)
    pm = panel_mean(s, limit=float(winsor_limit), min_count=min_count)

    flag_col = np.where(observed, flag, None).ravel()
    frame = pl.DataFrame(
        {
            entity_col: pl.Series(np.repeat(ent_labels, n_time)),
            time_col: pl.Series(np.tile(date_labels, n_ent)),
            prefix: s.ravel(),
            f"{prefix}_cv": np.where(observed, thr, np.nan).ravel(),
            f"{prefix}_excess": excess.ravel(),
            f"{prefix}_flag": pl.Series(list(flag_col), dtype=pl.Boolean),
            f"{prefix}_run": run.ravel(),
            f"{prefix}_rank": rank.ravel(),
            "breadth": np.tile(br, n_ent),
            "panel_mean": np.tile(pm, n_ent),
        }
    )
    return frame
