"""Double/debiased machine learning with leak-safe cross-fitting.

Implements the partially linear model

.. math::

    Y = \\theta D + g(X) + U, \\qquad D = m(X) + V,
    \\qquad E[U \\mid X, D] = 0, \\; E[V \\mid X] = 0,

where ``D`` is the feature whose effect you want to measure and ``X`` is the
rest of the (possibly very wide) panel feature matrix. Regressing ``Y`` on
``D`` and ``X`` directly gives an estimate of ``theta`` whose bias does not
vanish at the ``sqrt(n)`` rate once ``X`` is high-dimensional or the nuisance
functions are fitted by a regularised learner -- regularisation bias and
overfitting bias both leak into the coefficient of interest.

Two devices fix it (Chernozhukov et al. 2018):

1. **Neyman-orthogonal score.** Partial ``X`` out of both sides first and
   regress residual on residual:
   ``theta = E[(D - m(X))(Y - l(X))] / E[(D - m(X))^2]``. The score's derivative
   with respect to the nuisance functions is zero at the truth, so first-order
   errors in ``l`` and ``m`` do not propagate into ``theta``.
2. **Cross-fitting.** The nuisance predictions used for observation ``i`` are
   produced by a model that never saw observation ``i``. Here the folds come
   from PanelKit's **purged and embargoed** splitters
   (:class:`~polars_features.core.model_selection.PurgedKFold`), so the folds are
   leak-safe in the temporal sense as well: a label whose horizon overlaps the
   held-out block is purged, and an embargo is applied after it.

Post-double-selection LASSO
---------------------------
:func:`post_double_selection` implements the Belloni-Chernozhukov-Hansen
alternative: LASSO ``Y`` on ``X``, LASSO ``D`` on ``X``, then run OLS of ``Y``
on ``D`` and the **union** of the two selected sets. Selecting twice is what
protects against omitting a control that matters a lot for ``D`` but only a
little for ``Y`` -- the classic way naive single-selection LASSO goes wrong. The
penalty level is the theory-driven "rigorous" choice
``lambda = 2 c sqrt(n) Phi^{-1}(1 - gamma / (2p))`` with iteratively estimated
penalty loadings, not a cross-validated one.

Everything is pure NumPy: the default learner is the rigorous LASSO implemented
here by coordinate descent. Any object with ``fit`` / ``predict`` (e.g. a
scikit-learn regressor) can be passed instead, and is imported by the caller --
this module adds no dependency.

Leak-safety notes
-----------------
* Nuisance models are fitted on training folds only; the fitted
  :class:`DoubleMLTransformer` freezes them and applies them unchanged at
  ``transform`` time.
* For an **overlapping-horizon target** (a ``h``-period forward return, a local
  projection) pass ``horizon=h``; the splitter then purges overlapping labels
  and an ``embargo`` of at least ``h`` is strongly recommended (plan section 7.5).

References
----------
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W.
& Robins, J. (2018). *Double/debiased machine learning for treatment and
structural parameters*, Econometrics Journal 21(1).
Belloni, A., Chernozhukov, V. & Hansen, C. (2014). *Inference on treatment
effects after selection among high-dimensional controls*, ReStud 81(2).
Belloni, A., Chen, D., Chernozhukov, V. & Hansen, C. (2012). *Sparse models and
methods for optimal instruments*, Econometrica 80(6).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from polars_features.core.model_selection import PurgedKFold
from polars_features.core.panel_frame import PanelFrame, as_panel
from polars_features.core.protocol import PanelTransformer
from polars_features.econ._common import factorize, norm_cdf, norm_ppf, ols

__all__ = [
    "rigorous_lasso",
    "post_lasso",
    "lasso_penalty",
    "post_double_selection",
    "dml_partial_linear",
    "DMLResult",
    "PDSResult",
    "DoubleMLTransformer",
]


# --------------------------------------------------------------------------- #
# Rigorous (plug-in penalty) LASSO
# --------------------------------------------------------------------------- #
def lasso_penalty(
    n: int, p: int, *, c: float = 1.1, gamma: float | None = None
) -> float:
    """Belloni-Chernozhukov-Hansen "rigorous" penalty level.

    Returns the ``alpha`` of the objective
    ``(1 / (2n)) ||y - X b||^2 + alpha * sum_j psi_j |b_j|``, which corresponds
    to the BCH penalty ``lambda = 2 c sqrt(n) Phi^{-1}(1 - gamma / (2p))``.

    Parameters
    ----------
    n, p : int
        Sample size and number of candidate controls.
    c : float, default=1.1
        Slack constant; ``> 1`` is required by the theory.
    gamma : float, optional
        Confidence level of the penalty. Defaults to ``0.1 / log(max(n, 3))``.

    Returns
    -------
    float
    """
    if n < 1 or p < 1:
        raise ValueError(f"`n` and `p` must be positive, got n={n}, p={p}.")
    g = gamma if gamma is not None else 0.1 / np.log(max(n, 3))
    q = 1.0 - g / (2.0 * p)
    return float(c * norm_ppf(min(max(q, 0.5 + 1e-12), 1 - 1e-15)) / np.sqrt(n))


def _coordinate_descent(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    loadings: np.ndarray,
    *,
    max_iter: int,
    tol: float,
    beta0: np.ndarray | None = None,
) -> np.ndarray:
    """Cyclic coordinate descent for a weighted-L1 penalised least squares fit."""
    n, p = x.shape
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    resid = y - x @ beta
    norms = (x * x).sum(axis=0) / n
    active = norms > 0
    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            if not active[j]:
                continue
            bj = beta[j]
            rho = float(x[:, j] @ resid) / n + norms[j] * bj
            thr = alpha * loadings[j]
            new = np.sign(rho) * max(abs(rho) - thr, 0.0) / norms[j]
            if new != bj:
                resid -= x[:, j] * (new - bj)
                beta[j] = new
                max_change = max(max_change, abs(new - bj))
        if max_change < tol:
            break
    return beta


@dataclass
class LassoFit:
    """A fitted rigorous LASSO.

    Attributes
    ----------
    coef : ndarray, shape (p,)
        Slopes on the original (uncentred) scale.
    intercept : float
    selected : ndarray of int
        Indices of the non-zero coefficients.
    alpha : float
        Penalty level actually used.
    loadings : ndarray
        Final penalty loadings.
    n_iter : int
        Number of loading-update passes.
    """

    coef: np.ndarray
    intercept: float
    selected: np.ndarray
    alpha: float
    loadings: np.ndarray
    n_iter: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict for a design matrix on the original scale."""
        return np.asarray(x, dtype=float) @ self.coef + self.intercept


def rigorous_lasso(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c: float = 1.1,
    gamma: float | None = None,
    n_loading_iter: int = 3,
    max_iter: int = 2000,
    tol: float = 1e-9,
) -> LassoFit:
    """LASSO with the BCH data-driven penalty and heteroskedastic loadings.

    The penalty **level** is set by theory rather than cross-validation
    (:func:`lasso_penalty`) and the per-covariate **loadings** are refined
    iteratively from the residuals,
    ``psi_j = sqrt( (1/n) sum_i x_ij^2 e_i^2 )`` -- the heteroskedasticity-robust
    form of BCH. Cross-validating the penalty would be both slower and, inside a
    cross-fitting loop, an extra place for information to leak.

    Parameters
    ----------
    x : ndarray, shape (n, p)
    y : ndarray, shape (n,)
    c : float, default=1.1
    gamma : float, optional
    n_loading_iter : int, default=3
        Number of loading refinement passes (BCH suggest a small number).
    max_iter, tol :
        Coordinate-descent controls.

    Returns
    -------
    LassoFit
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"`x` has {x.shape[0]} rows but `y` has {y.shape[0]}.")
    n, p = x.shape
    if p == 0:
        return LassoFit(
            np.zeros(0), float(y.mean()), np.zeros(0, dtype=int), 0.0, np.zeros(0), 0
        )
    xm = x.mean(axis=0)
    ym = float(y.mean())
    xc = x - xm
    yc = y - ym
    alpha = lasso_penalty(n, p, c=c, gamma=gamma)

    # Initial loadings from the BCH "pilot": psi_j = sqrt(mean(x_j^2 * y^2)).
    loadings = np.sqrt(np.maximum((xc**2 * (yc**2)[:, None]).mean(axis=0), 1e-300))
    beta = np.zeros(p)
    it = 0
    for pass_no in range(1, n_loading_iter + 1):
        it = pass_no
        beta = _coordinate_descent(
            xc, yc, alpha, loadings, max_iter=max_iter, tol=tol, beta0=beta
        )
        resid = yc - xc @ beta
        new_loadings = np.sqrt(
            np.maximum((xc**2 * (resid**2)[:, None]).mean(axis=0), 1e-300)
        )
        if np.allclose(new_loadings, loadings, rtol=1e-6):
            loadings = new_loadings
            break
        loadings = new_loadings

    selected = np.flatnonzero(beta != 0.0)
    return LassoFit(
        coef=beta,
        intercept=ym - float(xm @ beta),
        selected=selected,
        alpha=alpha,
        loadings=loadings,
        n_iter=it,
    )


# --------------------------------------------------------------------------- #
# Learner plumbing
# --------------------------------------------------------------------------- #
def _ridge_fit(
    x: np.ndarray, y: np.ndarray, ridge: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Closed-form ridge on centred data, returned as a prediction callable."""
    xm = x.mean(axis=0)
    ym = float(y.mean())
    xc = x - xm
    k = xc.shape[1]
    scale = float((xc**2).sum()) / max(xc.shape[0], 1) / max(k, 1)
    lam = ridge * max(scale, 1e-12) * xc.shape[0]
    beta = np.linalg.solve(xc.T @ xc + lam * np.eye(k), xc.T @ (y - ym))
    return lambda z: (np.asarray(z, dtype=float) - xm) @ beta + ym


def post_lasso(
    x: np.ndarray, y: np.ndarray, **kwargs: object
) -> Callable[[np.ndarray], np.ndarray]:
    """Post-LASSO: select with the rigorous LASSO, then refit OLS unpenalised.

    LASSO shrinkage biases the *predictions*, and inside DML that bias survives
    the orthogonalisation and shows up as bias in ``theta``. Refitting OLS on the
    selected support removes the shrinkage while keeping the selection, which is
    why Belloni-Chernozhukov-Hansen recommend post-LASSO over LASSO for exactly
    this role. This is the default nuisance learner.

    Parameters
    ----------
    x : ndarray, shape (n, p)
    y : ndarray, shape (n,)
    **kwargs
        Forwarded to :func:`rigorous_lasso`.

    Returns
    -------
    callable
        ``predict(x_new) -> ndarray``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    fit = rigorous_lasso(x, y, **kwargs)  # type: ignore[arg-type]
    sel = np.asarray(fit.selected, dtype=int)
    if sel.size == 0 or sel.size >= x.shape[0] - 1:
        if sel.size == 0:
            mean = float(y.mean())
            return lambda z, _m=mean: np.full(np.asarray(z).shape[0], _m)
        sel = sel[: max(x.shape[0] // 2, 1)]
    design = np.column_stack([np.ones(x.shape[0]), x[:, sel]])
    beta = ols(design, y)[0]

    def _predict(z: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z, dtype=float))
        return beta[0] + z[:, sel] @ beta[1:]

    return _predict


def _make_learner(spec: object, ridge: float = 1e-3) -> Callable:
    """Turn a learner spec into a ``fit(x, y) -> predict(x)`` factory."""
    if callable(spec) and not hasattr(spec, "fit"):
        return spec  # type: ignore[return-value]
    if isinstance(spec, str):
        if spec == "post-lasso":
            return post_lasso
        if spec == "lasso":
            return lambda x, y: rigorous_lasso(x, y).predict
        if spec == "ridge":
            return lambda x, y: _ridge_fit(x, y, ridge)
        if spec == "ols":

            def _ols_fit(x: np.ndarray, y: np.ndarray) -> Callable:
                design = np.column_stack([np.ones(x.shape[0]), x])
                beta = ols(design, y)[0]
                return lambda z: beta[0] + np.asarray(z, dtype=float) @ beta[1:]

            return _ols_fit
        raise ValueError(
            f"unknown learner {spec!r}; expected 'post-lasso', 'lasso', "
            "'ridge', 'ols', a callable `fit(x, y) -> predict`, or an object "
            "with fit/predict."
        )
    if hasattr(spec, "fit") and hasattr(spec, "predict"):
        from copy import deepcopy

        def _sk_fit(x: np.ndarray, y: np.ndarray) -> Callable:
            model = deepcopy(spec)
            model.fit(x, y)  # type: ignore[attr-defined]
            return lambda z: np.asarray(model.predict(z), dtype=float).ravel()  # type: ignore[attr-defined]

        return _sk_fit
    raise TypeError(f"cannot interpret learner spec of type {type(spec).__name__!r}.")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class DMLResult:
    """A cross-fitted double machine learning estimate.

    Attributes
    ----------
    treatment : str
    theta, std_error, tstat, pvalue : float
    conf_int : tuple of float
        Two-sided ``1 - alpha`` normal interval.
    y_resid, d_resid : ndarray
        Out-of-fold residuals ``Y - l_hat(X)`` and ``D - m_hat(X)``. These are
        the orthogonalised, cross-fitted **features** -- what you feed a
        downstream model when you want the part of ``D`` that the rest of the
        panel cannot explain.
    n_obs, n_folds : int
    horizon, embargo : int
        Purge/embargo settings the folds were built with.
    fold_thetas : ndarray
        Per-fold estimate, for a quick stability check.
    learner : str
    """

    treatment: str
    theta: float
    std_error: float
    tstat: float
    pvalue: float
    conf_int: tuple[float, float]
    y_resid: np.ndarray
    d_resid: np.ndarray
    n_obs: int
    n_folds: int
    horizon: int
    embargo: int
    fold_thetas: np.ndarray
    learner: str
    extra: dict = field(default_factory=dict)

    def summary(self) -> pl.DataFrame:
        """One-row coefficient table."""
        return pl.DataFrame(
            {
                "term": [self.treatment],
                "estimate": [self.theta],
                "std_error": [self.std_error],
                "t_stat": [self.tstat],
                "p_value": [self.pvalue],
                "ci_lower": [self.conf_int[0]],
                "ci_upper": [self.conf_int[1]],
            }
        )


@dataclass
class PDSResult:
    """A post-double-selection LASSO estimate.

    Attributes
    ----------
    treatment : str
    theta, std_error, tstat, pvalue : float
    conf_int : tuple of float
    selected : list of str
        Union of the controls selected by the two LASSO steps.
    selected_y, selected_d : list of str
        The two individual selections.
    n_obs : int
    """

    treatment: str
    theta: float
    std_error: float
    tstat: float
    pvalue: float
    conf_int: tuple[float, float]
    selected: list[str]
    selected_y: list[str]
    selected_d: list[str]
    n_obs: int

    def summary(self) -> pl.DataFrame:
        """One-row coefficient table."""
        return pl.DataFrame(
            {
                "term": [self.treatment],
                "estimate": [self.theta],
                "std_error": [self.std_error],
                "t_stat": [self.tstat],
                "p_value": [self.pvalue],
                "ci_lower": [self.conf_int[0]],
                "ci_upper": [self.conf_int[1]],
                "n_selected": [len(self.selected)],
            }
        )


# --------------------------------------------------------------------------- #
# Data extraction
# --------------------------------------------------------------------------- #
def _dml_arrays(
    panel: PanelFrame, y: str, d: str, controls: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Collect ``(y, d, X, time_codes, control_names)`` from a panel."""
    cols = [y, d, *controls]
    frame = (
        panel.sort_panel()
        .lazy()
        .select(
            [
                pl.col(panel.time_col),
                *[pl.col(c).cast(pl.Float64).alias(c) for c in cols],
            ]
        )
        .collect()
        .filter(
            pl.all_horizontal(
                [pl.col(c).is_not_null() & pl.col(c).is_not_nan() for c in cols]
            )
        )
    )
    if frame.height == 0:
        raise ValueError("no complete observations for the DML fit.")
    tcodes, _ = factorize(frame[panel.time_col].to_numpy())
    yv = frame[y].to_numpy()
    dv = frame[d].to_numpy()
    xv = (
        np.column_stack([frame[c].to_numpy() for c in controls])
        if controls
        else np.zeros((frame.height, 0))
    )
    return yv, dv, xv, tcodes, list(controls)


def _resolve_controls(
    panel: PanelFrame, y: str, d: str, controls: Sequence[str] | None
) -> list[str]:
    """Default the control set to every other numeric feature column."""
    if controls is not None:
        missing = [c for c in controls if c not in panel.columns]
        if missing:
            raise ValueError(
                f"control column(s) {missing} not found. Available: {panel.columns}."
            )
        return list(controls)
    schema = panel.schema
    return [c for c in panel.feature_cols if c not in (y, d) and schema[c].is_numeric()]


# --------------------------------------------------------------------------- #
# Post-double selection
# --------------------------------------------------------------------------- #
def post_double_selection(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    y: str,
    d: str,
    controls: Sequence[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
    c: float = 1.1,
    gamma: float | None = None,
    alpha: float = 0.05,
) -> PDSResult:
    """Belloni-Chernozhukov-Hansen post-double-selection LASSO.

    Runs two rigorous LASSOs -- ``y`` on the controls and ``d`` on the controls --
    then estimates ``theta`` by OLS of ``y`` on ``d`` **and the union** of the two
    selected control sets, with HC1 standard errors. Selecting on the treatment
    equation as well as the outcome equation is what makes the resulting
    confidence interval uniformly valid: a control that predicts ``d`` strongly
    but ``y`` weakly would be dropped by single selection and its omission would
    bias ``theta``.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
    y : str
        Outcome.
    d : str
        Treatment / feature of interest.
    controls : sequence of str, optional
        Defaults to every other numeric feature column.
    entity, time : str, optional
    c, gamma : float
        Penalty tuning (see :func:`lasso_penalty`).
    alpha : float, default=0.05
        Confidence-interval level.

    Returns
    -------
    PDSResult
    """
    panel = (
        data
        if isinstance(data, PanelFrame)
        else as_panel(data, entity=entity, time=time)
    )
    ctrl = _resolve_controls(panel, y, d, controls)
    yv, dv, xv, _, names = _dml_arrays(panel, y, d, ctrl)
    n = yv.size

    if xv.shape[1] == 0:
        sel_y: list[str] = []
        sel_d: list[str] = []
        union_idx = np.zeros(0, dtype=int)
    else:
        fit_y = rigorous_lasso(xv, yv, c=c, gamma=gamma)
        fit_d = rigorous_lasso(xv, dv, c=c, gamma=gamma)
        sel_y = [names[i] for i in fit_y.selected]
        sel_d = [names[i] for i in fit_d.selected]
        union_idx = np.union1d(fit_y.selected, fit_d.selected).astype(int)

    design = np.column_stack(
        [np.ones(n), dv, *([xv[:, union_idx]] if union_idx.size else [])]
    )
    beta, resid, xtx_inv = ols(design, yv)
    scores = design * resid[:, None]
    meat = scores.T @ scores
    k = design.shape[1]
    hc1 = n / max(n - k, 1)
    vcov = hc1 * (xtx_inv @ meat @ xtx_inv)
    theta = float(beta[1])
    se = float(np.sqrt(max(vcov[1, 1], 0.0)))
    tstat = theta / se if se > 0 else float("nan")
    pvalue = (
        2.0 * (1.0 - float(norm_cdf(abs(tstat))))
        if np.isfinite(tstat)
        else float("nan")
    )
    z = float(norm_ppf(1.0 - alpha / 2.0))
    return PDSResult(
        treatment=d,
        theta=theta,
        std_error=se,
        tstat=tstat,
        pvalue=pvalue,
        conf_int=(theta - z * se, theta + z * se),
        selected=[names[i] for i in union_idx],
        selected_y=sel_y,
        selected_d=sel_d,
        n_obs=n,
    )


# --------------------------------------------------------------------------- #
# Cross-fitted DML
# --------------------------------------------------------------------------- #
def dml_partial_linear(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    y: str,
    d: str,
    controls: Sequence[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
    n_folds: int = 5,
    horizon: int = 0,
    embargo: int = 0,
    learner: object = "post-lasso",
    ridge: float = 1e-3,
    alpha: float = 0.05,
) -> DMLResult:
    """Cross-fitted DML for the partially linear model.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
    y : str
        Outcome.
    d : str
        Treatment / feature of interest.
    controls : sequence of str, optional
        Defaults to every other numeric feature column.
    entity, time : str, optional
    n_folds : int, default=5
        Number of cross-fitting folds.
    horizon : int, default=0
        Label horizon used for **purging**. Set this to the horizon of an
        overlapping target (e.g. ``5`` for a 5-day forward return).
    embargo : int, default=0
        Time-steps embargoed after each held-out block. For an overlapping
        target this should be at least ``horizon``.
    learner : str or callable or estimator, default="post-lasso"
        ``"post-lasso"`` (rigorous BCH LASSO selection + unpenalised OLS refit),
        ``"lasso"``, ``"ridge"``, ``"ols"``, a callable
        ``fit(X, y) -> predict``, or any object with ``fit`` / ``predict``
        (e.g. a scikit-learn regressor -- import it yourself; this module adds
        no dependency).
    ridge : float, default=1e-3
        Ridge strength when ``learner="ridge"``.
    alpha : float, default=0.05

    Returns
    -------
    DMLResult

    Raises
    ------
    ValueError
        If the panel has too few unique dates for ``n_folds`` folds, or if the
        cross-fitted treatment residual has no variation.

    Notes
    -----
    Folds are produced by
    :class:`~polars_features.core.model_selection.PurgedKFold` over the sorted
    unique time index, so they are contiguous in time, purged and embargoed. A
    follow-up will offer combinatorial purged CV (``validation/_cv.py``) as an
    alternative fold source; the score and variance formulae are unchanged.
    """
    panel = (
        data
        if isinstance(data, PanelFrame)
        else as_panel(data, entity=entity, time=time)
    )
    ctrl = _resolve_controls(panel, y, d, controls)
    yv, dv, xv, tcodes, _ = _dml_arrays(panel, y, d, ctrl)
    n = yv.size
    if n_folds < 2:
        raise ValueError(f"`n_folds` must be >= 2, got {n_folds}.")

    fit_fn = _make_learner(learner, ridge=ridge)
    learner_name = learner if isinstance(learner, str) else type(learner).__name__

    # Fold source: PanelKit's purged/embargoed splitter over the unique times.
    split_frame = pl.DataFrame(
        {"__entity": np.zeros(n, dtype=np.int64), "__time": tcodes}
    )
    splitter = PurgedKFold(
        n_splits=n_folds, horizon=horizon, embargo=embargo, return_indices=True
    )
    split_panel = PanelFrame(split_frame, entity="__entity", time="__time")

    y_res = np.full(n, np.nan)
    d_res = np.full(n, np.nan)
    fold_thetas: list[float] = []
    for train_pos, test_pos in splitter.split(split_panel):
        train_mask = np.isin(tcodes, train_pos)
        test_mask = np.isin(tcodes, test_pos)
        if train_mask.sum() < 2 or test_mask.sum() == 0:
            continue
        if xv.shape[1] == 0:
            ly = float(yv[train_mask].mean())
            ld = float(dv[train_mask].mean())
            yr = yv[test_mask] - ly
            dr = dv[test_mask] - ld
        else:
            pred_y = fit_fn(xv[train_mask], yv[train_mask])
            pred_d = fit_fn(xv[train_mask], dv[train_mask])
            yr = yv[test_mask] - np.asarray(pred_y(xv[test_mask]), dtype=float).ravel()
            dr = dv[test_mask] - np.asarray(pred_d(xv[test_mask]), dtype=float).ravel()
        y_res[test_mask] = yr
        d_res[test_mask] = dr
        denom = float(dr @ dr)
        if denom > 0:
            fold_thetas.append(float(dr @ yr) / denom)

    ok = np.isfinite(y_res) & np.isfinite(d_res)
    if ok.sum() < 3:
        raise ValueError(
            "cross-fitting produced too few out-of-fold residuals; reduce "
            "`n_folds`, `horizon` or `embargo`."
        )
    yr = y_res[ok]
    dr = d_res[ok]
    n_eff = yr.size
    jac = float(dr @ dr) / n_eff
    d_scale = float(np.var(dv)) if dv.size else 0.0
    if jac <= 1e-10 * max(d_scale, 1.0):
        raise ValueError(
            "the cross-fitted treatment residual has no variation: the controls "
            f"explain {d!r} perfectly, so its effect is not identified."
        )
    theta = float(dr @ yr) / (n_eff * jac)
    psi = (yr - theta * dr) * dr
    var = float(psi @ psi) / n_eff / (jac * jac) / n_eff
    se = float(np.sqrt(max(var, 0.0)))
    tstat = theta / se if se > 0 else float("nan")
    pvalue = (
        2.0 * (1.0 - float(norm_cdf(abs(tstat))))
        if np.isfinite(tstat)
        else float("nan")
    )
    z = float(norm_ppf(1.0 - alpha / 2.0))

    return DMLResult(
        treatment=d,
        theta=theta,
        std_error=se,
        tstat=tstat,
        pvalue=pvalue,
        conf_int=(theta - z * se, theta + z * se),
        y_resid=y_res,
        d_resid=d_res,
        n_obs=n_eff,
        n_folds=n_folds,
        horizon=horizon,
        embargo=embargo,
        fold_thetas=np.asarray(fold_thetas, dtype=float),
        learner=str(learner_name),
        extra={"n_controls": xv.shape[1], "controls": ctrl},
    )


# --------------------------------------------------------------------------- #
# Transformer
# --------------------------------------------------------------------------- #
class DoubleMLTransformer(PanelTransformer):
    """Cross-fitted DML effect estimate plus orthogonalised residual features.

    ``fit`` (training rows only):

    1. runs :func:`dml_partial_linear` with purged / embargoed folds and stores
       ``theta_``, ``std_error_`` and ``conf_int_``;
    2. refits the two nuisance functions ``l(X)`` and ``m(X)`` on **all** the
       training rows and freezes them.

    ``transform`` applies the frozen nuisance models and emits

    * ``<prefix>y_resid`` -- ``y - l_hat(X)``
    * ``<prefix>d_resid`` -- ``d - m_hat(X)``
    * ``<prefix>effect``  -- ``theta_ * d_resid``, the debiased contribution of
      the treatment.

    Parameters
    ----------
    y, d : str
    controls : sequence of str, optional
    n_folds, horizon, embargo, learner, ridge, alpha :
        Forwarded to :func:`dml_partial_linear`.
    prefix : str, default="dml_"

    Attributes
    ----------
    result_ : DMLResult
    theta_, std_error_ : float
    conf_int_ : tuple of float

    Warnings
    --------
    Residuals emitted for the *training* rows are in-sample (they use the
    all-training nuisance fit). Use ``result_.y_resid`` / ``result_.d_resid``
    when you want the out-of-fold versions.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        *,
        y: str,
        d: str,
        controls: Sequence[str] | None = None,
        n_folds: int = 5,
        horizon: int = 0,
        embargo: int = 0,
        learner: object = "post-lasso",
        ridge: float = 1e-3,
        alpha: float = 0.05,
        prefix: str = "dml_",
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self.y = y
        self.d = d
        self.controls = list(controls) if controls is not None else None
        self.n_folds = int(n_folds)
        self.horizon = int(horizon)
        self.embargo = int(embargo)
        self.learner = learner
        self.ridge = float(ridge)
        self.alpha = float(alpha)
        self.prefix = prefix
        self.result_: DMLResult | None = None
        self.theta_: float = float("nan")
        self.std_error_: float = float("nan")
        self.conf_int_: tuple[float, float] = (float("nan"), float("nan"))
        self._controls_: list[str] = []
        self._pred_y = None
        self._pred_d = None

    def _fit(self, panel: PanelFrame) -> None:
        ctrl = _resolve_controls(panel, self.y, self.d, self.controls)
        self.result_ = dml_partial_linear(
            panel,
            y=self.y,
            d=self.d,
            controls=ctrl,
            n_folds=self.n_folds,
            horizon=self.horizon,
            embargo=self.embargo,
            learner=self.learner,
            ridge=self.ridge,
            alpha=self.alpha,
        )
        self.theta_ = self.result_.theta
        self.std_error_ = self.result_.std_error
        self.conf_int_ = self.result_.conf_int
        self._controls_ = ctrl

        yv, dv, xv, _, _ = _dml_arrays(panel, self.y, self.d, ctrl)
        if xv.shape[1] == 0:
            ymean, dmean = float(yv.mean()), float(dv.mean())
            self._pred_y = lambda z, _m=ymean: np.full(z.shape[0], _m)
            self._pred_d = lambda z, _m=dmean: np.full(z.shape[0], _m)
        else:
            fit_fn = _make_learner(self.learner, ridge=self.ridge)
            self._pred_y = fit_fn(xv, yv)
            self._pred_d = fit_fn(xv, dv)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        frame = panel.sort_panel().collect()
        ctrl = self._controls_
        missing = [c for c in [self.d, *ctrl] if c not in frame.columns]
        if missing:
            raise ValueError(
                f"column(s) {missing} required by the fitted DoubleMLTransformer "
                f"are missing. Available: {frame.columns}."
            )
        xv = (
            np.column_stack(
                [frame[c].cast(pl.Float64).fill_null(0.0).to_numpy() for c in ctrl]
            )
            if ctrl
            else np.zeros((frame.height, 0))
        )
        dv = frame[self.d].cast(pl.Float64).to_numpy()
        d_resid = dv - np.asarray(self._pred_d(xv), dtype=float).ravel()
        cols = {
            f"{self.prefix}d_resid": d_resid,
            f"{self.prefix}effect": self.theta_ * d_resid,
        }
        if self.y in frame.columns:
            yv = frame[self.y].cast(pl.Float64).to_numpy()
            cols[f"{self.prefix}y_resid"] = (
                yv - np.asarray(self._pred_y(xv), dtype=float).ravel()
            )
        out = frame.with_columns(
            [pl.Series(name, values) for name, values in cols.items()]
        )
        return PanelFrame(
            out, entity=panel.entity_col, time=panel.time_col, validate=False
        )

    @property
    def output_names(self) -> list[str]:
        """Names of the emitted feature columns."""
        return [
            f"{self.prefix}y_resid",
            f"{self.prefix}d_resid",
            f"{self.prefix}effect",
        ]
