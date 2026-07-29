"""General-purpose minimisation wrappers around :mod:`scipy.optimize`.

Provides a small, uniform result container (:class:`OptimizeResult`) for

* :func:`minimize` — local minimisation of a scalar objective;
* :func:`global_minimize` — stochastic global minimisation over a box
  (differential evolution);
* :func:`least_squares` — bound-constrained nonlinear least squares from
  a residual vector (the form nonlinear curve-fitting problems take).

This module has no dependency on the Rust kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize as _opt

__all__ = ["OptimizeResult", "minimize", "global_minimize", "least_squares"]

Objective = Callable[[NDArray[np.float64]], float]
"""Scalar objective over a parameter vector."""

Residual = Callable[[NDArray[np.float64]], NDArray[np.float64]]
"""Vector-valued residual function for least-squares problems."""


@dataclass
class OptimizeResult:
    """Uniform result container for the optimization routines.

    Attributes
    ----------
    x : ndarray, shape (P,)
        Best parameter vector found.
    fun : float
        Objective value at ``x``. For :func:`least_squares` this is the
        cost, i.e. half the sum of squared residuals (scipy's ``cost``),
        not the residual vector.
    success : bool
        Whether the solver reports convergence.
    message : str
        Solver termination message, passed through from scipy.
    n_eval : int
        Number of objective evaluations (scipy's ``nfev``).
    history : dict or None
        Reserved for iteration histories; currently always None.
    """

    x: NDArray[np.float64]
    fun: float
    success: bool
    message: str
    n_eval: int
    history: dict[str, Any] | None = None


def _to_result(res: Any, fun: float | None = None) -> OptimizeResult:
    return OptimizeResult(
        x=np.atleast_1d(np.asarray(res.x, dtype=float)),
        fun=float(res.fun) if fun is None else float(fun),
        success=bool(res.success),
        message=str(res.message),
        n_eval=int(getattr(res, "nfev", 0)),
        history=None,
    )


def minimize(
    func: Objective,
    x0: ArrayLike,
    method: str = "Nelder-Mead",
    bounds: Sequence[tuple[float, float]] | Any | None = None,
    jac: Callable[..., Any] | str | bool | None = None,
    **kw: Any,
) -> OptimizeResult:
    """Local minimisation of a scalar objective.

    Thin wrapper around :func:`scipy.optimize.minimize` returning an
    :class:`OptimizeResult`. The default method Nelder-Mead is
    derivative-free and robust for noisy or non-smooth objectives; switch
    to a gradient-based method (e.g. ``"L-BFGS-B"``) and pass ``jac``
    for smooth problems with many parameters.

    Parameters
    ----------
    func : callable
        ``func(x) -> float``, the objective to minimise.
    x0 : array_like, shape (P,)
        Starting point.
    method : str, optional
        Any solver name accepted by :func:`scipy.optimize.minimize`.
    bounds : sequence of (min, max) pairs or scipy Bounds, optional
        Parameter bounds. Only honoured by methods that support bounds
        (Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, trust-constr).
    jac : callable, str or bool, optional
        Gradient of the objective, or a finite-difference scheme name;
        ignored by derivative-free methods.
    **kw
        Additional keyword arguments passed to
        :func:`scipy.optimize.minimize`, e.g. ``tol`` or
        ``options={"maxiter": ...}``.

    Returns
    -------
    OptimizeResult
        ``success`` and ``message`` reflect the solver's own convergence
        report; no exception is raised on non-convergence.
    """
    res = _opt.minimize(func, np.asarray(x0, dtype=float), method=method, jac=jac, bounds=bounds, **kw)
    return _to_result(res)


def global_minimize(
    func: Objective,
    bounds: Sequence[tuple[float, float]],
    method: str = "differential_evolution",
    seed: int | np.random.Generator | None = None,
    **kw: Any,
) -> OptimizeResult:
    """Global minimisation over a box via differential evolution.

    Wraps :func:`scipy.optimize.differential_evolution`, a stochastic
    population-based global optimizer that does not require a starting
    point and is suited to multimodal objectives where local solvers get
    trapped.

    Parameters
    ----------
    func : callable
        ``func(x) -> float``, the objective to minimise.
    bounds : sequence of (min, max) pairs
        Box constraints for each parameter; required and finite.
    method : str, optional
        Global optimization strategy. Currently only
        ``"differential_evolution"`` is supported.
    seed : int or numpy.random.Generator, optional
        Randomness source. Differential evolution is *stochastic*:
        repeated runs explore differently and may find different local
        minima. Pass a fixed ``seed`` for reproducible results (e.g. in
        pipelines and tests); leave None for fresh randomness per run.
    **kw
        Additional keyword arguments passed to
        :func:`scipy.optimize.differential_evolution`, e.g. ``maxiter``,
        ``popsize``, ``tol``, ``polish`` or ``workers``.

    Returns
    -------
    OptimizeResult
        Best point found. Note that ``success=True`` only means the
        algorithm terminated normally, not that the point is certified
        global.

    Raises
    ------
    ValueError
        If ``method`` names an unsupported strategy.
    """
    if method != "differential_evolution":
        raise ValueError(
            f"unknown global method {method!r}; only 'differential_evolution' "
            "is supported"
        )
    res = _opt.differential_evolution(func, bounds, seed=seed, **kw)
    return _to_result(res)


def least_squares(
    residual: Residual,
    x0: ArrayLike,
    bounds: tuple[ArrayLike, ArrayLike] = (-np.inf, np.inf),
    **kw: Any,
) -> OptimizeResult:
    """Bound-constrained nonlinear least squares from a residual vector.

    Wraps :func:`scipy.optimize.least_squares`, which minimises
    ``0.5 * sum(residual(x)**2)`` subject to box bounds. Use this when a
    fitting problem is naturally expressed as a residual (e.g.
    ``lambda p: model(x, *p) - y``); for named-parameter curve fitting
    with uncertainties, prefer :func:`precision_physkit.fitting.curve_fit`.

    Parameters
    ----------
    residual : callable
        ``residual(x) -> ndarray, shape (M,)``, the residual vector.
    x0 : array_like, shape (P,)
        Starting point; must be feasible with respect to ``bounds``
        (scipy raises otherwise).
    bounds : tuple of array_like, optional
        Lower and upper bounds per parameter; unbounded by default.
    **kw
        Additional keyword arguments passed to
        :func:`scipy.optimize.least_squares`, e.g. ``method="lm"``,
        ``loss="soft_l1"`` for outlier robustness, or ``x_scale``.

    Returns
    -------
    OptimizeResult
        ``fun`` holds the final cost ``0.5 * sum(residual**2)`` (scipy's
        ``cost`` convention). Evaluate ``residual(result.x)`` again if
        the residual vector itself is needed.

    Raises
    ------
    ValueError
        Propagated from scipy if ``x0`` is infeasible or shapes are
        inconsistent.
    """
    res = _opt.least_squares(residual, np.asarray(x0, dtype=float), bounds=bounds, **kw)
    return _to_result(res, fun=float(res.cost))
