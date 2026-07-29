"""Parameter estimation: nonlinear curve fitting and linear least squares.

This module wraps :func:`scipy.optimize.curve_fit` for nonlinear models
and delegates linear problems to the Rust kernel ``precision_physkit._core``
(imported lazily inside each function, so this module remains importable
while the extension is not built):

* :func:`linear_lstsq` — truncated-SVD least squares for a single data
  vector;
* :func:`iterative_multichannel` — iteratively reweighted least squares
  for multi-channel systems with unknown per-channel noise variances;
* :func:`polyfit` — polynomial fitting as a convenience wrapper
  demonstrating the kernel-based linear solver.

All estimators return a :class:`FitResult` with named parameters, 1-sigma
uncertainties, covariance, residuals and summary statistics, and report
failures through the result object instead of raising (unless explicitly
requested).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize as _opt

__all__ = [
    "FitResult",
    "curve_fit",
    "linear_lstsq",
    "iterative_multichannel",
    "polyfit",
]


def _load_core():
    """Import the Rust kernel lazily (it may not be built at import time)."""
    from . import _core

    return _core


@dataclass
class FitResult:
    """Container for the outcome of a parameter estimation.

    Attributes
    ----------
    params : dict[str, float]
        Estimated parameter values keyed by parameter name.
    perr : dict[str, float]
        1-sigma (standard) uncertainties per parameter, computed as
        ``sqrt(diag(cov))``. Entries are ``inf`` when the covariance
        could not be determined and ``nan`` for non-positive variances.
    cov : ndarray, shape (P, P)
        Covariance matrix of the estimated parameters. Empty ``(0, 0)``
        array for failed nonlinear fits.
    resid : ndarray, shape (N,)
        Residual vector ``y - model(x)`` (unweighted). Empty array for
        failed nonlinear fits.
    success : bool
        Whether the estimation completed successfully. For
        :func:`iterative_multichannel` this reflects convergence of the
        iteration; the returned parameters are then still the best
        effort of the last iteration.
    message : str
        Human-readable status or failure description.
    stats : dict
        Summary statistics. Common keys are ``"chi2"`` (weighted by
        ``1/sigma**2`` when 1-D ``sigma`` was given), ``"redchi"``
        (``chi2 / dof``), ``"r2"`` (coefficient of determination) and
        ``"dof"``. Solvers may add keys such as ``"rank"``, ``"n_iter"``
        or ``"weights"``.
    """

    params: dict[str, float]
    perr: dict[str, float]
    cov: NDArray[np.floating[Any]]
    resid: NDArray[np.floating[Any]]
    success: bool
    message: str
    stats: dict[str, Any] = field(default_factory=dict)


def _model_param_names(model: Callable[..., Any]) -> list[str] | None:
    """Infer parameter names from the model signature (all args after x)."""
    try:
        sig = inspect.signature(model)
    except (TypeError, ValueError):
        return None
    names = [
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return names[1:] if names else None


def _resolve_names(
    param_names: list[str] | tuple[str, ...] | None,
    inferred: list[str] | None,
    n: int,
) -> list[str]:
    """Build a name list of length ``n``: explicit > inferred > p0…p{n-1}."""
    names = list(param_names) if param_names is not None else list(inferred or [])
    if len(names) < n:
        names.extend(f"p{i}" for i in range(len(names), n))
    return names[:n]


def _perr_from_cov(cov: NDArray[Any]) -> NDArray[np.float64]:
    """1-sigma uncertainties from a covariance matrix diagonal."""
    diag = np.diag(np.asarray(cov, dtype=float))
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.where(diag >= 0, diag, np.nan))


def _fit_stats(
    resid: NDArray[np.float64],
    y: NDArray[np.float64],
    n_params: int,
    sigma: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """Goodness-of-fit statistics: chi2, reduced chi2, R^2, dof."""
    n = y.size
    dof = n - n_params
    if sigma is not None and sigma.ndim == 1:
        chi2 = float(np.sum((resid / sigma) ** 2))
    else:
        chi2 = float(np.sum(resid**2))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "chi2": chi2,
        "redchi": chi2 / dof if dof > 0 else float("nan"),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "dof": int(dof),
    }


def curve_fit(
    model: Callable[..., ArrayLike],
    x: ArrayLike,
    y: ArrayLike,
    p0: ArrayLike | None = None,
    sigma: ArrayLike | None = None,
    bounds: tuple[ArrayLike, ArrayLike] = (-np.inf, np.inf),
    param_names: list[str] | tuple[str, ...] | None = None,
    raise_on_failure: bool = False,
    **kw: Any,
) -> FitResult:
    """Nonlinear least-squares fit of ``model(x, *params)`` to data.

    Thin wrapper around :func:`scipy.optimize.curve_fit` returning a
    :class:`FitResult`. Parameter names are taken from ``param_names`` if
    given, otherwise inferred from the model's signature (every
    positional parameter after ``x``), otherwise synthesised as
    ``p0`` … ``p{P-1}``.

    Parameters
    ----------
    model : callable
        ``model(x, *params) -> y_model`` with the independent variable as
        first argument.
    x, y : array_like, shape (N,)
        Independent variable and measured data.
    p0 : array_like, optional
        Initial parameter guess. If None, scipy introspects the model
        signature and starts at 1 for every parameter.
    sigma : array_like, optional
        1-sigma uncertainties of ``y`` (1-D) or full covariance (2-D),
        passed through to scipy. When 1-D, ``stats["chi2"]`` is the
        weighted sum ``sum((resid / sigma)**2)``.
    bounds : tuple of array_like, optional
        Lower and upper bounds per parameter; unbounded by default.
    param_names : sequence of str, optional
        Explicit parameter names, overriding signature inference.
    raise_on_failure : bool, optional
        If True, re-raise solver exceptions instead of returning a
        ``FitResult`` with ``success=False``.
    **kw
        Additional keyword arguments for :func:`scipy.optimize.curve_fit`
        (e.g. ``absolute_sigma=True``, ``maxfev=...``).

    Returns
    -------
    FitResult
        On failure (e.g. non-convergence) ``success`` is False,
        ``message`` carries the original exception, and known parameter
        names map to ``nan``.

    Raises
    ------
    Exception
        Only when ``raise_on_failure=True``; scipy's original exception
        is re-raised unchanged.

    Notes
    -----
    When the covariance cannot be estimated (flat directions in parameter
    space), scipy returns ``inf`` entries in the covariance and the
    corresponding ``perr`` values are ``inf`` — treat those parameters as
    unconstrained by the data. Pass ``absolute_sigma=True`` to interpret
    ``sigma`` in a statistical (rather than relative) sense, which makes
    ``cov`` and ``perr`` carry absolute units.

    Examples
    --------
    >>> def gauss(x, amp, mu, sig):
    ...     return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    >>> result = curve_fit(gauss, xdata, ydata, p0=[1.0, 0.0, 1.0])  # doctest: +SKIP
    >>> result.params["mu"], result.perr["mu"]  # doctest: +SKIP
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    sigma_arr = None if sigma is None else np.asarray(sigma, dtype=float)
    inferred = _model_param_names(model)
    try:
        popt, pcov = _opt.curve_fit(
            model, x_arr, y_arr, p0=p0, sigma=sigma_arr, bounds=bounds, **kw
        )
    except Exception as exc:
        if raise_on_failure:
            raise
        n_hint = len(p0) if p0 is not None else len(inferred or [])
        names = _resolve_names(param_names, inferred, n_hint)
        nan = float("nan")
        return FitResult(
            params={name: nan for name in names},
            perr={name: nan for name in names},
            cov=np.empty((0, 0)),
            resid=np.empty(0),
            success=False,
            message=f"{type(exc).__name__}: {exc}",
            stats={},
        )
    popt = np.atleast_1d(np.asarray(popt, dtype=float))
    names = _resolve_names(param_names, inferred, popt.size)
    resid = y_arr - np.asarray(model(x_arr, *popt), dtype=float)
    perr = _perr_from_cov(pcov)
    stats = _fit_stats(resid, y_arr, popt.size, sigma=sigma_arr)
    return FitResult(
        params=dict(zip(names, popt.tolist(), strict=True)),
        perr=dict(zip(names, perr.tolist(), strict=True)),
        cov=np.asarray(pcov, dtype=float),
        resid=resid,
        success=True,
        message="ok",
        stats=stats,
    )


def linear_lstsq(
    design: ArrayLike,
    y: ArrayLike,
    rcond: float | None = None,
    param_names: list[str] | tuple[str, ...] | None = None,
) -> FitResult:
    """Linear least squares via truncated SVD (Rust kernel).

    Solves ``min_theta ||design @ theta - y||_2`` using a singular value
    decomposition of the design matrix. Singular values smaller than
    ``rcond`` times the largest singular value are *truncated* (treated
    as zero and excluded from the pseudo-inverse). This is the standard
    numerical stabilisation for ill-conditioned or rank-deficient design
    matrices: directions of parameter space that the data do not
    constrain are projected out instead of amplifying noise, at the cost
    of a small bias. The effective solution set is reported via
    ``stats["rank"]``.

    Parameters
    ----------
    design : array_like, shape (N, P)
        Design matrix; row ``i`` holds the basis functions evaluated at
        measurement ``i``.
    y : array_like, shape (N,)
        Measured data vector.
    rcond : float, optional
        Relative cut-off for small singular values. None lets the kernel
        use its default (machine-precision scaled).
    param_names : sequence of str, optional
        Parameter names; defaults to ``p0`` … ``p{P-1}``.

    Returns
    -------
    FitResult
        ``stats`` additionally contains ``"rank"`` (effective numerical
        rank after truncation) and ``"rcond"``. If the design matrix is
        rank-deficient, ``message`` reports it (``success`` stays True:
        the truncated solution is still the best unbiased estimate in the
        solvable subspace).

    Raises
    ------
    ValueError
        If shapes are inconsistent (``design`` not 2-D, or ``y`` length
        mismatching ``design`` rows).
    ImportError
        If the Rust extension ``precision_physkit._core`` is not built.
    """
    design_arr = np.asarray(design, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if design_arr.ndim != 2:
        raise ValueError(f"design must be a 2-D (N, P) array, got {design_arr.ndim}-D")
    if y_arr.ndim != 1 or y_arr.shape[0] != design_arr.shape[0]:
        raise ValueError(
            "y must be 1-D with one entry per design row; got "
            f"{y_arr.shape} vs design {design_arr.shape}"
        )
    core = _load_core()
    theta, resid, rank, cov = core.svd_lstsq(design_arr, y_arr, rcond=rcond)
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    cov = np.asarray(cov, dtype=float)
    resid = np.asarray(resid, dtype=float)
    names = _resolve_names(param_names, None, theta.size)
    stats = _fit_stats(resid, y_arr, theta.size)
    stats["rank"] = int(rank)
    stats["rcond"] = rcond
    deficient = int(rank) < theta.size
    message = (
        f"rank-deficient design matrix: rank {int(rank)} < {theta.size} parameters"
        if deficient
        else "ok"
    )
    return FitResult(
        params=dict(zip(names, theta.tolist(), strict=True)),
        perr=dict(zip(names, _perr_from_cov(cov).tolist(), strict=True)),
        cov=cov,
        resid=resid,
        success=True,
        message=message,
        stats=stats,
    )


def iterative_multichannel(
    design: ArrayLike,
    y: ArrayLike,
    group_sizes: list[int] | tuple[int, ...],
    max_iter: int = 100,
    tol: float = 1e-12,
    param_names: list[str] | tuple[str, ...] | None = None,
) -> FitResult:
    """Iteratively reweighted least squares for multi-channel systems.

    Performs linear parameter estimation when several data channels (each
    contributing a contiguous block of rows in ``design``/``y``) share the
    same physical parameters but have *unknown, mutually different* noise
    variances. The kernel alternates between two steps until the
    parameter change drops below ``tol`` or ``max_iter`` is reached:

    1. estimate each channel's noise variance from its residuals and set
       the channel weight to the inverse variance;
    2. solve the weighted linear least-squares problem for the shared
       parameters.

    This is the Gauss-Markov / feasible-GLS strategy: with consistent
    weight estimates the weighted solution is asymptotically efficient,
    and down-weights channels that are dominated by noise.

    Parameters
    ----------
    design : array_like, shape (N, P)
        Joint design matrix of all stacked channels.
    y : array_like, shape (N,)
        Stacked data vector of all channels.
    group_sizes : sequence of int
        Row counts of the channel blocks; ``sum(group_sizes)`` must equal
        ``N``. E.g. ``(1000, 1000)`` for two channels of 1000 samples.
    max_iter : int, optional
        Maximum number of weight/parameter alternations.
    tol : float, optional
        Convergence tolerance on the parameter update.
    param_names : sequence of str, optional
        Parameter names; defaults to ``p0`` … ``p{P-1}``.

    Returns
    -------
    FitResult
        ``success`` reflects convergence of the iteration; on
        non-convergence the parameters of the last iteration are still
        returned. ``stats`` additionally contains ``"n_iter"`` and the
        final per-channel ``"weights"``.

    Raises
    ------
    ValueError
        If shapes are inconsistent or ``group_sizes`` does not sum to the
        number of rows.
    ImportError
        If the Rust extension ``precision_physkit._core`` is not built.
    """
    design_arr = np.asarray(design, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if design_arr.ndim != 2:
        raise ValueError(f"design must be a 2-D (N, P) array, got {design_arr.ndim}-D")
    if y_arr.ndim != 1 or y_arr.shape[0] != design_arr.shape[0]:
        raise ValueError(
            "y must be 1-D with one entry per design row; got "
            f"{y_arr.shape} vs design {design_arr.shape}"
        )
    groups = [int(g) for g in group_sizes]
    if any(g <= 0 for g in groups) or sum(groups) != design_arr.shape[0]:
        raise ValueError(
            f"group_sizes must be positive and sum to N={design_arr.shape[0]}, "
            f"got {groups}"
        )
    core = _load_core()
    theta, weights, n_iter, converged, cov = core.iterative_wls(
        design_arr, y_arr, groups, max_iter=max_iter, tol=tol
    )
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    cov = np.asarray(cov, dtype=float)
    resid = y_arr - design_arr @ theta
    names = _resolve_names(param_names, None, theta.size)
    stats = _fit_stats(resid, y_arr, theta.size)
    stats["n_iter"] = int(n_iter)
    stats["weights"] = np.asarray(weights, dtype=float)
    stats["tol"] = float(tol)
    message = (
        f"converged after {int(n_iter)} iterations"
        if converged
        else f"did not converge within {max_iter} iterations"
    )
    return FitResult(
        params=dict(zip(names, theta.tolist(), strict=True)),
        perr=dict(zip(names, _perr_from_cov(cov).tolist(), strict=True)),
        cov=cov,
        resid=resid,
        success=bool(converged),
        message=message,
        stats=stats,
    )


def polyfit(
    x: ArrayLike,
    y: ArrayLike,
    deg: int,
) -> FitResult:
    """Fit a polynomial of degree ``deg`` via :func:`linear_lstsq`.

    Builds a Vandermonde design matrix with *ascending* powers and solves
    it with the truncated-SVD kernel, demonstrating kernel-based linear
    estimation.

    Parameters
    ----------
    x, y : array_like, shape (N,)
        Independent variable and measured data.
    deg : int
        Polynomial degree (>= 0).

    Returns
    -------
    FitResult
        Parameters are named ``c0`` … ``c{deg}``, where ``ci`` multiplies
        ``x**i`` — i.e. ``c0`` is the constant term. Note this is the
        *opposite* ordering of :func:`numpy.polyfit`.

    Raises
    ------
    ValueError
        If ``deg`` is negative or the inputs are inconsistent.
    ImportError
        If the Rust extension ``precision_physkit._core`` is not built.

    Notes
    -----
    Vandermonde systems become severely ill-conditioned as the degree
    grows; the kernel's singular-value truncation keeps the solution
    usable, but for degrees beyond ~5-7 consider centering and scaling
    ``x`` (e.g. to zero mean, unit standard deviation) before fitting.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.shape != y_arr.shape:
        raise ValueError(
            f"x and y must be 1-D with identical length, got {x_arr.shape} and {y_arr.shape}"
        )
    if int(deg) < 0:
        raise ValueError(f"deg must be >= 0, got {deg}")
    design = np.vander(x_arr, N=int(deg) + 1, increasing=True)
    names = [f"c{i}" for i in range(int(deg) + 1)]
    return linear_lstsq(design, y_arr, param_names=names)
