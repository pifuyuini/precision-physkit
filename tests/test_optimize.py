"""Quantitative tests for precision_physkit.optimize minimisation wrappers."""

import numpy as np
import pytest

from precision_physkit import optimize


def test_minimize_quadratic_converges():
    """Nelder-Mead finds the exact minimum of a shifted quadratic."""

    def objective(v):
        return (v[0] - 3.0) ** 2 + (v[1] + 1.0) ** 2 + 7.0

    res = optimize.minimize(objective, x0=[0.0, 0.0])
    assert res.success
    assert res.x[0] == pytest.approx(3.0, abs=1e-3)
    assert res.x[1] == pytest.approx(-1.0, abs=1e-3)
    assert res.fun == pytest.approx(7.0, abs=1e-6)
    assert res.n_eval > 0


def test_minimize_with_bounds_respects_box():
    """Bounded L-BFGS-B stays inside the box and hits the constrained min."""

    def objective(v):
        return (v[0] - 10.0) ** 2

    res = optimize.minimize(
        objective, x0=[0.0], method="L-BFGS-B", bounds=[(-5.0, 5.0)]
    )
    assert res.success
    assert res.x[0] == pytest.approx(5.0, abs=1e-3)
    assert res.fun == pytest.approx(25.0, abs=1e-4)


def test_global_minimize_finds_global_minimum():
    """Differential evolution (fixed seed) finds the global min of Rastrigin."""

    def rastrigin(v):
        v = np.asarray(v)
        return float(10.0 * v.size + np.sum(v**2 - 10.0 * np.cos(2.0 * np.pi * v)))

    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    res = optimize.global_minimize(rastrigin, bounds, seed=42)
    assert res.success
    assert res.fun < 0.01
    assert np.all(np.abs(res.x) < 0.1)
    # Reproducible with the same seed.
    res2 = optimize.global_minimize(rastrigin, bounds, seed=42)
    assert np.allclose(res.x, res2.x)
    assert res.fun == pytest.approx(res2.fun)


def test_global_minimize_rejects_unknown_method():
    with pytest.raises(ValueError):
        optimize.global_minimize(lambda v: 0.0, [(-1.0, 1.0)], method="basinhopping")


def test_least_squares_residual_converges(rng):
    """Least-squares on an exponential-decay residual recovers the model."""
    a_true, tau_true = 2.5, 0.7
    t = np.linspace(0.0, 5.0, 200)
    y = a_true * np.exp(-t / tau_true) + 0.01 * rng.normal(size=t.size)

    def residual(p):
        return p[0] * np.exp(-t / p[1]) - y

    res = optimize.least_squares(residual, x0=[1.0, 1.0])
    assert res.success
    assert res.x[0] == pytest.approx(a_true, rel=1e-2)
    assert res.x[1] == pytest.approx(tau_true, rel=2e-2)
    # fun is scipy's cost = 0.5 * sum(residual^2), consistent with noise.
    expected_cost = 0.5 * float(np.sum(residual(res.x) ** 2))
    assert res.fun == pytest.approx(expected_cost, rel=1e-9)
    assert res.fun < 0.02


def test_least_squares_with_bounds():
    """Bounds are honoured: the unconstrained minimiser lies outside."""

    def residual(p):
        # Unconstrained minimum of 0.5*(p-10)^2 is at p=10, outside the box.
        return np.array([p[0] - 10.0])

    # Start from an interior point (scipy's trf stalls when x0 sits exactly
    # on the bound opposite the descent direction).
    res = optimize.least_squares(residual, x0=[1.0], bounds=([0.0], [3.0]))
    assert res.success
    assert 0.0 <= res.x[0] <= 3.0
    assert res.x[0] == pytest.approx(3.0, abs=1e-4)
