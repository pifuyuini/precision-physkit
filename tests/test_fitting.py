"""Quantitative tests for precision_physkit.fitting parameter estimation."""

import numpy as np
import pytest

from precision_physkit import fitting


def _gauss(x, amp, mu, sig):
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def test_curve_fit_gaussian_recovers_params(rng):
    """Gaussian parameters are recovered and 1-sigma errors are finite."""
    amp_true, mu_true, sig_true = 2.5, 1.3, 0.4
    x = np.linspace(-2.0, 4.0, 400)
    y = _gauss(x, amp_true, mu_true, sig_true) + 0.01 * rng.normal(size=x.size)
    res = fitting.curve_fit(_gauss, x, y, p0=[1.0, 0.0, 1.0])
    assert res.success
    assert res.params["amp"] == pytest.approx(amp_true, rel=1e-2)
    assert res.params["mu"] == pytest.approx(mu_true, abs=1e-2)
    assert res.params["sig"] == pytest.approx(sig_true, rel=2e-2)
    # All 1-sigma uncertainties are finite and positive.
    for name in ("amp", "mu", "sig"):
        assert np.isfinite(res.perr[name])
        assert res.perr[name] > 0.0
    assert res.cov.shape == (3, 3)
    assert np.all(np.diag(res.cov) > 0.0)
    assert res.resid.shape == x.shape
    assert res.stats["r2"] > 0.99
    assert res.stats["dof"] == x.size - 3


def test_curve_fit_explicit_param_names(rng):
    """Explicit param_names override signature inference."""

    def model(x, a, b):
        return a * x + b

    x = np.linspace(0.0, 10.0, 100)
    y = 2.0 * x - 1.0 + 0.01 * rng.normal(size=x.size)
    res = fitting.curve_fit(model, x, y, p0=[0.0, 0.0], param_names=["slope", "offset"])
    assert set(res.params) == {"slope", "offset"}
    assert res.params["slope"] == pytest.approx(2.0, rel=1e-3)


def test_linear_lstsq_recovers_coefficients(rng):
    """Truncated-SVD least squares recovers known theta; cov diag positive."""
    n, p = 400, 3
    design = rng.normal(size=(n, p))
    theta_true = np.array([1.5, -2.0, 0.7])
    y = design @ theta_true + 0.01 * rng.normal(size=n)
    res = fitting.linear_lstsq(design, y, param_names=["a", "b", "c"])
    assert res.success
    theta = np.array([res.params[k] for k in ("a", "b", "c")])
    assert np.allclose(theta, theta_true, atol=0.01)
    assert res.stats["rank"] == p
    assert res.cov.shape == (p, p)
    assert np.all(np.diag(res.cov) > 0.0)
    # Residual norm consistent with the injected noise level.
    assert float(np.std(res.resid)) == pytest.approx(0.01, rel=0.3)


def test_linear_lstsq_rcond_truncates_rank(rng):
    """A tiny singular value survives the default rcond but not rcond=1e-6."""
    n = 200
    u, _ = np.linalg.qr(rng.normal(size=(n, 3)))
    v, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    design = u @ np.diag([1.0, 0.5, 1e-9]) @ v.T
    y = rng.normal(size=n)
    res_default = fitting.linear_lstsq(design, y)
    res_trunc = fitting.linear_lstsq(design, y, rcond=1e-6)
    assert res_default.stats["rank"] == 3
    assert res_trunc.stats["rank"] == 2
    assert "rank-deficient" in res_trunc.message
    assert res_trunc.success  # truncated solution is still reported


def test_iterative_multichannel_heteroscedastic(rng):
    """Two channels with 1:10 noise ratio: theta recovered, weights ~100:1."""
    n1 = n2 = 400
    t1 = np.linspace(0.0, 1.0, n1)
    t2 = np.linspace(0.0, 1.0, n2)
    design = np.vstack(
        [
            np.column_stack([np.ones(n1), t1]),
            np.column_stack([np.ones(n2), t2]),
        ]
    )
    theta_true = np.array([2.0, 3.0])
    s1, s2 = 0.05, 0.5
    y = np.concatenate(
        [
            design[:n1] @ theta_true + s1 * rng.normal(size=n1),
            design[n1:] @ theta_true + s2 * rng.normal(size=n2),
        ]
    )
    res = fitting.iterative_multichannel(
        design, y, (n1, n2), param_names=["offset", "slope"]
    )
    assert res.success  # iteration converged
    assert res.params["offset"] == pytest.approx(theta_true[0], abs=0.02)
    assert res.params["slope"] == pytest.approx(theta_true[1], abs=0.05)
    weights = np.asarray(res.stats["weights"])
    assert weights.shape == (2,)
    # Weights are inverse variances: ratio must reflect (s2/s1)^2 = 100.
    ratio = weights[0] / weights[1]
    assert 40.0 < ratio < 250.0
    assert res.stats["n_iter"] >= 2
    assert np.all(np.diag(res.cov) > 0.0)


def test_polyfit_recovers_line(rng):
    """polyfit recovers a known straight line; c0 is the constant term."""
    c0_true, c1_true = 1.5, -2.5
    x = np.linspace(-3.0, 3.0, 200)
    y = c0_true + c1_true * x + 0.01 * rng.normal(size=x.size)
    res = fitting.polyfit(x, y, deg=1)
    assert res.success
    assert res.params["c0"] == pytest.approx(c0_true, abs=0.01)
    assert res.params["c1"] == pytest.approx(c1_true, abs=0.01)
    assert res.stats["rank"] == 2


def test_polyfit_quadratic(rng):
    """Quadratic fit on noiseless data is essentially exact."""
    x = np.linspace(-2.0, 2.0, 101)
    y = 0.5 - 1.0 * x + 2.0 * x**2
    res = fitting.polyfit(x, y, deg=2)
    assert res.success
    assert res.params["c0"] == pytest.approx(0.5, abs=1e-6)
    assert res.params["c1"] == pytest.approx(-1.0, abs=1e-6)
    assert res.params["c2"] == pytest.approx(2.0, abs=1e-6)
