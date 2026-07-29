"""Error-path tests: invalid inputs must raise ValueError (or re-raise).

Covers the Rust kernel ``precision_physkit._core`` argument validation (reached
through the public spectral API) and shape-consistency checks in the
spectral and fitting wrappers.
"""

import numpy as np
import pytest

from precision_physkit import fitting, spectral


# ---------------------------------------------------------------------------
# Rust kernel validation via precision_physkit.spectral
# ---------------------------------------------------------------------------

def test_core_rejects_nan_and_inf(rng):
    fs = 1000.0
    x = rng.normal(size=8192)
    x_nan = x.copy()
    x_nan[100] = np.nan
    x_inf = x.copy()
    x_inf[100] = np.inf
    for bad in (x_nan, x_inf):
        with pytest.raises(ValueError):
            spectral.lpsd(bad, fs, Jdes=20, Kdes=10)
        with pytest.raises(ValueError):
            spectral.lcsd(bad, x, fs, Jdes=20, Kdes=10)
        with pytest.raises(ValueError):
            spectral.lcoherence(x, bad, fs, Jdes=20, Kdes=10)
        with pytest.raises(ValueError):
            spectral.ltransfer(bad, x, fs, Jdes=20, Kdes=10)
        with pytest.raises(ValueError):
            spectral.lcsd_matrix(np.column_stack([bad, x]), fs, Jdes=20, Kdes=10)


def test_core_rejects_bad_fs(rng):
    x = rng.normal(size=1024)
    for bad_fs in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            spectral.welch_psd(x, bad_fs)
        with pytest.raises(ValueError):
            spectral.lpsd(x, bad_fs, Jdes=10, Kdes=5)


def test_core_rejects_bad_jdes_kdes(rng):
    fs = 1000.0
    x = rng.normal(size=8192)
    with pytest.raises(ValueError):
        spectral.lpsd(x, fs, Jdes=1, Kdes=10)  # Jdes < 2
    with pytest.raises(ValueError):
        spectral.lpsd(x, fs, Jdes=10, Kdes=1)  # Kdes < 2


def test_core_rejects_bad_xi(rng):
    fs = 1000.0
    x = rng.normal(size=8192)
    for bad_xi in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            spectral.lpsd(x, fs, Jdes=20, Kdes=10, xi=bad_xi)


# ---------------------------------------------------------------------------
# Shape-consistency checks in the spectral wrappers
# ---------------------------------------------------------------------------

def test_spectral_rejects_shape_mismatch(rng):
    fs = 1000.0
    x = rng.normal(size=4096)
    y = rng.normal(size=2048)
    with pytest.raises(ValueError):
        spectral.welch_csd(x, y, fs)
    with pytest.raises(ValueError):
        spectral.welch_coherence(x, y, fs)
    with pytest.raises(ValueError):
        spectral.welch_transfer(x, y, fs)
    with pytest.raises(ValueError):
        spectral.lcsd(x, y, fs, Jdes=20, Kdes=10)
    with pytest.raises(ValueError):
        spectral.lcoherence(x, y, fs, Jdes=20, Kdes=10)
    with pytest.raises(ValueError):
        spectral.ltransfer(x, y, fs, Jdes=20, Kdes=10)


def test_spectral_rejects_bad_dimensionality(rng):
    fs = 1000.0
    x3d = rng.normal(size=(64, 2, 2))
    with pytest.raises(ValueError):
        spectral.welch_psd(x3d, fs)
    with pytest.raises(ValueError):
        spectral.lpsd(x3d, fs, Jdes=10, Kdes=5)
    with pytest.raises(ValueError):
        spectral.lcsd_matrix(rng.normal(size=1024), fs, Jdes=10, Kdes=5)  # 1-D
    with pytest.raises(ValueError):
        spectral.welch_psd(np.array([]), fs)  # empty


# ---------------------------------------------------------------------------
# Shape-consistency checks in the fitting wrappers
# ---------------------------------------------------------------------------

def test_linear_lstsq_rejects_shape_mismatch(rng):
    design = rng.normal(size=(100, 3))
    with pytest.raises(ValueError):
        fitting.linear_lstsq(design, rng.normal(size=50))  # y too short
    with pytest.raises(ValueError):
        fitting.linear_lstsq(rng.normal(size=100), rng.normal(size=100))  # 1-D design
    with pytest.raises(ValueError):
        fitting.linear_lstsq(design, rng.normal(size=(100, 1)))  # y not 1-D


def test_iterative_multichannel_rejects_bad_groups(rng):
    design = rng.normal(size=(100, 2))
    y = rng.normal(size=100)
    with pytest.raises(ValueError):
        fitting.iterative_multichannel(design, y, (60, 30))  # sums to 90 != 100
    with pytest.raises(ValueError):
        fitting.iterative_multichannel(design, y, (100, 0))  # zero-size group
    with pytest.raises(ValueError):
        fitting.iterative_multichannel(design, y, (-50, 150))  # negative group
    with pytest.raises(ValueError):
        fitting.iterative_multichannel(design, rng.normal(size=(100, 1)), (100,))


def test_polyfit_rejects_bad_input(rng):
    x = np.linspace(0.0, 1.0, 50)
    y = rng.normal(size=50)
    with pytest.raises(ValueError):
        fitting.polyfit(x, y, deg=-1)
    with pytest.raises(ValueError):
        fitting.polyfit(x, rng.normal(size=40), deg=1)  # length mismatch
    with pytest.raises(ValueError):
        fitting.polyfit(x.reshape(5, 10), y, deg=1)  # x not 1-D


def test_curve_fit_failure_reporting(rng):
    """A non-converging fit reports success=False; raise_on_failure re-raises."""

    def model5(x, a, b, c, d, e):
        return a * x**4 + b * x**3 + c * x**2 + d * x + e

    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 0.0])
    res = fitting.curve_fit(model5, x, y)  # 5 params > 3 data points
    assert not res.success
    assert res.message  # carries the original exception
    assert all(np.isnan(v) for v in res.params.values())
    assert res.cov.size == 0
    assert res.resid.size == 0
    with pytest.raises(TypeError):
        fitting.curve_fit(model5, x, y, raise_on_failure=True)
