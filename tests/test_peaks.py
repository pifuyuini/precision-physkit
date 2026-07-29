"""Quantitative tests for precision_physkit.peaks detection, fitting and area."""

import numpy as np
import pytest

from precision_physkit import peaks

_GAUSS_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _two_gaussians(x):
    return 1.5 * np.exp(-0.5 * ((x - 3.0) / 0.2) ** 2) + 2.0 * np.exp(
        -0.5 * ((x - 7.0) / 0.4) ** 2
    )


def test_find_peaks_two_gaussians():
    """Two well-separated Gaussians: exact positions, prominences, FWHM."""
    x = np.linspace(0.0, 10.0, 2001)
    y = _two_gaussians(x)
    res = peaks.find_peaks(y, prominence=0.1, x_axis=x)
    assert len(res) == 2
    dx = float(x[1] - x[0])
    assert np.allclose(res.positions, [3.0, 7.0], atol=2.0 * dx)
    # Prominences recover the amplitudes (valley between peaks is ~0).
    assert np.allclose(res.prominences, [1.5, 2.0], rtol=0.05)
    # Heights include the tiny overlap of the other Gaussian.
    assert np.allclose(res.heights, [1.5, 2.0], rtol=0.02)
    # FWHM at half prominence matches the analytic Gaussian value.
    expected_fwhm = np.array([_GAUSS_FWHM * 0.2, _GAUSS_FWHM * 0.4])
    assert np.allclose(res.fwhms, expected_fwhm, rtol=0.05)
    # Peaks carry fractional-index base positions and a positive area.
    for p in res:
        assert p.left_base < p.index < p.right_base
        assert p.area > 0.0


def test_find_peaks_empty_result():
    x = np.zeros(100)
    res = peaks.find_peaks(x)
    assert len(res) == 0
    assert res.positions.size == 0


def test_fit_peaks_gaussian_recovers_parameters(rng):
    """Gaussian fit recovers center/amplitude/FWHM within 5%."""
    x = np.linspace(0.0, 10.0, 2001)
    y = _two_gaussians(x) + 0.05 * rng.normal(size=x.size)
    det = peaks.find_peaks(y, prominence=0.5, x_axis=x)
    assert len(det) == 2
    fit = peaks.fit_peaks(x, y, [p.index for p in det], model="gaussian")

    expected = [(3.0, 1.5, _GAUSS_FWHM * 0.2), (7.0, 2.0, _GAUSS_FWHM * 0.4)]
    assert len(fit.peaks) == 2
    for peak, (c_true, a_true, w_true) in zip(fit.peaks, expected):
        assert peak.center == pytest.approx(c_true, rel=0.05)
        assert peak.amplitude == pytest.approx(a_true, rel=0.05)
        assert peak.fwhm == pytest.approx(w_true, rel=0.05)
        for key, err in peak.errors.items():
            assert np.isfinite(err), key
            assert err >= 0.0, key
        # Analytic component area: amplitude * sigma * sqrt(2*pi).
        sigma = peak.params["sigma"]
        assert peak.area == pytest.approx(a_true * sigma * np.sqrt(2.0 * np.pi), rel=0.05)
    assert abs(fit.baseline) < 0.05
    assert fit.covariance.shape == (7, 7)  # baseline + 2 peaks * 3 params
    assert fit.y_fit.shape == x.shape


def test_fit_peaks_rejects_bad_input(rng):
    x = np.linspace(0.0, 1.0, 100)
    y = rng.normal(size=x.size)
    with pytest.raises(ValueError):
        peaks.fit_peaks(x, y, [], model="gaussian")
    with pytest.raises(ValueError):
        peaks.fit_peaks(x, y, [10], model="lorentzianX")
    with pytest.raises(ValueError):
        peaks.fit_peaks(x, y, [200], model="gaussian")  # index out of range
    with pytest.raises(ValueError):
        peaks.fit_peaks(x, y, [10], model="gaussian", window=-1.0)


def test_peak_area_matches_analytic():
    """Trapezoidal integral of an isolated Gaussian matches its area."""
    x = np.linspace(0.0, 10.0, 4001)
    amp, sigma = 1.5, 0.2
    y = amp * np.exp(-0.5 * ((x - 3.0) / sigma) ** 2)
    area = peaks.peak_area(x, y, 2.0, 4.0, baseline="edge")
    analytic = amp * sigma * np.sqrt(2.0 * np.pi)  # +/-5 sigma window
    assert area == pytest.approx(analytic, rel=0.02)


def test_peak_area_constant_baseline():
    """An explicit baseline is subtracted as a constant level."""
    x = np.linspace(0.0, 10.0, 4001)
    y = 0.3 + np.exp(-0.5 * ((x - 5.0) / 0.3) ** 2)
    raw = peaks.peak_area(x, y, 4.0, 6.0, baseline=None)
    sub = peaks.peak_area(x, y, 4.0, 6.0, baseline=0.3)
    assert sub == pytest.approx(raw - 0.3 * 2.0, rel=1e-12)


def test_peak_area_rejects_bad_input():
    x = np.linspace(0.0, 1.0, 100)
    y = np.sin(x)
    with pytest.raises(ValueError):
        peaks.peak_area(x, y, 0.8, 0.2)  # left >= right
    with pytest.raises(ValueError):
        peaks.peak_area(x[::-1], y, 0.2, 0.8)  # not increasing
