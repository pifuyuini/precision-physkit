"""Quantitative tests for precision_physkit.filters."""

import numpy as np
import pytest
from scipy import signal

from precision_physkit import filters


def test_lowpass_stopband_attenuation():
    """A 6th-order Butterworth low-pass suppresses the stopband by > 40 dB."""
    fs = 1000.0
    t = np.arange(8192) / fs
    x = np.sin(2.0 * np.pi * 10.0 * t) + np.sin(2.0 * np.pi * 300.0 * t)
    y = filters.lowpass(x, fs, cutoff=50.0, order=6)
    assert y.shape == x.shape
    f, p_in = signal.welch(x, fs=fs, nperseg=2048)
    f, p_out = signal.welch(y, fs=fs, nperseg=2048)
    i300 = int(np.argmin(np.abs(f - 300.0)))
    attenuation_db = 10.0 * np.log10(p_out[i300] / p_in[i300])
    assert attenuation_db < -40.0
    # Passband tone passes essentially unattenuated.
    i10 = int(np.argmin(np.abs(f - 10.0)))
    pass_db = 10.0 * np.log10(p_out[i10] / p_in[i10])
    assert abs(pass_db) < 0.5


def test_highpass_and_bands():
    """High-pass, band-pass and band-stop separate tones as designed."""
    fs = 1000.0
    t = np.arange(8192) / fs
    lo = np.sin(2.0 * np.pi * 5.0 * t)
    hi = np.sin(2.0 * np.pi * 300.0 * t)
    x = lo + hi
    y_hp = filters.highpass(x, fs, cutoff=50.0, order=6)
    assert np.std(y_hp) == pytest.approx(np.std(hi), rel=0.05)

    y_bp = filters.bandpass(x, fs, cutoff=(250.0, 350.0), order=4)
    assert np.std(y_bp) == pytest.approx(np.std(hi), rel=0.05)

    y_bs = filters.bandstop(x, fs, cutoff=(250.0, 350.0), order=4)
    assert np.std(y_bs) == pytest.approx(np.std(lo), rel=0.05)


def test_notch_depth():
    """The IIR notch removes the 50 Hz line by > 40 dB, leaves 10 Hz alone."""
    fs = 1000.0
    t = np.arange(8192) / fs
    tone = np.sin(2.0 * np.pi * 50.0 * t)
    keep = np.sin(2.0 * np.pi * 10.0 * t)
    x = keep + tone
    y = filters.notch(x, fs, f0=50.0, q=30)
    f, p_in = signal.welch(x, fs=fs, nperseg=2048)
    f, p_out = signal.welch(y, fs=fs, nperseg=2048)
    i50 = int(np.argmin(np.abs(f - 50.0)))
    depth_db = 10.0 * np.log10(p_out[i50] / p_in[i50])
    assert depth_db < -40.0
    i10 = int(np.argmin(np.abs(f - 10.0)))
    assert abs(10.0 * np.log10(p_out[i10] / p_in[i10])) < 0.5


def test_filtfilt_zero_phase_keeps_peak_position():
    """Forward-backward filtering does not shift a sine's peak position."""
    fs = 1000.0
    t = np.arange(8192) / fs
    x = np.sin(2.0 * np.pi * 5.0 * t)
    y = filters.lowpass(x, fs, cutoff=50.0, order=4, zero_phase=True)
    # Peak index inside a clean interior period must be unchanged.
    seg = slice(3000, 3200)
    shift = int(np.argmax(y[seg]) - np.argmax(x[seg]))
    assert shift == 0
    # Zero-phase amplitude is the squared magnitude response: at 5 Hz with a
    # 50 Hz Butterworth the tone passes with essentially unit gain.
    assert np.std(y[1000:7000]) == pytest.approx(np.std(x[1000:7000]), rel=0.02)


def test_savgol_smooths_noise(rng):
    """Savitzky-Golay smoothing reduces the residual noise variance."""
    fs = 1000.0
    t = np.arange(4000) / fs
    clean = np.sin(2.0 * np.pi * 5.0 * t)
    x = clean + 0.3 * rng.normal(size=t.size)
    y = filters.savgol(x, window_length=31, polyorder=3)
    interior = slice(500, 3500)
    before = float(np.std(x[interior] - clean[interior]))
    after = float(np.std(y[interior] - clean[interior]))
    assert after < 0.5 * before


def test_savgol_derivative(rng):
    """The smoothed first derivative of a sine is a cosine (scaled by dt)."""
    fs = 1000.0
    t = np.arange(4000) / fs
    x = np.sin(2.0 * np.pi * 5.0 * t)
    dy = filters.savgol(x, window_length=31, polyorder=3, deriv=1) / (t[1] - t[0])
    interior = slice(500, 3500)
    expected = 2.0 * np.pi * 5.0 * np.cos(2.0 * np.pi * 5.0 * t)
    assert np.allclose(dy[interior], expected[interior], atol=1.0)


def test_moving_average_reduces_variance(rng):
    x = rng.normal(size=2000)
    y = filters.moving_average(x, 25)
    interior = slice(100, 1900)
    assert float(np.std(y[interior])) == pytest.approx(np.std(x) / 5.0, rel=0.2)


def test_filter_param_errors(rng):
    """Invalid parameters raise ValueError with a clear message."""
    x = rng.normal(size=512)
    with pytest.raises(ValueError):
        filters.lowpass(x, fs=-1.0, cutoff=10.0)
    with pytest.raises(ValueError):
        filters.lowpass(x, fs=1000.0, cutoff=600.0)  # above Nyquist
    with pytest.raises(ValueError):
        filters.lowpass(x, fs=1000.0, cutoff=0.0)
    with pytest.raises(ValueError):
        filters.lowpass(x, fs=1000.0, cutoff=10.0, order=0)
    with pytest.raises(ValueError):
        filters.bandpass(x, fs=1000.0, cutoff=(100.0, 50.0))  # low >= high
    with pytest.raises(ValueError):
        filters.bandstop(x, fs=1000.0, cutoff=(450.0, 550.0))  # high > Nyquist
    with pytest.raises(ValueError):
        filters.notch(x, fs=1000.0, f0=50.0, q=0.0)
    with pytest.raises(ValueError):
        filters.notch(x, fs=1000.0, f0=600.0)
    with pytest.raises(ValueError):
        filters.savgol(x, window_length=10, polyorder=2)  # even window
    with pytest.raises(ValueError):
        filters.savgol(x, window_length=11, polyorder=11)  # polyorder too large
    with pytest.raises(ValueError):
        filters.savgol(x, window_length=11, polyorder=3, deriv=4)  # deriv > polyorder
    with pytest.raises(ValueError):
        filters.moving_average(x, window=0)
