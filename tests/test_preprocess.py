"""Quantitative tests for precision_physkit.preprocess resampling and conditioning."""

import numpy as np
import pytest
from scipy import signal

from precision_physkit import preprocess


def test_downsample_length_and_antialias(rng):
    """q=4 decimation: exact length and strong suppression of aliased content.

    The 400 Hz component (above the new Nyquist of 125 Hz) must be
    attenuated by the anti-alias filter instead of folded back.
    """
    fs = 1000.0
    n = 32768
    t = np.arange(n) / fs
    x = np.sin(2.0 * np.pi * 10.0 * t) + np.sin(2.0 * np.pi * 400.0 * t)
    y = preprocess.downsample(x, 4)
    fs_new = fs / 4.0
    assert y.size == n // 4
    f, p = signal.welch(y, fs=fs_new, nperseg=2048)
    p_keep = p[np.argmin(np.abs(f - 10.0))]
    p_alias = p[np.argmin(np.abs(f - 100.0))]  # 400 Hz would alias to 100 Hz
    assert p_alias / p_keep < 1e-8


def test_downsample_rejects_bad_params(rng):
    x = rng.normal(size=256)
    with pytest.raises(ValueError):
        preprocess.downsample(x, 0)
    with pytest.raises(ValueError):
        preprocess.downsample(x, 2, ftype="butter")


def test_upsample_length_and_spectrum(rng):
    """Polyphase upsampling: exact length and preserved tone frequency."""
    fs = 250.0
    n = 4096
    t = np.arange(n) / fs
    x = np.sin(2.0 * np.pi * 20.0 * t)
    q = 4
    y = preprocess.upsample(x, q)
    assert y.size == n * q
    f_peak = np.fft.rfftfreq(y.size, d=1.0 / (fs * q))[
        np.argmax(np.abs(np.fft.rfft(y * np.hanning(y.size))))
    ]
    assert f_peak == pytest.approx(20.0, abs=0.5)
    with pytest.raises(ValueError):
        preprocess.upsample(x, 0)


def test_resample_preserves_sine_frequency(rng):
    """Rational-ratio resampling keeps a tone at its frequency."""
    fs_in, fs_out = 1000.0, 300.0
    n = 32768
    t = np.arange(n) / fs_in
    x = np.sin(2.0 * np.pi * 50.0 * t)
    y = preprocess.resample(x, fs_in, fs_out)
    assert abs(y.size - round(n * fs_out / fs_in)) <= 1
    freqs = np.fft.rfftfreq(y.size, d=1.0 / fs_out)
    peak = freqs[np.argmax(np.abs(np.fft.rfft(y * np.hanning(y.size))))]
    assert peak == pytest.approx(50.0, abs=0.5)
    with pytest.raises(ValueError):
        preprocess.resample(x, 0.0, fs_out)
    with pytest.raises(ValueError):
        preprocess.resample(x, fs_in, -1.0)


def test_interpolate_exact_at_nodes():
    """Every supported method reproduces the samples at the nodes."""
    t = np.linspace(0.0, 1.0, 11)
    x = np.sin(2.0 * np.pi * t)
    for method in ("linear", "cubic", "pchip", "akima"):
        y = preprocess.interpolate(t, x, t, method=method)
        assert np.allclose(y, x, atol=1e-12), method


def test_interpolate_between_nodes_and_out_of_range():
    """Interior values track the sine; out-of-range points become NaN."""
    t = np.linspace(0.0, 1.0, 11)
    x = np.sin(2.0 * np.pi * t)
    t_new = np.array([-0.5, 0.35, 0.65, 1.5])
    # Piecewise linear converges only as O(h^2); spline family is tighter.
    tolerances = {"linear": 6e-2, "cubic": 3e-2, "pchip": 3e-2, "akima": 3e-2}
    for method, tol in tolerances.items():
        y = preprocess.interpolate(t, x, t_new, method=method)
        assert np.isnan(y[0]) and np.isnan(y[3])
        assert y[1] == pytest.approx(np.sin(2.0 * np.pi * 0.35), abs=tol)
        assert y[2] == pytest.approx(np.sin(2.0 * np.pi * 0.65), abs=tol)


def test_interpolate_rejects_bad_input():
    t = np.array([0.0, 2.0, 1.0])
    x = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):  # not strictly increasing
        preprocess.interpolate(t, x, [0.5])
    with pytest.raises(ValueError):  # length mismatch
        preprocess.interpolate(np.arange(3.0), np.arange(4.0), [0.5])
    with pytest.raises(ValueError):  # unknown method
        preprocess.interpolate(np.arange(3.0), np.arange(3.0), [0.5], method="nearest")


def test_fill_gaps_no_nan_and_continuous(rng):
    """Filled gaps contain no NaN and no abnormal jumps at the boundaries."""
    fs = 1000.0
    t = np.arange(4000) / fs
    clean = np.sin(2.0 * np.pi * 5.0 * t) + 0.1 * rng.normal(size=t.size)
    x = clean.copy()
    x[1000:1050] = np.nan
    x[2000:2010] = np.nan
    y = preprocess.fill_gaps(x, method="pchip")
    assert not np.isnan(y).any()
    assert y.shape == x.shape
    # Largest step anywhere must be comparable to the signal's own steps.
    max_step_filled = float(np.abs(np.diff(y)).max())
    max_step_clean = float(np.abs(np.diff(clean)).max())
    assert max_step_filled < 5.0 * max_step_clean


def test_fill_gaps_noise_reproducible_and_edge_clamp(rng):
    """fill_noise is reproducible under a fixed seed; edges clamp to nearest."""
    fs = 1000.0
    t = np.arange(2000) / fs
    base = np.sin(2.0 * np.pi * 5.0 * t) + 0.1 * rng.normal(size=t.size)
    x = base.copy()
    x[500:520] = np.nan
    y1 = preprocess.fill_gaps(x, fill_noise=True, seed=7)
    y2 = preprocess.fill_gaps(x, fill_noise=True, seed=7)
    assert np.array_equal(y1, y2)
    assert not np.isnan(y1).any()

    x_edge = base.copy()
    x_edge[:5] = np.nan
    x_edge[-5:] = np.nan
    y_edge = preprocess.fill_gaps(x_edge)
    assert not np.isnan(y_edge).any()
    assert y_edge[0] == pytest.approx(base[5])
    assert y_edge[-1] == pytest.approx(base[-6])

    with pytest.raises(ValueError):
        preprocess.fill_gaps(np.array([np.nan, 1.0]))  # <2 valid samples
    with pytest.raises(ValueError):
        preprocess.fill_gaps(base.reshape(2, -1))  # not 1-D


def _psd_dynamic_range_db(x, fs):
    f, p = signal.welch(x, fs=fs, nperseg=4096)
    band = (f > 20.0) & (f < 480.0)
    return 10.0 * np.log10(p[band].max() / p[band].min())


def test_whiten_flattens_colored_noise(rng):
    """Whitening strongly reduces the spectral dynamic range of colored noise."""
    fs = 1000.0
    n = 32768
    white = rng.normal(size=n)
    b, a = signal.butter(4, 50.0 / (fs / 2.0))
    colored = signal.lfilter(b, a, white)
    y = preprocess.whiten(colored, fs)
    assert y.shape == colored.shape
    before = _psd_dynamic_range_db(colored, fs)
    after = _psd_dynamic_range_db(y, fs)
    assert after < 0.55 * before


def test_whiten_rejects_bad_input(rng):
    x = rng.normal(size=512)
    with pytest.raises(ValueError):
        preprocess.whiten(x, fs=0.0)
    with pytest.raises(ValueError):
        preprocess.whiten(x, fs=1000.0, method="ar")
    with pytest.raises(ValueError):
        preprocess.whiten(x.reshape(2, -1), fs=1000.0)
    with pytest.raises(ValueError):
        preprocess.whiten(np.ones(512), fs=1000.0)  # zero power
