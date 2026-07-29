"""Quantitative tests for precision_physkit.spectral estimators."""

import numpy as np
import pytest
from scipy import signal

from precision_physkit import spectral


def test_welch_psd_white_noise_level(white_noise):
    """Welch PSD of white noise matches the theoretical level 2*sigma^2/fs."""
    d = white_noise
    sp = spectral.welch_psd(d["x"], d["fs"], nperseg=4096)
    expected = 2.0 * d["sigma"] ** 2 / d["fs"]  # one-sided white-noise level
    band = (sp.f > 50.0) & (sp.f < 450.0)
    level = float(np.median(sp.values[band]))
    assert level == pytest.approx(expected, rel=0.05)
    assert sp.kind == "psd"
    assert sp.scale == "linear"
    assert sp.method == "welch"
    assert sp.meta["n_segments"] > 10


def test_welch_and_lpsd_agree_on_white_noise(white_noise):
    """Welch and LPSD levels agree in magnitude on the same white record."""
    d = white_noise
    sp_w = spectral.welch_psd(d["x"], d["fs"], nperseg=4096)
    sp_l = spectral.lpsd(d["x"], d["fs"], Jdes=100, Kdes=50)
    band_w = (sp_w.f > 30.0) & (sp_w.f < 300.0)
    band_l = (sp_l.f > 30.0) & (sp_l.f < 300.0)
    med_w = float(np.median(sp_w.values[band_w]))
    med_l = float(np.median(sp_l.values[band_l]))
    assert med_l == pytest.approx(med_w, rel=0.10)
    assert sp_l.scale == "log"


def test_lcoherence_of_identical_signals_is_one(white_noise):
    """Magnitude-squared coherence of x with itself is 1 at every bin."""
    d = white_noise
    coh = spectral.lcoherence(d["x"], d["x"], d["fs"], Jdes=50, Kdes=30)
    assert coh.kind == "coherence"
    assert np.all(coh.values >= 1.0 - 1e-9)
    assert np.all(coh.values <= 1.0 + 1e-9)


def test_lcoherence_of_independent_noise_is_small(rng):
    """Independent white channels have near-zero coherence (sanity bound)."""
    fs = 1000.0
    x = rng.normal(size=32768)
    y = rng.normal(size=32768)
    coh = spectral.lcoherence(x, y, fs, Jdes=50, Kdes=30)
    assert float(np.median(coh.values)) < 0.3


def test_ltransfer_recovers_known_fir(white_noise):
    """H1 transfer estimate recovers a short FIR response in the passband (10%).

    The FIR is kept short (9 taps) because LPSD uses short segments at
    high frequencies; systems whose impulse response approaches the
    segment length are biased by construction.
    """
    d = white_noise
    fs = d["fs"]
    b = signal.firwin(9, 200.0 / (fs / 2.0))
    y_full = signal.lfilter(b, 1.0, d["x"])
    x, y = d["x"][500:], y_full[500:]  # drop the start-up transient
    tf = spectral.ltransfer(x, y, fs, Jdes=60, Kdes=100, with_coherence=True)
    w, h = signal.freqz(b, worN=8192, fs=fs)
    h_true = np.interp(tf.f, w, np.abs(h))
    band = (tf.f > 30.0) & (tf.f < 150.0)
    assert band.sum() >= 3
    rel_err = np.abs(np.abs(tf.values[band]) - h_true[band]) / h_true[band]
    assert float(rel_err.max()) < 0.10
    assert tf.kind == "transfer"
    assert np.asarray(tf.meta["coherence"])[band].min() > 0.7


def test_lcsd_self_real_part_equals_lpsd(white_noise):
    """For y == x the CSD is real and equals the auto-PSD."""
    d = white_noise
    csd = spectral.lcsd(d["x"], d["x"], d["fs"], Jdes=50, Kdes=30)
    psd = spectral.lpsd(d["x"], d["fs"], Jdes=50, Kdes=30)
    assert csd.f.shape == psd.f.shape
    assert np.allclose(csd.values.real, psd.values, rtol=1e-6)
    assert np.abs(csd.values.imag).max() < 1e-9 * psd.values.mean()


def test_multichannel_shapes(rng):
    """(N, C) input yields (J, C) values, matching per-channel estimates."""
    fs = 1000.0
    x = rng.normal(size=(16384, 3))
    sp_w = spectral.welch_psd(x, fs)
    assert sp_w.values.ndim == 2
    assert sp_w.values.shape[1] == 3
    col1 = spectral.welch_psd(x[:, 1], fs)
    assert np.allclose(sp_w.values[:, 1], col1.values)

    sp_l = spectral.lpsd(x, fs, Jdes=30, Kdes=20)
    assert sp_l.values.ndim == 2
    assert sp_l.values.shape[0] == sp_l.f.shape[0]
    assert sp_l.values.shape[1] == 3
    col0 = spectral.lpsd(x[:, 0], fs, Jdes=30, Kdes=20)
    assert np.allclose(sp_l.f, col0.f)
    assert np.allclose(sp_l.values[:, 0], col0.values)

    coh = spectral.lcoherence(x, x, fs, Jdes=30, Kdes=20)
    assert coh.values.shape == sp_l.values.shape
    assert np.all(coh.values >= 1.0 - 1e-9)


def test_welch_psd_tone_frequencies(multi_sine):
    """Welch PSD places the tone peaks at the injected frequencies."""
    d = multi_sine
    sp = spectral.welch_psd(d["x"], d["fs"], nperseg=2048)
    resolution = sp.f[1] - sp.f[0]
    for f_true in d["freqs"]:
        window = (sp.f > f_true - 3.0) & (sp.f < f_true + 3.0)
        f_peak = sp.f[np.argmax(sp.values[window]) + np.flatnonzero(window)[0]]
        assert f_peak == pytest.approx(f_true, abs=resolution)


def test_lcsd_matrix_structure(rng):
    """lcsd_matrix returns a Hermitian (J, C, C) matrix with PSD diagonal."""
    fs = 1000.0
    x = rng.normal(size=(16384, 2))
    m = spectral.lcsd_matrix(x, fs, Jdes=25, Kdes=20)
    j, c = m.values.shape[0], x.shape[1]
    assert m.values.shape == (j, c, c)
    assert m.meta["n_channels"] == c
    # Hermitian symmetry.
    assert np.allclose(m.values, np.conj(np.transpose(m.values, (0, 2, 1))))
    # Diagonal equals the per-channel LPSD.
    lp = spectral.lpsd(x, fs, Jdes=25, Kdes=20)
    for ch in range(c):
        assert np.allclose(m.values[:, ch, ch].real, lp.values[:, ch], rtol=1e-6)
    # Independent channels: small normalized cross spectrum.
    off = np.abs(m.values[:, 0, 1]) ** 2 / (
        m.values[:, 0, 0].real * m.values[:, 1, 1].real
    )
    assert float(np.median(off)) < 0.1


def test_spectrum_to_asd(white_noise):
    """to_asd takes the square root and tags provenance; rejects non-PSD."""
    d = white_noise
    sp = spectral.welch_psd(d["x"], d["fs"], nperseg=4096)
    asd = sp.to_asd()
    assert asd.kind == "asd"
    assert np.allclose(asd.values, np.sqrt(sp.values))
    assert np.allclose(asd.f, sp.f)
    assert asd.meta["derived_from"] == "psd"
    # Calling on an ASD again returns an equivalent container.
    asd2 = asd.to_asd()
    assert np.allclose(asd2.values, asd.values)
    # Not defined for coherence/transfer/csd.
    coh = spectral.welch_coherence(d["x"][:8192], d["x"][:8192], d["fs"])
    with pytest.raises(ValueError):
        coh.to_asd()


def test_spectrum_to_dataframe(rng):
    """to_dataframe: single channel -> kind column; (J, C) -> ch0..chC-1."""
    fs = 1000.0
    x = rng.normal(size=8192)
    sp = spectral.welch_psd(x, fs)
    df = sp.to_dataframe()
    assert list(df.columns) == ["psd"]
    assert df.index.name == "frequency"
    assert len(df) == sp.f.size
    assert np.allclose(df.index.to_numpy(), sp.f)

    xm = rng.normal(size=(8192, 3))
    spm = spectral.welch_psd(xm, fs)
    dfm = spm.to_dataframe()
    assert list(dfm.columns) == ["ch0", "ch1", "ch2"]
    assert np.allclose(dfm["ch2"].to_numpy(), spm.values[:, 2])

    # The (J, C, C) CSD matrix is not exportable.
    m = spectral.lcsd_matrix(xm[:, :2], fs, Jdes=20, Kdes=10)
    with pytest.raises(ValueError):
        m.to_dataframe()
