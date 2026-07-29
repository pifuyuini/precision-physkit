"""Preprocessing utilities for uniformly sampled time series.

This module covers the first steps of a typical analysis pipeline:
resampling (integer-factor decimation and interpolation, rational-ratio
resampling), regridding by interpolation, filling of missing samples
(gaps marked as NaN), and frequency-domain whitening.

All functions return new arrays and never modify their inputs in place.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Literal

import numpy as np
import scipy.interpolate
import scipy.signal
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "downsample",
    "fill_gaps",
    "interpolate",
    "resample",
    "upsample",
    "whiten",
]

InterpMethod = Literal["linear", "cubic", "pchip", "akima"]
"""Interpolation methods supported by `interpolate` and `fill_gaps`."""

_SCIPY_INTERPOLATORS = {
    "cubic": scipy.interpolate.CubicSpline,
    "pchip": scipy.interpolate.PchipInterpolator,
    "akima": scipy.interpolate.Akima1DInterpolator,
}

_INTERP_METHODS = ("linear", "cubic", "pchip", "akima")


def downsample(
    x: ArrayLike,
    q: int,
    axis: int = -1,
    zero_phase: bool = True,
    ftype: str = "iir",
) -> NDArray[np.floating]:
    """Downsample a signal by an integer factor with anti-alias filtering.

    Wraps `scipy.signal.decimate`: the signal is first low-pass filtered to
    suppress components above the new Nyquist frequency (anti-aliasing) and
    then every ``q``-th sample is kept.

    Parameters
    ----------
    x : array_like
        Input signal. May be n-dimensional; decimation is applied along
        ``axis``.
    q : int
        Integer downsampling factor. Must be >= 1.
    axis : int, optional
        Axis along which to decimate. Default is -1.
    zero_phase : bool, optional
        If True (default), the anti-alias filter is applied forward and
        backward, producing zero phase distortion at the cost of an
        effectively doubled filter order. If False, filtering is causal
        (single pass), which introduces group delay.
    ftype : {"iir", "fir"}, optional
        Anti-alias filter family. ``"iir"`` (default) uses an 8th-order
        Chebyshev type I low-pass filter: steep transition band, small
        computational cost, nonlinear phase when ``zero_phase=False``.
        ``"fir"`` uses a 30-tap Hamming-windowed FIR filter: linear phase
        but a gentler transition band.

    Returns
    -------
    numpy.ndarray
        Downsampled signal whose length along ``axis`` is approximately
        ``x.shape[axis] / q`` (exact length follows
        `scipy.signal.decimate`).

    Raises
    ------
    ValueError
        If ``q`` is smaller than 1 or ``ftype`` is not ``"iir"``/``"fir"``.

    Notes
    -----
    Because of the anti-alias filter, decimation is the correct way to
    reduce the sampling rate of noisy or broadband data; plain slicing
    (``x[::q]``) aliases energy above the new Nyquist frequency into the
    retained band.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sin(2 * np.pi * 0.05 * np.arange(400.0))
    >>> downsample(x, 4).shape
    (100,)
    """
    if q < 1:
        raise ValueError(f"q must be >= 1, got {q}.")
    return scipy.signal.decimate(x, q, axis=axis, ftype=ftype, zero_phase=zero_phase)


def upsample(x: ArrayLike, q: int, axis: int = -1) -> NDArray[np.floating]:
    """Upsample a signal by an integer factor via polyphase interpolation.

    Wraps `scipy.signal.resample_poly` with ``up=q, down=1``: zeros are
    inserted between samples and the result is low-pass filtered with a
    Kaiser-windowed FIR anti-imaging filter.

    Parameters
    ----------
    x : array_like
        Input signal. May be n-dimensional; resampling is applied along
        ``axis``.
    q : int
        Integer upsampling factor. Must be >= 1.
    axis : int, optional
        Axis along which to upsample. Default is -1.

    Returns
    -------
    numpy.ndarray
        Upsampled signal with approximately ``x.shape[axis] * q`` samples
        along ``axis``.

    Raises
    ------
    ValueError
        If ``q`` is smaller than 1.

    Notes
    -----
    Polyphase upsampling preserves the spectral shape of the original
    signal below the original Nyquist frequency; the inserted samples are
    band-limited interpolants, not repetitions.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sin(2 * np.pi * 0.05 * np.arange(100.0))
    >>> upsample(x, 4).shape
    (400,)
    """
    if q < 1:
        raise ValueError(f"q must be >= 1, got {q}.")
    return scipy.signal.resample_poly(x, q, 1, axis=axis)


def resample(
    x: ArrayLike,
    fs_in: float,
    fs_out: float,
    axis: int = -1,
    max_denominator: int = 100_000,
) -> NDArray[np.floating]:
    """Resample a signal from ``fs_in`` to ``fs_out`` by a rational ratio.

    The frequency ratio ``fs_out / fs_in`` is approximated by a rational
    number ``up / down`` using `fractions.Fraction.limit_denominator`, and
    the signal is resampled with the polyphase method
    (`scipy.signal.resample_poly`), which applies a Kaiser-windowed
    anti-aliasing/anti-imaging FIR filter.

    Parameters
    ----------
    x : array_like
        Input signal sampled at ``fs_in``.
    fs_in : float
        Input sampling frequency in Hz. Must be positive.
    fs_out : float
        Target sampling frequency in Hz. Must be positive.
    axis : int, optional
        Axis along which to resample. Default is -1.
    max_denominator : int, optional
        Largest denominator allowed when approximating ``fs_out / fs_in``
        by a rational number. Default is 100000, which gives rate errors
        far below one ppm for typical frequencies.

    Returns
    -------
    numpy.ndarray
        Resampled signal whose effective sampling rate is
        ``fs_in * up / down``, which equals ``fs_out`` up to the rational
        approximation error.

    Raises
    ------
    ValueError
        If ``fs_in`` or ``fs_out`` is not positive, or if
        ``max_denominator`` is smaller than 1.

    Notes
    -----
    This polyphase (filtering) method is the right choice for arbitrary
    rate changes of general, noisy, or non-periodic signals: it makes no
    periodicity assumption and controls aliasing through a real filter.

    The Fourier method (`scipy.signal.resample`) instead truncates/zero-pads
    the FFT and assumes the signal is periodic across its endpoints. It is
    fast and spectrally exact for truly periodic data, but produces edge
    ringing (Gibbs-like leakage) for non-periodic signals and is
    parameterized by output *sample count* rather than by sampling rate.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(0, 1, 1 / 44100)
    >>> x = np.sin(2 * np.pi * 440 * t)
    >>> y = resample(x, 44100, 48000)
    >>> y.shape[0] == round(x.shape[0] * 48000 / 44100)
    True
    """
    if fs_in <= 0:
        raise ValueError(f"fs_in must be positive, got {fs_in}.")
    if fs_out <= 0:
        raise ValueError(f"fs_out must be positive, got {fs_out}.")
    if max_denominator < 1:
        raise ValueError(f"max_denominator must be >= 1, got {max_denominator}.")
    ratio = Fraction(float(fs_out) / float(fs_in)).limit_denominator(max_denominator)
    return scipy.signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=axis)


def interpolate(
    t: ArrayLike,
    x: ArrayLike,
    t_new: ArrayLike,
    method: InterpMethod = "linear",
) -> NDArray[np.floating]:
    """Interpolate a 1-D signal sampled at ``t`` onto a new grid ``t_new``.

    Parameters
    ----------
    t : array_like, shape (n,)
        Sampling coordinates of the input signal. Must be one-dimensional
        and strictly increasing.
    x : array_like, shape (n,)
        Signal values at ``t``.
    t_new : array_like
        New sampling coordinates. May have any shape; the result has the
        same shape.
    method : {"linear", "cubic", "pchip", "akima"}, optional
        Interpolation method:

        - ``"linear"`` (default): piecewise linear (`numpy.interp`).
        - ``"cubic"``: natural cubic spline
          (`scipy.interpolate.CubicSpline`); smoothest, but may overshoot
          between samples.
        - ``"pchip"``: monotone cubic Hermite interpolation
          (`scipy.interpolate.PchipInterpolator`); never overshoots,
          preserves monotonicity — usually the safest default for
          measured data.
        - ``"akima"``: Akima interpolation
          (`scipy.interpolate.Akima1DInterpolator`); robust against
          outliers, less smooth than a spline.

    Returns
    -------
    numpy.ndarray
        Interpolated values at ``t_new``, with the same shape as
        ``t_new``. Points of ``t_new`` outside the range of ``t`` are
        filled with NaN (no extrapolation is performed).

    Raises
    ------
    ValueError
        If ``t`` and ``x`` are not 1-D arrays of equal length (>= 2), if
        ``t`` is not strictly increasing, or if ``method`` is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 11)
    >>> x = np.sin(2 * np.pi * t)
    >>> interpolate(t, x, [0.25, 0.5], method="pchip").shape
    (2,)
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    t_new = np.asarray(t_new, dtype=float)
    if t.ndim != 1 or x.ndim != 1 or t.size != x.size:
        raise ValueError("t and x must be 1-D arrays of equal length.")
    if t.size < 2:
        raise ValueError("At least two samples are required.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing.")
    if method not in _INTERP_METHODS:
        raise ValueError(f"method must be one of {_INTERP_METHODS}, got {method!r}.")

    if method == "linear":
        y_new = np.interp(t_new, t, x)
    else:
        interpolator = _SCIPY_INTERPOLATORS[method](t, x)
        y_new = np.asarray(interpolator(t_new), dtype=float)
    y_new = np.array(y_new, dtype=float, copy=True)
    y_new[(t_new < t[0]) | (t_new > t[-1])] = np.nan
    return y_new


def _nan_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return the ``[start, stop)`` runs of True values in a boolean mask."""
    d = np.diff(mask.view(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _estimate_noise_std(samples: NDArray[np.floating]) -> float:
    """Estimate the white-noise standard deviation of a valid sample block.

    Uses ``std(diff(samples)) / sqrt(2)``, which isolates the
    high-frequency (noise) component and is insensitive to slow signal
    trends. Falls back to the plain standard deviation for very short
    blocks and to 0.0 for degenerate input.
    """
    if samples.size >= 3:
        return float(np.std(np.diff(samples), ddof=1) / np.sqrt(2.0))
    if samples.size >= 2:
        return float(np.std(samples, ddof=1))
    return 0.0


def fill_gaps(
    x: ArrayLike,
    method: InterpMethod = "pchip",
    fill_noise: bool = False,
    noise_std: float | None = None,
    seed: int | None = None,
) -> NDArray[np.floating]:
    """Fill NaN gaps in a 1-D signal by interpolation.

    Gaps (NaN runs) inside the valid range are filled by the chosen
    interpolation method. Leading and trailing NaN runs cannot be
    interpolated and are filled with the nearest valid sample (constant
    extension) rather than extrapolated.

    Parameters
    ----------
    x : array_like, shape (n,)
        Input signal in which NaN values mark missing samples.
    method : {"linear", "cubic", "pchip", "akima"}, optional
        Interpolation method used for interior gaps (see `interpolate`).
        Default is ``"pchip"``, which does not overshoot.
    fill_noise : bool, optional
        If True, Gaussian random perturbations are added on top of the
        interpolated values. A purely interpolated gap is artificially
        smooth (deterministic, near-zero high-frequency variance), which
        shows up in spectral analysis as an artificial low-variance,
        low-power segment; adding noise with a locally matched standard
        deviation keeps the filled signal approximately
        variance-stationary. Default is False.
    noise_std : float, optional
        Standard deviation of the added noise. If None (default), it is
        estimated per gap from the valid samples in a neighborhood of
        +/- 50 samples around the gap via
        ``std(diff) / sqrt(2)``. Must be >= 0 when given.
    seed : int, optional
        Seed for the random number generator used when
        ``fill_noise=True``, for reproducible results. Passed to
        `numpy.random.default_rng`.

    Returns
    -------
    numpy.ndarray
        A copy of ``x`` with all NaN values replaced.

    Raises
    ------
    ValueError
        If ``x`` is not 1-D, contains fewer than two valid (non-NaN)
        samples, if ``method`` is unknown, or if ``noise_std`` is
        negative.

    Notes
    -----
    Filled values are synthetic data. Downstream statistics that assume
    independent samples (e.g. confidence intervals, white-noise tests)
    should treat filled segments with care, even with ``fill_noise=True``.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sin(2 * np.pi * np.arange(64.0) / 16)
    >>> x[20:24] = np.nan
    >>> y = fill_gaps(x, fill_noise=True, seed=0)
    >>> bool(np.isnan(y).any())
    False
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array.")
    if method not in _INTERP_METHODS:
        raise ValueError(f"method must be one of {_INTERP_METHODS}, got {method!r}.")
    if noise_std is not None and noise_std < 0:
        raise ValueError(f"noise_std must be >= 0, got {noise_std}.")

    valid = ~np.isnan(x)
    if valid.all():
        return x.copy()
    if valid.sum() < 2:
        raise ValueError("At least two valid (non-NaN) samples are required.")

    grid = np.arange(x.size, dtype=float)
    gap_idx = grid[~valid]
    filled = x.copy()

    if method == "linear":
        filled[~valid] = np.interp(gap_idx, grid[valid], x[valid])
    else:
        interpolator = _SCIPY_INTERPOLATORS[method](grid[valid], x[valid])
        values = np.asarray(interpolator(gap_idx), dtype=float)
        # Clamp leading/trailing gaps to the nearest valid sample instead
        # of extrapolating.
        values[gap_idx < grid[valid][0]] = x[valid][0]
        values[gap_idx > grid[valid][-1]] = x[valid][-1]
        filled[~valid] = values

    if fill_noise:
        rng = np.random.default_rng(seed)
        global_std = _estimate_noise_std(x[valid])
        for start, stop in _nan_runs(~valid):
            lo = max(0, start - 50)
            hi = min(x.size, stop + 50)
            neighborhood = np.concatenate([x[lo:start], x[stop:hi]])
            neighborhood = neighborhood[~np.isnan(neighborhood)]
            if noise_std is not None:
                std = float(noise_std)
            elif neighborhood.size >= 3:
                std = _estimate_noise_std(neighborhood)
            else:
                std = global_std
            filled[start:stop] += rng.normal(0.0, std, size=stop - start)

    return filled


def whiten(
    x: ArrayLike,
    fs: float,
    method: str = "psd",
    nperseg: int | None = None,
) -> NDArray[np.floating]:
    """Whiten a signal in the frequency domain (spectral pre-whitening).

    The power spectral density (PSD) of the signal is estimated with
    Welch's method (a smoothed, averaged periodogram), the FFT of the
    signal is divided by the square root of that PSD, and the inverse FFT
    yields a time series whose spectrum is approximately flat (white).

    Parameters
    ----------
    x : array_like, shape (n,)
        Real-valued input signal.
    fs : float
        Sampling frequency in Hz. Must be positive.
    method : {"psd"}, optional
        Whitening method. Only ``"psd"`` (Welch-smoothed PSD) is
        currently implemented; the parameter reserves the interface for
        alternative estimators.
    nperseg : int, optional
        Segment length passed to `scipy.signal.welch`. Smaller values
        give a smoother (more strongly averaged) PSD estimate at lower
        frequency resolution; larger values resolve narrow spectral
        features at the cost of estimate variance. Default None lets
        `scipy.signal.welch` choose (256 samples).

    Returns
    -------
    numpy.ndarray
        Whitened signal of the same length as ``x``. Its overall
        amplitude scale is arbitrary (the normalization of the FFT and of
        Welch's PSD differ); what matters is that its spectrum is
        approximately flat.

    Raises
    ------
    ValueError
        If ``x`` is not 1-D, has fewer than two samples, if ``fs`` is not
        positive, if ``method`` is unknown, or if the signal has zero
        power (e.g. all samples equal) so that no PSD is defined.

    Notes
    -----
    Whitening is a standard pre-processing step for peak detection and
    parametric modeling: after flattening the background, spectral peaks
    (lines, resonances) stand out against a uniform noise floor, and
    least-squares fits in the time domain are no longer dominated by the
    strongest low-frequency components.

    Boundary effects: the FFT treats the signal as periodic, so
    discontinuities between the last and first sample leak across the
    spectrum and the first/last few samples of the output are less
    reliably whitened (wrap-around artifacts). Detrending or windowing
    the input before whitening reduces this effect. Strong narrow
    spectral lines are only partially whitened because Welch smoothing
    broadens them in the PSD estimate; reducing ``nperseg`` is *not* the
    remedy — increase it if narrow lines must be flattened more
    aggressively, accepting a noisier estimate.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(4096) / 1000.0
    >>> x = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.default_rng(0).normal(size=t.size)
    >>> y = whiten(x, fs=1000.0)
    >>> y.shape == x.shape
    True
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array.")
    if x.size < 2:
        raise ValueError("At least two samples are required.")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    if method != "psd":
        raise ValueError(f"method must be 'psd', got {method!r}.")

    freqs_welch, psd = scipy.signal.welch(x, fs=fs, nperseg=nperseg)
    if not np.any(psd > 0):
        raise ValueError("Signal has zero power; whitening is undefined.")

    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    psd_grid = np.interp(freqs, freqs_welch, psd)
    # Floor the PSD to avoid division by (near-)zero between line peaks.
    psd_grid = np.maximum(psd_grid, psd_grid.max() * 1e-12)

    spectrum = np.fft.rfft(x)
    whitened = np.fft.irfft(spectrum / np.sqrt(psd_grid), n=x.size)
    return whitened
