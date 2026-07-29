"""Frequency-selective filters for uniformly sampled time series.

Butterworth low-pass, high-pass, band-pass and band-stop filters in
second-order-section (SOS) form, a second-order IIR notch filter,
Savitzky-Golay polynomial smoothing/differentiation, and a plain moving
average. All IIR filters can be applied either with zero phase
(forward-backward) or causally (single pass).
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage
import scipy.signal
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "bandpass",
    "bandstop",
    "highpass",
    "lowpass",
    "moving_average",
    "notch",
    "savgol",
]


def _check_fs(fs: float) -> float:
    """Validate and return the sampling frequency."""
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive number, got {fs}.")
    return fs


def _check_order(order: int) -> int:
    """Validate and return the Butterworth filter order."""
    if not float(order).is_integer() or order < 1:
        raise ValueError(f"order must be a positive integer, got {order}.")
    return int(order)


def _check_cutoff(cutoff: float | ArrayLike, fs: float) -> None:
    """Validate that cutoff frequencies lie strictly inside (0, fs/2)."""
    values = np.atleast_1d(np.asarray(cutoff, dtype=float))
    nyquist = fs / 2.0
    if np.any(~np.isfinite(values)) or np.any(values <= 0) or np.any(values >= nyquist):
        raise ValueError(
            f"cutoff frequencies must satisfy 0 < fc < fs/2 "
            f"(fs/2 = {nyquist:g} Hz), got {cutoff}."
        )


def _butter_filter(
    x: ArrayLike,
    fs: float,
    cutoff: float | ArrayLike,
    order: int,
    btype: str,
    zero_phase: bool,
) -> NDArray[np.floating]:
    """Design a Butterworth SOS filter and apply it along the last axis."""
    fs = _check_fs(fs)
    order = _check_order(order)
    _check_cutoff(cutoff, fs)
    sos = scipy.signal.butter(order, cutoff, btype=btype, fs=fs, output="sos")
    if zero_phase:
        return scipy.signal.sosfiltfilt(sos, x)
    return scipy.signal.sosfilt(sos, x)


def lowpass(
    x: ArrayLike,
    fs: float,
    cutoff: float,
    order: int = 4,
    zero_phase: bool = True,
) -> NDArray[np.floating]:
    """Apply a Butterworth low-pass filter.

    Parameters
    ----------
    x : array_like
        Input signal. May be n-dimensional; filtering is applied along
        the last axis.
    fs : float
        Sampling frequency in Hz. Must be positive.
    cutoff : float
        -3 dB cutoff frequency in Hz. Must satisfy
        ``0 < cutoff < fs/2``.
    order : int, optional
        Filter order. Default is 4. With ``zero_phase=True`` the
        effective order is doubled (8) because the filter is applied
        twice.
    zero_phase : bool, optional
        If True (default), apply the filter forward and backward
        (`scipy.signal.sosfiltfilt`): zero group delay, no phase
        distortion, squared magnitude response. If False, apply it once
        causally (`scipy.signal.sosfilt`), which preserves the
        single-pass magnitude response but introduces frequency-dependent
        group delay.

    Returns
    -------
    numpy.ndarray
        Filtered signal with the same shape as ``x``.

    Raises
    ------
    ValueError
        If ``fs <= 0``, ``order < 1``, or ``cutoff`` is not in
        ``(0, fs/2)``.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(2000) / 1000.0
    >>> x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)
    >>> y = lowpass(x, fs=1000.0, cutoff=50.0)
    >>> y.shape == x.shape
    True
    """
    return _butter_filter(x, fs, cutoff, order, "lowpass", zero_phase)


def highpass(
    x: ArrayLike,
    fs: float,
    cutoff: float,
    order: int = 4,
    zero_phase: bool = True,
) -> NDArray[np.floating]:
    """Apply a Butterworth high-pass filter.

    Parameters
    ----------
    x : array_like
        Input signal. May be n-dimensional; filtering is applied along
        the last axis.
    fs : float
        Sampling frequency in Hz. Must be positive.
    cutoff : float
        -3 dB cutoff frequency in Hz. Must satisfy
        ``0 < cutoff < fs/2``.
    order : int, optional
        Filter order. Default is 4 (effectively 8 with
        ``zero_phase=True``).
    zero_phase : bool, optional
        If True (default), use forward-backward filtering
        (`scipy.signal.sosfiltfilt`) with zero phase distortion;
        otherwise filter causally (`scipy.signal.sosfilt`).

    Returns
    -------
    numpy.ndarray
        Filtered signal with the same shape as ``x``.

    Raises
    ------
    ValueError
        If ``fs <= 0``, ``order < 1``, or ``cutoff`` is not in
        ``(0, fs/2)``.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(2000) / 1000.0
    >>> x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)
    >>> y = highpass(x, fs=1000.0, cutoff=50.0)
    >>> y.shape == x.shape
    True
    """
    return _butter_filter(x, fs, cutoff, order, "highpass", zero_phase)


def bandpass(
    x: ArrayLike,
    fs: float,
    cutoff: tuple[float, float],
    order: int = 4,
    zero_phase: bool = True,
) -> NDArray[np.floating]:
    """Apply a Butterworth band-pass filter.

    Parameters
    ----------
    x : array_like
        Input signal. May be n-dimensional; filtering is applied along
        the last axis.
    fs : float
        Sampling frequency in Hz. Must be positive.
    cutoff : (float, float)
        Lower and upper -3 dB cutoff frequencies ``(f_low, f_high)`` in
        Hz. Must satisfy ``0 < f_low < f_high < fs/2``.
    order : int, optional
        Filter order per band edge. Default is 4 (the SOS has 2 * order
        sections, and ``zero_phase=True`` doubles the effective order
        again).
    zero_phase : bool, optional
        If True (default), use forward-backward filtering
        (`scipy.signal.sosfiltfilt`) with zero phase distortion;
        otherwise filter causally (`scipy.signal.sosfilt`).

    Returns
    -------
    numpy.ndarray
        Filtered signal with the same shape as ``x``.

    Raises
    ------
    ValueError
        If ``fs <= 0``, ``order < 1``, or the cutoff pair is not ordered
        within ``(0, fs/2)``.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(4000) / 1000.0
    >>> x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 100 * t)
    >>> y = bandpass(x, fs=1000.0, cutoff=(80.0, 120.0))
    >>> y.shape == x.shape
    True
    """
    low, high = _check_band(cutoff)
    return _butter_filter(x, fs, (low, high), order, "bandpass", zero_phase)


def bandstop(
    x: ArrayLike,
    fs: float,
    cutoff: tuple[float, float],
    order: int = 4,
    zero_phase: bool = True,
) -> NDArray[np.floating]:
    """Apply a Butterworth band-stop (band-reject) filter.

    Parameters
    ----------
    x : array_like
        Input signal. May be n-dimensional; filtering is applied along
        the last axis.
    fs : float
        Sampling frequency in Hz. Must be positive.
    cutoff : (float, float)
        Lower and upper -3 dB cutoff frequencies ``(f_low, f_high)`` of
        the rejected band in Hz. Must satisfy
        ``0 < f_low < f_high < fs/2``.
    order : int, optional
        Filter order per band edge. Default is 4.
    zero_phase : bool, optional
        If True (default), use forward-backward filtering
        (`scipy.signal.sosfiltfilt`) with zero phase distortion;
        otherwise filter causally (`scipy.signal.sosfilt`).

    Returns
    -------
    numpy.ndarray
        Filtered signal with the same shape as ``x``.

    Raises
    ------
    ValueError
        If ``fs <= 0``, ``order < 1``, or the cutoff pair is not ordered
        within ``(0, fs/2)``.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(4000) / 1000.0
    >>> x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 100 * t)
    >>> y = bandstop(x, fs=1000.0, cutoff=(80.0, 120.0))
    >>> y.shape == x.shape
    True
    """
    low, high = _check_band(cutoff)
    return _butter_filter(x, fs, (low, high), order, "bandstop", zero_phase)


def _check_band(cutoff: tuple[float, float]) -> tuple[float, float]:
    """Validate a (low, high) cutoff pair and return it as floats."""
    try:
        low, high = cutoff
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cutoff must be a (f_low, f_high) pair, got {cutoff!r}."
        ) from exc
    low, high = float(low), float(high)
    if low >= high:
        raise ValueError(f"cutoff must satisfy f_low < f_high, got ({low}, {high}).")
    return low, high


def notch(
    x: ArrayLike,
    fs: float,
    f0: float,
    q: float = 30,
    zero_phase: bool = True,
) -> NDArray[np.floating]:
    """Apply a second-order IIR notch (band-reject) filter at ``f0``.

    Designed with `scipy.signal.iirnotch`; typical use is removing power
    line interference (50/60 Hz) and its harmonics.

    Parameters
    ----------
    x : array_like
        Input signal. May be n-dimensional; filtering is applied along
        the last axis.
    fs : float
        Sampling frequency in Hz. Must be positive.
    f0 : float
        Notch center frequency in Hz. Must satisfy ``0 < f0 < fs/2``.
    q : float, optional
        Quality factor, ``q = f0 / bandwidth``: the rejected -3 dB band
        is ``f0 * (1 ± 1/(2q))``. Larger ``q`` gives a narrower notch.
        Default is 30. Must be positive.
    zero_phase : bool, optional
        If True (default), apply the filter forward and backward
        (`scipy.signal.filtfilt`) for zero phase distortion; otherwise
        filter causally (`scipy.signal.lfilter`).

    Returns
    -------
    numpy.ndarray
        Filtered signal with the same shape as ``x``.

    Raises
    ------
    ValueError
        If ``fs <= 0``, ``q <= 0``, or ``f0`` is not in ``(0, fs/2)``.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(2000) / 1000.0
    >>> x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)
    >>> y = notch(x, fs=1000.0, f0=50.0)
    >>> y.shape == x.shape
    True
    """
    fs = _check_fs(fs)
    _check_cutoff(f0, fs)
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}.")
    b, a = scipy.signal.iirnotch(f0, q, fs=fs)
    if zero_phase:
        return scipy.signal.filtfilt(b, a, x)
    return scipy.signal.lfilter(b, a, x)


def savgol(
    x: ArrayLike,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    axis: int = -1,
) -> NDArray[np.floating]:
    """Apply a Savitzky-Golay filter (local polynomial least squares).

    Wraps `scipy.signal.savgol_filter`. Each output sample is the value
    (or derivative) of a polynomial of degree ``polyorder`` fitted by
    least squares to the ``window_length`` samples centered on it. The
    filter smooths noise while preserving peak shapes and moments better
    than a moving average, and can directly estimate smoothed
    derivatives.

    Parameters
    ----------
    x : array_like
        Input signal. May be n-dimensional.
    window_length : int
        Length of the filter window. Must be a positive odd integer not
        exceeding the size of ``x`` along ``axis``.
    polyorder : int
        Degree of the fitted polynomial. Must be a non-negative integer
        smaller than ``window_length``.
    deriv : int, optional
        Order of the derivative to compute: 0 (default) smoothes,
        1 returns the first derivative, etc. Must satisfy
        ``0 <= deriv <= polyorder``. Derivatives are scaled per sample
        interval (``delta=1``); divide by the sampling interval to get
        physical derivatives.
    axis : int, optional
        Axis along which to filter. Default is -1.

    Returns
    -------
    numpy.ndarray
        Filtered signal with the same shape as ``x``.

    Raises
    ------
    ValueError
        If ``window_length`` is not a positive odd integer, if
        ``polyorder`` is negative or not smaller than ``window_length``,
        or if ``deriv`` is not in ``[0, polyorder]``.

    Notes
    -----
    At the boundaries the default mode ``"interp"`` of
    `scipy.signal.savgol_filter` is used: the polynomial fitted to the
    nearest full window is evaluated at the edge samples, which is
    unbiased but increases variance at the edges.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = np.sin(2 * np.pi * np.arange(200) / 50) + 0.2 * rng.normal(size=200)
    >>> y = savgol(x, window_length=11, polyorder=3)
    >>> y.shape == x.shape
    True
    """
    if not float(window_length).is_integer() or window_length < 1:
        raise ValueError(f"window_length must be a positive integer, got {window_length}.")
    window_length = int(window_length)
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}.")
    if not float(polyorder).is_integer() or polyorder < 0:
        raise ValueError(f"polyorder must be a non-negative integer, got {polyorder}.")
    polyorder = int(polyorder)
    if polyorder >= window_length:
        raise ValueError(
            f"polyorder ({polyorder}) must be smaller than window_length "
            f"({window_length})."
        )
    if not float(deriv).is_integer() or not 0 <= deriv <= polyorder:
        raise ValueError(
            f"deriv must be an integer in [0, polyorder={polyorder}], got {deriv}."
        )
    return scipy.signal.savgol_filter(
        x, window_length, polyorder, deriv=int(deriv), axis=axis
    )


def moving_average(
    x: ArrayLike, window: int, axis: int = -1
) -> NDArray[np.floating]:
    """Compute the equally weighted moving (running) average.

    Implemented with `scipy.ndimage.uniform_filter1d`.

    Parameters
    ----------
    x : array_like
        Input signal. May be n-dimensional.
    window : int
        Number of samples per averaging window. Must be a positive
        integer. Even values are allowed; the window is then centered
        with the half-sample convention of
        `scipy.ndimage.uniform_filter1d` (``origin=0``), i.e. output
        sample ``i`` averages ``x[i - (w-1)//2 : i + w//2 + 1]``.
    axis : int, optional
        Axis along which to average. Default is -1.

    Returns
    -------
    numpy.ndarray
        Averaged signal with the same shape as ``x``.

    Raises
    ------
    ValueError
        If ``window`` is not a positive integer.

    Notes
    -----
    Edge semantics: near the boundaries the window is truncated and the
    nearest edge sample is replicated (``mode="nearest"``) instead of
    zero-padding, so edge values are means over effectively fewer
    distinct samples but are not biased toward zero.

    A moving average is a poor low-pass filter (sinc-like frequency
    response with large side lobes); use `lowpass` for spectral
    separation and `savgol` when peak shapes should be preserved.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(10.0)
    >>> moving_average(x, 3)[0]
    0.5
    """
    if not float(window).is_integer() or window < 1:
        raise ValueError(f"window must be a positive integer, got {window}.")
    return scipy.ndimage.uniform_filter1d(x, size=int(window), axis=axis, mode="nearest")
