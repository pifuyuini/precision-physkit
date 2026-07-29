"""Peak detection, characterization, and model-based peak fitting.

`find_peaks` wraps `scipy.signal.find_peaks` and augments each detection
with prominence, base width, FWHM, and a numerically integrated area,
collected in a `PeakAnalysisResult`. `fit_peaks` refines detections by
fitting a sum of Gaussian, Lorentzian, or Voigt profiles plus a constant
baseline with `scipy.optimize.curve_fit`, returning per-peak parameters
with 1-sigma uncertainties. `peak_area` numerically integrates a peak
between two positions with optional baseline subtraction.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.optimize
import scipy.signal
import scipy.special
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "FittedPeak",
    "Peak",
    "PeakAnalysisResult",
    "PeakFitResult",
    "find_peaks",
    "fit_peaks",
    "peak_area",
]

PeakModel = Literal["gaussian", "lorentzian", "voigt"]
"""Line-shape models supported by `fit_peaks`."""

_PEAK_MODELS = ("gaussian", "lorentzian", "voigt")

_GAUSS_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))  # FWHM = _GAUSS_FWHM * sigma


@dataclass(frozen=True)
class Peak:
    """A single detected peak.

    All positions and widths are given in units of the ``x_axis`` passed
    to `find_peaks`, or in sample units when no axis was provided.

    Attributes
    ----------
    index : int
        Index of the peak maximum in the input array.
    position : float
        Coordinate of the peak maximum (``x_axis[index]`` or the index
        itself as a float).
    height : float
        Signal value at the peak maximum (measured from zero, i.e. not
        background-subtracted).
    prominence : float
        Peak prominence: the vertical distance between the maximum and
        its lowest contour line (see `scipy.signal.peak_prominences`).
    width : float
        Full width at the base of the prominence
        (``rel_height=1.0`` in `scipy.signal.peak_widths`), i.e. the full
        horizontal extent of the peak at its lowest contour line.
    fwhm : float
        Full width at half prominence (``rel_height=0.5``). For a peak
        sitting on a near-zero background this approximates the full
        width at half maximum.
    area : float
        Trapezoidal integral of the raw signal over
        ``[left_base, right_base]``. Includes any background under the
        peak; use `peak_area` with baseline subtraction or
        `fit_peaks` for background-free areas.
    left_base, right_base : float
        Fractional index positions where the prominence base contour
        intersects the signal (interpolation positions from
        `scipy.signal.peak_widths`).
    """

    index: int
    position: float
    height: float
    prominence: float
    width: float
    fwhm: float
    area: float
    left_base: float
    right_base: float


@dataclass
class PeakAnalysisResult:
    """Container returned by `find_peaks`.

    Attributes
    ----------
    peaks : list of Peak
        Detected peaks, ordered by position.
    x_axis : numpy.ndarray or None
        The coordinate axis used for positions/widths, or None when the
        analysis ran in sample-index units.
    """

    peaks: list[Peak]
    x_axis: NDArray[np.floating] | None = None

    def __len__(self) -> int:
        return len(self.peaks)

    def __iter__(self) -> Iterator[Peak]:
        return iter(self.peaks)

    @property
    def positions(self) -> NDArray[np.floating]:
        """Peak positions as an array (in ``x_axis`` or sample units)."""
        return np.array([p.position for p in self.peaks])

    @property
    def heights(self) -> NDArray[np.floating]:
        """Peak heights as an array."""
        return np.array([p.height for p in self.peaks])

    @property
    def prominences(self) -> NDArray[np.floating]:
        """Peak prominences as an array."""
        return np.array([p.prominence for p in self.peaks])

    @property
    def fwhms(self) -> NDArray[np.floating]:
        """Peak FWHMs as an array."""
        return np.array([p.fwhm for p in self.peaks])


def _trapezoid_span(
    y: NDArray[np.floating],
    x_axis: NDArray[np.floating] | None,
    left: float,
    right: float,
) -> float:
    """Integrate ``y`` trapezoidally between fractional indices.

    ``left``/``right`` are fractional sample positions; endpoint values
    are linearly interpolated. ``x_axis`` provides the physical
    coordinates; integration falls back to sample-index units when it is
    None.
    """
    grid = np.arange(y.size, dtype=float)
    inner = np.arange(np.ceil(left), right)
    ks = np.concatenate(([left], inner, [right]))
    ys = np.interp(ks, grid, y)
    xs = np.interp(ks, grid, x_axis) if x_axis is not None else ks
    return float(np.trapezoid(ys, xs))


def find_peaks(
    x: ArrayLike,
    height: float | None = None,
    prominence: float | None = None,
    distance: float | None = None,
    width: float | None = None,
    x_axis: ArrayLike | None = None,
) -> PeakAnalysisResult:
    """Detect peaks and characterize each with width, FWHM, and area.

    Thin wrapper around `scipy.signal.find_peaks` plus
    `scipy.signal.peak_widths`.

    Parameters
    ----------
    x : array_like, shape (n,)
        1-D signal to search.
    height : float, optional
        Minimum peak height (see `scipy.signal.find_peaks`).
    prominence : float, optional
        Minimum peak prominence (see `scipy.signal.find_peaks`).
    distance : float, optional
        Minimum horizontal distance between neighboring peaks, in
        samples (see `scipy.signal.find_peaks`).
    width : float, optional
        Minimum peak width at half prominence, in samples (see
        `scipy.signal.find_peaks`).
    x_axis : array_like, shape (n,), optional
        Physical coordinates of the samples (e.g. time or frequency).
        Must be 1-D, the same length as ``x``, and strictly increasing.
        When given, `Peak.position`, `Peak.width`, `Peak.fwhm`, and
        `Peak.area` are reported in physical units.

    Returns
    -------
    PeakAnalysisResult
        Detected peaks ordered by position. The result is empty
        (``len(result) == 0``) when no peak satisfies the criteria.

    Raises
    ------
    ValueError
        If ``x`` is not 1-D or ``x_axis`` is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 1001)
    >>> x = np.exp(-((t - 3) ** 2) / 0.02) + 2 * np.exp(-((t - 7) ** 2) / 0.08)
    >>> result = find_peaks(x, prominence=0.1, x_axis=t)
    >>> [round(p, 1) for p in result.positions]
    [3.0, 7.0]
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array.")
    axis = None
    if x_axis is not None:
        axis = np.asarray(x_axis, dtype=float)
        if axis.ndim != 1 or axis.size != x.size:
            raise ValueError("x_axis must be 1-D with the same length as x.")
        if np.any(np.diff(axis) <= 0):
            raise ValueError("x_axis must be strictly increasing.")

    indices, props = scipy.signal.find_peaks(
        x, height=height, prominence=prominence, distance=distance, width=width
    )
    peaks: list[Peak] = []
    if indices.size == 0:
        return PeakAnalysisResult(peaks=peaks, x_axis=axis)

    prominences = props.get("prominences")
    if prominences is None:
        prominences = scipy.signal.peak_prominences(x, indices)[0]
    widths_base, _, lefts_base, rights_base = scipy.signal.peak_widths(
        x, indices, rel_height=1.0
    )
    widths_half, _, lefts_half, rights_half = scipy.signal.peak_widths(
        x, indices, rel_height=0.5
    )

    grid = np.arange(x.size, dtype=float)
    for k, idx in enumerate(indices):
        left_b, right_b = float(lefts_base[k]), float(rights_base[k])
        if axis is not None:
            position = float(axis[idx])
            width_val = float(
                np.interp(right_b, grid, axis) - np.interp(left_b, grid, axis)
            )
            fwhm_val = float(
                np.interp(rights_half[k], grid, axis)
                - np.interp(lefts_half[k], grid, axis)
            )
        else:
            position = float(idx)
            width_val = float(widths_base[k])
            fwhm_val = float(widths_half[k])
        area = _trapezoid_span(x, axis, left_b, right_b)
        peaks.append(
            Peak(
                index=int(idx),
                position=position,
                height=float(x[idx]),
                prominence=float(prominences[k]),
                width=width_val,
                fwhm=fwhm_val,
                area=area,
                left_base=left_b,
                right_base=right_b,
            )
        )
    return PeakAnalysisResult(peaks=peaks, x_axis=axis)


@dataclass(frozen=True)
class FittedPeak:
    """Parameters of one peak component from `fit_peaks`.

    Attributes
    ----------
    center : float
        Fitted peak center, in units of the fit coordinate ``x``.
    amplitude : float
        Fitted peak height above the constant baseline.
    fwhm : float
        Full width at half maximum of the component. Exact for
        Gaussian (``2*sqrt(2*ln2)*sigma``) and Lorentzian (``2*gamma``)
        profiles; for Voigt profiles the Olivero-Longbothum approximation
        is used.
    area : float
        Analytic area of the component (baseline excluded):
        ``amplitude * sigma * sqrt(2*pi)`` for Gaussian,
        ``amplitude * pi * gamma`` for Lorentzian, and
        ``amplitude / voigt_profile(0, sigma, gamma)`` for Voigt.
    params : dict of str to float
        All fitted parameters of this component: ``center``,
        ``amplitude``, ``sigma``, and additionally ``gamma`` for the
        Voigt model.
    errors : dict of str to float
        1-sigma (standard error) uncertainties for every entry of
        ``params`` plus ``fwhm``, propagated from the fit covariance
        matrix.
    """

    center: float
    amplitude: float
    fwhm: float
    area: float
    params: dict[str, float]
    errors: dict[str, float]


@dataclass
class PeakFitResult:
    """Container returned by `fit_peaks`.

    Attributes
    ----------
    model : {"gaussian", "lorentzian", "voigt"}
        Line-shape model that was fitted.
    peaks : list of FittedPeak
        Fitted components, in the order of the input ``peak_indices``.
    baseline : float
        Fitted constant offset shared by all components.
    baseline_error : float
        1-sigma uncertainty of the baseline.
    x : numpy.ndarray
        The full input coordinate array.
    y_fit : numpy.ndarray
        The fitted model curve (baseline plus all components) evaluated
        on the full input ``x``, including regions excluded from the fit
        by ``window``.
    covariance : numpy.ndarray
        Full covariance matrix of the fitted parameters, ordered as
        ``[baseline, (center, amplitude, sigma[, gamma]) per peak]``.
    """

    model: str
    peaks: list[FittedPeak]
    baseline: float
    baseline_error: float
    x: NDArray[np.floating]
    y_fit: NDArray[np.floating]
    covariance: NDArray[np.floating]


def _gaussian(x: NDArray[np.floating], c: float, a: float, s: float) -> NDArray:
    return a * np.exp(-0.5 * ((x - c) / s) ** 2)


def _lorentzian(x: NDArray[np.floating], c: float, a: float, g: float) -> NDArray:
    return a * g**2 / ((x - c) ** 2 + g**2)


def _voigt(x: NDArray[np.floating], c: float, a: float, s: float, g: float) -> NDArray:
    return a * scipy.special.voigt_profile(x - c, s, g) / scipy.special.voigt_profile(
        0.0, s, g
    )


def _voigt_fwhm(sigma: float, gamma: float) -> float:
    """Olivero-Longbothum approximation of the Voigt FWHM."""
    f_g = _GAUSS_FWHM * sigma
    f_l = 2.0 * gamma
    return 0.5346 * f_l + np.sqrt(0.2166 * f_l**2 + f_g**2)


def _component_area(model: str, amplitude: float, sigma: float, gamma: float) -> float:
    """Analytic area of a single component (baseline excluded)."""
    if model == "gaussian":
        return float(amplitude * sigma * np.sqrt(2.0 * np.pi))
    if model == "lorentzian":
        return float(amplitude * np.pi * gamma)
    return float(amplitude / scipy.special.voigt_profile(0.0, sigma, gamma))


def _estimate_fwhm_samples(y: NDArray[np.floating], peak: int) -> float:
    """Estimate the half-prominence width of one peak in samples.

    Falls back to 3 samples for degenerate peaks (flat tops, edges).
    """
    try:
        width = scipy.signal.peak_widths(y, [peak], rel_height=0.5)[0][0]
    except (ValueError, IndexError, RuntimeError):
        return 3.0
    if not np.isfinite(width) or width <= 0:
        return 3.0
    return float(width)


def fit_peaks(
    x: ArrayLike,
    y: ArrayLike,
    peak_indices: Sequence[int],
    model: PeakModel = "gaussian",
    window: float | None = None,
) -> PeakFitResult:
    """Fit a sum of line-shape profiles plus a constant baseline to data.

    Each detected peak is modeled by one component; all components share
    a single constant baseline ``b``::

        y(x) = b + sum_k component_k(x; center, amplitude, ...)

    Initial guesses are derived from the detections: the center from the
    sample position, the amplitude from the height above a robust
    baseline estimate (median of the outer 10% samples), and the width
    from the measured half-prominence width.

    Parameters
    ----------
    x : array_like, shape (n,)
        Coordinate array. Must be 1-D and strictly increasing.
    y : array_like, shape (n,)
        Signal values.
    peak_indices : sequence of int
        Indices of the peaks in ``x``/``y``, e.g. ``[p.index for p in
        find_peaks(y)]``. At least one index is required.
    model : {"gaussian", "lorentzian", "voigt"}, optional
        Line-shape model. ``"gaussian"`` (default):
        ``A*exp(-(x-c)^2/(2*sigma^2))``; ``"lorentzian"``:
        ``A*gamma^2/((x-c)^2+gamma^2)``; ``"voigt"``: Voigt profile
        (convolution of Gaussian and Lorentzian) parameterized so that
        ``A`` is the peak height, with per-peak ``sigma`` (Gaussian) and
        ``gamma`` (Lorentzian) widths.
    window : float, optional
        Half-width (in units of ``x``) of the data window around each
        initial peak center used for the fit; only samples within
        ``|x - c0| <= window`` of any peak enter the least squares. If
        None (default), all samples are used.

    Returns
    -------
    PeakFitResult
        Per-peak parameters with 1-sigma uncertainties, the shared
        baseline, the fitted curve on the full input grid, and the full
        parameter covariance matrix.

    Raises
    ------
    ValueError
        If the inputs are inconsistent (shapes, monotonicity, empty
        ``peak_indices``, unknown ``model``, non-positive ``window``) or
        an index is out of range.
    RuntimeError
        If the least-squares optimization does not converge (raised by
        `scipy.optimize.curve_fit`).

    Notes
    -----
    Centers are constrained to the data range and widths/amplitudes to
    positive values (trust-region-reflective bounds). The Voigt model
    cannot distinguish ``sigma`` and ``gamma`` well for a single noisy
    peak; prefer it for high-SNR data or with several peaks.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(-10, 10, 2001)
    >>> y = 3 * np.exp(-((t - 1) ** 2) / 2) + 2 * np.exp(-((t + 4) ** 2) / 8)
    >>> idx = [p.index for p in find_peaks(y, prominence=0.5)]
    >>> res = fit_peaks(t, y, idx, model="gaussian")
    >>> [round(p.center, 1) for p in res.peaks]
    [-4.0, 1.0]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1-D arrays of equal length.")
    if x.size < 3:
        raise ValueError("At least three samples are required.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing.")
    if model not in _PEAK_MODELS:
        raise ValueError(f"model must be one of {_PEAK_MODELS}, got {model!r}.")
    indices = [int(i) for i in peak_indices]
    if not indices:
        raise ValueError("peak_indices must contain at least one index.")
    for i in indices:
        if not 0 <= i < x.size:
            raise ValueError(f"peak index {i} is out of range [0, {x.size}).")
    if window is not None and window <= 0:
        raise ValueError(f"window must be positive, got {window}.")

    n_peaks = len(indices)
    n_par = 4 if model == "voigt" else 3
    dx = float(np.median(np.diff(x)))

    # Initial baseline from the outer 10% of samples on both sides.
    edge = max(1, x.size // 10)
    b0 = float(np.median(np.concatenate([y[:edge], y[-edge:]])))

    p0 = [b0]
    lower = [-np.inf]
    upper = [np.inf]
    width_lb = dx * 1e-3
    mask = np.zeros(x.size, dtype=bool)
    for i in indices:
        c0 = float(x[i])
        a0 = max(float(y[i]) - b0, 1e-12)
        fwhm0 = _estimate_fwhm_samples(y, i) * dx
        if model == "gaussian":
            s0 = max(fwhm0 / _GAUSS_FWHM, width_lb)
            block, lb, ub = [c0, a0, s0], [x[0], 0.0, width_lb], [x[-1], np.inf, np.inf]
        elif model == "lorentzian":
            g0 = max(fwhm0 / 2.0, width_lb)
            block, lb, ub = [c0, a0, g0], [x[0], 0.0, width_lb], [x[-1], np.inf, np.inf]
        else:
            s0 = max(fwhm0 / 3.6, width_lb)
            g0 = max(fwhm0 / 3.6, width_lb)
            block = [c0, a0, s0, g0]
            lb = [x[0], 0.0, width_lb, width_lb]
            ub = [x[-1], np.inf, np.inf, np.inf]
        p0.extend(block)
        lower.extend(lb)
        upper.extend(ub)
        if window is not None:
            mask |= np.abs(x - c0) <= window
    if window is None:
        mask[:] = True

    def component(xx: NDArray[np.floating], params: Sequence[float]) -> NDArray:
        if model == "gaussian":
            return _gaussian(xx, *params)
        if model == "lorentzian":
            return _lorentzian(xx, *params)
        return _voigt(xx, *params)

    def composite(xx: NDArray[np.floating], *p: float) -> NDArray:
        out = np.full(xx.shape, p[0])
        for k in range(n_peaks):
            base = 1 + k * n_par
            out = out + component(xx, p[base : base + n_par])
        return out

    popt, pcov = scipy.optimize.curve_fit(
        composite,
        x[mask],
        y[mask],
        p0=p0,
        bounds=(lower, upper),
        max_nfev=10000,
    )
    perr = np.sqrt(np.diag(pcov))

    peaks: list[FittedPeak] = []
    for k in range(n_peaks):
        base = 1 + k * n_par
        sl = slice(base, base + n_par)
        p_k, e_k = popt[sl], perr[sl]
        center, amplitude, sigma = float(p_k[0]), float(p_k[1]), float(p_k[2])
        gamma = float(p_k[3]) if model == "voigt" else sigma
        if model == "gaussian":
            fwhm = float(_GAUSS_FWHM * sigma)
            fwhm_err = float(_GAUSS_FWHM * e_k[2])
        elif model == "lorentzian":
            fwhm = 2.0 * gamma
            fwhm_err = float(2.0 * e_k[2])
        else:
            fwhm = float(_voigt_fwhm(sigma, gamma))
            # Propagate the (sigma, gamma) covariance through the
            # Olivero-Longbothum approximation with a numeric gradient.
            eps = 1e-6
            grad = np.array(
                [
                    (_voigt_fwhm(sigma + eps, gamma) - _voigt_fwhm(sigma - eps, gamma))
                    / (2 * eps),
                    (_voigt_fwhm(sigma, gamma + eps) - _voigt_fwhm(sigma, gamma - eps))
                    / (2 * eps),
                ]
            )
            cov_block = pcov[base + 2 : base + 4, base + 2 : base + 4]
            fwhm_err = float(np.sqrt(max(grad @ cov_block @ grad, 0.0)))
        params = {"center": center, "amplitude": amplitude, "sigma": sigma}
        errors = {"center": float(e_k[0]), "amplitude": float(e_k[1]), "sigma": float(e_k[2])}
        if model == "voigt":
            params["gamma"] = gamma
            errors["gamma"] = float(e_k[3])
        errors["fwhm"] = fwhm_err
        peaks.append(
            FittedPeak(
                center=center,
                amplitude=amplitude,
                fwhm=fwhm,
                area=_component_area(model, amplitude, sigma, gamma),
                params=params,
                errors=errors,
            )
        )

    return PeakFitResult(
        model=model,
        peaks=peaks,
        baseline=float(popt[0]),
        baseline_error=float(perr[0]),
        x=x,
        y_fit=composite(x, *popt),
        covariance=pcov,
    )


def peak_area(
    x: ArrayLike,
    y: ArrayLike,
    left: float,
    right: float,
    baseline: float | Literal["edge"] | None = None,
) -> float:
    """Numerically integrate a peak between two x positions.

    The signal is integrated trapezoidally over ``[left, right]``;
    endpoint values are linearly interpolated from the neighboring
    samples.

    Parameters
    ----------
    x : array_like, shape (n,)
        Coordinate array. Must be 1-D and strictly increasing.
    y : array_like, shape (n,)
        Signal values.
    left, right : float
        Integration bounds in units of ``x``. Must satisfy
        ``left < right``. Bounds outside the data range are clamped to
        the nearest sample value (constant extension via `numpy.interp`).
    baseline : float, "edge", or None, optional
        Baseline handling: None (default) integrates the raw signal; a
        float subtracts that constant level; ``"edge"`` subtracts the
        straight line connecting ``(left, y(left))`` and
        ``(right, y(right))`` — the usual valley-to-valley definition of
        peak area for peaks on a sloping background.

    Returns
    -------
    float
        The integrated area in units of ``x * y``.

    Raises
    ------
    ValueError
        If ``x``/``y`` are inconsistent, ``x`` is not strictly
        increasing, ``left >= right``, or ``baseline`` is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(-5, 5, 1001)
    >>> y = np.exp(-(t**2) / 2)
    >>> area = peak_area(t, y, -3, 3)
    >>> abs(area - np.sqrt(2 * np.pi)) < 0.01
    True
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1-D arrays of equal length.")
    if x.size < 2:
        raise ValueError("At least two samples are required.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing.")
    if not left < right:
        raise ValueError(f"left must be smaller than right, got ({left}, {right}).")
    if baseline is not None and baseline != "edge" and not isinstance(baseline, (int, float)):
        raise ValueError(f"baseline must be None, a float, or 'edge', got {baseline!r}.")

    inner = x[(x > left) & (x < right)]
    xs = np.concatenate(([left], inner, [right]))
    ys = np.interp(xs, x, y)
    if baseline == "edge":
        slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
        ys = ys - (ys[0] + slope * (xs - xs[0]))
    elif isinstance(baseline, (int, float)):
        ys = ys - float(baseline)
    return float(np.trapezoid(ys, xs))
