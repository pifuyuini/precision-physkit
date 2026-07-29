"""Spectral estimation on linear and logarithmic frequency scales.

This module provides a unified interface to two estimator families:

* classical FFT-based estimators with a uniform (linear) frequency axis,
  wrapped from :mod:`scipy.signal` (Welch's method and derived quantities);
* the LPSD family of logarithmic-frequency estimators implemented in the
  Rust kernel ``precision_physkit._core`` (imported lazily inside each function, so
  this module remains importable while the extension is not built).

All estimators return a :class:`Spectrum` container holding the frequency
axis, the estimated values and descriptive metadata, so downstream code
(plotting, reporting, archival) does not need to know which estimator
produced the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "Spectrum",
    "welch_psd",
    "welch_csd",
    "welch_coherence",
    "welch_transfer",
    "lpsd",
    "lcsd",
    "lcoherence",
    "ltransfer",
    "lcsd_matrix",
]

SpectrumKind = Literal["psd", "csd", "coherence", "transfer", "asd"]
"""Physical quantity stored in :attr:`Spectrum.values`."""

SpectrumScale = Literal["linear", "log"]
"""Spacing of the frequency axis: uniform or logarithmic."""


def _load_core():
    """Import the Rust kernel lazily (it may not be built at import time)."""
    from . import _core

    return _core


@dataclass
class Spectrum:
    """Container for a single- or multi-channel spectral estimate.

    Attributes
    ----------
    f : ndarray, shape (J,)
        Frequency bin centers in hertz.
    values : ndarray, shape (J,), (J, C) or (J, C, C)
        Estimated spectral values. Real-valued for ``"psd"``,
        ``"coherence"`` and ``"asd"``; complex-valued for ``"csd"`` and
        ``"transfer"``. Multi-channel estimates are stacked along the
        second axis; :func:`lcsd_matrix` additionally returns the full
        cross-spectral density matrix with shape ``(J, C, C)``.
    kind : {"psd", "csd", "coherence", "transfer", "asd"}
        Physical quantity stored in ``values`` (power spectral density,
        cross spectral density, magnitude-squared coherence, transfer
        function, or amplitude spectral density).
    scale : {"linear", "log"}
        Whether ``f`` is uniformly (``"linear"``) or logarithmically
        (``"log"``) spaced. Logarithmic spectra come from the LPSD
        estimator family.
    method : str
        Name of the estimator that produced the spectrum, e.g. ``"welch"``
        or ``"lpsd"``.
    meta : dict
        Free-form metadata: sampling frequency, estimator parameters and,
        for transfer functions estimated with ``with_coherence=True``, the
        magnitude-squared coherence under the key ``"coherence"``.
    """

    f: NDArray[np.floating[Any]]
    values: NDArray[Any]
    kind: SpectrumKind
    scale: SpectrumScale
    method: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_asd(self) -> Spectrum:
        """Convert a power spectral density to an amplitude spectral density.

        The amplitude spectral density (ASD) is the element-wise positive
        square root of the PSD and carries units of ``unit/sqrt(Hz)``
        instead of ``unit**2/Hz``. It is the conventional representation
        for noise floors in precision metrology.

        Returns
        -------
        Spectrum
            A new container with ``kind="asd"`` and
            ``values = sqrt(self.values)``; ``meta`` gains a
            ``"derived_from": "psd"`` entry. Calling this method on a
            spectrum that already is an ASD returns a shallow copy.

        Raises
        ------
        ValueError
            If :attr:`kind` is neither ``"psd"`` nor ``"asd"``. The square
            root of a cross spectrum, coherence or transfer function is
            not a meaningful amplitude spectral density.
        """
        if self.kind == "asd":
            return replace(self, meta=dict(self.meta))
        if self.kind != "psd":
            raise ValueError(
                "to_asd() is only defined for PSD spectra, "
                f"got kind={self.kind!r}"
            )
        return Spectrum(
            f=self.f.copy(),
            values=np.sqrt(self.values),
            kind="asd",
            scale=self.scale,
            method=self.method,
            meta={**self.meta, "derived_from": "psd"},
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Export the spectrum to a :class:`pandas.DataFrame`.

        The frequency axis becomes the index (named ``"frequency"``).
        Single-channel values are stored in one column named after
        :attr:`kind`; multi-channel values of shape ``(J, C)`` are stored
        in columns ``ch0`` … ``ch{C-1}``. Complex values are kept as
        complex dtype.

        Returns
        -------
        pandas.DataFrame
            Table with one row per frequency bin.

        Raises
        ------
        ValueError
            If ``values`` is neither 1-D nor 2-D with leading axis matching
            ``f`` (e.g. the ``(J, C, C)`` output of :func:`lcsd_matrix`).
        """
        import pandas as pd

        f = np.asarray(self.f)
        values = np.asarray(self.values)
        if values.ndim == 1:
            data: dict[str, NDArray[Any]] = {self.kind: values}
        elif values.ndim == 2 and values.shape[0] == f.shape[0]:
            data = {f"ch{j}": values[:, j] for j in range(values.shape[1])}
        else:
            raise ValueError(
                "to_dataframe() supports 1-D values or 2-D (J, C) values, "
                f"got shape {values.shape}"
            )
        return pd.DataFrame(data, index=pd.Index(f, name="frequency"))


def _check_fs(fs: float) -> None:
    if not np.isscalar(fs) or not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive finite scalar, got {fs!r}")


def _as_1d_or_2d(x: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError(
            f"{name} must be a 1-D (N,) or 2-D (N, C) array, got {arr.ndim}-D"
        )
    if arr.size == 0 or (arr.ndim == 2 and arr.shape[1] == 0):
        raise ValueError(f"{name} must be non-empty")
    return arr


def _as_pair(x: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x_arr = _as_1d_or_2d(x, "x")
    y_arr = _as_1d_or_2d(y, "y")
    if x_arr.shape != y_arr.shape:
        raise ValueError(
            "x and y must have identical shapes: both (N,) or both (N, C); "
            f"got {x_arr.shape} and {y_arr.shape}"
        )
    return x_arr, y_arr


def _default_nperseg(n: int) -> int:
    """Heuristic segment length: about eight non-overlapping segments.

    Clamped to ``[8, n]`` so very short records still produce a valid,
    if coarse, estimate. With the default 50 % overlap this corresponds to
    roughly fifteen averaging segments.
    """
    return int(min(n, max(8, n // 8)))


def _welch_meta(
    fs: float,
    nperseg: int,
    noverlap: int | None,
    window: str,
    detrend: str,
    n: int,
) -> dict[str, Any]:
    noverlap_eff = nperseg // 2 if noverlap is None else int(noverlap)
    step = max(1, nperseg - noverlap_eff)
    n_segments = 1 + (n - nperseg) // step if n >= nperseg else 0
    return {
        "fs": float(fs),
        "nperseg": int(nperseg),
        "noverlap": noverlap_eff,
        "window": window,
        "detrend": detrend,
        "n_segments": int(n_segments),
    }


def _lpsd_meta(fs: float, Jdes: int, Kdes: int, xi: float) -> dict[str, Any]:
    return {"fs": float(fs), "Jdes": int(Jdes), "Kdes": int(Kdes), "xi": float(xi)}


def welch_psd(
    x: ArrayLike,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str = "constant",
) -> Spectrum:
    """Power spectral density via Welch's averaged periodogram method.

    Parameters
    ----------
    x : array_like, shape (N,) or (N, C)
        Time series. For a 2-D input each of the ``C`` columns is treated
        as an independent channel and estimated separately; the result has
        shape ``(J, C)``.
    fs : float
        Sampling frequency in hertz.
    nperseg : int, optional
        Segment length. Defaults to a heuristic of about eight
        non-overlapping segments (``max(8, N // 8)``), a reasonable
        bias/variance compromise for unattended batch processing.
    noverlap : int, optional
        Segment overlap in samples; defaults to ``nperseg // 2`` (50 %).
    window : str, optional
        Window function passed to :func:`scipy.signal.welch`.
    detrend : str, optional
        Per-segment detrending passed to :func:`scipy.signal.welch`;
        ``"constant"`` removes the segment mean.

    Returns
    -------
    Spectrum
        ``kind="psd"``, ``scale="linear"``, one-sided spectrum for real
        input. ``meta`` records the effective estimator parameters,
        including the number of averaged segments (``"n_segments"``).

    Raises
    ------
    ValueError
        If ``fs`` is not positive, or ``x`` is neither 1-D nor 2-D.
    """
    _check_fs(fs)
    x_arr = _as_1d_or_2d(x, "x")
    if nperseg is None:
        nperseg = _default_nperseg(x_arr.shape[0])
    f, pxx = signal.welch(
        x_arr,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        axis=0,
    )
    return Spectrum(
        f=np.asarray(f, dtype=float),
        values=np.asarray(pxx),
        kind="psd",
        scale="linear",
        method="welch",
        meta=_welch_meta(fs, nperseg, noverlap, window, detrend, x_arr.shape[0]),
    )


def welch_csd(
    x: ArrayLike,
    y: ArrayLike,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str = "constant",
) -> Spectrum:
    """Cross spectral density between two signals via Welch's method.

    Parameters
    ----------
    x, y : array_like, both (N,) or both (N, C) with identical shape
        Input signal pairs. For 2-D inputs the cross spectrum is computed
        column-wise between channel pairs ``(x[:, j], y[:, j])``.
    fs : float
        Sampling frequency in hertz.
    nperseg, noverlap, window, detrend
        See :func:`welch_psd`.

    Returns
    -------
    Spectrum
        ``kind="csd"`` with complex values; ``Sxy[k]`` is the cross
        spectrum of ``x`` and ``y`` (conjugate-symmetric convention of
        :func:`scipy.signal.csd`).

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different shapes or ``fs`` is not positive.
    """
    _check_fs(fs)
    x_arr, y_arr = _as_pair(x, y)
    if nperseg is None:
        nperseg = _default_nperseg(x_arr.shape[0])
    f, sxy = signal.csd(
        x_arr,
        y_arr,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        axis=0,
    )
    return Spectrum(
        f=np.asarray(f, dtype=float),
        values=np.asarray(sxy),
        kind="csd",
        scale="linear",
        method="welch",
        meta=_welch_meta(fs, nperseg, noverlap, window, detrend, x_arr.shape[0]),
    )


def welch_coherence(
    x: ArrayLike,
    y: ArrayLike,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str = "constant",
) -> Spectrum:
    """Magnitude-squared coherence between two signals via Welch's method.

    The coherence ``|Sxy|**2 / (Sxx * Syy)`` lies in ``[0, 1]`` and
    measures the linear correlation between ``x`` and ``y`` per frequency
    bin; values near 1 indicate a reliable linear relation.

    Parameters
    ----------
    x, y : array_like, both (N,) or both (N, C) with identical shape
        Input signal pairs (column-wise for 2-D input).
    fs : float
        Sampling frequency in hertz.
    nperseg, noverlap, window, detrend
        See :func:`welch_psd`.

    Returns
    -------
    Spectrum
        ``kind="coherence"`` with real values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different shapes or ``fs`` is not positive.
    """
    _check_fs(fs)
    x_arr, y_arr = _as_pair(x, y)
    if nperseg is None:
        nperseg = _default_nperseg(x_arr.shape[0])
    f, coh = signal.coherence(
        x_arr,
        y_arr,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        axis=0,
    )
    return Spectrum(
        f=np.asarray(f, dtype=float),
        values=np.asarray(coh),
        kind="coherence",
        scale="linear",
        method="welch",
        meta=_welch_meta(fs, nperseg, noverlap, window, detrend, x_arr.shape[0]),
    )


def welch_transfer(
    x: ArrayLike,
    y: ArrayLike,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str = "constant",
    with_coherence: bool = False,
) -> Spectrum:
    """H1 transfer function estimate from input ``x`` to output ``y``.

    The H1 estimator ``H = Sxy / Sxx`` minimises the effect of noise on
    the output ``y`` and is the standard choice when the input is measured
    cleanly. Frequency bins where the input auto-spectrum ``Sxx`` vanishes
    yield ``inf``/``nan`` and should be discarded together with bins of
    low coherence.

    Parameters
    ----------
    x, y : array_like, both (N,) or both (N, C) with identical shape
        Input/output signal pairs (column-wise for 2-D input).
    fs : float
        Sampling frequency in hertz.
    nperseg, noverlap, window, detrend
        See :func:`welch_psd`.
    with_coherence : bool, optional
        If True, additionally compute the magnitude-squared coherence and
        store it in ``meta["coherence"]`` (same frequency grid). The
        coherence is the standard quality indicator for the transfer
        function estimate.

    Returns
    -------
    Spectrum
        ``kind="transfer"`` with complex values.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different shapes or ``fs`` is not positive.
    """
    _check_fs(fs)
    x_arr, y_arr = _as_pair(x, y)
    if nperseg is None:
        nperseg = _default_nperseg(x_arr.shape[0])
    f, sxy = signal.csd(
        x_arr,
        y_arr,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        axis=0,
    )
    _, pxx = signal.welch(
        x_arr,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        axis=0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        transfer = sxy / pxx
    meta = _welch_meta(fs, nperseg, noverlap, window, detrend, x_arr.shape[0])
    if with_coherence:
        _, coh = signal.coherence(
            x_arr,
            y_arr,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            axis=0,
        )
        meta["coherence"] = np.asarray(coh)
    return Spectrum(
        f=np.asarray(f, dtype=float),
        values=np.asarray(transfer),
        kind="transfer",
        scale="linear",
        method="welch",
        meta=meta,
    )


def _pairwise_core(
    core_fn: Any,
    x: ArrayLike,
    y: ArrayLike,
    fs: float,
    Jdes: int,
    Kdes: int,
    xi: float,
    parallel: bool,
) -> tuple[NDArray[np.floating[Any]], NDArray[Any]]:
    """Apply a 1-D two-channel kernel function column-wise.

    All channels share the same record length and estimator parameters, so
    every column pair yields the same logarithmic frequency grid; the grid
    of the first column is returned.
    """
    x_arr, y_arr = _as_pair(x, y)
    if x_arr.ndim == 1:
        f, v = core_fn(x_arr, y_arr, fs, Jdes=Jdes, Kdes=Kdes, xi=xi, parallel=parallel)
        return np.asarray(f, dtype=float), np.asarray(v)
    f: NDArray[np.floating[Any]] | None = None
    cols = []
    for j in range(x_arr.shape[1]):
        fj, vj = core_fn(
            x_arr[:, j], y_arr[:, j], fs, Jdes=Jdes, Kdes=Kdes, xi=xi, parallel=parallel
        )
        if f is None:
            f = np.asarray(fj, dtype=float)
        cols.append(np.asarray(vj))
    if f is None:  # unreachable: _as_pair rejects empty inputs
        raise RuntimeError("no channels to process")
    return f, np.stack(cols, axis=1)


_LPSD_NOTES = """
    Notes
    -----
    LPSD [1]_ estimates the spectrum on a logarithmically spaced frequency
    grid. Unlike Welch's method it uses a *different segment length at
    every frequency bin*: long segments at low frequencies give high
    frequency resolution, short segments at high frequencies give many
    averages and hence low variance. This makes LPSD well suited to
    precision measurements over wide frequency bands (e.g. several decades
    from mHz to kHz), where a fixed segment length either wastes data at
    high frequencies or resolves too little at low frequencies.

    Because the algorithm derives each bin's resolution from the desired
    overlap and averaging count, the number of returned frequency points
    ``J`` generally differs from the desired value ``Jdes``; always use
    the returned frequency axis ``f``, not a reconstructed grid.

    References
    ----------
    .. [1] M. Tröbs and G. Heinzel, "Improved spectrum estimation from
       digitized time series on a logarithmic frequency axis",
       Measurement 39 (2006), 120-129.
"""


def lpsd(
    x: ArrayLike,
    fs: float,
    Jdes: int = 200,
    Kdes: int = 100,
    xi: float = 0.5,
    parallel: bool = True,
) -> Spectrum:
    """Power spectral density on a logarithmic frequency axis (LPSD).

    Parameters
    ----------
    x : array_like, shape (N,) or (N, C)
        Time series. Multi-channel input of shape ``(N, C)`` is passed to
        the kernel directly and estimated channel-wise; the result has
        shape ``(J, C)``.
    fs : float
        Sampling frequency in hertz.
    Jdes : int, optional
        Desired number of logarithmically spaced frequency bins. The
        actual number of bins ``J`` differs in general (see Notes).
    Kdes : int, optional
        Desired number of averaged segments per frequency bin; controls
        the variance of the estimate.
    xi : float, optional
        Fractional segment overlap in ``[0, 1)``; 0.5 is 50 % overlap.
    parallel : bool, optional
        Let the kernel use multiple threads.

    Returns
    -------
    Spectrum
        ``kind="psd"``, ``scale="log"``. ``meta`` records the request
        parameters (``"Jdes"``, ``"Kdes"``, ``"xi"``, ``"fs"``).

    Raises
    ------
    ValueError
        If ``fs`` is not positive or ``x`` is neither 1-D nor 2-D.
    ImportError
        If the Rust extension ``precision_physkit._core`` is not built.
    """
    _check_fs(fs)
    x_arr = _as_1d_or_2d(x, "x")
    core = _load_core()
    f, pxx = core.lpsd_psd(x_arr, fs, Jdes=Jdes, Kdes=Kdes, xi=xi, parallel=parallel)
    return Spectrum(
        f=np.asarray(f, dtype=float),
        values=np.asarray(pxx),
        kind="psd",
        scale="log",
        method="lpsd",
        meta=_lpsd_meta(fs, Jdes, Kdes, xi),
    )


lpsd.__doc__ = (lpsd.__doc__ or "") + _LPSD_NOTES


def lcsd(
    x: ArrayLike,
    y: ArrayLike,
    fs: float,
    Jdes: int = 200,
    Kdes: int = 100,
    xi: float = 0.5,
    parallel: bool = True,
) -> Spectrum:
    """Cross spectral density on a logarithmic frequency axis (LPSD).

    Parameters
    ----------
    x, y : array_like, both (N,) or both (N, C) with identical shape
        Input signal pairs. For 2-D inputs the kernel is applied
        column-wise to the channel pairs ``(x[:, j], y[:, j])``; all
        columns share the same logarithmic frequency grid.
    fs : float
        Sampling frequency in hertz.
    Jdes, Kdes, xi, parallel
        See :func:`lpsd`.

    Returns
    -------
    Spectrum
        ``kind="csd"``, ``scale="log"``, with complex values.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different shapes or ``fs`` is not positive.
    ImportError
        If the Rust extension ``precision_physkit._core`` is not built.
    """
    _check_fs(fs)
    core = _load_core()
    f, sxy = _pairwise_core(core.lpsd_csd, x, y, fs, Jdes, Kdes, xi, parallel)
    return Spectrum(
        f=f,
        values=sxy,
        kind="csd",
        scale="log",
        method="lpsd",
        meta=_lpsd_meta(fs, Jdes, Kdes, xi),
    )


lcsd.__doc__ = (lcsd.__doc__ or "") + _LPSD_NOTES


def lcoherence(
    x: ArrayLike,
    y: ArrayLike,
    fs: float,
    Jdes: int = 200,
    Kdes: int = 100,
    xi: float = 0.5,
    parallel: bool = True,
) -> Spectrum:
    """Magnitude-squared coherence on a logarithmic frequency axis (LPSD).

    Parameters
    ----------
    x, y : array_like, both (N,) or both (N, C) with identical shape
        Input signal pairs (column-wise for 2-D input).
    fs : float
        Sampling frequency in hertz.
    Jdes, Kdes, xi, parallel
        See :func:`lpsd`.

    Returns
    -------
    Spectrum
        ``kind="coherence"``, ``scale="log"``, with real values in
        ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different shapes or ``fs`` is not positive.
    ImportError
        If the Rust extension ``precision_physkit._core`` is not built.
    """
    _check_fs(fs)
    core = _load_core()
    f, coh = _pairwise_core(core.lpsd_coherence, x, y, fs, Jdes, Kdes, xi, parallel)
    return Spectrum(
        f=f,
        values=coh,
        kind="coherence",
        scale="log",
        method="lpsd",
        meta=_lpsd_meta(fs, Jdes, Kdes, xi),
    )


lcoherence.__doc__ = (lcoherence.__doc__ or "") + _LPSD_NOTES


def ltransfer(
    x: ArrayLike,
    y: ArrayLike,
    fs: float,
    Jdes: int = 200,
    Kdes: int = 100,
    xi: float = 0.5,
    parallel: bool = True,
    with_coherence: bool = False,
) -> Spectrum:
    """H1 transfer function estimate on a logarithmic frequency axis.

    Computes ``H1 = Sxy / Sxx`` from input ``x`` to output ``y`` with the
    LPSD estimator.

    Parameters
    ----------
    x, y : array_like, both (N,) or both (N, C) with identical shape
        Input/output signal pairs (column-wise for 2-D input).
    fs : float
        Sampling frequency in hertz.
    Jdes, Kdes, xi, parallel
        See :func:`lpsd`.
    with_coherence : bool, optional
        If True, additionally estimate the magnitude-squared coherence on
        the same grid and store it in ``meta["coherence"]``. This roughly
        doubles the runtime; the coherence is the standard quality
        indicator for the transfer function estimate.

    Returns
    -------
    Spectrum
        ``kind="transfer"``, ``scale="log"``, with complex values.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different shapes or ``fs`` is not positive.
    ImportError
        If the Rust extension ``precision_physkit._core`` is not built.
    """
    _check_fs(fs)
    core = _load_core()
    f, transfer = _pairwise_core(core.lpsd_transfer, x, y, fs, Jdes, Kdes, xi, parallel)
    meta = _lpsd_meta(fs, Jdes, Kdes, xi)
    if with_coherence:
        _, coh = _pairwise_core(core.lpsd_coherence, x, y, fs, Jdes, Kdes, xi, parallel)
        meta["coherence"] = coh
    return Spectrum(
        f=f,
        values=transfer,
        kind="transfer",
        scale="log",
        method="lpsd",
        meta=meta,
    )


ltransfer.__doc__ = (ltransfer.__doc__ or "") + _LPSD_NOTES


def lcsd_matrix(
    X: ArrayLike,
    fs: float,
    Jdes: int = 200,
    Kdes: int = 100,
    xi: float = 0.5,
    parallel: bool = True,
) -> Spectrum:
    """Full cross-spectral density matrix of a multi-channel recording.

    Parameters
    ----------
    X : array_like, shape (N, C)
        Multi-channel time series, one column per channel.
    fs : float
        Sampling frequency in hertz.
    Jdes, Kdes, xi, parallel
        See :func:`lpsd`.

    Returns
    -------
    Spectrum
        ``kind="csd"``, ``scale="log"``. ``values`` has shape
        ``(J, C, C)``: ``values[k]`` is the (Hermitian) cross-spectral
        density matrix of all channel pairs at frequency ``f[k]``.
        ``meta["n_channels"]`` records ``C``. Note that
        :meth:`Spectrum.to_dataframe` does not support this 3-D shape.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D or ``fs`` is not positive.
    ImportError
        If the Rust extension ``precision_physkit._core`` is not built.
    """
    _check_fs(fs)
    x_arr = np.asarray(X, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError(f"X must be a 2-D (N, C) array, got {x_arr.ndim}-D")
    core = _load_core()
    f, s_matrix = core.lpsd_csd_matrix(
        x_arr, fs, Jdes=Jdes, Kdes=Kdes, xi=xi, parallel=parallel
    )
    meta = _lpsd_meta(fs, Jdes, Kdes, xi)
    meta["n_channels"] = int(x_arr.shape[1])
    return Spectrum(
        f=np.asarray(f, dtype=float),
        values=np.asarray(s_matrix),
        kind="csd",
        scale="log",
        method="lpsd",
        meta=meta,
    )


lcsd_matrix.__doc__ = (lcsd_matrix.__doc__ or "") + _LPSD_NOTES
