//! Python bindings for the LPSD spectral family. Signatures here are the
//! public contract of `precision_physkit._core`; the heavy computation runs in
//! `crate::lpsd` with the GIL released.

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray3};
use pyo3::prelude::*;

use crate::{convert, lpsd};

fn channel_refs(channels: &[Vec<f64>]) -> Vec<&[f64]> {
    channels.iter().map(|c| c.as_slice()).collect()
}

/// Estimate one-sided power spectral density on a logarithmic frequency axis
/// (LPSD; Tröbs & Heinzel 2005 / LTPDA `iLPSD` algorithm).
///
/// Parameters
/// ----------
/// x : array_like
///     Time series, shape (N,) or (N, C); columns are processed as
///     independent channels. A (1, N) row vector is transposed to (N,)
///     following MATLAB semantics.
/// fs : float
///     Sample rate in Hz; must be positive and finite.
/// Jdes : int, optional
///     Desired number of logarithmic frequency points (default 200). The
///     actual number of points J generally differs from Jdes by design.
/// Kdes : int, optional
///     Desired number of averaged segments per frequency (default 100).
/// xi : float, optional
///     Segment overlap fraction, 0 <= xi < 1 (default 0.5).
/// parallel : bool, optional
///     Evaluate frequency points in parallel (default True).
///
/// Returns
/// -------
/// f : ndarray, shape (J,)
///     Frequency points in Hz, strictly increasing within [fs/N, fs/2].
/// pxx : ndarray, shape (J,) or (J, C)
///     One-sided PSD estimate in unit^2/Hz. Shape (J,) for vector input,
///     (J, C) for matrix input.
///
/// Raises
/// ------
/// ValueError
///     If x contains NaN/Inf, fs <= 0, Jdes < 2, Kdes < 2, xi is not in
///     [0, 1), or N is shorter than the segment length required at the
///     lowest frequency point.
#[pyfunction(signature = (x, fs, Jdes=200, Kdes=100, xi=0.5, parallel=true))]
pub fn lpsd_psd<'py>(
    py: Python<'py>,
    x: Bound<'py, PyAny>,
    fs: f64,
    Jdes: usize,
    Kdes: usize,
    xi: f64,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, PyObject)> {
    let (channels, was_vector) = convert::to_channels(convert::to_owned_dyn(py, &x, "x")?, "x")?;
    let refs = channel_refs(&channels);
    let (f, rows) = py.allow_threads(|| lpsd::psd_channels(&refs, fs, Jdes, Kdes, xi, parallel))?;
    let py_f = Array1::from_vec(f).into_pyarray_bound(py);
    let j = rows.len();
    let py_pxx: PyObject = if was_vector {
        let col: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        Array1::from_vec(col)
            .into_pyarray_bound(py)
            .unbind()
            .into_any()
    } else {
        let c = channels.len();
        let flat: Vec<f64> = rows.concat();
        Array2::from_shape_vec((j, c), flat)
            .expect("rows and channel count are consistent by construction")
            .into_pyarray_bound(py)
            .unbind()
            .into_any()
    };
    Ok((py_f, py_pxx))
}

/// Convert a 1D signal argument with MATLAB row-vector semantics.
fn signal_arg(py: Python<'_>, obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    convert::to_vector(convert::to_owned_dyn(py, obj, name)?, name)
}

macro_rules! pair_binding {
    ($(#[$meta:meta])* $name:ident, $core:ident, $out:ty) => {
        $(#[$meta])*
        #[pyfunction(signature = (x, y, fs, Jdes=200, Kdes=100, xi=0.5, parallel=true))]
        pub fn $name<'py>(
            py: Python<'py>,
            x: Bound<'py, PyAny>,
            y: Bound<'py, PyAny>,
            fs: f64,
            Jdes: usize,
            Kdes: usize,
            xi: f64,
            parallel: bool,
        ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<$out>>)> {
            let xv = signal_arg(py, &x, "x")?;
            let yv = signal_arg(py, &y, "y")?;
            let (f, out) =
                py.allow_threads(|| lpsd::$core(&xv, &yv, fs, Jdes, Kdes, xi, parallel))?;
            Ok((
                Array1::from_vec(f).into_pyarray_bound(py),
                Array1::from_vec(out).into_pyarray_bound(py),
            ))
        }
    };
}

pair_binding!(
    /// Estimate the one-sided cross power spectral density of two signals on
    /// a logarithmic frequency axis.
    ///
    /// Parameters
    /// ----------
    /// x, y : array_like
    ///     Time series of equal length, shape (N,); a (1, N) row vector is
    ///     transposed following MATLAB semantics.
    /// fs : float
    ///     Sample rate in Hz; must be positive and finite.
    /// Jdes : int, optional
    ///     Desired number of frequency points (default 200); the actual
    ///     number J generally differs by design.
    /// Kdes : int, optional
    ///     Desired number of averaged segments per frequency (default 100).
    /// xi : float, optional
    ///     Segment overlap fraction, 0 <= xi < 1 (default 0.5).
    /// parallel : bool, optional
    ///     Evaluate frequency points in parallel (default True).
    ///
    /// Returns
    /// -------
    /// f : ndarray, shape (J,)
    ///     Frequency points in Hz.
    /// sxy : ndarray of complex128, shape (J,)
    ///     Cross spectrum Sxy = mean_k conj(A_x) * A_y in unit^2/Hz, with
    ///     A the single-frequency DFT sum x[l] * exp(-i*omega*l). For
    ///     y = H * x this gives Sxy = H * Sxx.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If x or y contains NaN/Inf, lengths differ, fs <= 0, Jdes < 2,
    ///     Kdes < 2, xi is not in [0, 1), or the data are too short for the
    ///     lowest frequency point.
    lpsd_csd, csd_pair, Complex64
);

pair_binding!(
    /// Estimate the magnitude-squared coherence of two signals on a
    /// logarithmic frequency axis: coh = |Sxy|^2 / (Sxx * Syy), in [0, 1].
    ///
    /// Parameters
    /// ----------
    /// x, y : array_like
    ///     Time series of equal length, shape (N,); a (1, N) row vector is
    ///     transposed following MATLAB semantics.
    /// fs : float
    ///     Sample rate in Hz; must be positive and finite.
    /// Jdes : int, optional
    ///     Desired number of frequency points (default 200); the actual
    ///     number J generally differs by design.
    /// Kdes : int, optional
    ///     Desired number of averaged segments per frequency (default 100).
    /// xi : float, optional
    ///     Segment overlap fraction, 0 <= xi < 1 (default 0.5).
    /// parallel : bool, optional
    ///     Evaluate frequency points in parallel (default True).
    ///
    /// Returns
    /// -------
    /// f : ndarray, shape (J,)
    ///     Frequency points in Hz.
    /// coh : ndarray, shape (J,)
    ///     Magnitude-squared coherence in [0, 1]. NaN where Sxx * Syy == 0
    ///     (e.g. constant input), where coherence is undefined.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If x or y contains NaN/Inf, lengths differ, fs <= 0, Jdes < 2,
    ///     Kdes < 2, xi is not in [0, 1), or the data are too short for the
    ///     lowest frequency point.
    lpsd_coherence, coherence_pair, f64
);

pair_binding!(
    /// Estimate the H1 frequency response H = Sxy / Sxx (complex) on a
    /// logarithmic frequency axis.
    ///
    /// Parameters
    /// ----------
    /// x, y : array_like
    ///     Input and output time series of equal length, shape (N,); a
    ///     (1, N) row vector is transposed following MATLAB semantics.
    /// fs : float
    ///     Sample rate in Hz; must be positive and finite.
    /// Jdes : int, optional
    ///     Desired number of frequency points (default 200); the actual
    ///     number J generally differs by design.
    /// Kdes : int, optional
    ///     Desired number of averaged segments per frequency (default 100).
    /// xi : float, optional
    ///     Segment overlap fraction, 0 <= xi < 1 (default 0.5).
    /// parallel : bool, optional
    ///     Evaluate frequency points in parallel (default True).
    ///
    /// Returns
    /// -------
    /// f : ndarray, shape (J,)
    ///     Frequency points in Hz.
    /// H : ndarray of complex128, shape (J,)
    ///     Complex H1 estimate Sxy / Sxx. NaN where Sxx == 0 (e.g. constant
    ///     input).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If x or y contains NaN/Inf, lengths differ, fs <= 0, Jdes < 2,
    ///     Kdes < 2, xi is not in [0, 1), or the data are too short for the
    ///     lowest frequency point.
    lpsd_transfer, transfer_pair, Complex64
);

/// Estimate the full cross-spectral matrix of a multi-channel signal on a
/// logarithmic frequency axis.
///
/// Parameters
/// ----------
/// X : array_like
///     Time series, shape (N, C) (or (N,) / (1, N) for a single channel);
///     columns are the channels.
/// fs : float
///     Sample rate in Hz; must be positive and finite.
/// Jdes : int, optional
///     Desired number of frequency points (default 200); the actual number
///     J generally differs by design.
/// Kdes : int, optional
///     Desired number of averaged segments per frequency (default 100).
/// xi : float, optional
///     Segment overlap fraction, 0 <= xi < 1 (default 0.5).
/// parallel : bool, optional
///     Evaluate frequency points in parallel (default True).
///
/// Returns
/// -------
/// f : ndarray, shape (J,)
///     Frequency points in Hz.
/// S : ndarray of complex128, shape (J, C, C)
///     Cross-spectral matrix, S[j, a, b] = mean_k conj(A_a) * A_b at f[j]
///     in unit^2/Hz. Diagonal entries are the real channel PSDs (imaginary
///     part exactly zero); S[j] is Hermitian.
///
/// Raises
/// ------
/// ValueError
///     If X contains NaN/Inf, fs <= 0, Jdes < 2, Kdes < 2, xi is not in
///     [0, 1), or the data are too short for the lowest frequency point.
#[pyfunction(signature = (X, fs, Jdes=200, Kdes=100, xi=0.5, parallel=true))]
pub fn lpsd_csd_matrix<'py>(
    py: Python<'py>,
    X: Bound<'py, PyAny>,
    fs: f64,
    Jdes: usize,
    Kdes: usize,
    xi: f64,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray3<Complex64>>)> {
    let (channels, _) = convert::to_channels(convert::to_owned_dyn(py, &X, "X")?, "X")?;
    let c = channels.len();
    let refs = channel_refs(&channels);
    let (f, flat) = py.allow_threads(|| lpsd::csd_matrix(&refs, fs, Jdes, Kdes, xi, parallel))?;
    let j = f.len();
    let s = Array3::from_shape_vec((j, c, c), flat).expect("flat layout is J*C*C by construction");
    Ok((
        Array1::from_vec(f).into_pyarray_bound(py),
        s.into_pyarray_bound(py),
    ))
}
