//! precision_physkit high-performance Rust core, exposed to Python as `precision_physkit._core`.
//!
//! Contents
//! --------
//! - LPSD family: power spectral density, cross-spectral density,
//!   magnitude-squared coherence, H1 transfer function and full CSD matrix
//!   on a logarithmic frequency axis (Tröbs & Heinzel 2005 / LTPDA `iLPSD`).
//! - Fitting kernels: truncated-SVD least squares and iterative weighted
//!   least squares for multi-channel systems.
//!
//! All heavy computation runs with the GIL released and is parallelized over
//! frequency points with rayon.

#![forbid(unsafe_code)]
#![allow(non_snake_case)] // Python API uses MATLAB-style names (Jdes, Kdes, A, ...)

mod convert;
mod fit;
mod fit_py;
mod freqs;
mod lpsd;
mod lpsd_py;
mod window;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod testutil;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// Error type shared by the computational cores.
///
/// Carries a human-readable message; mapped to Python `ValueError` at the
/// binding boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreError(pub String);

impl CoreError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl std::fmt::Display for CoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CoreError {}

impl From<CoreError> for PyErr {
    fn from(err: CoreError) -> PyErr {
        PyValueError::new_err(err.0)
    }
}

/// precision_physkit._core — Rust kernels for spectral estimation and fitting.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lpsd_py::lpsd_psd, m)?)?;
    m.add_function(wrap_pyfunction!(lpsd_py::lpsd_csd, m)?)?;
    m.add_function(wrap_pyfunction!(lpsd_py::lpsd_coherence, m)?)?;
    m.add_function(wrap_pyfunction!(lpsd_py::lpsd_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(lpsd_py::lpsd_csd_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(fit_py::svd_lstsq, m)?)?;
    m.add_function(wrap_pyfunction!(fit_py::iterative_wls, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
