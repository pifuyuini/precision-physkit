//! Shared conversion helpers between Python array-likes and owned Rust
//! containers. Anything accepted is first routed through
//! `numpy.ascontiguousarray(..., dtype=float64)`, so lists, non-contiguous
//! views and integer arrays all work.

use ndarray::{ArrayD, Ix2};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::CoreError;

/// Convert an arbitrary array-like to an owned C-contiguous float64 array.
pub fn to_owned_dyn(py: Python<'_>, obj: &Bound<'_, PyAny>, name: &str) -> PyResult<ArrayD<f64>> {
    let numpy = py.import_bound("numpy")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", "float64")?;
    let arr_obj = numpy
        .getattr("ascontiguousarray")?
        .call((obj,), Some(&kwargs))?;
    let array = arr_obj
        .downcast::<PyArrayDyn<f64>>()
        .map_err(|_| CoreError::new(format!("{name} could not be converted to a float64 array")))?;
    let readonly = array.readonly();
    Ok(readonly.as_array().to_owned())
}

/// Interpret an array as a single data column with MATLAB semantics:
/// `(N,)` -> N samples; `(1, N)` row vector -> transposed to N samples;
/// `(N, 1)` -> N samples. Anything else is rejected.
pub fn to_vector(arr: ArrayD<f64>, name: &str) -> PyResult<Vec<f64>> {
    match arr.ndim() {
        1 => Ok(arr.iter().copied().collect()),
        2 => {
            let shape = arr.shape().to_vec();
            let (r, c) = (shape[0], shape[1]);
            if r == 1 || c == 1 {
                Ok(arr.iter().copied().collect())
            } else {
                Err(CoreError::new(format!(
                    "{name} must be one-dimensional, got shape ({r}, {c})"
                ))
                .into())
            }
        }
        d => Err(CoreError::new(format!(
            "{name} must be a 1D or 2D array, got {d} dimensions"
        ))
        .into()),
    }
}

/// Interpret an array as a set of column channels with MATLAB semantics.
/// Returns `(channels, was_vector)`; `was_vector` is true for `(N,)` input
/// and for a `(1, N)` row vector (transposed to a single channel), false
/// for a genuine `(N, C)` matrix.
pub fn to_channels(arr: ArrayD<f64>, name: &str) -> PyResult<(Vec<Vec<f64>>, bool)> {
    match arr.ndim() {
        1 => Ok((vec![arr.iter().copied().collect()], true)),
        2 => {
            let mat = arr
                .into_dimensionality::<Ix2>()
                .map_err(|_| CoreError::new(format!("{name} must be a 1D or 2D array")))?;
            let (r, c) = (mat.nrows(), mat.ncols());
            if r == 1 && c != 1 {
                // MATLAB row-vector semantics: transpose (1, N) to (N, 1)
                Ok((vec![mat.iter().copied().collect()], true))
            } else {
                let channels = (0..c)
                    .map(|ci| mat.column(ci).iter().copied().collect::<Vec<f64>>())
                    .collect();
                Ok((channels, false))
            }
        }
        d => Err(CoreError::new(format!(
            "{name} must be a 1D or 2D array, got {d} dimensions"
        ))
        .into()),
    }
}

/// Interpret an array as a dense row-major matrix `(nrows, ncols, data)`.
pub fn to_matrix(arr: ArrayD<f64>, name: &str) -> PyResult<(usize, usize, Vec<f64>)> {
    match arr.ndim() {
        2 => {
            let shape = arr.shape().to_vec();
            let (r, c) = (shape[0], shape[1]);
            Ok((r, c, arr.iter().copied().collect()))
        }
        d => Err(CoreError::new(format!("{name} must be a 2D matrix, got {d} dimensions")).into()),
    }
}
