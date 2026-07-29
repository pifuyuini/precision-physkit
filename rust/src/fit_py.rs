//! Python bindings for the fitting kernels. Signatures here are the public
//! contract of `precision_physkit._core`; computation runs in `crate::fit` with the
//! GIL released.

use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::{convert, fit};

fn dvec_to_vec(v: &DVector<f64>) -> Vec<f64> {
    v.iter().copied().collect()
}

fn dmat_to_row_major(m: &DMatrix<f64>) -> Vec<f64> {
    let (r, c) = (m.nrows(), m.ncols());
    let mut out = Vec::with_capacity(r * c);
    for i in 0..r {
        for j in 0..c {
            out.push(m[(i, j)]);
        }
    }
    out
}

fn design_args(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
) -> PyResult<(usize, usize, Vec<f64>, Vec<f64>)> {
    let (n, m, a_data) = convert::to_matrix(convert::to_owned_dyn(py, a, "A")?, "A")?;
    let y_data = convert::to_vector(convert::to_owned_dyn(py, y, "y")?, "y")?;
    Ok((n, m, a_data, y_data))
}

/// Linear least squares via SVD with singular-value truncation.
///
/// Solves min_x ||A x - y||_2 as x = V S^+ U^T y, keeping only singular
/// values s_i > rcond * s_max.
///
/// Parameters
/// ----------
/// A : array_like, shape (N, M)
///     Design matrix.
/// y : array_like, shape (N,)
///     Observation vector.
/// rcond : float, optional
///     Relative cutoff for small singular values. Default is the
///     machine-precision-based value max(N, M) * eps.
///
/// Returns
/// -------
/// x : ndarray, shape (M,)
///     Least-squares solution.
/// resid : ndarray, shape (N,)
///     Residual vector y - A @ x.
/// rank : int
///     Numerical rank: number of singular values above the cutoff.
/// cov : ndarray, shape (M, M)
///     Covariance estimate sigma^2 * (A^T A)^+ (truncated pseudo-inverse),
///     with sigma^2 = ||resid||^2 / (N - rank), or 0 if N <= rank.
///
/// Raises
/// ------
/// ValueError
///     If A or y contains NaN/Inf, shapes are inconsistent, or rcond is not
///     positive and finite.
#[pyfunction(signature = (A, y, rcond=None))]
pub fn svd_lstsq<'py>(
    py: Python<'py>,
    A: Bound<'py, PyAny>,
    y: Bound<'py, PyAny>,
    rcond: Option<f64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    Bound<'py, PyArray2<f64>>,
)> {
    let (n, m, a_data, y_data) = design_args(py, &A, &y)?;
    let res = py.allow_threads(|| {
        let a = DMatrix::from_row_slice(n, m, &a_data);
        let yv = DVector::from_vec(y_data);
        fit::svd_lstsq_core(&a, &yv, rcond)
    })?;
    let cov = dmat_to_row_major(&res.cov);
    Ok((
        Array1::from_vec(dvec_to_vec(&res.theta)).into_pyarray_bound(py),
        Array1::from_vec(dvec_to_vec(&res.resid)).into_pyarray_bound(py),
        res.rank,
        Array2::from_shape_vec((m, m), cov)
            .expect("cov is M*M by construction")
            .into_pyarray_bound(py),
    ))
}

/// Iterative weighted least squares for a multi-channel linear system.
///
/// The rows of A (N, M) and y (N,) are partitioned into G consecutive
/// channel blocks of lengths group_sizes (which must sum to N). Starting
/// from uniform weights, the iteration is:
///
/// 1. weighted least squares theta = argmin sum_g w_g ||A_g theta - y_g||^2,
///    solved by scaling the rows of block g with sqrt(w_g) and applying a
///    truncated-SVD solve (cutoff max(N, M) * eps * s_max);
/// 2. per-channel noise variance from the residuals of the unscaled model:
///    sigma_g^2 = ||y_g - A_g theta||^2 / n_g (floored away from zero);
/// 3. weights w_g = 1 / sigma_g^2, normalized to unit mean (the common
///    factor does not change theta, it only keeps the system scaled);
/// 4. stop when ||theta_new - theta_old|| < tol * ||theta_old||
///    (converged = True) or when max_iter iterations are exhausted.
///
/// Parameters
/// ----------
/// A : array_like, shape (N, M)
///     Design matrix; rows are grouped into channels by group_sizes.
/// y : array_like, shape (N,)
///     Observation vector.
/// group_sizes : sequence of int
///     Number of rows per channel; must be non-empty, all >= 1, and sum
///     to N.
/// max_iter : int, optional
///     Maximum number of iterations (default 100).
/// tol : float, optional
///     Relative parameter-change convergence tolerance (default 1e-12).
///
/// Returns
/// -------
/// theta : ndarray, shape (M,)
///     Final parameter estimate.
/// weights : ndarray, shape (G,)
///     Final per-channel weights, normalized to unit mean; inversely
///     proportional to the estimated channel noise variances.
/// n_iter : int
///     Number of iterations actually performed.
/// converged : bool
///     Whether the relative parameter change fell below tol.
/// cov : ndarray, shape (M, M)
///     (A^T W A)^+ from the final weighted solve (truncated pseudo-inverse
///     with the normalized weights; absolute covariance requires rescaling
///     by the true common noise factor).
///
/// Raises
/// ------
/// ValueError
///     If A or y contains NaN/Inf, group_sizes is empty, contains zeros or
///     does not sum to N, max_iter < 1, or tol is not positive and finite.
#[pyfunction(signature = (A, y, group_sizes, max_iter=100, tol=1e-12))]
pub fn iterative_wls<'py>(
    py: Python<'py>,
    A: Bound<'py, PyAny>,
    y: Bound<'py, PyAny>,
    group_sizes: Vec<usize>,
    max_iter: usize,
    tol: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    bool,
    Bound<'py, PyArray2<f64>>,
)> {
    let (n, m, a_data, y_data) = design_args(py, &A, &y)?;
    let res = py.allow_threads(|| {
        let a = DMatrix::from_row_slice(n, m, &a_data);
        let yv = DVector::from_vec(y_data);
        fit::iterative_wls_core(&a, &yv, &group_sizes, max_iter, tol)
    })?;
    let cov = dmat_to_row_major(&res.cov);
    Ok((
        Array1::from_vec(dvec_to_vec(&res.theta)).into_pyarray_bound(py),
        Array1::from_vec(res.weights).into_pyarray_bound(py),
        res.n_iter,
        res.converged,
        Array2::from_shape_vec((m, m), cov)
            .expect("cov is M*M by construction")
            .into_pyarray_bound(py),
    ))
}
