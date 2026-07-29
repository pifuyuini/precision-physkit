//! Fitting kernels: truncated-SVD least squares and iterative weighted
//! least squares for multi-channel systems (nalgebra-backed).

use nalgebra::{DMatrix, DVector};

use crate::CoreError;

/// Result of [`svd_lstsq_core`].
pub struct LstsqResult {
    /// Least-squares solution, length M.
    pub theta: DVector<f64>,
    /// Residual vector `y - A * theta`, length N.
    pub resid: DVector<f64>,
    /// Numerical rank: number of singular values above the cutoff.
    pub rank: usize,
    /// Covariance estimate `sigma^2 * (A^T A)^+` (truncated pseudo-inverse),
    /// with `sigma^2 = ||resid||^2 / (N - rank)` (0 if `N <= rank`).
    pub cov: DMatrix<f64>,
}

/// Result of [`iterative_wls_core`].
pub struct WlsResult {
    /// Final parameter estimate, length M.
    pub theta: DVector<f64>,
    /// Final per-channel weights, normalized to unit mean, length G.
    pub weights: Vec<f64>,
    /// Number of WLS iterations actually performed.
    pub n_iter: usize,
    /// Whether the relative parameter change fell below `tol`.
    pub converged: bool,
    /// `(A^T W A)^+` from the final iteration (truncated pseudo-inverse
    /// via the same SVD solve; W holds the normalized final weights).
    pub cov: DMatrix<f64>,
}

fn default_rcond(n: usize, m: usize) -> f64 {
    n.max(m) as f64 * f64::EPSILON
}

fn validate_design(a: &DMatrix<f64>, y: &DVector<f64>) -> Result<(), CoreError> {
    let (n, m) = (a.nrows(), a.ncols());
    if n == 0 || m == 0 {
        return Err(CoreError::new(format!(
            "design matrix A must be non-empty, got shape ({n}, {m})"
        )));
    }
    if y.len() != n {
        return Err(CoreError::new(format!(
            "y length {} does not match the {} rows of A",
            y.len(),
            n
        )));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::new("design matrix A contains NaN or Inf"));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::new("observation vector y contains NaN or Inf"));
    }
    Ok(())
}

fn validate_rcond(rcond: Option<f64>) -> Result<(), CoreError> {
    if let Some(rc) = rcond {
        if !rc.is_finite() || rc <= 0.0 {
            return Err(CoreError::new(format!(
                "rcond must be positive and finite, got {rc}"
            )));
        }
    }
    Ok(())
}

/// SVD solution with singular-value truncation at `rcond * sigma_max`.
struct TruncatedSolve {
    theta: DVector<f64>,
    rank: usize,
    /// `V * S^{-2} * V^T` over the kept singular values, i.e. `(A^T A)^+`.
    cov_unscaled: DMatrix<f64>,
}

fn solve_truncated(a: &DMatrix<f64>, y: &DVector<f64>, rcond: f64) -> TruncatedSolve {
    let m = a.ncols();
    let svd = a.clone().svd(true, true);
    let u = svd.u.expect("U requested from SVD");
    let v_t = svd.v_t.expect("V^T requested from SVD");
    let s = svd.singular_values;
    let smax = s.iter().copied().fold(0.0f64, f64::max);
    let thresh = rcond * smax;
    let uty = u.transpose() * y;

    let mut theta = DVector::zeros(m);
    let mut cov = DMatrix::zeros(m, m);
    let mut rank = 0usize;
    for i in 0..s.len() {
        if s[i] > thresh {
            rank += 1;
            let v_i = v_t.row(i).transpose().into_owned();
            theta.axpy(uty[i] / s[i], &v_i, 1.0);
            cov.ger(1.0 / (s[i] * s[i]), &v_i, &v_i, 1.0);
        }
    }
    TruncatedSolve {
        theta,
        rank,
        cov_unscaled: cov,
    }
}

/// Linear least squares via SVD with singular-value truncation.
///
/// Solves `min_theta ||A theta - y||_2` as `theta = V S^+ U^T y`, keeping
/// only singular values `s_i > rcond * s_max`. `rcond = None` selects the
/// machine-precision-based default `max(N, M) * eps`.
pub fn svd_lstsq_core(
    a: &DMatrix<f64>,
    y: &DVector<f64>,
    rcond: Option<f64>,
) -> Result<LstsqResult, CoreError> {
    validate_design(a, y)?;
    validate_rcond(rcond)?;
    let (n, m) = (a.nrows(), a.ncols());
    let rcond = rcond.unwrap_or_else(|| default_rcond(n, m));

    let sol = solve_truncated(a, y, rcond);
    let resid = y - a * &sol.theta;
    let dof = n.saturating_sub(sol.rank);
    let sigma2 = if dof > 0 {
        resid.norm_squared() / dof as f64
    } else {
        0.0
    };
    let cov = sol.cov_unscaled * sigma2;
    Ok(LstsqResult {
        theta: sol.theta,
        resid,
        rank: sol.rank,
        cov,
    })
}

/// Iterative weighted least squares for a multi-channel system.
///
/// The rows of the design matrix `A` (N, M) and of `y` (N,) are partitioned
/// into G consecutive channel blocks of lengths `group_sizes` (summing to N).
/// The iteration is:
///
/// 1. weighted least squares `theta = argmin sum_g w_g ||A_g theta - y_g||^2`,
///    solved by row-scaling block `g` with `sqrt(w_g)` and applying the
///    truncated SVD solve (default rcond);
/// 2. per-channel noise variance from the residuals of the *unscaled* model:
///    `sigma_g^2 = ||y_g - A_g theta||^2 / n_g` (floored away from zero);
/// 3. weights `w_g = 1 / sigma_g^2`, normalized to unit mean (the common
///    factor does not change `theta` but keeps the system scaled);
/// 4. stop when `||theta_new - theta_old|| < tol * ||theta_old||`
///    (converged) or after `max_iter` iterations.
///
/// Weights start at `w_g = 1`, so the first iteration is ordinary least
/// squares. Returns the final `theta`, the final normalized weights, the
/// iteration count, the convergence flag, and `cov = (A^T W A)^+` from the
/// final weighted solve (with the normalized weights; absolute covariance
/// requires rescaling by the true common noise factor).
pub fn iterative_wls_core(
    a: &DMatrix<f64>,
    y: &DVector<f64>,
    group_sizes: &[usize],
    max_iter: usize,
    tol: f64,
) -> Result<WlsResult, CoreError> {
    validate_design(a, y)?;
    let (n, m) = (a.nrows(), a.ncols());
    if group_sizes.is_empty() {
        return Err(CoreError::new("group_sizes must not be empty"));
    }
    if group_sizes.iter().any(|&s| s == 0) {
        return Err(CoreError::new("group sizes must all be at least 1"));
    }
    let total: usize = group_sizes.iter().sum();
    if total != n {
        return Err(CoreError::new(format!(
            "group_sizes sum to {total} but A has {n} rows"
        )));
    }
    if max_iter == 0 {
        return Err(CoreError::new("max_iter must be at least 1"));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(CoreError::new(format!(
            "tol must be positive and finite, got {tol}"
        )));
    }

    let g = group_sizes.len();
    let mut bounds = Vec::with_capacity(g);
    let mut start = 0usize;
    for &len in group_sizes {
        bounds.push((start, len));
        start += len;
    }

    let rcond = default_rcond(n, m);
    let mut weights: Vec<f64> = vec![1.0; g];
    let mut theta = DVector::zeros(m);
    let mut cov = DMatrix::zeros(m, m);
    let mut n_iter = 0usize;
    let mut converged = false;

    for it in 1..=max_iter {
        n_iter = it;
        // 1. weighted least squares via row scaling
        let mut a_w = a.clone();
        let mut y_w = y.clone();
        for (gi, &(off, len)) in bounds.iter().enumerate() {
            let sc = weights[gi].sqrt();
            for r in off..off + len {
                for c in 0..m {
                    a_w[(r, c)] *= sc;
                }
                y_w[r] *= sc;
            }
        }
        let sol = solve_truncated(&a_w, &y_w, rcond);
        let theta_new = sol.theta;
        cov = sol.cov_unscaled;

        // 2./3. per-channel residual variance -> normalized inverse-variance weights
        let mut inv = vec![0.0; g];
        for (gi, &(off, len)) in bounds.iter().enumerate() {
            let resid = y.rows(off, len) - a.rows(off, len) * &theta_new;
            let mut s2 = resid.norm_squared() / len as f64;
            if s2 == 0.0 {
                s2 = f64::MIN_POSITIVE;
            }
            inv[gi] = 1.0 / s2;
        }
        let mean_inv = inv.iter().sum::<f64>() / g as f64;
        for gi in 0..g {
            weights[gi] = inv[gi] / mean_inv;
        }

        // 4. convergence on the relative parameter change
        let diff = (&theta_new - &theta).norm();
        let scale = theta.norm().max(1e-300);
        theta = theta_new;
        if it > 1 && diff / scale < tol {
            converged = true;
            break;
        }
    }

    Ok(WlsResult {
        theta,
        weights,
        n_iter,
        converged,
        cov,
    })
}

#[cfg(test)]
mod fit_tests {
    use super::*;
    use crate::testutil::XorShift64;

    fn poly_design(t: &[f64]) -> DMatrix<f64> {
        let n = t.len();
        let mut a = DMatrix::zeros(n, 3);
        for (i, &ti) in t.iter().enumerate() {
            a[(i, 0)] = 1.0;
            a[(i, 1)] = ti;
            a[(i, 2)] = ti * ti;
        }
        a
    }

    #[test]
    fn svd_lstsq_recovers_known_solution_and_rank() {
        let n = 200;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let a = poly_design(&t);
        let theta_true = [1.5, -2.0, 0.5];
        let mut rng = XorShift64::new(3);
        let y = DVector::from_fn(n, |i, _| {
            theta_true[0] + theta_true[1] * t[i] + theta_true[2] * t[i] * t[i] + 1e-4 * rng.normal()
        });

        let res = svd_lstsq_core(&a, &y, None).unwrap();
        assert_eq!(res.rank, 3);
        for i in 0..3 {
            assert!(
                (res.theta[i] - theta_true[i]).abs() < 1e-2,
                "theta[{i}] = {}, expected {}",
                res.theta[i],
                theta_true[i]
            );
        }
        // residual vector consistent with y - A*theta
        let resid_ref = &y - &a * &res.theta;
        assert!((&res.resid - &resid_ref).norm() < 1e-12);
        // covariance symmetric with positive diagonal
        for i in 0..3 {
            assert!(res.cov[(i, i)] > 0.0);
            for j in 0..3 {
                assert!((res.cov[(i, j)] - res.cov[(j, i)]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn svd_lstsq_detects_rank_deficiency() {
        let n = 100;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        // column 2 = 2*col0 + 3*col1  =>  numerical rank 2
        let mut a = DMatrix::zeros(n, 3);
        for (i, &ti) in t.iter().enumerate() {
            a[(i, 0)] = 1.0;
            a[(i, 1)] = ti;
            a[(i, 2)] = 2.0 + 3.0 * ti;
        }
        let y = DVector::from_fn(n, |i, _| 1.0 + 2.0 * t[i]);
        let res = svd_lstsq_core(&a, &y, None).unwrap();
        assert_eq!(res.rank, 2);
    }

    #[test]
    fn iterative_wls_recovers_theta_and_weights_reflect_noise_ratio() {
        let n1 = 256;
        let n2 = 256;
        let n = n1 + n2;
        let theta_true = [2.0, -0.5];
        let mut rng = XorShift64::new(11);

        // channel 1: sigma = 0.01, channel 2: sigma = 1.0
        let mut a = DMatrix::zeros(n, 2);
        let mut y = DVector::zeros(n);
        for i in 0..n1 {
            let t = i as f64 / n1 as f64;
            a[(i, 0)] = 1.0;
            a[(i, 1)] = t;
            y[i] = theta_true[0] + theta_true[1] * t + 0.01 * rng.normal();
        }
        for i in 0..n2 {
            let t = i as f64 / n2 as f64;
            a[(n1 + i, 0)] = 1.0;
            a[(n1 + i, 1)] = t;
            y[n1 + i] = theta_true[0] + theta_true[1] * t + 1.0 * rng.normal();
        }

        let res = iterative_wls_core(&a, &y, &[n1, n2], 100, 1e-10).unwrap();
        assert!(res.converged, "iteration did not converge");
        for i in 0..2 {
            assert!(
                (res.theta[i] - theta_true[i]).abs() < 0.02,
                "theta[{i}] = {}, expected {}",
                res.theta[i],
                theta_true[i]
            );
        }
        let ratio = res.weights[0] / res.weights[1];
        assert!(
            ratio > 1e3,
            "weight ratio {ratio} should reflect (sigma2/sigma1)^2 = 1e4"
        );
        let wsum: f64 = res.weights.iter().sum();
        assert!((wsum - 2.0).abs() < 1e-9, "weights normalized to unit mean");
    }
}
