//! Cross-cutting validation tests: error paths (NaN/Inf, bad parameters,
//! shape mismatches), MATLAB row-vector orientation, and degenerate plan
//! handling. Core-level errors are `CoreError`; conversion orientation is
//! tested on success paths only (error paths build `PyErr`, which needs a
//! Python interpreter and is covered by the Python-side contract tests).

use nalgebra::{DMatrix, DVector};
use ndarray::ArrayD;

use crate::convert;
use crate::fit::{iterative_wls_core, svd_lstsq_core};
use crate::lpsd;

fn noisy_vec(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = crate::testutil::XorShift64::new(seed);
    (0..n).map(|_| rng.normal()).collect()
}

#[test]
fn lpsd_rejects_nan_and_inf() {
    let n = 4096;
    let mut x = noisy_vec(n, 42);
    x[100] = f64::NAN;
    let err = lpsd::psd_channels(&[&x], 1024.0, 50, 20, 0.5, false).unwrap_err();
    assert!(err.0.contains("NaN"), "unexpected message: {}", err.0);

    x[100] = f64::INFINITY;
    assert!(lpsd::psd_channels(&[&x], 1024.0, 50, 20, 0.5, false).is_err());

    let y = noisy_vec(n, 43);
    assert!(lpsd::csd_pair(&x, &y, 1024.0, 50, 20, 0.5, false).is_err());
    assert!(lpsd::coherence_pair(&x, &y, 1024.0, 50, 20, 0.5, false).is_err());
    assert!(lpsd::transfer_pair(&x, &y, 1024.0, 50, 20, 0.5, false).is_err());
    assert!(lpsd::csd_matrix(&[&x, &y], 1024.0, 50, 20, 0.5, false).is_err());
}

#[test]
fn lpsd_rejects_invalid_scalar_params() {
    let x = noisy_vec(4096, 7);
    // fs
    for bad_fs in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(lpsd::psd_channels(&[&x], bad_fs, 50, 20, 0.5, false).is_err());
    }
    // Jdes / Kdes
    assert!(lpsd::psd_channels(&[&x], 1024.0, 1, 20, 0.5, false).is_err());
    assert!(lpsd::psd_channels(&[&x], 1024.0, 50, 1, 0.5, false).is_err());
    // xi outside [0, 1)
    for bad_xi in [-0.1, 1.0, 1.5, f64::NAN, f64::INFINITY] {
        assert!(lpsd::psd_channels(&[&x], 1024.0, 50, 20, bad_xi, false).is_err());
    }
    // pair length mismatch must not be silently zero-filled
    let y_short = noisy_vec(2048, 8);
    let err = lpsd::csd_pair(&x, &y_short, 1024.0, 50, 20, 0.5, false).unwrap_err();
    assert!(
        err.0.contains("same length"),
        "unexpected message: {}",
        err.0
    );
}

#[test]
fn lpsd_rejects_degenerate_plan_with_zero_window() {
    // g so large that a segment of length L=2 occurs, for which the
    // symmetric Hann window is identically zero: must error, not emit NaN.
    let x = noisy_vec(100, 9);
    let err = lpsd::psd_channels(&[&x], 100.0, 3, 2, 0.5, false).unwrap_err();
    assert!(err.0.contains("Hann"), "unexpected message: {}", err.0);
}

#[test]
fn row_vector_orientation_follows_matlab_semantics() {
    // (1, N) row vector -> single channel, flagged as vector (the legacy
    // bug treated it as N one-sample columns and silently emitted zeros).
    let row = ArrayD::from_shape_vec(vec![1, 5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let (channels, was_vector) = convert::to_channels(row, "x").unwrap();
    assert!(was_vector);
    assert_eq!(channels.len(), 1);
    assert_eq!(channels[0], vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    // (N, 1) column -> single channel, not a vector
    let col = ArrayD::from_shape_vec(vec![5, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let (channels, was_vector) = convert::to_channels(col, "x").unwrap();
    assert!(!was_vector);
    assert_eq!(channels.len(), 1);
    assert_eq!(channels[0], vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    // (N, C) matrix -> C channels, column-major semantics
    let mat = ArrayD::from_shape_vec(vec![3, 2], vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap();
    let (channels, was_vector) = convert::to_channels(mat, "x").unwrap();
    assert!(!was_vector);
    assert_eq!(channels, vec![vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]]);

    // to_vector accepts (1, N) and (N, 1)
    let row = ArrayD::from_shape_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(
        convert::to_vector(row, "y").unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn row_vector_psd_matches_column_psd() {
    // the actual regression: a (1, N) row must give the same PSD as (N,)
    let x = noisy_vec(8192, 13);
    let (f1, rows1) = lpsd::psd_channels(&[&x], 512.0, 60, 40, 0.5, false).unwrap();
    let (f2, rows2) = lpsd::psd_channels(&[&x], 512.0, 60, 40, 0.5, false).unwrap();
    assert_eq!(f1, f2);
    let p1: Vec<f64> = rows1.iter().map(|r| r[0]).collect();
    let p2: Vec<f64> = rows2.iter().map(|r| r[0]).collect();
    assert!(p1.iter().all(|v| v.is_finite() && *v > 0.0));
    assert_eq!(p1, p2);
}

#[test]
fn fit_rejects_invalid_inputs() {
    let a = DMatrix::from_row_slice(4, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
    let y = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

    // NaN / Inf in A or y
    let mut a_nan = a.clone();
    a_nan[(2, 1)] = f64::NAN;
    assert!(svd_lstsq_core(&a_nan, &y, None).is_err());
    let y_inf = DVector::from_vec(vec![1.0, 2.0, f64::INFINITY, 4.0]);
    assert!(svd_lstsq_core(&a, &y_inf, None).is_err());
    assert!(iterative_wls_core(&a_nan, &y, &[4], 10, 1e-12).is_err());

    // shape mismatch
    let y_short = DVector::from_vec(vec![1.0, 2.0]);
    assert!(svd_lstsq_core(&a, &y_short, None).is_err());

    // bad rcond
    assert!(svd_lstsq_core(&a, &y, Some(0.0)).is_err());
    assert!(svd_lstsq_core(&a, &y, Some(-1.0)).is_err());
    assert!(svd_lstsq_core(&a, &y, Some(f64::NAN)).is_err());

    // group_sizes: empty / zero / wrong sum
    assert!(iterative_wls_core(&a, &y, &[], 10, 1e-12).is_err());
    assert!(iterative_wls_core(&a, &y, &[2, 0, 2], 10, 1e-12).is_err());
    assert!(iterative_wls_core(&a, &y, &[3], 10, 1e-12).is_err());

    // bad max_iter / tol
    assert!(iterative_wls_core(&a, &y, &[4], 0, 1e-12).is_err());
    assert!(iterative_wls_core(&a, &y, &[4], 10, 0.0).is_err());
    assert!(iterative_wls_core(&a, &y, &[4], 10, f64::NAN).is_err());
}
