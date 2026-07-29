//! Core of the LPSD spectral family (Tröbs & Heinzel 2005; LTPDA `iLPSD`).
//!
//! Per frequency point `f_j` with segment length `L_j`:
//! - hop `D = max(1, floor((1 - xi) * L))`, segment count `K = floor((N-L)/D) + 1`;
//! - each segment is demeaned and multiplied by a symmetric Hann window;
//! - the single-frequency DFT `A = sum_l x[l] * exp(-i*omega*l)` with
//!   `omega = 2*pi*f/fs` is evaluated with a precomputed cos/sin lookup table
//!   (twiddle table, generated once per frequency point). This replaces the
//!   rotation recursion of the legacy implementation, whose phase drifts
//!   along the segment and across segments — a hard error for cross spectra.
//!   Table lookup keeps every tap at full libm accuracy and turns the inner
//!   loop into two plain dot products;
//! - normalization `mean_k |A_k|^2 * 2 / (fs * sum(w^2))` (one-sided PSD,
//!   factor 2).
//!
//! Cross-spectral convention: with the DFT defined as
//! `A(f) = sum_l x[l] * exp(-i*omega*l)`, the cross spectrum is accumulated
//! as `Sxy = mean_k conj(A_x) * A_y`, so for `y = H * x` the H1 estimator
//! `Sxy / Sxx` recovers `H` (not its conjugate). Coherence is the
//! magnitude-squared `|Sxy|^2 / (Sxx * Syy)` in `[0, 1]`.

use num_complex::Complex64;
use rayon::prelude::*;

use crate::freqs::{get_freqs, hop, FreqPlan};
use crate::window::WindowCache;
use crate::CoreError;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// Normalized auto/cross spectra of a channel pair at one frequency point.
pub struct PairSpectrum {
    pub sxx: f64,
    pub syy: f64,
    pub sxy: Complex64,
}

/// Validate the scalar LPSD parameters.
pub fn validate_params(
    n: usize,
    fs: f64,
    jdes: usize,
    kdes: usize,
    xi: f64,
) -> Result<(), CoreError> {
    if n < 2 {
        return Err(CoreError::new(format!(
            "data length must be at least 2, got {n}"
        )));
    }
    if !fs.is_finite() || fs <= 0.0 {
        return Err(CoreError::new(format!(
            "fs must be a positive finite number, got {fs}"
        )));
    }
    if jdes < 2 {
        return Err(CoreError::new(format!(
            "Jdes must be at least 2, got {jdes}"
        )));
    }
    if kdes < 2 {
        return Err(CoreError::new(format!(
            "Kdes must be at least 2, got {kdes}"
        )));
    }
    if !xi.is_finite() || !(0.0..1.0).contains(&xi) {
        return Err(CoreError::new(format!(
            "xi must satisfy 0 <= xi < 1, got {xi}"
        )));
    }
    Ok(())
}

/// Validate channel layout: non-empty, equal lengths, no NaN/Inf.
fn validate_channels(channels: &[&[f64]]) -> Result<usize, CoreError> {
    if channels.is_empty() {
        return Err(CoreError::new("at least one data channel is required"));
    }
    let n = channels[0].len();
    for (i, ch) in channels.iter().enumerate() {
        if ch.len() != n {
            return Err(CoreError::new(format!(
                "all channels must have the same length, channel 0 has {n} but channel {i} has {}",
                ch.len()
            )));
        }
        if ch.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::new(format!("channel {i} contains NaN or Inf")));
        }
    }
    Ok(n)
}

/// Validate a frequency pair (x, y) and return the common length.
fn validate_pair(x: &[f64], y: &[f64]) -> Result<usize, CoreError> {
    if x.len() != y.len() {
        return Err(CoreError::new(format!(
            "x and y must have the same length, got {} and {}",
            x.len(),
            y.len()
        )));
    }
    validate_channels(&[x, y])
}

/// Sanity-check the plan against the data length.
fn check_plan(plan: &FreqPlan, n: usize, fs: f64) -> Result<(), CoreError> {
    if plan.f.is_empty() {
        return Err(CoreError::new(
            "frequency plan is empty: data too short for the given fs (need at least 3 samples)",
        ));
    }
    if plan.lengths.iter().any(|&l| l == 2) {
        return Err(CoreError::new(
            "frequency plan contains a segment of length L=2 for which the symmetric Hann \
             window is identically zero; increase Jdes or the data length",
        ));
    }
    let l_min_freq = plan.lengths[0]; // lengths are non-increasing: max is at f[0]
    if n < l_min_freq {
        return Err(CoreError::new(format!(
            "data length {n} is shorter than the segment length {l_min_freq} required at the \
             lowest frequency point ({:.6} Hz); provide more data or reduce Jdes/Kdes",
            plan.f[0].max(fs / n as f64)
        )));
    }
    Ok(())
}

/// Generate the twiddle lookup tables `cos(omega*l)`, `sin(omega*l)` once
/// per frequency point. Each entry is a direct libm call, so there is no
/// accumulated phase error along the segment.
fn twiddle_tables(len: usize, omega: f64) -> (Vec<f64>, Vec<f64>) {
    let mut cos_t = Vec::with_capacity(len);
    let mut sin_t = Vec::with_capacity(len);
    for i in 0..len {
        let (s, c) = (omega * i as f64).sin_cos();
        cos_t.push(c);
        sin_t.push(s);
    }
    (cos_t, sin_t)
}

/// Single-frequency DFT of one demeaned, windowed segment:
/// `A = sum_l (x[l] - mean) * w[l] * exp(-i*omega*l)`.
fn dft_segment(seg: &[f64], w: &[f64], cos_t: &[f64], sin_t: &[f64]) -> Complex64 {
    let len = seg.len();
    debug_assert_eq!(w.len(), len);
    debug_assert_eq!(cos_t.len(), len);
    debug_assert_eq!(sin_t.len(), len);
    let mean = seg.iter().sum::<f64>() / len as f64;
    let mut re = 0.0;
    let mut im = 0.0;
    for l in 0..len {
        let v = (seg[l] - mean) * w[l];
        re += v * cos_t[l];
        im -= v * sin_t[l];
    }
    Complex64::new(re, im)
}

/// Run `f(cache, j)` for all frequency indices, in parallel over rayon
/// workers (each owning a `WindowCache`) or sequentially.
fn run_freqs<T, F>(n_freqs: usize, parallel: bool, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(&mut WindowCache, usize) -> T + Sync,
{
    if parallel {
        (0..n_freqs)
            .into_par_iter()
            .map_init(WindowCache::new, |cache, j| f(cache, j))
            .collect()
    } else {
        let mut cache = WindowCache::new();
        (0..n_freqs).map(|j| f(&mut cache, j)).collect()
    }
}

/// PSD of every channel at frequency index `j`; one row of C values.
fn eval_psd_freq(
    channels: &[&[f64]],
    plan: &FreqPlan,
    j: usize,
    xi: f64,
    fs: f64,
    cache: &mut WindowCache,
) -> Vec<f64> {
    let len = plan.lengths[j];
    let k = plan.segments[j];
    let d = hop(len, xi);
    let entry = cache.get(len);
    let w = &entry.w;
    let sum_w_sq = entry.sum_w_sq;
    let omega = TWO_PI * plan.f[j] / fs;
    let (cos_t, sin_t) = twiddle_tables(len, omega);
    let norm = 2.0 / (fs * sum_w_sq * k as f64);
    channels
        .iter()
        .map(|ch| {
            let mut acc = 0.0;
            for seg in 0..k {
                let start = seg * d;
                let a = dft_segment(&ch[start..start + len], w, &cos_t, &sin_t);
                acc += a.norm_sqr();
            }
            acc * norm
        })
        .collect()
}

/// Auto/cross spectra of (x, y) at frequency index `j`.
fn eval_pair_freq(
    x: &[f64],
    y: &[f64],
    plan: &FreqPlan,
    j: usize,
    xi: f64,
    fs: f64,
    cache: &mut WindowCache,
) -> PairSpectrum {
    let len = plan.lengths[j];
    let k = plan.segments[j];
    let d = hop(len, xi);
    let entry = cache.get(len);
    let w = &entry.w;
    let sum_w_sq = entry.sum_w_sq;
    let omega = TWO_PI * plan.f[j] / fs;
    let (cos_t, sin_t) = twiddle_tables(len, omega);
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = Complex64::new(0.0, 0.0);
    for seg in 0..k {
        let start = seg * d;
        let ax = dft_segment(&x[start..start + len], w, &cos_t, &sin_t);
        let ay = dft_segment(&y[start..start + len], w, &cos_t, &sin_t);
        sxx += ax.norm_sqr();
        syy += ay.norm_sqr();
        sxy += ax.conj() * ay;
    }
    let norm = 2.0 / (fs * sum_w_sq * k as f64);
    PairSpectrum {
        sxx: sxx * norm,
        syy: syy * norm,
        sxy: sxy.scale(norm),
    }
}

/// Full CSD matrix at frequency index `j`, row-major C*C:
/// `S[a][b] = mean_k conj(A_a) * A_b`; the diagonal is the real PSD.
fn eval_matrix_freq(
    channels: &[&[f64]],
    plan: &FreqPlan,
    j: usize,
    xi: f64,
    fs: f64,
    cache: &mut WindowCache,
) -> Vec<Complex64> {
    let c = channels.len();
    let len = plan.lengths[j];
    let k = plan.segments[j];
    let d = hop(len, xi);
    let entry = cache.get(len);
    let w = &entry.w;
    let sum_w_sq = entry.sum_w_sq;
    let omega = TWO_PI * plan.f[j] / fs;
    let (cos_t, sin_t) = twiddle_tables(len, omega);
    let mut acc = vec![Complex64::new(0.0, 0.0); c * c];
    let mut a = Vec::with_capacity(c);
    for seg in 0..k {
        let start = seg * d;
        a.clear();
        for ch in channels {
            a.push(dft_segment(&ch[start..start + len], w, &cos_t, &sin_t));
        }
        for row in 0..c {
            let conj_row = a[row].conj();
            for (col, a_col) in a.iter().enumerate() {
                acc[row * c + col] += conj_row * a_col;
            }
        }
    }
    let norm = 2.0 / (fs * sum_w_sq * k as f64);
    acc.iter_mut().for_each(|v| *v = v.scale(norm));
    acc
}

/// One-sided PSD of every channel. Returns `(f, rows)` where `rows[j][c]`
/// is the PSD of channel `c` at `f[j]`.
pub fn psd_channels(
    channels: &[&[f64]],
    fs: f64,
    jdes: usize,
    kdes: usize,
    xi: f64,
    parallel: bool,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), CoreError> {
    let n = validate_channels(channels)?;
    validate_params(n, fs, jdes, kdes, xi)?;
    let plan = get_freqs(n, fs, jdes, kdes, xi);
    check_plan(&plan, n, fs)?;
    let rows = run_freqs(plan.f.len(), parallel, |cache, j| {
        eval_psd_freq(channels, &plan, j, xi, fs, cache)
    });
    Ok((plan.f, rows))
}

/// Normalized auto/cross spectra of (x, y) at all planned frequencies.
pub fn pair_spectra(
    x: &[f64],
    y: &[f64],
    fs: f64,
    jdes: usize,
    kdes: usize,
    xi: f64,
    parallel: bool,
) -> Result<(Vec<f64>, Vec<PairSpectrum>), CoreError> {
    let n = validate_pair(x, y)?;
    validate_params(n, fs, jdes, kdes, xi)?;
    let plan = get_freqs(n, fs, jdes, kdes, xi);
    check_plan(&plan, n, fs)?;
    let spectra = run_freqs(plan.f.len(), parallel, |cache, j| {
        eval_pair_freq(x, y, &plan, j, xi, fs, cache)
    });
    Ok((plan.f, spectra))
}

/// Cross power spectral density `Sxy = mean_k conj(A_x) * A_y` (one-sided).
pub fn csd_pair(
    x: &[f64],
    y: &[f64],
    fs: f64,
    jdes: usize,
    kdes: usize,
    xi: f64,
    parallel: bool,
) -> Result<(Vec<f64>, Vec<Complex64>), CoreError> {
    let (f, spectra) = pair_spectra(x, y, fs, jdes, kdes, xi, parallel)?;
    Ok((f, spectra.iter().map(|p| p.sxy).collect()))
}

/// Magnitude-squared coherence `|Sxy|^2 / (Sxx * Syy)`, in `[0, 1]`.
/// Yields NaN where `Sxx * Syy == 0` (e.g. constant input), where the
/// coherence is undefined.
pub fn coherence_pair(
    x: &[f64],
    y: &[f64],
    fs: f64,
    jdes: usize,
    kdes: usize,
    xi: f64,
    parallel: bool,
) -> Result<(Vec<f64>, Vec<f64>), CoreError> {
    let (f, spectra) = pair_spectra(x, y, fs, jdes, kdes, xi, parallel)?;
    let coh = spectra
        .iter()
        .map(|p| {
            let den = p.sxx * p.syy;
            if den > 0.0 {
                (p.sxy.norm_sqr() / den).clamp(0.0, 1.0)
            } else {
                f64::NAN
            }
        })
        .collect();
    Ok((f, coh))
}

/// H1 transfer function estimate `H = Sxy / Sxx` (complex).
/// Yields NaN where `Sxx == 0` (e.g. constant input).
pub fn transfer_pair(
    x: &[f64],
    y: &[f64],
    fs: f64,
    jdes: usize,
    kdes: usize,
    xi: f64,
    parallel: bool,
) -> Result<(Vec<f64>, Vec<Complex64>), CoreError> {
    let (f, spectra) = pair_spectra(x, y, fs, jdes, kdes, xi, parallel)?;
    let h = spectra
        .iter()
        .map(|p| {
            if p.sxx > 0.0 {
                p.sxy / p.sxx
            } else {
                Complex64::new(f64::NAN, f64::NAN)
            }
        })
        .collect();
    Ok((f, h))
}

/// Full cross-spectral matrix for C channels. Returns `(f, flat)` where
/// `flat[j*C*C + a*C + b] = S[a][b]` at `f[j]`; diagonal entries are the
/// real channel PSDs (imaginary part exactly zero).
pub fn csd_matrix(
    channels: &[&[f64]],
    fs: f64,
    jdes: usize,
    kdes: usize,
    xi: f64,
    parallel: bool,
) -> Result<(Vec<f64>, Vec<Complex64>), CoreError> {
    let n = validate_channels(channels)?;
    validate_params(n, fs, jdes, kdes, xi)?;
    let plan = get_freqs(n, fs, jdes, kdes, xi);
    check_plan(&plan, n, fs)?;
    let rows = run_freqs(plan.f.len(), parallel, |cache, j| {
        eval_matrix_freq(channels, &plan, j, xi, fs, cache)
    });
    Ok((plan.f, rows.concat()))
}

#[cfg(test)]
mod lpsd_tests {
    use super::*;
    use crate::testutil::XorShift64;

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    #[test]
    fn white_noise_psd_median_matches_theory() {
        let n = 1 << 16;
        let fs = 1024.0;
        let sigma = 1.7;
        let mut rng = XorShift64::new(0x1234_5678_9abc_def0);
        let x: Vec<f64> = (0..n).map(|_| sigma * rng.normal()).collect();

        let (_f, rows) = psd_channels(&[&x], fs, 100, 60, 0.5, true).unwrap();
        let p: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        assert!(p.iter().all(|v| v.is_finite() && *v > 0.0));
        let med = median(p);
        let expected = 2.0 * sigma * sigma / fs; // one-sided PSD of white noise
        let rel = (med - expected).abs() / expected;
        assert!(
            rel < 0.25,
            "median PSD {med} vs expected {expected} (rel err {rel:.3})"
        );
    }

    #[test]
    fn sine_tone_produces_peak_at_its_frequency() {
        let n = 1 << 15;
        let fs = 1024.0;
        let tone = 64.0;
        let mut rng = XorShift64::new(99);
        let x: Vec<f64> = (0..n)
            .map(|i| 1.5 * (TWO_PI * tone * i as f64 / fs).sin() + 0.05 * rng.normal())
            .collect();

        let (f, rows) = psd_channels(&[&x], fs, 150, 80, 0.5, false).unwrap();
        let (idx, _) = rows
            .iter()
            .enumerate()
            .max_by(|a, b| a.1[0].partial_cmp(&b.1[0]).unwrap())
            .unwrap();
        let f_peak = f[idx];
        assert!(
            (f_peak - tone).abs() / tone < 0.15,
            "peak at {f_peak} Hz, expected near {tone} Hz"
        );
    }

    #[test]
    fn coherence_of_identical_signals_is_one() {
        let n = 1 << 15;
        let fs = 1024.0;
        let mut rng = XorShift64::new(5);
        let x: Vec<f64> = (0..n).map(|_| rng.normal()).collect();

        let (_f, coh) = coherence_pair(&x, &x, fs, 120, 80, 0.5, true).unwrap();
        assert!(!coh.is_empty());
        for (j, &c) in coh.iter().enumerate() {
            assert!(
                (c - 1.0).abs() < 1e-6,
                "coherence at index {j} is {c}, expected 1"
            );
        }
    }

    #[test]
    fn coherence_of_independent_noise_is_small() {
        let n = 1 << 15;
        let fs = 1024.0;
        let mut rng_x = XorShift64::new(6);
        let mut rng_y = XorShift64::new(7);
        let x: Vec<f64> = (0..n).map(|_| rng_x.normal()).collect();
        let y: Vec<f64> = (0..n).map(|_| rng_y.normal()).collect();

        let (_f, coh) = coherence_pair(&x, &y, fs, 120, 80, 0.5, false).unwrap();
        assert!(coh.iter().all(|&c| (0.0..=1.0).contains(&c)));
        let med = median(coh);
        assert!(med < 0.3, "median coherence {med} >= 0.3");
    }

    #[test]
    fn transfer_recovers_known_fir_magnitude() {
        let n = 1 << 16;
        let fs = 1024.0;
        let mut rng = XorShift64::new(21);
        let x: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        // y = 0.5 x[n] + 0.5 x[n-1]  =>  H(f) = exp(-i*pi*f/fs) * cos(pi*f/fs)
        let mut y = vec![0.0; n];
        y[0] = 0.5 * x[0];
        for i in 1..n {
            y[i] = 0.5 * x[i] + 0.5 * x[i - 1];
        }

        let (f, h) = transfer_pair(&x, &y, fs, 150, 100, 0.5, true).unwrap();
        let plan = get_freqs(n, fs, 150, 100, 0.5);
        let mut checked = 0;
        for (j, &fj) in f.iter().enumerate() {
            // passband points with enough averaging to keep the estimate tight
            if fj < fs / 8.0 && plan.segments[j] >= 10 {
                let expected = (std::f64::consts::PI * fj / fs).cos();
                let got = h[j].norm();
                assert!(
                    (got - expected).abs() / expected < 0.10,
                    "|H({fj:.2} Hz)| = {got}, expected {expected}"
                );
                checked += 1;
            }
        }
        assert!(checked >= 3, "only {checked} passband points checked");
    }
}
