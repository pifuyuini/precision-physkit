//! Logarithmic frequency planning for the LPSD algorithm
//! (Tröbs & Heinzel 2005; LTPDA / iLPSD `getFreqs`).
//!
//! The plan starts at `fmin = fs/N`, ends at `fmax = fs/2`, and steps with a
//! resolution `r_j` that blends the geometric spacing `f_j * g` with the
//! averaging-imposed resolution `r_avg` via `sqrt(r0 * r_avg)`, floored at
//! `fmin`. The resolution is quantized through the segment length
//! `L = floor(fs/r)` and recomputed as `r = fs/L`. Because of the blending
//! and quantization, the actual number of points `J` generally differs from
//! `Jdes` — this is by design.

/// Frequency plan for one LPSD run.
#[derive(Debug, Clone)]
pub struct FreqPlan {
    /// Evaluation frequencies in Hz, strictly increasing.
    pub f: Vec<f64>,
    /// Segment length `L_j` per frequency point.
    pub lengths: Vec<usize>,
    /// Number of averaged segments `K_j` per frequency point.
    pub segments: Vec<usize>,
}

/// Segment hop size `D = max(1, floor((1 - xi) * L))`.
pub fn hop(len: usize, xi: f64) -> usize {
    let d = ((1.0 - xi) * len as f64).floor() as isize;
    if d < 1 {
        1
    } else {
        d as usize
    }
}

/// Build the logarithmic frequency plan for `n` samples at rate `fs`.
///
/// Assumes `n >= 2` and validated `fs/Jdes/Kdes/xi`; invalid plans are
/// caught by the caller's parameter validation and plan checks.
pub fn get_freqs(n: usize, fs: f64, jdes: usize, kdes: usize, xi: f64) -> FreqPlan {
    let fmin = fs / n as f64;
    let fmax = fs / 2.0;
    let r_avg = fmin * (1.0 + (1.0 - xi) * (kdes as f64 - 1.0));
    let g = (n as f64 / 2.0).powf(1.0 / (jdes as f64 - 1.0)) - 1.0;

    let mut f = Vec::with_capacity(jdes);
    let mut lengths = Vec::with_capacity(jdes);
    let mut segments = Vec::with_capacity(jdes);

    let mut fj = fmin;
    while fj < fmax {
        let mut rj = fj * g;
        if rj < r_avg {
            rj = (rj * r_avg).sqrt();
        }
        if rj < fmin {
            rj = fmin;
        }
        let mut lj = (fs / rj).floor() as usize;
        if lj == 0 {
            lj = 1;
        }
        rj = fs / lj as f64;

        f.push(fj);
        lengths.push(lj);
        let d = hop(lj, xi);
        segments.push(n.saturating_sub(lj) / d + 1);

        fj += rj;
    }

    FreqPlan {
        f,
        lengths,
        segments,
    }
}

#[cfg(test)]
mod freq_tests {
    use super::*;

    #[test]
    fn freq_plan_strictly_increasing_and_bounded() {
        let cases: &[(usize, f64, usize, usize, f64)] = &[
            (100_000, 1000.0, 200, 100, 0.5),
            (4096, 250.0, 50, 10, 0.0),
            (4096, 250.0, 50, 10, 0.9),
            (65_536, 1024.0, 300, 2, 0.25),
        ];
        for &(n, fs, jdes, kdes, xi) in cases {
            let plan = get_freqs(n, fs, jdes, kdes, xi);
            assert!(!plan.f.is_empty(), "plan must not be empty: {cases:?}");
            let fmin = fs / n as f64;
            assert!(
                plan.f[0] >= fmin - 1e-12,
                "f[0]={} must be >= fs/N={fmin}",
                plan.f[0]
            );
            assert!(
                plan.f.windows(2).all(|w| w[1] > w[0]),
                "frequencies must be strictly increasing"
            );
            assert!(
                *plan.f.last().unwrap() <= fs / 2.0,
                "last frequency must not exceed fs/2"
            );
            assert_eq!(plan.f.len(), plan.lengths.len());
            assert_eq!(plan.f.len(), plan.segments.len());
            assert!(plan.lengths.iter().all(|&l| l >= 1));
            assert!(plan.segments.iter().all(|&k| k >= 1));
        }
    }
}
