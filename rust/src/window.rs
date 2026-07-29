//! Symmetric Hann window and a per-length cache.

use std::collections::HashMap;

/// Symmetric Hann window of length `len`: `w[i] = 0.5 * (1 - cos(2*pi*i/(len-1)))`.
///
/// This matches MATLAB `hann(len)` (denominator `len - 1`), not the periodic
/// (DFT-even) variant used by scipy's default. A length-1 window degenerates
/// to `[1.0]`; length 2 is identically zero and is rejected by the caller's
/// plan check instead of being silently produced.
pub fn hann_window(len: usize) -> Vec<f64> {
    if len <= 1 {
        return vec![1.0];
    }
    let denom = (len - 1) as f64;
    (0..len)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / denom).cos()))
        .collect()
}

/// A cached window: coefficients plus their squared norm `sum(w^2)`,
/// which is the only window statistic the PSD normalization needs.
pub struct WindowEntry {
    pub w: Vec<f64>,
    pub sum_w_sq: f64,
}

/// Cache of windows keyed by segment length, so a window is generated at
/// most once per distinct `L`. With rayon each worker thread owns its cache
/// (via `map_init`), so no synchronization is needed.
#[derive(Default)]
pub struct WindowCache {
    map: HashMap<usize, WindowEntry>,
}

impl WindowCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&mut self, len: usize) -> &WindowEntry {
        self.map.entry(len).or_insert_with(|| {
            let w = hann_window(len);
            let sum_w_sq = w.iter().map(|v| v * v).sum();
            WindowEntry { w, sum_w_sq }
        })
    }
}
