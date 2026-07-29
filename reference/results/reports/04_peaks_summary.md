# 04 Peak Analysis — Results Summary

## Scope

A synthetic time series (fs = 4096 Hz, T = 120 s) contained 4 spectral lines,
{50.0: 0.8, 120.0: 0.4, 250.0: 0.25, 400.0: 0.15}, plus white noise with
σ = 0.1. The workflow was `welch_psd` (nperseg = 16384,
0.2500 Hz resolution) → ASD → `find_peaks` (prominence ≥ 0.002) →
`fit_peaks` (Gaussian model, ±3.0 Hz window).

## Key Results

- ASD noise floor: measured median 0.0022 V/√Hz and theoretical value
  0.0022 V/√Hz, in agreement
- 4 peaks detected at [ 50. 120. 250. 400.] Hz, in one-to-one correspondence
  with the injected lines, with no false positives or missed detections
- Maximum center-frequency bias after Gaussian refinement:
  |Δf| = 0.0003 Hz, well below the 0.2500 Hz frequency resolution
- Fitted FWHM ≈ 0.481 Hz, corresponding to the Hann-window main-lobe width
  determined by nperseg and consistent across all lines
- Fitted amplitude ratio amp(50 Hz)/amp(120 Hz) = 2.002, compared with the true
  ratio of 2.000, recovering both amplitude ordering and quantitative ratio
- Shared baseline: 5.58e-04 V/√Hz

## Artifacts

- `reference/results/figures/04_peaks_spectrum.png`
- `reference/results/data/04_peaks.csv` (4-row peak table)
- `reference/results/reports/04_peaks_summary.md`
