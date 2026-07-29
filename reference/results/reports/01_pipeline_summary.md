# 01 Time-Series Pipeline — Results Summary

## Scope

This example generated a synthetic dual-channel precision-measurement time
series (fs = 2048 Hz, T = 60 s, N = 122880) containing 1/f noise, white noise,
injected spectral lines, and 444 samples in NaN gaps. It exercised the complete
raw data → metadata → preprocessing → spectral analysis → export workflow, with
6 processing stages recorded in the metadata.

## Key Results

- Raw data: 122880 samples × 2 channels @ 2048 Hz; 444 samples in NaN gaps
  (ch0: ['10000+64', '60000+200'], ch1: ['30000+100', '110000+80'])
- After preprocessing: 61440 samples @ 1024 Hz; post-downsampling
  RMS = [0.7754, 0.5443] V
- `welch_psd`: 4097 linearly spaced frequency bins, nperseg = 8192, averaged
  over 14 segments
- `lpsd`: 90 logarithmically spaced frequency bins spanning 0.017 – 494.9 Hz
- Detected frequencies for the injected spectral lines
  (true value → Welch estimate / LPSD estimate):
  - ch0: 50 Hz → 50.00 / 49.41 Hz, 137 Hz → 137.00 / 140.16 Hz,
    400 Hz → 400.00 / 400.49 Hz
  - ch1: 50 Hz → 50.00 / 49.41 Hz, 290 Hz → 290.00 / 281.87 Hz
  - Comparison: the Welch linear-frequency grid has 0.125 Hz resolution and
    localizes the lines accurately. The LPSD logarithmic grid is sparse at high
    frequencies and therefore provides coarser localization, but offers much
    finer resolution below 1 Hz. The estimators are complementary.

## Artifacts

- `reference/data/raw/01_precision_run_raw.npz`
- `reference/data/meta/01_precision_run.meta.toml` (logs for 6 processing stages)
- `reference/results/figures/01_timeseries_gaps.png`
- `reference/results/figures/01_psd_comparison.png`
- `reference/results/data/01_pipeline_spectra.npz`
- `reference/results/reports/01_pipeline_summary.md`
