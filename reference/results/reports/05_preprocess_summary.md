# 05 Preprocessing Comparison — Results Summary

## Scope

Three independent experiments were performed: `whiten` on 1/f colored noise;
`fill_gaps` on a 100-sample NaN gap; and a comparison of anti-aliased
`downsample` (q=4, 1024→256 Hz) with naive slicing. The fixed random seed was
20260724.

## Key Results

### 1. `whiten` (1/f + white noise, N = 32768, fs = 1024 Hz)

- Ratio of mean PSD in the low-frequency band (1–5 Hz) to that in the
  high-frequency band (100–300 Hz): 19.1 dB before whitening and 0.0 dB after
  whitening, demonstrating spectral flattening.

### 2. `fill_gaps` (0.5 Hz + 2 Hz slowly varying signal, σ=0.10 noise, 100-sample gap ≈ 0.10 s)

- RMS error in the filled segment relative to the gap-free reference:
  0.2292, or 30% of the signal RMS (0.7683). When the gap is shorter than the
  period of the fastest component, PCHIP follows the waveform well;
  `fill_noise=True` adds matched noise to maintain stable variance, with no
  remaining NaN values.
- Caution: filled values are synthetic. If a gap is longer than the period of
  the fastest signal component, interpolation cannot recover the true waveform
  and can only preserve statistical stationarity. In a preliminary long-gap
  experiment for this script, the error reached 1.4 times the signal RMS.

### 3. `downsample` (q = 4, new Nyquist frequency = 128 Hz, 200 Hz out-of-band tone)

- Naive slicing with x[::4] aliases the 200 Hz tone to 56 Hz. The ratio of
  50–62 Hz band power for naive slicing versus anti-aliased downsampling is
  36.3 dB, showing that the anti-aliasing filter suppresses the aliased tone
  below the noise floor.

## Artifacts

- `reference/results/figures/05_preprocess.png`
- `reference/results/reports/05_preprocess_summary.md`
