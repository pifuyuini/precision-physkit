# 02 Logarithmic-Frequency Spectral Estimator Family — Results Summary

## Scope

A known FIR low-pass filter (129 taps, 400 Hz cutoff, fs = 4096 Hz) served as
the system under test. The input was white noise x, and the output was
y = fir(x) + 5% independent noise. The example computed `lpsd`, `lcsd`,
`lcoherence`, and `ltransfer` on a logarithmic frequency grid and compared the
estimated response with the theoretical |H(f)|.

## Key Results

- Logarithmic frequency grid: 112 bins (Jdes = 200, with 112 bins returned)
  spanning 0.033 – 1947.2 Hz
- Mean passband coherence (30–300 Hz): 0.9995
- Mean stopband coherence (>700 Hz): 0.0003; independent output noise dominates
  and the coherence collapses as expected
- Relative bias of the passband |H| estimate: median -0.010%, with a 90th
  percentile absolute bias of 0.162%
- Conclusion: `ltransfer` quantitatively agrees with theoretical |H| in the
  passband where coherence is near 1. The transfer-function estimate is
  unreliable in the low-coherence stopband, demonstrating that coherence is
  the relevant estimator-quality criterion.

## Artifacts

- `reference/results/figures/02_lpsd_family.png`
- `reference/results/data/02_lpsd_family.npz`
- `reference/results/reports/02_spectral_logscale_summary.md`
