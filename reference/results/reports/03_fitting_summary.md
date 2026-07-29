# 03 Fitting and Optimization — Results Summary

## Scope

Four quantitative experiments were performed: `curve_fit` on an exponentially
decaying sinusoid; an `rcond` truncation comparison for `linear_lstsq` on an
ill-conditioned Vandermonde matrix; two-channel joint estimation with
`iterative_multichannel`; and `global_minimize` on the Rastrigin function. The
fixed random seed was 20260722.

## Key Results

### 1. `curve_fit` (true value → estimate ± 1σ)

- amp: 2.5 → 2.510 ± 0.034
- tau: 1.5 → 1.494 ± 0.029
- freq: 3.0 → 3.0013 ± 0.0021
- phase: 0.7 → 0.687 ± 0.014
- Reduced χ² = 1.007 (σ is known accurately, so the expected value is ≈1);
  R² = 0.9476

### 2. `linear_lstsq` (degree-10 Vandermonde, cond = 5.88e+11)

| Variant | rank | Interpolation prediction RMS | Maximum parameter 1σ |
|---|---|---|---|
| raw, rcond=None | 11 | 0.0270 | 1.21e+00 |
| raw, rcond=1e-10 | 8 | 0.0355 | 1.70e-02 |
| raw, rcond=1e-8 | 5 | 0.9118 | 3.21e-04 |
| scaled, rcond=None | 11 | 0.0270 | 4.66e-01 |

- For an ill-conditioned matrix, a good fit does not imply correct parameters.
  Without truncation, prediction RMS remains acceptable while parameter
  uncertainty becomes very large. Setting rcond=1e-10 reduces parameter
  uncertainty by approximately two orders of magnitude at negligible
  prediction cost. Setting rcond=1e-8 over-truncates the system (rank 5) and
  severely degrades the prediction, illustrating the bias-variance trade-off.
- Structural remedy: scaling the independent variable reduces
  cond = 5.46e+03 and yields a stable solution without truncation.

### 3. `iterative_multichannel` (σ₁=0.05, σ₂=1.0, 4 shared parameters)

- Convergence: converged after 5 iterations; final weights
  [1.9954, 0.004648], with a ratio of 429 ≈ (σ₂/σ₁)² = 400. Iterative
  reweighting therefore assigns weights approximately inversely proportional
  to variance.
- Parameter RMS error: 0.0043 for the joint estimate versus 0.0390 for naive,
  unweighted stacking.

### 4. `global_minimize` (2-dimensional Rastrigin function)

- Global result: x = (2.48e-10, 1.80e-09), f = 0.00e+00 after
  2133 evaluations, with success = True (true solution x=0, f=0).
- Control: local Nelder-Mead optimization initialized at (4.5, -3.5) becomes
  trapped in a local minimum at (4.97, -3.98), with f = 40.79.

## Artifacts

- `reference/results/figures/03_fitting_optimize.png`
- `reference/results/data/03_fitting_results.json`
- `reference/results/reports/03_fitting_summary.md`
