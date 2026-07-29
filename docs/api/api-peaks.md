# `precision_physkit.peaks` API

```python
from precision_physkit import peaks
```

## Result structures

`Peak` is an immutable detected-peak record containing index, position, height,
prominence, width, FWHM, area, and integration bounds. `PeakAnalysisResult`
contains the detected peaks and vector convenience properties.

`FittedPeak` is an immutable line-shape result containing center, amplitude,
FWHM, analytic area, model parameters, and uncertainty fields. `PeakFitResult`
contains the model name, fitted peaks, baseline and its uncertainty, the input
axis, the fitted curve evaluated across that axis, and the full covariance.

## Detection

```python
peaks.find_peaks(
    x, height=None, prominence=None, distance=None, width=None, x_axis=None,
)
```

Wraps SciPy peak detection and width measurement. If `x_axis` is omitted,
positions and widths use sample coordinates. If supplied, it must be finite,
strictly increasing, and match `x`. The quick detection area includes background;
use line-shape fitting or `peak_area(..., baseline=...)` for corrected area.

## Line-shape fitting

```python
peaks.fit_peaks(
    x, y, peak_indices, model="gaussian", window=None,
)
```

Fits a shared constant baseline plus the sum of all peak components. Models are
`"gaussian"`, `"lorentzian"`, and `"voigt"`. `window` limits each peak to a
local region and is important for dense spectra. Centers and widths are bounded
to physically meaningful values.

Voigt sigma and gamma can be poorly identifiable for a single noisy peak; prefer
high-SNR or joint fits. SciPy `RuntimeError` can propagate on non-convergence, so
production pipelines should catch and report it.

## Numerical area

```python
area = peaks.peak_area(x, y, left, right, baseline=None)
```

Integrates by the trapezoid rule with linearly interpolated boundary values.
`baseline=None` integrates the raw signal; a numeric baseline is constant;
`baseline="edge"` subtracts the line connecting the two boundary values.

## Standard workflow

```python
detected = peaks.find_peaks(y, prominence=0.5, x_axis=x)
if len(detected) == 0:
    raise RuntimeError("no peaks detected")

fit = peaks.fit_peaks(
    x, y, [peak.index for peak in detected], model="gaussian", window=3.0,
)
for peak in fit.peaks:
    print(peak.center, peak.errors["center"], peak.fwhm, peak.area)
```

Validate fitted FWHM against sampling or spectral resolution, and state the
baseline convention whenever reporting area.

## See also

[Spectral analysis](api-spectral.md) · [Fitting](api-fitting.md)
