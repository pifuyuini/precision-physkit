# precision-physkit user guide

## Overview

`precision-physkit` provides reproducible workflows for sampled physical data:
metadata and provenance, resampling and gap handling, digital filters, Welch and
logarithmic spectra, parameter fitting, numerical optimization, peak analysis,
and publication-oriented plotting.

```python
from precision_physkit import (
    filters, fitting, meta, optimize, peaks, plotting, preprocess, spectral,
)
```

The Rust extension `precision_physkit._core` accelerates the LPSD family and the
linear least-squares routines. Other modules remain usable if the extension is
unavailable.

## Install from a clean clone

```bash
uv sync --locked
uv run pytest -q
uv run cargo test --manifest-path rust/Cargo.toml --locked
```

If no lock file exists yet, run `uv sync` to resolve the environment before
switching to locked installs.

## A complete pipeline

### 1. Register the raw dataset

```python
import numpy as np
from precision_physkit import __version__, meta

raw_path = "data/raw/01_precision_run_raw.npz"
meta_path = "data/meta/01_precision_run.meta.toml"
np.savez(raw_path, t=t, data=data, fs=np.array(fs))

doc = meta.create_meta(
    "01-precision-run",
    instrument="daq-01",
    experiment="noise-floor survey",
    files=["01_precision_run_raw.npz"],
    format="npz",
    fs=fs,
    n_samples=data.shape[0],
    t_start=float(t[0]),
    t_end=float(t[-1]),
    channels=[
        {"name": "sensor_a", "unit": "V", "quantity": "voltage"},
        {"name": "sensor_b", "unit": "V", "quantity": "voltage"},
    ],
)
meta.save_meta(doc, meta_path)
meta.log_stage(
    meta_path, "acquire", "acquisition", __version__,
    params={"fs": float(fs), "seed": 20260720}, outputs=[raw_path],
)
```

Raw files are immutable. `params` accepts native TOML-compatible Python values;
convert NumPy scalars with `float()` or `int()`.

### 2. Fill gaps, filter, and downsample

```python
from precision_physkit import filters, preprocess

filled = np.column_stack([
    preprocess.fill_gaps(data[:, j], fill_noise=True, seed=100 + j)
    for j in range(data.shape[1])
])
meta.log_stage(meta_path, "fill_gaps", "precision_physkit.preprocess.fill_gaps",
               __version__, params={"method": "pchip", "fill_noise": True, "seed": 100})

# Butterworth filters act on the last axis; transpose (N, C) data.
lowpassed = filters.lowpass(filled.T, fs, cutoff=480.0, order=4).T
meta.log_stage(meta_path, "lowpass", "precision_physkit.filters.lowpass",
               __version__, params={"cutoff": 480.0, "order": 4, "zero_phase": True})

decimated = preprocess.downsample(lowpassed, 2, axis=0)
meta.log_stage(meta_path, "downsample", "precision_physkit.preprocess.downsample",
               __version__, params={"q": 2, "zero_phase": True, "ftype": "iir"})
```

Fill gaps before filtering because NaNs propagate through filters. Downsampling
already includes a general anti-alias filter, but a task-specific low-pass cutoff
usually gives clearer control.

### 3. Compare linear and logarithmic spectra

```python
from precision_physkit import spectral

welch = spectral.welch_psd(decimated, fs / 2, nperseg=8192).to_asd()
logsp = spectral.lpsd(decimated, fs / 2, Jdes=150, Kdes=60).to_asd()
```

Both return `Spectrum`. Use `Spectrum.f`; never reconstruct the frequency grid
from `nperseg` or `Jdes`. Welch has uniform linear resolution and is preferred
for precise line location. LPSD uses long low-frequency and short high-frequency
segments, making it useful across many decades.

### 4. Export artifacts and plot

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from precision_physkit import plotting

np.savez("artifacts/data/01_spectra.npz",
         f_welch=welch.f, asd_welch=welch.values,
         f_lpsd=logsp.f, asd_lpsd=logsp.values)

with plotting.temp_style(["ysy_academic", "tab4"]):
    fig, ax = plt.subplots()
    ax.loglog(welch.f, welch.values)
    fig.savefig("artifacts/figures/01_spectra.pdf", bbox_inches="tight")
    plt.close(fig)

meta.log_stage(
    meta_path, "export", "pipeline", __version__,
    params={"formats": ["npz", "pdf"]},
    outputs=["artifacts/data/01_spectra.npz", "artifacts/figures/01_spectra.pdf"],
)
```

Reference outputs suitable for regression comparison belong in `reference/`;
new runtime output belongs in `artifacts/`.

## Method selection

- **Metadata:** `create_meta`, `validate_meta`, `save_meta`, `load_meta`,
  `log_stage`.
- **Preprocessing:** use `downsample` for an integer ratio, `resample` for an
  arbitrary ratio, `interpolate` for a new time grid, `fill_gaps` for NaNs, and
  `whiten` for detection rather than absolute spectral levels.
- **Filters:** Butterworth low/high/band filters, notch filtering, Savitzky-Golay
  smoothing or derivatives, and moving averages.
- **Spectra:** Welch for uniform resolution; LPSD for wide-band logarithmic
  coverage. Use coherence with every transfer-function estimate.
- **Fitting:** `curve_fit` for named nonlinear parameters and uncertainty;
  `linear_lstsq` for SVD-controlled linear systems; `iterative_multichannel`
  when channels share parameters but have different unknown noise.
- **Optimization:** use bounded global search for multimodal objectives, then a
  local refinement. Fix the seed for reproducibility.
- **Peaks:** detect, fit a line shape, then report center uncertainty, width,
  amplitude, and baseline-corrected area.
- **Plotting:** create a style first, enter `temp_style`, then draw only data.

## Numerical cautions

- `fill_gaps` creates synthetic samples. Interpolation is credible only for gaps
  short relative to the fastest relevant period; edge gaps use constant extension.
- Never downsample by slicing. Energy above the new Nyquist frequency aliases into
  the retained band.
- Whitening flattens spectral shape but has arbitrary absolute amplitude and can
  show FFT wraparound artifacts at the ends.
- For transfer functions, reject non-finite values and low-coherence bins.
  Long-memory systems can bias both H1 and coherence; compare several segment
  lengths before drawing conclusions.
- With `curve_fit(..., sigma=...)`, set `absolute_sigma=True` when `sigma`
  represents absolute measurement uncertainty.
- `polyfit` returns coefficients in ascending order: `c0, c1, ...`.
- `global_minimize(..., seed=...)` is stochastic; `success=True` means normal
  termination, not proof of the global optimum.
- `fit_peaks` may propagate SciPy `RuntimeError` on non-convergence.
- `temp_style` restores Matplotlib defaults on exit, not the prior style stack.

## Further reading

- [Architecture](architecture.md)
- [Pipeline contract](api/api-pipeline.md)
- [Metadata API](api/api-meta.md)
- [Preprocessing API](api/api-preprocess.md)
- [Filters API](api/api-filters.md)
- [Spectral API](api/api-spectral.md)
- [Fitting API](api/api-fitting.md)
- [Optimization API](api/api-optimize.md)
- [Peaks API](api/api-peaks.md)
- [Plotting API](api/api-plotting.md)
