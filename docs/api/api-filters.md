# `precision_physkit.filters` API

```python
from precision_physkit import filters
```

## Butterworth filters

```python
filters.lowpass(x, fs, cutoff, order=4, zero_phase=True)
filters.highpass(x, fs, cutoff, order=4, zero_phase=True)
filters.bandpass(x, fs, cutoff, order=4, zero_phase=True)
filters.bandstop(x, fs, cutoff, order=4, zero_phase=True)
```

The low/high-pass cutoff is a scalar; band filters accept
`(f_low, f_high)`. Frequencies must satisfy `0 < cutoff < fs/2`, or
`0 < f_low < f_high < fs/2`. Filters use second-order sections.

`zero_phase=True` applies forward-backward filtering: zero group delay, squared
single-pass magnitude response, and doubled effective order. `False` is causal
and introduces frequency-dependent delay. Filtering acts on the last axis.

## Notch filter

```python
filters.notch(x, fs, f0, q=30, zero_phase=True)
```

A second-order IIR notch, commonly used for 50/60 Hz interference.
`q = f0 / bandwidth`; larger `q` gives a narrower notch. Require
`0 < f0 < fs/2` and `q > 0`.

## Smoothing

```python
filters.savgol(x, window_length, polyorder, deriv=0, axis=-1)
filters.moving_average(x, window, axis=-1)
```

Savitzky-Golay smoothing preserves polynomial peak structure and can estimate
derivatives. `window_length` must be positive and odd, `polyorder` smaller than
the window, and `deriv <= polyorder`. Derivatives use sample spacing one; divide
by the physical interval as needed.

The moving average accepts even or odd positive windows and uses nearest-value
boundary extension.

## Patterns

```python
from precision_physkit import filters, preprocess

# Target-specific anti-alias filter, then integer decimation.
x_lp = filters.lowpass(x, fs=2048.0, cutoff=240.0, order=4)
x_ds = preprocess.downsample(x_lp, 4)

# Butterworth filters operate on the last axis.
filtered = filters.lowpass(data.T, fs=1000.0, cutoff=100.0).T
smoothed = filters.savgol(data, window_length=51, polyorder=3, axis=0)
```

For `sosfiltfilt` or `filtfilt`, very short signals can fail because of padding
requirements. Use causal filtering only when latency or streaming semantics
require it, and document the phase delay.

## See also

[Preprocessing](api-preprocess.md) · [Spectral analysis](api-spectral.md)
