# `precision_physkit.preprocess` API

```python
from precision_physkit import preprocess
```

## Resampling

```python
preprocess.downsample(x, q, axis=-1, zero_phase=True, ftype="iir")
preprocess.upsample(x, q, axis=-1)
preprocess.resample(x, fs_in, fs_out, axis=-1, max_denominator=100_000)
```

`downsample` wraps `scipy.signal.decimate` and applies anti-alias filtering before
integer decimation. `q` is an integer at least one; `ftype` is `"iir"` or `"fir"`.
Never replace it with `x[::q]`.

`upsample` uses polyphase resampling with `up=q`. `resample` approximates
`fs_out/fs_in` as a rational ratio bounded by `max_denominator`, then uses
polyphase anti-alias/anti-imaging filtering. Sample rates must be positive.

## Interpolation

```python
preprocess.interpolate(t, x, t_new, method="linear")
```

Resamples a one-dimensional signal onto `t_new`. `t` must be finite, strictly
increasing, and match `x`; methods are `"linear"`, `"cubic"`, and `"pchip"`.
Points outside the observed interval become NaN rather than being extrapolated.

## Gap filling

```python
preprocess.fill_gaps(
    x, method="pchip", fill_noise=False, noise_std=None, seed=None,
)
```

Fills NaNs in a one-dimensional signal. Internal gaps are interpolated; leading
and trailing gaps use nearest-value constant extension. With `fill_noise=True`,
matched random noise is added to filled samples; fix `seed` for reproducibility.
`noise_std` overrides local estimation.

Gap filling synthesizes data. It cannot recover structure in gaps longer than the
fastest relevant physical period, and every use should be recorded in metadata.

## Whitening

```python
preprocess.whiten(x, fs, method="psd", nperseg=None)
```

Estimates a Welch PSD, divides the real FFT by its square root, and transforms
back to a spectrally flattened time series. Only `method="psd"` is supported.
Input must be finite and one-dimensional.

Whitened absolute amplitude is arbitrary because FFT and Welch normalization
differ. FFT periodicity also makes edge samples vulnerable to wraparound
artifacts. Use whitening for detection or conditioning, not absolute levels.

## Patterns

```python
# Integer ratio.
q = int(fs_in / fs_out)
x_out = preprocess.downsample(x, q)

# Arbitrary ratio.
x_out = preprocess.resample(x, fs_in, fs_out)

# Reproducible gap filling before spectral analysis.
x_filled = preprocess.fill_gaps(x, method="pchip", fill_noise=True, seed=42)

# Irregular timestamps to a uniform grid.
t_uniform = np.arange(t[0], t[-1], 1 / fs)
x_uniform = preprocess.interpolate(t, x, t_uniform, method="pchip")
```

## See also

[Filters](api-filters.md) · [Spectral analysis](api-spectral.md) ·
[Pipeline](api-pipeline.md)
