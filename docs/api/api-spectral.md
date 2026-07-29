# `precision_physkit.spectral` API

```python
from precision_physkit import spectral
```

## `Spectrum`

```python
@dataclass
class Spectrum:
    f: NDArray
    values: NDArray
    kind: Literal["psd", "csd", "coherence", "transfer", "asd"]
    scale: Literal["linear", "log"]
    method: str
    meta: dict
```

`values` is `(J,)`, `(J, C)`, or `(J, C, C)`. PSD, ASD, and coherence are
real; CSD and transfer functions are complex. `to_asd()` returns the square root
of a PSD and rejects other kinds. `to_dataframe()` supports one- and two-dimensional
values but not a full CSD matrix.

## Welch family

```python
spectral.welch_psd(x, fs, nperseg=None, noverlap=None,
                   window="hann", detrend="constant")
spectral.welch_csd(x, y, fs, nperseg=None, noverlap=None,
                   window="hann", detrend="constant")
spectral.welch_coherence(x, y, fs, nperseg=None, noverlap=None,
                         window="hann", detrend="constant")
spectral.welch_transfer(x, y, fs, nperseg=None, noverlap=None,
                        window="hann", detrend="constant",
                        with_coherence=False)
```

The default segment length is `max(8, N // 8)` and default overlap is 50%.
Real input gives a one-sided spectrum. Two-dimensional PSD input is processed
column by column. Coherence is `|Sxy|²/(Sxx*Syy)`. Transfer functions use the
H1 convention `H = Sxy/Sxx`; with `with_coherence=True`, matching coherence is
stored in `result.meta["coherence"]`.

Welch uses a uniformly spaced linear frequency axis and is preferred for precise
line location or when a specific constant resolution is required.

## LPSD family

```python
spectral.lpsd(x, fs, Jdes=200, Kdes=100, xi=0.5, parallel=True)
spectral.lcsd(x, y, fs, Jdes=200, Kdes=100, xi=0.5, parallel=True)
spectral.lcoherence(x, y, fs, Jdes=200, Kdes=100, xi=0.5, parallel=True)
spectral.ltransfer(x, y, fs, Jdes=200, Kdes=100, xi=0.5,
                   parallel=True, with_coherence=False)
spectral.lcsd_matrix(X, fs, Jdes=200, Kdes=100, xi=0.5, parallel=True)
```

These functions use `precision_physkit._core`. LPSD assigns a different segment
length to every logarithmic frequency bin: long segments at low frequency improve
resolution, while short segments at high frequency allow more averages.

- `Jdes` controls desired logarithmic frequency density; actual `J` generally
  differs, so always use the returned `Spectrum.f`.
- `Kdes` controls desired averages and variance. Increasing it smooths estimates
  but can reduce effective low-frequency resolution.
- `xi` is overlap in `[0, 1)`; 0.5 is a standard compromise.
- Coverage is approximately `fs/N` to `fs/2`; lower frequencies require more data.
- Require finite input, positive finite `fs`, `Jdes >= 2`, `Kdes >= 2`, equal
  paired lengths, and enough samples for the longest segment.

`lcsd_matrix` returns `(J,C,C)` Hermitian matrices with real PSDs on the diagonal.
The one-sided PSD normalization is `2/(fs*sum(w²))`; CSD uses
`conj(Ax)*Ay`, so H1 recovers `Sxy/Sxx`.

## Quality controls

```python
H = spectral.ltransfer(x, y, fs, with_coherence=True)
good = np.isfinite(H.values) & (H.meta["coherence"] > 0.9)
f_good, H_good = H.f[good], H.values[good]
```

Reject bins with zero `Sxx`, non-finite transfer values, or inadequate coherence.
Segmented estimators assume the system is approximately stationary within a
segment. A long impulse response can bias both H1 and coherence; compare several
Welch segment lengths and consider longer data or smaller `Kdes`. Coherence alone
cannot validate a long-memory system because it is biased by the same mechanism.

Do not take square roots of CSD or transfer functions. Use `to_asd()` only for PSD.

## See also

[Preprocessing](api-preprocess.md) · [Filters](api-filters.md)
