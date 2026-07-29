"""Example 05: preprocessing comparisons.

Demonstrate spectral whitening, NaN-gap filling, and anti-aliasing
downsampling. Generated files are written below ``artifacts/``.

Run from the repository root::

    uv run python examples/05_preprocess.py

"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from precision_physkit import plotting, preprocess, spectral

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "artifacts" / "figures"
REPORT_DIR = REPO / "artifacts" / "reports"

SEED = 20260724


def pink_noise(rng: np.random.Generator, n: int, fs: float, std: float) -> np.ndarray:
    """Generate normalized 1/f noise by spectral shaping."""
    white = rng.normal(size=n)
    spectrum = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum[1:] /= np.sqrt(f[1:])
    spectrum[0] = 0.0
    out = np.fft.irfft(spectrum, n=n)
    return out * (std / np.std(out))


def main() -> None:
    for d in (FIG_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------
    print("[1/3] Whitening 1/f colored noise ...")
    fs1, n1 = 1024.0, 32768
    colored = pink_noise(rng, n1, fs1, 1.0) + rng.normal(0.0, 0.05, n1)
    whitened = preprocess.whiten(colored, fs1, nperseg=4096)
    psd_before = spectral.welch_psd(colored, fs1, nperseg=4096)
    psd_after = spectral.welch_psd(whitened, fs1, nperseg=4096)

    def band_ratio_db(psd: spectral.Spectrum, lo=(1.0, 5.0), hi=(100.0, 300.0)) -> float:
        """Return the low-band to high-band mean PSD ratio in dB."""
        m_lo = (psd.f >= lo[0]) & (psd.f <= lo[1])
        m_hi = (psd.f >= hi[0]) & (psd.f <= hi[1])
        return float(10.0 * np.log10(psd.values[m_lo].mean() / psd.values[m_hi].mean()))

    ratio_before = band_ratio_db(psd_before)
    ratio_after = band_ratio_db(psd_after)
    print(f"      Low/high-band PSD ratio: {ratio_before:.1f} dB -> {ratio_after:.1f} dB")

    # ------------------------------------------------------------------
    print("[2/3] Filling a NaN gap ...")
    fs2, n2 = 1024.0, 16384
    t2 = np.arange(n2) / fs2
    # The gap is shorter than the fastest component's period.
    clean = (np.sin(2 * np.pi * 0.5 * t2) + 0.4 * np.sin(2 * np.pi * 2.0 * t2 + 1.0)
             + rng.normal(0.0, 0.10, n2))
    gap_start, gap_len = 8000, 100
    gapped = clean.copy()
    gapped[gap_start : gap_start + gap_len] = np.nan
    filled = preprocess.fill_gaps(gapped, method="pchip", fill_noise=True, seed=SEED)
    gap_err = filled[gap_start : gap_start + gap_len] - clean[gap_start : gap_start + gap_len]
    gap_rms = float(np.sqrt(np.mean(gap_err**2)))
    sig_rms = float(np.std(clean))
    print(f"      Gap length {gap_len} samples; filled-region RMS error {gap_rms:.4f} "
          f"(signal RMS {sig_rms:.4f}, ratio {gap_rms / sig_rms:.2f})")

    # ------------------------------------------------------------------
    print("[3/3] Comparing anti-aliasing downsampling with naive slicing ...")
    fs3, n3 = 1024.0, 32768
    q = 4                      # 1024 Hz -> 256 Hz; new Nyquist frequency is 128 Hz.
    t3 = np.arange(n3) / fs3
    sig = (np.sin(2 * np.pi * 20 * t3) + 0.7 * np.sin(2 * np.pi * 200 * t3)
           + rng.normal(0.0, 0.05, n3))
    dec = preprocess.downsample(sig, q)          # Includes anti-alias filtering.
    naive = sig[::q]                             # Aliases 200 Hz to 56 Hz.
    fs_new = fs3 / q
    psd_dec = spectral.welch_psd(dec, fs_new, nperseg=4096)
    psd_naive = spectral.welch_psd(naive, fs_new, nperseg=4096)

    def band_power(psd: spectral.Spectrum, f_lo: float, f_hi: float) -> float:
        m = (psd.f >= f_lo) & (psd.f <= f_hi)
        return float(np.trapezoid(psd.values[m], psd.f[m]))

    # Compare band power around the 56 Hz alias.
    alias_naive = band_power(psd_naive, 50.0, 62.0)
    alias_dec = band_power(psd_dec, 50.0, 62.0)
    alias_db = 10.0 * np.log10(alias_naive / alias_dec)
    print(f"      50-62 Hz alias-band power, naive / anti-aliased = {alias_db:.1f} dB")

    # ------------------------------------------------------------------
    print("[plot] Rendering the comparison figure ...")
    with plotting.temp_style(["ysy_academic", "tab4"]):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

        ax = axes[0]
        ax.loglog(psd_before.f, psd_before.values, label="before (1/f + white)")
        ax.loglog(psd_after.f, psd_after.values, label="after whiten")
        ax.set_xlabel("frequency / Hz")
        ax.set_ylabel("PSD")
        ax.set_title(f"(a) whiten: band ratio {ratio_before:.1f} → "
                     f"{ratio_after:.1f} dB")
        ax.legend(fontsize=8)

        ax = axes[1]
        sl = slice(gap_start - 600, gap_start + gap_len + 600)
        ax.plot(t2[sl], clean[sl], lw=1.0, label="true (gap-free)")
        ax.plot(t2[sl], filled[sl], "--", lw=1.0, label="fill_gaps")
        ax.axvspan(t2[gap_start], t2[gap_start + gap_len], color="0.85",
                   zorder=0, label="gap")
        ax.set_xlabel("t / s")
        ax.set_title(f"(b) fill_gaps: gap RMS err {gap_rms:.3f}")
        ax.legend(fontsize=8)

        ax = axes[2]
        ax.semilogy(psd_naive.f, psd_naive.values, lw=1.0,
                    label="naive x[::q] (aliased)")
        ax.semilogy(psd_dec.f, psd_dec.values, lw=1.0,
                    label="downsample (anti-alias)")
        ax.axvline(56.0, color="0.5", ls=":", lw=1.0, label="alias of 200 Hz")
        ax.axvline(fs_new / 2, color="0.7", ls="--", lw=1.0,
                   label="new Nyquist")
        ax.set_xlabel("frequency / Hz")
        ax.set_ylabel("PSD")
        ax.set_title(f"(c) downsample q={q}: alias suppressed {alias_db:.1f} dB")
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(FIG_DIR / "05_preprocess.png", dpi=200)
        plt.close(fig)
    print(f"      -> {FIG_DIR.relative_to(REPO)}/05_preprocess.png")

    summary = f"""# 05 Preprocessing comparisons

## Results
### 1. whiten
- Low/high-band PSD ratio: {ratio_before:.1f} dB before and {ratio_after:.1f} dB after whitening.

### 2. fill_gaps
- Gap length: {gap_len} samples ({gap_len / fs2:.2f} s).
- Filled-region RMS error: {gap_rms:.4f}, or {gap_rms / sig_rms:.0%} of signal RMS.

### 3. downsample
- Decimation factor: {q}; new Nyquist frequency: {fs_new / 2:.0f} Hz.
- Naive/anti-aliased 50-62 Hz band-power ratio: {alias_db:.1f} dB.

## Generated files
- `artifacts/figures/05_preprocess.png`
- `artifacts/reports/05_preprocess_summary.md`
"""
    (REPORT_DIR / "05_preprocess_summary.md").write_text(summary, encoding="utf-8")
    print("Example 05 complete.")


if __name__ == "__main__":
    main()
