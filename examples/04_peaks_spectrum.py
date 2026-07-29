"""Example 04: spectral-line detection and peak refinement.

Generate four known tones in white noise, estimate their ASD, detect and fit
the peaks, and export a CSV table. Generated files are written below
``artifacts/``.

Run from the repository root::

    uv run python examples/04_peaks_spectrum.py

"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from precision_physkit import peaks, plotting, spectral

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "artifacts" / "figures"
DATA_DIR = REPO / "artifacts" / "data"
REPORT_DIR = REPO / "artifacts" / "reports"

SEED = 20260723
FS = 4096.0
DURATION = 120.0
LINES = {50.0: 0.8, 120.0: 0.4, 250.0: 0.25, 400.0: 0.15}  # Frequency [Hz] -> amplitude [V]
NOISE_STD = 0.1
NPERSEG = 16384
PROMINENCE = 2e-3     # Minimum find_peaks prominence [V/sqrt(Hz)]
FIT_WINDOW = 3.0      # Peak-fit half-window [Hz]


def main() -> None:
    for d in (FIG_DIR, DATA_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("[1/4] Generating the tone record and estimating Welch ASD ...")
    rng = np.random.default_rng(SEED)
    n = int(FS * DURATION)
    t = np.arange(n) / FS
    x = rng.normal(0.0, NOISE_STD, n)
    for f0, amp in LINES.items():
        x = x + amp * np.sin(2.0 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
    asd = spectral.welch_psd(x, FS, nperseg=NPERSEG).to_asd()
    floor = float(np.median(asd.values))
    floor_theory = NOISE_STD * np.sqrt(2.0 / FS)
    df = float(asd.f[1] - asd.f[0])
    print(f"      N = {n}; frequency resolution {df:.4f} Hz; "
          f"ASD floor {floor:.4f} (theory {floor_theory:.4f}) V/sqrt(Hz)")

    print("[2/4] Detecting peaks and refining Gaussian fits ...")
    found = peaks.find_peaks(asd.values, prominence=PROMINENCE, x_axis=asd.f)
    print(f"      Detected {len(found)} peaks: {np.round(found.positions, 2)} Hz")
    fit = peaks.fit_peaks(
        asd.f, asd.values, [p.index for p in found],
        model="gaussian", window=FIT_WINDOW,
    )
    for p in fit.peaks:
        print(f"      center {p.center:8.3f} ± {p.errors['center']:.4f} Hz, "
              f"amp {p.amplitude:.4f}, FWHM {p.fwhm:.3f} Hz")

    print("[3/4] Rendering the figure ...")
    with plotting.temp_style(["ysy_academic", "tab4"]):
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=True,
                                 height_ratios=[2.2, 1.0])
        ax = axes[0]
        ax.semilogy(asd.f, asd.values, lw=0.9, label="welch ASD")
        ax.semilogy(fit.x, fit.y_fit, lw=1.2, label="gaussian fit (sum)")
        for p in fit.peaks:
            ax.plot(p.center, p.amplitude + fit.baseline, "v", ms=7)
            ax.annotate(f"{p.center:.2f} Hz", (p.center, p.amplitude + fit.baseline),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)
        ax.axhline(floor, color="0.5", ls=":", lw=1.0,
                   label=f"noise floor ≈ {floor:.3f}")
        ax.set_ylabel("ASD / V/$\\sqrt{\\mathrm{Hz}}$")
        ax.set_title(f"Line detection: find_peaks + fit_peaks "
                     f"(fs = {FS:.0f} Hz, nperseg = {NPERSEG})")
        ax.legend(fontsize=8)

        ax = axes[1]
        resid_db = 10.0 * np.log10(
            np.maximum(asd.values, 1e-30) / np.maximum(fit.y_fit, 1e-30))
        ax.plot(asd.f, resid_db, lw=0.8)
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_ylabel("data/fit / dB")
        ax.set_xlabel("frequency / Hz")
        ax.set_xlim(10.0, 500.0)
        ax.set_ylim(-6, 6)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "04_peaks_spectrum.png", dpi=200)
        plt.close(fig)
    print(f"      -> {FIG_DIR.relative_to(REPO)}/04_peaks_spectrum.png")

    print("[4/4] Writing the peak table and summary ...")
    true_freqs = sorted(LINES)
    det_sorted = sorted(found.peaks, key=lambda p: p.position)
    fit_sorted = sorted(fit.peaks, key=lambda p: p.center)
    rows = []
    for f_true, det, fp in zip(true_freqs, det_sorted, fit_sorted, strict=True):
        rows.append({
            "true_freq_Hz": f_true,
            "true_amplitude_V": LINES[f_true],
            "det_position_Hz": det.position,
            "det_height": det.height,
            "det_prominence": det.prominence,
            "fit_center_Hz": fp.center,
            "fit_center_err_Hz": fp.errors["center"],
            "fit_amplitude": fp.amplitude,
            "fit_fwhm_Hz": fp.fwhm,
            "fit_area": fp.area,
            "center_bias_Hz": fp.center - f_true,
        })
    table = pd.DataFrame(rows)
    table.to_csv(DATA_DIR / "04_peaks.csv", index=False, float_format="%.6g")

    max_bias = float(np.abs(table["center_bias_Hz"]).max())
    amp_ratio_fit = table["fit_amplitude"].iloc[0] / table["fit_amplitude"].iloc[1]
    amp_ratio_true = LINES[50.0] / LINES[120.0]

    summary = f"""# 04 Peak analysis

## Results
- ASD noise floor: measured median {floor:.4f} V/sqrt(Hz), theory {floor_theory:.4f} V/sqrt(Hz).
- Detected {len(found)} peaks at {np.round(found.positions, 2)} Hz.
- Maximum refined center-frequency error: {max_bias:.4f} Hz.
- Mean fitted FWHM: {float(table['fit_fwhm_Hz'].mean()):.3f} Hz.
- Fitted amplitude ratio (50 Hz / 120 Hz): {amp_ratio_fit:.3f}; true ratio: {amp_ratio_true:.3f}.
- Shared baseline: {fit.baseline:.2e} V/sqrt(Hz).

## Generated files
- `artifacts/figures/04_peaks_spectrum.png`
- `artifacts/data/04_peaks.csv`
- `artifacts/reports/04_peaks_summary.md`
"""
    (REPORT_DIR / "04_peaks_summary.md").write_text(summary, encoding="utf-8")

    print("---- Summary ----")
    print(table[["true_freq_Hz", "fit_center_Hz", "center_bias_Hz",
                 "fit_fwhm_Hz"]].to_string(index=False))
    print("Example 04 complete.")


if __name__ == "__main__":
    main()
