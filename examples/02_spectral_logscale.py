"""Example 02: the logarithmic-frequency LPSD estimator family.

Estimate PSD, CSD, coherence, and transfer response for a known FIR system,
compare the estimate with theory, and write generated files below
``artifacts/``.

Run from the repository root::

    uv run python examples/02_spectral_logscale.py

"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from precision_physkit import plotting, spectral

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "artifacts" / "figures"
DATA_DIR = REPO / "artifacts" / "data"
REPORT_DIR = REPO / "artifacts" / "reports"

SEED = 20260721
FS = 4096.0
DURATION = 30.0
FIR_TAPS = 129
FIR_CUTOFF = 400.0      # FIR low-pass cutoff [Hz]
OUT_NOISE_FRAC = 0.05   # Output noise deviation relative to the filtered signal
JDES, KDES, XI = 200, 100, 0.5


def main() -> None:
    for d in (FIG_DIR, DATA_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("[1/4] Generating x and y = fir(x) + independent noise ...")
    rng = np.random.default_rng(SEED)
    n = int(FS * DURATION)
    h = scipy.signal.firwin(FIR_TAPS, FIR_CUTOFF, fs=FS)
    x = rng.normal(0.0, 1.0, n)
    y_clean = scipy.signal.fftconvolve(x, h, mode="same")
    y = y_clean + rng.normal(0.0, OUT_NOISE_FRAC * np.std(y_clean), n)
    print(f"      N = {n}; FIR low-pass {FIR_CUTOFF:.0f} Hz / {FIR_TAPS} taps; "
          f"output noise {OUT_NOISE_FRAC:.0%}")

    print("[2/4] Estimating lpsd, lcsd, lcoherence, and ltransfer ...")
    psd_x = spectral.lpsd(x, FS, Jdes=JDES, Kdes=KDES, xi=XI)
    psd_y = spectral.lpsd(y, FS, Jdes=JDES, Kdes=KDES, xi=XI)
    csd = spectral.lcsd(x, y, FS, Jdes=JDES, Kdes=KDES, xi=XI)
    coh = spectral.lcoherence(x, y, FS, Jdes=JDES, Kdes=KDES, xi=XI)
    transfer = spectral.ltransfer(x, y, FS, Jdes=JDES, Kdes=KDES, xi=XI)
    f = transfer.f
    print(f"      Log-frequency axis: {f.size} bins, {f[0]:.3f}-{f[-1]:.1f} Hz")

    # Interpolate the theoretical response onto the LPSD frequency grid.
    w_theory, h_theory = scipy.signal.freqz(h, worN=2.0 * np.pi * f / FS)
    mag_theory = np.abs(h_theory)

    print("[3/4] Rendering the four-panel figure ...")
    with plotting.temp_style(["ysy_academic", "tab4"]):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
        ax = axes[0, 0]
        ax.loglog(psd_x.f, psd_x.values, label="PSD of x (input)")
        ax.loglog(psd_y.f, psd_y.values, label="PSD of y (output)")
        ax.set_ylabel("PSD / V$^2$/Hz")
        ax.set_title("(a) auto spectra (lpsd)")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        ax.loglog(csd.f, np.abs(csd.values), label="|Sxy| (lcsd)")
        ax.set_ylabel("|CSD| / V$^2$/Hz")
        ax.set_title("(b) cross spectrum magnitude")
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        ax.semilogx(coh.f, coh.values, label="coherence (lcoherence)")
        ax.axvline(FIR_CUTOFF, color="0.5", ls=":", lw=1.0, label="FIR cutoff")
        ax.set_xlabel("frequency / Hz")
        ax.set_ylabel("magnitude-squared coherence")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("(c) coherence")
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        ax.loglog(f, np.abs(transfer.values), label="|H| estimate (ltransfer)")
        ax.loglog(f, mag_theory, "--", label="|H| theory (freqz)")
        ax.set_xlabel("frequency / Hz")
        ax.set_ylabel("|H(f)|")
        ax.set_title("(d) transfer function vs theory")
        ax.legend(fontsize=8)

        fig.suptitle("LPSD family on a known FIR system "
                     f"(fs = {FS:.0f} Hz, N = {n})")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "02_lpsd_family.png", dpi=200)
        plt.close(fig)
    print(f"      -> {FIG_DIR.relative_to(REPO)}/02_lpsd_family.png")

    print("[4/4] Exporting arrays and the summary ...")
    np.savez(
        DATA_DIR / "02_lpsd_family.npz",
        f=f, psd_x=psd_x.values, psd_y=psd_y.values,
        csd=csd.values, coherence=coh.values,
        transfer=transfer.values, mag_theory=mag_theory,
        fir_taps=h, fs=np.array(FS),
    )

    # Summarize coherence and transfer-response error.
    passband = (f >= 30.0) & (f <= 300.0)
    stopband = f >= 700.0
    coh_pass = float(np.mean(coh.values[passband]))
    coh_stop = float(np.mean(coh.values[stopband]))
    rel_err = np.abs(transfer.values[passband]) / mag_theory[passband] - 1.0
    rel_err_med = float(np.median(rel_err))
    rel_err_p90 = float(np.percentile(np.abs(rel_err), 90))

    summary = f"""# 02 LPSD estimator family

## Results
- Log-frequency axis: {f.size} bins (requested Jdes = {JDES}), spanning {f[0]:.3f}-{f[-1]:.1f} Hz.
- Mean passband coherence (30-300 Hz): {coh_pass:.4f}.
- Mean stopband coherence (>700 Hz): {coh_stop:.4f}.
- Passband transfer relative error: median {rel_err_med:+.3%}, 90th percentile absolute error {rel_err_p90:.3%}.

## Generated files
- `artifacts/figures/02_lpsd_family.png`
- `artifacts/data/02_lpsd_family.npz`
- `artifacts/reports/02_spectral_logscale_summary.md`
"""
    (REPORT_DIR / "02_spectral_logscale_summary.md").write_text(summary, encoding="utf-8")

    print("---- Summary ----")
    print(f"Coherence: passband {coh_pass:.4f} / stopband {coh_stop:.4f}")
    print(f"Passband |H| relative error: median {rel_err_med:+.3%}, p90 {rel_err_p90:.3%}")
    print("Example 02 complete.")


if __name__ == "__main__":
    main()
