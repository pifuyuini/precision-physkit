"""Example 01: end-to-end precision time-series processing.

The example generates a two-channel record with colored noise, tones, and NaN
gaps; records provenance; preprocesses the signals; compares Welch and LPSD
spectra; and writes every generated file below ``artifacts/``.

Run from the repository root::

    uv run python examples/01_pipeline_timeseries.py

Generated files include raw and processed data, provenance metadata, figures,
and a Markdown summary under ``artifacts/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from precision_physkit import __version__, filters, meta, plotting, preprocess, spectral

# --------------------------------------------------------------------------
# Paths and reproducible global parameters.
# --------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
RAW_PATH = REPO / "artifacts" / "data" / "raw" / "01_precision_run_raw.npz"
META_PATH = REPO / "artifacts" / "data" / "meta" / "01_precision_run.meta.toml"
FIG_DIR = REPO / "artifacts" / "figures"
DATA_DIR = REPO / "artifacts" / "data"
REPORT_DIR = REPO / "artifacts" / "reports"

SEED = 20260720
FS = 2048.0          # Original sample rate [Hz]
DURATION = 60.0      # Record duration [s]
FS_OUT = 1024.0      # Downsampled rate [Hz]
DS_FACTOR = int(FS / FS_OUT)
LINES_CH0 = {50.0: 0.8, 137.0: 0.3, 400.0: 0.15}   # Frequency [Hz] -> amplitude [V]
LINES_CH1 = {50.0: 0.5, 290.0: 0.2}
WHITE_STD = (0.10, 0.15)   # Per-channel white-noise standard deviations [V]
PINK_STD = (0.50, 0.40)    # Per-channel 1/f-noise standard deviations [V]
# Gaps are represented as (start sample, length) pairs.
GAPS = {0: [(10_000, 64), (60_000, 200)], 1: [(30_000, 100), (110_000, 80)]}


def pink_noise(rng: np.random.Generator, n: int, fs: float, std: float) -> np.ndarray:
    """Generate 1/f noise by spectral shaping and normalize its deviation."""
    white = rng.normal(size=n)
    spectrum = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum[1:] /= np.sqrt(f[1:])
    spectrum[0] = 0.0
    out = np.fft.irfft(spectrum, n=n)
    return out * (std / np.std(out))


def synthesize() -> tuple[np.ndarray, np.ndarray]:
    """Return time and two-channel data containing deterministic NaN gaps."""
    rng = np.random.default_rng(SEED)
    n = int(FS * DURATION)
    t = np.arange(n) / FS
    channels = []
    for ch, (lines, w_std, p_std) in enumerate(
        zip((LINES_CH0, LINES_CH1), WHITE_STD, PINK_STD, strict=True)
    ):
        sig = pink_noise(rng, n, FS, p_std) + rng.normal(0.0, w_std, n)
        for freq, amp in lines.items():
            sig = sig + amp * np.sin(2.0 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
        for start, length in GAPS[ch]:
            sig[start : start + length] = np.nan
        channels.append(sig)
    return t, np.column_stack(channels)


def main() -> None:
    for d in (RAW_PATH.parent, META_PATH.parent, FIG_DIR, DATA_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("[1/6] Generating and saving the raw two-channel record ...")
    t, data = synthesize()
    n, n_ch = data.shape
    n_nan = int(np.isnan(data).sum())
    np.savez(RAW_PATH, t=t, data=data, fs=np.array(FS))
    print(f"      N = {n} samples x {n_ch} channels; {n_nan} NaNs -> {RAW_PATH.relative_to(REPO)}")

    print("[2/6] Creating provenance metadata ...")
    doc = meta.create_meta(
        "01-precision-run",
        instrument="synthetic-daq",
        experiment="precision_physkit example 01",
        operator="precision_physkit",
        description="Synthetic two-channel record with 1/f noise, white noise, tones, and NaN gaps",
        files=[RAW_PATH.name],
        format="npz",
        fs=FS,
        n_samples=n,
        t_start=0.0,
        t_end=float(t[-1]),
        channels=[
            {"name": "sensor_a", "unit": "V", "quantity": "voltage"},
            {"name": "sensor_b", "unit": "V", "quantity": "voltage"},
        ],
    )
    meta.save_meta(doc, META_PATH)
    meta.log_stage(
        META_PATH, "acquire", "examples/01_pipeline_timeseries.py", __version__,
        params={"seed": SEED, "fs": FS, "duration_s": DURATION, "n_nan": n_nan},
        outputs=[str(RAW_PATH.relative_to(REPO))],
    )
    print(f"      -> {META_PATH.relative_to(REPO)}")

    print("[3/6] Preprocessing: fill_gaps -> lowpass -> downsample ...")
    raw = np.load(RAW_PATH)
    filled = np.column_stack([
        preprocess.fill_gaps(raw["data"][:, j], fill_noise=True, seed=SEED + j)
        for j in range(n_ch)
    ])
    meta.log_stage(
        META_PATH, "fill_gaps", "precision_physkit.preprocess.fill_gaps", __version__,
        params={"method": "pchip", "fill_noise": True, "seed": SEED},
    )
    # Filters operate on the last axis, so transpose (N, C) around the call.
    lowpassed = filters.lowpass(filled.T, FS, cutoff=480.0, order=4).T
    meta.log_stage(
        META_PATH, "lowpass", "precision_physkit.filters.lowpass", __version__,
        params={"cutoff": 480.0, "order": 4, "zero_phase": True},
    )
    decimated = preprocess.downsample(lowpassed, DS_FACTOR, axis=0)
    meta.log_stage(
        META_PATH, "downsample", "precision_physkit.preprocess.downsample", __version__,
        params={"q": DS_FACTOR, "zero_phase": True, "ftype": "iir"},
    )
    rms = np.std(decimated, axis=0)
    print(f"      {decimated.shape[0]} samples after downsampling @ {FS_OUT:.0f} Hz; "
          f"RMS = [{rms[0]:.4f}, {rms[1]:.4f}] V")

    print("[4/6] Estimating Welch and LPSD spectra ...")
    welch = spectral.welch_psd(decimated, FS_OUT, nperseg=8192).to_asd()
    logsp = spectral.lpsd(decimated, FS_OUT, Jdes=150, Kdes=60).to_asd()
    meta.log_stage(
        META_PATH, "spectral_analysis", "precision_physkit.spectral.welch_psd+lpsd", __version__,
        params={"welch_nperseg": 8192, "Jdes": 150, "Kdes": 60, "xi": 0.5},
        outputs=[str((DATA_DIR / "01_pipeline_spectra.npz").relative_to(REPO))],
    )
    print(f"      Welch: {welch.f.size} bins ({welch.meta['n_segments']} averaged segments); "
          f"LPSD: {logsp.f.size} log bins ({logsp.f[0]:.3f}-{logsp.f[-1]:.1f} Hz)")

    print("[5/6] Rendering figures ...")
    with plotting.temp_style(["ysy_academic", "tab4"]):
        # Figure 1: the record around a channel-0 gap.
        start, length = GAPS[0][1]
        sl = slice(start - 400, start + length + 400)
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
        axes[0].plot(t[sl], data[sl, 0], label="raw (with gap)")
        axes[0].axvspan(t[start], t[start + length], color="0.85", zorder=0,
                        label="gap region")
        axes[0].set_ylabel("ch0 / V")
        axes[0].set_title("Raw record around a NaN gap")
        axes[0].legend(loc="upper right")
        axes[1].plot(t[sl], filled[sl, 0], label="filled (pchip + noise)")
        axes[1].axvspan(t[start], t[start + length], color="0.85", zorder=0)
        axes[1].set_xlabel("t / s")
        axes[1].set_ylabel("ch0 / V")
        axes[1].legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "01_timeseries_gaps.png", dpi=200)
        plt.close(fig)

        # Figure 2: Welch and LPSD ASD estimates for both channels.
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
        for j, ax in enumerate(axes):
            ax.loglog(welch.f, welch.values[:, j], label="welch ASD")
            ax.loglog(logsp.f, logsp.values[:, j], label="lpsd ASD")
            lines = LINES_CH0 if j == 0 else LINES_CH1
            for k, freq in enumerate(lines):
                ax.axvline(freq, color="0.5", ls=":", lw=1.0,
                           label="true line" if k == 0 else None)
            ax.set_ylabel("ASD / V/$\\sqrt{\\mathrm{Hz}}$")
            ax.set_title(f"channel {j}")
            ax.legend(loc="lower left", fontsize=8)
        axes[1].set_xlabel("frequency / Hz")
        axes[1].set_xlim(1.0, FS_OUT / 2)
        fig.suptitle("Welch (linear axis) vs LPSD (log axis)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "01_psd_comparison.png", dpi=200)
        plt.close(fig)
    print(f"      -> {FIG_DIR.relative_to(REPO)}/01_timeseries_gaps.png, 01_psd_comparison.png")

    print("[6/6] Exporting spectral data and the summary ...")
    np.savez(
        DATA_DIR / "01_pipeline_spectra.npz",
        f_welch=welch.f, asd_welch=welch.values,
        f_lpsd=logsp.f, asd_lpsd=logsp.values,
        fs_out=np.array(FS_OUT),
    )

    # Detect each injected line in a five-bin neighborhood of its nominal bin.
    def detect_lines(spec: spectral.Spectrum, lines: dict[float, float], ch: int) -> list[float]:
        found = []
        for freq in lines:
            k0 = int(np.argmin(np.abs(spec.f - freq)))
            lo, hi = max(0, k0 - 2), min(spec.f.size, k0 + 3)
            found.append(float(spec.f[lo + int(np.argmax(spec.values[lo:hi, ch]))]))
        return found

    detected = {
        f"ch{j}": {
            "welch": detect_lines(welch, lines, j),
            "lpsd": detect_lines(logsp, lines, j),
        }
        for j, lines in enumerate((LINES_CH0, LINES_CH1))
    }

    meta.log_stage(
        META_PATH, "export", "examples/01_pipeline_timeseries.py", __version__,
        params={"formats": ["png", "npz", "md"]},
        outputs=[
            "artifacts/figures/01_timeseries_gaps.png",
            "artifacts/figures/01_psd_comparison.png",
            "artifacts/data/01_pipeline_spectra.npz",
            "artifacts/reports/01_pipeline_summary.md",
        ],
    )
    n_stages = len(meta.load_meta(META_PATH)["processing"])

    summary = f"""# 01 End-to-end time-series pipeline

## Results
- Raw data: {n} samples x {n_ch} channels at {FS:.0f} Hz with {n_nan} NaNs.
- Processed data: {decimated.shape[0]} samples at {FS_OUT:.0f} Hz; RMS = [{rms[0]:.4f}, {rms[1]:.4f}] V.
- Welch: {welch.f.size} linear bins from {welch.meta['n_segments']} averaged segments.
- LPSD: {logsp.f.size} logarithmic bins spanning {logsp.f[0]:.3f}-{logsp.f[-1]:.1f} Hz.
- Provenance metadata contains {n_stages} processing stages.
- Detected channel-0 lines: {detected['ch0']}.
- Detected channel-1 lines: {detected['ch1']}.

## Generated files
- `artifacts/data/raw/01_precision_run_raw.npz`
- `artifacts/data/meta/01_precision_run.meta.toml`
- `artifacts/figures/01_timeseries_gaps.png`
- `artifacts/figures/01_psd_comparison.png`
- `artifacts/data/01_pipeline_spectra.npz`
- `artifacts/reports/01_pipeline_summary.md`
"""
    (REPORT_DIR / "01_pipeline_summary.md").write_text(summary, encoding="utf-8")

    print("---- Summary ----")
    print(f"Detected lines: ch0 {detected['ch0']} Hz, ch1 {detected['ch1']} Hz")
    print(f"Metadata stages: {n_stages}. Example 01 complete.")


if __name__ == "__main__":
    main()
