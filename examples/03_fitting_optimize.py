"""Example 03: parameter fitting and numerical optimization.

The example covers nonlinear curve fitting, truncated-SVD linear least
squares, iterative multi-channel estimation, and global versus local
optimization. Generated files are written below ``artifacts/``.

Run from the repository root::

    uv run python examples/03_fitting_optimize.py

"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from precision_physkit import fitting, optimize, plotting

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "artifacts" / "figures"
DATA_DIR = REPO / "artifacts" / "data"
REPORT_DIR = REPO / "artifacts" / "reports"

SEED = 20260722


def _jsonable(obj):
    """Convert NumPy scalars and arrays into JSON-compatible values."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def demo_curve_fit(rng: np.random.Generator) -> dict:
    """Fit a noisy exponentially decaying sinusoid."""
    true = {"amp": 2.5, "tau": 1.5, "freq": 3.0, "phase": 0.7}
    t = np.linspace(0.0, 6.0, 600)
    sigma = 0.15

    def damped_sine(t, amp, tau, freq, phase):
        return amp * np.exp(-t / tau) * np.sin(2.0 * np.pi * freq * t + phase)

    y_clean = damped_sine(t, **true)
    y = y_clean + rng.normal(0.0, sigma, t.size)
    res = fitting.curve_fit(
        damped_sine, t, y, p0=[2.0, 1.0, 2.5, 0.0],
        sigma=np.full(t.size, sigma), absolute_sigma=True,
    )
    assert res.success, res.message
    return {
        "true": true, "fitted": res.params, "perr": res.perr,
        "redchi": res.stats["redchi"], "r2": res.stats["r2"],
        "noise_sigma": sigma,
        "_plot": (t, y, y_clean, damped_sine(t, *res.params.values())),
    }


def demo_linear_lstsq(rng: np.random.Generator) -> dict:
    """Compare rcond truncation and input scaling for a Vandermonde system."""
    deg, n = 10, 200
    x = np.linspace(0.0, 10.0, n)
    coef_true = np.zeros(deg + 1)
    coef_true[:3] = [1.0, 0.5, -0.02]          # The true signal is quadratic.
    design_raw = np.vander(x, N=deg + 1, increasing=True)
    y = design_raw @ coef_true + rng.normal(0.0, 0.1, n)

    # Evaluate interpolation quality against the noiseless truth.
    xf = np.linspace(0.0, 10.0, 2001)
    y_true_fine = np.vander(xf, N=deg + 1, increasing=True) @ coef_true

    def evaluate(design: np.ndarray, design_fine: np.ndarray, rcond):
        res = fitting.linear_lstsq(design, y, rcond=rcond)
        theta = np.array([res.params[f"p{i}"] for i in range(deg + 1)])
        pred_rms = float(np.sqrt(np.mean((design_fine @ theta - y_true_fine) ** 2)))
        return {
            "rcond": rcond, "rank": res.stats["rank"],
            "pred_rms": pred_rms, "perr_max": float(max(res.perr.values())),
            "theta": theta,
        }

    vander_fine = np.vander(xf, N=deg + 1, increasing=True)
    variants = {
        "raw_rcond_None": evaluate(design_raw, vander_fine, None),
        "raw_rcond_1e-10": evaluate(design_raw, vander_fine, 1e-10),
        "raw_rcond_1e-8": evaluate(design_raw, vander_fine, 1e-8),
    }
    # Scale the input to improve conditioning.
    xs, xsf = (x - x.mean()) / x.std(), (xf - x.mean()) / x.std()
    variants["scaled_rcond_None"] = evaluate(
        np.vander(xs, N=deg + 1, increasing=True),
        np.vander(xsf, N=deg + 1, increasing=True), None,
    )
    return {
        "deg": deg, "cond_raw": float(np.linalg.cond(design_raw)),
        "cond_scaled": float(np.linalg.cond(np.vander(xs, N=deg + 1, increasing=True))),
        "coef_true": coef_true, "noise_sigma": 0.1,
        "variants": {k: {kk: vv for kk, vv in v.items() if kk != "theta"}
                     for k, v in variants.items()},
        "_plot": (x, y, coef_true, {k: v["theta"] for k, v in variants.items()}),
    }


def demo_iterative_multichannel(rng: np.random.Generator) -> dict:
    """Estimate shared parameters from channels with different noise levels."""
    t = np.linspace(0.0, 5.0, 400)
    basis = np.column_stack([
        np.ones_like(t), t, np.sin(2.0 * np.pi * 0.8 * t), np.cos(2.0 * np.pi * 0.8 * t)
    ])
    theta_true = np.array([1.2, -0.3, 0.8, 0.4])
    sig1, sig2 = 0.05, 1.0
    y1 = basis @ theta_true + rng.normal(0.0, sig1, t.size)
    y2 = basis @ theta_true + rng.normal(0.0, sig2, t.size)
    design = np.vstack([basis, basis])
    y = np.concatenate([y1, y2])

    joint = fitting.iterative_multichannel(design, y, group_sizes=[t.size, t.size])
    unweighted = fitting.linear_lstsq(design, y)
    def to_arr(result: fitting.FitResult) -> np.ndarray:
        return np.array([result.params[f"p{i}"] for i in range(4)])

    def err(result: fitting.FitResult) -> float:
        return float(np.sqrt(np.mean((to_arr(result) - theta_true) ** 2)))
    return {
        "theta_true": theta_true, "channel_sigmas": [sig1, sig2],
        "joint_params": to_arr(joint), "joint_rms_err": err(joint),
        "joint_success": bool(joint.success), "joint_message": joint.message,
        "n_iter": joint.stats["n_iter"], "weights": joint.stats["weights"],
        "unweighted_rms_err": err(unweighted),
        "_plot": (t, basis @ theta_true, y1, y2, basis @ to_arr(joint)),
    }


def demo_global_minimize() -> dict:
    """Compare global Rastrigin optimization with local Nelder-Mead."""
    def rastrigin(v: np.ndarray) -> float:
        v = np.asarray(v)
        return float(10 * v.size + np.sum(v**2 - 10 * np.cos(2 * np.pi * v)))

    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    glob = optimize.global_minimize(rastrigin, bounds, seed=SEED)
    loc = optimize.minimize(rastrigin, [4.5, -3.5])  # The local solve finds a local minimum.
    return {
        "global_x": glob.x, "global_fun": glob.fun,
        "global_success": bool(glob.success), "n_eval": glob.n_eval,
        "local_x": loc.x, "local_fun": loc.fun,
        "_plot": (rastrigin, bounds, glob.x, loc.x),
    }


def main() -> None:
    for d in (FIG_DIR, DATA_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("[1/5] curve_fit: decaying sinusoid ...")
    r1 = demo_curve_fit(rng)
    print(f"      amp {r1['fitted']['amp']:.3f}+/-{r1['perr']['amp']:.3f} (true 2.5), "
          f"tau {r1['fitted']['tau']:.3f}+/-{r1['perr']['tau']:.3f} (true 1.5), "
          f"redchi {r1['redchi']:.3f}")

    print("[2/5] linear_lstsq: ill-conditioned Vandermonde comparison ...")
    r2 = demo_linear_lstsq(rng)
    for name, v in r2["variants"].items():
        print(f"      {name:>18}: rank={v['rank']:2d}, pred_rms={v['pred_rms']:.4f}, "
              f"perr_max={v['perr_max']:.2e}")
    print(f"      cond: raw {r2['cond_raw']:.2e} -> scaled {r2['cond_scaled']:.2e}")

    print("[3/5] iterative_multichannel: joint two-channel estimate ...")
    r3 = demo_iterative_multichannel(rng)
    w = r3["weights"]
    print(f"      weight ratio w1/w2 = {w[0] / w[1]:.0f} ((sigma2/sigma1)^2 = 400); "
          f"joint RMS error {r3['joint_rms_err']:.4f} vs unweighted {r3['unweighted_rms_err']:.4f}")

    print("[4/5] global_minimize: Rastrigin ...")
    r4 = demo_global_minimize()
    print(f"      global: x={np.round(r4['global_x'], 8)}, fun={r4['global_fun']:.2e} "
          f"({r4['n_eval']} evaluations); local NM: fun={r4['local_fun']:.2f}")

    print("[5/5] Rendering the figure and writing JSON and Markdown ...")
    with plotting.temp_style(["ysy_academic", "tab4"]):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
        # (a) Decaying-sinusoid fit.
        t, y, y_clean, y_fit = r1["_plot"]
        ax = axes[0, 0]
        ax.plot(t, y, ".", ms=2, label="noisy data")
        ax.plot(t, y_clean, "--", label="true model")
        ax.plot(t, y_fit, label="curve_fit")
        ax.set_xlabel("t / s")
        ax.set_title("(a) damped-sine curve_fit")
        ax.legend(fontsize=8)

        # (b) Vandermonde coefficient comparison.
        x, yv, coef_true, thetas = r2["_plot"]
        ax = axes[0, 1]
        idx = np.arange(coef_true.size)
        width = 0.25
        ax.semilogy(idx - width, np.abs(coef_true) + 1e-16, "s", label="true")
        ax.semilogy(idx, np.abs(thetas["raw_rcond_None"]) + 1e-16, "o", ms=4,
                    label="rcond=None (rank 11)")
        ax.semilogy(idx + width, np.abs(thetas["raw_rcond_1e-10"]) + 1e-16, "^", ms=4,
                    label="rcond=1e-10 (rank 8)")
        ax.set_xlabel("coefficient index i of $x^i$")
        ax.set_ylabel("$|c_i|$")
        ax.set_xticks(idx)
        ax.set_title("(b) Vandermonde coefficients")
        ax.legend(fontsize=8)

        # (c) Joint two-channel estimate.
        t3, model3, y1, y2, fit3 = r3["_plot"]
        ax = axes[1, 0]
        ax.plot(t3, y2, ".", ms=1.5, color="0.6", label="ch2 ($\\sigma$=1.0)")
        ax.plot(t3, y1, ".", ms=1.5, label="ch1 ($\\sigma$=0.05)")
        ax.plot(t3, model3, "--", label="true model")
        ax.plot(t3, fit3, label="joint IRLS fit")
        ax.set_xlabel("t / s")
        ax.set_title("(c) iterative_multichannel")
        ax.legend(fontsize=8)

        # (d) Rastrigin contours and estimated minima.
        rastrigin, bounds, gx, lx = r4["_plot"]
        ax = axes[1, 1]
        grid = np.linspace(-5.12, 5.12, 300)
        xx, yy = np.meshgrid(grid, grid)
        zz = 20 + xx**2 - 10 * np.cos(2 * np.pi * xx) + yy**2 - 10 * np.cos(2 * np.pi * yy)
        cs = ax.contour(xx, yy, zz, levels=np.logspace(0, 2.6, 12), cmap="viridis")
        ax.clabel(cs, inline=True, fontsize=6)
        ax.plot(*gx, "*", ms=14, label=f"global ({gx[0]:.1e}, {gx[1]:.1e})")
        ax.plot(*lx, "x", ms=9, label=f"local NM ({lx[0]:.2f}, {lx[1]:.2f})")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title("(d) Rastrigin global_minimize")
        ax.legend(fontsize=8, loc="upper right")

        fig.suptitle("precision_physkit fitting & optimize demos")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "03_fitting_optimize.png", dpi=200)
        plt.close(fig)

    results = {
        "seed": SEED,
        "curve_fit": {k: v for k, v in r1.items() if k != "_plot"},
        "linear_lstsq": {k: v for k, v in r2.items() if k != "_plot"},
        "iterative_multichannel": {k: v for k, v in r3.items() if k != "_plot"},
        "global_minimize": {k: v for k, v in r4.items() if k != "_plot"},
    }
    with open(DATA_DIR / "03_fitting_results.json", "w", encoding="utf-8") as fh:
        json.dump(_jsonable(results), fh, indent=2, ensure_ascii=False)

    v = r2["variants"]
    w = r3["weights"]
    summary = f"""# 03 Fitting and optimization

## Results
### 1. curve_fit (truth -> estimate +/- 1 sigma)
- amp: 2.5 -> {r1['fitted']['amp']:.3f} +/- {r1['perr']['amp']:.3f}
- tau: 1.5 -> {r1['fitted']['tau']:.3f} +/- {r1['perr']['tau']:.3f}
- freq: 3.0 -> {r1['fitted']['freq']:.4f} +/- {r1['perr']['freq']:.4f}
- phase: 0.7 -> {r1['fitted']['phase']:.3f} +/- {r1['perr']['phase']:.3f}
- Reduced chi-square = {r1['redchi']:.3f}; R-squared = {r1['r2']:.4f}.

### 2. linear_lstsq (degree-10 Vandermonde, cond = {r2['cond_raw']:.2e})
| Variant | rank | interpolation RMS | maximum parameter uncertainty |
|---|---|---|---|
| raw, rcond=None | {v['raw_rcond_None']['rank']} | {v['raw_rcond_None']['pred_rms']:.4f} | {v['raw_rcond_None']['perr_max']:.2e} |
| raw, rcond=1e-10 | {v['raw_rcond_1e-10']['rank']} | {v['raw_rcond_1e-10']['pred_rms']:.4f} | {v['raw_rcond_1e-10']['perr_max']:.2e} |
| raw, rcond=1e-8 | {v['raw_rcond_1e-8']['rank']} | {v['raw_rcond_1e-8']['pred_rms']:.4f} | {v['raw_rcond_1e-8']['perr_max']:.2e} |
| scaled, rcond=None | {v['scaled_rcond_None']['rank']} | {v['scaled_rcond_None']['pred_rms']:.4f} | {v['scaled_rcond_None']['perr_max']:.2e} |

- Scaled-input condition number: {r2['cond_scaled']:.2e}.

### 3. iterative_multichannel
- Status: {r3['joint_message']}; weights [{w[0]:.4f}, {w[1]:.6f}].
- Parameter RMS error: joint {r3['joint_rms_err']:.4f} vs unweighted {r3['unweighted_rms_err']:.4f}.

### 4. global_minimize (two-dimensional Rastrigin)
- Global: x = ({r4['global_x'][0]:.2e}, {r4['global_x'][1]:.2e}), f = {r4['global_fun']:.2e},
  {r4['n_eval']} evaluations, success = {r4['global_success']}.
- Local Nelder-Mead: x = ({r4['local_x'][0]:.2f}, {r4['local_x'][1]:.2f}), f = {r4['local_fun']:.2f}.

## Generated files
- `artifacts/figures/03_fitting_optimize.png`
- `artifacts/data/03_fitting_results.json`
- `artifacts/reports/03_fitting_summary.md`
"""
    (REPORT_DIR / "03_fitting_summary.md").write_text(summary, encoding="utf-8")
    print("Example 03 complete.")


if __name__ == "__main__":
    main()
