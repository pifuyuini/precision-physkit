# precision-physkit pipeline contract

## Directory roles

| Path | Contract |
|---|---|
| `data/raw/` | Immutable acquired inputs |
| `data/meta/` | One `<name>.meta.toml` record per dataset |
| `artifacts/data/` | Derived NPZ, JSON, or CSV data |
| `artifacts/figures/` | Figures; PDF preferred, PNG for previews |
| `artifacts/reports/` | Markdown summaries |
| `reference/` | Curated comparison snapshots, not runtime output |

Related files share an `NN_` prefix, for example
`data/raw/01_run_raw.npz`, `data/meta/01_run.meta.toml`,
`artifacts/data/01_spectra.npz`, and `artifacts/figures/01_spectra.pdf`.

## Stages

1. **Register:** create and save metadata, then log acquisition.
2. **Preprocess:** commonly `fill_gaps` → task-specific `lowpass` →
   `downsample` or `resample`; log each operation immediately.
3. **Analyze:** choose spectral, fitting, optimization, or peak functions and
   record all estimator parameters.
4. **Export:** write derived files below `artifacts/`, then log one `export`
   entry listing every output.
5. **Audit:** compare metadata processing outputs with files on disk.

## Minimal example

```python
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from precision_physkit import __version__, filters, meta, plotting, preprocess, spectral

raw_path = Path("data/raw/02_demo_raw.npz")
meta_path = Path("data/meta/02_demo.meta.toml")
fs, fs_out = 2048.0, 1024.0
np.savez(raw_path, t=t, data=data, fs=np.array(fs))

doc = meta.create_meta(
    "02-demo", files=[raw_path.name], format="npz", fs=fs,
    n_samples=data.shape[0], t_start=float(t[0]), t_end=float(t[-1]),
    channels=[{"name": "ch0", "unit": "V", "quantity": "voltage"}],
)
meta.save_meta(doc, meta_path)
meta.log_stage(meta_path, "acquire", "pipeline", __version__,
               params={"fs": fs}, outputs=[str(raw_path)])

filled = np.column_stack([
    preprocess.fill_gaps(data[:, j], fill_noise=True, seed=j)
    for j in range(data.shape[1])
])
meta.log_stage(meta_path, "fill_gaps", "precision_physkit.preprocess.fill_gaps",
               __version__, params={"method": "pchip", "fill_noise": True, "seed": 0})

filtered = filters.lowpass(filled.T, fs, cutoff=480.0, order=4).T
meta.log_stage(meta_path, "lowpass", "precision_physkit.filters.lowpass",
               __version__, params={"cutoff": 480.0, "order": 4})

reduced = preprocess.downsample(filtered, 2, axis=0)
spec = spectral.lpsd(reduced, fs_out, Jdes=150, Kdes=60).to_asd()
np.savez("artifacts/data/02_spectra.npz", f=spec.f, asd=spec.values)

style = plotting.get_adaptive_subplot_style()
with plotting.temp_style(["tab4"], extra_style=style):
    fig, ax = plt.subplots()
    ax.loglog(spec.f, spec.values[:, 0])
    fig.savefig("artifacts/figures/02_asd.pdf", bbox_inches="tight")
    plt.close(fig)

meta.log_stage(
    meta_path, "export", "pipeline", __version__,
    params={"formats": ["npz", "pdf"]},
    outputs=["artifacts/data/02_spectra.npz", "artifacts/figures/02_asd.pdf"],
)
```

Always export `Spectrum.f` with its values. Random operations must use a fixed
seed recorded in metadata.

## Audit and resume

```python
doc = meta.load_meta("data/meta/02_demo.meta.toml")
done = {entry["stage"] for entry in doc["processing"]}
outputs = [path for entry in doc["processing"] for path in entry["outputs"]]
if "downsample" not in done:
    ...  # run only the missing stage
```

Metadata is the durable pipeline state. Do not overwrite raw data or repeat a
completed stage without explicitly recording why.

## See also

[Metadata](api-meta.md) · [Preprocessing](api-preprocess.md) ·
[Filters](api-filters.md) · [Spectra](api-spectral.md) ·
[Plotting](api-plotting.md)
