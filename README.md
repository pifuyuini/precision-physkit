# precision-physkit

`precision-physkit` is a Python toolkit for precision-measurement data,
uniformly sampled time-series signals, and spectral analysis. It combines a
Python API for analysis workflows with a Rust extension for numerically
intensive routines.

## Status

This project is distributed from source only. Prebuilt wheels are not currently
built or published, so installation compiles the Rust extension locally.

The public Python package is imported as `precision_physkit`.

## Features

- Dataset metadata and auditable processing-stage records
- Resampling, interpolation, gap filling, whitening, and digital filters
- Welch and logarithmic-frequency spectral estimators
- Power and cross spectra, coherence, and transfer-function estimation
- Curve fitting, linear least squares, optimization, and peak analysis
- Plotting helpers and deterministic, end-to-end scientific examples

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A stable [Rust toolchain](https://rustup.rs/) and the platform C linker

On Linux, the standard compiler toolchain is typically provided by the
distribution's build-essential package. On macOS, install the Xcode Command
Line Tools. On Windows, install the Microsoft C++ Build Tools.

## Install from source

```bash
git clone https://github.com/pifuyuini/precision-physkit.git
cd precision-physkit
uv sync --locked
```

`uv sync` creates the project environment and invokes the maturin build backend
to compile and install `precision_physkit._core`.

Confirm the installation:

```bash
uv run python -c "import precision_physkit, precision_physkit._core; print(precision_physkit.__version__)"
```

## Quickstart

The following example estimates the amplitude spectral density of a synthetic
50 Hz signal:

```python
import numpy as np

from precision_physkit import spectral

fs = 1024.0
t = np.arange(int(4 * fs)) / fs
x = np.sin(2 * np.pi * 50.0 * t)

asd = spectral.welch_psd(x, fs, nperseg=1024).to_asd()
peak_frequency = asd.f[np.argmax(asd.values)]

print(f"Peak frequency: {peak_frequency:.1f} Hz")
```

## Examples and reference results

The scripts in [`examples/`](examples/) cover the complete data-processing
pipeline, logarithmic spectral analysis, fitting and optimization, peak
analysis, and preprocessing. Run an example from the repository root:

```bash
uv run python examples/01_pipeline_timeseries.py
```

Examples write generated files under `artifacts/` by default, leaving the
repository's versioned reference material unchanged. The reproducibility
snapshot in [`reference/`](reference/) contains the data provenance, numerical
outputs, figures, and concise reports associated with the documented examples.

## Tests

Run both the Python and Rust test suites:

```bash
uv run pytest -q
uv run cargo test --manifest-path rust/Cargo.toml --locked
```

Formatting and lint checks:

```bash
uv run ruff check src tests examples
cargo fmt --manifest-path rust/Cargo.toml --check
```

## Documentation

- [User guide](docs/user-guide.md)
- [API reference](docs/api/)
- [Architecture and data provenance](docs/architecture.md)

## Citation

For research use, cite the specific tagged GitHub release used in the analysis,
including its version and repository URL. When an archival DOI becomes
available, prefer the DOI record while still reporting the exact software
version. Do not cite a mutable branch such as `main` as the software version.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow. Please
report vulnerabilities according to [SECURITY.md](SECURITY.md).

## License

This project is licensed under the [MIT License](LICENSE).
