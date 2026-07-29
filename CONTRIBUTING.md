# Contributing

Contributions that improve correctness, reproducibility, documentation, and
scientific usability are welcome.

## Development setup

Install Python 3.12 or newer, uv, a stable Rust toolchain, and the platform C
linker. Then run:

```bash
git clone https://github.com/pifuyuini/precision-physkit.git
cd precision-physkit
uv sync --locked
```

To rebuild the extension explicitly during development:

```bash
uv run maturin develop --uv
```

## Before opening a pull request

Run the relevant checks:

```bash
uv run ruff check src tests examples
cargo fmt --manifest-path rust/Cargo.toml --check
uv run cargo test --manifest-path rust/Cargo.toml --locked
uv run pytest -q
```

Keep changes focused and include tests for behavior changes. Changes to
scientific algorithms must document assumptions, preserve deterministic random
seeds where applicable, and update the relevant API documentation and reference
metrics. Generated example output belongs under `artifacts/`; do not overwrite
the versioned `reference/` snapshot as part of an unrelated change.

Please describe the motivation, implementation, validation performed, and any
numerical or platform-dependent limitations in the pull request.
