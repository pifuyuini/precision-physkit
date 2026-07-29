# `precision_physkit.meta` API

Metadata schema 1 records dataset identity, source, contents, and append-only
processing history.

```python
from precision_physkit import meta
```

## Schema

```toml
schema_version = 1

[id]
uuid = "..."
name = "run-001"
created_at = "2026-07-20T08:00:00+00:00"

[source]
instrument = "daq-01"
experiment = "noise-floor survey"
operator = ""
description = ""

[data]
files = ["run-001_raw.npz"]
format = "npz"
fs = 2048.0
n_samples = 122880
t_start = 0.0
t_end = 60.0

[[data.channels]]
name = "sensor_a"
unit = "V"
quantity = "voltage"

[[processing]]
stage = "lowpass"
tool = "precision_physkit.filters.lowpass"
tool_version = "0.1.0"
params = { cutoff = 480.0, order = 4 }
at = "2026-07-20T08:05:00+00:00"
outputs = ["artifacts/data/run-001_lowpass.npz"]
```

Only `schema_version`, `id`, `source`, `data`, and `processing` are accepted at
the top level. Sample rate must be positive, sample count non-negative, time
ranges ordered, channel names non-empty, and all timestamps ISO-8601 parseable.

## Functions

### `create_meta`

```python
meta.create_meta(
    name, *,
    instrument="", experiment="", operator="", description="",
    files=None, format="csv", fs=None, n_samples=None,
    t_start=None, t_end=None, channels=None,
)
```

Creates a schema-1 dictionary with a generated UUID, current UTC timestamp, and
empty processing history. Unknown optional numeric fields should remain `None`.

### `validate_meta`

```python
problems = meta.validate_meta(document)
```

Returns `list[str]`; an empty list means valid. Invalid input is reported rather
than raised.

### `save_meta` and `load_meta`

```python
meta.save_meta(document, path)
document = meta.load_meta(path)
```

Both validate. `save_meta` raises `ValueError` before writing an invalid document.
`load_meta` can raise `FileNotFoundError`, `tomllib.TOMLDecodeError`, or
`ValueError`.

### `log_stage`

```python
document = meta.log_stage(
    path_or_dict, stage, tool, tool_version, params=None, outputs=None,
)
```

Appends a UTC-timestamped processing entry. A path is loaded, updated, validated,
and saved; a dictionary is updated in place. `stage` must be non-empty. Parameters
may contain only strings, numbers, booleans, and nested lists or dictionaries of
those types. Convert NumPy scalars first. Use fully qualified tool names and
`precision_physkit.__version__`.

## Provenance pattern

```python
from precision_physkit import __version__, meta

doc = meta.create_meta(
    "run-001", files=["run-001_raw.npz"], format="npz",
    fs=2048.0, n_samples=122880,
    channels=[{"name": "sensor_a", "unit": "V", "quantity": "voltage"}],
)
assert meta.validate_meta(doc) == []
path = "data/meta/run-001.meta.toml"
meta.save_meta(doc, path)
meta.log_stage(
    path, "acquire", "acquisition", __version__,
    params={"fs": 2048.0}, outputs=["data/raw/run-001_raw.npz"],
)
```

Every dataset needs metadata, and every operation—including export—needs a
processing entry. See [the pipeline contract](api-pipeline.md).
