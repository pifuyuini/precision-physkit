"""Tests for precision_physkit.meta: TOML metadata identity documents."""

import copy
import uuid

import pytest

from precision_physkit import meta


def _full_meta():
    return meta.create_meta(
        "run-001",
        instrument="scope",
        experiment="calibration",
        operator="tester",
        description="synthetic run",
        files=["run-001.csv"],
        format="csv",
        fs=1000.0,
        n_samples=100000,
        t_start=0.0,
        t_end=100.0,
        channels=[
            {"name": "ch1", "unit": "V", "quantity": "voltage"},
            {"name": "ch2"},  # unit/quantity default to ""
        ],
    )


def test_create_and_validate():
    d = _full_meta()
    assert meta.validate_meta(d) == []
    assert d["schema_version"] == meta.SCHEMA_VERSION == 1
    # Generated identity fields are well-formed.
    uuid.UUID(d["id"]["uuid"])
    assert d["id"]["name"] == "run-001"
    assert d["processing"] == []
    # Channel defaults are filled in.
    assert d["data"]["channels"][1] == {"name": "ch2", "unit": "", "quantity": ""}


def test_create_rejects_invalid_fields():
    with pytest.raises(ValueError):
        meta.create_meta("")  # empty name
    with pytest.raises(ValueError):
        meta.create_meta("x", fs=0.0)
    with pytest.raises(ValueError):
        meta.create_meta("x", n_samples=-1)
    with pytest.raises(ValueError):
        meta.create_meta("x", t_start=2.0, t_end=1.0)
    with pytest.raises(ValueError):
        meta.create_meta("x", channels=[{"unit": "V"}])  # missing name
    with pytest.raises(TypeError):
        meta.create_meta("x", channels=["not-a-dict"])


def test_save_load_roundtrip(tmp_path):
    d = _full_meta()
    path = tmp_path / "run-001.meta.toml"
    meta.save_meta(d, path)
    loaded = meta.load_meta(path)
    assert meta.validate_meta(loaded) == []
    # Every field that was actually set round-trips exactly, with types.
    assert loaded["id"] == d["id"]
    assert loaded["source"] == d["source"]
    assert loaded["schema_version"] == d["schema_version"]
    assert loaded["data"]["files"] == ["run-001.csv"]
    assert isinstance(loaded["data"]["fs"], float)
    assert loaded["data"]["fs"] == 1000.0
    assert isinstance(loaded["data"]["n_samples"], int)
    assert loaded["data"]["n_samples"] == 100000
    assert loaded["data"]["t_start"] == 0.0
    assert loaded["data"]["t_end"] == 100.0
    # Channels survive as an array of tables.
    assert loaded["data"]["channels"] == d["data"]["channels"]


def test_optional_none_fields_are_omitted(tmp_path):
    """None-valued optional fields are omitted from the TOML document."""
    d = meta.create_meta("minimal", fs=1000.0)
    path = tmp_path / "minimal.toml"
    meta.save_meta(d, path)
    text = path.read_text(encoding="utf-8")
    assert "n_samples" not in text
    assert "t_start" not in text
    loaded = meta.load_meta(path)
    # Absent keys read back as None via .get and still validate.
    assert loaded["data"].get("n_samples") is None
    assert loaded["data"].get("t_start") is None
    assert loaded["data"]["fs"] == 1000.0
    assert meta.validate_meta(loaded) == []


def test_validate_reports_problems():
    d = _full_meta()
    bad = copy.deepcopy(d)
    del bad["id"]["name"]
    problems = meta.validate_meta(bad)
    assert any("id.name" in p for p in problems)

    bad_fs = copy.deepcopy(d)
    bad_fs["data"]["fs"] = -1.0
    problems = meta.validate_meta(bad_fs)
    assert any("data.fs" in p for p in problems)

    bad_version = copy.deepcopy(d)
    bad_version["schema_version"] = 99
    problems = meta.validate_meta(bad_version)
    assert any("schema_version" in p for p in problems)

    bad_uuid = copy.deepcopy(d)
    bad_uuid["id"]["uuid"] = "not-a-uuid"
    problems = meta.validate_meta(bad_uuid)
    assert any("id.uuid" in p for p in problems)


def test_save_refuses_invalid_document(tmp_path):
    d = _full_meta()
    d["data"]["fs"] = -5.0
    with pytest.raises(ValueError):
        meta.save_meta(d, tmp_path / "bad.toml")
    # Nothing was written.
    assert not (tmp_path / "bad.toml").exists()


def test_log_stage_dict_appends():
    d = _full_meta()
    out = meta.log_stage(
        d,
        "lowpass",
        "precision_physkit.filters.lowpass",
        "0.1.0",
        params={"cutoff": 50.0, "order": 4},
        outputs=["run-001-lp.csv"],
    )
    assert out is d  # dict updated in place and returned
    assert len(d["processing"]) == 1
    entry = d["processing"][0]
    assert entry["stage"] == "lowpass"
    assert entry["tool"] == "precision_physkit.filters.lowpass"
    assert entry["params"] == {"cutoff": 50.0, "order": 4}
    assert entry["outputs"] == ["run-001-lp.csv"]
    assert isinstance(entry["at"], str) and entry["at"]
    assert meta.validate_meta(d) == []


def test_log_stage_path_roundtrip(tmp_path):
    """log_stage on a path updates the file; the result reloads cleanly."""
    path = tmp_path / "run.toml"
    meta.save_meta(_full_meta(), path)
    meta.log_stage(path, "downsample", "precision_physkit.preprocess.downsample", "0.1.0",
                   params={"q": 4})
    meta.log_stage(path, "whiten", "precision_physkit.preprocess.whiten", "0.1.0",
                   outputs=["run-001-white.csv"])
    loaded = meta.load_meta(path)
    assert [e["stage"] for e in loaded["processing"]] == ["downsample", "whiten"]
    assert loaded["processing"][0]["params"] == {"q": 4}
    assert loaded["processing"][1]["outputs"] == ["run-001-white.csv"]
    assert meta.validate_meta(loaded) == []


def test_log_stage_rejects_bad_input(tmp_path):
    d = _full_meta()
    with pytest.raises(ValueError):
        meta.log_stage(d, "", "tool", "1.0")  # empty stage
    with pytest.raises(TypeError):
        meta.log_stage(123, "stage", "tool", "1.0")  # unsupported target
    with pytest.raises(TypeError):
        meta.log_stage(d, "stage", "tool", "1.0", params={"bad": object()})


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        meta.load_meta(tmp_path / "does-not-exist.toml")
