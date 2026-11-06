"""Offline byte and transaction checks for published-artifact replay."""

import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from safetensors import TensorSpec, serialize_file

from llm_loop_fix import replay
from llm_loop_fix.replay import audit_safetensors, replay_sparse_edit


def _save_bits(path, arrays):
    specs = {
        name: TensorSpec(
            dtype="bfloat16",
            shape=array.shape,
            data_ptr=array.ctypes.data,
            data_len=array.nbytes,
        )
        for name, array in arrays.items()
    }
    serialize_file(specs, str(path), metadata={"fixture": "raw-bf16"})


def test_audit_raw_bf16_and_signed_zero(tmp_path):
    bits = np.array([0, 32768, 16256, 49024], dtype="<u2").reshape(2, 2)
    path = tmp_path / "base.safetensors"
    _save_bits(path, {"target": bits, "untouched": bits.copy()})
    original = path.read_bytes()

    report = audit_safetensors(path)

    assert report["size_bytes"] == len(original)
    assert report["file_sha256"] == hashlib.sha256(original).hexdigest()
    assert report["tensors"]["target"] == {
        "dtype": "BF16",
        "shape": [2, 2],
        "sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
    }
    assert report["tensors"]["target"] == report["tensors"]["untouched"]
    assert hashlib.sha256(bits.tobytes()).hexdigest() != hashlib.sha256(
        np.array([0, 0, 16256, 49024], dtype="<u2").tobytes()
    ).hexdigest()
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "payload",
    [b"", struct.pack("<Q", 2 * 1024 * 1024), struct.pack("<Q", 1024) + b"{}"],
)
def test_audit_rejects_invalid_header(tmp_path, payload):
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        audit_safetensors(path)


def _replay_fixture(tmp_path, indices=None, values=None):
    bits = np.array([0, 0, 16256, 49024], dtype="<u2").reshape(2, 2)
    edited = bits.copy().reshape(-1)
    edited[[0, 3]] = [32768, 16256]
    base = tmp_path / "base.safetensors"
    edit = tmp_path / "edit.npz"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "output.safetensors"
    _save_bits(base, {"target": bits, "untouched": bits.copy()})
    np.savez_compressed(
        edit,
        t0_idx=np.array([0, 3], dtype=np.int64) if indices is None else indices,
        t0_val=np.array([32768, 16256], dtype="<u2") if values is None else values,
    )
    manifest.write_text(json.dumps({
        "changed_tensors": 1,
        "changed_entries_total": 2,
        "tensors": [{
            "name": "target", "shape": [2, 2], "dtype": "bfloat16", "form": "sparse",
            "changed_entries": 2, "changed_rows": 2, "changed_columns": 2,
            "base_sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
            "edited_sha256": hashlib.sha256(edited.tobytes()).hexdigest(),
        }],
    }), encoding="utf-8")
    arguments = {
        "expected_base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "expected_edit_sha256": hashlib.sha256(edit.read_bytes()).hexdigest(),
        "expected_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    return (base, edit, manifest, output), arguments


def test_sparse_replay_preserves_baseline_and_untouched_bits(tmp_path):
    paths, arguments = _replay_fixture(tmp_path)
    original = paths[0].read_bytes()

    report = replay_sparse_edit(*paths, **arguments)

    assert report["changed_entries_total"] == 2
    assert report["unchanged_tensors"] == 1
    assert report["tensor_count"] == 2
    assert report["tensors"][0]["changed_columns"] == 2
    assert audit_safetensors(paths[3])["tensors"]["untouched"] == audit_safetensors(paths[0])["tensors"]["untouched"]
    assert paths[0].read_bytes() == original
    assert list(tmp_path.glob("*.staging")) == []


@pytest.mark.parametrize("field", ["expected_base_sha256", "expected_edit_sha256", "expected_manifest_sha256"])
def test_replay_rejects_wrong_input_hash(tmp_path, field):
    paths, arguments = _replay_fixture(tmp_path)
    arguments[field] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        replay_sparse_edit(*paths, **arguments)
    assert not paths[3].exists()
    assert list(tmp_path.glob("*.staging")) == []


@pytest.mark.parametrize("indices", [
    np.array([0, 0]), np.array([-1, 3]), np.array([0, 4]),
    np.array([0.0, 3.0]), np.array([0, 3], dtype=object),
])
def test_replay_rejects_unsafe_sparse_indices(tmp_path, indices):
    paths, arguments = _replay_fixture(tmp_path, indices=indices)
    original = paths[0].read_bytes()
    with pytest.raises(ValueError):
        replay_sparse_edit(*paths, **arguments)
    assert paths[0].read_bytes() == original
    assert not paths[3].exists()
    assert list(tmp_path.glob("*.staging")) == []


def test_replay_never_overwrites_existing_output(tmp_path):
    paths, arguments = _replay_fixture(tmp_path)
    paths[3].write_bytes(b"user-owned output")
    with pytest.raises(FileExistsError):
        replay_sparse_edit(*paths, **arguments)
    assert paths[3].read_bytes() == b"user-owned output"


@pytest.mark.parametrize("field,value", [
    ("dtype", "float16"), ("shape", [4, 1]), ("base_sha256", "0" * 64),
    ("edited_sha256", "0" * 64), ("changed_entries", 1),
])
def test_replay_rejects_tensor_contract_mismatch(tmp_path, field, value):
    paths, arguments = _replay_fixture(tmp_path)
    original = paths[0].read_bytes()
    manifest = json.loads(paths[2].read_text())
    manifest["tensors"][0][field] = value
    paths[2].write_text(json.dumps(manifest), encoding="utf-8")
    arguments["expected_manifest_sha256"] = hashlib.sha256(paths[2].read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        replay_sparse_edit(*paths, **arguments)
    assert paths[0].read_bytes() == original
    assert not paths[3].exists()
    assert list(tmp_path.glob("*.staging")) == []


def test_replay_rejects_huge_npy_shape_before_allocation(tmp_path):
    paths, arguments = _replay_fixture(tmp_path)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<i8", "fortran_order": False, "shape": (2**40,)}
    )
    values = io.BytesIO()
    np.save(values, np.array([32768, 16256], dtype="<u2"), allow_pickle=False)
    with zipfile.ZipFile(paths[1], "w") as archive:
        archive.writestr("t0_idx.npy", header.getvalue())
        archive.writestr("t0_val.npy", values.getvalue())
    arguments["expected_edit_sha256"] = hashlib.sha256(paths[1].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="shape/byte mismatch"):
        replay_sparse_edit(*paths, **arguments)
    assert not paths[3].exists()
    assert list(tmp_path.glob("*.staging")) == []


def test_replay_removes_staging_after_fresh_load_failure(tmp_path, monkeypatch):
    paths, arguments = _replay_fixture(tmp_path)
    original = paths[0].read_bytes()
    actual_audit = replay.audit_safetensors

    def failing_audit(path):
        if path.suffix == ".staging":
            raise RuntimeError("Injected fresh-load failure")
        return actual_audit(path)

    monkeypatch.setattr(replay, "audit_safetensors", failing_audit)
    with pytest.raises(RuntimeError, match="Injected fresh-load failure"):
        replay_sparse_edit(*paths, **arguments)
    assert paths[0].read_bytes() == original
    assert not paths[3].exists()
    assert list(tmp_path.glob("*.staging")) == []


def test_replay_never_overwrites_destination_claimed_before_promotion(tmp_path, monkeypatch):
    paths, arguments = _replay_fixture(tmp_path)
    actual_link = os.link

    def claim_destination(source, destination):
        destination.write_bytes(b"concurrent user output")
        return actual_link(source, destination)

    monkeypatch.setattr(replay.os, "link", claim_destination)
    with pytest.raises(FileExistsError):
        replay_sparse_edit(*paths, **arguments)
    assert paths[3].read_bytes() == b"concurrent user output"
    assert list(tmp_path.glob("*.staging")) == []


def test_replay_command_emits_byte_only_report(tmp_path):
    paths, arguments = _replay_fixture(tmp_path)
    command = [sys.executable, "-m", "llm_loop_fix.replay"]
    for name, path in zip(("base", "edit", "manifest", "output"), paths):
        command.extend([f"--{name}", str(path)])
    for name in ("base", "edit", "manifest"):
        command.extend([f"--{name}-sha256", arguments[f"expected_{name}_sha256"]])
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    report = json.loads(completed.stdout)
    assert report["claim_scope"] == "stored_tensor_bytes_only"
    assert report["changed_entries_total"] == 2
    assert report["output_file_sha256"] == hashlib.sha256(paths[3].read_bytes()).hexdigest()
