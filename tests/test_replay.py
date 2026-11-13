"""Offline byte and transaction checks for published-artifact replay."""

import hashlib
import importlib
import io
import json
import os
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from safetensors import TensorSpec, serialize_file

from llm_loop_fix import replay
from llm_loop_fix.replay import audit_safetensors, replay_sparse_edit


@pytest.mark.parametrize("module_name,blocked", [
    ("llm_loop_fix", ("torch", "numpy", "safetensors", "transformers")),
    ("llm_loop_fix.records", ("torch", "numpy", "safetensors", "transformers")),
    ("llm_loop_fix.cli", ("torch", "numpy", "safetensors", "transformers")),
    ("llm_loop_fix.replay", ("torch", "transformers")),
])
def test_offline_package_import_boundary(module_name, blocked):
    source = str(Path(__file__).resolve().parents[1] / "src")
    script = f"""
import importlib
import sys
sys.path.insert(0, {source!r})
blocked = {blocked!r}
def guard(event, arguments):
    if event == 'import' and arguments[0].split('.')[0] in blocked:
        raise AssertionError('forbidden offline import: ' + arguments[0])
sys.addaudithook(guard)
module = importlib.import_module({module_name!r})
assert not set(blocked).intersection(sys.modules)
if {module_name!r} == 'llm_loop_fix':
    assert set(module.__all__).issubset(dir(module))
    assert {{'detector', 'localize', 'manifest', 'patch', 'probe', 'verify'}}.issubset(dir(module))
    try:
        getattr(module, 'unknown_offline_export')
    except AttributeError:
        pass
    else:
        raise AssertionError('unknown package export accepted')
    try:
        module.PatchConfig
    except AssertionError as error:
        assert 'forbidden offline import:' in str(error)
    else:
        raise AssertionError('numerical lookup bypassed import guard')
    assert 'PatchConfig' not in module.__dict__
    assert not set(blocked).intersection(sys.modules)
if {module_name!r} == 'llm_loop_fix.cli':
    import contextlib
    import io
    cases = [(['--help'], 0, 'diagnose-records'),
             (['diagnose-records'], 2, 'required'),
             (['run', '--model', 'synthetic', '--prompts', 'unused', '--output-dir', 'never-export'], 2, 'run is disabled')]
    for arguments, status, marker in cases:
        output, errors = io.StringIO(), io.StringIO()
        sys.argv = ['llm-loop-fix'] + arguments
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            try:
                module.main()
            except SystemExit as error:
                assert error.code == status
            else:
                raise AssertionError('CLI exit guard was bypassed')
        assert marker in output.getvalue() + errors.getvalue()
        assert not set(blocked).intersection(sys.modules)
    sys.argv = ['llm-loop-fix', 'diagnose', '--model', 'synthetic', '--prompts', 'unused']
    try:
        module.main()
    except AssertionError as error:
        assert 'forbidden offline import: torch' in str(error)
    else:
        raise AssertionError('experimental diagnosis lost numerical entry')
    assert not set(blocked).intersection(sys.modules)
"""
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True,
                            timeout=20, env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name,owner", [
    ("LoopAnalysis", "detector"), ("detect_loop_onset", "detector"),
    ("TrajectoryActivations", "probe"), ("collect_contrastive_activations", "probe"),
    ("CulpritMatrix", "localize"), ("LocalizationReport", "localize"), ("localize_loop_culprits", "localize"),
    ("BaselineTensorIdentity", "patch"), ("ModelBaseline", "patch"), ("capture_model_baseline", "patch"),
    ("MatrixEditMetrics", "patch"), ("PatchBudget", "patch"), ("PatchConfig", "patch"),
    ("PatchProposal", "patch"), ("PatchResult", "patch"), ("PatchTarget", "patch"),
    ("apply_patch_proposal", "patch"), ("apply_surgical_patch", "patch"), ("freeze_patch_proposal", "patch"),
    ("restore_weights", "patch"), ("SweepCandidate", "verify"), ("SweepResult", "verify"),
    ("patch_trial", "verify"), ("run_alpha_sweep", "verify"), ("MANIFEST_SCHEMA", "manifest"),
    ("build_edit_manifest", "manifest"), ("write_edit_manifest", "manifest"),
    ("load_and_validate_manifest", "manifest"), ("validate_edit_manifest", "manifest"),
])
def test_package_root_export_identity_and_cache(name, owner):
    package = importlib.import_module("llm_loop_fix")
    expected = getattr(importlib.import_module("llm_loop_fix." + owner), name)
    assert name in package.__all__ and name in dir(package)
    assert getattr(package, name) is expected
    assert package.__dict__[name] is expected
    assert getattr(package, name) is expected


@pytest.mark.parametrize("name", ["detector", "localize", "manifest", "patch", "probe", "verify"])
def test_package_root_submodule_identity_and_cache(name):
    package = importlib.import_module("llm_loop_fix")
    expected = importlib.import_module("llm_loop_fix." + name)
    assert package.__getattr__(name) is expected
    assert getattr(package, name) is expected
    assert package.__dict__[name] is expected
    assert name in dir(package)


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


def test_wp08_validated_checkpoint_export_preserves_auxiliary_mtp_tensors(tmp_path):
    import torch
    import torch.nn as nn
    from llm_loop_fix.localize import CulpritMatrix, LocalizationReport
    from llm_loop_fix.manifest import (
        CheckpointCoveragePlan,
        export_validated_checkpoint,
        load_and_validate_export_manifest,
        validate_export_manifest,
    )
    from llm_loop_fix.patch import (
        PatchBudget,
        PatchConfig,
        apply_patch_proposal,
        capture_model_baseline,
        freeze_patch_proposal,
    )

    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.down_proj = nn.Linear(4, 4, bias=False)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 4)
            self.layers = nn.ModuleList([TinyBlock()])

    torch.manual_seed(0)
    model = TinyModel().to(dtype=torch.bfloat16).eval()
    base_ckpt = tmp_path / "base.safetensors"

    def to_u16(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().contiguous().reshape(-1).view(torch.uint16).numpy().reshape(t.shape)

    mtp_bits = np.array([[16256, 16384], [49024, 0]], dtype="<u2")
    raw_tensors = {
        "model.embed_tokens.weight": to_u16(model.embed_tokens.weight),
        "lm_head.weight": to_u16(model.embed_tokens.weight),
        "model.layers.0.mlp.down_proj.weight": to_u16(model.layers[0].mlp.down_proj.weight),
        "mtp.layers.0.mlp.down_proj.weight": mtp_bits.copy(),
    }
    _save_bits(base_ckpt, raw_tensors)
    base_file_sha = hashlib.sha256(base_ckpt.read_bytes()).hexdigest()
    base_audit = audit_safetensors(base_ckpt)

    baseline = capture_model_baseline(model)
    budget = PatchBudget(6.0, 1, 1, 16, 16, 1.0, 1.0)
    report = LocalizationReport(
        layer_directions={0: torch.tensor([1.0, 0.0, 0.0, 0.0])},
        layer_separation_norms={0: 2.5},
        ranked_culprits=[
            CulpritMatrix("layers.0.mlp.down_proj", 0, [4, 4], 0.5, 1.0, 1.0, [0, 1], [0.6, 0.4], 2, 4),
        ],
    )
    proposal = freeze_patch_proposal(
        report,
        PatchConfig(alpha=1.5, top_k_matrices=1, mode="top_k_columns", top_k_columns=2, budget=budget, baseline=baseline),
    )
    confirmation_report = {
        "schema_version": 1,
        "track": "validated_repair",
        "outcome": "patched",
        "accepted": True,
        "confirmation_evaluation_sha256": "a" * 64,
        "proposal_sha256": proposal.sha256,
        "baseline_sha256": baseline.sha256,
        "rejection_reasons": (),
    }

    incomplete_coverage = CheckpointCoveragePlan(
        registered_parameter_map={
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.layers.0.mlp.down_proj.weight": "layers.0.mlp.down_proj.weight",
        },
        tied_parameter_aliases={"lm_head.weight": "embed_tokens.weight"},
        auxiliary_passthrough_tensors=(),
    )

    apply_patch_proposal(model, proposal)

    # Missing mtp.* tensor coverage must fail closed without creating output or staging artifacts
    out_dir = tmp_path / "exported"
    with pytest.raises(ValueError, match="Incomplete or inconsistent checkpoint tensor coverage"):
        export_validated_checkpoint(
            base_checkpoint_path=base_ckpt,
            expected_base_file_sha256=base_file_sha,
            model=model,
            proposal=proposal,
            confirmation_report=confirmation_report,
            coverage=incomplete_coverage,
            output_dir=out_dir,
        )
    assert not out_dir.exists()
    assert list(tmp_path.glob("*.staging")) == []

    # Unaccepted confirmation report must be refused
    full_coverage = CheckpointCoveragePlan(
        registered_parameter_map={
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.layers.0.mlp.down_proj.weight": "layers.0.mlp.down_proj.weight",
        },
        tied_parameter_aliases={"lm_head.weight": "embed_tokens.weight"},
        auxiliary_passthrough_tensors=("mtp.layers.0.mlp.down_proj.weight",),
    )
    rejected_report = {**confirmation_report, "accepted": False, "outcome": "validation_failed", "rejection_reasons": ("clean_perplexity_ratio_exceeded",)}
    with pytest.raises(ValueError, match="accepted validated_repair confirmation report"):
        export_validated_checkpoint(
            base_checkpoint_path=base_ckpt,
            expected_base_file_sha256=base_file_sha,
            model=model,
            proposal=proposal,
            confirmation_report=rejected_report,
            coverage=full_coverage,
            output_dir=out_dir,
        )
    assert not out_dir.exists()

    # Complete coverage export succeeds atomically and preserves auxiliary mtp.* tensor bit-for-bit
    exported = export_validated_checkpoint(
        base_checkpoint_path=base_ckpt,
        expected_base_file_sha256=base_file_sha,
        model=model,
        proposal=proposal,
        confirmation_report=confirmation_report,
        coverage=full_coverage,
        output_dir=out_dir,
    )
    assert out_dir.is_dir()
    assert list(tmp_path.glob("*.staging")) == []
    assert exported["coverage"]["auxiliary_passthrough_tensors"] == ["mtp.layers.0.mlp.down_proj.weight"]

    loaded_manifest = load_and_validate_export_manifest(out_dir / "export_manifest.json")
    assert loaded_manifest == exported

    fresh_ckpt = audit_safetensors(out_dir / "model.safetensors")
    assert fresh_ckpt["tensors"]["mtp.layers.0.mlp.down_proj.weight"] == base_audit["tensors"]["mtp.layers.0.mlp.down_proj.weight"]
    assert fresh_ckpt["tensors"]["lm_head.weight"] == base_audit["tensors"]["lm_head.weight"]
    assert fresh_ckpt["tensors"]["model.embed_tokens.weight"] == base_audit["tensors"]["model.embed_tokens.weight"]
    assert fresh_ckpt["tensors"]["model.layers.0.mlp.down_proj.weight"]["sha256"] != base_audit["tensors"]["model.layers.0.mlp.down_proj.weight"]["sha256"]

    # Refuses accidental overwrite of existing output_dir
    with pytest.raises(FileExistsError):
        export_validated_checkpoint(
            base_checkpoint_path=base_ckpt,
            expected_base_file_sha256=base_file_sha,
            model=model,
            proposal=proposal,
            confirmation_report=confirmation_report,
            coverage=full_coverage,
            output_dir=out_dir,
        )

    # Tampering with manifest counts fails self-digest & cross-field checks
    tampered = dict(loaded_manifest)
    tampered["coverage"] = dict(tampered["coverage"], auxiliary_passthrough_count=0)
    with pytest.raises(ValueError, match="inconsistent"):
        validate_export_manifest(tampered)
