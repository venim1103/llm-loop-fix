"""Deterministic JSON provenance and weight-edit manifest auditing (micro_classifier style)."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from .localize import LocalizationReport
from .patch import PatchConfig, PatchProposal, PatchResult, _tensor_sha256, capture_model_baseline
from .replay import _hash_range, _layout, audit_safetensors
from .verify import SweepResult

MANIFEST_SCHEMA = "llm-loop-fix/edit-manifest/1"
CHECKPOINT_EXPORT_SCHEMA = "llm-loop-fix/validated-checkpoint-export/1"


def _unique_object_hook(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in manifest: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value):
    raise ValueError(f"Non-finite JSON constant not allowed in manifest: {value}")


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize object to canonical UTF-8 JSON bytes with sorted keys and no NaN/Inf."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest_object(obj: Any) -> str:
    """Compute SHA-256 hex digest of an object's canonical JSON representation."""
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def build_edit_manifest(
    model_id: str,
    config: PatchConfig,
    patch_result: PatchResult,
    localization: LocalizationReport,
    sweep: Optional[SweepResult] = None,
    prompts_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a verifiable weight-edit manifest recording exact parameter & bitwise changes."""
    prompts_sha256 = None
    if prompts_path and Path(prompts_path).is_file():
        prompts_sha256 = hashlib.sha256(Path(prompts_path).read_bytes()).hexdigest()

    body: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "model_id": model_id,
        "prompts_sha256": prompts_sha256,
        "intervention": {
            "formula": "W_new = W_orig - alpha * outer(u, mask_K * (u @ W_orig))",
            "mode": config.mode,
            "alpha": float(config.alpha),
            "top_k_matrices": int(config.top_k_matrices),
            "top_k_columns": int(config.top_k_columns),
            "sparse_column_fraction": float(config.sparse_column_fraction),
        },
        "summary": {
            "total_model_params": int(patch_result.total_model_params),
            "touched_params": int(patch_result.modified_params_count),
            "bitwise_changed_params": int(patch_result.bitwise_changed_params_count),
            "touched_percentage": round(patch_result.modified_fraction * 100.0, 6),
            "modified_modules": list(patch_result.modified_modules),
        },
        "edited_matrices": [
            {
                "module_name": m.module_name,
                "layer_idx": m.layer_idx,
                "shape": m.shape,
                "num_selected_columns": len(m.selected_columns),
                "selected_columns": m.selected_columns,
                "touched_entries": m.touched_entries,
                "bitwise_changed_entries": m.bitwise_changed_entries,
                "relative_frobenius_change": m.relative_frobenius_change,
                "pre_edit_sha256": m.pre_edit_sha256,
                "post_edit_sha256": m.post_edit_sha256,
            }
            for m in patch_result.matrix_metrics
        ],
        "localization": localization.to_summary_dict(top_k=max(5, config.top_k_matrices)),
    }

    if sweep is not None:
        body["verification"] = {
            "best_alpha": float(sweep.best_alpha),
            "baseline_loop_rate": round(sweep.baseline_loop_rate, 4),
            "best_loop_rate": round(sweep.best_loop_rate, 4),
            "baseline_perplexity": round(sweep.baseline_perplexity, 4),
            "best_perplexity": round(sweep.best_perplexity, 4),
            "rescued_count": int(sweep.best_rescued_count),
            "regressed_count": int(sweep.best_regressed_count),
            "candidates": [
                {
                    "alpha": float(c.alpha),
                    "loop_rate": round(c.loop_rate, 4),
                    "cap_hit_rate": round(c.cap_hit_rate, 4),
                    "whitespace_tail_count": int(c.whitespace_tail_count),
                    "perplexity": round(c.perplexity, 4),
                    "rescued_count": int(c.rescued_count),
                    "regressed_count": int(c.regressed_count),
                }
                for c in sweep.candidates
            ],
        }

    body["manifest_sha256"] = digest_object(body)
    return body


def write_edit_manifest(path: str | Path, manifest: Dict[str, Any]) -> str:
    """Validate and write a manifest JSON file, returning its file SHA-256."""
    validate_edit_manifest(manifest)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    out.write_text(text, encoding="utf-8")
    return hashlib.sha256(out.read_bytes()).hexdigest()


def load_and_validate_manifest(path: str | Path) -> Dict[str, Any]:
    """Load an edit manifest from disk with strict duplicate-key and non-finite guards."""
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(
        raw, object_pairs_hook=_unique_object_hook, parse_constant=_reject_nonfinite
    )
    validate_edit_manifest(data)
    return data


def validate_edit_manifest(manifest: Dict[str, Any]) -> None:
    """Verify schema, required sections, cross-field invariants, and cryptographic digest integrity."""
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Manifest must declare schema '{MANIFEST_SCHEMA}'")
    for req in ("model_id", "intervention", "summary", "edited_matrices", "manifest_sha256"):
        if req not in manifest:
            raise ValueError(f"Manifest missing required field: {req}")
    if not isinstance(manifest["model_id"], str) or not manifest["model_id"].strip():
        raise ValueError("Manifest model_id must be a nonempty string")
    if not isinstance(manifest["intervention"], dict) or not isinstance(manifest["summary"], dict):
        raise ValueError("Manifest intervention and summary sections must be JSON objects")
    edited_matrices = manifest["edited_matrices"]
    if not isinstance(edited_matrices, list):
        raise ValueError("Manifest edited_matrices must be a list")

    summary = manifest["summary"]
    touched = summary.get("touched_params")
    changed = summary.get("bitwise_changed_params")
    total = summary.get("total_model_params")
    mod_modules = summary.get("modified_modules")
    if (
        type(total) is not int
        or type(touched) is not int
        or type(changed) is not int
        or total < 0
        or touched < 0
        or changed < 0
        or changed > touched
        or touched > total
    ):
        raise ValueError("Manifest summary parameter counts are invalid or inconsistent")
    if not isinstance(mod_modules, list) or len(mod_modules) != len(edited_matrices):
        raise ValueError("Manifest modified_modules must match edited_matrices length")

    for entry in edited_matrices:
        if not isinstance(entry, dict):
            raise ValueError("Manifest edited_matrices entries must be JSON objects")
        sel_cols = entry.get("selected_columns")
        num_cols = entry.get("num_selected_columns")
        m_touched = entry.get("touched_entries")
        m_changed = entry.get("bitwise_changed_entries")
        if (
            not isinstance(sel_cols, list)
            or type(num_cols) is not int
            or num_cols != len(sel_cols)
            or type(m_touched) is not int
            or type(m_changed) is not int
            or m_changed < 0
            or m_changed > m_touched
        ):
            raise ValueError("Manifest matrix entry counts or selected_columns are inconsistent")
        for digest_key in ("pre_edit_sha256", "post_edit_sha256"):
            digest_val = entry.get(digest_key)
            if (
                not isinstance(digest_val, str)
                or len(digest_val) != 64
                or any(ch not in "0123456789abcdef" for ch in digest_val)
            ):
                raise ValueError(f"Manifest matrix entry has invalid {digest_key}")

    recorded_hash = manifest["manifest_sha256"]
    if (
        not isinstance(recorded_hash, str)
        or len(recorded_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in recorded_hash)
    ):
        raise ValueError("Manifest SHA-256 digest format is invalid")
    copy_without_hash = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    expected_hash = digest_object(copy_without_hash)
    if recorded_hash != expected_hash:
        raise ValueError("Manifest SHA-256 integrity check failed")


@dataclass(frozen=True)
class CheckpointCoveragePlan:
    """Complete partition of baseline checkpoint tensors across runtime parameters, ties, and auxiliaries."""

    registered_parameter_map: Dict[str, str]
    tied_parameter_aliases: Dict[str, str] = None  # type: ignore[assignment]
    auxiliary_passthrough_tensors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.registered_parameter_map) is not dict or not self.registered_parameter_map:
            raise ValueError("Checkpoint coverage requires a nonempty registered_parameter_map")
        for ckpt_name, runtime_name in self.registered_parameter_map.items():
            if type(ckpt_name) is not str or not ckpt_name or type(runtime_name) is not str or not runtime_name:
                raise ValueError("Registered parameter mapping requires nonempty string keys")
        aliases = {} if self.tied_parameter_aliases is None else self.tied_parameter_aliases
        if type(aliases) is not dict:
            raise ValueError("tied_parameter_aliases must be a dictionary")
        for ckpt_name, target_runtime_name in aliases.items():
            if (
                type(ckpt_name) is not str
                or not ckpt_name
                or type(target_runtime_name) is not str
                or not target_runtime_name
            ):
                raise ValueError("Tied parameter aliases require nonempty string keys")
        object.__setattr__(self, "tied_parameter_aliases", dict(aliases))
        if type(self.auxiliary_passthrough_tensors) not in (tuple, list) or any(
            type(name) is not str or not name for name in self.auxiliary_passthrough_tensors
        ):
            raise ValueError("auxiliary_passthrough_tensors must be nonempty tensor name strings")
        aux_tuple = tuple(sorted(self.auxiliary_passthrough_tensors))
        if len(set(aux_tuple)) != len(aux_tuple):
            raise ValueError("Duplicate auxiliary passthrough tensor names")
        object.__setattr__(self, "auxiliary_passthrough_tensors", aux_tuple)
        reg_keys = set(self.registered_parameter_map)
        alias_keys = set(self.tied_parameter_aliases)
        aux_keys = set(aux_tuple)
        if (reg_keys & alias_keys) or (reg_keys & aux_keys) or (alias_keys & aux_keys):
            raise ValueError("Checkpoint coverage categories must be pairwise disjoint")

    def validate_against_checkpoint(self, checkpoint_audit: dict, model: nn.Module) -> None:
        """Fail closed unless every checkpoint tensor is accounted for and shapes/dtypes match."""
        ckpt_tensors = checkpoint_audit.get("tensors")
        if not isinstance(ckpt_tensors, dict) or not ckpt_tensors:
            raise ValueError("Checkpoint audit contains no tensors")
        covered = (
            set(self.registered_parameter_map)
            | set(self.tied_parameter_aliases)
            | set(self.auxiliary_passthrough_tensors)
        )
        if covered != set(ckpt_tensors):
            missing = sorted(set(ckpt_tensors) - covered)
            extra = sorted(covered - set(ckpt_tensors))
            raise ValueError(
                f"Incomplete or inconsistent checkpoint tensor coverage (missing={missing[:5]}, extra={extra[:5]})"
            )
        runtime_params = dict(model.named_parameters(remove_duplicate=False))
        for ckpt_name, runtime_name in self.registered_parameter_map.items():
            if runtime_name not in runtime_params:
                raise ValueError(f"Registered parameter missing from runtime model: {runtime_name}")
            param = runtime_params[runtime_name]
            info = ckpt_tensors[ckpt_name]
            if list(param.shape) != info["shape"]:
                raise ValueError(f"Checkpoint shape mismatch for {ckpt_name}")
            if _safetensors_dtype_to_torch(info["dtype"]) != param.dtype:
                raise ValueError(f"Checkpoint dtype mismatch for {ckpt_name}")
        for ckpt_name, target_runtime_name in self.tied_parameter_aliases.items():
            if target_runtime_name not in runtime_params:
                raise ValueError(f"Tied alias target missing from runtime model: {target_runtime_name}")
            param = runtime_params[target_runtime_name]
            info = ckpt_tensors[ckpt_name]
            if list(param.shape) != info["shape"] or _safetensors_dtype_to_torch(info["dtype"]) != param.dtype:
                raise ValueError(f"Tied alias shape/dtype mismatch for {ckpt_name}")


def _safetensors_dtype_to_torch(st_dtype: str) -> torch.dtype:
    mapping = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F64": torch.float64,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }
    if st_dtype not in mapping:
        raise ValueError(f"Unsupported safetensors dtype: {st_dtype}")
    return mapping[st_dtype]


def validate_export_manifest(manifest: Dict[str, Any]) -> None:
    """Verify schema, cross-field invariants, coverage counts, and digest of an export manifest."""
    if not isinstance(manifest, dict) or manifest.get("schema") != CHECKPOINT_EXPORT_SCHEMA:
        raise ValueError(f"Export manifest must declare schema '{CHECKPOINT_EXPORT_SCHEMA}'")
    required = (
        "outcome",
        "track",
        "base_file_sha256",
        "output_file_sha256",
        "checkpoint_filename",
        "baseline_sha256",
        "proposal_sha256",
        "confirmation_evaluation_sha256",
        "coverage",
        "edited_tensors",
        "manifest_sha256",
    )
    for key in required:
        if key not in manifest:
            raise ValueError(f"Export manifest missing required field: {key}")
    if manifest["outcome"] != "patched" or manifest["track"] != "validated_repair":
        raise ValueError("Export manifest requires accepted validated_repair patched outcome")
    for digest_key in (
        "base_file_sha256",
        "output_file_sha256",
        "baseline_sha256",
        "proposal_sha256",
        "confirmation_evaluation_sha256",
        "manifest_sha256",
    ):
        val = manifest[digest_key]
        if not isinstance(val, str) or len(val) != 64 or any(ch not in "0123456789abcdef" for ch in val):
            raise ValueError(f"Export manifest has invalid digest for {digest_key}")

    cov = manifest["coverage"]
    if not isinstance(cov, dict):
        raise ValueError("Export manifest coverage must be a JSON object")
    total = cov.get("total_checkpoint_tensors")
    reg_cnt = cov.get("registered_parameter_count")
    alias_cnt = cov.get("tied_alias_count")
    aux_cnt = cov.get("auxiliary_passthrough_count")
    aux_list = cov.get("auxiliary_passthrough_tensors")
    if (
        type(total) is not int
        or type(reg_cnt) is not int
        or type(alias_cnt) is not int
        or type(aux_cnt) is not int
        or min(total, reg_cnt, alias_cnt, aux_cnt) < 0
        or reg_cnt + alias_cnt + aux_cnt != total
    ):
        raise ValueError("Export manifest coverage counts are inconsistent")
    if not isinstance(aux_list, list) or len(aux_list) != aux_cnt or any(type(x) is not str or not x for x in aux_list):
        raise ValueError("Export manifest auxiliary_passthrough_tensors list is invalid")

    edited = manifest["edited_tensors"]
    if not isinstance(edited, list) or not edited:
        raise ValueError("Export manifest edited_tensors must be a nonempty list")
    for item in edited:
        if not isinstance(item, dict):
            raise ValueError("Export manifest edited_tensors entries must be JSON objects")
        for dkey in ("base_sha256", "edited_sha256"):
            dval = item.get(dkey)
            if not isinstance(dval, str) or len(dval) != 64 or any(ch not in "0123456789abcdef" for ch in dval):
                raise ValueError(f"Export manifest edited tensor has invalid {dkey}")

    recorded_hash = manifest["manifest_sha256"]
    copy_without_hash = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if recorded_hash != digest_object(copy_without_hash):
        raise ValueError("Export manifest SHA-256 integrity check failed")


def load_and_validate_export_manifest(path: str | Path) -> Dict[str, Any]:
    """Load an export manifest from disk with strict duplicate-key and non-finite guards."""
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw, object_pairs_hook=_unique_object_hook, parse_constant=_reject_nonfinite)
    validate_export_manifest(data)
    return data


@torch.no_grad()
def export_validated_checkpoint(
    *,
    base_checkpoint_path: str | Path,
    expected_base_file_sha256: str,
    model: nn.Module,
    proposal: PatchProposal,
    confirmation_report: Dict[str, Any],
    coverage: CheckpointCoveragePlan,
    output_dir: str | Path,
    checkpoint_filename: str = "model.safetensors",
    manifest_filename: str = "export_manifest.json",
) -> Dict[str, Any]:
    """Stage, fresh-load verify, and atomically publish an accepted validated-repair checkpoint."""
    base_path = Path(base_checkpoint_path)
    out_dir = Path(output_dir)
    if not isinstance(proposal, PatchProposal):
        raise ValueError("Checkpoint export requires a frozen PatchProposal")
    if not isinstance(coverage, CheckpointCoveragePlan):
        raise ValueError("Checkpoint export requires a CheckpointCoveragePlan")
    if proposal.alpha <= 0.0 or not proposal.targets:
        raise ValueError("Checkpoint export requires a non-zero proposal")
    if (
        not isinstance(confirmation_report, dict)
        or confirmation_report.get("track") != "validated_repair"
        or confirmation_report.get("outcome") != "patched"
        or confirmation_report.get("accepted") is not True
        or tuple(confirmation_report.get("rejection_reasons", ())) != ()
        or confirmation_report.get("proposal_sha256") != proposal.sha256
        or confirmation_report.get("baseline_sha256") != proposal.baseline.sha256
    ):
        raise ValueError("Checkpoint export requires an accepted validated_repair confirmation report matching the proposal")
    conf_eval_sha = confirmation_report.get("confirmation_evaluation_sha256")
    if not isinstance(conf_eval_sha, str) or len(conf_eval_sha) != 64:
        raise ValueError("Confirmation report missing valid confirmation_evaluation_sha256")
    if not checkpoint_filename or "/" in checkpoint_filename or not manifest_filename or "/" in manifest_filename or checkpoint_filename == manifest_filename:
        raise ValueError("Invalid export artifact filenames")
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError("Export output directory already exists; refusing overwrite")
    if not base_path.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {base_path}")
    if out_dir.parent.resolve() == base_path.resolve().parent and out_dir.name == base_path.name:
        raise FileExistsError("Export output directory cannot collide with base checkpoint")

    base_audit = audit_safetensors(base_path)
    if base_audit["file_sha256"] != expected_base_file_sha256:
        raise ValueError("Base checkpoint file SHA-256 mismatch")
    coverage.validate_against_checkpoint(base_audit, model)

    runtime_params = dict(model.named_parameters(remove_duplicate=False))
    runtime_to_ckpt: Dict[str, str] = {
        runtime_name: ckpt_name for ckpt_name, runtime_name in coverage.registered_parameter_map.items()
    }
    if len(runtime_to_ckpt) != len(coverage.registered_parameter_map):
        raise ValueError("Duplicate runtime parameter target in coverage map")

    target_by_runtime = {target.parameter.name: target for target in proposal.targets}
    for alias_ckpt, target_runtime in coverage.tied_parameter_aliases.items():
        if target_runtime in target_by_runtime:
            raise ValueError(f"Cannot export tied alias whose target parameter was mutated: {alias_ckpt}")

    # Verify runtime model matches baseline on all untouched parameters and matches expected post-edit bits on targets
    current_baseline = capture_model_baseline(model)
    current_param_identities = {
        tensor.name: tensor for tensor in current_baseline.tensors if tensor.kind == "parameter"
    }
    for base_tensor in proposal.baseline.tensors:
        if base_tensor.kind != "parameter":
            continue
        if base_tensor.name not in target_by_runtime:
            if current_param_identities.get(base_tensor.name) != base_tensor:
                raise ValueError(f"Untouched runtime parameter drifted from baseline: {base_tensor.name}")

    edited_updates: Dict[str, tuple[bytes, str, str, list[int], list[int]]] = {}
    for runtime_name, target in target_by_runtime.items():
        if runtime_name not in runtime_to_ckpt:
            raise ValueError(f"Edited runtime parameter not mapped in checkpoint coverage: {runtime_name}")
        ckpt_name = runtime_to_ckpt[runtime_name]
        ckpt_info = base_audit["tensors"][ckpt_name]
        if ckpt_info["sha256"] != target.parameter.sha256:
            raise ValueError(f"Checkpoint baseline tensor hash mismatch for {ckpt_name}")
        live_param = runtime_params[runtime_name]
        live_sha = _tensor_sha256(live_param)
        if live_sha == target.parameter.sha256:
            raise ValueError(f"Target parameter in runtime model is unedited (no-op): {runtime_name}")
        raw_bytes = live_param.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        if hashlib.sha256(raw_bytes).hexdigest() != live_sha:
            raise ValueError(f"Raw byte serialization digest mismatch for {runtime_name}")
        edited_updates[ckpt_name] = (
            raw_bytes,
            target.parameter.sha256,
            live_sha,
            list(target.parameter.shape),
            list(target.columns),
        )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", suffix=".staging", dir=out_dir.parent))
    try:
        staged_ckpt = staging_dir / checkpoint_filename
        staged_manifest = staging_dir / manifest_filename
        shutil.copyfile(base_path, staged_ckpt)
        header, data_start = _layout(staged_ckpt)
        with staged_ckpt.open("r+b") as dest:
            for ckpt_name, (raw_bytes, _pre_sha, _post_sha, _shape, _cols) in edited_updates.items():
                begin, end = header[ckpt_name]["data_offsets"]
                if end - begin != len(raw_bytes):
                    raise ValueError(f"Checkpoint byte span mismatch for {ckpt_name}")
                dest.seek(data_start + begin)
                dest.write(raw_bytes)
            dest.flush()
            os.fsync(dest.fileno())

        fresh_audit = audit_safetensors(staged_ckpt)
        if _layout(staged_ckpt) != (header, data_start) or set(fresh_audit["tensors"]) != set(base_audit["tensors"]):
            raise ValueError("Fresh-load checkpoint layout or tensor set mismatch")
        expected_tensors = {name: dict(info) for name, info in base_audit["tensors"].items()}
        for ckpt_name, (_raw, _pre, post_sha, _shape, _cols) in edited_updates.items():
            expected_tensors[ckpt_name]["sha256"] = post_sha
        if fresh_audit["tensors"] != expected_tensors:
            raise ValueError("Fresh-load checkpoint tensor preservation or edited hash mismatch")

        # Verify immutable baseline file was untouched
        if base_path.stat().st_size != base_audit["size_bytes"]:
            raise ValueError("Base checkpoint file size changed during export")
        with base_path.open("rb") as src:
            if _hash_range(src, 0, base_audit["size_bytes"]) != expected_base_file_sha256:
                raise ValueError("Base checkpoint file bytes changed during export")

        export_body: Dict[str, Any] = {
            "schema": CHECKPOINT_EXPORT_SCHEMA,
            "outcome": "patched",
            "track": "validated_repair",
            "base_file_sha256": expected_base_file_sha256,
            "output_file_sha256": fresh_audit["file_sha256"],
            "checkpoint_filename": checkpoint_filename,
            "baseline_sha256": proposal.baseline.sha256,
            "proposal_sha256": proposal.sha256,
            "confirmation_evaluation_sha256": conf_eval_sha,
            "coverage": {
                "total_checkpoint_tensors": len(fresh_audit["tensors"]),
                "registered_parameter_count": len(coverage.registered_parameter_map),
                "tied_alias_count": len(coverage.tied_parameter_aliases),
                "auxiliary_passthrough_count": len(coverage.auxiliary_passthrough_tensors),
                "auxiliary_passthrough_tensors": list(coverage.auxiliary_passthrough_tensors),
            },
            "edited_tensors": [
                {
                    "checkpoint_name": ckpt_name,
                    "shape": shape,
                    "selected_columns": cols,
                    "base_sha256": pre_sha,
                    "edited_sha256": post_sha,
                }
                for ckpt_name, (_raw, pre_sha, post_sha, shape, cols) in sorted(edited_updates.items())
            ],
        }
        export_body["manifest_sha256"] = digest_object(export_body)
        staged_manifest.write_text(
            json.dumps(export_body, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_and_validate_export_manifest(staged_manifest)

        if out_dir.exists() or out_dir.is_symlink():
            raise FileExistsError("Export destination appeared concurrently before atomic promotion")
        os.rename(staging_dir, out_dir)
        return export_body
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
