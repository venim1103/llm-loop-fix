"""Deterministic JSON provenance and weight-edit manifest auditing (micro_classifier style)."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Literal, Optional
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors_torch
import torch
import torch.nn as nn

from .localize import LocalizationReport
from .patch import PatchConfig, PatchProposal, PatchResult, _tensor_sha256, capture_model_baseline
from .replay import _hash_range, _layout, audit_safetensors
from .verify import SweepResult

MANIFEST_SCHEMA = "llm-loop-fix/edit-manifest/1"
CHECKPOINT_EXPORT_SCHEMA = "llm-loop-fix/validated-checkpoint-export/1"
PATCH_DISTRIBUTION_SCHEMA = "llm-loop-fix/exact-patch-distribution/1"


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


def validate_patch_distribution_manifest(manifest: Dict[str, Any]) -> None:
    """Verify schema, license/attribution provenance, cross-field invariants, and digest."""
    if not isinstance(manifest, dict) or manifest.get("schema") != PATCH_DISTRIBUTION_SCHEMA:
        raise ValueError(f"Patch distribution manifest must declare schema '{PATCH_DISTRIBUTION_SCHEMA}'")
    required = (
        "base_model_repo",
        "base_model_revision",
        "base_model_license",
        "patch_license",
        "attribution",
        "base_file_sha256",
        "accepted_checkpoint_file_sha256",
        "export_manifest_sha256",
        "proposal_sha256",
        "confirmation_evaluation_sha256",
        "slices_filename",
        "slices_file_sha256",
        "payload_bytes",
        "tensors",
        "manifest_sha256",
    )
    for key in required:
        if key not in manifest:
            raise ValueError(f"Patch distribution manifest missing required field: {key}")
    for text_key in (
        "base_model_repo",
        "base_model_revision",
        "base_model_license",
        "patch_license",
        "attribution",
        "slices_filename",
    ):
        val = manifest[text_key]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"Patch distribution manifest requires nonempty {text_key}")
    if "/" in manifest["slices_filename"]:
        raise ValueError("Patch distribution slices_filename must be a plain filename")
    for digest_key in (
        "base_file_sha256",
        "accepted_checkpoint_file_sha256",
        "export_manifest_sha256",
        "proposal_sha256",
        "confirmation_evaluation_sha256",
        "slices_file_sha256",
        "manifest_sha256",
    ):
        val = manifest[digest_key]
        if not isinstance(val, str) or len(val) != 64 or any(ch not in "0123456789abcdef" for ch in val):
            raise ValueError(f"Patch distribution manifest has invalid digest for {digest_key}")

    payload_bytes = manifest["payload_bytes"]
    if type(payload_bytes) is not int or payload_bytes <= 0:
        raise ValueError("Patch distribution payload_bytes must be a positive integer")

    tensors = manifest["tensors"]
    if not isinstance(tensors, list) or not tensors:
        raise ValueError("Patch distribution tensors must be a nonempty list")
    seen_names = set()
    computed_payload = 0
    for entry in tensors:
        if not isinstance(entry, dict):
            raise ValueError("Patch distribution tensor entry must be a JSON object")
        name = entry.get("checkpoint_name")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        cols = entry.get("selected_columns")
        entry_bytes = entry.get("payload_bytes")
        if type(name) is not str or not name or name in seen_names:
            raise ValueError("Patch distribution tensor checkpoint_name must be unique and nonempty")
        seen_names.add(name)
        if dtype not in ("BF16", "F16", "F32"):
            raise ValueError(f"Unsupported patch distribution stored dtype: {dtype}")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(dim) is not int or dim <= 0 for dim in shape)
        ):
            raise ValueError("Patch distribution tensor shape must be a positive 2D integer list")
        if (
            not isinstance(cols, list)
            or not cols
            or any(type(c) is not int or not 0 <= c < shape[1] for c in cols)
            or len(set(cols)) != len(cols)
            or cols != sorted(cols)
        ):
            raise ValueError("Patch distribution selected_columns must be strictly increasing in-range integers")
        elem_size = 2 if dtype in ("BF16", "F16") else 4
        expected_entry_bytes = shape[0] * len(cols) * elem_size
        if type(entry_bytes) is not int or entry_bytes != expected_entry_bytes:
            raise ValueError("Patch distribution tensor payload_bytes mismatch")
        computed_payload += entry_bytes
        for dkey in ("base_sha256", "edited_sha256"):
            dval = entry.get(dkey)
            if not isinstance(dval, str) or len(dval) != 64 or any(ch not in "0123456789abcdef" for ch in dval):
                raise ValueError(f"Patch distribution tensor has invalid {dkey}")
            if entry["base_sha256"] == entry["edited_sha256"]:
                raise ValueError("Patch distribution tensor cannot carry an ineffective no-op slice")

    if computed_payload != payload_bytes:
        raise ValueError("Patch distribution total payload_bytes does not match tensor sum")

    recorded_hash = manifest["manifest_sha256"]
    copy_without_hash = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if recorded_hash != digest_object(copy_without_hash):
        raise ValueError("Patch distribution manifest SHA-256 integrity check failed")


def load_and_validate_patch_distribution_manifest(path: str | Path) -> Dict[str, Any]:
    """Load a patch distribution manifest with strict duplicate-key and non-finite guards."""
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw, object_pairs_hook=_unique_object_hook, parse_constant=_reject_nonfinite)
    validate_patch_distribution_manifest(data)
    return data


@torch.no_grad()
def export_exact_patch_distribution(
    *,
    base_checkpoint_path: str | Path,
    accepted_checkpoint_path: str | Path,
    export_manifest_path: str | Path,
    expected_export_manifest_sha256: str,
    base_model_repo: str,
    base_model_revision: str,
    base_model_license: str,
    patch_license: str,
    attribution: str,
    output_dir: str | Path,
    slices_filename: str = "patch_slices.safetensors",
    manifest_filename: str = "patch_manifest.json",
) -> Dict[str, Any]:
    """Export exact stored-dtype replacement column slices bound to baseline and export hashes."""
    base_path = Path(base_checkpoint_path)
    accepted_path = Path(accepted_checkpoint_path)
    exp_manifest_path = Path(export_manifest_path)
    out_dir = Path(output_dir)

    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError("Patch distribution output directory already exists; refusing overwrite")
    if not base_path.is_file() or not accepted_path.is_file() or not exp_manifest_path.is_file():
        raise FileNotFoundError("Base checkpoint, accepted checkpoint, and export manifest must exist")
    if (
        not isinstance(expected_export_manifest_sha256, str)
        or len(expected_export_manifest_sha256) != 64
        or hashlib.sha256(exp_manifest_path.read_bytes()).hexdigest() != expected_export_manifest_sha256
    ):
        raise ValueError("Export manifest file SHA-256 mismatch")

    exp_manifest = load_and_validate_export_manifest(exp_manifest_path)
    base_audit = audit_safetensors(base_path)
    accepted_audit = audit_safetensors(accepted_path)
    if base_audit["file_sha256"] != exp_manifest["base_file_sha256"]:
        raise ValueError("Base checkpoint SHA-256 does not match export manifest")
    if accepted_audit["file_sha256"] != exp_manifest["output_file_sha256"]:
        raise ValueError("Accepted checkpoint SHA-256 does not match export manifest")
    if set(base_audit["tensors"]) != set(accepted_audit["tensors"]):
        raise ValueError("Base and accepted checkpoint tensor sets differ")

    edited_by_name = {entry["checkpoint_name"]: entry for entry in exp_manifest["edited_tensors"]}
    if len(edited_by_name) != len(exp_manifest["edited_tensors"]) or not edited_by_name:
        raise ValueError("Export manifest has duplicate or empty edited_tensors")

    # Verify all non-edited tensors are bit-identical between base and accepted checkpoints
    for name, info in base_audit["tensors"].items():
        if name not in edited_by_name and accepted_audit["tensors"][name] != info:
            raise ValueError(f"Non-edited checkpoint tensor drifted between base and accepted artifact: {name}")

    slice_tensors: Dict[str, torch.Tensor] = {}
    tensor_entries = []
    total_payload_bytes = 0

    with safe_open(str(base_path), framework="pt", device="cpu") as base_reader, safe_open(
        str(accepted_path), framework="pt", device="cpu"
    ) as acc_reader:
        for ckpt_name in sorted(edited_by_name):
            entry = edited_by_name[ckpt_name]
            base_info = base_audit["tensors"][ckpt_name]
            acc_info = accepted_audit["tensors"][ckpt_name]
            if base_info["sha256"] != entry["base_sha256"] or acc_info["sha256"] != entry["edited_sha256"]:
                raise ValueError(f"Checkpoint tensor digest mismatch against export manifest for {ckpt_name}")
            if base_info["dtype"] not in ("BF16", "F16", "F32") or base_info["dtype"] != acc_info["dtype"]:
                raise ValueError(f"Unsupported or mismatched stored dtype for {ckpt_name}")

            base_t = base_reader.get_tensor(ckpt_name).contiguous()
            acc_t = acc_reader.get_tensor(ckpt_name).contiguous()
            if base_t.ndim != 2 or tuple(base_t.shape) != tuple(acc_t.shape):
                raise ValueError(f"Expected matching 2D tensors for {ckpt_name}")

            cols = sorted(int(c) for c in entry["selected_columns"])
            if not cols or len(set(cols)) != len(cols) or cols[0] < 0 or cols[-1] >= base_t.shape[1]:
                raise ValueError(f"Invalid selected_columns for {ckpt_name}")

            # Verify unselected columns are bit-identical between base and accepted tensors
            mask = torch.ones(base_t.shape[1], dtype=torch.bool)
            mask[cols] = False
            if mask.any():
                if _tensor_sha256(base_t[:, mask]) != _tensor_sha256(acc_t[:, mask]):
                    raise ValueError(f"Unselected columns differ between base and accepted checkpoint for {ckpt_name}")

            col_idx_tensor = torch.tensor(cols, dtype=torch.int64)
            values_slice = acc_t[:, col_idx_tensor].contiguous()
            slice_bytes = values_slice.numel() * values_slice.element_size()
            total_payload_bytes += slice_bytes

            slice_tensors[f"{ckpt_name}::columns"] = col_idx_tensor
            slice_tensors[f"{ckpt_name}::values"] = values_slice
            tensor_entries.append(
                {
                    "checkpoint_name": ckpt_name,
                    "dtype": base_info["dtype"],
                    "shape": list(base_t.shape),
                    "selected_columns": cols,
                    "payload_bytes": slice_bytes,
                    "base_sha256": entry["base_sha256"],
                    "edited_sha256": entry["edited_sha256"],
                }
            )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", suffix=".staging", dir=out_dir.parent))
    try:
        staged_slices = staging_dir / slices_filename
        staged_manifest = staging_dir / manifest_filename
        save_safetensors_torch(slice_tensors, str(staged_slices), metadata={"schema": PATCH_DISTRIBUTION_SCHEMA})
        slices_file_sha = hashlib.sha256(staged_slices.read_bytes()).hexdigest()

        dist_body: Dict[str, Any] = {
            "schema": PATCH_DISTRIBUTION_SCHEMA,
            "base_model_repo": base_model_repo,
            "base_model_revision": base_model_revision,
            "base_model_license": base_model_license,
            "patch_license": patch_license,
            "attribution": attribution,
            "base_file_sha256": exp_manifest["base_file_sha256"],
            "accepted_checkpoint_file_sha256": exp_manifest["output_file_sha256"],
            "export_manifest_sha256": exp_manifest["manifest_sha256"],
            "proposal_sha256": exp_manifest["proposal_sha256"],
            "confirmation_evaluation_sha256": exp_manifest["confirmation_evaluation_sha256"],
            "slices_filename": slices_filename,
            "slices_file_sha256": slices_file_sha,
            "payload_bytes": total_payload_bytes,
            "tensors": tensor_entries,
        }
        dist_body["manifest_sha256"] = digest_object(dist_body)
        staged_manifest.write_text(
            json.dumps(dist_body, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_and_validate_patch_distribution_manifest(staged_manifest)

        if out_dir.exists() or out_dir.is_symlink():
            raise FileExistsError("Patch distribution destination appeared concurrently before atomic promotion")
        os.rename(staging_dir, out_dir)
        return dist_body
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


@torch.no_grad()
def apply_exact_patch_distribution(
    *,
    target_base_checkpoint_path: str | Path,
    distribution_dir: str | Path,
    expected_manifest_sha256: str,
    output_path: Optional[str | Path] = None,
    manifest_filename: str = "patch_manifest.json",
    on_already_applied: Literal["report", "refuse"] = "report",
) -> Dict[str, Any]:
    """Apply an exact stored-dtype patch bundle to a verified baseline checkpoint without compounding."""
    if on_already_applied not in ("report", "refuse"):
        raise ValueError("on_already_applied must be 'report' or 'refuse'")
    target_path = Path(target_base_checkpoint_path)
    dist_dir = Path(distribution_dir)
    manifest_path = dist_dir / manifest_filename
    if not target_path.is_file():
        raise FileNotFoundError(f"Target checkpoint not found: {target_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Patch distribution manifest not found: {manifest_path}")
    if (
        not isinstance(expected_manifest_sha256, str)
        or len(expected_manifest_sha256) != 64
        or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_manifest_sha256
    ):
        raise ValueError("Patch distribution manifest file SHA-256 mismatch")

    manifest = load_and_validate_patch_distribution_manifest(manifest_path)
    slices_path = dist_dir / manifest["slices_filename"]
    if not slices_path.is_file():
        raise FileNotFoundError(f"Patch slices file not found: {slices_path}")
    if hashlib.sha256(slices_path.read_bytes()).hexdigest() != manifest["slices_file_sha256"]:
        raise ValueError("Patch distribution slices file SHA-256 mismatch")

    target_audit = audit_safetensors(target_path)

    # Detect already-applied state explicitly so arithmetic updates can never compound
    if target_audit["file_sha256"] == manifest["accepted_checkpoint_file_sha256"] or all(
        target_audit["tensors"].get(entry["checkpoint_name"], {}).get("sha256") == entry["edited_sha256"]
        for entry in manifest["tensors"]
    ):
        if on_already_applied == "refuse":
            raise ValueError("Patch distribution is already applied to target checkpoint; refusing repeat application")
        return {
            "schema": PATCH_DISTRIBUTION_SCHEMA,
            "status": "already_applied",
            "applied": False,
            "target_file_sha256": target_audit["file_sha256"],
            "accepted_checkpoint_file_sha256": manifest["accepted_checkpoint_file_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        }

    if target_audit["file_sha256"] != manifest["base_file_sha256"]:
        raise ValueError("Patch distribution base checkpoint SHA-256 mismatch")
    if output_path is None:
        raise ValueError("output_path is required when applying a patch distribution to an unpatched baseline")

    out_path = Path(output_path)
    if out_path.exists() or out_path.is_symlink() or out_path.resolve() == target_path.resolve():
        raise FileExistsError("Patch application output must be a fresh path separate from the baseline")

    slices_audit = audit_safetensors(slices_path)
    expected_slice_keys = {
        f"{entry['checkpoint_name']}::{suffix}"
        for entry in manifest["tensors"]
        for suffix in ("columns", "values")
    }
    if set(slices_audit["tensors"]) != expected_slice_keys:
        raise ValueError("Patch slices file tensor keys do not match manifest")

    replacements: Dict[str, tuple[bytes, str, str]] = {}
    with safe_open(str(target_path), framework="pt", device="cpu") as base_reader, safe_open(
        str(slices_path), framework="pt", device="cpu"
    ) as slice_reader:
        for entry in manifest["tensors"]:
            ckpt_name = entry["checkpoint_name"]
            base_info = target_audit["tensors"].get(ckpt_name)
            if base_info != {
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "sha256": entry["base_sha256"],
            }:
                raise ValueError(f"Target baseline tensor contract mismatch for {ckpt_name}")

            col_tensor = slice_reader.get_tensor(f"{ckpt_name}::columns").contiguous()
            val_tensor = slice_reader.get_tensor(f"{ckpt_name}::values").contiguous()
            expected_cols = entry["selected_columns"]
            if (
                col_tensor.dtype != torch.int64
                or col_tensor.ndim != 1
                or col_tensor.tolist() != expected_cols
            ):
                raise ValueError(f"Patch slice columns mismatch for {ckpt_name}")
            if (
                val_tensor.dtype != _safetensors_dtype_to_torch(entry["dtype"])
                or tuple(val_tensor.shape) != (entry["shape"][0], len(expected_cols))
            ):
                raise ValueError(f"Patch slice values shape/dtype mismatch for {ckpt_name}")
            if val_tensor.is_floating_point() and not bool(torch.isfinite(val_tensor).all()):
                raise ValueError(f"Nonfinite values in patch slice for {ckpt_name}")

            working = base_reader.get_tensor(ckpt_name).contiguous().clone()
            working[:, col_tensor] = val_tensor
            post_sha = _tensor_sha256(working)
            if post_sha != entry["edited_sha256"]:
                raise ValueError(f"Reconstructed tensor hash mismatch after slice replacement for {ckpt_name}")
            raw_bytes = working.reshape(-1).view(torch.uint8).numpy().tobytes()
            replacements[ckpt_name] = (raw_bytes, entry["base_sha256"], post_sha)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{out_path.name}.", suffix=".staging", dir=out_path.parent
    )
    os.close(descriptor)
    staging = Path(temporary)
    try:
        shutil.copyfile(target_path, staging)
        header, data_start = _layout(staging)
        with staging.open("r+b") as dest:
            for ckpt_name, (raw_bytes, _pre_sha, _post_sha) in replacements.items():
                begin, end = header[ckpt_name]["data_offsets"]
                if end - begin != len(raw_bytes):
                    raise ValueError(f"Checkpoint byte span mismatch for {ckpt_name}")
                dest.seek(data_start + begin)
                dest.write(raw_bytes)
            dest.flush()
            os.fsync(dest.fileno())

        fresh_audit = audit_safetensors(staging)
        if _layout(staging) != (header, data_start) or set(fresh_audit["tensors"]) != set(target_audit["tensors"]):
            raise ValueError("Fresh-load patched checkpoint layout or tensor set mismatch")
        expected_tensors = {name: dict(info) for name, info in target_audit["tensors"].items()}
        for ckpt_name, (_raw, _pre, post_sha) in replacements.items():
            expected_tensors[ckpt_name]["sha256"] = post_sha
        if fresh_audit["tensors"] != expected_tensors:
            raise ValueError("Fresh-load patched checkpoint tensor preservation or hash mismatch")
        if fresh_audit["file_sha256"] != manifest["accepted_checkpoint_file_sha256"]:
            raise ValueError("Patched checkpoint file SHA-256 does not match accepted checkpoint digest")

        # Verify immutable baseline file was untouched
        if target_path.stat().st_size != target_audit["size_bytes"]:
            raise ValueError("Target baseline checkpoint size changed during patch application")
        with target_path.open("rb") as src:
            if _hash_range(src, 0, target_audit["size_bytes"]) != manifest["base_file_sha256"]:
                raise ValueError("Target baseline checkpoint bytes changed during patch application")

        os.link(staging, out_path)
        return {
            "schema": PATCH_DISTRIBUTION_SCHEMA,
            "status": "applied",
            "applied": True,
            "base_file_sha256": manifest["base_file_sha256"],
            "output_file_sha256": fresh_audit["file_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "payload_bytes": manifest["payload_bytes"],
            "edited_tensors": len(replacements),
        }
    finally:
        staging.unlink(missing_ok=True)
