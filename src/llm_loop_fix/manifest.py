"""Deterministic JSON provenance and weight-edit manifest auditing (micro_classifier style)."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .localize import LocalizationReport
from .patch import PatchConfig, PatchResult
from .verify import SweepResult

MANIFEST_SCHEMA = "llm-loop-fix/edit-manifest/1"


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
    """Verify schema, required sections, and cryptographic digest integrity."""
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Manifest must declare schema '{MANIFEST_SCHEMA}'")
    for req in ("model_id", "intervention", "summary", "edited_matrices", "manifest_sha256"):
        if req not in manifest:
            raise ValueError(f"Manifest missing required field: {req}")

    recorded_hash = manifest["manifest_sha256"]
    copy_without_hash = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    expected_hash = digest_object(copy_without_hash)
    if recorded_hash != expected_hash:
        raise ValueError("Manifest SHA-256 integrity check failed")
