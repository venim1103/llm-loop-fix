"""Surgical top-K column, sparse fractional, and rank-1 weight patching with bitwise auditing."""

from dataclasses import dataclass, field
import hashlib
import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from .localize import LocalizationReport
from .probe import find_output_projections


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Compute deterministic SHA-256 digest of a tensor's raw bytes on CPU."""
    cpu_contig = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu_contig.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _bitwise_changed_entries(original: torch.Tensor, edited: torch.Tensor) -> int:
    if original.shape != edited.shape or original.dtype != edited.dtype or original.device != edited.device:
        raise ValueError("stored-bit comparison requires identical shape, dtype and device")
    original_bytes = original.detach().contiguous().reshape(-1).view(torch.uint8).reshape(original.numel(), original.element_size())
    edited_bytes = edited.detach().contiguous().reshape(-1).view(torch.uint8).reshape(edited.numel(), edited.element_size())
    return int((original_bytes != edited_bytes).any(dim=-1).sum().item())


class RestorationError(RuntimeError):
    """Fatal failure to recover the original stored weight identity."""


@dataclass
class MatrixEditMetrics:
    """Detailed audit metrics for a single modified weight matrix."""

    module_name: str
    layer_idx: int
    shape: List[int]
    selected_columns: List[int]
    touched_entries: int
    bitwise_changed_entries: int
    relative_frobenius_change: float
    pre_edit_sha256: str
    post_edit_sha256: str


@dataclass
class PatchConfig:
    """Configuration for surgical weight-space editing.

    Modes:
      - `top_k_columns`: Edit exactly `top_k_columns` (e.g. 64) per culprit matrix:
            W_new = W - alpha * outer(u, mask_K * (u @ W))
      - `sparse_columns`: Edit `sparse_column_fraction` of columns per culprit matrix.
      - `rank1`: Full rank-1 directional projection across all columns.
    """

    alpha: float = 1.0
    top_k_matrices: int = 2
    mode: str = "top_k_columns"  # "top_k_columns", "sparse_columns", or "rank1"
    top_k_columns: int = 64
    sparse_column_fraction: float = 0.05


@dataclass
class PatchResult:
    """Metadata describing the applied weight edit across the model."""

    modified_modules: List[str]
    total_model_params: int
    modified_params_count: int
    bitwise_changed_params_count: int
    modified_fraction: float
    matrix_metrics: List[MatrixEditMetrics] = field(default_factory=list)


@torch.no_grad()
def apply_surgical_patch(
    model: nn.Module,
    report: LocalizationReport,
    config: PatchConfig,
    backup_store: Optional[Dict[str, torch.Tensor]] = None,
) -> PatchResult:
    """Apply atomically on Python failures; this is not repair acceptance."""
    targets = find_output_projections(model)
    original_parameters = {name: module.weight for name, (_idx, module) in targets.items()}
    selected = report.ranked_culprits[:config.top_k_matrices]
    names = set()
    storage_ids = set()
    for culprit in selected:
        name = culprit.module_name
        if name in names:
            raise ValueError("Duplicate patch target")
        names.add(name)
        if name not in targets:
            continue
        weight = targets[name][1].weight
        storage_id = (weight.device, weight.untyped_storage().data_ptr())
        if storage_id in storage_ids:
            raise ValueError("Patch targets share weight storage")
        storage_ids.add(storage_id)
        if backup_store is not None and name in backup_store:
            baseline = backup_store[name]
            if not isinstance(baseline, torch.Tensor) or baseline.shape != weight.shape or baseline.dtype != weight.dtype:
                raise ValueError("Backup does not identify the current baseline tensor")
            if baseline.device == weight.device and baseline.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr():
                raise ValueError("Backup aliases live weight storage")
            if _tensor_sha256(baseline) != _tensor_sha256(weight):
                raise ValueError("Patch must start from the immutable supplied baseline")
    owned_backup: Dict[str, torch.Tensor] = {}
    try:
        result = _apply_surgical_patch(model, report, config, owned_backup)
        if backup_store is not None:
            for name, baseline in owned_backup.items():
                backup_store.setdefault(name, baseline)
        return result
    except BaseException:
        try:
            for name in owned_backup:
                if model.get_submodule(name) is not targets[name][1] or targets[name][1].weight is not original_parameters[name]:
                    raise RestorationError("Target identity changed during the failed mutation")
            restore_weights(model, owned_backup)
        except BaseException as error:
            raise RestorationError("Fatal patch restoration failure; model is not validated") from error
        raise


@torch.no_grad()
def _apply_surgical_patch(
    model: nn.Module,
    report: LocalizationReport,
    config: PatchConfig,
    backup_store: Optional[Dict[str, torch.Tensor]] = None,
) -> PatchResult:
    """Apply directional projection or top-K column over-projection to culprit matrices.

    Implements the MIT CSAIL Qwen3.5-0.8B intervention formula:
        W_new = W_orig - alpha * outer(u, mask_K * (u @ W_orig))
    and records post-cast relative change as well as bitwise parameter changes
    after casting back to the module's target dtype (e.g. bfloat16/float16/float32).
    """
    proj_modules = find_output_projections(model)
    total_params = sum(p.numel() for p in model.parameters())
    modified_params = 0
    bitwise_changed_total = 0
    modified_names: List[str] = []
    metrics_list: List[MatrixEditMetrics] = []

    selected = report.ranked_culprits[: config.top_k_matrices]
    for culprit in selected:
        name = culprit.module_name
        if name not in proj_modules:
            raise ValueError(f"Unknown patch target: {name}")
        try:
            actual_module = model.get_submodule(name)
        except AttributeError as error:
            raise ValueError(f"Patch target lacks a canonical model path: {name}") from error
        if actual_module is not proj_modules[name][1]:
            raise ValueError(f"Patch target module identity mismatch: {name}")
    for culprit in selected:
        name = culprit.module_name
        layer_idx, mod = proj_modules[name]
        if backup_store is not None and name not in backup_store:
            backup_store[name] = mod.weight.detach().clone()

        orig_weight = mod.weight.detach()
        if not bool(torch.isfinite(orig_weight).all()):
            raise ValueError("Nonfinite baseline weight")
        pre_sha = _tensor_sha256(orig_weight)
        device = orig_weight.device
        dtype = orig_weight.dtype
        u = report.layer_directions[layer_idx].to(device=device, dtype=torch.float32)
        w = orig_weight.to(dtype=torch.float32)
        d_out, d_in = w.shape

        if config.mode == "rank1":
            u_t_w = torch.mv(w.t(), u)
            delta_w = torch.outer(u, u_t_w)
            w_new = w - config.alpha * delta_w
            cols = list(range(d_in))
            touched = w.numel()
        elif config.mode in ("top_k_columns", "sparse_columns"):
            if config.mode == "top_k_columns":
                num_cols = min(d_in, max(1, config.top_k_columns))
            else:
                num_cols = max(1, int(round(d_in * config.sparse_column_fraction)))

            if num_cols <= len(culprit.top_column_indices):
                cols = culprit.top_column_indices[:num_cols]
            else:
                u_t_w = torch.mv(w.t(), u)
                _, top_idxs = torch.topk(u_t_w.abs(), k=num_cols)
                cols = [int(i) for i in top_idxs.tolist()]

            w_new = w.clone()
            col_tensor = torch.tensor(cols, device=device, dtype=torch.long)
            w_sub = w[:, col_tensor]
            proj_coeffs = torch.mv(w_sub.t(), u)
            w_new[:, col_tensor] = w_sub - config.alpha * torch.outer(u, proj_coeffs)
            touched = d_out * len(cols)
        else:
            raise ValueError(f"Unsupported patch mode: {config.mode}")

        w_new_cast = w_new.to(dtype=dtype)
        if not bool(torch.isfinite(w_new_cast).all()):
            raise ValueError("Nonfinite stored proposal")
        stored_original = orig_weight.to(dtype=torch.float64)
        baseline_norm = float(torch.linalg.vector_norm(stored_original).item())
        delta_norm = float(torch.linalg.vector_norm(w_new_cast.to(dtype=torch.float64) - stored_original).item())
        rel_frob = delta_norm / baseline_norm if baseline_norm else (0.0 if delta_norm == 0.0 else float("inf"))
        if not math.isfinite(rel_frob):
            raise ValueError("Nonfinite stored perturbation metric")
        bitwise_diff = _bitwise_changed_entries(orig_weight, w_new_cast)

        mod.weight.copy_(w_new_cast)
        post_sha = _tensor_sha256(mod.weight.detach())

        modified_params += touched
        bitwise_changed_total += bitwise_diff
        modified_names.append(name)
        metrics_list.append(
            MatrixEditMetrics(
                module_name=name,
                layer_idx=layer_idx,
                shape=[d_out, d_in],
                selected_columns=cols,
                touched_entries=touched,
                bitwise_changed_entries=bitwise_diff,
                relative_frobenius_change=rel_frob,
                pre_edit_sha256=pre_sha,
                post_edit_sha256=post_sha,
            )
        )

    return PatchResult(
        modified_modules=modified_names,
        total_model_params=total_params,
        modified_params_count=modified_params,
        bitwise_changed_params_count=bitwise_changed_total,
        modified_fraction=modified_params / max(1, total_params),
        matrix_metrics=metrics_list,
    )


@torch.no_grad()
def restore_weights(model: nn.Module, backup_store: Dict[str, torch.Tensor]) -> None:
    """Prevalidate all backups and verify exact restored bytes, or fail fatally."""
    if not backup_store:
        return
    try:
        proj_modules = find_output_projections(model)
        restores = []
        for name, orig_w in backup_store.items():
            if name not in proj_modules:
                raise RestorationError(f"Cannot restore missing target: {name}")
            _idx, mod = proj_modules[name]
            canonical = model.get_submodule(name)
            if canonical is not mod or not isinstance(orig_w, torch.Tensor) or orig_w.shape != mod.weight.shape or orig_w.dtype != mod.weight.dtype:
                raise RestorationError(f"Cannot restore incompatible target: {name}")
            if orig_w.device == mod.weight.device and orig_w.untyped_storage().data_ptr() == mod.weight.untyped_storage().data_ptr():
                raise RestorationError("Cannot restore from a live-alias backup")
            restores.append((mod, orig_w, _tensor_sha256(orig_w)))
        for mod, orig_w, expected_hash in restores:
            mod.weight.copy_(orig_w.to(device=mod.weight.device))
        for mod, _orig_w, expected_hash in restores:
            if _tensor_sha256(mod.weight) != expected_hash:
                raise RestorationError("Restored tensor bits do not match the baseline")
    except BaseException as error:
        raise RestorationError("Fatal weight restoration failure") from error
