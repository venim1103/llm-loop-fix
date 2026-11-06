"""Surgical top-K column, sparse fractional, and rank-1 weight patching with bitwise auditing."""

from dataclasses import dataclass, field
import hashlib
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
            continue
        layer_idx, mod = proj_modules[name]
        if backup_store is not None and name not in backup_store:
            backup_store[name] = mod.weight.detach().clone()

        orig_weight = mod.weight.detach()
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
        stored_original = orig_weight.to(dtype=torch.float64)
        baseline_norm = float(torch.linalg.vector_norm(stored_original).item())
        delta_norm = float(torch.linalg.vector_norm(w_new_cast.to(dtype=torch.float64) - stored_original).item())
        rel_frob = delta_norm / baseline_norm if baseline_norm else (0.0 if delta_norm == 0.0 else float("inf"))
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
    """Restore original weights from `backup_store`."""
    proj_modules = find_output_projections(model)
    for name, orig_w in backup_store.items():
        if name in proj_modules:
            _idx, mod = proj_modules[name]
            mod.weight.copy_(orig_w.to(device=mod.weight.device, dtype=mod.weight.dtype))
