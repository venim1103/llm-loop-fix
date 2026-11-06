"""Surgical rank-1 and sparse column weight patching without gradient descent."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from .localize import LocalizationReport
from .probe import find_output_projections


@dataclass
class PatchConfig:
    """Configuration for surgical weight-space editing."""

    alpha: float = 1.0
    top_k_matrices: int = 2
    mode: str = "sparse_columns"  # "rank1" or "sparse_columns"
    sparse_column_fraction: float = 0.05  # fraction of columns in culprit matrix to edit


@dataclass
class PatchResult:
    """Metadata describing the applied weight edit."""

    modified_modules: List[str]
    total_model_params: int
    modified_params_count: int
    modified_fraction: float


@torch.no_grad()
def apply_surgical_patch(
    model: nn.Module,
    report: LocalizationReport,
    config: PatchConfig,
    backup_store: Optional[Dict[str, torch.Tensor]] = None,
) -> PatchResult:
    """Apply directional projection or sparse column attenuation to top culprit matrices.

    If `backup_store` dict is provided, stores clones of original weights before editing
    so `restore_weights` can revert the model cleanly during alpha sweeps.
    """
    proj_modules = find_output_projections(model)
    total_params = sum(p.numel() for p in model.parameters())
    modified_params = 0
    modified_names: List[str] = []

    selected = report.ranked_culprits[: config.top_k_matrices]
    for culprit in selected:
        name = culprit.module_name
        if name not in proj_modules:
            continue
        layer_idx, mod = proj_modules[name]
        if backup_store is not None and name not in backup_store:
            backup_store[name] = mod.weight.detach().clone()

        device = mod.weight.device
        dtype = mod.weight.dtype
        u = report.layer_directions[layer_idx].to(device=device, dtype=torch.float32)
        w = mod.weight.detach().to(dtype=torch.float32)

        if config.mode == "rank1":
            # Full rank-1 orthogonal projection: W_new = W - alpha * u (u^T W)
            u_t_w = torch.mv(w.t(), u)  # [d_in]
            delta_w = torch.outer(u, u_t_w)
            w_new = w - config.alpha * delta_w
            modified_params += w.numel()
        elif config.mode == "sparse_columns":
            # Attenuate only the top offending columns along direction u
            d_out, d_in = w.shape
            num_cols = max(1, int(round(d_in * config.sparse_column_fraction)))
            if num_cols <= len(culprit.top_column_indices):
                cols = culprit.top_column_indices[:num_cols]
            else:
                u_t_w = torch.mv(w.t(), u)
                _, top_idxs = torch.topk(u_t_w.abs(), k=num_cols)
                cols = top_idxs.tolist()

            w_new = w.clone()
            col_tensor = torch.tensor(cols, device=device, dtype=torch.long)
            w_sub = w[:, col_tensor]  # [d_out, num_cols]
            proj_coeffs = torch.mv(w_sub.t(), u)  # [num_cols]
            w_new[:, col_tensor] = w_sub - config.alpha * torch.outer(u, proj_coeffs)
            modified_params += d_out * len(cols)
        else:
            raise ValueError(f"Unsupported patch mode: {config.mode}")

        mod.weight.copy_(w_new.to(dtype=dtype))
        modified_names.append(name)

    return PatchResult(
        modified_modules=modified_names,
        total_model_params=total_params,
        modified_params_count=modified_params,
        modified_fraction=modified_params / max(1, total_params),
    )


@torch.no_grad()
def restore_weights(model: nn.Module, backup_store: Dict[str, torch.Tensor]) -> None:
    """Restore original weights from `backup_store`."""
    proj_modules = find_output_projections(model)
    for name, orig_w in backup_store.items():
        if name in proj_modules:
            _idx, mod = proj_modules[name]
            mod.weight.copy_(orig_w.to(device=mod.weight.device, dtype=mod.weight.dtype))
