"""Contrastive repetition vector extraction and culprit weight matrix attribution."""

from dataclasses import dataclass
from typing import Dict, List
import torch
import torch.nn as nn

from .probe import TrajectoryActivations, find_output_projections


@dataclass
class CulpritMatrix:
    """Attribution score and top offending neuron columns for a single Linear projection."""

    module_name: str
    layer_idx: int
    shape: List[int]
    static_alignment_score: float
    activation_contribution_score: float
    combined_score: float
    top_column_indices: List[int]
    top_column_scores: List[float]


@dataclass
class LocalizationReport:
    """Full diagnostic report mapping layer-wise repetition directions and culprit weights."""

    layer_directions: Dict[int, torch.Tensor]
    layer_separation_norms: Dict[int, float]
    ranked_culprits: List[CulpritMatrix]

    def to_summary_dict(self, top_k: int = 5) -> dict:
        """Serialize a lightweight diagnostic summary suitable for JSON or LLM inspection."""
        return {
            "layer_separation_norms": {
                str(k): round(v, 4) for k, v in self.layer_separation_norms.items()
            },
            "top_culprits": [
                {
                    "module_name": c.module_name,
                    "layer_idx": c.layer_idx,
                    "shape": c.shape,
                    "combined_score": round(c.combined_score, 5),
                    "activation_contribution_score": round(
                        c.activation_contribution_score, 5
                    ),
                    "static_alignment_score": round(c.static_alignment_score, 5),
                    "top_column_indices": c.top_column_indices[:16],
                    "top_column_scores": [round(s, 5) for s in c.top_column_scores[:16]],
                }
                for c in self.ranked_culprits[:top_k]
            ],
        }


@torch.no_grad()
def localize_loop_culprits(
    model: nn.Module,
    activations: TrajectoryActivations,
    top_columns_per_matrix: int = 128,
) -> LocalizationReport:
    """Compute per-layer repetition unit vectors `u` and rank output matrices by loop contribution."""
    layer_directions: Dict[int, torch.Tensor] = {}
    layer_norms: Dict[int, float] = {}

    all_layers = sorted(
        set(activations.residual_loop_sum.keys())
        & set(activations.residual_clean_sum.keys())
    )
    if not all_layers:
        raise ValueError(
            "Both looping and clean token activations are required to compute contrastive directions."
        )

    for idx in all_layers:
        loop_mean = (
            activations.residual_loop_sum[idx] / activations.residual_loop_count[idx]
        )
        clean_mean = (
            activations.residual_clean_sum[idx] / activations.residual_clean_count[idx]
        )
        delta = loop_mean - clean_mean
        norm = float(torch.linalg.norm(delta).item())
        layer_norms[idx] = norm
        if norm > 1e-8:
            layer_directions[idx] = delta / norm
        else:
            layer_directions[idx] = torch.zeros_like(delta)

    proj_modules = find_output_projections(model)
    culprits: List[CulpritMatrix] = []

    for name, (layer_idx, mod) in proj_modules.items():
        if layer_idx not in layer_directions:
            continue
        u = layer_directions[layer_idx]
        # Weight shape in nn.Linear: [d_out, d_in] where d_out == d_model
        w = mod.weight.detach().float().cpu()
        if w.shape[0] != u.shape[0]:
            continue

        # 1. Static alignment: how strongly columns of W align with u
        # u_T_W has shape [d_in]: projection of each column onto u
        u_t_w = torch.mv(w.t(), u)
        w_frob = float(torch.linalg.norm(w).item()) + 1e-8
        static_score = float(torch.linalg.norm(u_t_w).item()) / w_frob

        # 2. Activation-weighted contribution along u
        if (
            name in activations.module_input_loop_sum
            and name in activations.module_input_clean_sum
        ):
            m_loop = (
                activations.module_input_loop_sum[name]
                / activations.module_input_loop_count[name]
            )
            m_clean = (
                activations.module_input_clean_sum[name]
                / activations.module_input_clean_count[name]
            )
            delta_m = m_loop - m_clean
            # Per-column contribution to the repetition direction u: c_j = (u^T W_{:, j}) * delta_m_j
            col_contributions = u_t_w * delta_m
            act_score = float(col_contributions.sum().item())
        else:
            col_contributions = u_t_w.abs()
            act_score = static_score

        k = min(top_columns_per_matrix, col_contributions.numel())
        top_vals, top_idxs = torch.topk(col_contributions, k=k)

        combined = act_score if act_score > 0 else static_score * 0.1
        culprits.append(
            CulpritMatrix(
                module_name=name,
                layer_idx=layer_idx,
                shape=list(w.shape),
                static_alignment_score=static_score,
                activation_contribution_score=act_score,
                combined_score=combined,
                top_column_indices=[int(i) for i in top_idxs.tolist()],
                top_column_scores=[float(v) for v in top_vals.tolist()],
            )
        )

    culprits.sort(key=lambda c: c.combined_score, reverse=True)
    return LocalizationReport(
        layer_directions=layer_directions,
        layer_separation_norms=layer_norms,
        ranked_culprits=culprits,
    )
