"""Contrastive repetition vector extraction and culprit weight matrix attribution."""

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import TYPE_CHECKING, Dict, List, Literal, Optional
import torch
import torch.nn as nn

from .probe import TrajectoryActivations, find_output_projections

if TYPE_CHECKING:
    from .patch import PatchProposal


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
    positive_column_count: int = 0
    total_column_count: int = 0


@dataclass
class LocalizationReport:
    """Full diagnostic report mapping layer-wise repetition directions and culprit weights."""

    layer_directions: Dict[int, torch.Tensor]
    layer_separation_norms: Dict[int, float]
    ranked_culprits: List[CulpritMatrix]
    estimator: str = "signed_writer"
    abstained: bool = False
    abstention_reason: Optional[str] = None

    def to_summary_dict(self, top_k: int = 5) -> dict:
        """Serialize a lightweight diagnostic summary suitable for JSON or LLM inspection."""
        summary: dict = {
            "estimator": self.estimator,
            "abstained": self.abstained,
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
                    "positive_column_count": c.positive_column_count,
                    "top_column_indices": c.top_column_indices[:16],
                    "top_column_scores": [round(s, 5) for s in c.top_column_scores[:16]],
                }
                for c in self.ranked_culprits[:top_k]
            ],
        }
        if self.abstention_reason is not None:
            summary["abstention_reason"] = self.abstention_reason
        return summary


def _contrastive_delta(loop_sum, clean_sum, loop_count, clean_count, *, label, width=None):
    if type(loop_count) is not int or type(clean_count) is not int or loop_count <= 0 or clean_count <= 0:
        raise ValueError(f"Positive plain integer activation counts required for {label}")
    if (not isinstance(loop_sum, torch.Tensor) or not isinstance(clean_sum, torch.Tensor)
            or loop_sum.ndim != 1 or loop_sum.numel() == 0 or loop_sum.shape != clean_sum.shape
            or not loop_sum.is_floating_point() or not clean_sum.is_floating_point()
            or loop_sum.device.type == "meta" or clean_sum.device.type == "meta"
            or (width is not None and loop_sum.numel() != width)):
        raise ValueError(f"Matching one-dimensional activation vectors required for {label}")
    delta = loop_sum.float().cpu() / loop_count - clean_sum.float().cpu() / clean_count
    if not bool(torch.isfinite(delta).all()):
        raise ValueError(f"Nonfinite activation difference for {label}")
    return delta


def compute_direction_stability(
    activations_a: TrajectoryActivations,
    activations_b: TrajectoryActivations,
    *,
    min_direction_norm: float = 1e-8,
) -> Dict[int, float]:
    """Compute per-layer cosine similarity between contrastive directions from two subsets."""
    if isinstance(min_direction_norm, bool) or not isinstance(min_direction_norm, (int, float)) or not math.isfinite(min_direction_norm) or min_direction_norm < 0:
        raise ValueError("min_direction_norm must be finite and nonnegative")
    threshold = max(1e-8, float(min_direction_norm))
    shared_layers = sorted(
        set(activations_a.residual_loop_sum.keys())
        & set(activations_a.residual_clean_sum.keys())
        & set(activations_b.residual_loop_sum.keys())
        & set(activations_b.residual_clean_sum.keys())
    )
    if not shared_layers:
        raise ValueError("Both activation subsets must share at least one layer with loop and clean states")
    cosines: Dict[int, float] = {}
    for idx in shared_layers:
        delta_a = _contrastive_delta(
            activations_a.residual_loop_sum[idx], activations_a.residual_clean_sum[idx],
            activations_a.residual_loop_count.get(idx, 0), activations_a.residual_clean_count.get(idx, 0), label=idx,
        )
        delta_b = _contrastive_delta(
            activations_b.residual_loop_sum[idx], activations_b.residual_clean_sum[idx],
            activations_b.residual_loop_count.get(idx, 0), activations_b.residual_clean_count.get(idx, 0),
            label=idx, width=delta_a.numel(),
        )
        norm_a = float(torch.linalg.norm(delta_a.double()).item())
        norm_b = float(torch.linalg.norm(delta_b.double()).item())
        if norm_a <= threshold or norm_b <= threshold:
            cosines[idx] = 0.0
        else:
            cosines[idx] = float(torch.dot(delta_a / norm_a, delta_b / norm_b).item())
    return cosines


@torch.no_grad()
def localize_loop_culprits(
    model: nn.Module,
    activations: TrajectoryActivations,
    top_columns_per_matrix: int = 128,
    *,
    estimator: Literal["signed_writer", "static_alignment_control"] = "signed_writer",
    min_direction_norm: float = 1e-8,
    min_positive_columns: int = 1,
) -> LocalizationReport:
    """Compute per-layer repetition unit vectors `u` and rank output matrices by signed loop contribution."""
    if type(top_columns_per_matrix) is not int or top_columns_per_matrix < 1:
        raise ValueError("top_columns_per_matrix must be a positive integer")
    if type(estimator) is not str or estimator not in ("signed_writer", "static_alignment_control"):
        raise ValueError("estimator must be 'signed_writer' or 'static_alignment_control'")
    if isinstance(min_direction_norm, bool) or not isinstance(min_direction_norm, (int, float)) or not math.isfinite(min_direction_norm) or min_direction_norm < 0:
        raise ValueError("min_direction_norm must be finite and nonnegative")
    if type(min_positive_columns) is not int or min_positive_columns < 1:
        raise ValueError("min_positive_columns must be a positive integer")

    norm_threshold = max(1e-8, float(min_direction_norm))
    layer_directions: Dict[int, torch.Tensor] = {}
    layer_norms: Dict[int, float] = {}
    active_layers = set()

    all_layers = sorted(
        set(activations.residual_loop_sum.keys())
        & set(activations.residual_clean_sum.keys())
    )
    if not all_layers:
        raise ValueError(
            "Both looping and clean token activations are required to compute contrastive directions."
        )

    for idx in all_layers:
        delta = _contrastive_delta(
            activations.residual_loop_sum[idx], activations.residual_clean_sum[idx],
            activations.residual_loop_count.get(idx, 0), activations.residual_clean_count.get(idx, 0), label=idx,
        )
        norm = float(torch.linalg.norm(delta.double()).item())
        layer_norms[idx] = norm
        if norm > norm_threshold:
            layer_directions[idx] = delta / norm
            active_layers.add(idx)
        else:
            layer_directions[idx] = torch.zeros_like(delta)

    if not active_layers:
        return LocalizationReport(
            layer_directions=layer_directions,
            layer_separation_norms=layer_norms,
            ranked_culprits=[],
            estimator=estimator,
            abstained=True,
            abstention_reason="negligible_direction_norm",
        )

    proj_modules = find_output_projections(model)
    culprits: List[CulpritMatrix] = []

    for name, (layer_idx, mod) in proj_modules.items():
        if layer_idx not in active_layers:
            continue
        u = layer_directions[layer_idx]
        # Weight shape in nn.Linear: [d_out, d_in] where d_out == d_model
        w = mod.weight.detach().float().cpu()
        if w.ndim != 2 or not bool(torch.isfinite(w).all()):
            raise ValueError(f"Nonfinite or malformed projection weight for {name}")
        if w.shape[0] != u.shape[0]:
            continue

        # 1. Static alignment: how strongly columns of W align with u
        u_t_w = torch.mv(w.t(), u)
        if not bool(torch.isfinite(u_t_w).all()):
            raise ValueError(f"Nonfinite projection alignment for {name}")
        w_frob = float(torch.linalg.norm(w.double()).item()) + 1e-8
        static_score = float(torch.linalg.norm(u_t_w.double()).item()) / w_frob

        # 2. Signed activation-weighted contribution or explicitly declared static control
        if estimator == "signed_writer":
            if (
                name not in activations.module_input_loop_sum
                or name not in activations.module_input_clean_sum
            ):
                raise ValueError(
                    f"Missing module input activations for signed writer attribution: {name}"
                )
            delta_m = _contrastive_delta(
                activations.module_input_loop_sum[name], activations.module_input_clean_sum[name],
                activations.module_input_loop_count.get(name, 0), activations.module_input_clean_count.get(name, 0),
                label=name, width=w.shape[1],
            )
            # Per-column signed contribution: c_j = (u^T W_{:, j}) * delta_m_j
            col_contributions = u_t_w * delta_m
            act_score = float(col_contributions.sum().item())
        else:
            col_contributions = u_t_w.abs()
            act_score = static_score
        if not bool(torch.isfinite(col_contributions).all()) or not math.isfinite(act_score):
            raise ValueError(f"Nonfinite writer contributions for {name}")

        # Strictly select positive writer columns; never include zero or protective (negative) columns
        pos_indices = torch.nonzero(col_contributions > 0, as_tuple=False).flatten()
        pos_count = int(pos_indices.numel())
        if pos_count < min_positive_columns:
            continue

        pos_scores = col_contributions[pos_indices]
        k = min(top_columns_per_matrix, pos_count)
        top_vals, rel_idxs = torch.topk(pos_scores, k=k)
        top_idxs = pos_indices[rel_idxs]

        combined = float(pos_scores.sum().item()) if estimator == "signed_writer" else static_score
        if not math.isfinite(combined):
            raise ValueError(f"Nonfinite writer score for {name}")
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
                positive_column_count=pos_count,
                total_column_count=int(col_contributions.numel()),
            )
        )

    if not culprits:
        return LocalizationReport(
            layer_directions=layer_directions,
            layer_separation_norms=layer_norms,
            ranked_culprits=[],
            estimator=estimator,
            abstained=True,
            abstention_reason="no_positive_writer_columns",
        )

    culprits.sort(key=lambda c: c.combined_score, reverse=True)
    return LocalizationReport(
        layer_directions=layer_directions,
        layer_separation_norms=layer_norms,
        ranked_culprits=culprits,
        estimator=estimator,
        abstained=False,
        abstention_reason=None,
    )


def build_random_direction_control(proposal: "PatchProposal", *, seed: int) -> "PatchProposal":
    """Match support and unit-direction norm, not measured stored perturbation norm."""
    from .patch import PatchProposal, PatchTarget

    if not isinstance(proposal, PatchProposal):
        raise ValueError("Random direction control requires an immutable PatchProposal")
    if type(seed) is not int:
        raise ValueError("Random direction control seed must be a plain integer")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    layer_dirs: Dict[tuple[int, int], tuple[float, ...]] = {}
    control_targets: List[PatchTarget] = []
    for target in proposal.targets:
        key = (target.layer_idx, target.parameter.shape[0])
        if key not in layer_dirs:
            vector = torch.randn((target.parameter.shape[0],), generator=gen, dtype=torch.float32)
            norm = float(torch.linalg.vector_norm(vector).item())
            if not math.isfinite(norm) or norm <= 1e-8:
                raise ValueError("Random direction draw is nonfinite or negligible")
            layer_dirs[key] = tuple((vector / norm).tolist())
        control_targets.append(
            PatchTarget(
                parameter=target.parameter,
                layer_idx=target.layer_idx,
                direction=layer_dirs[key],
                columns=target.columns,
            )
        )
    return PatchProposal(
        baseline=proposal.baseline,
        budget=proposal.budget,
        alpha=proposal.alpha,
        targets=tuple(control_targets),
        working_dtype=proposal.working_dtype,
    )


def build_random_column_control(proposal: "PatchProposal", *, seed: int) -> "PatchProposal":
    """Construct a direction- and support-cardinality-matched random column control proposal."""
    from .patch import PatchProposal, PatchTarget

    if not isinstance(proposal, PatchProposal):
        raise ValueError("Random column control requires an immutable PatchProposal")
    if type(seed) is not int:
        raise ValueError("Random column control seed must be a plain integer")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    control_targets: List[PatchTarget] = []
    for target in proposal.targets:
        d_in = target.parameter.shape[1]
        k = len(target.columns)
        perm = torch.randperm(d_in, generator=gen)[:k].tolist()
        control_targets.append(
            PatchTarget(
                parameter=target.parameter,
                layer_idx=target.layer_idx,
                direction=target.direction,
                columns=tuple(int(idx) for idx in perm),
            )
        )
    return PatchProposal(
        baseline=proposal.baseline,
        budget=proposal.budget,
        alpha=proposal.alpha,
        targets=tuple(control_targets),
        working_dtype=proposal.working_dtype,
    )


def isolate_layer_proposal(proposal: "PatchProposal", *, layer_idx: int) -> "PatchProposal":
    """Construct a single-layer isolated intervention proposal from a multi-layer proposal."""
    from .patch import PatchProposal

    if not isinstance(proposal, PatchProposal):
        raise ValueError("Layer isolation requires an immutable PatchProposal")
    if type(layer_idx) is not int or layer_idx < 0:
        raise ValueError("Layer isolation requires a nonnegative integer layer_idx")
    isolated = tuple(target for target in proposal.targets if target.layer_idx == layer_idx)
    if not isolated:
        raise ValueError(f"Proposal contains no targets in layer {layer_idx}")
    return PatchProposal(
        baseline=proposal.baseline,
        budget=proposal.budget,
        alpha=proposal.alpha,
        targets=isolated,
        working_dtype=proposal.working_dtype,
    )


@dataclass(frozen=True)
class ControlConstructionIdentity:
    """Frozen existing control recipe, not authenticated experimental provenance."""

    targeted: "PatchProposal"
    kind: Literal["random_direction", "random_columns", "layer_isolated"]
    seed: Optional[int] = None
    layer_idx: Optional[int] = None
    control_proposal: "PatchProposal" = field(init=False)

    def __post_init__(self) -> None:
        from .patch import PatchProposal

        if (type(self.targeted) is not PatchProposal or not self.targeted.alpha or not self.targeted.targets
                or type(self.kind) is not str or self.kind not in ("random_direction", "random_columns", "layer_isolated")):
            raise ValueError("Control construction requires a nonzero frozen target and an existing control kind")
        if self.kind == "layer_isolated":
            if (self.seed is not None or type(self.layer_idx) is not int
                    or self.layer_idx not in {target.layer_idx for target in self.targeted.targets}):
                raise ValueError("Layer control requires a declared target layer without a random seed")
            control = isolate_layer_proposal(self.targeted, layer_idx=self.layer_idx)
        else:
            if type(self.seed) is not int or not 0 <= self.seed < 2 ** 32 or self.layer_idx is not None:
                raise ValueError("Random control requires a bounded plain seed without a layer selector")
            control = (build_random_direction_control(self.targeted, seed=self.seed) if self.kind == "random_direction"
                       else build_random_column_control(self.targeted, seed=self.seed))
        object.__setattr__(self, "control_proposal", control)

    @property
    def scope(self) -> str:
        return "declared_numerical_control_construction_only"

    @property
    def sha256(self) -> str:
        payload = {"schema_version": 1, "scope": self.scope, "builder": "unit-direction-support-controls-v1",
                   "targeted_proposal_sha256": self.targeted.sha256, "kind": self.kind,
                   "seed": self.seed, "layer_idx": self.layer_idx,
                   "control_proposal_sha256": self.control_proposal.sha256}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()
