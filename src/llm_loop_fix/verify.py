"""Closed-loop alpha sweep verification for loop reduction and perplexity stability."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .detector import detect_loop_onset
from .localize import LocalizationReport
from .patch import PatchConfig, PatchResult, apply_surgical_patch, restore_weights


@dataclass
class SweepCandidate:
    """Evaluation metrics for a single alpha value."""

    alpha: float
    loop_rate: float
    mean_loop_ratio: float
    perplexity: float
    patch_result: PatchResult


@dataclass
class SweepResult:
    """Summary of an alpha sweep with the selected optimal configuration."""

    best_alpha: float
    baseline_loop_rate: float
    best_loop_rate: float
    baseline_perplexity: float
    best_perplexity: float
    candidates: List[SweepCandidate]


@torch.no_grad()
def compute_sequence_perplexity(
    model: nn.Module, reference_sequences: List[torch.Tensor]
) -> float:
    """Compute token-level perplexity across clean reference sequences."""
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    for seq in reference_sequences:
        if seq.numel() < 2:
            continue
        inp = seq.unsqueeze(0).to(device)
        out = model(inp)
        logits = out.logits if hasattr(out, "logits") else out
        shift_logits = logits[0, :-1, :].float()
        shift_labels = inp[0, 1:]
        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        total_nll += float(loss.item())
        total_tokens += shift_labels.numel()

    if total_tokens == 0:
        return 1.0
    avg_nll = min(20.0, total_nll / total_tokens)
    return float(math.exp(avg_nll))


@torch.no_grad()
def run_alpha_sweep(
    model: nn.Module,
    report: LocalizationReport,
    generate_fn: Callable[[nn.Module], List[Sequence[int]]],
    reference_sequences: List[torch.Tensor],
    alphas: Sequence[float] = (0.0, 0.5, 1.0, 1.5),
    top_k_matrices: int = 2,
    mode: str = "sparse_columns",
    sparse_column_fraction: float = 0.05,
    max_perplexity_ratio: float = 1.25,
    min_ngram: int = 3,
    min_repeats: int = 3,
) -> SweepResult:
    """Sweep across candidate alpha values, pick the best loop-reducing edit, and apply it."""
    backup: Dict[str, torch.Tensor] = {}
    candidates: List[SweepCandidate] = []

    for alpha in alphas:
        restore_weights(model, backup)
        cfg = PatchConfig(
            alpha=alpha,
            top_k_matrices=top_k_matrices,
            mode=mode,
            sparse_column_fraction=sparse_column_fraction,
        )
        if alpha == 0.0:
            patch_res = PatchResult(
                modified_modules=[],
                total_model_params=sum(p.numel() for p in model.parameters()),
                modified_params_count=0,
                modified_fraction=0.0,
            )
        else:
            patch_res = apply_surgical_patch(model, report, cfg, backup_store=backup)

        generated = generate_fn(model)
        analyses = [
            detect_loop_onset(seq, min_ngram=min_ngram, min_repeats=min_repeats)
            for seq in generated
        ]
        loop_rate = sum(1 for a in analyses if a.is_looping) / max(1, len(analyses))
        mean_ratio = sum(a.loop_ratio for a in analyses) / max(1, len(analyses))
        ppl = compute_sequence_perplexity(model, reference_sequences)

        candidates.append(
            SweepCandidate(
                alpha=alpha,
                loop_rate=loop_rate,
                mean_loop_ratio=mean_ratio,
                perplexity=ppl,
                patch_result=patch_res,
            )
        )

    baseline = candidates[0]
    valid = [
        c
        for c in candidates
        if c.perplexity <= baseline.perplexity * max_perplexity_ratio
    ]
    if not valid:
        valid = [baseline]

    best = min(valid, key=lambda c: (c.loop_rate, c.mean_loop_ratio, c.perplexity))

    # Re-apply the winning patch cleanly
    restore_weights(model, backup)
    if best.alpha > 0.0:
        win_cfg = PatchConfig(
            alpha=best.alpha,
            top_k_matrices=top_k_matrices,
            mode=mode,
            sparse_column_fraction=sparse_column_fraction,
        )
        apply_surgical_patch(model, report, win_cfg)

    return SweepResult(
        best_alpha=best.alpha,
        baseline_loop_rate=baseline.loop_rate,
        best_loop_rate=best.loop_rate,
        baseline_perplexity=baseline.perplexity,
        best_perplexity=best.perplexity,
        candidates=candidates,
    )
