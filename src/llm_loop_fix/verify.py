"""Closed-loop alpha sweep verification with paired rescue/regression accounting."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .detector import LoopAnalysis, detect_loop_onset
from .localize import LocalizationReport
from .patch import PatchConfig, PatchResult, RestorationError, _tensor_sha256, apply_surgical_patch, restore_weights
from .probe import find_output_projections


@dataclass
class SweepCandidate:
    """Evaluation metrics for a single alpha value across the validation set."""

    alpha: float
    loop_rate: float
    cap_hit_rate: float
    whitespace_tail_count: int
    mean_loop_ratio: float
    perplexity: float
    rescued_count: int
    regressed_count: int
    net_rescued: int
    patch_result: PatchResult


@dataclass
class SweepResult:
    """Summary of an alpha sweep with the selected optimal configuration."""

    best_alpha: float
    baseline_loop_rate: float
    best_loop_rate: float
    baseline_perplexity: float
    best_perplexity: float
    best_rescued_count: int
    best_regressed_count: int
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
    alphas: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0, 6.0),
    top_k_matrices: int = 2,
    mode: str = "top_k_columns",
    top_k_columns: int = 64,
    sparse_column_fraction: float = 0.05,
    max_perplexity_ratio: float = 1.25,
    max_regressions_allowed: int = 0,
    min_ngram: int = 3,
    min_repeats: int = 3,
    max_new_tokens: Optional[int] = None,
    eos_token_ids: Optional[Set[int]] = None,
    whitespace_token_ids: Optional[Set[int]] = None,
) -> SweepResult:
    """Return legacy health-only candidate evidence, restoring every trial."""
    backup: Dict[str, torch.Tensor] = {}
    candidates: List[SweepCandidate] = []
    baseline_analyses: Optional[List[LoopAnalysis]] = None
    original_targets = {}
    for name, (_layer_idx, module) in find_output_projections(model).items():
        parameter = module.weight
        identity = (parameter.shape, parameter.dtype, parameter.device, parameter.stride(), parameter.storage_offset(), parameter.untyped_storage().data_ptr())
        original_targets[name] = (module, parameter, parameter.detach(), identity, _tensor_sha256(parameter))

    def restore_trial_impl() -> None:
        identity_changed = set(find_output_projections(model)) != set(original_targets)
        for name, (module, parameter, _view, identity, _digest) in original_targets.items():
            try:
                current_module = model.get_submodule(name)
            except AttributeError:
                identity_changed = True
                continue
            if current_module is not module or getattr(current_module, "weight", None) is not parameter:
                identity_changed = True
                continue
            current_identity = (parameter.shape, parameter.dtype, parameter.device, parameter.stride(), parameter.storage_offset(), parameter.untyped_storage().data_ptr())
            if current_identity != identity:
                identity_changed = True
        for name, baseline in backup.items():
            if name not in original_targets or _tensor_sha256(baseline) != original_targets[name][4]:
                raise RestorationError("Fatal trial baseline identity mismatch")
        if identity_changed:
            try:
                restore_weights(model, backup)
            finally:
                try:
                    for name, baseline in backup.items():
                        original_view = original_targets[name][2]
                        original_view.copy_(baseline.to(device=original_view.device))
                        if _tensor_sha256(original_view) != original_targets[name][4]:
                            raise RestorationError("Original trial storage did not recover its baseline")
                except BaseException as error:
                    raise RestorationError("Fatal trial identity and original-storage restoration failure") from error
            raise RestorationError("Fatal trial target identity change; model is not validated")
        restore_weights(model, backup)
        for name, (_module, _parameter, original_view, _identity, digest) in original_targets.items():
            if _tensor_sha256(original_view) != digest:
                raise RestorationError(f"Fatal trial baseline mismatch after cleanup: {name}")

    def restore_trial() -> None:
        try:
            restore_trial_impl()
        except RestorationError:
            raise
        except BaseException as error:
            raise RestorationError("Fatal trial identity audit/restoration failure; model is not validated") from error

    for alpha in alphas:
        restore_trial()
        cfg = PatchConfig(
            alpha=alpha,
            top_k_matrices=top_k_matrices,
            mode=mode,
            top_k_columns=top_k_columns,
            sparse_column_fraction=sparse_column_fraction,
        )
        if alpha == 0.0:
            patch_res = PatchResult(
                modified_modules=[],
                total_model_params=sum(p.numel() for p in model.parameters()),
                modified_params_count=0,
                bitwise_changed_params_count=0,
                modified_fraction=0.0,
                matrix_metrics=[],
            )
        else:
            patch_res = apply_surgical_patch(model, report, cfg, backup_store=backup)

        try:
            generated = generate_fn(model)
            analyses = [
                detect_loop_onset(
                    seq,
                    min_ngram=min_ngram,
                    min_repeats=min_repeats,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_token_ids,
                    whitespace_token_ids=whitespace_token_ids,
                )
                for seq in generated
            ]
            if baseline_analyses is None:
                baseline_analyses = analyses

            n_items = max(1, len(analyses))
            loop_rate = sum(1 for a in analyses if a.is_looping) / n_items
            cap_rate = sum(1 for a in analyses if a.hit_token_cap) / n_items
            ws_tails = sum(1 for a in analyses if a.has_whitespace_tail)
            mean_ratio = sum(a.loop_ratio for a in analyses) / n_items
            ppl = compute_sequence_perplexity(model, reference_sequences)

            rescued = sum(
                1
                for b, c in zip(baseline_analyses, analyses)
                if b.is_looping and not c.is_looping
            )
            regressed = sum(
                1
                for b, c in zip(baseline_analyses, analyses)
                if (not b.is_looping and c.is_looping)
                or (not b.has_whitespace_tail and c.has_whitespace_tail)
            )

            candidates.append(
                SweepCandidate(
                    alpha=alpha,
                    loop_rate=loop_rate,
                    cap_hit_rate=cap_rate,
                    whitespace_tail_count=ws_tails,
                    mean_loop_ratio=mean_ratio,
                    perplexity=ppl,
                    rescued_count=rescued,
                    regressed_count=regressed,
                    net_rescued=rescued - regressed,
                    patch_result=patch_res,
                )
            )
        finally:
            restore_trial()

    baseline = candidates[0]
    valid = [
        c
        for c in candidates
        if c.perplexity <= baseline.perplexity * max_perplexity_ratio
        and c.whitespace_tail_count <= baseline.whitespace_tail_count
        and c.regressed_count <= max_regressions_allowed
    ]
    if not valid:
        valid = [baseline]

    best = min(
        valid,
        key=lambda c: (
            c.loop_rate,
            -c.net_rescued,
            c.mean_loop_ratio,
            c.perplexity,
        ),
    )

    return SweepResult(
        best_alpha=best.alpha,
        baseline_loop_rate=baseline.loop_rate,
        best_loop_rate=best.loop_rate,
        baseline_perplexity=baseline.perplexity,
        best_perplexity=best.perplexity,
        best_rescued_count=best.rescued_count,
        best_regressed_count=best.regressed_count,
        candidates=candidates,
    )
