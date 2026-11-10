"""Closed-loop alpha sweep verification with paired rescue/regression accounting."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set
import math
import hashlib
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from .detector import LoopAnalysis, detect_loop_onset
from .evaluation import InstructionRegistry, evaluate_generation_batch
from .localize import LocalizationReport
from .patch import PatchConfig, PatchProposal, PatchResult, RestorationError, _tensor_sha256, apply_patch_proposal, apply_surgical_patch, capture_model_baseline, restore_weights
from .probe import find_output_projections
from .records import EvaluationRunIdentity, GenerationRecord, pair_generation_records


@dataclass(frozen=True)
class ProposalEvaluationIdentity:
    run_identity: EvaluationRunIdentity
    proposal: PatchProposal

    def __post_init__(self) -> None:
        if type(self.run_identity) is not EvaluationRunIdentity or type(self.proposal) is not PatchProposal:
            raise ValueError("Proposal evaluation requires frozen run/numerical declarations")
        if self.run_identity.profile.track != "validated_repair" or any(item.role not in ("diagnostic", "development") for item in self.run_identity.dataset.items):
            raise ValueError("Proposal evaluation requires declared repair diagnostic/development roles")

    @property
    def scope(self) -> str:
        return "declared_run_numerical_proposal_only"

    @property
    def sha256(self) -> str:
        payload = {"schema_version": 1, "scope": self.scope, "run_sha256": self.run_identity.sha256,
                   "baseline_sha256": self.proposal.baseline.sha256, "proposal_sha256": self.proposal.sha256}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()

    @property
    def baseline_candidate_id(self) -> str:
        return "baseline:" + self.proposal.baseline.sha256

    @property
    def proposal_candidate_id(self) -> str:
        return "proposal:" + self.proposal.sha256

    def pair_records(self, baseline: Sequence[GenerationRecord], candidate: Sequence[GenerationRecord]) -> tuple[tuple[GenerationRecord, GenerationRecord], ...]:
        original = self.run_identity.validate_generations(baseline, candidate_id=self.baseline_candidate_id)
        edited = self.run_identity.validate_generations(candidate, candidate_id=self.proposal_candidate_id)
        return pair_generation_records(original, edited, expected_keys=self.run_identity.expected_keys)

    def summarize_quality(self, baseline: Sequence[GenerationRecord], candidate: Sequence[GenerationRecord]) -> dict:
        """Count recorded quality transitions without accepting or selecting an edit."""
        pairs = self.pair_records(baseline, candidate)
        phase_counts = [{key: 0 for key in ("correct", "wrong", "missing", "unavailable", "error")} for _ in range(2)]
        transitions = {key: 0 for key in ("both_correct", "rescued", "regressed", "both_wrong", "not_comparable")}
        for original, edited in pairs:
            outcomes = []
            for counts, record in zip(phase_counts, (original, edited)):
                quality = record.quality
                if quality is None:
                    outcome = "missing"
                elif quality.grader_status == "graded":
                    outcome = "correct" if quality.correctness else "wrong"
                else:
                    outcome = quality.grader_status
                counts[outcome] += 1
                outcomes.append(outcome)
            if (any(outcome not in ("correct", "wrong") for outcome in outcomes)
                    or (original.quality.extractor_id, original.quality.grader_id) != (
                        edited.quality.extractor_id, edited.quality.grader_id)):
                transitions["not_comparable"] += 1
            elif outcomes == ["correct", "correct"]:
                transitions["both_correct"] += 1
            elif outcomes == ["wrong", "correct"]:
                transitions["rescued"] += 1
            elif outcomes == ["correct", "wrong"]:
                transitions["regressed"] += 1
            else:
                transitions["both_wrong"] += 1
        complete = transitions["not_comparable"] == 0
        return {"scope": "recorded_quality_counts_only", "proposal_evaluation_sha256": self.sha256,
            "paired_samples": len(pairs),
                "baseline": phase_counts[0], "candidate": phase_counts[1], "transitions": transitions,
                "quality_complete": complete,
                "net_correct_samples": transitions["rescued"] - transitions["regressed"] if complete else None}


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


def _build_trial_restorer(model: nn.Module, backup: Dict[str, torch.Tensor]) -> Callable[[], None]:
    """Retain whole-model identity and return the common fatal cleanup guard."""
    original_modules = dict(model.named_modules(remove_duplicate=False))
    original_state = {}
    for kind, registry in (("parameter", model.named_parameters(remove_duplicate=False)), ("buffer", model.named_buffers(remove_duplicate=False))):
        for name, tensor in registry:
            identity = (tensor.shape, tensor.dtype, tensor.device, tensor.stride(), tensor.storage_offset(), tensor.untyped_storage().data_ptr())
            original_state[kind, name] = (tensor, tensor.detach(), identity, _tensor_sha256(tensor))
    original_targets = {}
    for name, (_layer_idx, module) in find_output_projections(model).items():
        parameter = module.weight
        identity = (parameter.shape, parameter.dtype, parameter.device, parameter.stride(), parameter.storage_offset(), parameter.untyped_storage().data_ptr())
        original_targets[name] = (module, parameter, parameter.detach(), identity, _tensor_sha256(parameter))

    def restore_trial_impl() -> None:
        current_modules = dict(model.named_modules(remove_duplicate=False))
        identity_changed = current_modules.keys() != original_modules.keys() or any(current_modules[name] is not module for name, module in original_modules.items() if name in current_modules)
        current_state = {(kind, name): tensor for kind, registry in (("parameter", model.named_parameters(remove_duplicate=False)), ("buffer", model.named_buffers(remove_duplicate=False))) for name, tensor in registry}
        if current_state.keys() != original_state.keys():
            identity_changed = True
        for key, (tensor, _view, identity, _digest) in original_state.items():
            if current_state.get(key) is not tensor:
                identity_changed = True
                continue
            current_identity = (tensor.shape, tensor.dtype, tensor.device, tensor.stride(), tensor.storage_offset(), tensor.untyped_storage().data_ptr())
            if current_identity != identity:
                identity_changed = True
        if set(find_output_projections(model)) != set(original_targets):
            identity_changed = True
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
        for (kind, name), (_tensor, original_view, _identity, digest) in original_state.items():
            if _tensor_sha256(original_view) != digest:
                raise RestorationError(f"Fatal model baseline mismatch after cleanup: {kind} {name}")

    def restore_trial() -> None:
        try:
            restore_trial_impl()
        except RestorationError:
            raise
        except BaseException as error:
            raise RestorationError("Fatal trial identity audit/restoration failure; model is not validated") from error

    return restore_trial


@contextmanager
def patch_trial(model: nn.Module, proposal: PatchProposal) -> Iterator[PatchResult]:
    """Temporarily apply a frozen proposal and verify full cleanup on every exit."""
    backup: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        restore_trial = _build_trial_restorer(model, backup)
        try:
            result = apply_patch_proposal(model, proposal, backup)
            yield result
        finally:
            with torch.no_grad():
                restore_trial()


def _freeze_record_prompt_contexts(
    identity: ProposalEvaluationIdentity, expected_prompt_contexts: dict[str, tuple[tuple[int, ...], bool]] | None,
) -> dict[str, tuple[tuple[int, ...], bool]] | None:
    if len(identity.run_identity.expected_keys) > 16:
        raise ValueError("Record trial requires at most 16 planned samples")
    return identity.run_identity.snapshot_prompt_contexts(expected_prompt_contexts)


def collect_proposal_record_pairs(
    model: nn.Module, identity: ProposalEvaluationIdentity,
    record_callback: Callable[[nn.Module, str], Sequence[GenerationRecord]], *,
    expected_prompt_contexts: dict[str, tuple[tuple[int, ...], bool]] | None = None,
) -> tuple[tuple[GenerationRecord, GenerationRecord], ...]:
    if not isinstance(model, nn.Module) or type(identity) is not ProposalEvaluationIdentity or not callable(record_callback):
        raise ValueError("Record trial requires a model, frozen declaration and callback")
    prompt_contexts = _freeze_record_prompt_contexts(identity, expected_prompt_contexts)
    if capture_model_baseline(model) != identity.proposal.baseline:
        raise ValueError("Record trial model does not match the complete registered baseline")

    def collect(candidate_id: str) -> tuple[GenerationRecord, ...]:
        return identity.run_identity.validate_generations(record_callback(model, candidate_id), candidate_id=candidate_id,
                                                          expected_prompt_contexts=prompt_contexts)

    with patch_trial(model, replace(identity.proposal, alpha=0.0)):
        baseline = collect(identity.baseline_candidate_id)
    with patch_trial(model, identity.proposal):
        candidate = collect(identity.proposal_candidate_id)
        pairs = identity.pair_records(baseline, candidate)
    return pairs


def evaluate_proposal_record_pairs(
    model: nn.Module, identity: ProposalEvaluationIdentity,
    record_callback: Callable[[nn.Module, str], Sequence[GenerationRecord]], *,
    registry: InstructionRegistry | None = None,
    expected_prompt_contexts: dict[str, tuple[tuple[int, ...], bool]] | None = None,
) -> tuple[tuple[GenerationRecord, GenerationRecord], ...]:
    if not callable(record_callback):
        raise ValueError("Graded record trial requires a callback")
    if type(identity) is not ProposalEvaluationIdentity:
        raise ValueError("Graded record trial requires a frozen declaration")
    prompt_contexts = _freeze_record_prompt_contexts(identity, expected_prompt_contexts)

    def graded_callback(active_model: nn.Module, candidate_id: str) -> tuple[GenerationRecord, ...]:
        records = record_callback(active_model, candidate_id)
        if prompt_contexts is not None:
            records = identity.run_identity.validate_generations(records, candidate_id=candidate_id,
                                                                 expected_prompt_contexts=prompt_contexts)
        return evaluate_generation_batch(identity.run_identity, records,
                                         candidate_id=candidate_id, registry=registry)

    return collect_proposal_record_pairs(model, identity, graded_callback, expected_prompt_contexts=prompt_contexts)


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
    restore_trial = _build_trial_restorer(model, backup)

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
