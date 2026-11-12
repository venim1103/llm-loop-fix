"""Closed-loop alpha sweep verification with paired rescue/regression accounting."""

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set
import math
import hashlib
import json
import random
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F

from .detector import LoopAnalysis, detect_loop_onset
from .evaluation import InstructionRegistry, evaluate_generation_batch
from .localize import LocalizationReport
from .patch import PatchConfig, PatchProposal, PatchResult, RestorationError, _tensor_sha256, apply_patch_proposal, apply_surgical_patch, capture_model_baseline, restore_weights
from .probe import find_output_projections
from .records import EvaluationRunIdentity, GenerationRecord, pair_generation_records


_PROBLEM_RESAMPLING_MAX_CELLS = 524288


@dataclass(frozen=True)
class ProblemResamplingSpec:
    replicates: int
    seed: int
    central_percent: int = field(default=95, init=False)
    method: str = field(default="python-mt19937-inclusive-40-bin-v1", init=False)
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if (type(self.replicates) is not int or not 128 <= self.replicates <= 1024
                or type(self.seed) is not int or not 0 <= self.seed < 2 ** 32):
            raise ValueError("Problem resampling requires bounded plain-integer replicates/seed")
        if (type(self.central_percent) is not int or self.central_percent != 95
            or type(self.schema_version) is not int or self.schema_version != 1
            or type(self.method) is not str or self.method != "python-mt19937-inclusive-40-bin-v1"):
            raise ValueError("Problem resampling requires exact versioned method/schema fields")

    @property
    def sha256(self) -> str:
        payload = {"schema_version": self.schema_version, "method": self.method,
                   "central_percent": self.central_percent, "replicates": self.replicates, "seed": self.seed}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


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

    def pair_records(
        self, baseline: Sequence[GenerationRecord], candidate: Sequence[GenerationRecord], *,
        expected_prompt_contexts: dict[str, tuple[tuple[int, ...], bool]] | None = None,
    ) -> tuple[tuple[GenerationRecord, GenerationRecord], ...]:
        contexts = self.run_identity.snapshot_prompt_contexts(expected_prompt_contexts)
        original = self.run_identity.validate_generations(baseline, candidate_id=self.baseline_candidate_id,
                                                         expected_prompt_contexts=contexts)
        edited = self.run_identity.validate_generations(candidate, candidate_id=self.proposal_candidate_id,
                                                       expected_prompt_contexts=contexts)
        return pair_generation_records(original, edited, expected_keys=self.run_identity.expected_keys)

    def summarize_quality(
        self, baseline: Sequence[GenerationRecord], candidate: Sequence[GenerationRecord], *,
        expected_prompt_contexts: dict[str, tuple[tuple[int, ...], bool]] | None = None,
    ) -> dict:
        """Count recorded quality transitions without accepting or selecting an edit."""
        pairs = self.pair_records(baseline, candidate, expected_prompt_contexts=expected_prompt_contexts)
        return {"scope": "recorded_quality_counts_only", "proposal_evaluation_sha256": self.sha256,
                **_paired_quality_counts(pairs)}

    def summarize_quality_by_item(
        self, baseline: Sequence[GenerationRecord], candidate: Sequence[GenerationRecord], *,
        expected_prompt_contexts: dict[str, tuple[tuple[int, ...], bool]] | None = None,
    ) -> tuple[dict, ...]:
        """Keep descriptive sample counts grouped by problem, without uncertainty claims."""
        pairs = self.pair_records(baseline, candidate, expected_prompt_contexts=expected_prompt_contexts)
        return _paired_quality_by_item(pairs, self.sha256)

    def summarize_quality_resampling(
        self, baseline: Sequence[GenerationRecord], candidate: Sequence[GenerationRecord], *,
        resampling_spec: ProblemResamplingSpec,
        expected_prompt_contexts: dict[str, tuple[tuple[int, ...], bool]] | None = None,
    ) -> dict:
        """Resample complete recorded problems without coverage or acceptance claims."""
        if type(resampling_spec) is not ProblemResamplingSpec:
            raise ValueError("A frozen problem resampling specification is required")
        ProblemResamplingSpec.__post_init__(resampling_spec)
        item_count = len(self.run_identity.dataset.items)
        if item_count * resampling_spec.replicates > _PROBLEM_RESAMPLING_MAX_CELLS:
            raise ValueError("Problem resampling exceeds the sampled-problem work budget")
        pairs = self.pair_records(baseline, candidate, expected_prompt_contexts=expected_prompt_contexts)
        rows = _paired_quality_by_item(pairs, self.sha256)
        result = {"scope": "recorded_problem_resampling_only", "proposal_evaluation_sha256": self.sha256,
                  "resampling_spec_sha256": resampling_spec.sha256, "method": resampling_spec.method,
                  "seed": resampling_spec.seed,
                  "central_percent": resampling_spec.central_percent, "item_count": item_count,
                  "samples_per_item": self.run_identity.profile.samples_per_item, "paired_samples": len(pairs),
                  "by_item": rows, "available": False, "reason": None, "point_delta": None,
                  "central_interval": None, "replicates_planned": resampling_spec.replicates,
                  "replicates_used": 0, "sampled_problem_cells": 0,
                  "confidence_coverage_claimed": False, "acceptance_decision": None}
        contracts = {(record.quality.extractor_id, record.quality.grader_id)
                     for pair in pairs for record in pair if record.quality is not None}
        if any(not row["quality_complete"] for row in rows):
            result["reason"] = "quality_incomplete"
        elif len(contracts) != 1:
            result["reason"] = "quality_contracts_differ_across_items"
        elif item_count < 2:
            result["reason"] = "fewer_than_two_items"
        else:
            nets = [row["net_correct_samples"] for row in rows]
            denominator = len(pairs)
            rng = random.Random(resampling_spec.seed)
            draws = [sum(rng.choices(nets, k=item_count)) / denominator
                     for _ in range(resampling_spec.replicates)]
            quantiles = statistics.quantiles(draws, n=40, method="inclusive")
            result.update({"available": True, "point_delta": sum(nets) / denominator,
                           "central_interval": (quantiles[0], quantiles[-1]),
                           "replicates_used": resampling_spec.replicates,
                           "sampled_problem_cells": item_count * resampling_spec.replicates})
        return result


def _paired_quality_by_item(pairs: Sequence[tuple[GenerationRecord, GenerationRecord]], identity_sha256: str) -> tuple[dict, ...]:
    pairs_by_item = {}
    for pair in pairs:
        pairs_by_item.setdefault(pair[0].key.item_id, []).append(pair)
    return tuple({"scope": "recorded_quality_by_item_only", "proposal_evaluation_sha256": identity_sha256,
                  "item_id": item_id, **_paired_quality_counts(pairs_by_item[item_id])}
                 for item_id in sorted(pairs_by_item))


def _paired_quality_counts(pairs: Sequence[tuple[GenerationRecord, GenerationRecord]]) -> dict:
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
    return {"paired_samples": len(pairs), "baseline": phase_counts[0], "candidate": phase_counts[1],
            "transitions": transitions, "quality_complete": complete,
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


@dataclass(frozen=True)
class ValidatedRepairPolicy:
    """Frozen acceptance thresholds for validated-repair confirmation and preservation."""

    min_usable_improvement_lower_bound: float = 0.0
    max_clean_regressions: int = 0
    max_new_tail_degenerations: int = 0
    max_perplexity_ratio: float = 1.25
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_usable_improvement_lower_bound, bool)
            or not isinstance(self.min_usable_improvement_lower_bound, (int, float))
            or not math.isfinite(self.min_usable_improvement_lower_bound)
            or self.min_usable_improvement_lower_bound < 0.0
        ):
            raise ValueError("Validated repair policy requires finite nonnegative improvement lower bound")
        if type(self.max_clean_regressions) is not int or self.max_clean_regressions < 0:
            raise ValueError("Validated repair policy requires nonnegative integer max_clean_regressions")
        if type(self.max_new_tail_degenerations) is not int or self.max_new_tail_degenerations < 0:
            raise ValueError("Validated repair policy requires nonnegative integer max_new_tail_degenerations")
        if (
            isinstance(self.max_perplexity_ratio, bool)
            or not isinstance(self.max_perplexity_ratio, (int, float))
            or not math.isfinite(self.max_perplexity_ratio)
            or self.max_perplexity_ratio <= 0.0
        ):
            raise ValueError("Validated repair policy requires finite positive max_perplexity_ratio")
        object.__setattr__(self, "min_usable_improvement_lower_bound", float(self.min_usable_improvement_lower_bound))
        object.__setattr__(self, "max_perplexity_ratio", float(self.max_perplexity_ratio))

    @property
    def sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "min_usable_improvement_lower_bound": self.min_usable_improvement_lower_bound,
            "max_clean_regressions": self.max_clean_regressions,
            "max_new_tail_degenerations": self.max_new_tail_degenerations,
            "max_perplexity_ratio": self.max_perplexity_ratio,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
        ).hexdigest()


@dataclass(frozen=True)
class ConfirmationEvaluationIdentity:
    """Frozen binding between a candidate proposal, untouched confirmation, and clean-preservation suite."""

    confirmation_run: EvaluationRunIdentity
    clean_run: EvaluationRunIdentity
    proposal: PatchProposal
    resampling_spec: ProblemResamplingSpec
    policy: ValidatedRepairPolicy = field(default_factory=ValidatedRepairPolicy)

    def __post_init__(self) -> None:
        if (
            type(self.confirmation_run) is not EvaluationRunIdentity
            or type(self.clean_run) is not EvaluationRunIdentity
            or type(self.proposal) is not PatchProposal
            or type(self.resampling_spec) is not ProblemResamplingSpec
            or type(self.policy) is not ValidatedRepairPolicy
        ):
            raise ValueError("Confirmation evaluation requires frozen run, proposal, resampling, and policy contracts")
        if self.confirmation_run.profile.track != "validated_repair" or self.clean_run.profile.track != "validated_repair":
            raise ValueError("Confirmation evaluation requires validated_repair profiles")
        if len(self.confirmation_run.dataset.items) < 2 or any(
            item.role != "confirmation" for item in self.confirmation_run.dataset.items
        ):
            raise ValueError("Confirmation run requires at least two untouched confirmation items")
        if not self.clean_run.dataset.items or any(
            item.role != "clean_reference" for item in self.clean_run.dataset.items
        ):
            raise ValueError("Clean preservation run requires clean_reference items")
        conf_p = self.confirmation_run.profile
        clean_p = self.clean_run.profile
        if (
            conf_p.model_repo,
            conf_p.model_revision,
            conf_p.tokenizer_repo,
            conf_p.tokenizer_revision,
            conf_p.chat_template_sha256,
            conf_p.backend,
            conf_p.backend_version,
            conf_p.dependency_lock_sha256,
        ) != (
            clean_p.model_repo,
            clean_p.model_revision,
            clean_p.tokenizer_repo,
            clean_p.tokenizer_revision,
            clean_p.chat_template_sha256,
            clean_p.backend,
            clean_p.backend_version,
            clean_p.dependency_lock_sha256,
        ):
            raise ValueError("Confirmation and clean preservation runs must share identical model/runtime identity")
        if self.proposal.alpha <= 0.0 or not self.proposal.targets:
            raise ValueError("Confirmation evaluation requires a non-zero frozen candidate proposal")

    @property
    def sha256(self) -> str:
        payload = {
            "schema_version": 1,
            "confirmation_run_sha256": self.confirmation_run.sha256,
            "clean_run_sha256": self.clean_run.sha256,
            "baseline_sha256": self.proposal.baseline.sha256,
            "proposal_sha256": self.proposal.sha256,
            "resampling_spec_sha256": self.resampling_spec.sha256,
            "policy_sha256": self.policy.sha256,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
        ).hexdigest()

    @property
    def baseline_candidate_id(self) -> str:
        return "baseline:" + self.proposal.baseline.sha256

    @property
    def proposal_candidate_id(self) -> str:
        return "proposal:" + self.proposal.sha256


def _is_usable_correct(record: GenerationRecord) -> Optional[bool]:
    quality = record.quality
    if quality is None or quality.grader_status != "graded" or quality.correctness is None:
        return None
    return bool(quality.correctness is True and quality.usability == "usable")


def _record_has_tail_degeneration(record: GenerationRecord) -> Optional[bool]:
    health = record.health
    if health is None:
        return None
    if health.tail_degeneration_detected is True:
        return True
    if health.token_tail is not None and health.token_tail.status == "complete":
        return bool(health.token_tail.meets_threshold)
    if health.tail_degeneration_detected is False:
        return False
    return None


def select_development_candidate(
    run_identity: EvaluationRunIdentity,
    baseline_records: Sequence[GenerationRecord],
    candidates: Sequence[tuple[PatchProposal, Sequence[GenerationRecord]]],
    *,
    min_net_usable_rescued: int = 1,
    max_usable_regressions: int = 0,
    expected_prompt_contexts: dict[str, tuple[tuple[int, ...], bool]] | None = None,
) -> Optional[PatchProposal]:
    """Select the smallest effective candidate beating baseline on development usable-task outcomes."""
    if type(run_identity) is not EvaluationRunIdentity:
        raise ValueError("Development selection requires a frozen EvaluationRunIdentity")
    if run_identity.profile.track != "validated_repair" or any(
        item.role not in ("diagnostic", "development") for item in run_identity.dataset.items
    ):
        raise ValueError("Development selection requires validated_repair diagnostic/development roles")
    if type(min_net_usable_rescued) is not int or min_net_usable_rescued < 1:
        raise ValueError("min_net_usable_rescued must be a positive integer")
    if type(max_usable_regressions) is not int or max_usable_regressions < 0:
        raise ValueError("max_usable_regressions must be a nonnegative integer")
    if not candidates:
        return None

    contexts = run_identity.snapshot_prompt_contexts(expected_prompt_contexts)
    passing: List[tuple[tuple[int, float, int, str], PatchProposal]] = []

    for proposal, candidate_records in candidates:
        eval_id = ProposalEvaluationIdentity(run_identity, proposal)
        if proposal.alpha <= 0.0 or not proposal.targets:
            continue
        pairs = eval_id.pair_records(
            baseline_records, candidate_records, expected_prompt_contexts=contexts
        )
        rescued = 0
        regressed = 0
        complete = True
        for base_rec, cand_rec in pairs:
            base_ok = _is_usable_correct(base_rec)
            cand_ok = _is_usable_correct(cand_rec)
            if base_ok is None or cand_ok is None:
                complete = False
                break
            if not base_ok and cand_ok:
                rescued += 1
            elif base_ok and not cand_ok:
                regressed += 1
        if not complete:
            continue
        net_rescued = rescued - regressed
        if net_rescued < min_net_usable_rescued or regressed > max_usable_regressions:
            continue
        support_entries = sum(len(t.columns) * t.parameter.shape[0] for t in proposal.targets)
        # Prefer larger net gain first, then smallest effective edit on ties (support, alpha, sha256)
        sort_key = (-net_rescued, float(proposal.alpha), support_entries, proposal.sha256)
        passing.append((sort_key, proposal))

    if not passing:
        return None
    passing.sort(key=lambda entry: entry[0])
    return passing[0][1]


def evaluate_validated_repair_confirmation(
    identity: ConfirmationEvaluationIdentity,
    *,
    confirmation_baseline: Sequence[GenerationRecord],
    confirmation_candidate: Sequence[GenerationRecord],
    clean_baseline: Sequence[GenerationRecord],
    clean_candidate: Sequence[GenerationRecord],
    baseline_perplexity: float,
    candidate_perplexity: float,
    expected_confirmation_contexts: dict[str, tuple[tuple[int, ...], bool]] | None = None,
    expected_clean_contexts: dict[str, tuple[tuple[int, ...], bool]] | None = None,
) -> dict:
    """Enforce frozen validated-repair gates on untouched confirmation and clean-preservation records."""
    if type(identity) is not ConfirmationEvaluationIdentity:
        raise ValueError("Validated repair confirmation requires a frozen ConfirmationEvaluationIdentity")
    for label, value in (("baseline_perplexity", baseline_perplexity), ("candidate_perplexity", candidate_perplexity)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and strictly positive")

    conf_contexts = identity.confirmation_run.snapshot_prompt_contexts(expected_confirmation_contexts)
    clean_contexts = identity.clean_run.snapshot_prompt_contexts(expected_clean_contexts)

    conf_base_valid = identity.confirmation_run.validate_generations(
        confirmation_baseline, candidate_id=identity.baseline_candidate_id, expected_prompt_contexts=conf_contexts
    )
    conf_cand_valid = identity.confirmation_run.validate_generations(
        confirmation_candidate, candidate_id=identity.proposal_candidate_id, expected_prompt_contexts=conf_contexts
    )
    conf_pairs = pair_generation_records(
        conf_base_valid, conf_cand_valid, expected_keys=identity.confirmation_run.expected_keys
    )

    clean_base_valid = identity.clean_run.validate_generations(
        clean_baseline, candidate_id=identity.baseline_candidate_id, expected_prompt_contexts=clean_contexts
    )
    clean_cand_valid = identity.clean_run.validate_generations(
        clean_candidate, candidate_id=identity.proposal_candidate_id, expected_prompt_contexts=clean_contexts
    )
    clean_pairs = pair_generation_records(
        clean_base_valid, clean_cand_valid, expected_keys=identity.clean_run.expected_keys
    )

    rejection_reasons: Set[str] = set()

    # 1. Confirmation usable-task improvement interval
    conf_transitions = {"both_usable_correct": 0, "usable_rescued": 0, "usable_regressed": 0, "both_unsuccessful": 0, "not_comparable": 0}
    by_item_nets: Dict[str, int] = {item.item_id: 0 for item in identity.confirmation_run.dataset.items}
    conf_contracts = set()
    conf_quality_complete = True
    for base_rec, cand_rec in conf_pairs:
        base_ok = _is_usable_correct(base_rec)
        cand_ok = _is_usable_correct(cand_rec)
        if base_ok is None or cand_ok is None:
            conf_transitions["not_comparable"] += 1
            conf_quality_complete = False
            continue
        assert base_rec.quality is not None and cand_rec.quality is not None
        conf_contracts.add((base_rec.quality.extractor_id, base_rec.quality.grader_id))
        conf_contracts.add((cand_rec.quality.extractor_id, cand_rec.quality.grader_id))
        if (base_rec.quality.extractor_id, base_rec.quality.grader_id) != (
            cand_rec.quality.extractor_id,
            cand_rec.quality.grader_id,
        ):
            conf_transitions["not_comparable"] += 1
            conf_quality_complete = False
            continue
        if base_ok and cand_ok:
            conf_transitions["both_usable_correct"] += 1
        elif not base_ok and cand_ok:
            conf_transitions["usable_rescued"] += 1
            by_item_nets[base_rec.key.item_id] += 1
        elif base_ok and not cand_ok:
            conf_transitions["usable_regressed"] += 1
            by_item_nets[base_rec.key.item_id] -= 1
        else:
            conf_transitions["both_unsuccessful"] += 1

    point_delta: Optional[float] = None
    central_interval: Optional[tuple[float, float]] = None
    if not conf_quality_complete:
        rejection_reasons.add("confirmation_quality_incomplete")
    elif len(conf_contracts) != 1:
        rejection_reasons.add("confirmation_quality_contracts_differ")
    else:
        item_ids = sorted(by_item_nets)
        item_count = len(item_ids)
        if item_count * identity.resampling_spec.replicates > _PROBLEM_RESAMPLING_MAX_CELLS:
            raise ValueError("Confirmation resampling exceeds the sampled-problem work budget")
        nets = [by_item_nets[item_id] for item_id in item_ids]
        denominator = len(conf_pairs)
        rng = random.Random(identity.resampling_spec.seed)
        draws = [sum(rng.choices(nets, k=item_count)) / denominator for _ in range(identity.resampling_spec.replicates)]
        quantiles = statistics.quantiles(draws, n=40, method="inclusive")
        point_delta = sum(nets) / denominator
        central_interval = (float(quantiles[0]), float(quantiles[-1]))
        if central_interval[0] <= identity.policy.min_usable_improvement_lower_bound:
            rejection_reasons.add("confirmation_improvement_interval_not_positive")

    # 2. Clean preservation: correctness losses and new whitespace/control tail degeneration
    clean_regressions = 0
    new_clean_tail_degenerations = 0
    clean_quality_complete = True
    clean_tail_complete = True
    for base_rec, cand_rec in clean_pairs:
        base_ok = _is_usable_correct(base_rec)
        cand_ok = _is_usable_correct(cand_rec)
        if base_ok is None or cand_ok is None:
            clean_quality_complete = False
        elif base_ok and not cand_ok:
            clean_regressions += 1

        base_tail = _record_has_tail_degeneration(base_rec)
        cand_tail = _record_has_tail_degeneration(cand_rec)
        if base_tail is None or cand_tail is None:
            clean_tail_complete = False
        elif not base_tail and cand_tail:
            new_clean_tail_degenerations += 1

    if not clean_quality_complete:
        rejection_reasons.add("clean_preservation_quality_incomplete")
    elif clean_regressions > identity.policy.max_clean_regressions:
        rejection_reasons.add("clean_preservation_regression_exceeded")

    if not clean_tail_complete:
        rejection_reasons.add("clean_preservation_tail_observation_incomplete")
    elif new_clean_tail_degenerations > identity.policy.max_new_tail_degenerations:
        rejection_reasons.add("clean_preservation_tail_degeneration_exceeded")

    # 3. Clean-corpus perplexity ratio gate
    perplexity_ratio = float(candidate_perplexity) / float(baseline_perplexity)
    if perplexity_ratio > identity.policy.max_perplexity_ratio:
        rejection_reasons.add("clean_perplexity_ratio_exceeded")

    ordered_reasons = tuple(sorted(rejection_reasons))
    accepted = len(ordered_reasons) == 0
    return {
        "schema_version": 1,
        "track": "validated_repair",
        "outcome": "patched" if accepted else "validation_failed",
        "accepted": accepted,
        "confirmation_evaluation_sha256": identity.sha256,
        "proposal_sha256": identity.proposal.sha256,
        "baseline_sha256": identity.proposal.baseline.sha256,
        "policy_sha256": identity.policy.sha256,
        "resampling_spec_sha256": identity.resampling_spec.sha256,
        "confirmation_point_delta": point_delta,
        "confirmation_central_interval": central_interval,
        "confirmation_transitions": conf_transitions,
        "clean_regressions": clean_regressions,
        "new_clean_tail_degenerations": new_clean_tail_degenerations,
        "baseline_perplexity": float(baseline_perplexity),
        "candidate_perplexity": float(candidate_perplexity),
        "perplexity_ratio": perplexity_ratio,
        "rejection_reasons": ordered_reasons,
    }
