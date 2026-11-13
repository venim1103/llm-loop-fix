"""Closed-loop alpha sweep verification with paired rescue/regression accounting."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
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
from .patch import ModelBaseline, PatchConfig, PatchProposal, PatchResult, RestorationError, _tensor_sha256, _validate_patch_config, apply_patch_proposal, apply_surgical_patch, capture_model_baseline, restore_weights
from .probe import find_output_projections
from .records import EvaluationRunIdentity, GenerationRecord, pair_generation_records, record_to_json


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

    def summarize_outcomes(
        self, baseline: Sequence[GenerationRecord], candidate: Sequence[GenerationRecord], *,
        expected_prompt_contexts: dict[str, tuple[tuple[int, ...], bool]] | None = None,
    ) -> dict:
        """Keep reported usability and health distinct from recorded correctness."""
        pairs = self.pair_records(baseline, candidate, expected_prompt_contexts=expected_prompt_contexts)
        return _paired_outcome_summary(pairs, self.sha256)

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


@dataclass(frozen=True)
class DevelopmentGridIdentity:
    """Declared alpha-only grid; no data admission, selection or confirmation."""

    run_identity: EvaluationRunIdentity
    proposals: tuple[PatchProposal, ...]

    def __post_init__(self) -> None:
        if (type(self.run_identity) is not EvaluationRunIdentity
                or self.run_identity.profile.track != "validated_repair"
                or any(item.role != "development" for item in self.run_identity.dataset.items)):
            raise ValueError("Development grid requires one frozen repair run with only development roles")
        if (type(self.proposals) is not tuple or not 0 < len(self.proposals) <= 16
                or any(type(proposal) is not PatchProposal for proposal in self.proposals)):
            raise ValueError("Development grid requires 1 to 16 immutable numerical proposals")
        planned_samples = len(self.run_identity.expected_keys) * (len(self.proposals) + 1)
        if planned_samples > 4096:
            raise ValueError("Development grid exceeds the total planned-sample envelope")
        if any(proposal.alpha == 0.0 or not proposal.targets for proposal in self.proposals):
            raise ValueError("Development candidates require nonzero alpha and nonempty support; baseline is separate")
        if len({proposal.alpha for proposal in self.proposals}) != len(self.proposals):
            raise ValueError("Development grid requires unique nonzero alphas")
        baseline_proposal = replace(self.proposals[0], alpha=0.0)
        if any(replace(proposal, alpha=0.0) != baseline_proposal for proposal in self.proposals):
            raise ValueError("Development grid candidates must share baseline, direction, support and budget")
        object.__setattr__(self, "proposals", tuple(sorted(self.proposals, key=lambda proposal: proposal.alpha)))

    @property
    def scope(self) -> str:
        return "declared_development_alpha_grid_only"

    @property
    def sha256(self) -> str:
        payload = {"schema_version": 1, "scope": self.scope, "run_sha256": self.run_identity.sha256,
                   "proposal_sha256s": [proposal.sha256 for proposal in self.proposals]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()

    @property
    def baseline_proposal(self) -> PatchProposal:
        return replace(self.proposals[0], alpha=0.0)

    @property
    def baseline_candidate_id(self) -> str:
        return "baseline:" + self.baseline_proposal.baseline.sha256

    @property
    def candidate_identities(self) -> tuple[ProposalEvaluationIdentity, ...]:
        return tuple(ProposalEvaluationIdentity(self.run_identity, proposal) for proposal in self.proposals)

    def summarize_quality(
        self, baseline: Sequence[GenerationRecord], candidates: dict[str, Sequence[GenerationRecord]], *,
        expected_prompt_contexts: dict[str, tuple[tuple[int, ...], bool]] | None = None,
    ) -> dict:
        """Join every recorded candidate before reporting descriptive quality."""
        paired_batches = self._pair_batches(baseline, candidates, expected_prompt_contexts=expected_prompt_contexts)
        rows = tuple({
            "alpha": identity.proposal.alpha, "candidate_id": identity.proposal_candidate_id,
            "quality": {"scope": "recorded_quality_counts_only", "proposal_evaluation_sha256": identity.sha256,
                        **_paired_quality_counts(pairs)},
            "by_item": _paired_quality_by_item(pairs, identity.sha256),
        } for identity, pairs in paired_batches)
        return {"scope": "recorded_development_grid_quality_only", "development_grid_sha256": self.sha256,
                "baseline_candidate_id": self.baseline_candidate_id, "candidates": rows,
                "selection_decision": None, "repair_acceptance_validated": False}

    def summarize_outcomes(
        self, baseline: Sequence[GenerationRecord], candidates: dict[str, Sequence[GenerationRecord]], *,
        expected_prompt_contexts: dict[str, tuple[tuple[int, ...], bool]] | None = None,
    ) -> dict:
        """Report all recorded outcomes without selecting or accepting an edit."""
        paired_batches = self._pair_batches(baseline, candidates, expected_prompt_contexts=expected_prompt_contexts)
        rows = tuple({"alpha": identity.proposal.alpha, "candidate_id": identity.proposal_candidate_id,
                      "outcomes": _paired_outcome_summary(pairs, identity.sha256)}
                     for identity, pairs in paired_batches)
        return {"scope": "recorded_development_grid_outcomes_only", "development_grid_sha256": self.sha256,
                "baseline_candidate_id": self.baseline_candidate_id, "candidates": rows,
                "selection_decision": None, "repair_acceptance_validated": False}

    def _pair_batches(
        self, baseline: Sequence[GenerationRecord], candidates: dict[str, Sequence[GenerationRecord]], *,
        expected_prompt_contexts: dict[str, tuple[tuple[int, ...], bool]] | None,
    ) -> tuple[tuple[ProposalEvaluationIdentity, tuple[tuple[GenerationRecord, GenerationRecord], ...]], ...]:
        identities = self.candidate_identities
        if (type(candidates) is not dict or any(type(key) is not str for key in candidates)
                or set(candidates) != {identity.proposal_candidate_id for identity in identities}):
            raise ValueError("Development grid requires exactly the declared candidate batches")
        batches = dict(candidates)
        contexts = self.run_identity.snapshot_prompt_contexts(expected_prompt_contexts)
        paired_batches = tuple(identity.pair_records(baseline, batches[identity.proposal_candidate_id],
                                                     expected_prompt_contexts=contexts)
                               for identity in identities)
        return tuple(zip(identities, paired_batches))


@dataclass(frozen=True)
class RecordedBaseline:
    """Pinned recorded outputs, not proof that a native zero-edit run occurred."""

    run_identity: EvaluationRunIdentity
    baseline: ModelBaseline
    records: tuple[GenerationRecord, ...]
    prompt_contexts: tuple[tuple[str, tuple[int, ...], bool], ...]
    records_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (type(self.run_identity) is not EvaluationRunIdentity or type(self.baseline) is not ModelBaseline
                or self.run_identity.profile.track != "validated_repair"
                or any(item.role != "development" for item in self.run_identity.dataset.items)):
            raise ValueError("Recorded baseline requires a frozen development run and registered baseline")
        if (type(self.records) is not tuple or not 0 < len(self.records) <= 16
                or any(type(record) is not GenerationRecord for record in self.records)):
            raise ValueError("Recorded baseline requires at most 16 immutable generation records")
        if (type(self.prompt_contexts) is not tuple or not 0 < len(self.prompt_contexts) <= 16
                or any(type(context) is not tuple or len(context) != 3 or type(context[0]) is not str
                       for context in self.prompt_contexts)
                or len({context[0] for context in self.prompt_contexts}) != len(self.prompt_contexts)):
            raise ValueError("Recorded baseline requires unique immutable independent prompt contexts")
        contexts = {item_id: (tokens, thinking) for item_id, tokens, thinking in self.prompt_contexts}
        records = self.run_identity.validate_generations(
            self.records, candidate_id="baseline:" + self.baseline.sha256, expected_prompt_contexts=contexts)
        digest = hashlib.sha256()
        serialized_bytes = 0
        for record in records:
            encoded = record_to_json(record).encode("utf-8")
            serialized_bytes += 8 + len(encoded)
            if serialized_bytes > 8 * 1024 * 1024:
                raise ValueError("Recorded baseline exceeds the serialized-record byte envelope")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "prompt_contexts", tuple(sorted(self.prompt_contexts)))
        object.__setattr__(self, "records_sha256", digest.hexdigest())

    @property
    def scope(self) -> str:
        return "declared_run_registered_baseline_records_only"

    @property
    def cache_key(self) -> str:
        payload = {"schema_version": 1, "scope": self.scope, "run_sha256": self.run_identity.sha256,
                   "baseline_sha256": self.baseline.sha256, "prompt_contexts": self.prompt_contexts}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()

    @property
    def sha256(self) -> str:
        payload = {"schema_version": 1, "cache_key": self.cache_key,
                   "records_encoding": "length-prefixed-generation-json-v1", "records_sha256": self.records_sha256}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()

    def records_for_grid(
        self, grid: DevelopmentGridIdentity, *, expected_sha256: str,
        expected_prompt_contexts: dict[str, tuple[tuple[int, ...], bool]],
    ) -> tuple[GenerationRecord, ...]:
        if type(grid) is not DevelopmentGridIdentity:
            raise ValueError("Recorded baseline reuse requires an admitted development grid")
        if type(expected_sha256) is not str or expected_sha256 != self.sha256:
            raise ValueError("Recorded baseline content digest does not match the independent pin")
        if self.run_identity.sha256 != grid.run_identity.sha256 or self.baseline != grid.baseline_proposal.baseline:
            raise ValueError("Recorded baseline does not match the full grid run/numerical identity")
        contexts = self.run_identity.snapshot_prompt_contexts(expected_prompt_contexts)
        if contexts is None or tuple(sorted((item_id, tokens, thinking) for item_id, (tokens, thinking) in contexts.items())) != self.prompt_contexts:
            raise ValueError("Recorded baseline does not match independent grid prompt contexts")
        return self.records


def _pairs_by_item(
    pairs: Sequence[tuple[GenerationRecord, GenerationRecord]],
) -> dict[str, list[tuple[GenerationRecord, GenerationRecord]]]:
    pairs_by_item: dict[str, list[tuple[GenerationRecord, GenerationRecord]]] = {}
    for pair in pairs:
        pairs_by_item.setdefault(pair[0].key.item_id, []).append(pair)
    return pairs_by_item


def _paired_quality_by_item(pairs: Sequence[tuple[GenerationRecord, GenerationRecord]], identity_sha256: str) -> tuple[dict, ...]:
    pairs_by_item = _pairs_by_item(pairs)
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


def _paired_boolean_counts(values: Sequence[tuple[bool | None, bool | None]]) -> dict:
    phases = [{key: 0 for key in ("true", "false", "unavailable")} for _ in range(2)]
    transitions = {key: 0 for key in ("both_true", "both_false", "true_to_false", "false_to_true", "not_comparable")}
    for original, edited in values:
        for counts, value in zip(phases, (original, edited)):
            counts["unavailable" if value is None else "true" if value else "false"] += 1
        if original is None or edited is None:
            key = "not_comparable"
        elif original == edited:
            key = "both_true" if original else "both_false"
        else:
            key = "true_to_false" if original else "false_to_true"
        transitions[key] += 1
    return {"baseline": phases[0], "candidate": phases[1], "transitions": transitions}


def _recorded_health_flags(record: GenerationRecord) -> dict[str, bool | None]:
    health = record.health
    repetition = health.repetition if health is not None else None
    tail = health.token_tail if health is not None else None
    return {
        "cap_hit": record.finish_reason == "length",
        "literal_repetition": bool(repetition.spans) if repetition is not None and repetition.status == "complete" else None,
        "declared_token_tail": tail.meets_threshold if tail is not None and tail.status == "complete" else None,
        "thinking_closure_complete": health.thinking.closure_complete if health is not None else None,
        "thinking_malformed": health.thinking.malformed if health is not None else None,
        "reported_tail_degeneration": health.tail_degeneration_detected if health is not None else None,
    }


def _paired_outcome_counts(pairs: Sequence[tuple[GenerationRecord, GenerationRecord]]) -> dict:
    usability = [{key: 0 for key in ("usable", "empty", "invalid_format", "unavailable", "missing")} for _ in range(2)]
    finish = [{key: 0 for key in ("stop", "length", "cancelled", "error")} for _ in range(2)]
    usable_pairs: list[tuple[bool | None, bool | None]] = []
    for original, edited in pairs:
        for index, record in enumerate((original, edited)):
            usability[index][record.quality.usability if record.quality is not None else "missing"] += 1
            finish[index][record.finish_reason] += 1
        if (original.quality is None or edited.quality is None
                or original.quality.usability == "unavailable" or edited.quality.usability == "unavailable"
                or original.quality.extractor_id != edited.quality.extractor_id):
            usable_pairs.append((None, None))
        else:
            usable_pairs.append((original.quality.usability == "usable", edited.quality.usability == "usable"))
    flags = tuple((_recorded_health_flags(original), _recorded_health_flags(edited)) for original, edited in pairs)
    return {
        "quality": _paired_quality_counts(pairs),
        "usability": {"baseline": usability[0], "candidate": usability[1],
                      "transitions": _paired_boolean_counts(usable_pairs)["transitions"]},
        "finish": {"baseline": finish[0], "candidate": finish[1]},
        "health": {name: _paired_boolean_counts(tuple((original[name], edited[name]) for original, edited in flags))
                   for name in flags[0][0]},
    }


def _paired_outcome_summary(pairs: Sequence[tuple[GenerationRecord, GenerationRecord]], identity_sha256: str) -> dict:
    grouped = _pairs_by_item(pairs)
    return {"scope": "recorded_quality_usability_health_counts_only", "proposal_evaluation_sha256": identity_sha256,
            **_paired_outcome_counts(pairs),
            "by_item": tuple({"item_id": item_id, **_paired_outcome_counts(grouped[item_id])} for item_id in sorted(grouped)),
            "selection_decision": None, "repair_acceptance_validated": False}


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
    if any(not isinstance(seq, torch.Tensor) or seq.ndim != 1 for seq in reference_sequences):
        raise ValueError("Perplexity reference sequences must be one-dimensional tensors")
    if any(seq.numel() >= 2 and (seq.dtype != torch.int64 or bool((seq < 0).any())) for seq in reference_sequences):
        raise ValueError("Eligible perplexity references require nonnegative int64 token IDs")
    if not any(seq.numel() >= 2 for seq in reference_sequences):
        raise ValueError("Perplexity requires an actual predicted reference token")
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
        measured_loss = float(loss.item())
        if not math.isfinite(measured_loss) or measured_loss < 0.0:
            raise ValueError("Perplexity requires finite nonnegative measured loss")
        total_nll += measured_loss
        total_tokens += shift_labels.numel()

    if total_tokens == 0:
        raise ValueError("Perplexity requires an actual predicted reference token")
    avg_nll = total_nll / total_tokens
    if not math.isfinite(avg_nll):
        raise ValueError("Perplexity requires finite mean measured loss")
    try:
        return math.exp(avg_nll)
    except OverflowError as error:
        raise ValueError("Measured perplexity is not representable as a finite float") from error


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


def audit_control_perturbations(
    model: nn.Module, targeted: PatchProposal, controls: dict[str, PatchProposal],
) -> dict:
    """Measure stored numerical controls without forwards, selection or matching claims."""
    if (not isinstance(model, nn.Module) or type(targeted) is not PatchProposal
            or type(controls) is not dict or not 0 < len(controls) <= 8):
        raise ValueError("Control comparison requires a model, target and 1 to 8 declared controls")
    if (any(type(label) is not str or not 0 < len(label) <= 64 or label != label.strip() for label in controls)
            or any(type(proposal) is not PatchProposal for proposal in controls.values())):
        raise ValueError("Control comparison requires bounded plain labels and frozen proposals")
    declarations = dict(controls)
    proposals = (targeted, *declarations.values())
    if any(proposal.alpha == 0.0 or not proposal.targets for proposal in proposals):
        raise ValueError("Control comparison requires nonzero proposals with declared support")
    if any(proposal.baseline != targeted.baseline or proposal.budget != targeted.budget
           or proposal.working_dtype != targeted.working_dtype for proposal in proposals):
        raise ValueError("Control comparison requires one complete baseline and shared numerical constraints")
    if capture_model_baseline(model) != targeted.baseline:
        raise ValueError("Control comparison model does not match the registered baseline")

    def audit(proposal: PatchProposal) -> dict:
        with patch_trial(model, proposal) as result:
            if result.baseline_sha256 != proposal.baseline.sha256 or result.proposal_sha256 != proposal.sha256:
                raise ValueError("Control audit does not identify the baseline/numerical proposal")
            expected_modules = [target.parameter.name.removesuffix(".weight") for target in proposal.targets]
            if [metric.module_name for metric in result.matrix_metrics] != expected_modules:
                raise ValueError("Control audit does not cover the complete proposal")
            delta_squared = 0.0
            for metric in result.matrix_metrics:
                baseline_norm = metric.baseline_frobenius_norm
                delta_norm = metric.stored_delta_frobenius_norm
                if (type(baseline_norm) not in (int, float) or type(delta_norm) not in (int, float)
                        or not math.isfinite(baseline_norm) or not math.isfinite(delta_norm)
                        or baseline_norm < 0 or delta_norm < 0):
                    raise ValueError("Control comparison requires finite measured absolute norms")
                delta_squared += delta_norm ** 2
            stored_norm = math.sqrt(delta_squared)
            if not math.isfinite(stored_norm):
                raise ValueError("Control comparison has a nonfinite total stored norm")
            snapshot = {"proposal_sha256": proposal.sha256, "alpha": proposal.alpha,
                        "support_entries": result.modified_params_count,
                        "bitwise_changed_entries": result.bitwise_changed_params_count,
                        "stored_delta_frobenius_norm": stored_norm,
                        "matrix_metrics": tuple(asdict(metric) for metric in result.matrix_metrics)}
        return snapshot

    target_audit = audit(targeted)
    rows = []
    for label in sorted(declarations):
        control_audit = audit(declarations[label])
        rows.append({
            "label": label, "audit": control_audit,
            "stored_norm_difference": control_audit["stored_delta_frobenius_norm"] - target_audit["stored_delta_frobenius_norm"],
            "support_entries_difference": control_audit["support_entries"] - target_audit["support_entries"],
            "bitwise_changed_entries_difference": control_audit["bitwise_changed_entries"] - target_audit["bitwise_changed_entries"],
        })
    return {"scope": "registered_stored_control_comparison_only", "baseline_sha256": targeted.baseline.sha256,
            "targeted": target_audit, "controls": tuple(rows), "norm_matching_validated": False,
            "causal_utility_measured": False, "repair_acceptance_validated": False}


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
    if (isinstance(max_perplexity_ratio, bool) or not isinstance(max_perplexity_ratio, (int, float))
            or not math.isfinite(max_perplexity_ratio) or max_perplexity_ratio < 0.0):
        raise ValueError("Alpha sweep requires a finite nonnegative perplexity-ratio guard")
    if type(max_regressions_allowed) is not int or max_regressions_allowed < 0:
        raise ValueError("Alpha sweep requires a plain nonnegative regression-count guard")
    configs = tuple(PatchConfig(
        alpha=alpha, top_k_matrices=top_k_matrices, mode=mode,
        top_k_columns=top_k_columns, sparse_column_fraction=sparse_column_fraction,
    ) for alpha in alphas)
    if not configs:
        raise ValueError("Alpha sweep requires a nonempty grid")
    for config in configs:
        _validate_patch_config(config)
    if len({config.alpha for config in configs}) != len(configs):
        raise ValueError("Alpha sweep requires unique alpha values")
    if not any(config.alpha == 0.0 for config in configs):
        configs += (replace(configs[0], alpha=0.0),)

    backup: Dict[str, torch.Tensor] = {}
    candidates: List[SweepCandidate] = []
    analyses_by_alpha: Dict[float, List[LoopAnalysis]] = {}
    restore_trial = _build_trial_restorer(model, backup)

    for cfg in configs:
        restore_trial()
        alpha = cfg.alpha
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
            if not analyses:
                raise ValueError("Alpha sweep requires nonempty generated cohorts")
            if analyses_by_alpha and len(analyses) != len(next(iter(analyses_by_alpha.values()))):
                raise ValueError("Alpha sweep cohorts must have the same number of positional samples")
            analyses_by_alpha[alpha] = analyses

            n_items = len(analyses)
            loop_rate = sum(1 for a in analyses if a.is_looping) / n_items
            cap_rate = sum(1 for a in analyses if a.hit_token_cap) / n_items
            ws_tails = sum(1 for a in analyses if a.has_whitespace_tail)
            mean_ratio = sum(a.loop_ratio for a in analyses) / n_items
            ppl = compute_sequence_perplexity(model, reference_sequences)
            if type(ppl) not in (int, float) or not math.isfinite(ppl) or ppl < 1.0:
                raise ValueError("Alpha sweep requires finite measured perplexity of at least one")

            candidates.append(
                SweepCandidate(
                    alpha=alpha,
                    loop_rate=loop_rate,
                    cap_hit_rate=cap_rate,
                    whitespace_tail_count=ws_tails,
                    mean_loop_ratio=mean_ratio,
                    perplexity=ppl,
                    rescued_count=0,
                    regressed_count=0,
                    net_rescued=0,
                    patch_result=patch_res,
                )
            )
        finally:
            restore_trial()

    baseline = next(candidate for candidate in candidates if candidate.alpha == 0.0)
    baseline_analyses = analyses_by_alpha[0.0]
    candidates = [baseline] + [candidate for candidate in candidates if candidate is not baseline]
    for candidate in candidates:
        pairs = list(zip(baseline_analyses, analyses_by_alpha[candidate.alpha]))
        candidate.rescued_count = sum(1 for before, after in pairs if before.is_looping and not after.is_looping)
        candidate.regressed_count = sum(1 for before, after in pairs if
                                        (not before.is_looping and after.is_looping)
                                        or (not before.has_whitespace_tail and after.has_whitespace_tail))
        candidate.net_rescued = candidate.rescued_count - candidate.regressed_count

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
