"""Multi-mode repetition loop, ceiling truncation, and tail degeneration detector."""

import re
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import List, Optional, Sequence, Set

from .records import DatasetManifest, EvaluationProfile, EvaluationRunIdentity, GenerationHealth, GenerationRecord, RepetitionObservation, RepetitionSpan, SampleKey, TokenTailObservation, _observe_thinking, _token_ids_sha256, pair_generation_records, record_to_json


@dataclass
class LoopAnalysis:
    """Result of analyzing a token sequence for pathological repetition or truncation."""

    is_looping: bool
    onset_index: Optional[int]
    period_length: Optional[int]
    repeat_count: int
    loop_ratio: float
    failure_mode: str = "clean"  # "clean", "ngram_loop", "cap_truncation", "whitespace_tail"
    hit_token_cap: bool = False
    has_whitespace_tail: bool = False


@dataclass(frozen=True)
class RepetitionSpec:
    min_period: int = 3
    max_period: int = 48
    min_repeats: int = 3
    max_tokens: int = 4096
    algorithm: str = "greedy-exact-tandem-v1"

    def __post_init__(self) -> None:
        for value in (self.min_period, self.max_period, self.min_repeats, self.max_tokens):
            if type(value) is not int or value < 1:
                raise ValueError("Repetition limits must be positive integers")
        if self.min_period > self.max_period or self.max_period > 48 or self.min_repeats < 2 or self.max_tokens > 32768 or self.min_period * self.min_repeats > self.max_tokens:
            raise ValueError("Unsupported or inconsistent repetition scan limits")
        if self.algorithm != "greedy-exact-tandem-v1":
            raise ValueError("Unsupported repetition observation algorithm")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def observe_token_repetition(generation: GenerationRecord, spec: RepetitionSpec) -> RepetitionObservation:
    """Report bounded exact tandem spans, not pathological-loop or utility labels."""
    if not isinstance(generation, GenerationRecord) or not isinstance(spec, RepetitionSpec):
        raise ValueError("Repetition observation requires immutable generation and spec")
    tokens = generation.generated_token_ids
    token_count = len(tokens)
    token_digest = _token_ids_sha256(tokens)
    if generation.finish_reason in ("cancelled", "error"):
        return RepetitionObservation(spec.sha256, token_count, "unavailable", reason="backend_failure", token_ids_sha256=token_digest)
    if token_count > spec.max_tokens:
        return RepetitionObservation(spec.sha256, token_count, "unavailable", reason="token_budget_exceeded", token_ids_sha256=token_digest)
    spans = []
    start = 0
    while start + spec.min_period * spec.min_repeats <= token_count:
        max_period = min(spec.max_period, (token_count - start) // spec.min_repeats)
        for period in range(spec.min_period, max_period + 1):
            pattern = tokens[start:start + period]
            end = start + period
            repeats = 1
            while end + period <= token_count and tokens[end:end + period] == pattern:
                repeats += 1
                end += period
            if repeats >= spec.min_repeats:
                spans.append(RepetitionSpan(start, end, period, repeats))
                start = end
                break
        else:
            start += 1
    return RepetitionObservation(spec.sha256, token_count, "complete", tuple(spans), token_ids_sha256=token_digest)


@dataclass(frozen=True)
class TokenTailSpec:
    """Declared tokenizer token classes; empirical use requires mapping review."""

    tokenizer_repo: str
    tokenizer_revision: str
    whitespace_token_ids: tuple[int, ...]
    eos_token_ids: tuple[int, ...]
    min_run_tokens: int = 16
    max_tokens: int = 4096
    algorithm: str = "declared-whitespace-tail-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer_repo, str) or not self.tokenizer_repo.strip() or not isinstance(self.tokenizer_revision, str) or re.fullmatch(r"[0-9a-f]{40}", self.tokenizer_revision) is None:
            raise ValueError("Token-tail spec requires a pinned tokenizer identity")
        for name in ("whitespace_token_ids", "eos_token_ids"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(type(value) is not int or value < 0 for value in values) or len(set(values)) != len(values):
                raise ValueError("Token-tail vocabulary must be immutable unique nonnegative IDs")
            object.__setattr__(self, name, tuple(sorted(values)))
        if not self.whitespace_token_ids or set(self.whitespace_token_ids) & set(self.eos_token_ids):
            raise ValueError("Whitespace classes must be declared and disjoint from EOS")
        if type(self.min_run_tokens) is not int or type(self.max_tokens) is not int or not 1 <= self.min_run_tokens <= self.max_tokens <= 32768:
            raise ValueError("Token-tail limits must be positive bounded integers")
        if self.algorithm != "declared-whitespace-tail-v1":
            raise ValueError("Unsupported token-tail observation algorithm")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def observe_token_tail(generation: GenerationRecord, spec: TokenTailSpec) -> TokenTailObservation:
    """Observe a declared token-class suffix, not task correctness or degeneration."""
    if not isinstance(generation, GenerationRecord) or not isinstance(spec, TokenTailSpec):
        raise ValueError("Token-tail observation requires immutable generation and spec")
    tokens = generation.generated_token_ids
    token_count = len(tokens)
    token_digest = _token_ids_sha256(tokens)
    if generation.finish_reason in ("cancelled", "error"):
        return TokenTailObservation(spec.sha256, token_digest, token_count, "unavailable", reason="backend_failure")
    if token_count > spec.max_tokens:
        return TokenTailObservation(spec.sha256, token_digest, token_count, "unavailable", reason="token_budget_exceeded")
    eos_excluded = bool(tokens and generation.finish_reason == "stop" and tokens[-1] in spec.eos_token_ids)
    end = token_count - int(eos_excluded)
    start = end
    whitespace_ids = frozenset(spec.whitespace_token_ids)
    while start > 0 and tokens[start - 1] in whitespace_ids:
        start -= 1
    return TokenTailObservation(spec.sha256, token_digest, token_count, "complete", start, end, end - start >= spec.min_run_tokens, eos_excluded)


def observe_generation_health(
    generation: GenerationRecord, *, thinking_required: bool, repetition_spec: RepetitionSpec | None = None,
    tail_spec: TokenTailSpec | None = None, profile: EvaluationProfile | None = None
) -> GenerationHealth:
    """Observe finish/framing and optional exact repetition, not pathological loops."""
    if profile is not None:
        if not isinstance(profile, EvaluationProfile) or generation.profile_id != profile.sha256 or thinking_required != profile.thinking:
            raise ValueError("Generation health requires the exact evaluation profile/thinking contract")
        if len(generation.generated_token_ids) > profile.max_new_tokens or generation.prompt_tokens + len(generation.generated_token_ids) > profile.max_model_len:
            raise ValueError("Generation health input exceeds the declared profile token envelope")
    if tail_spec is not None:
        if not isinstance(tail_spec, TokenTailSpec) or profile is None:
            raise ValueError("Token-tail health requires a declared spec and evaluation profile")
        if (tail_spec.tokenizer_repo, tail_spec.tokenizer_revision, tail_spec.eos_token_ids) != (profile.tokenizer_repo, profile.tokenizer_revision, tuple(sorted(profile.eos_token_ids))):
            raise ValueError("Token-tail tokenizer/EOS identity does not match the profile")
    thinking = _observe_thinking(generation.generated_text, thinking_required, generation.thinking_open_from_prompt)
    repetition = observe_token_repetition(generation, repetition_spec) if repetition_spec is not None else None
    detected = bool(repetition.spans) if repetition is not None and repetition.status == "complete" else None
    token_tail = replace(observe_token_tail(generation, tail_spec), profile_sha256=profile.sha256) if tail_spec is not None else None
    version = "finish-thinking-patterns-v3" if token_tail is not None else "finish-thinking-repetition-v2" if repetition is not None else "finish-thinking-v1"
    return GenerationHealth(generation.finish_reason == "length", thinking, repetition_detected=detected, observation_version=version, repetition=repetition, token_tail=token_tail)


@dataclass(frozen=True)
class NoEditDiagnosis:
    outcome: str
    dataset_sha256: str
    profile_sha256: str
    evidence_sha256: str
    sample_keys: tuple[SampleKey, ...]
    reasons: tuple[str, ...]
    cap_hits: int

    def __post_init__(self) -> None:
        if self.outcome not in ("no_patch", "insufficient_evidence"):
            raise ValueError("Diagnosis cannot report an accepted repair")
        for digest in (self.dataset_sha256, self.profile_sha256, self.evidence_sha256):
            if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("Diagnosis requires canonical identity digests")
        if not isinstance(self.sample_keys, tuple) or not self.sample_keys or any(not isinstance(key, SampleKey) for key in self.sample_keys) or self.sample_keys != tuple(sorted(set(self.sample_keys))):
            raise ValueError("Diagnosis requires immutable complete unique ordered sample keys")
        if not isinstance(self.reasons, tuple) or not self.reasons or any(not isinstance(reason, str) or not reason.strip() for reason in self.reasons) or self.reasons != tuple(sorted(set(self.reasons))):
            raise ValueError("Diagnosis requires immutable ordered reasons")
        if type(self.cap_hits) is not int or not 0 <= self.cap_hits <= len(self.sample_keys):
            raise ValueError("Diagnosis cap count is invalid")

    @property
    def scope(self) -> str:
        return "recorded_evidence_only"

    @property
    def localization_authorized(self) -> bool:
        return False


def diagnose_generation_batch(
    run_identity: EvaluationRunIdentity, generations: Sequence[GenerationRecord], *, candidate_id: str,
    expected_prompt_contexts: dict[str, tuple[tuple[int, ...], bool]] | None = None,
) -> NoEditDiagnosis:
    if not isinstance(run_identity, EvaluationRunIdentity):
        raise ValueError("Batch diagnosis requires a frozen run declaration")
    if any(item.role not in ("diagnostic", "development") for item in run_identity.dataset.items):
        raise ValueError("Batch diagnosis must not select using protected dataset roles")
    if type(generations) not in (tuple, list) or not 0 < len(generations) <= 128 or len(run_identity.expected_keys) > 128:
        raise ValueError("Batch diagnosis requires at most 128 complete plain records")
    records = run_identity.validate_generations(generations, candidate_id=candidate_id,
                                               expected_prompt_contexts=expected_prompt_contexts)
    return diagnose_generation_records(records, dataset=run_identity.dataset, profile=run_identity.profile, run_identity=run_identity)


def diagnose_generation_records(
    generations: Sequence[GenerationRecord], *, dataset: DatasetManifest, profile: EvaluationProfile,
    run_identity: EvaluationRunIdentity | None = None
) -> NoEditDiagnosis:
    """Abstain from recorded channels; this is not calibrated repair evidence."""
    if not isinstance(dataset, DatasetManifest) or not isinstance(profile, EvaluationProfile):
        raise ValueError("Diagnosis requires frozen dataset/profile declarations")
    if any(item.role not in ("diagnostic", "development") for item in dataset.items):
        raise ValueError("Diagnosis must not select using protected dataset roles")
    if run_identity is not None and (not isinstance(run_identity, EvaluationRunIdentity) or run_identity.dataset.sha256 != dataset.sha256 or run_identity.profile.sha256 != profile.sha256):
        raise ValueError("Diagnosis dataset/profile does not match the frozen evaluation run")
    if not isinstance(generations, Sequence) or len(dataset.items) * profile.samples_per_item > 128 or len(generations) > 128:
        raise ValueError("Diagnosis requires a bounded record sequence")
    expected = tuple(SampleKey(item.item_id, index, profile.sampling_seed)
                     for item in dataset.items for index in range(profile.samples_per_item))
    pairs = pair_generation_records(generations, generations, expected_keys=expected)
    records = tuple(original for original, _ in pairs)
    if sum(len(record.generated_token_ids) for record in records) > 32768 or sum(len(record.generated_text) for record in records) > 1048576:
        raise ValueError("Diagnosis exceeds its token/text envelope")
    reasons = set()
    for record in records:
        if run_identity is not None:
            run_identity.validate_generation(next(item for item in dataset.items if item.item_id == record.key.item_id), record)
        if record.profile_id != profile.sha256 or record.prompt_tokens > profile.max_model_len - profile.max_new_tokens:
            raise ValueError("Diagnosis generation/profile identity or token envelope mismatch")
        observed = observe_generation_health(record, thinking_required=profile.thinking, profile=profile)
        health = record.health
        if health is None:
            reasons.add("health_unavailable")
        else:
            if health.cap_hit != observed.cap_hit or health.thinking != observed.thinking:
                raise ValueError("Diagnosis health does not match actual framing")
            if health.thinking.required and (health.thinking.closure_complete is not True or health.thinking.malformed is not False):
                reasons.add("thinking_framing_incomplete_or_unavailable")
            if health.repetition is None or health.repetition.status != "complete" or health.token_tail is None or health.token_tail.status != "complete":
                reasons.add("pattern_observations_unavailable")
            if health.repetition_detected is True:
                reasons.add("literal_repetition_observed")
            if health.token_tail is not None and health.token_tail.meets_threshold is True:
                reasons.add("declared_tail_observed")
            if health.tail_degeneration_detected is True:
                reasons.add("reported_tail_degeneration")
        quality = record.quality
        if quality is None:
            reasons.add("quality_unavailable")
        else:
            if quality.extractor_id != profile.extraction_version or quality.grader_id != f"{profile.grader}@{profile.grader_version}":
                raise ValueError("Diagnosis quality/profile identity mismatch")
            if quality.grader_status != "graded" or quality.correctness is None:
                reasons.add("quality_unavailable")
            elif quality.correctness is not True or quality.usability != "usable":
                reasons.add("recorded_task_quality_failure")
        if record.finish_reason in ("cancelled", "error"):
            reasons.add("backend_failure")
    evidence = hashlib.sha256(json.dumps([json.loads(record_to_json(record)) for record in records], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
    return NoEditDiagnosis(
        "insufficient_evidence" if reasons else "no_patch", dataset.sha256, profile.sha256, evidence,
        tuple(record.key for record in records), tuple(sorted(reasons)) or ("no_recorded_task_framing_or_literal_pattern_failure",),
        sum(record.finish_reason == "length" for record in records),
    )


def detect_loop_onset(
    token_ids: Sequence[int],
    min_ngram: int = 3,
    max_ngram: int = 48,
    min_repeats: int = 3,
    tail_window: int = 160,
    max_new_tokens: Optional[int] = None,
    eos_token_ids: Optional[Set[int]] = None,
    whitespace_token_ids: Optional[Set[int]] = None,
    whitespace_tail_threshold: int = 16,
) -> LoopAnalysis:
    """Detect whether a token sequence enters a repetitive loop, hits token cap, or degenerates.

    Classifies trajectories into:
      - `whitespace_tail`: Pathological run of whitespace/control tokens at the end of generation.
      - `ngram_loop`: Contiguous periodic block or high-density tail n-gram recurrence.
      - `cap_truncation`: Reached `max_new_tokens` without emitting an EOS token.
      - `clean`: Productive generation that terminated normally.
    """
    n = len(token_ids)
    if n == 0:
        return LoopAnalysis(
            is_looping=False,
            onset_index=None,
            period_length=None,
            repeat_count=1,
            loop_ratio=0.0,
            failure_mode="clean",
        )

    hit_cap = bool(
        max_new_tokens is not None
        and n >= max_new_tokens
        and (not eos_token_ids or token_ids[-1] not in eos_token_ids)
    )

    # 0. Check for whitespace / single-token tail degeneration (e.g. from over-scaled alpha)
    if whitespace_token_ids and n >= whitespace_tail_threshold:
        tail_run = 0
        for tok in reversed(token_ids):
            if tok in whitespace_token_ids:
                tail_run += 1
            else:
                break
        if tail_run >= whitespace_tail_threshold:
            onset = n - tail_run
            return LoopAnalysis(
                is_looping=True,
                onset_index=onset,
                period_length=1,
                repeat_count=tail_run,
                loop_ratio=round(tail_run / max(1, n), 4),
                failure_mode="whitespace_tail",
                hit_token_cap=hit_cap,
                has_whitespace_tail=True,
            )

    if n >= min_ngram * min_repeats:
        best_onset: Optional[int] = None
        best_period: Optional[int] = None
        best_repeats: int = 1

        # 1. Contiguous periodic block detection: [A][A][A]...
        max_k = min(max_ngram, n // min_repeats)
        for k in range(min_ngram, max_k + 1):
            i = 0
            while i <= n - k * min_repeats:
                pattern = token_ids[i : i + k]
                repeats = 1
                pos = i + k
                while pos + k <= n and list(token_ids[pos : pos + k]) == list(pattern):
                    repeats += 1
                    pos += k
                if repeats >= min_repeats:
                    if best_onset is None or i < best_onset:
                        best_onset = i
                        best_period = k
                        best_repeats = repeats
                    break
                i += 1

        if best_onset is not None:
            loop_tokens = n - best_onset
            return LoopAnalysis(
                is_looping=True,
                onset_index=best_onset,
                period_length=best_period,
                repeat_count=best_repeats,
                loop_ratio=round(loop_tokens / max(1, n), 4),
                failure_mode="ngram_loop",
                hit_token_cap=hit_cap,
            )

        # 2. Non-contiguous n-gram flood in the tail window (e.g. reasoning loop variations)
        start_scan = max(0, n - tail_window)
        tail = list(token_ids[start_scan:])
        if len(tail) >= min_ngram * min_repeats:
            ngram_positions: dict[tuple[int, ...], List[int]] = {}
            for idx in range(len(tail) - min_ngram + 1):
                gram = tuple(tail[idx : idx + min_ngram])
                ngram_positions.setdefault(gram, []).append(start_scan + idx)

            for _gram, positions in ngram_positions.items():
                if len(positions) >= min_repeats + 1:
                    onset = positions[0]
                    avg_period = int(
                        round(
                            (positions[-1] - positions[0])
                            / max(1, len(positions) - 1)
                        )
                    )
                    return LoopAnalysis(
                        is_looping=True,
                        onset_index=onset,
                        period_length=max(min_ngram, avg_period),
                        repeat_count=len(positions),
                        loop_ratio=round((n - onset) / max(1, n), 4),
                        failure_mode="ngram_loop",
                        hit_token_cap=hit_cap,
                    )

    # 3. Token-ceiling truncation (e.g. unclosed thinking block running out of budget)
    if hit_cap:
        onset = max(0, n - min(tail_window, max(1, n // 2)))
        return LoopAnalysis(
            is_looping=True,
            onset_index=onset,
            period_length=None,
            repeat_count=1,
            loop_ratio=round((n - onset) / max(1, n), 4),
            failure_mode="cap_truncation",
            hit_token_cap=True,
        )

    return LoopAnalysis(
        is_looping=False,
        onset_index=None,
        period_length=None,
        repeat_count=1,
        loop_ratio=0.0,
        failure_mode="clean",
        hit_token_cap=False,
    )
