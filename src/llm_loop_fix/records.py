"""Versioned, immutable evaluation inputs for the evidence-gated pipeline."""

import hashlib
import json
import math
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from itertools import combinations, product
from pathlib import Path
from typing import Iterable, Literal, Protocol, Sequence


DatasetRole = Literal[
    "diagnostic", "development", "confirmation", "benchmark", "clean_reference"
]
DATASET_ROLES = frozenset(
    {"diagnostic", "development", "confirmation", "benchmark", "clean_reference"}
)
SOURCE_MODEL_REPO = "Qwen/Qwen3.5-0.8B"
SOURCE_MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
_VLLM_FINISH_REASONS: dict[str, Literal["stop", "length", "cancelled", "error"]] = {
    "stop": "stop", "length": "length", "abort": "cancelled", "error": "error"
}


class VllmCompletionOutput(Protocol):
    @property
    def index(self) -> int: ...
    @property
    def text(self) -> str: ...
    @property
    def token_ids(self) -> Sequence[int]: ...
    @property
    def finish_reason(self) -> str | None: ...
    @property
    def stop_reason(self) -> int | str | None: ...


class VllmRequestOutput(Protocol):
    @property
    def request_id(self) -> str: ...
    @property
    def prompt_token_ids(self) -> Sequence[int] | None: ...
    @property
    def outputs(self) -> Sequence[VllmCompletionOutput]: ...
    @property
    def finished(self) -> bool: ...


def _required_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError("unsupported chat role")
        _required_text(self.content, "message content")


@dataclass(frozen=True)
class InstructionSpec:
    instruction_id: str
    kwargs_json: str = "{}"

    def __post_init__(self) -> None:
        _required_text(self.instruction_id, "instruction_id")
        _required_text(self.kwargs_json, "kwargs_json")
        if len(self.kwargs_json) > 65536:
            raise ValueError("instruction metadata exceeds the 64-KiB character budget")
        payload = json.loads(self.kwargs_json, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if payload is not None and not isinstance(payload, dict):
            raise ValueError("instruction kwargs must be an object or null")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(canonical) > 65536:
            raise ValueError("canonical instruction metadata exceeds the character budget")
        object.__setattr__(self, "kwargs_json", canonical)

    @property
    def kwargs(self) -> dict[str, object]:
        return json.loads(self.kwargs_json) or {}


@dataclass(frozen=True)
class EvaluationItem:
    item_id: str
    source: str
    source_revision: str
    role: DatasetRole
    prompt: str | None = None
    reference_answer: str | None = None
    evaluator_id: str | None = None
    subject: str | None = None
    length_group: str | None = None
    messages: tuple[ChatMessage, ...] = ()
    instructions: tuple[InstructionSpec, ...] = ()
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        for name in ("item_id", "source", "source_revision"):
            _required_text(getattr(self, name), name)
        if not isinstance(self.messages, tuple) or any(
            not isinstance(message, ChatMessage) for message in self.messages
        ):
            raise ValueError("messages must be an immutable tuple of ChatMessages")
        if (self.prompt is None) == (not self.messages):
            raise ValueError("exactly one of prompt or messages is required")
        if self.prompt is not None:
            _required_text(self.prompt, "prompt")
        if not isinstance(self.role, str) or self.role not in DATASET_ROLES:
            raise ValueError(f"unknown dataset role: {self.role}")
        if not isinstance(self.instructions, tuple) or any(
            not isinstance(instruction, InstructionSpec) for instruction in self.instructions
        ):
            raise ValueError("instructions must be an immutable tuple of InstructionSpecs")
        for name in ("reference_answer", "evaluator_id", "subject", "length_group"):
            value = getattr(self, name)
            if value is not None:
                _required_text(value, name)


def load_evaluation_items(
    path: Path | str, *, source: str, source_revision: str, role: DatasetRole
) -> tuple[EvaluationItem, ...]:
    """Assign a pinned source and immutable role; never synthesize missing IDs."""
    items = []
    identities = set()
    allowed = {"id", "prompt", "messages", "instructions", "reference_answer", "evaluator_id", "subject", "length_group"}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            if not isinstance(record, dict) or set(record) - allowed:
                raise ValueError("expected an item object with only supported fields")
            item = EvaluationItem(
                item_id=record["id"],
                prompt=record.get("prompt"),
                messages=_chat_messages(record.get("messages", [])),
                instructions=_instruction_specs(record.get("instructions", [])),
                source=source,
                source_revision=source_revision,
                role=role,
                **{name: value for name, value in record.items() if name not in {"id", "prompt", "messages", "instructions"}},
            )
            if item.item_id in identities:
                raise ValueError(f"duplicate item ID: {item.item_id}")
            identities.add(item.item_id)
            items.append(item)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    if not items:
        raise ValueError("evaluation input must not be empty")
    return tuple(items)


def _chat_messages(values: object) -> tuple[ChatMessage, ...]:
    if not isinstance(values, list) or any(not isinstance(value, dict) for value in values):
        raise ValueError("messages must be a JSON array of message objects")
    return tuple(ChatMessage(**value) for value in values)


def _instruction_specs(values: object) -> tuple[InstructionSpec, ...]:
    if not isinstance(values, list):
        raise ValueError("instructions must be a JSON array")
    instructions = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {"instruction_id", "kwargs"}:
            raise ValueError("instruction object requires only instruction_id and kwargs")
        instructions.append(InstructionSpec(
            value["instruction_id"], json.dumps(value["kwargs"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        ))
    return tuple(instructions)


def _nonnegative_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True, order=True)
class SampleKey:
    item_id: str
    sample_index: int
    seed: int

    def __post_init__(self) -> None:
        _required_text(self.item_id, "item_id")
        _nonnegative_integer(self.sample_index, "sample_index")
        _nonnegative_integer(self.seed, "seed")


@dataclass(frozen=True)
class QualityObservation:
    extractor_id: str
    grader_id: str
    extracted_answer: str | None
    usability: Literal["empty", "invalid_format", "usable", "unavailable"]
    correctness: bool | None
    grader_status: Literal["graded", "unavailable", "error"]
    reason: str | None = None
    instruction_following: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.extractor_id, "extractor_id")
        _required_text(self.grader_id, "grader_id")
        if self.extracted_answer is not None:
            _required_text(self.extracted_answer, "extracted_answer")
        if self.usability not in ("empty", "invalid_format", "usable", "unavailable"):
            raise ValueError("unknown answer usability")
        if self.grader_status not in ("graded", "unavailable", "error"):
            raise ValueError("unknown grader status")
        if self.grader_status == "graded":
            if type(self.correctness) is not bool:
                raise ValueError("graded correctness must be a boolean")
        elif self.correctness is not None or not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("unavailable/error grading requires no correctness and an explicit reason")
        if self.usability == "usable" and self.extracted_answer is None:
            raise ValueError("usable answer requires an extracted answer")
        if self.correctness is True and self.usability != "usable":
            raise ValueError("correctness requires a usable extracted answer")
        if self.reason is not None:
            _required_text(self.reason, "reason")
        if not isinstance(self.instruction_following, tuple) or any(type(outcome) is not bool for outcome in self.instruction_following):
            raise ValueError("instruction outcomes must be an immutable boolean tuple")
        if self.instruction_following and self.grader_status != "graded":
            raise ValueError("instruction outcomes require completed grading")
        if self.instruction_following and self.grader_status == "graded" and self.correctness != all(self.instruction_following):
            raise ValueError("strict instruction correctness must match every instruction outcome")


@dataclass(frozen=True)
class ThinkingObservation:
    required: bool
    opening_from_prompt: bool | None
    open_delimiter_observed: bool
    close_delimiter_observed: bool
    closure_complete: bool | None
    malformed: bool | None

    def __post_init__(self) -> None:
        for name in ("required", "open_delimiter_observed", "close_delimiter_observed"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("opening_from_prompt", "closure_complete", "malformed"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be boolean or unavailable")
        if self.opening_from_prompt is None and (self.closure_complete is not None or self.malformed is not None):
            raise ValueError("unknown prompt context requires unavailable framing conclusions")
        if self.closure_complete is True and (not self.close_delimiter_observed or self.malformed is not False):
            raise ValueError("complete closure requires an observed close and valid framing")


def _token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    payload = {"encoding": "nonnegative-token-ids-v1", "token_ids": token_ids}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class RepetitionSpan:
    start: int
    end: int
    period: int
    repeats: int

    def __post_init__(self) -> None:
        for name in ("start", "end", "period", "repeats"):
            _nonnegative_integer(getattr(self, name), name)
        if self.period < 1 or self.repeats < 2 or self.end - self.start != self.period * self.repeats:
            raise ValueError("Repetition span must cover exact complete periodic blocks")


@dataclass(frozen=True)
class RepetitionObservation:
    spec_sha256: str
    token_count: int
    status: Literal["complete", "unavailable"]
    spans: tuple[RepetitionSpan, ...] = ()
    reason: str | None = None
    token_ids_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec_sha256, str) or len(self.spec_sha256) != 64 or any(character not in "0123456789abcdef" for character in self.spec_sha256):
            raise ValueError("Repetition observation requires a canonical spec digest")
        _nonnegative_integer(self.token_count, "token_count")
        if self.token_ids_sha256 is not None and (not isinstance(self.token_ids_sha256, str) or len(self.token_ids_sha256) != 64 or any(character not in "0123456789abcdef" for character in self.token_ids_sha256)):
            raise ValueError("Repetition token identity requires a canonical SHA256 digest")
        if self.status not in ("complete", "unavailable") or not isinstance(self.spans, tuple) or any(not isinstance(span, RepetitionSpan) for span in self.spans):
            raise ValueError("Repetition observation requires immutable spans and status")
        previous_end = 0
        for span in self.spans:
            if span.start < previous_end or span.end > self.token_count:
                raise ValueError("Repetition spans must be ordered, nonoverlapping and in bounds")
            previous_end = span.end
        if self.status == "unavailable":
            if self.spans:
                raise ValueError("Unavailable repetition cannot claim completed spans")
            _required_text(self.reason, "reason")
        elif self.reason is not None:
            raise ValueError("Completed repetition observation cannot carry an unavailable reason")


@dataclass(frozen=True)
class TokenTailObservation:
    spec_sha256: str
    token_ids_sha256: str
    token_count: int
    status: Literal["complete", "unavailable"]
    start: int | None = None
    end: int | None = None
    meets_threshold: bool | None = None
    terminal_eos_excluded: bool | None = None
    reason: str | None = None
    profile_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("spec_sha256", "token_ids_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("Token-tail observation requires canonical spec/token digests")
        if self.profile_sha256 is not None and (not isinstance(self.profile_sha256, str) or len(self.profile_sha256) != 64 or any(character not in "0123456789abcdef" for character in self.profile_sha256)):
            raise ValueError("Token-tail profile identity requires a canonical digest")
        _nonnegative_integer(self.token_count, "token_count")
        if self.status == "complete":
            if type(self.start) is not int or type(self.end) is not int or not 0 <= self.start <= self.end <= self.token_count or type(self.meets_threshold) is not bool or type(self.terminal_eos_excluded) is not bool:
                raise ValueError("Completed token tail requires exact bounded positions and observations")
            if self.end != self.token_count - int(self.terminal_eos_excluded) or self.reason is not None:
                raise ValueError("Token-tail endpoint must match its recorded terminal EOS")
            if self.meets_threshold and self.start == self.end:
                raise ValueError("Empty token tails cannot meet a positive threshold")
        elif self.status == "unavailable":
            if any(value is not None for value in (self.start, self.end, self.meets_threshold, self.terminal_eos_excluded)):
                raise ValueError("Unavailable token tail cannot claim completed positions or flags")
            _required_text(self.reason, "reason")
        else:
            raise ValueError("Unsupported token-tail observation status")

    @property
    def run_length(self) -> int | None:
        return self.end - self.start if self.status == "complete" else None


@dataclass(frozen=True)
class GenerationHealth:
    cap_hit: bool
    thinking: ThinkingObservation
    repetition_detected: bool | None = None
    tail_degeneration_detected: bool | None = None
    observation_version: str = "finish-thinking-v1"
    repetition: RepetitionObservation | None = None
    token_tail: TokenTailObservation | None = None

    def __post_init__(self) -> None:
        if type(self.cap_hit) is not bool or not isinstance(self.thinking, ThinkingObservation):
            raise ValueError("health requires observed cap status and immutable thinking observations")
        for name in ("repetition_detected", "tail_degeneration_detected"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be boolean or unavailable")
        _required_text(self.observation_version, "observation_version")
        if self.observation_version == "finish-thinking-repetition-v2" and self.repetition is None:
            raise ValueError("Versioned repetition health requires its observation")
        if self.observation_version == "finish-thinking-patterns-v3" and self.token_tail is None:
            raise ValueError("Versioned token-tail health requires its observation")
        if self.repetition is not None:
            if not isinstance(self.repetition, RepetitionObservation) or self.repetition.token_ids_sha256 is None:
                raise ValueError("Attached repetition requires immutable token-bound observations")
            detected = bool(self.repetition.spans) if self.repetition.status == "complete" else None
            expected_version = "finish-thinking-patterns-v3" if self.token_tail is not None else "finish-thinking-repetition-v2"
            if self.repetition_detected != detected or self.observation_version != expected_version:
                raise ValueError("Repetition health/version does not match its observation")
        if self.token_tail is not None:
            if not isinstance(self.token_tail, TokenTailObservation) or self.token_tail.profile_sha256 is None or self.observation_version != "finish-thinking-patterns-v3":
                raise ValueError("Attached token tail requires immutable profile-bound observation")
            if self.tail_degeneration_detected is not None:
                raise ValueError("Declared token-tail classes do not establish calibrated degeneration")


@dataclass(frozen=True)
class GenerationRecord:
    experiment_id: str
    candidate_id: str
    profile_id: str
    key: SampleKey
    prompt_tokens: int
    generated_token_ids: tuple[int, ...]
    generated_text: str
    finish_reason: Literal["stop", "length", "cancelled", "error"]
    quality: QualityObservation | None = None
    thinking_open_from_prompt: bool | None = None
    health: GenerationHealth | None = None
    prompt_token_ids: tuple[int, ...] | None = None
    backend_request_id: str | None = None
    backend_finish_reason: str | None = None
    backend_stop_reason: int | str | None = None
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        for name in ("experiment_id", "candidate_id", "profile_id"):
            _required_text(getattr(self, name), name)
        if not isinstance(self.key, SampleKey):
            raise ValueError("key must be a SampleKey")
        _nonnegative_integer(self.prompt_tokens, "prompt_tokens")
        if not isinstance(self.generated_token_ids, tuple):
            raise ValueError("generated_token_ids must be an immutable tuple")
        for token in self.generated_token_ids:
            _nonnegative_integer(token, "token ID")
        if not isinstance(self.generated_text, str):
            raise ValueError("generated_text must be a string")
        if self.finish_reason not in ("stop", "length", "cancelled", "error"):
            raise ValueError("finish_reason must be an observed backend outcome")
        if self.quality is not None and not isinstance(self.quality, QualityObservation):
            raise ValueError("quality must be an immutable QualityObservation")
        if self.thinking_open_from_prompt is not None and type(self.thinking_open_from_prompt) is not bool:
            raise ValueError("thinking_open_from_prompt must be boolean or unavailable")
        if self.prompt_token_ids is not None:
            if not isinstance(self.prompt_token_ids, tuple) or len(self.prompt_token_ids) != self.prompt_tokens:
                raise ValueError("prompt token snapshot must be an immutable tuple matching prompt_tokens")
            for token in self.prompt_token_ids:
                _nonnegative_integer(token, "prompt token ID")
        if self.backend_request_id is not None:
            _required_text(self.backend_request_id, "backend_request_id")
        if self.backend_finish_reason is not None:
            if not isinstance(self.backend_finish_reason, str) or self.backend_finish_reason not in _VLLM_FINISH_REASONS or _VLLM_FINISH_REASONS[self.backend_finish_reason] != self.finish_reason:
                raise ValueError("native finish reason does not match the recorded finish outcome")
        if self.backend_stop_reason is not None:
            if isinstance(self.backend_stop_reason, str):
                _required_text(self.backend_stop_reason, "backend_stop_reason")
            else:
                _nonnegative_integer(self.backend_stop_reason, "backend_stop_reason")
            if self.backend_finish_reason != "stop":
                raise ValueError("native stop detail requires a native stop outcome")
        if self.health is not None:
            if not isinstance(self.health, GenerationHealth):
                raise ValueError("health must be an immutable GenerationHealth")
            if self.health.cap_hit != (self.finish_reason == "length") or self.health.thinking.opening_from_prompt != self.thinking_open_from_prompt:
                raise ValueError("health does not match actual finish/prompt context")
            repetition = self.health.repetition
            if repetition is not None:
                if repetition.token_count != len(self.generated_token_ids) or repetition.token_ids_sha256 != _token_ids_sha256(self.generated_token_ids):
                    raise ValueError("Repetition health does not identify exact generated tokens")
                if (repetition.reason == "backend_failure") != (self.finish_reason in ("cancelled", "error")):
                    raise ValueError("Repetition availability does not match actual backend finish")
            token_tail = self.health.token_tail
            if token_tail is not None:
                if token_tail.token_count != len(self.generated_token_ids) or token_tail.token_ids_sha256 != _token_ids_sha256(self.generated_token_ids) or token_tail.profile_sha256 != self.profile_id:
                    raise ValueError("Token-tail health does not identify exact generation/profile")
                if (token_tail.reason == "backend_failure") != (self.finish_reason in ("cancelled", "error")) or (token_tail.terminal_eos_excluded is True and self.finish_reason != "stop"):
                    raise ValueError("Token-tail availability/EOS does not match actual backend finish")


def record_vllm_generations(
    item: EvaluationItem,
    output: VllmRequestOutput,
    profile: "EvaluationProfile",
    *,
    experiment_id: str,
    candidate_id: str,
    expected_request_id: str,
    expected_prompt_token_ids: tuple[int, ...],
    thinking_open_from_prompt: bool,
    backend_version: str,
    run_identity: "EvaluationRunIdentity | None" = None,
) -> tuple[GenerationRecord, ...]:
    """Snapshot complete native outputs; declarations do not validate an engine."""
    if profile.backend != "vllm" or profile.backend_version != backend_version:
        raise ValueError("native backend/version does not match the profile")
    if run_identity is not None:
        if not isinstance(run_identity, EvaluationRunIdentity) or run_identity.profile != profile or experiment_id != run_identity.sha256:
            raise ValueError("Native output does not match the frozen evaluation run/profile")
        run_identity.validate_item(item)
    if profile.seed_policy != "engine_and_sampling_seed":
        raise ValueError("unsupported native sampling seed policy")
    _required_text(expected_request_id, "expected_request_id")
    _required_text(output.request_id, "native request_id")
    if output.request_id != expected_request_id:
        raise ValueError("native request identity mismatch")
    if type(output.finished) is not bool or not output.finished:
        raise ValueError("a complete nonstreaming native request is required")
    if type(thinking_open_from_prompt) is not bool:
        raise ValueError("known prompt thinking-prefix context is required")
    if not isinstance(expected_prompt_token_ids, tuple) or not expected_prompt_token_ids:
        raise ValueError("an independently planned immutable prompt token snapshot is required")
    for token in expected_prompt_token_ids:
        _nonnegative_integer(token, "expected prompt token ID")
    if not isinstance(output.prompt_token_ids, Sequence) or isinstance(output.prompt_token_ids, (str, bytes, bytearray)):
        raise ValueError("native prompt token IDs are required")
    prompt_ids = tuple(output.prompt_token_ids)
    for token in prompt_ids:
        _nonnegative_integer(token, "native prompt token ID")
    if prompt_ids != expected_prompt_token_ids:
        raise ValueError("native prompt token identity mismatch")
    if len(prompt_ids) > profile.max_model_len - profile.max_new_tokens:
        raise ValueError("native prompt exceeds the profile token envelope")
    if not isinstance(output.outputs, Sequence) or len(output.outputs) != profile.samples_per_item:
        raise ValueError("missing or unexpected native samples")
    indexed = {}
    for completion in output.outputs:
        _nonnegative_integer(completion.index, "native sample index")
        if completion.index >= profile.samples_per_item or completion.index in indexed:
            raise ValueError("duplicate or unexpected native sample index")
        if not isinstance(completion.finish_reason, str) or completion.finish_reason not in _VLLM_FINISH_REASONS:
            raise ValueError("unfinished or unsupported native finish reason")
        if not isinstance(completion.token_ids, Sequence) or isinstance(completion.token_ids, (str, bytes, bytearray)):
            raise ValueError("native generated token IDs must be a sequence of integers")
        tokens = tuple(completion.token_ids)
        if len(tokens) > profile.max_new_tokens:
            raise ValueError("native generation exceeds the profile token envelope")
        stop = completion.stop_reason
        if stop is not None:
            if isinstance(stop, str):
                if stop not in profile.stop_sequences:
                    raise ValueError("native stop string is not declared in the profile")
            elif type(stop) is not int or stop not in profile.eos_token_ids:
                raise ValueError("native stop token is not declared in the profile")
        indexed[completion.index] = GenerationRecord(
            experiment_id, candidate_id, profile.sha256,
            SampleKey(item.item_id, completion.index, profile.sampling_seed),
            len(prompt_ids), tokens, completion.text, _VLLM_FINISH_REASONS[completion.finish_reason],
            thinking_open_from_prompt=thinking_open_from_prompt,
            prompt_token_ids=prompt_ids, backend_request_id=output.request_id,
            backend_finish_reason=completion.finish_reason, backend_stop_reason=stop,
        )
    return tuple(indexed[index] for index in range(profile.samples_per_item))


def _health_contract(health: GenerationHealth | None) -> tuple[str, bool, str | None, str | None] | None:
    if health is None:
        return None
    return health.observation_version, health.thinking.required, health.repetition.spec_sha256 if health.repetition is not None else None, health.token_tail.spec_sha256 if health.token_tail is not None else None


def pair_generation_records(
    baseline: Iterable[GenerationRecord],
    candidate: Iterable[GenerationRecord],
    *,
    expected_keys: Iterable[SampleKey],
) -> tuple[tuple[GenerationRecord, GenerationRecord], ...]:
    """Require the independently planned key set, not just equal output lengths."""
    expected = tuple(expected_keys)
    if not expected or any(not isinstance(key, SampleKey) for key in expected):
        raise ValueError("expected keys must be a nonempty collection of SampleKeys")
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate expected sample key")
    indexed = []
    for label, records in (("baseline", baseline), ("candidate", candidate)):
        mapping = {}
        identities = set()
        for record in records:
            if not isinstance(record, GenerationRecord):
                raise ValueError(f"{label} contains an invalid generation record")
            if record.key in mapping:
                raise ValueError(f"{label} duplicate sample key: {record.key}")
            identities.add((record.experiment_id, record.candidate_id, record.profile_id, _health_contract(record.health)))
            mapping[record.key] = record
        if set(mapping) != set(expected):
            raise ValueError(f"{label} has missing or unexpected sample keys")
        if len(identities) != 1:
            raise ValueError(f"{label} mixes experiment, candidate, profile, or health identities")
        indexed.append(mapping)
    pairs = []
    for key in sorted(expected):
        original, edited = indexed[0][key], indexed[1][key]
        if (original.experiment_id, original.profile_id, original.prompt_tokens, original.prompt_token_ids, original.thinking_open_from_prompt) != (
            edited.experiment_id, edited.profile_id, edited.prompt_tokens, edited.prompt_token_ids, edited.thinking_open_from_prompt
        ):
            raise ValueError(f"paired experiment/profile/prompt mismatch: {key}")
        if _health_contract(original.health) != _health_contract(edited.health):
            raise ValueError(f"paired health observation contract mismatch: {key}")
        pairs.append((original, edited))
    return tuple(pairs)


@dataclass(frozen=True)
class EvaluationProfile:
    track: Literal["source_reproduction", "validated_repair"]
    task: Literal["math500", "gsm8k", "ifeval"]
    model_repo: str
    model_revision: str
    tokenizer_repo: str
    tokenizer_revision: str
    chat_template_sha256: str
    backend: str
    backend_version: str
    grader: str
    grader_version: str
    dependency_lock_sha256: str
    eos_token_ids: tuple[int, ...]
    stop_sequences: tuple[str, ...] = ()
    dtype: str = "bfloat16"
    thinking: bool = True
    add_generation_prompt: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    max_new_tokens: int = 32768
    max_model_len: int = 34816
    engine_seed: int = 0
    sampling_seed: int = 0
    samples_per_item: int = 4
    seed_policy: str = "engine_and_sampling_seed"
    prompt_template: str = "math_qwen_boxed_v1"
    extraction_version: str = "source-v1"
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if self.track not in ("source_reproduction", "validated_repair"):
            raise ValueError("unknown evaluation track")
        if self.task not in ("math500", "gsm8k", "ifeval"):
            raise ValueError("unknown evaluation task")
        for name in ("model_repo", "tokenizer_repo", "backend", "backend_version",
                     "grader", "grader_version", "dtype", "seed_policy", "prompt_template", "extraction_version"):
            _required_text(getattr(self, name), name)
        for name in ("model_revision", "tokenizer_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be an exact lowercase 40-digit revision")
        for name in ("chat_template_sha256", "dependency_lock_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        for name in ("thinking", "add_generation_prompt"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("temperature", "top_p", "presence_penalty"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or not -2 <= self.presence_penalty <= 2:
            raise ValueError("invalid decoding range")
        for name in ("top_k", "max_new_tokens", "max_model_len", "samples_per_item", "engine_seed", "sampling_seed"):
            _nonnegative_integer(getattr(self, name), name)
        if not self.top_k or not self.max_new_tokens or not self.samples_per_item or self.max_model_len <= self.max_new_tokens:
            raise ValueError("invalid sample count, token limits, or top_k")
        if not isinstance(self.eos_token_ids, tuple) or not self.eos_token_ids:
            raise ValueError("eos_token_ids must be a nonempty immutable tuple")
        for token in self.eos_token_ids:
            _nonnegative_integer(token, "EOS token ID")
        if len(set(self.eos_token_ids)) != len(self.eos_token_ids):
            raise ValueError("duplicate EOS token ID")
        if not isinstance(self.stop_sequences, tuple):
            raise ValueError("stop_sequences must be an immutable tuple")
        for stop in self.stop_sequences:
            _required_text(stop, "stop sequence")
        if self.track == "source_reproduction":
            if (self.model_repo, self.model_revision, self.tokenizer_repo, self.tokenizer_revision) != (
                SOURCE_MODEL_REPO, SOURCE_MODEL_REVISION, SOURCE_MODEL_REPO, SOURCE_MODEL_REVISION
            ):
                raise ValueError("source-reproduction model/tokenizer identity does not match the pinned source")
            if self.task in ("math500", "gsm8k") and (self.grader, self.grader_version) != ("math-verify", "0.9.0"):
                raise ValueError("source math grader must be math-verify 0.9.0")
            if self.task == "ifeval" and self.grader != "ifeval-strict":
                raise ValueError("source IFEval grader must use strict prompt-level checking")
            expected_samples = 4 if self.task == "math500" else 2
            expected_template = "ifeval_user_v1" if self.task == "ifeval" else "math_qwen_boxed_v1"
            if (self.dtype, self.thinking, self.add_generation_prompt, self.temperature,
                self.top_p, self.top_k, self.presence_penalty, self.max_new_tokens,
                self.max_model_len, self.engine_seed, self.sampling_seed, self.samples_per_item,
                self.prompt_template, self.extraction_version, self.stop_sequences, self.seed_policy) != (
                "bfloat16", True, True, 0.6, 0.95, 20, 1.5, 32768, 34816, 0, 0,
                expected_samples, expected_template, "source-v1", (), "engine_and_sampling_seed"
            ):
                raise ValueError("source-reproduction protocol settings do not match the pinned source")

    def vllm_sampling_kwargs(self, *, backend_version: str) -> dict[str, object]:
        if self.backend != "vllm" or self.backend_version != backend_version:
            raise ValueError("native backend/version does not match the profile")
        if self.seed_policy != "engine_and_sampling_seed":
            raise ValueError("unsupported native sampling seed policy")
        return {
            "n": self.samples_per_item, "temperature": self.temperature,
            "top_p": self.top_p, "top_k": self.top_k,
            "presence_penalty": self.presence_penalty, "max_tokens": self.max_new_tokens,
            "seed": self.sampling_seed, "stop": list(self.stop_sequences),
            "stop_token_ids": list(self.eos_token_ids), "ignore_eos": False,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(record_to_json(self).encode("utf-8")).hexdigest()


def record_to_json(record: EvaluationItem | GenerationRecord | EvaluationProfile) -> str:
    names = {EvaluationItem: "evaluation_item", GenerationRecord: "generation", EvaluationProfile: "evaluation_profile"}
    if type(record) not in names:
        raise ValueError("unsupported record type")
    payload = asdict(record)
    if isinstance(record, EvaluationItem):
        payload["instructions"] = [
            {"instruction_id": instruction.instruction_id, "kwargs": json.loads(instruction.kwargs_json)}
            for instruction in record.instructions
        ]
    payload["record_type"] = names[type(record)]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def record_from_json(text: str) -> EvaluationItem | GenerationRecord | EvaluationProfile:
    payload = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int:
        raise ValueError("a versioned record object is required")
    if payload.pop("schema_version") != 1:
        raise ValueError("unsupported record schema version")
    record_type = payload.pop("record_type", None)
    try:
        if record_type == "evaluation_item":
            payload["messages"] = _chat_messages(payload.get("messages", []))
            payload["instructions"] = _instruction_specs(payload.get("instructions", []))
            return EvaluationItem(**payload)
        if record_type == "generation":
            payload["key"] = SampleKey(**payload["key"])
            if payload.get("prompt_token_ids") is not None:
                if not isinstance(payload["prompt_token_ids"], list):
                    raise ValueError("prompt_token_ids must be a JSON array")
                payload["prompt_token_ids"] = tuple(payload["prompt_token_ids"])
            if payload.get("quality") is not None:
                outcomes = payload["quality"].get("instruction_following", [])
                if not isinstance(outcomes, list):
                    raise ValueError("instruction_following must be a JSON array")
                payload["quality"]["instruction_following"] = tuple(outcomes)
                payload["quality"] = QualityObservation(**payload["quality"])
            if payload.get("health") is not None:
                payload["health"]["thinking"] = ThinkingObservation(**payload["health"]["thinking"])
                repetition = payload["health"].get("repetition")
                if repetition is not None:
                    if not isinstance(repetition, dict) or not isinstance(repetition.get("spans"), list):
                        raise ValueError("Repetition observation requires a JSON object with span array")
                    repetition["spans"] = tuple(RepetitionSpan(**span) for span in repetition["spans"])
                    payload["health"]["repetition"] = RepetitionObservation(**repetition)
                token_tail = payload["health"].get("token_tail")
                if token_tail is not None:
                    if not isinstance(token_tail, dict):
                        raise ValueError("Token-tail observation requires a JSON object")
                    payload["health"]["token_tail"] = TokenTailObservation(**token_tail)
                payload["health"] = GenerationHealth(**payload["health"])
            tokens = payload["generated_token_ids"]
            if not isinstance(tokens, list):
                raise ValueError("generated_token_ids must be a JSON array")
            payload["generated_token_ids"] = tuple(tokens)
            return GenerationRecord(**payload)
        if record_type == "evaluation_profile":
            for name in ("eos_token_ids", "stop_sequences"):
                if not isinstance(payload[name], list):
                    raise ValueError(f"{name} must be a JSON array")
                payload[name] = tuple(payload[name])
            return EvaluationProfile(**payload)
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid record fields: {error}") from error
    raise ValueError("unsupported record type")


@dataclass(frozen=True)
class DatasetManifest:
    items: tuple[EvaluationItem, ...]
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple) or not self.items or any(
            not isinstance(item, EvaluationItem) for item in self.items
        ):
            raise ValueError("dataset items must be a nonempty immutable tuple")
        identities = set()
        content_roles = {}
        for item in self.items:
            if item.item_id in identities:
                raise ValueError(f"duplicate item ID: {item.item_id}")
            identities.add(item.item_id)
            content = (item.prompt, item.messages)
            if content in content_roles and content_roles[content] != item.role:
                raise ValueError("exact prompt overlap across dataset roles")
            content_roles[content] = item.role

    @property
    def sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "items": [json.loads(record_to_json(item)) for item in sorted(self.items, key=lambda item: item.item_id)],
        }
        data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class EvaluationRunIdentity:
    dataset: DatasetManifest
    profile: EvaluationProfile
    code_revision: str
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if type(self.dataset) is not DatasetManifest or type(self.profile) is not EvaluationProfile:
            raise ValueError("Evaluation run requires frozen dataset/profile declarations")
        if not isinstance(self.code_revision, str) or len(self.code_revision) != 40 or any(character not in "0123456789abcdef" for character in self.code_revision):
            raise ValueError("Evaluation run requires an exact code revision")
        if self.profile.seed_policy != "engine_and_sampling_seed" or len(self.dataset.items) * self.profile.samples_per_item > 4096:
            raise ValueError("Evaluation run requires a bounded declared sampling schedule")

    @property
    def scope(self) -> str:
        return "declared_data_profile_code_only"

    @property
    def sha256(self) -> str:
        payload = {"schema_version": self.schema_version, "record_type": "evaluation_run_identity",
                   "dataset_sha256": self.dataset.sha256, "profile_sha256": self.profile.sha256,
                   "code_revision": self.code_revision}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    @property
    def expected_keys(self) -> tuple[SampleKey, ...]:
        return tuple(sorted(SampleKey(item.item_id, index, self.profile.sampling_seed)
                            for item in self.dataset.items for index in range(self.profile.samples_per_item)))

    def validate_generation(self, item: EvaluationItem, generation: GenerationRecord) -> None:
        self.validate_item(item)
        if not isinstance(generation, GenerationRecord):
            raise ValueError("A generation record is required for run validation")
        if generation.experiment_id != self.sha256 or generation.profile_id != self.profile.sha256:
            raise ValueError("Generation does not identify the frozen evaluation run/profile")
        if generation.key.item_id != item.item_id or generation.key.sample_index >= self.profile.samples_per_item or generation.key.seed != self.profile.sampling_seed:
            raise ValueError("Generation sample is outside the frozen run schedule")

    def validate_item(self, item: EvaluationItem) -> None:
        if not isinstance(item, EvaluationItem) or item not in self.dataset.items:
            raise ValueError("Evaluation item does not match the frozen run dataset")

    def to_json(self) -> str:
        payload = {"schema_version": self.schema_version, "record_type": "evaluation_run_identity",
                   "scope": self.scope, "dataset_sha256": self.dataset.sha256,
                   "profile_sha256": self.profile.sha256, "sha256": self.sha256,
                   "code_revision": self.code_revision,
                   "items": [json.loads(record_to_json(item)) for item in sorted(self.dataset.items, key=lambda item: item.item_id)],
                   "profile": json.loads(record_to_json(self.profile))}
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(text.encode("utf-8")) > 1048576:
            raise ValueError("Run JSON exceeds the one-MiB UTF-8 budget")
        return text

    @classmethod
    def from_json(cls, text: str, *, expected_sha256: str | None = None) -> "EvaluationRunIdentity":
        if not isinstance(text, str) or len(text) > 1048576 or len(text.encode("utf-8")) > 1048576:
            raise ValueError("Run JSON exceeds the one-MiB UTF-8 budget")
        if expected_sha256 is not None and (not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256)):
            raise ValueError("Expected run identity must be a canonical SHA-256")
        try:
            payload = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        except (json.JSONDecodeError, RecursionError) as error:
            raise ValueError("Invalid run JSON") from error
        fields = {"schema_version", "record_type", "scope", "dataset_sha256", "profile_sha256",
                  "sha256", "code_revision", "items", "profile"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("Run JSON requires exact declaration fields")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1 or payload["record_type"] != "evaluation_run_identity" or payload["scope"] != "declared_data_profile_code_only":
            raise ValueError("Unsupported run schema/type/scope")
        if not isinstance(payload["items"], list) or not 0 < len(payload["items"]) <= 4096 or not isinstance(payload["profile"], dict):
            raise ValueError("Run JSON requires bounded item/profile declarations")
        try:
            items = tuple(record_from_json(json.dumps(item, allow_nan=False)) for item in payload["items"])
            profile = record_from_json(json.dumps(payload["profile"], allow_nan=False))
        except (TypeError, RecursionError) as error:
            raise ValueError("Invalid nested run declarations") from error
        if any(type(item) is not EvaluationItem for item in items) or type(profile) is not EvaluationProfile:
            raise ValueError("Run JSON requires evaluation items and a profile")
        identity = cls(DatasetManifest(items), profile, payload["code_revision"])
        if payload["dataset_sha256"] != identity.dataset.sha256 or payload["profile_sha256"] != identity.profile.sha256 or payload["sha256"] != identity.sha256 or (expected_sha256 is not None and expected_sha256 != identity.sha256):
            raise ValueError("Run JSON identity mismatch")
        return identity


@dataclass(frozen=True)
class DatasetOverlap:
    left_item_id: str
    right_item_id: str
    left_role: DatasetRole
    right_role: DatasetRole
    kind: Literal["normalized_exact", "near_duplicate"]
    similarity: float


@dataclass(frozen=True)
class DatasetOverlapAudit:
    dataset_sha256: str
    near_duplicate_threshold: float
    compared_pairs: int
    findings: tuple[DatasetOverlap, ...]
    normalization: str = "nfkc-lower-whitespace-v1"
    similarity_method: str = "difflib-sequence-autojunk-disabled-v1"
    schema_version: int = field(default=1, init=False)

    @property
    def review_required(self) -> bool:
        return bool(self.findings)


def audit_dataset_overlaps(
    dataset: DatasetManifest,
    *,
    near_duplicate_threshold: float = 0.9,
    max_comparisons: int = 100000,
    max_prompt_characters: int = 16384,
) -> DatasetOverlapAudit:
    """Flag cross-role prompt similarity for review; never infer data independence."""
    if not isinstance(dataset, DatasetManifest):
        raise ValueError("a validated dataset manifest is required")
    if type(near_duplicate_threshold) not in (int, float) or not math.isfinite(near_duplicate_threshold) or not 0 < near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be finite and in (0, 1]")
    _nonnegative_integer(max_comparisons, "max_comparisons")
    _nonnegative_integer(max_prompt_characters, "max_prompt_characters")
    if not max_prompt_characters:
        raise ValueError("max_prompt_characters must be positive")
    groups = {}
    for item in sorted(dataset.items, key=lambda item: item.item_id):
        groups.setdefault(item.role, []).append(item)
    group_pairs = tuple(combinations(sorted(groups), 2))
    planned_comparisons = sum(len(groups[left]) * len(groups[right]) for left, right in group_pairs)
    if planned_comparisons > max_comparisons:
        raise ValueError("cross-role comparison budget exceeded; no partial audit returned")
    normalized = {}
    for item in dataset.items:
        messages = item.messages or (ChatMessage("user", item.prompt),)
        if sum(len(message.content) for message in messages) > max_prompt_characters:
            raise ValueError("prompt character budget exceeded; no partial audit returned")
        payload = [
            (message.role, " ".join(unicodedata.normalize("NFKC", message.content).lower().split()))
            for message in messages
        ]
        if sum(len(content) for _, content in payload) > max_prompt_characters:
            raise ValueError("normalized prompt character budget exceeded; no partial audit returned")
        normalized[item.item_id] = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    findings = []
    for left_role, right_role in group_pairs:
        for left, right in product(groups[left_role], groups[right_role]):
            left_text, right_text = normalized[left.item_id], normalized[right.item_id]
            score = SequenceMatcher(None, left_text, right_text, autojunk=False).ratio()
            if score >= near_duplicate_threshold:
                findings.append(DatasetOverlap(
                    left.item_id, right.item_id, left.role, right.role,
                    "normalized_exact" if left_text == right_text else "near_duplicate", score
                ))
    return DatasetOverlapAudit(dataset.sha256, near_duplicate_threshold, planned_comparisons, tuple(findings))