"""Versioned, immutable evaluation inputs for the evidence-gated pipeline."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal


DatasetRole = Literal[
    "diagnostic", "development", "confirmation", "benchmark", "clean_reference"
]
DATASET_ROLES = frozenset(
    {"diagnostic", "development", "confirmation", "benchmark", "clean_reference"}
)
SOURCE_MODEL_REPO = "Qwen/Qwen3.5-0.8B"
SOURCE_MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


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
    allowed = {"id", "prompt", "messages", "reference_answer", "evaluator_id", "subject", "length_group"}
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
                source=source,
                source_revision=source_revision,
                role=role,
                **{name: value for name, value in record.items() if name not in {"id", "prompt", "messages"}},
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
class GenerationRecord:
    experiment_id: str
    candidate_id: str
    profile_id: str
    key: SampleKey
    prompt_tokens: int
    generated_token_ids: tuple[int, ...]
    generated_text: str
    finish_reason: Literal["stop", "length", "cancelled", "error"]
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
            identities.add((record.experiment_id, record.candidate_id, record.profile_id))
            mapping[record.key] = record
        if set(mapping) != set(expected):
            raise ValueError(f"{label} has missing or unexpected sample keys")
        if len(identities) != 1:
            raise ValueError(f"{label} mixes experiment, candidate, or profile identities")
        indexed.append(mapping)
    pairs = []
    for key in sorted(expected):
        original, edited = indexed[0][key], indexed[1][key]
        if (original.experiment_id, original.profile_id, original.prompt_tokens) != (
            edited.experiment_id, edited.profile_id, edited.prompt_tokens
        ):
            raise ValueError(f"paired experiment/profile/prompt mismatch: {key}")
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

    @property
    def sha256(self) -> str:
        return hashlib.sha256(record_to_json(self).encode("utf-8")).hexdigest()


def record_to_json(record: EvaluationItem | GenerationRecord | EvaluationProfile) -> str:
    names = {EvaluationItem: "evaluation_item", GenerationRecord: "generation", EvaluationProfile: "evaluation_profile"}
    if type(record) not in names:
        raise ValueError("unsupported record type")
    payload = asdict(record)
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
            return EvaluationItem(**payload)
        if record_type == "generation":
            payload["key"] = SampleKey(**payload["key"])
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