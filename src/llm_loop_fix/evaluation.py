"""Portable source-reproduction extraction, independent of generation health."""

import importlib
import importlib.metadata
import random
import signal
import threading
from copy import deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence

from .detector import observe_generation_health
from .records import EvaluationItem, EvaluationProfile, GenerationRecord, QualityObservation


class InstructionChecker(Protocol):
    def build_description(self, **kwargs: object) -> str: ...
    def get_instruction_args(self) -> Mapping[str, object] | None: ...
    def get_instruction_args_keys(self) -> Sequence[str] | None: ...
    def check_following(self, value: str) -> bool: ...


@dataclass(frozen=True)
class InstructionRegistry:
    revision: str
    factories: Mapping[str, Callable[[str], InstructionChecker]]

    def __post_init__(self) -> None:
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("a declared registry revision is required")
        if not self.factories or any(not isinstance(name, str) or not name.strip() or not callable(factory)
                                     for name, factory in self.factories.items()):
            raise ValueError("a nonempty instruction factory mapping is required")
        object.__setattr__(self, "factories", MappingProxyType(dict(self.factories)))


MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


@dataclass(frozen=True)
class MathExtraction:
    answer_text: str
    answer: str | None
    box_markers: int
    close_delimiter_observed: bool


def build_math_prompt(problem: str) -> str:
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("math problem must be a nonempty string")
    return problem + "\n" + MATH_INSTRUCTION


def extract_math_answer(text: str) -> MathExtraction:
    """Match the pinned source's marker precedence and malformed-last-box rule."""
    if not isinstance(text, str):
        raise ValueError("response must be a string")
    closed = "</think>" in text
    answer_text = text.rsplit("</think>", 1)[-1].strip() if closed else text
    marker = "\\boxed"
    position = answer_text.rfind(marker)
    if position < 0:
        marker = "\\fbox"
        position = answer_text.rfind(marker)
    answer = None
    if position >= 0:
        rest = answer_text[position + len(marker):]
        body = rest.lstrip()
        if body.startswith("{"):
            depth = 0
            for offset, character in enumerate(body):
                if character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                    if depth == 0:
                        answer = body[1:offset].strip() or None
                        break
        elif marker == "\\boxed" and rest.startswith(" "):
            answer = rest.strip().split("$", 1)[0].split("\n", 1)[0].strip() or None
    return MathExtraction(
        answer_text, answer, answer_text.count("\\boxed") + answer_text.count("\\fbox"), closed
    )


def extract_ifeval_response(text: str, *, thinking: bool) -> str:
    if not isinstance(text, str) or type(thinking) is not bool:
        raise ValueError("response must be a string and thinking must be a boolean")
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return "" if thinking else text.strip()


def evaluate_math_generation(
    item: EvaluationItem, generation: GenerationRecord, profile: EvaluationProfile
) -> GenerationRecord:
    """Grade a matched response without turning infrastructure failure into wrongness."""
    if profile.task not in ("math500", "gsm8k"):
        raise ValueError("a math task profile is required")
    if item.instructions:
        raise ValueError("math items must not carry IFEval instruction metadata")
    if item.item_id != generation.key.item_id or generation.profile_id != profile.sha256:
        raise ValueError("item/generation/profile identity mismatch")
    if item.evaluator_id is not None and item.evaluator_id != profile.grader:
        raise ValueError("item evaluator does not match the profile")
    if profile.grader != "math-verify" or profile.grader_version != "0.9.0":
        raise ValueError("only the pinned math-verify grader is implemented")
    extraction = extract_math_answer(generation.generated_text)
    usability = "usable" if extraction.answer is not None else (
        "invalid_format" if extraction.answer_text.strip() else "empty"
    )

    def result(correctness, status, reason=None):
        quality = QualityObservation(
            profile.extraction_version, f"{profile.grader}@{profile.grader_version}",
            extraction.answer, usability, correctness, status, reason
        )
        return replace(generation, quality=quality, health=observe_generation_health(generation, thinking_required=profile.thinking))

    if generation.finish_reason in ("cancelled", "error"):
        return result(None, "unavailable", f"backend_{generation.finish_reason}")
    if item.reference_answer is None:
        return result(None, "unavailable", "missing_reference_answer")
    if generation.prompt_tokens > profile.max_model_len - profile.max_new_tokens or len(generation.generated_token_ids) > profile.max_new_tokens:
        return result(None, "error", "generation_exceeds_profile_token_limits")
    if extraction.answer is None:
        return result(False, "graded", "no_extracted_answer")
    if threading.current_thread() is not threading.main_thread():
        return result(None, "unavailable", "grader_requires_main_thread_timeouts")
    if hasattr(signal, "getitimer") and signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        return result(None, "unavailable", "active_alarm_would_be_overwritten")
    try:
        for package, version in (("math-verify", "0.9.0"), ("latex2sympy2-extended", "1.11.0"),
                                 ("antlr4-python3-runtime", "4.13.2")):
            if importlib.metadata.version(package) != version:
                return result(None, "unavailable", "grader_dependency_version_mismatch")
        grader = importlib.import_module("math_verify")
        timeout_error = importlib.import_module("math_verify.errors").TimeoutException
    except (ImportError, importlib.metadata.PackageNotFoundError):
        return result(None, "unavailable", "grader_dependency_missing")
    try:
        gold = grader.parse("\\boxed{" + item.reference_answer + "}", parsing_timeout=5, raise_on_error=True)
        prediction = grader.parse("\\boxed{" + extraction.answer + "}", parsing_timeout=5, raise_on_error=True)
        if not gold:
            return result(None, "unavailable", "unparseable_reference_answer")
        correctness = bool(prediction) and bool(grader.verify(gold, prediction, timeout_seconds=5, raise_on_error=True))
        return result(correctness, "graded")
    except (Exception, timeout_error) as error:
        return result(None, "error", type(error).__name__)


def evaluate_ifeval_generation(
    item: EvaluationItem,
    generation: GenerationRecord,
    profile: EvaluationProfile,
    *,
    registry: InstructionRegistry,
) -> GenerationRecord:
    """Apply every fixed constraint; a declared registry is not proof of code parity."""
    if profile.task != "ifeval" or profile.grader != "ifeval-strict":
        raise ValueError("a strict IFEval profile is required")
    if generation.profile_id != profile.sha256 or item.item_id != generation.key.item_id:
        raise ValueError("item/generation/profile identity mismatch")
    if item.evaluator_id is not None and item.evaluator_id != profile.grader:
        raise ValueError("item evaluator does not match the profile")
    response = extract_ifeval_response(generation.generated_text, thinking=profile.thinking)

    def result(correctness, status, reason=None, outcomes=()):
        quality = QualityObservation(
            profile.extraction_version, f"{profile.grader}@{profile.grader_version}",
            response or None, "usable" if response else "empty", correctness, status, reason, outcomes
        )
        return replace(generation, quality=quality, health=observe_generation_health(generation, thinking_required=profile.thinking))

    if registry.revision != profile.grader_version:
        return result(None, "unavailable", "registry_revision_mismatch")
    if not item.instructions:
        return result(None, "unavailable", "missing_instruction_constraints")
    if generation.finish_reason in ("cancelled", "error"):
        return result(None, "unavailable", f"backend_{generation.finish_reason}")
    if generation.prompt_tokens > profile.max_model_len - profile.max_new_tokens or len(generation.generated_token_ids) > profile.max_new_tokens:
        return result(None, "error", "generation_exceeds_profile_token_limits")
    if any(instruction.instruction_id not in registry.factories for instruction in item.instructions):
        return result(None, "unavailable", "unknown_instruction_id")
    if threading.current_thread() is not threading.main_thread() or any(
        not hasattr(signal, name) for name in ("SIGALRM", "ITIMER_REAL", "getitimer", "alarm")
    ):
        return result(None, "unavailable", "grader_requires_main_thread_unix_timeout")
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        return result(None, "unavailable", "active_alarm_would_be_overwritten")

    def interrupt(signum, frame):
        raise TimeoutError("instruction checking deadline exceeded")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_random_state = random.getstate()
    signal.signal(signal.SIGALRM, interrupt)
    try:
        signal.alarm(5)
        outcomes = []
        for specification in item.instructions:
            checker = registry.factories[specification.instruction_id](specification.instruction_id)
            keys = tuple(checker.get_instruction_args_keys() or ())
            arguments = {key: value for key, value in specification.kwargs.items() if value is not None}
            if "prompt" in keys:
                if item.prompt is None:
                    return result(None, "unavailable", "literal_prompt_required_by_instruction")
                if "prompt" in arguments and arguments["prompt"] != item.prompt:
                    return result(None, "unavailable", "instruction_prompt_conflicts_with_item")
                arguments["prompt"] = item.prompt
            if set(arguments) != set(keys):
                return result(None, "unavailable", "instruction_arguments_not_fully_fixed")
            fixed_arguments = deepcopy(arguments)
            checker.build_description(**arguments)
            resolved = dict(checker.get_instruction_args() or {})
            if resolved != fixed_arguments:
                return result(None, "unavailable", "instruction_arguments_changed_or_randomized")
            outcome = bool(response) and checker.check_following(response)
            if type(outcome) is not bool:
                return result(None, "error", "instruction_checker_returned_nonboolean")
            outcomes.append(outcome)
        return result(all(outcomes), "graded", outcomes=tuple(outcomes))
    except Exception as error:
        return result(None, "error", type(error).__name__)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        random.setstate(previous_random_state)