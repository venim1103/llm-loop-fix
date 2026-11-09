import pytest
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import random
import signal
import math_verify
import llm_loop_fix.evaluation as evaluation
from math_verify.errors import TimeoutException

from llm_loop_fix.detector import RepetitionSpec, TokenTailSpec, observe_generation_health
from llm_loop_fix.evaluation import (
    MATH_INSTRUCTION,
    build_math_prompt,
    build_run_chat_template_input,
    extract_ifeval_response,
    extract_math_answer,
    evaluate_math_generation,
    evaluate_ifeval_generation,
    evaluate_generation_batch,
    InstructionRegistry,
)
from llm_loop_fix.records import (
    DatasetManifest, EvaluationItem, EvaluationProfile, EvaluationRunIdentity, GenerationRecord, SampleKey,
    SOURCE_MODEL_REPO, SOURCE_MODEL_REVISION, SOURCE_CHAT_TEMPLATE_SHA256, record_from_json, record_to_json,
    ChatMessage, InstructionSpec,
)


@pytest.mark.parametrize(
    "text,expected,markers",
    [
        (r"answer \boxed{4}", "4", 1),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}", 1),
        (r"\boxed{1} then \boxed{2}", "2", 2),
        (r"\boxed{1} then \boxed{unfinished", None, 2),
        (r"\boxed{1} then \boxed{}", None, 2),
        (r"\fbox{3}", "3", 1),
        (r"\boxed{1} then \fbox{2}", "1", 2),
        (r"\boxed { 3 }", "3", 1),
        ("\\boxed 5$ trailing", "5", 1),
        ("\\boxed 5\ntrailing", "5", 1),
        ("\\boxed\t5", None, 1),
        (r"\fbox 5", None, 1),
        ("no answer", None, 0),
        ("", None, 0),
    ],
)
def test_source_math_extractor_golden_cases(text, expected, markers):
    extraction = extract_math_answer(text)
    assert extraction.answer == expected
    assert extraction.box_markers == markers
    assert not extraction.close_delimiter_observed


def test_math_reads_full_response_without_closure_even_if_truncated():
    extraction = extract_math_answer(r"<think>reasoning \boxed{4} continued until token cap")
    assert extraction.answer == "4"
    assert not extraction.close_delimiter_observed


def test_math_and_ifeval_read_only_after_last_closure():
    text = r"<think>\boxed{1}</think>\boxed{2}</think>  \boxed{3}  "
    extraction = extract_math_answer(text)
    assert extraction.answer == "3"
    assert extraction.answer_text == r"\boxed{3}"
    assert extraction.box_markers == 1
    assert extraction.close_delimiter_observed
    assert extract_ifeval_response(text, thinking=True) == r"\boxed{3}"


def test_ifeval_does_not_read_unclosed_thinking_as_an_answer():
    text = "<think> unfinished but possibly instruction-compliant text"
    assert extract_ifeval_response(text, thinking=True) == ""
    assert extract_ifeval_response("  response  ", thinking=False) == "response"
    assert extract_ifeval_response("  ", thinking=False) == ""


def test_math_prompt_preserves_problem_and_exact_source_instruction():
    assert build_math_prompt("problem\n") == "problem\n\n" + MATH_INSTRUCTION
    with pytest.raises(ValueError):
        build_math_prompt(" ")


@pytest.fixture
def math_response():
    profile = EvaluationProfile(
        "source_reproduction", "math500", SOURCE_MODEL_REPO, SOURCE_MODEL_REVISION,
        SOURCE_MODEL_REPO, SOURCE_MODEL_REVISION, "a" * 64, "offline-fixture", "fixture-v1",
        "math-verify", "0.9.0", "b" * 64, (2,),
    )
    item = EvaluationItem("a", "fixture", "v1", "development", "question", r"\frac{1}{2}")
    generation = GenerationRecord("experiment-v1", "base", profile.sha256, SampleKey("a", 0, 0),
                                  10, (1, 2), r"<think>reasoning \boxed{0.5}", "length",
                                  thinking_open_from_prompt=False)
    return item, generation, profile


@pytest.mark.parametrize("task,samples", [("math500", 4), ("gsm8k", 2), ("ifeval", 2)])
def test_run_chat_input_binds_task_user_turn_without_reference_metadata(math_response, task, samples):
    item, _, profile = math_response
    profile = replace(profile, task=task, samples_per_item=samples, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256,
                      prompt_template="ifeval_user_v1" if task == "ifeval" else "math_qwen_boxed_v1",
                      grader="ifeval-strict" if task == "ifeval" else "math-verify")
    run_identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    result = build_run_chat_template_input(run_identity, item, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    assert result == {"conversation": [{"role": "user", "content": item.prompt if task == "ifeval" else build_math_prompt(item.prompt)}],
                      "tokenize": True, "return_dict": False, "add_generation_prompt": True, "enable_thinking": True}
    assert "reference_answer" not in result and "instructions" not in result


def test_run_chat_input_isolates_nested_mutation_and_preserves_non_thinking(math_response):
    item, _, profile = math_response
    profile = replace(profile, track="validated_repair", thinking=False, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    before = identity.to_json()
    first = build_run_chat_template_input(identity, item, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    assert first["enable_thinking"] is False
    first["conversation"][0]["role"] = "system"
    first["conversation"][0]["content"] = "altered"
    first["enable_thinking"] = True
    second = build_run_chat_template_input(identity, item, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    assert second["conversation"] == [{"role": "user", "content": build_math_prompt(item.prompt)}]
    assert second["enable_thinking"] is False and identity.to_json() == before


@pytest.mark.parametrize("changed", ["run", "item_type", "item_id", "reference", "role", "source", "messages",
                                     "math_instructions", "prompt_template", "task_template", "tokenizer", "caller_hash",
                                     "large_utf8", "large_chars"])
def test_run_chat_input_refuses_stale_or_unsupported_preparation(math_response, changed):
    item, _, profile = math_response
    profile = replace(profile, track="validated_repair", chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    if changed == "prompt_template":
        profile = replace(profile, prompt_template="unreviewed-template")
    elif changed == "task_template":
        profile = replace(profile, task="ifeval")
    elif changed == "tokenizer":
        profile = replace(profile, tokenizer_repo="other/tokenizer")
    elif changed == "messages":
        item = replace(item, prompt=None, messages=(ChatMessage("user", "question"),))
    elif changed == "math_instructions":
        item = replace(item, instructions=(InstructionSpec("controlled:fixture", "{}"),))
    elif changed == "large_utf8":
        item = replace(item, prompt="\u00e9" * 32769)
    elif changed == "large_chars":
        item = replace(item, prompt="p" * 65537)
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    supplied_identity = identity
    template_hash = SOURCE_CHAT_TEMPLATE_SHA256
    if changed == "run":
        supplied_identity = None
    elif changed == "item_type":
        item = object()
    elif changed == "item_id":
        item = replace(item, item_id="not-in-run")
    elif changed == "reference":
        item = replace(item, reference_answer="changed-reference")
    elif changed == "role":
        item = replace(item, role="confirmation")
    elif changed == "source":
        item = replace(item, source_revision="different-source")
    elif changed == "caller_hash":
        template_hash = "f" * 64
    before = identity.to_json()
    with pytest.raises(ValueError):
        build_run_chat_template_input(supplied_identity, item, chat_template_sha256=template_hash)
    assert identity.to_json() == before


def test_run_chat_input_preserves_exact_utf8_bound_without_token_budget_claim(math_response):
    item, _, profile = math_response
    item = replace(item, prompt="p" * 65536)
    profile = replace(profile, track="validated_repair", task="ifeval", prompt_template="ifeval_user_v1",
                      chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    result = build_run_chat_template_input(identity, item, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    assert result["conversation"] == [{"role": "user", "content": item.prompt}]


def test_run_chat_input_ifeval_excludes_reference_and_instruction_metadata(math_response):
    item, _, profile = math_response
    item = replace(item, reference_answer="REFERENCE_MUST_NOT_BE_SENT", evaluator_id="ifeval-strict",
                   instructions=(InstructionSpec("controlled:fixture", '{"private_metadata":"DO_NOT_SEND"}'),))
    profile = replace(profile, track="validated_repair", task="ifeval", prompt_template="ifeval_user_v1",
                      chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    result = build_run_chat_template_input(identity, item, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    assert result["conversation"] == [{"role": "user", "content": item.prompt}]
    assert "REFERENCE_MUST_NOT_BE_SENT" not in str(result) and "DO_NOT_SEND" not in str(result)


def test_complete_run_prompt_context_capture_orders_items_once_and_freezes_context_mapping(math_response):
    item, _, profile = math_response
    second = replace(item, item_id="b", prompt="second synthetic question")
    profile = replace(profile, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    identity = EvaluationRunIdentity(DatasetManifest((second, item)), profile, "e" * 40)
    before = identity.to_json()
    contexts = {"b": False, "a": True}
    calls = []
    produced = []

    def apply_template(**kwargs):
        calls.append(kwargs)
        contexts.clear()
        assert kwargs["truncation"] is False and kwargs["return_dict"] is False
        produced.append([len(calls), 7])
        return produced[-1]

    result = evaluation.capture_run_prompt_contexts(
        identity, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256,
        thinking_open_from_prompt=contexts, apply_chat_template=apply_template,
    )
    assert result == {"a": ((1, 7), True), "b": ((2, 7), False)}
    assert [call["conversation"][0]["content"] for call in calls] == [build_math_prompt(item.prompt), build_math_prompt(second.prompt)]
    assert len(calls) == 2 and not contexts
    produced[0][0] = 99
    assert result["a"] == ((1, 7), True) and identity.to_json() == before
    assert identity.snapshot_prompt_contexts(result) == result


def test_complete_run_prompt_context_capture_reorder_parity_and_input_isolation(math_response):
    item, _, profile = math_response
    second = replace(item, item_id="b", prompt="second synthetic question")
    profile = replace(profile, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    expected = None
    for items in ((item, second), (second, item)):
        identity = EvaluationRunIdentity(DatasetManifest(items), profile, "e" * 40)
        before = identity.to_json()
        calls = []

        def apply_template(**kwargs):
            calls.append(kwargs["conversation"][0]["content"])
            kwargs["conversation"][0]["content"] = "callback-local change"
            return (len(calls), 7)

        result = evaluation.capture_run_prompt_contexts(
            identity, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256,
            thinking_open_from_prompt={"a": True, "b": False}, apply_chat_template=apply_template,
        )
        assert calls == [build_math_prompt(item.prompt), build_math_prompt(second.prompt)]
        assert identity.to_json() == before
        if expected is not None:
            assert result == expected
        expected = result


@pytest.mark.parametrize("changed", ["run", "contexts_type", "missing", "extra", "context_type", "key_type",
                                     "callback", "schedule", "template_hash", "late_item"])
def test_complete_run_prompt_context_capture_preflights_every_item_before_callbacks(math_response, changed):
    item, _, profile = math_response
    profile = replace(profile, track="validated_repair", chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    items = (item, replace(item, item_id="b", prompt="second synthetic question"))
    if changed == "schedule":
        items = tuple(replace(item, item_id=str(index), prompt="question " + str(index)) for index in range(9))
        profile = replace(profile, samples_per_item=2)
    elif changed == "late_item":
        items = (item, replace(items[1], prompt="p" * 65537))
    identity = EvaluationRunIdentity(DatasetManifest(items), profile, "e" * 40)
    before = identity.to_json()
    supplied_identity = None if changed == "run" else identity
    contexts = {entry.item_id: True for entry in items}
    if changed == "contexts_type":
        contexts = tuple(contexts.items())
    elif changed == "missing":
        contexts.pop("b")
    elif changed == "extra":
        contexts["not-in-run"] = False
    elif changed == "context_type":
        contexts["b"] = 1
    elif changed == "key_type":
        contexts[1] = contexts.pop("b")
    calls = []

    def forbidden_callback(**kwargs):
        calls.append(True)
        raise AssertionError("invalid complete-run input reached a callback")

    with pytest.raises(ValueError):
        evaluation.capture_run_prompt_contexts(
            supplied_identity, chat_template_sha256="f" * 64 if changed == "template_hash" else SOURCE_CHAT_TEMPLATE_SHA256,
            thinking_open_from_prompt=contexts, apply_chat_template=None if changed == "callback" else forbidden_callback,
        )
    assert not calls and identity.to_json() == before


@pytest.mark.parametrize("output", ["boolean", "lazy", "over_profile"])
def test_complete_run_prompt_context_capture_refuses_bad_output_before_remaining_callbacks(math_response, output):
    item, _, profile = math_response
    profile = replace(profile, track="validated_repair", max_model_len=12, max_new_tokens=4,
                      chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    items = tuple(replace(item, item_id=str(index), prompt="question " + str(index)) for index in range(3))
    identity = EvaluationRunIdentity(DatasetManifest(items), profile, "e" * 40)
    calls = []

    def forbidden_generator():
        raise AssertionError("lazy token output must not be consumed")
        yield 1

    def apply_template(**kwargs):
        calls.append(True)
        if len(calls) == 1:
            return [1, 7]
        return [True] if output == "boolean" else forbidden_generator() if output == "lazy" else [1] * 9

    with pytest.raises(ValueError, match="Prompt tokenizer"):
        evaluation.capture_run_prompt_contexts(
            identity, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256,
            thinking_open_from_prompt={entry.item_id: True for entry in items}, apply_chat_template=apply_template,
        )
    assert len(calls) == 2


@pytest.mark.parametrize("failure", [RuntimeError, MemoryError, KeyboardInterrupt])
def test_complete_run_prompt_context_capture_interruptions_return_no_partial_mapping(math_response, failure):
    item, _, profile = math_response
    profile = replace(profile, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    items = (item, replace(item, item_id="b", prompt="second synthetic question"))
    identity = EvaluationRunIdentity(DatasetManifest(items), profile, "e" * 40)
    before = identity.to_json()
    contexts = {"a": True, "b": False}
    calls = []

    def apply_template(**kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise failure("controlled complete-run capture interruption")
        return [1, 7]

    with pytest.raises(failure):
        evaluation.capture_run_prompt_contexts(
            identity, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256,
            thinking_open_from_prompt=contexts, apply_chat_template=apply_template,
        )
    assert len(calls) == 2 and contexts == {"a": True, "b": False} and identity.to_json() == before


def test_complete_run_prompt_context_capture_exact_and_excess_aggregate_planned_tokens(math_response):
    item, _, profile = math_response
    profile = replace(profile, track="validated_repair", samples_per_item=1, max_model_len=65537, max_new_tokens=1,
                      chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    items = tuple(replace(item, item_id=str(index), prompt="question " + str(index)) for index in range(6))
    tokens = (1,) * 65536
    calls = []

    def apply_template(**kwargs):
        calls.append(True)
        return tokens

    identity = EvaluationRunIdentity(DatasetManifest(items[:4]), profile, "e" * 40)
    result = evaluation.capture_run_prompt_contexts(
        identity, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256,
        thinking_open_from_prompt={entry.item_id: False for entry in items[:4]}, apply_chat_template=apply_template,
    )
    assert sum(len(context[0]) for context in result.values()) == 262144 and len(calls) == 4
    calls.clear()
    identity = replace(identity, dataset=DatasetManifest(items))
    with pytest.raises(ValueError, match="aggregate planned prompt token envelope"):
        evaluation.capture_run_prompt_contexts(
            identity, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256,
            thinking_open_from_prompt={entry.item_id: False for entry in items}, apply_chat_template=apply_template,
        )
    assert len(calls) == 5


def test_complete_run_prompt_context_capture_explicit_context_is_not_profile_thinking(math_response):
    item, _, profile = math_response
    profile = replace(profile, track="validated_repair", thinking=False, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    calls = []

    def apply_template(**kwargs):
        calls.append(kwargs)
        assert kwargs["enable_thinking"] is False
        return [1, 7]

    result = evaluation.capture_run_prompt_contexts(
        identity, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256,
        thinking_open_from_prompt={"a": True}, apply_chat_template=apply_template,
    )
    assert result == {"a": ((1, 7), True)} and len(calls) == 1


def test_run_prompt_token_handoff_freezes_single_explicit_nontruncating_callback(math_response):
    item, _, profile = math_response
    profile = replace(profile, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    before = identity.to_json()
    produced = [1, 3, 4]
    calls = []

    def apply_template(**kwargs):
        calls.append(kwargs)
        assert kwargs == {"conversation": [{"role": "user", "content": build_math_prompt(item.prompt)}],
                          "tokenize": True, "return_dict": False, "add_generation_prompt": True,
                          "enable_thinking": True, "truncation": False}
        kwargs["conversation"][0]["content"] = "callback-local mutation"
        return produced

    result = evaluation.tokenize_run_chat_template_input(
        identity, item, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256, apply_chat_template=apply_template,
    )
    produced[0] = 9
    assert result == (1, 3, 4) and type(result) is tuple
    assert len(calls) == 1 and identity.to_json() == before


def test_run_prompt_token_handoff_preserves_non_thinking_ifeval_without_private_metadata(math_response):
    item, _, profile = math_response
    item = replace(item, reference_answer="PRIVATE_REFERENCE",
                   instructions=(InstructionSpec("controlled:fixture", '{"private":"PRIVATE_CHECKER"}'),))
    profile = replace(profile, track="validated_repair", task="ifeval", thinking=False,
                      prompt_template="ifeval_user_v1", grader="ifeval-strict",
                      chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    produced = (0, 3, 4)
    calls = []

    def apply_template(**kwargs):
        calls.append(kwargs)
        assert kwargs["conversation"] == [{"role": "user", "content": item.prompt}]
        assert kwargs["enable_thinking"] is False and kwargs["truncation"] is False
        assert "PRIVATE_REFERENCE" not in str(kwargs) and "PRIVATE_CHECKER" not in str(kwargs)
        return produced

    result = evaluation.tokenize_run_chat_template_input(
        identity, item, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256, apply_chat_template=apply_template,
    )
    assert result == produced and result is not produced and len(calls) == 1


@pytest.mark.parametrize("produced", [None, True, "1", b"1", {"input_ids": [1]}, [], [True], [-1],
                                      [1.0], [[1]], (1, "2"), range(2)])
def test_run_prompt_token_handoff_refuses_invalid_plain_token_snapshots(math_response, produced):
    item, _, profile = math_response
    profile = replace(profile, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    calls = []

    def apply_template(**kwargs):
        calls.append(kwargs)
        return produced

    with pytest.raises(ValueError, match="Prompt tokenizer"):
        evaluation.tokenize_run_chat_template_input(
            identity, item, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256, apply_chat_template=apply_template,
        )
    assert len(calls) == 1


@pytest.mark.parametrize("container", ["generator", "list_subclass"])
def test_run_prompt_token_handoff_refuses_custom_iterables_without_consumption(math_response, container):
    item, _, profile = math_response
    profile = replace(profile, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)

    def lazy_tokens():
        raise AssertionError("must not consume lazy tokens")
        yield 1

    class TokenList(list):
        def __iter__(self):
            raise AssertionError("must not consume custom tokens")

    produced = lazy_tokens() if container == "generator" else TokenList([1])
    with pytest.raises(ValueError, match="plain token list/tuple"):
        evaluation.tokenize_run_chat_template_input(
            identity, item, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256,
            apply_chat_template=lambda **kwargs: produced,
        )


@pytest.mark.parametrize("changed", ["reference", "template", "callback", "item_type"])
def test_run_prompt_token_handoff_preflights_before_callback(math_response, changed):
    item, _, profile = math_response
    profile = replace(profile, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    before = identity.to_json()
    template_hash = SOURCE_CHAT_TEMPLATE_SHA256
    calls = []

    def apply_template(**kwargs):
        calls.append(kwargs)
        raise AssertionError("must not execute an inadmissible callback")

    if changed == "reference":
        item = replace(item, reference_answer="stale")
    elif changed == "template":
        template_hash = "f" * 64
    elif changed == "item_type":
        item = object()
    with pytest.raises(ValueError):
        evaluation.tokenize_run_chat_template_input(
            identity, item, chat_template_sha256=template_hash,
            apply_chat_template=None if changed == "callback" else apply_template,
        )
    assert not calls and identity.to_json() == before


def test_run_prompt_token_handoff_refuses_over_profile_and_absolute_bounds_without_truncating(math_response):
    item, _, profile = math_response
    profile = replace(profile, track="validated_repair", max_model_len=16, max_new_tokens=8,
                      chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    calls = []
    produced = list(range(8))

    def apply_template(**kwargs):
        calls.append(kwargs)
        assert kwargs["truncation"] is False
        return produced

    result = evaluation.tokenize_run_chat_template_input(
        identity, item, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256, apply_chat_template=apply_template,
    )
    assert result == tuple(range(8))
    produced.append(8)
    with pytest.raises(ValueError, match="bounded nonempty"):
        evaluation.tokenize_run_chat_template_input(
            identity, item, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256, apply_chat_template=apply_template,
        )
    assert len(produced) == 9 and result == tuple(range(8))
    identity = EvaluationRunIdentity(DatasetManifest((item,)), replace(profile, max_model_len=65545), "e" * 40)
    produced = list(range(65536))
    result = evaluation.tokenize_run_chat_template_input(
        identity, item, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256, apply_chat_template=apply_template,
    )
    assert len(result) == 65536 and result[-1] == 65535
    produced.append(65536)
    with pytest.raises(ValueError, match="bounded nonempty"):
        evaluation.tokenize_run_chat_template_input(
            identity, item, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256, apply_chat_template=apply_template,
        )
    assert len(calls) == 4 and len(produced) == 65537


@pytest.mark.parametrize("failure", [RuntimeError, MemoryError, KeyboardInterrupt])
def test_run_prompt_token_handoff_propagates_callback_interruptions_without_snapshot(math_response, failure):
    item, _, profile = math_response
    profile = replace(profile, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    before = identity.to_json()
    calls = []

    def apply_template(**kwargs):
        calls.append(kwargs)
        raise failure("controlled tokenization interruption")

    with pytest.raises(failure, match="controlled tokenization interruption"):
        evaluation.tokenize_run_chat_template_input(
            identity, item, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256, apply_chat_template=apply_template,
        )
    assert len(calls) == 1 and identity.to_json() == before


def test_bounded_math_grading_preserves_correct_truncated_unclosed_answer(math_response):
    item, generation, profile = math_response
    graded = evaluate_math_generation(item, generation, profile)
    assert graded.quality.correctness is True
    assert graded.quality.usability == "usable"
    assert graded.finish_reason == "length"
    assert graded.health.cap_hit is True
    assert graded.health.thinking.closure_complete is False
    assert graded.health.thinking.malformed is False
    assert graded.health.repetition_detected is None
    assert generation.quality is None
    assert record_from_json(record_to_json(graded)) == graded


def test_grading_batch_preserves_complete_canonical_math_identity(math_response):
    item, generation, profile = math_response
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    records = tuple(replace(generation, experiment_id=identity.sha256, key=replace(generation.key, sample_index=index),
                            prompt_token_ids=tuple(range(10))) for index in range(profile.samples_per_item))
    result = evaluate_generation_batch(identity, list(reversed(records)), candidate_id="base")
    assert tuple(record.key for record in result) == identity.expected_keys
    assert all(record.quality.correctness is True and record.quality.grader_status == "graded" for record in result)
    assert all(record.experiment_id == identity.sha256 and record.profile_id == profile.sha256 for record in result)
    assert all(record.prompt_token_ids == records[0].prompt_token_ids and record.health.cap_hit for record in result)
    assert all(record.quality is None and record.health is None for record in records)


@pytest.fixture
def complete_math_batch(math_response):
    item, generation, profile = math_response
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    records = tuple(replace(generation, experiment_id=identity.sha256, key=replace(generation.key, sample_index=index),
                            prompt_token_ids=tuple(range(10))) for index in range(profile.samples_per_item))
    return identity, records


def test_grading_batch_preserves_controlled_ifeval_identity(ifeval_response):
    item, generation, profile, registry = ifeval_response
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    records = tuple(replace(generation, experiment_id=identity.sha256, key=replace(generation.key, sample_index=index),
                            prompt_token_ids=tuple(range(10))) for index in range(profile.samples_per_item))
    result = evaluate_generation_batch(identity, list(reversed(records)), candidate_id="base", registry=registry)
    assert tuple(record.key for record in result) == identity.expected_keys
    assert all(record.quality.correctness is True and record.quality.instruction_following == (True,) for record in result)
    assert all(record.quality is None for record in records)


@pytest.mark.parametrize("changed", ["key", "health", "item"])
def test_grading_batch_refuses_late_drift_before_any_parser(complete_math_batch, monkeypatch, changed):
    identity, records = complete_math_batch
    if changed == "key":
        records = records[:-1] + (replace(records[-1], key=replace(records[-1].key, seed=1)),)
    elif changed == "health":
        records = tuple(replace(record, health=observe_generation_health(record, thinking_required=True)) for record in records)
        records = records[:-1] + (replace(records[-1], generated_text=r"</think>\boxed{0.5}"),)
    else:
        second = replace(identity.dataset.items[0], item_id="b", prompt="second", evaluator_id="wrong-grader")
        identity = replace(identity, dataset=DatasetManifest(identity.dataset.items + (second,)))
        records += tuple(replace(record, key=replace(record.key, item_id="b")) for record in records)
        records = tuple(replace(record, experiment_id=identity.sha256) for record in records)
    calls = []

    def forbidden_parser(*args, **kwargs):
        calls.append(True)
        raise AssertionError("complete preflight must precede parsing")

    monkeypatch.setattr(math_verify, "parse", forbidden_parser)
    with pytest.raises(ValueError):
        evaluate_generation_batch(identity, records, candidate_id="base")
    assert not calls


@pytest.mark.parametrize("changed", ["run", "count", "unbounded", "math_registry", "ifeval_registry"])
def test_grading_batch_refuses_invalid_admission(complete_math_batch, ifeval_response, changed):
    identity, records = complete_math_batch
    registry = None
    if changed == "run":
        identity = None
    elif changed == "count":
        records = records[:1] * 17
    elif changed == "unbounded":
        def forbidden_generator():
            raise AssertionError("unbounded input must not be consumed")
            yield None
        records = forbidden_generator()
    elif changed == "math_registry":
        registry = ifeval_response[3]
    else:
        identity = replace(identity, profile=ifeval_response[2])
    with pytest.raises(ValueError):
        evaluate_generation_batch(identity, records, candidate_id="base", registry=registry)


@pytest.mark.parametrize("outcome", ["missing_reference", "backend_error", "parser_error"])
def test_grading_batch_preserves_unavailable_or_error_quality(complete_math_batch, monkeypatch, outcome):
    identity, records = complete_math_batch
    if outcome == "missing_reference":
        identity = replace(identity, dataset=DatasetManifest((replace(identity.dataset.items[0], reference_answer=None),)))
        records = tuple(replace(record, experiment_id=identity.sha256) for record in records)
    elif outcome == "backend_error":
        records = tuple(replace(record, finish_reason="error") for record in records)
    else:
        def failed_parser(*args, **kwargs):
            raise TimeoutException("controlled timeout")
        monkeypatch.setattr(math_verify, "parse", failed_parser)
    result = evaluate_generation_batch(identity, records, candidate_id="base")
    assert all(record.quality.correctness is None for record in result)
    assert all(record.quality.grader_status == ("error" if outcome == "parser_error" else "unavailable") for record in result)
    assert tuple(record.key for record in result) == identity.expected_keys
    assert all(record.quality is None for record in records)


@pytest.mark.parametrize("failure", [RuntimeError, MemoryError, KeyboardInterrupt])
def test_grading_batch_interruptions_return_no_partial_batch(complete_math_batch, monkeypatch, failure):
    identity, records = complete_math_batch
    before = tuple(record_to_json(record) for record in records)
    original = evaluation.evaluate_math_generation
    calls = []

    def interrupted_grader(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise failure("controlled interrupted batch")
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluation, "evaluate_math_generation", interrupted_grader)
    with pytest.raises(failure):
        evaluate_generation_batch(identity, records, candidate_id="base")
    assert len(calls) == 2
    assert tuple(record_to_json(record) for record in records) == before


@pytest.mark.parametrize("changed", ["generated_text", "generated_token_ids", "backend_request_id", "quality", "health"])
def test_grading_batch_refuses_corrupted_grader_output(complete_math_batch, monkeypatch, changed):
    identity, records = complete_math_batch
    records = tuple(replace(record, health=observe_generation_health(record, thinking_required=True)) for record in records)
    original = evaluation.evaluate_math_generation
    values = {"generated_text": "altered", "generated_token_ids": (9,), "backend_request_id": "altered",
              "quality": None, "health": None}

    def corrupted_grader(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, **{changed: values[changed]})

    monkeypatch.setattr(evaluation, "evaluate_math_generation", corrupted_grader)
    with pytest.raises(ValueError, match="grading result"):
        evaluate_generation_batch(identity, records, candidate_id="base")
    assert all(record.quality is None for record in records)


def test_grading_batch_preserves_exact_attached_pattern_health(complete_math_batch):
    identity, records = complete_math_batch
    first = replace(records[0], generated_text=r"reasoning</think>\boxed{0.5}", thinking_open_from_prompt=True,
                    generated_token_ids=(1, 3, 4) * 3 + (9,) * 3 + (2,), finish_reason="stop")
    profile = identity.profile
    health = observe_generation_health(first, thinking_required=True, profile=profile,
                                       repetition_spec=RepetitionSpec(max_tokens=64),
                                       tail_spec=TokenTailSpec(profile.tokenizer_repo, profile.tokenizer_revision, (9,), (2,), max_tokens=64))
    records = tuple(replace(first, key=record.key, health=health) for record in records)
    result = evaluate_generation_batch(identity, records, candidate_id="base")
    assert all(record.health is health and record.quality.correctness is True for record in result)
    assert all(record_from_json(record_to_json(record)) == record for record in result)


def test_grading_batch_refuses_nonrecord_grader_output(complete_math_batch, monkeypatch):
    identity, records = complete_math_batch
    monkeypatch.setattr(evaluation, "evaluate_math_generation", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="grading result"):
        evaluate_generation_batch(identity, records, candidate_id="base")


def test_grading_batch_retains_existing_native_metadata(complete_math_batch):
    identity, records = complete_math_batch
    records = tuple(replace(record, backend_request_id="native-fixture", backend_finish_reason="length") for record in records)
    result = evaluate_generation_batch(identity, records, candidate_id="base")
    assert all(record.backend_request_id == "native-fixture" and record.backend_finish_reason == "length" for record in result)
    assert all(record.generated_token_ids == original.generated_token_ids and record.generated_text == original.generated_text
               for record, original in zip(result, records))


def test_grading_batch_registry_mismatch_stays_unavailable(ifeval_response):
    item, generation, profile, registry = ifeval_response
    identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    records = tuple(replace(generation, experiment_id=identity.sha256, key=replace(generation.key, sample_index=index),
                            prompt_token_ids=tuple(range(10))) for index in range(profile.samples_per_item))
    result = evaluate_generation_batch(identity, records, candidate_id="base", registry=replace(registry, revision="other"))
    assert all(record.quality.correctness is None and record.quality.grader_status == "unavailable" for record in result)
    assert all(record.quality.reason == "registry_revision_mismatch" for record in result)


def test_run_bound_math_grading_preserves_matched_record(math_response):
    item, generation, profile = math_response
    run_identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    generation = replace(generation, experiment_id=run_identity.sha256)
    graded = evaluate_math_generation(item, generation, profile, run_identity=run_identity)
    assert graded.quality.correctness is True
    assert graded.experiment_id == run_identity.sha256
    assert graded.key in run_identity.expected_keys
    assert run_identity.scope == "declared_data_profile_code_only"


def test_run_bound_math_grading_refuses_same_id_reference_drift(math_response, monkeypatch):
    item, generation, profile = math_response
    run_identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    generation = replace(generation, experiment_id=run_identity.sha256)
    calls = []

    def forbidden_parser(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Reference drift must refuse before parsing")

    monkeypatch.setattr(math_verify, "parse", forbidden_parser)
    with pytest.raises(ValueError, match="frozen run dataset"):
        evaluate_math_generation(replace(item, reference_answer="7"), generation, profile, run_identity=run_identity)
    assert not calls


def test_run_bound_ifeval_grading_preserves_fixed_items(ifeval_response):
    item, generation, profile, registry = ifeval_response
    run_identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    generation = replace(generation, experiment_id=run_identity.sha256)
    graded = evaluate_ifeval_generation(item, generation, profile, registry=registry, run_identity=run_identity)
    assert graded.quality.correctness is True
    assert graded.experiment_id == run_identity.sha256


def test_run_bound_ifeval_grading_refuses_fixed_instruction_drift(ifeval_response):
    item, generation, profile, registry = ifeval_response
    run_identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    generation = replace(generation, experiment_id=run_identity.sha256)
    calls = []

    def forbidden_factory(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Instruction drift must refuse before checker construction")

    registry = replace(registry, factories={"fixture:constraint": forbidden_factory})
    changed = replace(item, instructions=(InstructionSpec("fixture:constraint", '{"expected":"other"}'),))
    with pytest.raises(ValueError, match="frozen run dataset"):
        evaluate_ifeval_generation(changed, generation, profile, registry=registry, run_identity=run_identity)
    assert not calls


@pytest.mark.parametrize("field,value", [("prompt", "changed question"), ("role", "confirmation")])
def test_run_bound_math_grading_refuses_complete_item_drift(math_response, field, value):
    item, generation, profile = math_response
    run_identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    generation = replace(generation, experiment_id=run_identity.sha256)
    with pytest.raises(ValueError, match="frozen run dataset"):
        evaluate_math_generation(replace(item, **{field: value}), generation, profile, run_identity=run_identity)


@pytest.mark.parametrize("changed", ["experiment", "sample", "seed"])
def test_run_bound_math_grading_refuses_stale_run_schedule(math_response, changed):
    item, generation, profile = math_response
    run_identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    generation = replace(generation, experiment_id=run_identity.sha256)
    if changed == "experiment":
        generation = replace(generation, experiment_id="other-experiment")
    else:
        generation = replace(generation, key=replace(generation.key, sample_index=profile.samples_per_item) if changed == "sample" else replace(generation.key, seed=1))
    with pytest.raises(ValueError, match="frozen evaluation run/profile|frozen run schedule"):
        evaluate_math_generation(item, generation, profile, run_identity=run_identity)


def test_wrong_math_answer_is_distinct_from_unavailable_grading(math_response):
    item, generation, profile = math_response
    wrong = evaluate_math_generation(item, replace(generation, generated_text=r"\boxed{7}"), profile)
    assert wrong.quality.correctness is False
    assert wrong.quality.grader_status == "graded"
    absent = evaluate_math_generation(replace(item, reference_answer=None), generation, profile)
    assert absent.quality.correctness is None
    assert absent.quality.reason == "missing_reference_answer"


@pytest.mark.parametrize("finish", ["error", "cancelled"])
def test_backend_failure_is_not_a_wrong_answer(math_response, finish):
    item, generation, profile = math_response
    graded = evaluate_math_generation(item, replace(generation, finish_reason=finish), profile)
    assert graded.quality.correctness is None
    assert graded.quality.reason == f"backend_{finish}"


def test_empty_and_unboxed_math_outputs_remain_explicit(math_response):
    item, generation, profile = math_response
    for text, usability in (("", "empty"), ("answer is 0.5", "invalid_format")):
        graded = evaluate_math_generation(item, replace(generation, generated_text=text), profile)
        assert graded.quality.usability == usability
        assert graded.quality.correctness is False


def test_math_grader_refuses_threads_and_active_alarms(math_response, monkeypatch):
    item, generation, profile = math_response
    with ThreadPoolExecutor(max_workers=1) as executor:
        graded = executor.submit(evaluate_math_generation, item, generation, profile).result()
    assert graded.quality.reason == "grader_requires_main_thread_timeouts"
    monkeypatch.setattr(signal, "getitimer", lambda timer: (1.0, 0.0))
    assert evaluate_math_generation(item, generation, profile).quality.reason == "active_alarm_would_be_overwritten"


def test_math_grader_version_mismatch_is_explicit(math_response, monkeypatch):
    item, generation, profile = math_response
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "wrong-version")
    assert evaluate_math_generation(item, generation, profile).quality.reason == "grader_dependency_version_mismatch"


def test_math_grader_refuses_mismatched_records(math_response):
    item, generation, profile = math_response
    with pytest.raises(ValueError, match="identity"):
        evaluate_math_generation(replace(item, item_id="other"), generation, profile)


@pytest.mark.parametrize("error", [RuntimeError("controlled failure"), TimeoutException("controlled timeout")])
def test_math_grader_retains_operational_errors_without_wrongness(math_response, monkeypatch, error):
    item, generation, profile = math_response

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(math_verify, "parse", fail)
    graded = evaluate_math_generation(item, generation, profile)
    assert graded.quality.correctness is None
    assert graded.quality.grader_status == "error"
    assert graded.quality.reason == type(error).__name__


def test_math_grader_rejects_generation_outside_profile_limits(math_response):
    item, generation, profile = math_response
    invalid = replace(generation, prompt_tokens=profile.max_model_len)
    graded = evaluate_math_generation(item, invalid, profile)
    assert graded.quality.correctness is None
    assert graded.quality.reason == "generation_exceeds_profile_token_limits"


def test_real_math_grading_restores_signal_handler(math_response):
    item, generation, profile = math_response
    before = signal.getsignal(signal.SIGALRM)
    assert evaluate_math_generation(item, generation, profile).quality.correctness is True
    assert signal.getsignal(signal.SIGALRM) == before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


@pytest.mark.parametrize(
    "prediction,reference,expected",
    [(r"x^2+2x+1", r"(x+1)^2", True),
     (r"\{1,2\}", r"\{2,1\}", True),
     ("2", r"\sqrt{4}", True),
     ("(0,1)", "[0,1]", False),
     ("0.3", r"\frac{1}{2}", False)],
)
def test_real_math_grader_golden_equivalence_cases(math_response, prediction, reference, expected):
    item, generation, profile = math_response
    item = replace(item, reference_answer=reference)
    generation = replace(generation, generated_text="\\boxed{" + prediction + "}", finish_reason="stop")
    graded = evaluate_math_generation(item, generation, profile)
    assert graded.quality.grader_status == "graded"
    assert graded.quality.correctness is expected


def test_source_prompt_opening_makes_generated_closing_tag_valid(math_response):
    item, generation, profile = math_response
    generation = replace(generation, generated_text=r"reasoning</think>\boxed{0.5}",
                         thinking_open_from_prompt=True, finish_reason="stop")
    graded = evaluate_math_generation(item, generation, profile)
    assert graded.quality.correctness is True
    assert graded.health.thinking.opening_from_prompt is True
    assert graded.health.thinking.open_delimiter_observed is False
    assert graded.health.thinking.closure_complete is True
    assert graded.health.thinking.malformed is False


def test_math_grading_does_not_silently_ignore_instruction_metadata(math_response):
    item, generation, profile = math_response
    item = replace(item, instructions=(InstructionSpec("fixture:word-count", '{"count":2}'),))
    with pytest.raises(ValueError, match="instruction metadata"):
        evaluate_math_generation(item, generation, profile)


@pytest.fixture
def ifeval_response(math_response):
    _, generation, math_profile = math_response
    profile = replace(math_profile, task="ifeval", samples_per_item=2, prompt_template="ifeval_user_v1",
                      grader="ifeval-strict", grader_version="controlled-registry-v1")
    item = EvaluationItem("a", "fixture", "v1", "development", "literal prompt", evaluator_id="ifeval-strict",
                          instructions=(InstructionSpec("fixture:constraint", '{"expected":"good"}'),))
    generation = replace(generation, profile_id=profile.sha256, generated_text="reasoning</think>good",
                         thinking_open_from_prompt=True, finish_reason="stop")

    class ControlledChecker:
        def __init__(self, instruction_id):
            self.arguments = {}

        def get_instruction_args_keys(self):
            return ["expected"]

        def build_description(self, **kwargs):
            self.arguments = kwargs
            return "controlled fixture"

        def get_instruction_args(self):
            return dict(self.arguments)

        def check_following(self, value):
            return value == self.arguments["expected"]

    registry = InstructionRegistry(profile.grader_version, {"fixture:constraint": ControlledChecker})
    return item, generation, profile, registry


def test_strict_ifeval_controller_records_every_constraint(ifeval_response):
    item, generation, profile, registry = ifeval_response
    graded = evaluate_ifeval_generation(item, generation, profile, registry=registry)
    assert graded.quality.correctness is True
    assert graded.quality.instruction_following == (True,)
    assert record_from_json(record_to_json(graded)) == graded
    failing = replace(item, instructions=item.instructions + (InstructionSpec("fixture:constraint", '{"expected":"other"}'),))
    graded = evaluate_ifeval_generation(failing, generation, profile, registry=registry)
    assert graded.quality.correctness is False
    assert graded.quality.instruction_following == (True, False)


@pytest.mark.parametrize("task", ["math", "ifeval"])
@pytest.mark.parametrize("status", ["graded", "unavailable", "error"])
def test_grading_preserves_pattern_health(math_response, ifeval_response, task, status):
    if task == "math":
        item, generation, profile = math_response
    else:
        item, generation, profile, registry = ifeval_response
    generation = replace(generation, generated_token_ids=(1, 3, 4) * 3 + (9,) * 3 + (2,),
                         finish_reason="error" if status == "unavailable" else "stop",
                         prompt_tokens=profile.max_model_len - profile.max_new_tokens + 1 if status == "error" else 10)
    health = observe_generation_health(
        generation, thinking_required=profile.thinking, repetition_spec=RepetitionSpec(max_tokens=128),
        tail_spec=TokenTailSpec(profile.tokenizer_repo, profile.tokenizer_revision, (9,),
                               tuple(sorted(profile.eos_token_ids)), min_run_tokens=3, max_tokens=128),
        profile=profile,
    )
    generation = replace(generation, health=health)
    if task == "math":
        graded = evaluate_math_generation(item, generation, profile)
    else:
        graded = evaluate_ifeval_generation(item, generation, profile, registry=registry)
    assert graded.quality.grader_status == status
    assert graded.quality.correctness is (True if status == "graded" else None)
    assert graded.health is health
    assert graded.health.tail_degeneration_detected is None
    assert graded.generated_token_ids == generation.generated_token_ids
    assert graded.finish_reason == generation.finish_reason
    assert generation.quality is None
    assert record_from_json(record_to_json(graded)) == graded


@pytest.mark.parametrize("task", ["math", "ifeval"])
@pytest.mark.parametrize("repetition", [False, True])
def test_grading_preserves_legacy_health_contract(math_response, ifeval_response, task, repetition):
    if task == "math":
        item, generation, profile = math_response
    else:
        item, generation, profile, registry = ifeval_response
    health = observe_generation_health(
        generation, thinking_required=profile.thinking,
        repetition_spec=RepetitionSpec(max_tokens=128) if repetition else None,
    )
    generation = replace(generation, health=health)
    if task == "math":
        graded = evaluate_math_generation(item, generation, profile)
    else:
        graded = evaluate_ifeval_generation(item, generation, profile, registry=registry)
    assert graded.health is health
    assert graded.quality.correctness is True


@pytest.mark.parametrize("task", ["math", "ifeval"])
@pytest.mark.parametrize("changed", ["text", "thinking_policy"])
def test_grading_refuses_stale_health_framing(math_response, ifeval_response, task, changed, monkeypatch):
    calls = []

    def forbidden_grader(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Incompatible health must refuse before grading")

    if task == "math":
        item, generation, profile = math_response
        monkeypatch.setattr(math_verify, "parse", forbidden_grader)
    else:
        item, generation, profile, registry = ifeval_response
        registry = replace(registry, factories={"fixture:constraint": forbidden_grader})
    health = observe_generation_health(generation, thinking_required=profile.thinking if changed == "text" else False)
    generation = replace(generation, health=health, generated_text="" if changed == "text" else generation.generated_text)
    with pytest.raises(ValueError, match="finish/thinking framing"):
        if task == "math":
            evaluate_math_generation(item, generation, profile)
        else:
            evaluate_ifeval_generation(item, generation, profile, registry=registry)
    assert not calls


def test_thinking_ifeval_without_closure_has_empty_usable_response(ifeval_response):
    item, generation, profile, registry = ifeval_response
    generation = replace(generation, generated_text="good", finish_reason="length")
    graded = evaluate_ifeval_generation(item, generation, profile, registry=registry)
    assert graded.quality.correctness is False
    assert graded.quality.usability == "empty"
    assert graded.health.cap_hit is True


def test_ifeval_cannot_pass_empty_or_unresolved_constraints(ifeval_response):
    item, generation, profile, registry = ifeval_response
    for instructions, reason in (((), "missing_instruction_constraints"),
                                 ((InstructionSpec("unknown"),), "unknown_instruction_id"),
                                 ((InstructionSpec("fixture:constraint"),), "instruction_arguments_not_fully_fixed")):
        graded = evaluate_ifeval_generation(replace(item, instructions=instructions), generation, profile, registry=registry)
        assert graded.quality.correctness is None
        assert graded.quality.reason == reason


def test_ifeval_registry_revision_and_mapping_are_frozen(ifeval_response):
    item, generation, profile, registry = ifeval_response
    with pytest.raises(TypeError):
        registry.factories["other"] = registry.factories["fixture:constraint"]
    graded = evaluate_ifeval_generation(item, generation, profile, registry=replace(registry, revision="other"))
    assert graded.quality.reason == "registry_revision_mismatch"


def test_ifeval_refuses_randomized_arguments_and_restores_rng(ifeval_response):
    item, generation, profile, registry = ifeval_response
    base = registry.factories["fixture:constraint"]

    class RandomizedChecker(base):
        def build_description(self, **kwargs):
            random.random()
            return super().build_description(expected="random fallback")

    registry = replace(registry, factories={"fixture:constraint": RandomizedChecker})
    before = random.getstate()
    graded = evaluate_ifeval_generation(item, generation, profile, registry=registry)
    assert graded.quality.correctness is None
    assert graded.quality.reason == "instruction_arguments_changed_or_randomized"
    assert random.getstate() == before


@pytest.mark.parametrize("failure", [TimeoutError("fixture timeout"), RuntimeError("fixture failure"), 1])
def test_ifeval_checker_errors_are_not_wrong_answers(ifeval_response, failure):
    item, generation, profile, registry = ifeval_response
    base = registry.factories["fixture:constraint"]

    class FailingChecker(base):
        def check_following(self, value):
            if isinstance(failure, Exception):
                raise failure
            return failure

    registry = replace(registry, factories={"fixture:constraint": FailingChecker})
    handler = signal.getsignal(signal.SIGALRM)
    graded = evaluate_ifeval_generation(item, generation, profile, registry=registry)
    assert graded.quality.correctness is None
    assert graded.quality.grader_status == "error"
    assert graded.quality.instruction_following == ()
    assert signal.getsignal(signal.SIGALRM) == handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_ifeval_preserves_explicit_repeat_target(ifeval_response):
    item, generation, profile, registry = ifeval_response
    base = registry.factories["fixture:constraint"]

    class RepeatChecker(base):
        def get_instruction_args_keys(self):
            return ["prompt_to_repeat"]

        def check_following(self, value):
            return value.startswith(self.arguments["prompt_to_repeat"])

    registry = replace(registry, factories={"fixture:constraint": RepeatChecker})
    item = replace(item, instructions=(InstructionSpec("fixture:constraint", '{"prompt_to_repeat":"good"}'),))
    assert evaluate_ifeval_generation(item, generation, profile, registry=registry).quality.correctness is True
    item = replace(item, instructions=(InstructionSpec("fixture:constraint"),))
    assert evaluate_ifeval_generation(item, generation, profile, registry=registry).quality.reason == "instruction_arguments_not_fully_fixed"


def test_ifeval_parameterless_instruction_accepts_null_args(ifeval_response):
    item, generation, profile, registry = ifeval_response
    base = registry.factories["fixture:constraint"]

    class ParameterlessChecker(base):
        def get_instruction_args_keys(self):
            return []

        def check_following(self, value):
            return True

    registry = replace(registry, factories={"fixture:constraint": ParameterlessChecker})
    item = replace(item, instructions=(InstructionSpec("fixture:constraint", "null"),))
    assert evaluate_ifeval_generation(item, generation, profile, registry=registry).quality.correctness is True


def test_ifeval_refuses_threads_and_existing_alarm(ifeval_response, monkeypatch):
    item, generation, profile, registry = ifeval_response
    with ThreadPoolExecutor(max_workers=1) as executor:
        graded = executor.submit(evaluate_ifeval_generation, item, generation, profile, registry=registry).result()
    assert graded.quality.reason == "grader_requires_main_thread_unix_timeout"
    monkeypatch.setattr(signal, "getitimer", lambda timer: (1.0, 0.0))
    graded = evaluate_ifeval_generation(item, generation, profile, registry=registry)
    assert graded.quality.reason == "active_alarm_would_be_overwritten"


def test_ifeval_rejects_in_place_nested_argument_mutation(ifeval_response):
    item, generation, profile, registry = ifeval_response
    base = registry.factories["fixture:constraint"]

    class MutatingChecker(base):
        def build_description(self, **kwargs):
            kwargs["expected"].append("changed")
            return super().build_description(**kwargs)

    registry = replace(registry, factories={"fixture:constraint": MutatingChecker})
    item = replace(item, instructions=(InstructionSpec("fixture:constraint", '{"expected":["good"]}'),))
    graded = evaluate_ifeval_generation(item, generation, profile, registry=registry)
    assert graded.quality.reason == "instruction_arguments_changed_or_randomized"
    assert item.instructions[0].kwargs == {"expected": ["good"]}


def test_ifeval_timeout_handler_restores_alarm_and_handler(ifeval_response, monkeypatch):
    item, generation, profile, registry = ifeval_response
    before = signal.getsignal(signal.SIGALRM)
    calls = []

    def expire(seconds):
        calls.append(seconds)
        if seconds:
            signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
        return 0

    monkeypatch.setattr(signal, "alarm", expire)
    graded = evaluate_ifeval_generation(item, generation, profile, registry=registry)
    assert graded.quality.reason == "TimeoutError"
    assert graded.quality.correctness is None
    assert calls == [5, 0]
    assert signal.getsignal(signal.SIGALRM) == before


def test_ifeval_abort_propagates_after_restoring_process_state(ifeval_response):
    item, generation, profile, registry = ifeval_response

    class ControlledAbort(BaseException):
        pass

    def abort(instruction_id):
        random.random()
        raise ControlledAbort()

    registry = replace(registry, factories={"fixture:constraint": abort})
    before = random.getstate(), signal.getsignal(signal.SIGALRM)
    with pytest.raises(ControlledAbort):
        evaluate_ifeval_generation(item, generation, profile, registry=registry)
    assert (random.getstate(), signal.getsignal(signal.SIGALRM)) == before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)