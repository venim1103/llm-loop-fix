import pytest

from llm_loop_fix.evaluation import (
    MATH_INSTRUCTION,
    build_math_prompt,
    extract_ifeval_response,
    extract_math_answer,
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