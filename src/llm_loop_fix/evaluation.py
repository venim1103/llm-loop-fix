"""Portable source-reproduction extraction, independent of generation health."""

from dataclasses import dataclass


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