"""Token and n-gram repetition loop detector for generated sequences."""

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class LoopAnalysis:
    """Result of analyzing a token sequence for pathological repetition loops."""

    is_looping: bool
    onset_index: Optional[int]
    period_length: Optional[int]
    repeat_count: int
    loop_ratio: float


def detect_loop_onset(
    token_ids: Sequence[int],
    min_ngram: int = 3,
    max_ngram: int = 48,
    min_repeats: int = 3,
    tail_window: int = 160,
) -> LoopAnalysis:
    """Detect whether a token sequence enters a repetitive loop and find its onset index.

    Scans for contiguous repeating blocks of length `k` in `[min_ngram, max_ngram]`
    that repeat at least `min_repeats` times consecutively, or high-density n-gram
    recurrence in the tail window.
    """
    n = len(token_ids)
    if n < min_ngram * min_repeats:
        return LoopAnalysis(
            is_looping=False,
            onset_index=None,
            period_length=None,
            repeat_count=1,
            loop_ratio=0.0,
        )

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
        )

    # 2. Non-contiguous n-gram flood in the tail window (e.g. slight variations in loop)
    start_scan = max(0, n - tail_window)
    tail = list(token_ids[start_scan:])
    if len(tail) >= min_ngram * min_repeats:
        ngram_positions: dict[tuple[int, ...], List[int]] = {}
        for idx in range(len(tail) - min_ngram + 1):
            gram = tuple(tail[idx : idx + min_ngram])
            ngram_positions.setdefault(gram, []).append(start_scan + idx)

        for gram, positions in ngram_positions.items():
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
                )

    return LoopAnalysis(
        is_looping=False,
        onset_index=None,
        period_length=None,
        repeat_count=1,
        loop_ratio=0.0,
    )
