"""Multi-mode repetition loop, ceiling truncation, and tail degeneration detector."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set


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
