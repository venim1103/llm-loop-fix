"""End-to-end unit tests for loop detection, contrastive probing, localization, and patching."""

import torch
import torch.nn as nn

from llm_loop_fix.detector import detect_loop_onset
from llm_loop_fix.localize import localize_loop_culprits
from llm_loop_fix.patch import PatchConfig, apply_surgical_patch, restore_weights
from llm_loop_fix.probe import collect_contrastive_activations
from llm_loop_fix.verify import run_alpha_sweep


class ToyMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.tanh(self.up_proj(x)))


class ToyDecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.mlp = ToyMLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class ToyLoopingCausalLM(nn.Module):
    """Synthetic small transformer where neuron 0 in layer 1's down_proj induces a 3-token loop."""

    def __init__(self, vocab_size: int = 16, d_model: int = 8, d_ff: int = 8):
        super().__init__()
        torch.manual_seed(42)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([ToyDecoderLayer(d_model, d_ff) for _ in range(2)])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Inject a strong "loop attractor" along dimension 0 in layer 1's MLP column 0
        with torch.no_grad():
            self.layers[1].mlp.up_proj.weight.zero_()
            self.layers[1].mlp.up_proj.weight[0, 0] = 5.0
            self.layers[1].mlp.down_proj.weight.zero_()
            self.layers[1].mlp.down_proj.weight[0, 0] = 10.0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(h)

    @torch.no_grad()
    def greedy_generate(self, prompt_ids: list[int], max_new_tokens: int = 18) -> list[int]:
        seq = list(prompt_ids)
        for _ in range(max_new_tokens):
            x = torch.tensor([seq], dtype=torch.long)
            h = self.embed(x)
            for layer in self.layers:
                h = layer(h)
            last_h = h[0, -1]
            # If the loop attractor feature (dim 0) is strongly positive, cycle tokens 7 -> 8 -> 9 -> 7
            if float(last_h[0].item()) > 2.5:
                cycle = {7: 8, 8: 9, 9: 7}
                next_tok = cycle.get(seq[-1], 7)
            else:
                # Productive termination sequence
                next_tok = min(15, seq[-1] + 1)
            seq.append(next_tok)
        return seq


def test_detect_loop_onset_contiguous():
    tokens = [1, 2, 3, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9]
    res = detect_loop_onset(tokens, min_ngram=3, min_repeats=3)
    assert res.is_looping is True
    assert res.onset_index == 3
    assert res.period_length == 3
    assert res.repeat_count == 4


def test_end_to_end_localization_and_surgical_fix():
    model = ToyLoopingCausalLM()
    # Set token 1 embedding dim 0 high (triggers loop) and token 2 dim 0 negative (clean)
    with torch.no_grad():
        model.embed.weight[1, 0] = 2.0
        for t in (7, 8, 9):
            model.embed.weight[t, 0] = 2.0
        for t in range(2, 7):
            model.embed.weight[t, 0] = -1.0
        for t in range(10, 16):
            model.embed.weight[t, 0] = -1.0

    loop_seq = torch.tensor(model.greedy_generate([1], max_new_tokens=15))
    clean_seq = torch.tensor(model.greedy_generate([2], max_new_tokens=15))

    acts, analyses = collect_contrastive_activations(
        model, [loop_seq, clean_seq], prompt_lengths=[1, 1], min_ngram=3, min_repeats=3
    )
    assert analyses[0].is_looping is True
    assert analyses[1].is_looping is False

    report = localize_loop_culprits(model, acts)
    top_culprit = report.ranked_culprits[0]
    assert top_culprit.module_name == "layers.1.mlp.down_proj"
    assert top_culprit.top_column_indices[0] == 0

    # Verify surgical sparse column patch eliminates the loop and can be restored
    backup = {}
    cfg = PatchConfig(
        alpha=1.0, top_k_matrices=1, mode="sparse_columns", sparse_column_fraction=0.125
    )
    patch_res = apply_surgical_patch(model, report, cfg, backup_store=backup)
    assert patch_res.modified_modules == ["layers.1.mlp.down_proj"]

    fixed_out = model.greedy_generate([1], max_new_tokens=15)
    fixed_analysis = detect_loop_onset(fixed_out[1:], min_ngram=3, min_repeats=3)
    assert fixed_analysis.is_looping is False

    restore_weights(model, backup)
    reverted_out = model.greedy_generate([1], max_new_tokens=15)
    assert detect_loop_onset(reverted_out[1:], min_ngram=3, min_repeats=3).is_looping is True

    # Test automated alpha sweep
    sweep = run_alpha_sweep(
        model=model,
        report=report,
        generate_fn=lambda m: [m.greedy_generate([1], max_new_tokens=15)[1:]],
        reference_sequences=[clean_seq],
        alphas=(0.0, 0.5, 1.0),
        top_k_matrices=1,
        mode="sparse_columns",
        sparse_column_fraction=0.125,
    )
    assert sweep.baseline_loop_rate == 1.0
    assert sweep.best_loop_rate == 0.0
    assert sweep.best_alpha == 1.0
