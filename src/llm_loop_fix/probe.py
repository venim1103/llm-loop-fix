"""Contrastive activation and layer input probing via PyTorch forward hooks."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

from .detector import LoopAnalysis, detect_loop_onset


@dataclass
class TrajectoryActivations:
    """Stores contrastive summary statistics across model layers and target projections."""

    num_looping_prompts: int = 0
    num_clean_prompts: int = 0
    # layer_idx -> [loop_mean_vec, clean_mean_vec] in R^{d_model}
    residual_loop_sum: Dict[int, torch.Tensor] = field(default_factory=dict)
    residual_clean_sum: Dict[int, torch.Tensor] = field(default_factory=dict)
    residual_loop_count: Dict[int, int] = field(default_factory=dict)
    residual_clean_count: Dict[int, int] = field(default_factory=dict)

    # module_name -> pre-projection input activation mean in R^{d_in}
    module_input_loop_sum: Dict[str, torch.Tensor] = field(default_factory=dict)
    module_input_clean_sum: Dict[str, torch.Tensor] = field(default_factory=dict)
    module_input_loop_count: Dict[str, int] = field(default_factory=dict)
    module_input_clean_count: Dict[str, int] = field(default_factory=dict)


def find_transformer_layers(model: nn.Module) -> List[nn.Module]:
    """Locate the sequential list of transformer decoder blocks across common architectures."""
    for path in [
        "model.language_model.layers",
        "language_model.layers",
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "layers",
        "blocks",
    ]:
        curr = model
        found = True
        for part in path.split("."):
            if hasattr(curr, part):
                curr = getattr(curr, part)
            else:
                found = False
                break
        if found and isinstance(curr, (nn.ModuleList, list)):
            return list(curr)

    # Fallback: find the first ModuleList inside the model
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            return list(module)
    raise ValueError("Could not automatically discover transformer layer list in model.")


def find_output_projections(
    model: nn.Module,
    target_suffixes: Tuple[str, ...] = ("down_proj", "o_proj", "c_proj", "out_proj"),
) -> Dict[str, Tuple[int, nn.Linear]]:
    """Return mapping of `module_name -> (layer_idx, nn.Linear)` for residual-writing matrices."""
    layers = find_transformer_layers(model)
    canonical_names = {id(module): name for name, module in model.named_modules()}

    results: Dict[str, Tuple[int, nn.Linear]] = {}
    for layer_idx, layer in enumerate(layers):
        for sub_name, sub_mod in layer.named_modules():
            if isinstance(sub_mod, nn.Linear) and any(
                sub_name.endswith(sfx) for sfx in target_suffixes
            ):
                full_name = canonical_names[id(sub_mod)]
                results[full_name] = (layer_idx, sub_mod)
    return results


@torch.no_grad()
def collect_contrastive_activations(
    model: nn.Module,
    sequences: List[torch.Tensor],
    prompt_lengths: List[int],
    min_ngram: int = 3,
    min_repeats: int = 3,
) -> Tuple[TrajectoryActivations, List[LoopAnalysis]]:
    """Run forward passes on generated sequences and aggregate looping vs. clean activations.

    Args:
        model: PyTorch transformer model.
        sequences: List of 1D token ID tensors `[seq_len]` containing prompt + generation.
        prompt_lengths: Prompt token length for each sequence.
        min_ngram: Minimum n-gram size for loop detection.
        min_repeats: Minimum consecutive repeats to classify as a loop.
    """
    if type(sequences) not in (list, tuple) or type(prompt_lengths) not in (list, tuple) or not sequences or len(sequences) != len(prompt_lengths):
        raise ValueError("Probe inputs require complete nonempty plain sequence/prompt-length batches")
    paired_inputs = tuple(zip(sequences, prompt_lengths))
    for sequence, prompt_length in paired_inputs:
        if (not isinstance(sequence, torch.Tensor) or sequence.ndim != 1 or sequence.numel() == 0
                or sequence.dtype not in (torch.int32, torch.int64) or type(prompt_length) is not int
                or not 0 <= prompt_length < sequence.numel()):
            raise ValueError("Probe inputs require integer token vectors and prompt lengths leaving generated tokens")
    model.eval()
    layers = find_transformer_layers(model)
    proj_modules = find_output_projections(model)

    stats = TrajectoryActivations()
    analyses: List[LoopAnalysis] = []

    current_layer_outputs: Dict[int, Dict[str, Tuple[torch.Tensor, int]]] = {}
    current_proj_inputs: Dict[str, Dict[str, Tuple[torch.Tensor, int]]] = {}
    hooks = []
    clean_slice = slice(0, 0)
    loop_slice = None

    def summarize_spans(hidden: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, int]]:
        summary = {}
        values = hidden.detach()[0]
        for label, span in (("clean", clean_slice), ("loop", loop_slice)):
            if span is not None and span.stop > span.start:
                vectors = values[span]
                summary[label] = (vectors.float().sum(dim=0).cpu(), vectors.shape[0])
        return summary

    def make_layer_hook(idx: int):
        def hook(_mod, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            current_layer_outputs[idx] = summarize_spans(hidden)
        return hook

    def make_proj_hook(name: str):
        def hook(_mod, inp, _out):
            x = inp[0] if isinstance(inp, tuple) else inp
            current_proj_inputs[name] = summarize_spans(x)
        return hook

    try:
        for idx, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(make_layer_hook(idx)))
        for name, (_idx, mod) in proj_modules.items():
            hooks.append(mod.register_forward_hook(make_proj_hook(name)))

        device = next(model.parameters()).device

        for seq, p_len in paired_inputs:
            gen_ids = seq[p_len:].tolist()
            analysis = detect_loop_onset(
                gen_ids, min_ngram=min_ngram, min_repeats=min_repeats
            )
            analyses.append(analysis)

            current_layer_outputs.clear()
            current_proj_inputs.clear()

            if analysis.is_looping and analysis.onset_index is not None:
                abs_onset = p_len + analysis.onset_index
                clean_slice = slice(max(0, p_len - 1), max(p_len, abs_onset))
                loop_slice = slice(abs_onset, len(seq))
            else:
                clean_slice = slice(max(0, p_len - 1), len(seq))
                loop_slice = None

            input_ids = seq.unsqueeze(0).to(device)
            model(input_ids)
            if loop_slice is not None:
                stats.num_looping_prompts += 1
            else:
                stats.num_clean_prompts += 1

            for idx, summary in current_layer_outputs.items():
                if "clean" in summary:
                    c_sum, c_count = summary["clean"]
                    stats.residual_clean_sum[idx] = (
                        stats.residual_clean_sum.get(idx, torch.zeros_like(c_sum)) + c_sum
                    )
                    stats.residual_clean_count[idx] = (
                        stats.residual_clean_count.get(idx, 0) + c_count
                    )

                if "loop" in summary:
                    l_sum, l_count = summary["loop"]
                    stats.residual_loop_sum[idx] = (
                        stats.residual_loop_sum.get(idx, torch.zeros_like(l_sum)) + l_sum
                    )
                    stats.residual_loop_count[idx] = (
                        stats.residual_loop_count.get(idx, 0) + l_count
                    )

            for name, summary in current_proj_inputs.items():
                if "clean" in summary:
                    c_sum, c_count = summary["clean"]
                    stats.module_input_clean_sum[name] = (
                        stats.module_input_clean_sum.get(name, torch.zeros_like(c_sum))
                        + c_sum
                    )
                    stats.module_input_clean_count[name] = (
                        stats.module_input_clean_count.get(name, 0) + c_count
                    )

                if "loop" in summary:
                    l_sum, l_count = summary["loop"]
                    stats.module_input_loop_sum[name] = (
                        stats.module_input_loop_sum.get(name, torch.zeros_like(l_sum))
                        + l_sum
                    )
                    stats.module_input_loop_count[name] = (
                        stats.module_input_loop_count.get(name, 0) + l_count
                    )
    finally:
        for h in hooks:
            h.remove()

    return stats, analyses
