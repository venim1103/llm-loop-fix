"""Contrastive activation and layer input probing via PyTorch forward hooks."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
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
    layer_to_idx = {id(layer): idx for idx, layer in enumerate(layers)}

    results: Dict[str, Tuple[int, nn.Linear]] = {}
    for layer_idx, layer in enumerate(layers):
        for sub_name, sub_mod in layer.named_modules():
            if isinstance(sub_mod, nn.Linear) and any(
                sub_name.endswith(sfx) for sfx in target_suffixes
            ):
                full_name = f"layers.{layer_idx}.{sub_name}"
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
    model.eval()
    layers = find_transformer_layers(model)
    proj_modules = find_output_projections(model)

    stats = TrajectoryActivations()
    analyses: List[LoopAnalysis] = []

    current_layer_outputs: Dict[int, torch.Tensor] = {}
    current_proj_inputs: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_layer_hook(idx: int):
        def hook(_mod, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            current_layer_outputs[idx] = hidden.detach().float().cpu()
        return hook

    def make_proj_hook(name: str):
        def hook(_mod, inp, _out):
            x = inp[0] if isinstance(inp, tuple) else inp
            current_proj_inputs[name] = x.detach().float().cpu()
        return hook

    for idx, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_layer_hook(idx)))
    for name, (_idx, mod) in proj_modules.items():
        hooks.append(mod.register_forward_hook(make_proj_hook(name)))

    device = next(model.parameters()).device

    try:
        for seq, p_len in zip(sequences, prompt_lengths):
            gen_ids = seq[p_len:].tolist()
            analysis = detect_loop_onset(
                gen_ids, min_ngram=min_ngram, min_repeats=min_repeats
            )
            analyses.append(analysis)

            current_layer_outputs.clear()
            current_proj_inputs.clear()

            input_ids = seq.unsqueeze(0).to(device)
            model(input_ids)

            if analysis.is_looping and analysis.onset_index is not None:
                stats.num_looping_prompts += 1
                abs_onset = p_len + analysis.onset_index
                clean_slice = slice(max(0, p_len - 1), max(p_len, abs_onset))
                loop_slice = slice(abs_onset, len(seq))
            else:
                stats.num_clean_prompts += 1
                clean_slice = slice(max(0, p_len - 1), len(seq))
                loop_slice = None

            for idx, hidden in current_layer_outputs.items():
                # hidden shape: [1, seq_len, d_model]
                h = hidden[0]
                if clean_slice.stop > clean_slice.start:
                    c_vecs = h[clean_slice]
                    c_sum = c_vecs.sum(dim=0)
                    stats.residual_clean_sum[idx] = (
                        stats.residual_clean_sum.get(idx, torch.zeros_like(c_sum)) + c_sum
                    )
                    stats.residual_clean_count[idx] = (
                        stats.residual_clean_count.get(idx, 0) + c_vecs.shape[0]
                    )

                if loop_slice is not None and loop_slice.stop > loop_slice.start:
                    l_vecs = h[loop_slice]
                    l_sum = l_vecs.sum(dim=0)
                    stats.residual_loop_sum[idx] = (
                        stats.residual_loop_sum.get(idx, torch.zeros_like(l_sum)) + l_sum
                    )
                    stats.residual_loop_count[idx] = (
                        stats.residual_loop_count.get(idx, 0) + l_vecs.shape[0]
                    )

            for name, x_in in current_proj_inputs.items():
                x = x_in[0]
                if clean_slice.stop > clean_slice.start:
                    c_in = x[clean_slice]
                    c_sum = c_in.sum(dim=0)
                    stats.module_input_clean_sum[name] = (
                        stats.module_input_clean_sum.get(name, torch.zeros_like(c_sum))
                        + c_sum
                    )
                    stats.module_input_clean_count[name] = (
                        stats.module_input_clean_count.get(name, 0) + c_in.shape[0]
                    )

                if loop_slice is not None and loop_slice.stop > loop_slice.start:
                    l_in = x[loop_slice]
                    l_sum = l_in.sum(dim=0)
                    stats.module_input_loop_sum[name] = (
                        stats.module_input_loop_sum.get(name, torch.zeros_like(l_sum))
                        + l_sum
                    )
                    stats.module_input_loop_count[name] = (
                        stats.module_input_loop_count.get(name, 0) + l_in.shape[0]
                    )
    finally:
        for h in hooks:
            h.remove()

    return stats, analyses
