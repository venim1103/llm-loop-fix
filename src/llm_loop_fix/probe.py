"""Contrastive activation and layer input probing via PyTorch forward hooks."""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple
import torch
import torch.nn as nn

from .detector import LoopAnalysis, detect_loop_onset


@dataclass(frozen=True)
class ProbeCaptureBudget:
    """Explicit collector work limits, not backend memory or execution deadlines.

    Summary elements count channel scalars transferred for all selected spans.
    Reduction elements count selected token/channel scalars before FP32 conversion.
    Both accumulate across sequences and overwritten or repeated hook invocations.
    """

    max_sequences: int
    max_sequence_tokens: int
    max_total_tokens: int
    max_summary_elements: int
    max_reduction_elements: int

    def __post_init__(self) -> None:
        for value in (self.max_sequences, self.max_sequence_tokens, self.max_total_tokens,
                      self.max_summary_elements, self.max_reduction_elements):
            if type(value) is not int or value < 0:
                raise ValueError("Probe capture budget requires nonnegative plain integer limits")


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


def generated_span_activation_bounds(
    *, sequence_tokens: int, prompt_tokens: int, generated_start: int, generated_end: int,
    convention: Literal["observed_token", "predictive_next_token"],
) -> Tuple[int, int]:
    """Map nonempty generated offsets to exclusive activation bounds.

    Coordinates do not authenticate tokens or select matched contrastive spans.
    """
    if any(type(value) is not int for value in (sequence_tokens, prompt_tokens, generated_start, generated_end)):
        raise ValueError("Generated span bounds require plain integer positions")
    if (sequence_tokens <= 0 or not 0 <= prompt_tokens < sequence_tokens
            or not 0 <= generated_start < generated_end <= sequence_tokens - prompt_tokens):
        raise ValueError("Generated span bounds require a nonempty in-range generation span")
    if type(convention) is not str or convention not in ("observed_token", "predictive_next_token"):
        raise ValueError("Generated span bounds require an explicit supported position convention")
    shift = 1 if convention == "predictive_next_token" else 0
    start, end = prompt_tokens + generated_start - shift, prompt_tokens + generated_end - shift
    if start < 0:
        raise ValueError("Predictive span requires a preceding token context")
    return start, end


def find_transformer_layers(model: nn.Module) -> List[nn.Module]:
    """Locate the sequential list of transformer decoder blocks across common architectures."""
    paths = [
        "model.language_model.layers",
        "language_model.layers",
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "layers",
        "blocks",
    ]
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    declared_qwen = type(model_type) is str and model_type in ("qwen3_5", "qwen3_5_text")
    if declared_qwen:
        paths = (["model.language_model.layers", "language_model.layers"]
                 if model_type == "qwen3_5" else ["model.layers", "layers"])
    for path in paths:
        curr = model
        found = True
        for part in path.split("."):
            if hasattr(curr, part):
                curr = getattr(curr, part)
            else:
                found = False
                break
        if found and isinstance(curr, (nn.ModuleList, list)):
            if declared_qwen and (not isinstance(curr, nn.ModuleList) or not len(curr)):
                raise ValueError("Declared Qwen3.5 text stack requires registered nonempty layers")
            return list(curr)

    if declared_qwen:
        raise ValueError("Declared Qwen3.5 text stack is missing or incompatible")
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
    sequences: List[torch.Tensor] | Tuple[torch.Tensor, ...],
    prompt_lengths: List[int] | Tuple[int, ...],
    min_ngram: int = 3,
    min_repeats: int = 3,
    *,
    capture_budget: ProbeCaptureBudget | None = None,
) -> Tuple[TrajectoryActivations, List[LoopAnalysis]]:
    """Run forward passes on generated sequences and aggregate looping vs. clean activations.

    Args:
        model: PyTorch transformer model.
        sequences: List of 1D token ID tensors `[seq_len]` containing prompt + generation.
        prompt_lengths: Prompt token length for each sequence.
        min_ngram: Minimum n-gram size for loop detection.
        min_repeats: Minimum consecutive repeats to classify as a loop.
        capture_budget: Optional complete-input and cumulative hook-work limits.
    """
    if capture_budget is not None and type(capture_budget) is not ProbeCaptureBudget:
        raise ValueError("Probe capture budget requires a frozen ProbeCaptureBudget")
    if type(sequences) not in (list, tuple) or type(prompt_lengths) not in (list, tuple) or not sequences or len(sequences) != len(prompt_lengths):
        raise ValueError("Probe inputs require complete nonempty plain sequence/prompt-length batches")
    if capture_budget is not None and len(sequences) > capture_budget.max_sequences:
        raise ValueError("Probe capture budget sequence limit exceeded")
    paired_inputs = tuple(zip(sequences, prompt_lengths))
    total_tokens = 0
    for sequence, prompt_length in paired_inputs:
        if (not isinstance(sequence, torch.Tensor) or sequence.ndim != 1 or sequence.numel() == 0
                or sequence.dtype not in (torch.int32, torch.int64) or type(prompt_length) is not int
                or not 0 <= prompt_length < sequence.numel()):
            raise ValueError("Probe inputs require integer token vectors and prompt lengths leaving generated tokens")
        total_tokens += sequence.numel()
        if capture_budget is not None and (sequence.numel() > capture_budget.max_sequence_tokens
                                           or total_tokens > capture_budget.max_total_tokens):
            raise ValueError("Probe capture budget token limit exceeded")
    input_metadata = tuple((tuple(sequence.shape), sequence.dtype) for sequence, _ in paired_inputs) if capture_budget is not None else ()
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
    expected_sequence_tokens = 0
    summary_elements = 0
    reduction_elements = 0

    def summarize_spans(hidden: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, int]]:
        nonlocal summary_elements, reduction_elements
        if capture_budget is not None and (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
                                           or hidden.shape[0] != 1 or hidden.shape[1] != expected_sequence_tokens
                                           or hidden.shape[2] == 0):
            raise ValueError("Probe capture budget requires full single-sequence activation shapes")
        summary = {}
        values = hidden.detach()[0]
        for label, span in (("clean", clean_slice), ("loop", loop_slice)):
            if span is not None and span.stop > span.start:
                vectors = values[span]
                if capture_budget is not None:
                    summary_elements += vectors.shape[1]
                    reduction_elements += vectors.numel()
                    if (summary_elements > capture_budget.max_summary_elements
                            or reduction_elements > capture_budget.max_reduction_elements):
                        raise ValueError("Probe capture budget cumulative hook-work limit exceeded")
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

        for input_index, (seq, p_len) in enumerate(paired_inputs):
            if capture_budget is not None and (tuple(seq.shape), seq.dtype) != input_metadata[input_index]:
                raise ValueError("Probe capture budget input metadata changed after preflight")
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

            expected_sequence_tokens = seq.numel()
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
