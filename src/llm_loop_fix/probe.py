"""Contrastive activation and layer input probing via PyTorch forward hooks."""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import torch
import torch.nn as nn

from .detector import LoopAnalysis, detect_loop_onset
from .records import GenerationRecord


_QWEN_TEXT_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text", "qwen2", "qwen3"})


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
    position_convention: Optional[Literal["observed_token", "predictive_next_token"]] = None


@dataclass(frozen=True)
class QwenTextAdapter:
    """Declared canonical shape snapshot, not authenticated architecture/runtime behavior."""

    model_type: str
    layer_prefix: str
    num_layers: int
    hidden_size: int
    projection_shapes: Dict[str, Tuple[int, int]]
    projection_layers: Dict[str, int]
    weight_dtypes: Dict[str, str]
    device: str
    position_convention: Literal["observed_token", "predictive_next_token"]
    backend: str = "pytorch_forward_hooks"


def inspect_qwen_text_adapter(
    model: nn.Module,
    *,
    position_convention: Literal["observed_token", "predictive_next_token"] = "observed_token",
    target_suffixes: Tuple[str, ...] = ("down_proj", "o_proj", "c_proj", "out_proj"),
) -> QwenTextAdapter:
    """Validate an explicit Qwen text stack and return its canonical module/shape contract."""
    if type(position_convention) is not str or position_convention not in ("observed_token", "predictive_next_token"):
        raise ValueError("Qwen text adapter requires an explicit supported position convention")
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if type(model_type) is not str or model_type not in _QWEN_TEXT_MODEL_TYPES:
        raise ValueError("Qwen text adapter requires an explicit supported Qwen config.model_type")
    layers = find_transformer_layers(model)
    if not layers:
        raise ValueError("Qwen adapter requires nonempty canonical text layers")
    canonical_modules = {id(module): name for name, module in model.named_modules()}
    layer_names = [canonical_modules[id(layer)] for layer in layers]
    layer_prefix = layer_names[0].rsplit(".", 1)[0] if "." in layer_names[0] else ""
    allowed_prefixes = ("model.language_model.layers", "language_model.layers") if model_type == "qwen3_5" else ("model.layers", "layers")
    if layer_prefix not in allowed_prefixes or layer_names != [f"{layer_prefix}.{index}" for index in range(len(layers))]:
        raise ValueError("Qwen adapter requires registered canonical text layer paths")
    projections = find_output_projections(model, target_suffixes=target_suffixes)
    if not projections:
        raise ValueError("Qwen text adapter requires at least one residual-writing output projection")
    hidden_size: Optional[int] = None
    device_str: Optional[str] = None
    projection_shapes: Dict[str, Tuple[int, int]] = {}
    projection_layers: Dict[str, int] = {}
    weight_dtypes: Dict[str, str] = {}
    for name, (layer_idx, mod) in projections.items():
        weight = getattr(mod, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[0] <= 0 or weight.shape[1] <= 0:
            raise ValueError(f"Qwen text adapter projection {name!r} has invalid 2D weight shape")
        out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
        if hidden_size is None:
            hidden_size = out_features
        elif out_features != hidden_size:
            raise ValueError("Qwen text adapter projections disagree on output hidden_size")
        mod_device = str(weight.device)
        if device_str is None:
            device_str = mod_device
        elif mod_device != device_str:
            raise ValueError("Qwen text adapter projections span inconsistent devices")
        projection_shapes[name] = (out_features, in_features)
        projection_layers[name] = layer_idx
        weight_dtypes[name] = str(weight.dtype)
    assert hidden_size is not None and device_str is not None
    return QwenTextAdapter(
        model_type=model_type,
        layer_prefix=layer_prefix,
        num_layers=len(layers),
        hidden_size=hidden_size,
        projection_shapes=projection_shapes,
        projection_layers=projection_layers,
        weight_dtypes=weight_dtypes,
        device=device_str,
        position_convention=position_convention,
    )


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
            if declared_qwen and (not isinstance(curr, nn.ModuleList) or not len(curr)
                                  or any(not isinstance(layer, nn.Module) for layer in curr)):
                raise ValueError("Declared Qwen3.5 text stack requires registered nonempty module layers")
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


def _explicit_analysis_slices(
    *,
    sequence_tokens: int,
    prompt_tokens: int,
    analysis: LoopAnalysis,
    convention: Literal["observed_token", "predictive_next_token"],
) -> Tuple[slice, Optional[slice]]:
    """Compute generated-span activation slices under an explicit position convention."""
    gen_len = sequence_tokens - prompt_tokens
    if analysis.is_looping and analysis.onset_index is not None:
        onset = analysis.onset_index
        if onset > 0:
            c_start, c_end = generated_span_activation_bounds(
                sequence_tokens=sequence_tokens,
                prompt_tokens=prompt_tokens,
                generated_start=0,
                generated_end=onset,
                convention=convention,
            )
            clean_slice = slice(c_start, c_end)
        else:
            clean_slice = slice(0, 0)
        l_start, l_end = generated_span_activation_bounds(
            sequence_tokens=sequence_tokens,
            prompt_tokens=prompt_tokens,
            generated_start=onset,
            generated_end=gen_len,
            convention=convention,
        )
        return clean_slice, slice(l_start, l_end)
    c_start, c_end = generated_span_activation_bounds(
        sequence_tokens=sequence_tokens,
        prompt_tokens=prompt_tokens,
        generated_start=0,
        generated_end=gen_len,
        convention=convention,
    )
    return slice(c_start, c_end), None


@torch.no_grad()
def collect_contrastive_activations(
    model: nn.Module,
    sequences: List[torch.Tensor] | Tuple[torch.Tensor, ...],
    prompt_lengths: List[int] | Tuple[int, ...],
    min_ngram: int = 3,
    min_repeats: int = 3,
    *,
    convention: Optional[Literal["observed_token", "predictive_next_token"]] = None,
    capture_budget: ProbeCaptureBudget | None = None,
) -> Tuple[TrajectoryActivations, List[LoopAnalysis]]:
    """Run forward passes on generated sequences and aggregate looping vs. clean activations.

    Args:
        model: PyTorch transformer model.
        sequences: List of 1D token ID tensors `[seq_len]` containing prompt + generation.
        prompt_lengths: Prompt token length for each sequence.
        min_ngram: Minimum n-gram size for loop detection.
        min_repeats: Minimum consecutive repeats to classify as a loop.
        convention: Optional explicit token-position convention (`"observed_token"` or
            `"predictive_next_token"`). When omitted, retains legacy boundary slicing.
        capture_budget: Optional complete-input and cumulative hook-work limits.
    """
    if convention is not None and (
        type(convention) is not str or convention not in ("observed_token", "predictive_next_token")
    ):
        raise ValueError("Probe position convention must be 'observed_token' or 'predictive_next_token'")
    if capture_budget is not None and type(capture_budget) is not ProbeCaptureBudget:
        raise ValueError("Probe capture budget requires a frozen ProbeCaptureBudget")
    if type(sequences) not in (list, tuple) or type(prompt_lengths) not in (list, tuple) or not sequences or len(sequences) != len(prompt_lengths):
        raise ValueError("Probe inputs require complete nonempty plain sequence/prompt-length batches")
    if capture_budget is not None and len(sequences) > capture_budget.max_sequences:
        raise ValueError("Probe capture budget sequence limit exceeded")
    paired_inputs = tuple(zip(sequences, prompt_lengths))
    total_tokens = 0
    preflight_analyses: List[LoopAnalysis] = []
    preflight_slices: List[Tuple[slice, Optional[slice]]] = []
    for sequence, prompt_length in paired_inputs:
        if (not isinstance(sequence, torch.Tensor) or sequence.ndim != 1 or sequence.numel() == 0
                or sequence.dtype not in (torch.int32, torch.int64) or type(prompt_length) is not int
                or not 0 <= prompt_length < sequence.numel()):
            raise ValueError("Probe inputs require integer token vectors and prompt lengths leaving generated tokens")
        total_tokens += sequence.numel()
        if capture_budget is not None and (sequence.numel() > capture_budget.max_sequence_tokens
                                           or total_tokens > capture_budget.max_total_tokens):
            raise ValueError("Probe capture budget token limit exceeded")
        if convention is not None:
            analysis = detect_loop_onset(
                sequence[prompt_length:].tolist(), min_ngram=min_ngram, min_repeats=min_repeats
            )
            preflight_analyses.append(analysis)
            preflight_slices.append(
                _explicit_analysis_slices(
                    sequence_tokens=sequence.numel(),
                    prompt_tokens=prompt_length,
                    analysis=analysis,
                    convention=convention,
                )
            )
    input_metadata = tuple((tuple(sequence.shape), sequence.dtype) for sequence, _ in paired_inputs) if capture_budget is not None else ()
    model.eval()
    layers = find_transformer_layers(model)
    proj_modules = find_output_projections(model)

    stats = TrajectoryActivations(position_convention=convention)
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
            if convention is not None:
                analysis = preflight_analyses[input_index]
                clean_slice, loop_slice = preflight_slices[input_index]
            else:
                gen_ids = seq[p_len:].tolist()
                analysis = detect_loop_onset(
                    gen_ids, min_ngram=min_ngram, min_repeats=min_repeats
                )
                if analysis.is_looping and analysis.onset_index is not None:
                    abs_onset = p_len + analysis.onset_index
                    clean_slice = slice(max(0, p_len - 1), max(p_len, abs_onset))
                    loop_slice = slice(abs_onset, len(seq))
                else:
                    clean_slice = slice(max(0, p_len - 1), len(seq))
                    loop_slice = None
            analyses.append(analysis)

            current_layer_outputs.clear()
            current_proj_inputs.clear()

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


@torch.no_grad()
def collect_recorded_contrastive_activations(
    model: nn.Module,
    records: Sequence[GenerationRecord],
    *,
    convention: Literal["observed_token", "predictive_next_token"],
    target_layers: Optional[Sequence[int]] = None,
    target_projections: Optional[Sequence[str]] = None,
    prompt_balanced: bool = True,
    capture_budget: ProbeCaptureBudget | None = None,
) -> TrajectoryActivations:
    """Reduce declared recorded spans; optional work bounds do not authenticate a run.

    Consumes persisted `GenerationRecord.health.repetition.spans` directly rather than
    recomputing repetition labels, maps spans under an explicit position convention,
    and hooks only requested layers and output projections. Prompt balancing gives
    each record unit weight, not each problem or an independently validated sample set.
    """
    if type(convention) is not str or convention not in ("observed_token", "predictive_next_token"):
        raise ValueError("Recorded probe requires an explicit supported position convention")
    if type(prompt_balanced) is not bool:
        raise ValueError("Recorded probe prompt_balanced flag must be boolean")
    if capture_budget is not None and type(capture_budget) is not ProbeCaptureBudget:
        raise ValueError("Probe capture budget requires a frozen ProbeCaptureBudget")
    if type(records) not in (list, tuple) or not records or any(not isinstance(r, GenerationRecord) for r in records):
        raise ValueError("Recorded probe requires a nonempty sequence of GenerationRecord instances")
    records = tuple(records)
    if capture_budget is not None and len(records) > capture_budget.max_sequences:
        raise ValueError("Probe capture budget sequence limit exceeded")
    if target_layers is not None:
        if (
            type(target_layers) not in (list, tuple)
            or not target_layers
            or any(type(idx) is not int or idx < 0 for idx in target_layers)
            or len(set(target_layers)) != len(target_layers)
        ):
            raise ValueError("Recorded probe target_layers must be nonempty unique nonnegative integers")
    if target_projections is not None:
        if (
            type(target_projections) not in (list, tuple)
            or not target_projections
            or any(type(name) is not str or not name for name in target_projections)
            or len(set(target_projections)) != len(target_projections)
        ):
            raise ValueError("Recorded probe target_projections must be nonempty unique module names")

    prepared_items: List[Tuple[Tuple[int, ...], slice, Tuple[slice, ...]]] = []
    total_tokens = 0
    declaration = (records[0].experiment_id, records[0].candidate_id, records[0].profile_id)
    repetition_spec = None
    seen_keys = set()
    for record in records:
        if (record.experiment_id, record.candidate_id, record.profile_id) != declaration:
            raise ValueError("Recorded probe batch identity must match one experiment/candidate/profile")
        if record.key in seen_keys:
            raise ValueError("Recorded probe refuses duplicate sample identity")
        seen_keys.add(record.key)
        if record.finish_reason in ("cancelled", "error"):
            raise ValueError("Recorded probe refuses cancelled or errored generation records")
        if record.prompt_token_ids is None or not record.generated_token_ids:
            raise ValueError("Recorded probe requires prompt_token_ids and nonempty generated_token_ids")
        if (
            record.health is None
            or record.health.repetition is None
            or record.health.repetition.status != "complete"
            or record.health.repetition.token_count != len(record.generated_token_ids)
        ):
            raise ValueError("Recorded probe requires complete attached repetition observations")
        if repetition_spec is not None and repetition_spec != record.health.repetition.spec_sha256:
            raise ValueError("Recorded probe repetition specification identity must match")
        repetition_spec = record.health.repetition.spec_sha256
        seq_len = record.prompt_tokens + len(record.generated_token_ids)
        total_tokens += seq_len
        if capture_budget is not None and (
            seq_len > capture_budget.max_sequence_tokens or total_tokens > capture_budget.max_total_tokens
        ):
            raise ValueError("Probe capture budget token limit exceeded")
        gen_len = len(record.generated_token_ids)
        spans = record.health.repetition.spans
        if spans:
            first_span = spans[0]
            if first_span.start > 0:
                c_start, c_end = generated_span_activation_bounds(
                    sequence_tokens=seq_len,
                    prompt_tokens=record.prompt_tokens,
                    generated_start=0,
                    generated_end=first_span.start,
                    convention=convention,
                )
                clean_slice = slice(c_start, c_end)
            else:
                clean_slice = slice(0, 0)
            loop_slices = tuple(
                slice(*generated_span_activation_bounds(
                    sequence_tokens=seq_len,
                    prompt_tokens=record.prompt_tokens,
                    generated_start=span.start,
                    generated_end=span.end,
                    convention=convention,
                )) for span in spans
            )
        else:
            c_start, c_end = generated_span_activation_bounds(
                sequence_tokens=seq_len,
                prompt_tokens=record.prompt_tokens,
                generated_start=0,
                generated_end=gen_len,
                convention=convention,
            )
            clean_slice = slice(c_start, c_end)
            loop_slices = ()
        prepared_items.append((record.prompt_token_ids + record.generated_token_ids, clean_slice, loop_slices))

    all_layers = find_transformer_layers(model)
    all_projs = find_output_projections(model)
    if target_layers is not None:
        if any(idx >= len(all_layers) for idx in target_layers):
            raise ValueError("Recorded probe target_layers index is out of range")
        selected_layers = [(idx, all_layers[idx]) for idx in target_layers]
    else:
        selected_layers = list(enumerate(all_layers))
    if target_projections is not None:
        missing = [name for name in target_projections if name not in all_projs]
        if missing:
            raise ValueError(f"Recorded probe target_projections not found: {missing[0]}")
        selected_projs = [(name, all_projs[name][1]) for name in target_projections]
    else:
        selected_projs = [(name, mod) for name, (_idx, mod) in all_projs.items()]

    model.eval()
    stats = TrajectoryActivations(position_convention=convention)
    current_layer_outputs: Dict[int, Dict[str, Tuple[torch.Tensor, int]]] = {}
    current_proj_inputs: Dict[str, Dict[str, Tuple[torch.Tensor, int]]] = {}
    hooks = []
    active_clean = slice(0, 0)
    active_loops: Tuple[slice, ...] = ()
    expected_sequence_tokens = 0
    summary_elements = 0
    reduction_elements = 0

    def summarize_spans(hidden: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, int]]:
        nonlocal summary_elements, reduction_elements
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 3
            or hidden.shape[0] != 1
            or hidden.shape[1] != expected_sequence_tokens
            or hidden.shape[2] == 0
        ):
            raise ValueError("Probe capture budget requires full single-sequence activation shapes")
        summary: Dict[str, Tuple[torch.Tensor, int]] = {}
        values = hidden.detach()[0]
        for label, spans in (("clean", (active_clean,)), ("loop", active_loops)):
            spans = tuple(span for span in spans if span.stop > span.start)
            if spans:
                count = sum(span.stop - span.start for span in spans)
                if capture_budget is not None:
                    summary_elements += values.shape[1]
                    reduction_elements += count * values.shape[1]
                    if (
                        summary_elements > capture_budget.max_summary_elements
                        or reduction_elements > capture_budget.max_reduction_elements
                    ):
                        raise ValueError("Probe capture budget cumulative hook-work limit exceeded")
                reduced = values[spans[0]].float().sum(dim=0)
                for span in spans[1:]:
                    reduced = reduced + values[span].float().sum(dim=0)
                summary[label] = (reduced.cpu(), count)
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
        for idx, layer in selected_layers:
            hooks.append(layer.register_forward_hook(make_layer_hook(idx)))
        for name, mod in selected_projs:
            hooks.append(mod.register_forward_hook(make_proj_hook(name)))

        device = next(model.parameters()).device
        for token_ids, clean_slice, loop_slices in prepared_items:
            current_layer_outputs.clear()
            current_proj_inputs.clear()
            active_clean = clean_slice
            active_loops = loop_slices
            expected_sequence_tokens = len(token_ids)
            seq_tensor = torch.tensor(token_ids, dtype=torch.int64)
            model(seq_tensor.unsqueeze(0).to(device))
            if loop_slices:
                stats.num_looping_prompts += 1
            else:
                stats.num_clean_prompts += 1

            for idx, summary in current_layer_outputs.items():
                if "clean" in summary:
                    c_sum, c_count = summary["clean"]
                    vec = (c_sum / float(c_count)) if prompt_balanced else c_sum
                    weight = 1 if prompt_balanced else c_count
                    stats.residual_clean_sum[idx] = stats.residual_clean_sum.get(idx, torch.zeros_like(vec)) + vec
                    stats.residual_clean_count[idx] = stats.residual_clean_count.get(idx, 0) + weight
                if "loop" in summary:
                    l_sum, l_count = summary["loop"]
                    vec = (l_sum / float(l_count)) if prompt_balanced else l_sum
                    weight = 1 if prompt_balanced else l_count
                    stats.residual_loop_sum[idx] = stats.residual_loop_sum.get(idx, torch.zeros_like(vec)) + vec
                    stats.residual_loop_count[idx] = stats.residual_loop_count.get(idx, 0) + weight

            for name, summary in current_proj_inputs.items():
                if "clean" in summary:
                    c_sum, c_count = summary["clean"]
                    vec = (c_sum / float(c_count)) if prompt_balanced else c_sum
                    weight = 1 if prompt_balanced else c_count
                    stats.module_input_clean_sum[name] = (
                        stats.module_input_clean_sum.get(name, torch.zeros_like(vec)) + vec
                    )
                    stats.module_input_clean_count[name] = stats.module_input_clean_count.get(name, 0) + weight
                if "loop" in summary:
                    l_sum, l_count = summary["loop"]
                    vec = (l_sum / float(l_count)) if prompt_balanced else l_sum
                    weight = 1 if prompt_balanced else l_count
                    stats.module_input_loop_sum[name] = (
                        stats.module_input_loop_sum.get(name, torch.zeros_like(vec)) + vec
                    )
                    stats.module_input_loop_count[name] = stats.module_input_loop_count.get(name, 0) + weight
    finally:
        for h in hooks:
            h.remove()

    return stats
