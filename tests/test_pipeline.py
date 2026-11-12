"""End-to-end unit tests for loop detection, over-projection patching, and provenance manifests."""

import hashlib
from dataclasses import FrozenInstanceError, replace
import struct
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import llm_loop_fix.evaluation as evaluation_module
import llm_loop_fix.patch as patch_module
import llm_loop_fix.verify as verify_module

from llm_loop_fix import BaselineTensorIdentity, ModelBaseline, PatchBudget, PatchProposal, PatchTarget, apply_patch_proposal, capture_model_baseline, freeze_patch_proposal, patch_trial
from llm_loop_fix.cli import main as cli_main
from llm_loop_fix.detector import RepetitionSpec, detect_loop_onset, diagnose_generation_batch, observe_generation_health
from llm_loop_fix.evaluation import InstructionRegistry
from llm_loop_fix.localize import localize_loop_culprits, CulpritMatrix, LocalizationReport
from llm_loop_fix.manifest import (
    build_edit_manifest,
    load_and_validate_manifest,
    write_edit_manifest,
)
from llm_loop_fix.patch import PatchConfig, RestorationError, apply_surgical_patch, restore_weights, _tensor_sha256, _bitwise_changed_entries
from llm_loop_fix.probe import ProbeCaptureBudget, TrajectoryActivations, collect_contrastive_activations, find_output_projections, find_transformer_layers, generated_span_activation_bounds
from llm_loop_fix.verify import run_alpha_sweep
from llm_loop_fix.verify import ProposalEvaluationIdentity, collect_proposal_record_pairs, evaluate_proposal_record_pairs
from llm_loop_fix.records import DatasetManifest, EvaluationItem, EvaluationProfile, EvaluationRunIdentity, GenerationRecord, InstructionSpec, SampleKey, SOURCE_MODEL_REPO, SOURCE_MODEL_REVISION, SOURCE_CHAT_TEMPLATE_SHA256


@pytest.mark.parametrize(
    "dtype,integer_dtype,format_code,negative_zero,nan_bits,one_bits",
    [(torch.bfloat16, torch.int16, "h", -32768, 0x7FC1, 0x3F80),
     (torch.float16, torch.int16, "h", -32768, 0x7E01, 0x3C00),
     (torch.float32, torch.int32, "i", -2147483648, 0x7FC00001, 0x3F800000)],
)
def test_raw_dtype_hash_and_bit_counts(dtype, integer_dtype, format_code, negative_zero, nan_bits, one_bits):
    bits = [0, negative_zero, nan_bits, one_bits]
    original = torch.tensor(bits, dtype=integer_dtype).view(dtype).reshape(2, 2)
    expected = hashlib.sha256(struct.pack("<" + format_code * 4, *bits)).hexdigest()
    assert _tensor_sha256(original) == expected
    assert _bitwise_changed_entries(original, original.clone()) == 0
    edited = original.clone()
    edited.view(integer_dtype)[0, 0] = negative_zero
    assert _bitwise_changed_entries(original, edited) == 1
    edited.view(integer_dtype)[1, 1] = one_bits + 1
    assert _bitwise_changed_entries(original, edited) == 2
    transposed_bits = [bits[0], bits[2], bits[1], bits[3]]
    assert _tensor_sha256(original.t()) == hashlib.sha256(struct.pack("<" + format_code * 4, *transposed_bits)).hexdigest()
    assert _tensor_sha256(original[1, 1]) == hashlib.sha256(struct.pack("<" + format_code, one_bits)).hexdigest()
    empty = torch.empty((0, 2), dtype=dtype)
    assert _tensor_sha256(empty) == hashlib.sha256(b"").hexdigest()
    assert _bitwise_changed_entries(empty, empty.clone()) == 0


@pytest.mark.parametrize("changed", [torch.ones((3, 2)), torch.ones((2, 2), dtype=torch.float16)])
def test_raw_bit_comparison_refuses_different_tensor_identity(changed):
    with pytest.raises(ValueError, match="shape, dtype and device"):
        _bitwise_changed_entries(torch.ones((2, 2)), changed)


class ToyMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.up_proj(x)))


class ToyDecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.mlp = ToyMLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class ToyLoopingCausalLM(nn.Module):
    """Synthetic model with a one-sided MLP feature that induces a 3-token loop."""

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
                next_tok = 10 if seq[-1] == 6 else min(15, seq[-1] + 1)
            seq.append(next_tok)
            if next_tok == 15:
                break
        return seq


@pytest.fixture
def tiny_qwen35_model():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    config = Qwen3_5Config(
        text_config={"vocab_size": 64, "hidden_size": 16, "intermediate_size": 32,
                     "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
                     "head_dim": 8, "max_position_embeddings": 64, "pad_token_id": 0,
                     "bos_token_id": 1, "eos_token_id": 2, "use_cache": False,
                     "linear_key_head_dim": 4, "linear_value_head_dim": 4,
                     "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                     "linear_conv_kernel_dim": 2, "layer_types": ["linear_attention", "full_attention"],
                     "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                         "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                       "out_hidden_size": 16, "num_position_embeddings": 16,
                       "patch_size": 2, "temporal_patch_size": 1, "spatial_merge_size": 2},
        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(42)
        model = Qwen3_5ForConditionalGeneration(config).to(dtype=torch.float32).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) <= 65536
    return model


def test_tiny_qwen35_text_runtime_names_shapes_and_finite_forward(tiny_qwen35_model):
    model = tiny_qwen35_model
    layers = find_transformer_layers(model)
    assert tuple(layers) == tuple(model.model.language_model.layers)
    projections = find_output_projections(model)
    expected = {"model.language_model.layers.0.linear_attn.out_proj": (0, (16, 8)),
                "model.language_model.layers.0.mlp.down_proj": (0, (16, 32)),
                "model.language_model.layers.1.self_attn.o_proj": (1, (16, 16)),
                "model.language_model.layers.1.mlp.down_proj": (1, (16, 32))}
    assert set(projections) == set(expected)
    for name, (index, module) in projections.items():
        assert (index, tuple(module.weight.shape)) == expected[name]
        assert model.get_submodule(name) is module
        assert module.weight.device.type == "cpu" and module.weight.dtype == torch.float32
    input_ids = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=False)
    assert output.logits.shape == (1, 8, 64) and torch.isfinite(output.logits).all()


def test_tiny_qwen35_cached_prefix_matches_teacher_forced_final_logits(tiny_qwen35_model):
    model = tiny_qwen35_model
    input_ids = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    with torch.inference_mode():
        complete = model(input_ids=input_ids, use_cache=False)
        prefix = model(input_ids=input_ids[:, :-1], use_cache=True)
        assert prefix.past_key_values is not None and prefix.past_key_values.get_seq_length() == 7
        continuation = model(input_ids=input_ids[:, -1:], use_cache=True, past_key_values=prefix.past_key_values)
    assert continuation.logits.shape == (1, 1, 64) and torch.isfinite(continuation.logits).all()
    assert continuation.past_key_values.get_seq_length() == 8
    torch.testing.assert_close(continuation.logits[:, -1], complete.logits[:, -1], atol=1e-6, rtol=1e-5)
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before


@pytest.mark.parametrize("prompt,start,end,convention,expected", [
    (3, 0, 2, "observed_token", (3, 5)), (3, 0, 2, "predictive_next_token", (2, 4)),
    (0, 0, 2, "observed_token", (0, 2)), (0, 1, 3, "predictive_next_token", (0, 2)),
    (3, 4, 5, "observed_token", (7, 8)), (3, 4, 5, "predictive_next_token", (6, 7)),
])
def test_generated_span_activation_bounds_maps_explicit_offsets(prompt, start, end, convention, expected):
    result = generated_span_activation_bounds(sequence_tokens=8, prompt_tokens=prompt, generated_start=start,
                                              generated_end=end, convention=convention)
    assert result == expected and isinstance(result, tuple) and result[1] - result[0] == end - start


@pytest.mark.parametrize("changes", [
    {"sequence_tokens": True}, {"prompt_tokens": 1.0}, {"generated_start": False}, {"generated_end": "2"},
    {"sequence_tokens": 0}, {"prompt_tokens": -1}, {"prompt_tokens": 8}, {"generated_start": -1},
    {"generated_start": 2, "generated_end": 2}, {"generated_start": 3, "generated_end": 2},
    {"generated_end": 6}, {"convention": "legacy"}, {"convention": []},
    {"prompt_tokens": 0, "convention": "predictive_next_token"},
])
def test_generated_span_activation_bounds_refuses_invalid_or_unpredictable_span(changes):
    arguments = dict(sequence_tokens=8, prompt_tokens=3, generated_start=0, generated_end=2, convention="observed_token")
    arguments.update(changes)
    with pytest.raises(ValueError):
        generated_span_activation_bounds(**arguments)


@pytest.mark.parametrize("generated_offset", [0, 1])
def test_tiny_qwen35_predictive_observed_positions_match_cached_prefix(tiny_qwen35_model, generated_offset):
    from transformers.utils.output_capturing import _active_collector, maybe_install_capturing_hooks

    model = tiny_qwen35_model
    assert sum(parameter.numel() for parameter in model.parameters()) == 15176
    maybe_install_capturing_hooks(model)
    before = capture_model_baseline(model)
    collector_before = _active_collector.get()
    hooks_before = {name: (dict(module._forward_hooks), dict(module._forward_pre_hooks)) for name, module in model.named_modules()}
    input_ids = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9]])
    prompt_tokens = 3
    target_position = prompt_tokens + generated_offset
    arguments = dict(sequence_tokens=8, prompt_tokens=prompt_tokens, generated_start=generated_offset, generated_end=generated_offset + 1)
    predictive_bounds = generated_span_activation_bounds(**arguments, convention="predictive_next_token")
    observed_bounds = generated_span_activation_bounds(**arguments, convention="observed_token")
    assert predictive_bounds == (target_position - 1, target_position) and observed_bounds == (target_position, target_position + 1)
    snapshots = {phase: {} for phase in ("full", "predictive", "observed")}
    phase = "full"
    firings, handles = [], []

    def snapshot(key, use_input=False):
        def capture(module, inputs, output):
            value = inputs[0] if use_input else output
            assert isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[0] == 1
            snapshots[phase][key] = value.detach().clone()
            firings.append((phase, key))
        return capture

    try:
        for index, layer in enumerate(find_transformer_layers(model)):
            handles.append(layer.register_forward_hook(snapshot(("residual", index))))
        for name, (_, projection) in find_output_projections(model).items():
            handles.append(projection.register_forward_hook(snapshot(("projection_input", name), True)))
        with torch.inference_mode():
            complete = model(input_ids=input_ids, use_cache=False)
            phase = "predictive"
            prefix = model(input_ids=input_ids[:, :target_position], use_cache=True)
            assert prefix.past_key_values.get_seq_length() == target_position
            phase = "observed"
            consumed = model(input_ids=input_ids[:, target_position:target_position + 1], use_cache=True,
                             past_key_values=prefix.past_key_values)
            assert consumed.past_key_values.get_seq_length() == target_position + 1
    finally:
        for handle in handles:
            handle.remove()
    assert len(firings) == 18
    assert {name: (dict(module._forward_hooks), dict(module._forward_pre_hooks)) for name, module in model.named_modules()} == hooks_before
    assert _active_collector.get() is collector_before
    assert capture_model_baseline(model) == before
    assert len(snapshots["full"]) == 6 and snapshots["full"].keys() == snapshots["predictive"].keys() == snapshots["observed"].keys()
    for key, complete_values in snapshots["full"].items():
        predictive, observed = snapshots["predictive"][key], snapshots["observed"][key]
        assert complete_values.shape[1] == 8 and predictive.shape[1] == target_position and observed.shape[1] == 1
        torch.testing.assert_close(predictive[0, -1], complete_values[0, predictive_bounds[0]], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(observed[0, 0], complete_values[0, observed_bounds[0]], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(prefix.logits[:, -1], complete.logits[:, predictive_bounds[0]], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(consumed.logits[:, -1], complete.logits[:, observed_bounds[0]], atol=1e-6, rtol=1e-5)
    assert not torch.equal(complete.logits[:, target_position - 1], complete.logits[:, target_position])


@pytest.mark.parametrize("interrupt", [False, True])
def test_tiny_qwen35_hidden_state_hook_boundaries_and_cleanup(tiny_qwen35_model, interrupt):
    from transformers.utils.output_capturing import _active_collector, maybe_install_capturing_hooks

    model = tiny_qwen35_model
    text = model.model.language_model
    input_ids = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    maybe_install_capturing_hooks(model)
    collector_before = _active_collector.get()
    hooks_before = {name: (dict(module._forward_hooks), dict(module._forward_pre_hooks)) for name, module in model.named_modules()}
    captured = {}
    handles = []
    output = None

    def boundary_hook(name):
        def capture(module, inputs, result):
            assert isinstance(inputs[0], torch.Tensor) and isinstance(result, torch.Tensor)
            captured[name] = (inputs[0].detach().clone(), result.detach().clone())
            if interrupt and name == "block1":
                raise RuntimeError("controlled hidden-state hook interruption")
        return capture

    try:
        for name, module in (("block0", text.layers[0]), ("block1", text.layers[1]), ("norm", text.norm)):
            handles.append(module.register_forward_hook(boundary_hook(name)))
        with torch.inference_mode():
            if interrupt:
                with pytest.raises(RuntimeError, match="controlled hidden-state hook interruption"):
                    model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
            else:
                output = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()
    assert {name: (dict(module._forward_hooks), dict(module._forward_pre_hooks)) for name, module in model.named_modules()} == hooks_before
    assert _active_collector.get() is collector_before
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before
    if interrupt:
        assert set(captured) == {"block0", "block1"}
        return
    assert output is not None and len(output.hidden_states) == 3
    assert set(captured) == {"block0", "block1", "norm"}
    for pair in captured.values():
        assert all(tensor.shape == (1, 8, 16) and torch.isfinite(tensor).all() for tensor in pair)
    expected_embeddings = text.embed_tokens.weight.index_select(0, input_ids[0]).unsqueeze(0)
    torch.testing.assert_close(captured["block0"][0], expected_embeddings, rtol=0, atol=0)
    torch.testing.assert_close(captured["block1"][0], captured["block0"][1], rtol=0, atol=0)
    torch.testing.assert_close(captured["norm"][0], captured["block1"][1], rtol=0, atol=0)
    torch.testing.assert_close(output.hidden_states[0], captured["block0"][0], rtol=0, atol=0)
    torch.testing.assert_close(output.hidden_states[1], captured["block0"][1], rtol=0, atol=0)
    torch.testing.assert_close(output.hidden_states[2], captured["norm"][1], rtol=0, atol=0)
    assert not torch.equal(captured["norm"][0], captured["norm"][1])


@pytest.mark.parametrize("summary_limit", [120, 72])
def test_tiny_qwen35_bounded_collector_matches_raw_reference_or_cleans_up(monkeypatch, tiny_qwen35_model, summary_limit):
    from transformers.utils.output_capturing import _active_collector, maybe_install_capturing_hooks

    model = tiny_qwen35_model
    assert sum(parameter.numel() for parameter in model.parameters()) == 15176
    assert model.config.text_config.use_cache is False
    maybe_install_capturing_hooks(model)
    before = capture_model_baseline(model)
    collector_before = _active_collector.get()
    hooks_before = {name: (dict(module._forward_hooks), dict(module._forward_pre_hooks)) for name, module in model.named_modules()}
    sequence = torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])
    layers, projections = find_transformer_layers(model), find_output_projections(model)
    layer_values, projection_values = {}, {}
    handles = []

    def snapshot(destination, key, use_input=False):
        def capture(module, inputs, output):
            value = inputs[0] if use_input else output
            assert isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[:2] == (1, 8)
            destination[key] = value.detach().clone()
        return capture

    try:
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(snapshot(layer_values, index)))
        for name, (_, projection) in projections.items():
            handles.append(projection.register_forward_hook(snapshot(projection_values, name, True)))
        with torch.inference_mode():
            model(input_ids=sequence.unsqueeze(0), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    assert set(layer_values) == {0, 1} and set(projection_values) == set(projections)
    assert sum(value.shape[2] for value in (*layer_values.values(), *projection_values.values())) == 120
    transfers = []
    original_cpu = torch.Tensor.cpu

    def vector_cpu(tensor, *args, **kwargs):
        assert tensor.ndim == 1 and tensor.dtype == torch.float32
        transfers.append(tensor.numel())
        return original_cpu(tensor, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "cpu", vector_cpu)
        budget = ProbeCaptureBudget(1, 8, 8, summary_limit, 960)
        if summary_limit < 120:
            with pytest.raises(ValueError, match="Probe capture budget cumulative"):
                collect_contrastive_activations(model, [sequence], [1], capture_budget=budget)
        else:
            stats, analyses = collect_contrastive_activations(model, [sequence], [1], capture_budget=budget)
    assert transfers == ([8, 32, 16, 16] if summary_limit < 120 else [8, 32, 16, 16, 32, 16])
    assert {name: (dict(module._forward_hooks), dict(module._forward_pre_hooks)) for name, module in model.named_modules()} == hooks_before
    assert _active_collector.get() is collector_before
    assert capture_model_baseline(model) == before
    if summary_limit < 120:
        return
    assert len(analyses) == 1 and not analyses[0].is_looping
    assert stats.num_clean_prompts == 1 and stats.num_looping_prompts == 0
    assert stats.residual_clean_count == {index: 8 for index in layer_values}
    assert stats.module_input_clean_count == {name: 8 for name in projection_values}
    for kind, values in (("residual", layer_values), ("module_input", projection_values)):
        assert getattr(stats, kind + "_clean_sum").keys() == values.keys()
        assert not getattr(stats, kind + "_loop_sum") and not getattr(stats, kind + "_loop_count")
        for key, value in values.items():
            assert torch.equal(getattr(stats, kind + "_clean_sum")[key], value[0].sum(dim=0))


@pytest.mark.parametrize("wrapped", [True, False])
def test_nested_text_discovery_does_not_choose_first_vision_module_list(wrapped):
    model = nn.Module()
    model.model = nn.Module()
    model.model.visual = nn.Module()
    model.model.visual.blocks = nn.ModuleList([ToyDecoderLayer(4, 8)])
    model.model.language_model = nn.Module()
    model.model.language_model.layers = nn.ModuleList([ToyDecoderLayer(4, 8), ToyDecoderLayer(4, 8)])
    text_layers = model.model.language_model.layers
    if not wrapped:
        model = model.model
    assert find_transformer_layers(model) == list(text_layers)
    projections = find_output_projections(model)
    prefix = "model." if wrapped else ""
    assert tuple(projections) == (prefix + "language_model.layers.0.mlp.down_proj", prefix + "language_model.layers.1.mlp.down_proj")
    assert tuple(index for index, _ in projections.values()) == (0, 1)
    assert all(module is layer.mlp.down_proj for (_, module), layer in zip(projections.values(), text_layers))


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_text"])
@pytest.mark.parametrize("invalid_stack", ["missing", "empty", "unregistered", "wrong_path", "none_layer", "late_none_layer"])
def test_declared_qwen_text_discovery_refuses_incompatible_stack(model_type, invalid_stack):
    model = nn.Module()
    model.config = SimpleNamespace(model_type=model_type)
    model.visual = nn.Module()
    model.visual.blocks = nn.ModuleList([ToyDecoderLayer(4, 8)])
    owner = model
    name = "layers"
    if model_type == "qwen3_5":
        model.language_model = nn.Module()
        owner = model.language_model
    if invalid_stack == "empty":
        setattr(owner, name, nn.ModuleList())
    elif invalid_stack == "unregistered":
        setattr(owner, name, [ToyDecoderLayer(4, 8)])
    elif invalid_stack == "none_layer":
        setattr(owner, name, nn.ModuleList([None]))
    elif invalid_stack == "late_none_layer":
        setattr(owner, name, nn.ModuleList([ToyDecoderLayer(4, 8), None]))
    elif invalid_stack == "wrong_path":
        model.transformer = nn.Module()
        model.transformer.h = nn.ModuleList([ToyDecoderLayer(4, 8)])
    with pytest.raises(ValueError, match="Declared Qwen3.5 text stack"):
        find_transformer_layers(model)
    with pytest.raises(ValueError, match="Declared Qwen3.5 text stack"):
        find_output_projections(model)


@pytest.mark.parametrize("model_type,wrapped", [
    ("qwen3_5", True), ("qwen3_5", False),
    ("qwen3_5_text", True), ("qwen3_5_text", False),
])
def test_declared_qwen_text_discovery_keeps_canonical_stack(model_type, wrapped):
    model = nn.Module()
    model.config = SimpleNamespace(model_type=model_type)
    model.visual = nn.Module()
    model.visual.blocks = nn.ModuleList([ToyDecoderLayer(4, 8)])
    owner = model
    if wrapped:
        model.model = nn.Module()
        owner = model.model
    if model_type == "qwen3_5":
        owner.language_model = nn.Module()
        owner = owner.language_model
    owner.layers = nn.ModuleList([ToyDecoderLayer(4, 8), ToyDecoderLayer(4, 8)])
    assert find_transformer_layers(model) == list(owner.layers)
    prefix = ("model." if wrapped else "") + ("language_model." if model_type == "qwen3_5" else "")
    projections = find_output_projections(model)
    assert tuple(projections) == (prefix + "layers.0.mlp.down_proj", prefix + "layers.1.mlp.down_proj")
    assert tuple(index for index, _ in projections.values()) == (0, 1)
    assert all(module is layer.mlp.down_proj for (_, module), layer in zip(projections.values(), owner.layers))


@pytest.fixture
def proposal_evaluation_case(numerical_patch_case):
    model, report = numerical_patch_case
    baseline_identity = capture_model_baseline(model)
    budget = PatchBudget(2.0, 1, 1, 2, 2, 1.0, 1.0)
    proposal = freeze_patch_proposal(report, PatchConfig(alpha=0.5, top_k_matrices=1, top_k_columns=1,
                                                       budget=budget, baseline=baseline_identity))
    profile = EvaluationProfile("validated_repair", "math500", SOURCE_MODEL_REPO, SOURCE_MODEL_REVISION,
                                SOURCE_MODEL_REPO, SOURCE_MODEL_REVISION, "a" * 64, "offline-fixture", "fixture-v1",
                                "math-verify", "0.9.0", "b" * 64, (2,), samples_per_item=2)
    item = EvaluationItem("a", "fixture", "v1", "development", "question", "4")
    run_identity = EvaluationRunIdentity(DatasetManifest((item,)), profile, "e" * 40)
    identity = ProposalEvaluationIdentity(run_identity, proposal)
    baseline = tuple(GenerationRecord(run_identity.sha256, identity.baseline_candidate_id, profile.sha256, SampleKey("a", index, 0),
                                      2, (3, 2), r"</think>\boxed{4}", "stop", thinking_open_from_prompt=True,
                                      prompt_token_ids=(1, 4)) for index in range(2))
    candidate = tuple(replace(record, candidate_id=identity.proposal_candidate_id) for record in baseline)
    return model, baseline_identity, identity, baseline, candidate


@pytest.fixture
def captured_proposal_run_case(proposal_evaluation_case):
    model, before, identity, baseline, _ = proposal_evaluation_case
    first = identity.run_identity.dataset.items[0]
    second = replace(first, item_id="b", prompt="second synthetic question")
    profile = replace(identity.run_identity.profile, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256)
    run = EvaluationRunIdentity(DatasetManifest((second, first)), profile, identity.run_identity.code_revision)
    identity = replace(identity, run_identity=run)
    prompt_ids = {"a": (1, 4), "b": (2, 4)}
    contexts = {"a": True, "b": False}
    baseline = tuple(replace(baseline[0], experiment_id=run.sha256, profile_id=profile.sha256, key=key,
                             prompt_token_ids=prompt_ids[key.item_id], thinking_open_from_prompt=contexts[key.item_id],
                             generated_text=r"</think>\boxed{4}" if key.item_id == "a" else r"<think>work</think>\boxed{4}")
                     for key in run.expected_keys)
    candidate = tuple(replace(record, candidate_id=identity.proposal_candidate_id) for record in baseline)
    template_calls = []

    def apply_template(**kwargs):
        content = kwargs["conversation"][0]["content"]
        item = first if content == evaluation_module.build_math_prompt(first.prompt) else second
        assert content == evaluation_module.build_math_prompt(item.prompt)
        assert kwargs["truncation"] is False and kwargs["enable_thinking"] is True
        template_calls.append(item.item_id)
        return list(prompt_ids[item.item_id])

    captured = evaluation_module.capture_run_prompt_contexts(
        run, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256,
        thinking_open_from_prompt=contexts, apply_chat_template=apply_template,
    )
    return model, before, identity, baseline, candidate, captured, template_calls


@pytest.mark.parametrize("scenario", ["complete", "prompt_drift", "missing_sample", "native_error"])
def test_native_record_graded_trial_join_preserves_snapshots_and_restoration(captured_proposal_run_case, monkeypatch, scenario):
    from types import SimpleNamespace
    from llm_loop_fix.records import record_vllm_generations
    import llm_loop_fix.verify as verify_module

    model, before, identity, _, _, captured, template_calls = captured_proposal_run_case
    profile = replace(identity.run_identity.profile, backend="vllm", backend_version="controlled-api-fixture")
    identity = replace(identity, run_identity=replace(identity.run_identity, profile=profile))
    run = identity.run_identity
    candidate_ids = (identity.baseline_candidate_id, identity.proposal_candidate_id)
    requests = {(candidate_id, item.item_id): candidate_id + ":" + item.item_id for candidate_id in candidate_ids for item in run.dataset.items}
    callbacks, graded_candidates, adapted = [], [], {}
    original_grade_batch = verify_module.evaluate_generation_batch

    class PromptDriftOutput(SimpleNamespace):
        @property
        def outputs(self):
            raise AssertionError("prompt drift must refuse before completion access")

    def grade_batch(run_identity, records, *, candidate_id, registry=None):
        graded_candidates.append(candidate_id)
        assert (capture_model_baseline(model) == before) == (candidate_id == identity.baseline_candidate_id)
        return original_grade_batch(run_identity, records, candidate_id=candidate_id, registry=registry)

    monkeypatch.setattr(verify_module, "evaluate_generation_batch", grade_batch)

    def callback(active_model, candidate_id):
        callbacks.append(candidate_id)
        assert (capture_model_baseline(active_model) == before) == (candidate_id == identity.baseline_candidate_id)
        records = []
        for item in sorted(run.dataset.items, key=lambda item: item.item_id):
            prompt_ids, context = captured[item.item_id]
            request_id = requests[candidate_id, item.item_id]
            text = r"</think>\boxed{4}" if context else r"<think>work</think>\boxed{4}"
            output = SimpleNamespace(request_id=request_id, finished=True, prompt_token_ids=list(prompt_ids),
                                     outputs=[SimpleNamespace(index=index, token_ids=[3, 2], text=text, finish_reason="stop", stop_reason=2)
                                              for index in reversed(range(profile.samples_per_item))])
            if candidate_id == identity.proposal_candidate_id and item.item_id == "b":
                if scenario == "prompt_drift":
                    output = PromptDriftOutput(request_id=request_id, finished=True, prompt_token_ids=[99, prompt_ids[1]])
                elif scenario == "missing_sample":
                    output.outputs.pop()
                elif scenario == "native_error":
                    completion = next(entry for entry in output.outputs if entry.index == 0)
                    completion.finish_reason, completion.stop_reason = "error", None
                    completion.token_ids, completion.text = [], ""
            snapshots = record_vllm_generations(item, output, profile, experiment_id=run.sha256, candidate_id=candidate_id,
                                                expected_request_id=request_id, expected_prompt_token_ids=prompt_ids,
                                                thinking_open_from_prompt=context, backend_version=profile.backend_version, run_identity=run)
            records.extend(snapshots)
            adapted[candidate_id, item.item_id] = snapshots
            output.prompt_token_ids[0] = 99
            output.outputs[0].text = "changed after immutable native snapshot"
            output.outputs[0].token_ids[:] = [99]
        return tuple(reversed(records))

    if scenario in ("prompt_drift", "missing_sample"):
        message = "native prompt token identity mismatch" if scenario == "prompt_drift" else "missing or unexpected native samples"
        with pytest.raises(ValueError, match=message):
            evaluate_proposal_record_pairs(model, identity, callback, expected_prompt_contexts=captured)
        assert graded_candidates == [identity.baseline_candidate_id]
    else:
        pairs = evaluate_proposal_record_pairs(model, identity, callback, expected_prompt_contexts=captured)
        assert tuple(original.key for original, _ in pairs) == run.expected_keys
        assert graded_candidates == list(candidate_ids)
        for pair in pairs:
            for record in pair:
                source = next(entry for entry in adapted[record.candidate_id, record.key.item_id] if entry.key == record.key)
                assert replace(record, quality=None, health=None) == source
                assert source.quality is None and source.health is None
                assert record.health is not None and record.quality is not None
                assert (record.prompt_token_ids, record.thinking_open_from_prompt) == captured[record.key.item_id]
                assert record.backend_request_id == requests[record.candidate_id, record.key.item_id]
                if record.backend_finish_reason == "error":
                    assert record.quality.correctness is None and record.quality.grader_status in ("unavailable", "error")
                    assert record.quality.reason
                    assert not record.generated_token_ids and record.backend_stop_reason is None
                else:
                    assert record.quality.correctness is True and record.backend_stop_reason == 2
        report = diagnose_generation_batch(run, tuple(pair[1] for pair in pairs), candidate_id=identity.proposal_candidate_id,
                                           expected_prompt_contexts=captured)
        assert report.outcome == "insufficient_evidence" and report.localization_authorized is False
    assert callbacks == list(candidate_ids) and template_calls == ["a", "b"]
    assert capture_model_baseline(model) == before


def test_captured_run_trial_join_preserves_complete_two_item_pairs(captured_proposal_run_case):
    model, before, identity, baseline, candidate, captured, template_calls = captured_proposal_run_case
    callbacks = []

    def callback(active_model, candidate_id):
        callbacks.append(candidate_id)
        assert (capture_model_baseline(active_model) == before) == (candidate_id == identity.baseline_candidate_id)
        return tuple(reversed(baseline if candidate_id == identity.baseline_candidate_id else candidate))

    pairs = evaluate_proposal_record_pairs(model, identity, callback, expected_prompt_contexts=captured)
    assert tuple(original.key for original, _ in pairs) == identity.run_identity.expected_keys
    assert len(pairs) == 4 and template_calls == ["a", "b"]
    assert callbacks == [identity.baseline_candidate_id, identity.proposal_candidate_id]
    for pair, sources in zip(pairs, zip(baseline, candidate)):
        for record, source in zip(pair, sources):
            assert record.quality.correctness is True and record.health is not None
            assert replace(record, quality=None, health=None) == source
            assert (record.prompt_token_ids, record.thinking_open_from_prompt) == captured[record.key.item_id]
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("phase", ["baseline", "proposal"])
@pytest.mark.parametrize("changed", ["tokens", "thinking"])
def test_captured_run_trial_join_refuses_late_item_drift_before_phase_grader(captured_proposal_run_case, monkeypatch, phase, changed):
    model, before, identity, baseline, candidate, captured, template_calls = captured_proposal_run_case
    selected = identity.baseline_candidate_id if phase == "baseline" else identity.proposal_candidate_id
    callbacks = []
    graders = []
    original = verify_module.evaluate_generation_batch

    def callback(active_model, candidate_id):
        callbacks.append(candidate_id)
        records = baseline if candidate_id == identity.baseline_candidate_id else candidate
        if candidate_id == selected:
            captured.clear()
            changed_records = []
            for record in records:
                if record.key.item_id == "b":
                    record = replace(record, prompt_token_ids=(2, 5)) if changed == "tokens" else replace(record, thinking_open_from_prompt=True)
                changed_records.append(record)
            return tuple(changed_records)
        return records

    def grade(run, generations, *, candidate_id, registry=None):
        graders.append(candidate_id)
        assert candidate_id != selected, "drifted item reached its phase grader"
        return original(run, generations, candidate_id=candidate_id, registry=registry)

    monkeypatch.setattr(verify_module, "evaluate_generation_batch", grade)
    with pytest.raises(ValueError, match="independent prompt/context"):
        evaluate_proposal_record_pairs(model, identity, callback, expected_prompt_contexts=captured)
    assert callbacks == ([identity.baseline_candidate_id] if phase == "baseline" else [identity.baseline_candidate_id, identity.proposal_candidate_id])
    assert graders == ([] if phase == "baseline" else [identity.baseline_candidate_id])
    assert template_calls == ["a", "b"] and capture_model_baseline(model) == before


def test_captured_run_trial_join_interrupted_capture_never_starts_trial(captured_proposal_run_case):
    model, before, identity, baseline, _, _, _ = captured_proposal_run_case
    template_calls = []
    record_calls = []

    def interrupted_template(**kwargs):
        template_calls.append(True)
        if len(template_calls) == 2:
            raise MemoryError("controlled capture allocation interruption")
        return [1, 4]

    def forbidden_records(active_model, candidate_id):
        record_calls.append(candidate_id)
        raise AssertionError("interrupted complete capture started a trial")

    with pytest.raises(MemoryError):
        captured = evaluation_module.capture_run_prompt_contexts(
            identity.run_identity, chat_template_sha256=SOURCE_CHAT_TEMPLATE_SHA256,
            thinking_open_from_prompt={"a": True, "b": False}, apply_chat_template=interrupted_template,
        )
        evaluate_proposal_record_pairs(model, identity, forbidden_records, expected_prompt_contexts=captured)
    assert len(template_calls) == 2 and not record_calls
    assert all(record.quality is None for record in baseline) and capture_model_baseline(model) == before


def test_captured_run_trial_join_readonly_diagnosis_does_not_promote_quality(captured_proposal_run_case):
    model, before, identity, baseline, candidate, captured, template_calls = captured_proposal_run_case
    pairs = evaluate_proposal_record_pairs(model, identity, lambda active_model, candidate_id: baseline if candidate_id == identity.baseline_candidate_id else candidate,
                                          expected_prompt_contexts=captured)
    for index, candidate_id in enumerate((identity.baseline_candidate_id, identity.proposal_candidate_id)):
        records = tuple(pair[index] for pair in pairs)
        report = diagnose_generation_batch(identity.run_identity, records, candidate_id=candidate_id,
                                           expected_prompt_contexts=captured)
        assert report.outcome == "insufficient_evidence" and report.localization_authorized is False
        assert report.scope == "recorded_evidence_only"
        assert all(record.quality.correctness is True for record in records)
    assert template_calls == ["a", "b"] and capture_model_baseline(model) == before


def test_captured_run_trial_join_noop_keeps_baseline_without_retokenizing(captured_proposal_run_case):
    model, before, identity, baseline, candidate, captured, template_calls = captured_proposal_run_case
    identity = replace(identity, proposal=replace(identity.proposal, alpha=0.0))
    candidate = tuple(replace(record, candidate_id=identity.proposal_candidate_id) for record in candidate)

    def callback(active_model, candidate_id):
        assert capture_model_baseline(active_model) == before
        return baseline if candidate_id == identity.baseline_candidate_id else candidate

    pairs = evaluate_proposal_record_pairs(model, identity, callback, expected_prompt_contexts=captured)
    assert len(pairs) == 4 and all(record.quality.correctness is True for pair in pairs for record in pair)
    assert template_calls == ["a", "b"] and capture_model_baseline(model) == before


@pytest.mark.parametrize("stage,failure", [("layer", RuntimeError), ("projection", RuntimeError), ("device", RuntimeError),
                                          ("layer", MemoryError), ("projection", KeyboardInterrupt)])
def test_legacy_probe_setup_failure_removes_every_installed_hook(monkeypatch, stage, failure):
    model = ToyLoopingCausalLM().eval()
    hook_registry = {module: dict(module._forward_hooks) for module in model.modules()}
    parameters = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}

    def failed_setup(*args, **kwargs):
        raise failure("controlled probe setup failure")

    def forbidden_forward(*args, **kwargs):
        raise AssertionError("failed setup must not forward a model")

    monkeypatch.setattr(model, "forward", forbidden_forward)
    if stage == "layer":
        monkeypatch.setattr(model.layers[1], "register_forward_hook", failed_setup)
    elif stage == "projection":
        monkeypatch.setattr(model.layers[0].mlp.down_proj, "register_forward_hook", failed_setup)
    else:
        monkeypatch.setattr(model, "parameters", failed_setup)
    with pytest.raises(failure, match="controlled probe setup failure"):
        collect_contrastive_activations(model, [torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])], [1])
    assert {module: dict(module._forward_hooks) for module in model.modules()} == hook_registry
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == parameters


@pytest.mark.parametrize("failure", [MemoryError, KeyboardInterrupt])
def test_legacy_probe_interrupted_forward_removes_hooks_without_parameter_changes(monkeypatch, failure):
    model = ToyLoopingCausalLM().eval()
    hook_registry = {module: dict(module._forward_hooks) for module in model.modules()}
    parameters = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}

    def failed_forward(*args, **kwargs):
        raise failure("controlled probe forward interruption")

    monkeypatch.setattr(model.layers[1], "forward", failed_forward)
    with pytest.raises(failure, match="controlled probe forward interruption"):
        collect_contrastive_activations(model, [torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])], [1])
    assert {module: dict(module._forward_hooks) for module in model.modules()} == hook_registry
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == parameters


def test_legacy_probe_success_preserves_raw_block_sums_and_preexisting_hooks():
    model = ToyLoopingCausalLM().eval()
    assert sum(parameter.numel() for parameter in model.parameters()) <= 1024
    parameters = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    observed = {}

    def retain_raw_output(module, inputs, output):
        observed["last_block"] = output.detach().float().cpu().clone()

    retained = model.layers[-1].register_forward_hook(retain_raw_output)
    hook_registry = {module: dict(module._forward_hooks) for module in model.modules()}
    try:
        stats, analyses = collect_contrastive_activations(model, [torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])], [1])
        assert len(analyses) == 1 and stats.num_clean_prompts == 1 and stats.num_looping_prompts == 0
        assert stats.residual_clean_count == {0: 8, 1: 8}
        assert torch.equal(stats.residual_clean_sum[1], observed["last_block"][0].sum(dim=0))
        assert {module: dict(module._forward_hooks) for module in model.modules()} == hook_registry
        assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == parameters
    finally:
        retained.remove()


@pytest.mark.parametrize("changed", ["extra_sequence", "extra_length"])
def test_legacy_probe_input_preflight_refuses_incomplete_pairing(changed):
    model = ToyLoopingCausalLM().eval()
    sequence = torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])
    sequences = [sequence, sequence] if changed == "extra_sequence" else [sequence]
    lengths = [1, 1] if changed == "extra_length" else [1]
    with pytest.raises(ValueError, match="Probe inputs"):
        collect_contrastive_activations(model, sequences, lengths)


@pytest.mark.parametrize("changed", ["empty", "negative_length", "boolean_length", "past_end", "no_generation",
                                     "rank", "empty_sequence", "float_tokens", "lazy_sequences", "lazy_lengths", "not_tensor"])
def test_legacy_probe_input_preflight_refuses_invalid_late_item_before_model_access(monkeypatch, changed):
    model = ToyLoopingCausalLM()
    parameters = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    hook_registry = {module: dict(module._forward_hooks) for module in model.modules()}
    sequence = torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])
    sequences = [sequence, sequence]
    lengths = [1, 1]
    if changed == "empty":
        sequences, lengths = [], []
    elif changed == "negative_length":
        lengths[-1] = -1
    elif changed == "boolean_length":
        lengths[-1] = True
    elif changed == "past_end":
        lengths[-1] = 9
    elif changed == "no_generation":
        lengths[-1] = 8
    elif changed == "rank":
        sequences[-1] = sequence.unsqueeze(0)
    elif changed == "empty_sequence":
        sequences[-1] = torch.empty(0, dtype=torch.long)
    elif changed == "float_tokens":
        sequences[-1] = sequence.float()
    elif changed == "not_tensor":
        sequences[-1] = [1, 2]
    else:
        def forbidden_iteration():
            raise AssertionError("lazy probe batches must not be consumed")
            yield sequence

        if changed == "lazy_sequences":
            sequences = forbidden_iteration()
        else:
            lengths = forbidden_iteration()

    def forbidden_model_access(*args, **kwargs):
        raise AssertionError("invalid probe input reached model mode/setup")

    monkeypatch.setattr(model, "eval", forbidden_model_access)
    with pytest.raises(ValueError, match="Probe inputs"):
        collect_contrastive_activations(model, sequences, lengths)
    assert model.training is True
    assert {module: dict(module._forward_hooks) for module in model.modules()} == hook_registry
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == parameters


def test_legacy_probe_input_preflight_preserves_complete_integer_batches_and_zero_prompt(monkeypatch):
    model = ToyLoopingCausalLM().eval()
    parameters = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    hook_registry = {module: dict(module._forward_hooks) for module in model.modules()}
    sequences = (torch.tensor([1, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32),
                 torch.tensor([1, 3, 4, 5, 6, 7, 8, 10], dtype=torch.int64))
    for container in (list, tuple):
        stats, analyses = collect_contrastive_activations(model, container(sequences), container((0, 1)))
        assert len(analyses) == 2 and stats.num_clean_prompts == 2 and stats.num_looping_prompts == 0
        assert stats.residual_clean_count == {0: 16, 1: 16}
        assert {module: dict(module._forward_hooks) for module in model.modules()} == hook_registry
        assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == parameters

    mutable_sequences = list(sequences)
    mutable_lengths = [0, 1]
    original_forward = model.forward

    def mutate_input_containers(input_ids):
        mutable_sequences.clear()
        mutable_lengths.clear()
        return original_forward(input_ids)

    monkeypatch.setattr(model, "forward", mutate_input_containers)
    stats, analyses = collect_contrastive_activations(model, mutable_sequences, mutable_lengths)
    assert len(analyses) == 2 and stats.num_clean_prompts == 2
    assert stats.residual_clean_count == {0: 16, 1: 16}
    assert not mutable_sequences and not mutable_lengths
    assert {module: dict(module._forward_hooks) for module in model.modules()} == hook_registry
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == parameters


def test_legacy_probe_capture_budget_refuses_invalid_object_before_model_access(monkeypatch):
    model = ToyLoopingCausalLM()

    def forbidden_model_access(*args, **kwargs):
        raise AssertionError("invalid capture budget reached model mode/setup")

    monkeypatch.setattr(model, "eval", forbidden_model_access)
    with pytest.raises(ValueError, match="Probe capture budget"):
        collect_contrastive_activations(model, [torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])], [1], capture_budget=object())
    assert model.training is True


@pytest.mark.parametrize("invalid", [True, -1, 1.0, "1"])
def test_legacy_probe_capture_budget_requires_frozen_plain_integer_limits(invalid):
    budget = ProbeCaptureBudget(2, 16, 32, 128, 2048)
    for field in vars(budget):
        with pytest.raises(ValueError, match="Probe capture budget"):
            replace(budget, **{field: invalid})
    with pytest.raises(AttributeError):
        setattr(budget, "max_summary_elements", 1)
    assert budget == ProbeCaptureBudget(2, 16, 32, 128, 2048)


@pytest.mark.parametrize("field,limit", [("max_sequences", 1), ("max_sequence_tokens", 7), ("max_total_tokens", 15)])
def test_legacy_probe_capture_budget_refuses_complete_input_excess_before_model_access(monkeypatch, field, limit):
    model = ToyLoopingCausalLM()
    parameters = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    registry = {module: dict(module._forward_hooks) for module in model.modules()}
    budget = replace(ProbeCaptureBudget(2, 16, 32, 128, 2048), **{field: limit})

    def forbidden_model_access(*args, **kwargs):
        raise AssertionError("over-budget complete input reached model mode/setup")

    monkeypatch.setattr(model, "eval", forbidden_model_access)
    sequence = torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])
    with pytest.raises(ValueError, match="Probe capture budget"):
        collect_contrastive_activations(model, [sequence, sequence], [1, 1], capture_budget=budget)
    assert model.training is True
    assert {module: dict(module._forward_hooks) for module in model.modules()} == registry
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == parameters


@pytest.mark.parametrize("scenario,limit,allowed", [("summary_single", 7, 0), ("reduction_single", 63, 0),
                                                 ("summary_total", 24, 3), ("reduction_total", 192, 3),
                                                 ("cross_sequence", 32, 4), ("repeated_hook", 32, 4)])
def test_legacy_probe_capture_budget_refuses_cumulative_work_before_conversion_and_transfer(monkeypatch, scenario, limit, allowed):
    model = ToyLoopingCausalLM().eval()
    parameters = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    registry = {module: dict(module._forward_hooks) for module in model.modules()}
    budget = ProbeCaptureBudget(2, 16, 32, 512, 4096)
    field = "max_reduction_elements" if scenario.startswith("reduction") else "max_summary_elements"
    budget = replace(budget, **{field: limit})
    sequence = torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])
    sequences = [sequence, sequence] if scenario == "cross_sequence" else [sequence]
    if scenario == "repeated_hook":
        def repeating_forward(input_ids):
            hidden = model.embed(input_ids)
            for block in (model.layers[0], model.layers[1], model.layers[0]):
                hidden = block(hidden)
            return model.lm_head(hidden)

        monkeypatch.setattr(model, "forward", repeating_forward)
    conversions, transfers = [], []
    original_float, original_cpu = torch.Tensor.float, torch.Tensor.cpu

    def counted_float(tensor, *args, **kwargs):
        conversions.append(tensor.numel())
        return original_float(tensor, *args, **kwargs)

    def vector_cpu(tensor, *args, **kwargs):
        assert tensor.ndim == 1
        transfers.append(tensor.numel())
        return original_cpu(tensor, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "float", counted_float)
        context.setattr(torch.Tensor, "cpu", vector_cpu)
        with pytest.raises(ValueError, match="Probe capture budget cumulative"):
            collect_contrastive_activations(model, sequences, [1] * len(sequences), capture_budget=budget)
    assert conversions == [64] * allowed and transfers == [8] * allowed
    assert {module: dict(module._forward_hooks) for module in model.modules()} == registry
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == parameters


@pytest.mark.parametrize("looping", [False, True])
def test_legacy_probe_capture_budget_exact_bound_preserves_legacy_sums(monkeypatch, looping):
    model = ToyLoopingCausalLM().eval()
    parameters = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    registry = {module: dict(module._forward_hooks) for module in model.modules()}
    sequence = torch.tensor([1] + [7, 8, 9] * 4) if looping else torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])
    budget = ProbeCaptureBudget(1, len(sequence), len(sequence), 64 if looping else 32, 4 * len(sequence) * 8)
    expected, expected_analyses = collect_contrastive_activations(model, [sequence], [1])
    actual, analyses = collect_contrastive_activations(model, [sequence], [1], capture_budget=budget)
    assert analyses == expected_analyses
    for field, value in vars(expected).items():
        observed = getattr(actual, field)
        if isinstance(value, dict):
            assert observed.keys() == value.keys()
            assert all(torch.equal(observed[key], entry) if isinstance(entry, torch.Tensor) else observed[key] == entry for key, entry in value.items())
        else:
            assert observed == value
    assert {module: dict(module._forward_hooks) for module in model.modules()} == registry
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == parameters


def test_legacy_probe_capture_budget_refuses_later_tensor_resize_before_its_forward(monkeypatch):
    model = ToyLoopingCausalLM().eval()
    parameters = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    registry = {module: dict(module._forward_hooks) for module in model.modules()}
    sequences = [torch.tensor([1, 3, 4, 5, 6, 7, 8, 9]), torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])]
    original_forward = model.forward
    forwards = []

    def resizing_forward(input_ids):
        forwards.append(True)
        if len(forwards) == 1:
            sequences[1].resize_(9).fill_(7)
        return original_forward(input_ids)

    monkeypatch.setattr(model, "forward", resizing_forward)
    with pytest.raises(ValueError, match="Probe capture budget input metadata changed"):
        collect_contrastive_activations(model, sequences, [1, 1], capture_budget=ProbeCaptureBudget(2, 8, 16, 128, 2048))
    assert forwards == [True]
    assert {module: dict(module._forward_hooks) for module in model.modules()} == registry
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == parameters


@pytest.mark.parametrize("changed", ["rank", "batch", "length", "channels"])
def test_legacy_probe_capture_budget_refuses_malformed_hook_shapes_and_cleans_up(monkeypatch, changed):
    model = ToyLoopingCausalLM().eval()
    parameters = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    registry = {module: dict(module._forward_hooks) for module in model.modules()}
    original_forward = model.layers[0].forward

    def malformed_forward(hidden):
        output = original_forward(hidden)
        if changed == "rank":
            return output[0]
        if changed == "batch":
            return output.expand(2, -1, -1)
        if changed == "length":
            return output[:, :-1]
        return output[:, :, :0]

    monkeypatch.setattr(model.layers[0], "forward", malformed_forward)
    with pytest.raises(ValueError, match="Probe capture budget requires full"):
        collect_contrastive_activations(model, [torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])], [1],
                                       capture_budget=ProbeCaptureBudget(1, 8, 8, 128, 2048))
    assert {module: dict(module._forward_hooks) for module in model.modules()} == registry
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == parameters


def test_legacy_probe_span_reduction_transfers_only_channel_sums(monkeypatch):
    model = ToyLoopingCausalLM().eval()
    parameters = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    hook_registry = {module: dict(module._forward_hooks) for module in model.modules()}
    original_cpu = torch.Tensor.cpu
    transfers = []

    def vector_only_cpu(tensor, *args, **kwargs):
        assert tensor.ndim == 1, "collector must reduce spans before CPU transfer"
        transfers.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "cpu", vector_only_cpu)
        stats, analyses = collect_contrastive_activations(model, [torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])], [1])
    assert transfers == [(8,)] * 4
    assert len(analyses) == 1 and stats.residual_clean_count == {0: 8, 1: 8}
    assert {module: dict(module._forward_hooks) for module in model.modules()} == hook_registry
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == parameters


@pytest.mark.parametrize("scenario", ["mixed", "loop", "zero_prompt", "repeated_hook"])
@pytest.mark.parametrize("tuple_output", [False, True])
def test_legacy_probe_span_reduction_matches_full_activation_reference(monkeypatch, scenario, tuple_output):
    model = ToyLoopingCausalLM().eval()
    assert sum(parameter.numel() for parameter in model.parameters()) <= 1024
    parameters = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    registry = {module: dict(module._forward_hooks) for module in model.modules()}
    if tuple_output:
        for layer in model.layers:
            original_forward = layer.forward

            def tuple_forward(*args, original=original_forward):
                return (original(*args), None)

            monkeypatch.setattr(layer, "forward", tuple_forward)

    def forward(input_ids):
        hidden = model.embed(input_ids)
        blocks = tuple(model.layers) + ((model.layers[0],) if scenario == "repeated_hook" else ())
        for block in blocks:
            output = block(hidden)
            hidden = output[0] if isinstance(output, tuple) else output
        return model.lm_head(hidden)

    monkeypatch.setattr(model, "forward", forward)
    clean = torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])
    loop = torch.tensor([1] + [7, 8, 9] * 4)
    sequences, lengths = [loop], [1]
    if scenario == "mixed":
        sequences, lengths = [clean, torch.tensor([1, 4, 5] + [7, 8, 9] * 4)], [1, 1]
    elif scenario == "zero_prompt":
        sequences, lengths = [loop[1:]], [0]
    elif scenario == "repeated_hook":
        sequences, lengths = [clean], [1]
    expected = TrajectoryActivations()
    expected_analyses = []
    layer_values, projection_values = {}, {}
    handles = []

    def capture_layer(index):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            layer_values[index] = value.detach().float().cpu().clone()
        return hook

    def capture_projection(name):
        def hook(module, inputs, output):
            projection_values[name] = inputs[0].detach().float().cpu().clone()
        return hook

    try:
        for index, layer in enumerate(model.layers):
            handles.append(layer.register_forward_hook(capture_layer(index)))
        for name, (_, projection) in find_output_projections(model).items():
            handles.append(projection.register_forward_hook(capture_projection(name)))
        for sequence, prompt_length in zip(sequences, lengths):
            layer_values.clear()
            projection_values.clear()
            with torch.no_grad():
                model(sequence.unsqueeze(0))
            analysis = detect_loop_onset(sequence[prompt_length:].tolist())
            expected_analyses.append(analysis)
            if analysis.is_looping and analysis.onset_index is not None:
                expected.num_looping_prompts += 1
                onset = prompt_length + analysis.onset_index
                spans = {"clean": slice(max(0, prompt_length - 1), max(prompt_length, onset)), "loop": slice(onset, len(sequence))}
            else:
                expected.num_clean_prompts += 1
                spans = {"clean": slice(max(0, prompt_length - 1), len(sequence))}
            for kind, values in (("residual", layer_values), ("module_input", projection_values)):
                for label, span in spans.items():
                    if span.stop <= span.start:
                        continue
                    sums = getattr(expected, kind + "_" + label + "_sum")
                    counts = getattr(expected, kind + "_" + label + "_count")
                    for key, value in values.items():
                        selected = value[0, span]
                        summed = selected.sum(dim=0)
                        sums[key] = sums.get(key, torch.zeros_like(summed)) + summed
                        counts[key] = counts.get(key, 0) + len(selected)
    finally:
        for handle in handles:
            handle.remove()
    original_cpu = torch.Tensor.cpu

    def vector_only_cpu(tensor, *args, **kwargs):
        assert tensor.ndim == 1, "collector retained a full activation tensor"
        return original_cpu(tensor, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "cpu", vector_only_cpu)
        actual, analyses = collect_contrastive_activations(model, sequences, lengths)
    assert analyses == expected_analyses
    for field, value in vars(expected).items():
        observed = getattr(actual, field)
        if isinstance(value, dict):
            assert observed.keys() == value.keys()
            assert all(torch.equal(observed[key], entry) if isinstance(entry, torch.Tensor) else observed[key] == entry for key, entry in value.items())
        else:
            assert observed == value
    assert {module: dict(module._forward_hooks) for module in model.modules()} == registry
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == parameters


def test_legacy_probe_span_reduction_interruption_removes_hooks(monkeypatch):
    model = ToyLoopingCausalLM().eval()
    parameters = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    registry = {module: dict(module._forward_hooks) for module in model.modules()}
    original_cpu = torch.Tensor.cpu
    transfers = []

    def interrupted_cpu(tensor, *args, **kwargs):
        assert tensor.ndim == 1
        transfers.append(tuple(tensor.shape))
        if len(transfers) == 2:
            raise MemoryError("controlled span reduction transfer interruption")
        return original_cpu(tensor, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "cpu", interrupted_cpu)
        with pytest.raises(MemoryError, match="controlled span reduction"):
            collect_contrastive_activations(model, [torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])], [1])
    assert len(transfers) == 2
    assert {module: dict(module._forward_hooks) for module in model.modules()} == registry
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == parameters


@pytest.mark.parametrize("original", ["correct", "wrong", "missing", "unavailable", "error"])
@pytest.mark.parametrize("edited", ["correct", "wrong", "missing", "unavailable", "error"])
def test_proposal_quality_summary_retains_independent_outcomes(proposal_evaluation_case, original, edited, monkeypatch):
    from llm_loop_fix.records import QualityObservation

    model, before, identity, baseline, candidate = proposal_evaluation_case

    def quality(outcome):
        if outcome == "missing":
            return None
        if outcome in ("correct", "wrong"):
            return QualityObservation("fixture-extractor", "fixture-grader", "4", "usable", outcome == "correct", "graded")
        return QualityObservation("fixture-extractor", "fixture-grader", None, "unavailable", None, outcome, "controlled outcome")

    def no_grader(*args, **kwargs):
        raise AssertionError("Readonly quality summary must not grade")

    monkeypatch.setattr("llm_loop_fix.verify.evaluate_generation_batch", no_grader)
    baseline = tuple(replace(record, quality=quality(original)) for record in baseline)
    candidate = tuple(replace(record, quality=quality(edited)) for record in candidate)
    saved = baseline, candidate, identity.sha256
    result = identity.summarize_quality(tuple(reversed(baseline)), candidate)
    complete = original in ("correct", "wrong") and edited in ("correct", "wrong")
    expected_transition = ({("correct", "correct"): "both_correct", ("wrong", "correct"): "rescued",
                            ("correct", "wrong"): "regressed", ("wrong", "wrong"): "both_wrong"}.get(
                                (original, edited), "not_comparable"))
    assert result == {
        "scope": "recorded_quality_counts_only", "proposal_evaluation_sha256": identity.sha256, "paired_samples": 2,
        "baseline": {key: 2 if key == original else 0 for key in ("correct", "wrong", "missing", "unavailable", "error")},
        "candidate": {key: 2 if key == edited else 0 for key in ("correct", "wrong", "missing", "unavailable", "error")},
        "transitions": {key: 2 if key == expected_transition else 0 for key in ("both_correct", "rescued", "regressed", "both_wrong", "not_comparable")},
        "quality_complete": complete,
        "net_correct_samples": 2 * (int(edited == "correct") - int(original == "correct")) if complete else None,
    }
    result["baseline"][original] = 100
    assert identity.summarize_quality(baseline, candidate)["baseline"][original] == 2
    assert (baseline, candidate, identity.sha256) == saved
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("changed", ["extractor_id", "grader_id"])
def test_proposal_quality_summary_keeps_different_quality_contracts_noncomparable(proposal_evaluation_case, changed):
    from llm_loop_fix.records import QualityObservation

    model, before, identity, baseline, candidate = proposal_evaluation_case
    quality = QualityObservation("fixture-extractor", "fixture-grader", "4", "usable", True, "graded")
    baseline = tuple(replace(record, quality=quality) for record in baseline)
    candidate = tuple(replace(record, quality=replace(quality, **{changed: "other-contract"})) for record in candidate)
    result = identity.summarize_quality(baseline, candidate)
    assert result["baseline"]["correct"] == result["candidate"]["correct"] == 2
    assert result["transitions"]["not_comparable"] == 2
    assert sum(result["transitions"].values()) == 2
    assert not result["quality_complete"] and result["net_correct_samples"] is None
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("changed", ["baseline_missing", "candidate_missing", "candidate_label", "prompt"])
def test_proposal_quality_summary_refuses_invalid_complete_pairs(proposal_evaluation_case, changed):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    if changed == "baseline_missing":
        baseline = baseline[:-1]
    elif changed == "candidate_missing":
        candidate = candidate[:-1]
    elif changed == "candidate_label":
        candidate = tuple(replace(record, candidate_id="opaque") for record in candidate)
    else:
        candidate = tuple(replace(record, prompt_token_ids=(1, 5)) for record in candidate)
    with pytest.raises(ValueError):
        identity.summarize_quality(baseline, candidate)
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("scenario", ["mixed", "missing", "incompatible"])
@pytest.mark.parametrize("reordered", [False, True])
def test_proposal_quality_by_item_preserves_grouped_counts(captured_proposal_run_case, scenario, reordered):
    from llm_loop_fix.records import QualityObservation

    model, before, identity, baseline, candidate, captured, template_calls = captured_proposal_run_case
    quality = QualityObservation("fixture-extractor", "fixture-grader", "4", "usable", True, "graded")
    baseline = tuple(replace(record, quality=replace(quality, correctness=record.key.item_id == "b")) for record in baseline)
    candidate = tuple(replace(record, quality=replace(quality, correctness=(record.key.item_id == "a") == (record.key.sample_index == 0)))
                      for record in candidate)
    if scenario != "mixed":
        candidate = tuple(replace(record, quality=None if scenario == "missing" else replace(record.quality, grader_id="other-grader"))
                          if record.key.item_id == "b" and record.key.sample_index == 0 else record for record in candidate)
    saved = baseline, candidate, identity.sha256, tuple(template_calls)
    if reordered:
        baseline, candidate = tuple(reversed(baseline)), tuple(reversed(candidate))
        identity = replace(identity, run_identity=replace(identity.run_identity, dataset=DatasetManifest(tuple(reversed(identity.run_identity.dataset.items)))))
    rows = identity.summarize_quality_by_item(baseline, candidate, expected_prompt_contexts=captured)
    aggregate = identity.summarize_quality(baseline, candidate, expected_prompt_contexts=captured)
    assert type(rows) is tuple and tuple(row["item_id"] for row in rows) == ("a", "b")
    assert all(row["scope"] == "recorded_quality_by_item_only" and row["proposal_evaluation_sha256"] == identity.sha256
               and row["paired_samples"] == 2 for row in rows)
    assert rows[0]["quality_complete"] and rows[0]["net_correct_samples"] == 1
    assert rows[0]["transitions"]["rescued"] == rows[0]["transitions"]["both_wrong"] == 1
    assert rows[1]["quality_complete"] is (scenario == "mixed")
    assert rows[1]["net_correct_samples"] == (-1 if scenario == "mixed" else None)
    assert rows[1]["transitions"]["regressed" if scenario == "mixed" else "not_comparable"] == 1
    for section in ("baseline", "candidate", "transitions"):
        assert {key: sum(row[section][key] for row in rows) for key in aggregate[section]} == aggregate[section]
    assert sum(row["paired_samples"] for row in rows) == aggregate["paired_samples"] == 4
    rows[0]["baseline"]["correct"] = 100
    assert identity.summarize_quality_by_item(baseline, candidate, expected_prompt_contexts=captured)[0]["baseline"]["correct"] == 0
    assert tuple(sorted(baseline, key=lambda record: record.key)) == saved[0]
    assert tuple(sorted(candidate, key=lambda record: record.key)) == saved[1]
    assert identity.sha256 == saved[2] and tuple(template_calls) == saved[3]
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("scenario", ["mixed", "zero", "positive", "negative"])
@pytest.mark.parametrize("reordered", [False, True])
def test_problem_quality_resampling_matches_two_item_reference(captured_proposal_run_case, scenario, reordered):
    import random
    from dataclasses import FrozenInstanceError
    from itertools import product
    from llm_loop_fix.records import QualityObservation
    from llm_loop_fix.verify import ProblemResamplingSpec

    model, before, identity, baseline, candidate, captured, template_calls = captured_proposal_run_case
    quality = QualityObservation("fixture-extractor", "fixture-grader", "4", "usable", True, "graded")
    baseline = tuple(replace(record, quality=replace(quality, correctness=record.key.item_id == "b")) for record in baseline)
    candidate = tuple(replace(record, quality=replace(quality, correctness=(record.key.item_id == "a") == (record.key.sample_index == 0)))
                      for record in candidate)
    if scenario != "mixed":
        baseline = tuple(replace(record, quality=replace(quality, correctness=scenario != "positive")) for record in baseline)
        candidate = tuple(replace(record, quality=replace(quality, correctness=scenario != "negative")) for record in candidate)
    saved = baseline, candidate, identity.sha256, tuple(template_calls), random.getstate()
    if reordered:
        baseline, candidate = tuple(reversed(baseline)), tuple(reversed(candidate))
        identity = replace(identity, run_identity=replace(identity.run_identity, dataset=DatasetManifest(identity.run_identity.dataset.items[::-1])))
    spec = ProblemResamplingSpec(128, 0)
    report = identity.summarize_quality_resampling(baseline, candidate, resampling_spec=spec,
                                                 expected_prompt_contexts=captured)
    nets = {"mixed": (1, -1), "zero": (0, 0), "positive": (2, 2), "negative": (-2, -2)}[scenario]
    exact = [sum(draw) / 4 for draw in product(nets, repeat=2)]
    assert report["available"] and report["reason"] is None and report["point_delta"] == sum(nets) / 4
    assert report["central_interval"] == (min(exact), max(exact))
    assert report["scope"] == "recorded_problem_resampling_only" and report["central_percent"] == 95
    assert report["seed"] == spec.seed
    assert report["resampling_spec_sha256"] == spec.sha256 and report["proposal_evaluation_sha256"] == identity.sha256
    assert report["replicates_used"] == 128 and report["sampled_problem_cells"] == 256
    assert report["confidence_coverage_claimed"] is False and report["acceptance_decision"] is None
    assert report["by_item"] == identity.summarize_quality_by_item(baseline, candidate, expected_prompt_contexts=captured)
    assert report == identity.summarize_quality_resampling(tuple(reversed(baseline)), tuple(reversed(candidate)),
                                                         resampling_spec=spec, expected_prompt_contexts=captured)
    assert (tuple(sorted(baseline, key=lambda record: record.key)), tuple(sorted(candidate, key=lambda record: record.key)),
            identity.sha256, tuple(template_calls), random.getstate()) == saved
    with pytest.raises(FrozenInstanceError):
        spec.seed = 1
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("scenario", ["missing", "unavailable", "error", "cross_pair_contract", "cross_item_contract", "one_item"])
def test_problem_quality_resampling_preserves_unavailable_without_random_draws(captured_proposal_run_case, monkeypatch, scenario):
    import random
    import llm_loop_fix.verify as verify_module
    from llm_loop_fix.records import QualityObservation

    model, before, identity, baseline, candidate, captured, template_calls = captured_proposal_run_case
    quality = QualityObservation("fixture-extractor", "fixture-grader", "4", "usable", True, "graded")
    baseline = tuple(replace(record, quality=quality) for record in baseline)
    candidate = tuple(replace(record, quality=quality) for record in candidate)
    if scenario == "one_item":
        run = replace(identity.run_identity, dataset=DatasetManifest(tuple(item for item in identity.run_identity.dataset.items if item.item_id == "a")))
        identity = replace(identity, run_identity=run)
        baseline = tuple(replace(record, experiment_id=run.sha256) for record in baseline if record.key.item_id == "a")
        candidate = tuple(replace(record, experiment_id=run.sha256) for record in candidate if record.key.item_id == "a")
        captured = {"a": captured["a"]}
    elif scenario == "cross_item_contract":
        baseline = tuple(replace(record, quality=replace(quality, grader_id="other")) if record.key.item_id == "b" else record for record in baseline)
        candidate = tuple(replace(record, quality=replace(quality, grader_id="other")) if record.key.item_id == "b" else record for record in candidate)
    else:
        changed = None if scenario == "missing" else (
            replace(quality, grader_status=scenario, correctness=None, reason="controlled_" + scenario) if scenario in ("unavailable", "error")
            else replace(quality, grader_id="other"))
        candidate = tuple(replace(record, quality=changed) if record.key.item_id == "b" and record.key.sample_index == 0 else record for record in candidate)
    saved = baseline, candidate, identity.sha256, tuple(template_calls), random.getstate()

    def forbidden(*args, **kwargs):
        pytest.fail("unavailable recorded resampling reached RNG construction")

    monkeypatch.setattr(verify_module.random, "Random", forbidden)
    report = identity.summarize_quality_resampling(baseline, candidate, resampling_spec=verify_module.ProblemResamplingSpec(128, 0),
                                                 expected_prompt_contexts=captured)
    reason = "fewer_than_two_items" if scenario == "one_item" else (
        "quality_contracts_differ_across_items" if scenario == "cross_item_contract" else "quality_incomplete")
    assert report["available"] is False and report["reason"] == reason
    assert report["point_delta"] is None and report["central_interval"] is None
    assert report["replicates_used"] == report["sampled_problem_cells"] == 0
    assert report["by_item"] == identity.summarize_quality_by_item(baseline, candidate, expected_prompt_contexts=captured)
    assert report["acceptance_decision"] is None and report["confidence_coverage_claimed"] is False
    assert (baseline, candidate, identity.sha256, tuple(template_calls), random.getstate()) == saved
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("replicates,seed", [(True, 0), (128, True), (1025, 0), (128, 2 ** 32)])
def test_problem_quality_resampling_refuses_invalid_frozen_specs(replicates, seed):
    from llm_loop_fix.verify import ProblemResamplingSpec

    with pytest.raises(ValueError, match="bounded plain-integer"):
        ProblemResamplingSpec(replicates, seed)


def test_problem_quality_resampling_preflights_work_and_spec_before_pairing(captured_proposal_run_case, monkeypatch):
    from types import SimpleNamespace
    import llm_loop_fix.verify as verify_module
    from llm_loop_fix.records import QualityObservation

    model, before, identity, baseline, candidate, captured, template_calls = captured_proposal_run_case
    spec = verify_module.ProblemResamplingSpec(128, 0)
    saved_calls = tuple(template_calls)
    quality = QualityObservation("fixture-extractor", "fixture-grader", "4", "usable", True, "graded")
    baseline = tuple(replace(record, quality=quality) for record in baseline)
    candidate = tuple(replace(record, quality=quality) for record in candidate)
    monkeypatch.setattr(verify_module, "_PROBLEM_RESAMPLING_MAX_CELLS", 256)
    report = identity.summarize_quality_resampling(baseline, candidate, resampling_spec=spec,
                                                 expected_prompt_contexts=captured)
    assert report["available"] and report["sampled_problem_cells"] == 256

    def forbidden(*args, **kwargs):
        pytest.fail("invalid resampling work/spec reached pairing")

    monkeypatch.setattr(verify_module.ProposalEvaluationIdentity, "pair_records", forbidden)
    with pytest.raises(ValueError, match="frozen problem resampling"):
        identity.summarize_quality_resampling(baseline, candidate, resampling_spec=SimpleNamespace(replicates=128, seed=0))
    for field_name, value in (("schema_version", True), ("central_percent", 95.0), ("method", "other")):
        malformed = verify_module.ProblemResamplingSpec(128, 0)
        object.__setattr__(malformed, field_name, value)
        with pytest.raises(ValueError, match="exact versioned method/schema"):
            identity.summarize_quality_resampling(baseline, candidate, resampling_spec=malformed)
    monkeypatch.setattr(verify_module, "_PROBLEM_RESAMPLING_MAX_CELLS", 255)
    with pytest.raises(ValueError, match="sampled-problem work budget"):
        identity.summarize_quality_resampling(baseline, candidate, resampling_spec=spec)
    assert tuple(template_calls) == saved_calls and capture_model_baseline(model) == before


def test_problem_quality_resampling_refuses_uniform_independent_context_drift(captured_proposal_run_case, monkeypatch):
    import llm_loop_fix.verify as verify_module
    from llm_loop_fix.records import QualityObservation

    model, before, identity, baseline, candidate, captured, template_calls = captured_proposal_run_case
    quality = QualityObservation("fixture-extractor", "fixture-grader", "4", "usable", True, "graded")
    baseline = tuple(replace(record, quality=quality, prompt_token_ids=(1, 5)) if record.key.item_id == "b" else replace(record, quality=quality)
                     for record in baseline)
    candidate = tuple(replace(record, quality=quality, prompt_token_ids=(1, 5)) if record.key.item_id == "b" else replace(record, quality=quality)
                      for record in candidate)
    saved_calls = tuple(template_calls)

    def forbidden(*args, **kwargs):
        pytest.fail("stale recorded resampling identity reached random draws")

    monkeypatch.setattr(verify_module.random, "Random", forbidden)
    with pytest.raises(ValueError):
        identity.summarize_quality_resampling(baseline, candidate, resampling_spec=verify_module.ProblemResamplingSpec(128, 0),
                                             expected_prompt_contexts=captured)
    assert tuple(template_calls) == saved_calls and capture_model_baseline(model) == before


def test_problem_quality_resampling_central_band_is_not_confidence_coverage(captured_proposal_run_case):
    import random
    from itertools import product
    from llm_loop_fix.records import QualityObservation
    from llm_loop_fix.verify import ProblemResamplingSpec

    model, before, identity, baseline, candidate, captured, template_calls = captured_proposal_run_case
    saved = baseline, candidate, identity.sha256, tuple(template_calls), random.getstate()
    quality = QualityObservation("fixture-extractor", "fixture-grader", "4", "usable", True, "graded")
    spec = ProblemResamplingSpec(128, 0)
    population_mean = (-1 + 1) / 2
    states = tuple(product((False, True), repeat=2))
    covered, cells = 0, 0
    for state in states:
        outcomes = dict(zip(("a", "b"), state))
        original = tuple(replace(record, quality=replace(quality, correctness=not outcomes[record.key.item_id]))
                         for record in baseline)
        edited = tuple(replace(record, quality=replace(quality, correctness=outcomes[record.key.item_id]))
                       for record in candidate)
        records_before = original, edited
        report = identity.summarize_quality_resampling(original, edited, resampling_spec=spec,
                                                     expected_prompt_contexts=captured)
        deltas = tuple(1 if outcome else -1 for outcome in state)
        assert report["available"] and report["point_delta"] == sum(deltas) / 2
        assert report["central_interval"] == (min(deltas), max(deltas))
        lower, upper = report["central_interval"]
        covered += lower <= population_mean <= upper
        cells += report["sampled_problem_cells"]
        assert report["confidence_coverage_claimed"] is False and report["acceptance_decision"] is None
        assert (original, edited) == records_before
    assert covered == 2 and len(states) == 4 and covered / len(states) == 0.5
    assert covered / len(states) < spec.central_percent / 100
    assert cells == 1024
    assert (baseline, candidate, identity.sha256, tuple(template_calls), random.getstate()) == saved
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("changed", ["baseline_missing", "candidate_missing", "uniform_drift"])
def test_proposal_quality_by_item_refuses_incomplete_or_independently_stale_pairs(captured_proposal_run_case, changed):
    model, before, identity, baseline, candidate, captured, template_calls = captured_proposal_run_case
    saved_calls = tuple(template_calls)
    if changed == "baseline_missing":
        baseline = baseline[:-1]
    elif changed == "candidate_missing":
        candidate = candidate[:-1]
    else:
        baseline = tuple(replace(record, prompt_token_ids=(1, 5)) if record.key.item_id == "b" else record for record in baseline)
        candidate = tuple(replace(record, prompt_token_ids=(1, 5)) if record.key.item_id == "b" else record for record in candidate)
    with pytest.raises(ValueError):
        identity.summarize_quality_by_item(baseline, candidate, expected_prompt_contexts=captured)
    assert tuple(template_calls) == saved_calls
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("method", ["pair_records", "summarize_quality"])
def test_proposal_pair_quality_independent_contexts_preserve_unbound_compatibility(proposal_evaluation_case, method):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    expected = {"a": ((1, 4), True)}
    operation = getattr(identity, method)
    assert operation(baseline, candidate, expected_prompt_contexts=expected) == operation(baseline, candidate)
    assert expected == {"a": ((1, 4), True)}
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("method", ["pair_records", "summarize_quality"])
def test_proposal_pair_quality_independent_contexts_refuse_uniform_drift(proposal_evaluation_case, method):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    baseline = tuple(replace(record, prompt_token_ids=(1, 5)) for record in baseline)
    candidate = tuple(replace(record, prompt_token_ids=(1, 5)) for record in candidate)
    operation = getattr(identity, method)
    operation(baseline, candidate)
    with pytest.raises(ValueError, match="independent prompt/context expectations"):
        operation(baseline, candidate, expected_prompt_contexts={"a": ((1, 4), True)})
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("method", ["pair_records", "summarize_quality"])
@pytest.mark.parametrize("changed", ["missing", "extra", "mutable_ids", "boolean_context", "negative_token", "over_budget"])
def test_proposal_pair_quality_independent_contexts_reject_bad_snapshots(proposal_evaluation_case, method, changed):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    expected = {"a": ((1, 4), True)}
    if changed == "missing":
        expected = {}
    elif changed == "extra":
        expected["b"] = ((1, 4), True)
    elif changed == "mutable_ids":
        expected["a"] = ([1, 4], True)
    elif changed == "boolean_context":
        expected["a"] = ((1, 4), 1)
    elif changed == "negative_token":
        expected["a"] = ((-1, 4), True)
    else:
        expected["a"] = ((1,) * 65537, True)
    with pytest.raises(ValueError):
        getattr(identity, method)(baseline, candidate, expected_prompt_contexts=expected)
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("method", ["pair_records", "summarize_quality"])
def test_proposal_pair_quality_independent_contexts_freeze_before_phase_validation(proposal_evaluation_case, method, monkeypatch):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    expected = {"a": ((1, 4), True)}
    calls = []
    original = EvaluationRunIdentity.validate_generations

    def validation_spy(self, records, **kwargs):
        assert kwargs["expected_prompt_contexts"] is not expected
        calls.append(kwargs["expected_prompt_contexts"].copy())
        result = original(self, records, **kwargs)
        expected["a"] = ((1, 5), False)
        return result

    monkeypatch.setattr(EvaluationRunIdentity, "validate_generations", validation_spy)
    getattr(identity, method)(baseline, candidate, expected_prompt_contexts=expected)
    assert calls == [{"a": ((1, 4), True)}] * 2
    assert expected == {"a": ((1, 5), False)}
    assert capture_model_baseline(model) == before


def test_proposal_evaluation_identity_binds_complete_declared_pairs_without_mutation(proposal_evaluation_case):
    model, baseline_identity, identity, baseline, candidate = proposal_evaluation_case
    pairs = identity.pair_records(tuple(reversed(baseline)), candidate)
    assert pairs == tuple(zip(baseline, candidate))
    assert identity.scope == "declared_run_numerical_proposal_only"
    assert identity.baseline_candidate_id == "baseline:" + baseline_identity.sha256
    assert identity.proposal_candidate_id == "proposal:" + identity.proposal.sha256
    assert capture_model_baseline(model) == baseline_identity
    with pytest.raises(FrozenInstanceError):
        identity.proposal = replace(identity.proposal, alpha=0.0)


@pytest.mark.parametrize("changed", ["alpha", "proposal_budget", "baseline", "run_code", "run_profile", "reference"])
def test_proposal_evaluation_identity_refuses_stale_declared_pairs(proposal_evaluation_case, changed):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    original_digest = identity.sha256
    if changed == "alpha":
        identity = replace(identity, proposal=replace(identity.proposal, alpha=0.0))
    elif changed == "proposal_budget":
        identity = replace(identity, proposal=replace(identity.proposal, budget=replace(identity.proposal.budget, max_alpha=1.5)))
    elif changed == "baseline":
        tensors = list(identity.proposal.baseline.tensors)
        target_names = {target.parameter.name for target in identity.proposal.targets}
        index = next(index for index, tensor in enumerate(tensors) if tensor.name not in target_names)
        tensors[index] = replace(tensors[index], sha256="f" * 64)
        identity = replace(identity, proposal=replace(identity.proposal, baseline=ModelBaseline(tuple(tensors))))
    elif changed == "run_code":
        identity = replace(identity, run_identity=replace(identity.run_identity, code_revision="f" * 40))
    elif changed == "run_profile":
        identity = replace(identity, run_identity=replace(identity.run_identity, profile=replace(identity.run_identity.profile, chat_template_sha256="d" * 64)))
    else:
        identity = replace(identity, run_identity=replace(identity.run_identity, dataset=DatasetManifest((replace(identity.run_identity.dataset.items[0], reference_answer="7"),))))
    assert identity.sha256 != original_digest
    with pytest.raises(ValueError):
        identity.pair_records(baseline, candidate)
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("changed", ["opaque_baseline", "opaque_proposal", "mixed", "missing_baseline", "missing_proposal", "prompt", "swapped"])
def test_proposal_evaluation_identity_refuses_incomplete_or_unbound_labels(proposal_evaluation_case, changed):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    if changed == "opaque_baseline":
        baseline = tuple(replace(record, candidate_id="base") for record in baseline)
    elif changed == "opaque_proposal":
        candidate = tuple(replace(record, candidate_id="edit") for record in candidate)
    elif changed == "mixed":
        candidate = (replace(candidate[0], candidate_id="edit"),) + candidate[1:]
    elif changed == "missing_baseline":
        baseline = baseline[:-1]
    elif changed == "missing_proposal":
        candidate = candidate[:-1]
    elif changed == "prompt":
        candidate = tuple(replace(record, prompt_token_ids=(1, 5)) for record in candidate)
    else:
        baseline, candidate = candidate, baseline
    with pytest.raises(ValueError):
        identity.pair_records(baseline, candidate)
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("changed", ["source_track", "confirmation", "benchmark", "clean_reference", "run_type", "proposal_type"])
def test_proposal_evaluation_identity_refuses_out_of_scope_declarations(proposal_evaluation_case, changed):
    model, before, identity, _, _ = proposal_evaluation_case
    run_identity = identity.run_identity
    proposal = identity.proposal
    if changed == "source_track":
        run_identity = replace(run_identity, profile=replace(run_identity.profile, track="source_reproduction", samples_per_item=4))
    elif changed in ("confirmation", "benchmark", "clean_reference"):
        run_identity = replace(run_identity, dataset=DatasetManifest((replace(run_identity.dataset.items[0], role=changed),)))
    elif changed == "run_type":
        run_identity = object()
    else:
        proposal = object()
    with pytest.raises(ValueError):
        ProposalEvaluationIdentity(run_identity, proposal)
    assert capture_model_baseline(model) == before


def test_record_trial_bridge_returns_complete_pairs_after_exact_restoration(proposal_evaluation_case):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    calls = []

    def callback(active_model, candidate_id):
        assert active_model is model
        calls.append(candidate_id)
        actual = capture_model_baseline(active_model)
        if candidate_id == identity.baseline_candidate_id:
            assert actual == before
            return baseline
        assert candidate_id == identity.proposal_candidate_id and actual != before
        return candidate

    pairs = collect_proposal_record_pairs(model, identity, callback)
    assert pairs == tuple(zip(baseline, candidate))
    assert calls == [identity.baseline_candidate_id, identity.proposal_candidate_id]
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("phase", ["baseline", "proposal"])
def test_prompt_bound_record_trial_refuses_drift_and_freezes_independent_expectations(proposal_evaluation_case, phase):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    expected = {"a": ((1, 4), True)}
    calls = []

    def callback(active_model, candidate_id):
        calls.append(candidate_id)
        expected["a"] = ((1, 5), True)
        records = baseline if candidate_id == identity.baseline_candidate_id else candidate
        if phase == "baseline" or candidate_id == identity.proposal_candidate_id:
            return tuple(replace(record, prompt_token_ids=(1, 5)) for record in records)
        return records

    with pytest.raises(ValueError, match="independent prompt/context"):
        collect_proposal_record_pairs(model, identity, callback, expected_prompt_contexts=expected)
    assert calls == ([identity.baseline_candidate_id] if phase == "baseline"
                     else [identity.baseline_candidate_id, identity.proposal_candidate_id])
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("thinking_context", [True, False])
def test_prompt_bound_record_trial_preserves_matching_records_and_isolates_caller_mapping(proposal_evaluation_case, thinking_context):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    baseline = tuple(replace(record, thinking_open_from_prompt=thinking_context) for record in baseline)
    candidate = tuple(replace(record, thinking_open_from_prompt=thinking_context) for record in candidate)
    expected = {"a": ((1, 4), thinking_context)}
    calls = []

    def callback(active_model, candidate_id):
        calls.append(candidate_id)
        expected.clear()
        actual = capture_model_baseline(active_model)
        assert (actual == before) is (candidate_id == identity.baseline_candidate_id)
        return tuple(reversed(baseline if candidate_id == identity.baseline_candidate_id else candidate))

    pairs = collect_proposal_record_pairs(model, identity, callback, expected_prompt_contexts=expected)
    assert pairs == tuple(zip(baseline, candidate))
    assert all(original is baseline[index] and edited is candidate[index] for index, (original, edited) in enumerate(pairs))
    assert calls == [identity.baseline_candidate_id, identity.proposal_candidate_id]
    assert not expected and capture_model_baseline(model) == before


@pytest.mark.parametrize("phase", ["baseline", "proposal"])
def test_prompt_bound_record_trial_refuses_thinking_context_drift_and_restores(proposal_evaluation_case, phase):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    calls = []

    def callback(active_model, candidate_id):
        calls.append(candidate_id)
        records = baseline if candidate_id == identity.baseline_candidate_id else candidate
        if phase == "baseline" or candidate_id == identity.proposal_candidate_id:
            return tuple(replace(record, thinking_open_from_prompt=False) for record in records)
        return records

    with pytest.raises(ValueError, match="independent prompt/context"):
        collect_proposal_record_pairs(model, identity, callback, expected_prompt_contexts={"a": ((1, 4), True)})
    assert len(calls) == (1 if phase == "baseline" else 2)
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("changed", ["mapping", "missing", "extra", "pair", "tokens", "empty",
                                     "bool_token", "negative_token", "thinking", "profile_limit"])
def test_prompt_bound_record_trial_preflights_plain_complete_bounded_expectations(proposal_evaluation_case, monkeypatch, changed):
    model, before, identity, _, _ = proposal_evaluation_case
    expected = {"a": ((1, 4), True)}

    def must_not_iterate():
        raise AssertionError("must not consume custom expectations")
        yield 1

    class ContextMap(dict):
        def __iter__(self):
            raise AssertionError("must not iterate a custom mapping")

    class ItemId(str):
        pass

    variants = [expected]
    if changed == "mapping":
        variants = [ContextMap(expected), must_not_iterate(), {1: ((1, 4), True)}, {ItemId("a"): ((1, 4), True)}]
    elif changed == "missing":
        variants = [{}]
    elif changed == "extra":
        variants = [dict(expected, extra=((1, 4), True))]
    elif changed == "pair":
        variants = [{"a": [(1, 4), True]}, {"a": ((1, 4),)}, {"a": ((1, 4), True, "extra")}]
    elif changed == "tokens":
        variants = [{"a": ([1, 4], True)}, {"a": (must_not_iterate(), True)}]
    elif changed == "empty":
        variants = [{"a": ((), True)}]
    elif changed == "bool_token":
        variants = [{"a": ((1, True), True)}]
    elif changed == "negative_token":
        variants = [{"a": ((1, -1), True)}]
    elif changed == "thinking":
        variants = [{"a": ((1, 4), 1)}]
    else:
        limit = identity.run_identity.profile.max_model_len - identity.run_identity.profile.max_new_tokens
        variants = [{"a": ((1,) * (limit + 1), True)}]

    def forbidden(*args, **kwargs):
        raise AssertionError("must preflight before model hashing, trial or callback")

    monkeypatch.setattr("llm_loop_fix.verify.capture_model_baseline", forbidden)
    monkeypatch.setattr("llm_loop_fix.verify.patch_trial", forbidden)
    for supplied in variants:
        with pytest.raises(ValueError, match="[Ii]ndependent prompt/context"):
            collect_proposal_record_pairs(model, identity, forbidden, expected_prompt_contexts=supplied)
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("phase", ["baseline", "proposal"])
@pytest.mark.parametrize("failure", [RuntimeError, torch.OutOfMemoryError, KeyboardInterrupt])
def test_prompt_bound_record_trial_restores_on_callback_interruptions(proposal_evaluation_case, phase, failure):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    calls = []

    def callback(active_model, candidate_id):
        calls.append(candidate_id)
        if candidate_id == (identity.baseline_candidate_id if phase == "baseline" else identity.proposal_candidate_id):
            raise failure("controlled prompt-bound callback interruption")
        return baseline if candidate_id == identity.baseline_candidate_id else candidate

    with pytest.raises(failure, match="controlled prompt-bound callback interruption"):
        collect_proposal_record_pairs(model, identity, callback, expected_prompt_contexts={"a": ((1, 4), True)})
    assert len(calls) == (1 if phase == "baseline" else 2)
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("bound", ["absolute", "aggregate"])
def test_prompt_bound_record_trial_refuses_absolute_and_aggregate_budgets_before_trial(proposal_evaluation_case, monkeypatch, bound):
    model, before, identity, _, _ = proposal_evaluation_case
    profile = replace(identity.run_identity.profile, max_new_tokens=1,
                      max_model_len=65538 if bound == "absolute" else 65537,
                      samples_per_item=2 if bound == "absolute" else 5)
    identity = replace(identity, run_identity=replace(identity.run_identity, profile=profile))
    expected = {"a": ((1,) * (65537 if bound == "absolute" else 65536), True)}

    def forbidden(*args, **kwargs):
        raise AssertionError("must reject prompt budgets before model hashing, trial or callback")

    monkeypatch.setattr("llm_loop_fix.verify.capture_model_baseline", forbidden)
    monkeypatch.setattr("llm_loop_fix.verify.patch_trial", forbidden)
    with pytest.raises(ValueError, match="[Ii]ndependent prompt/context"):
        collect_proposal_record_pairs(model, identity, forbidden, expected_prompt_contexts=expected)
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("phase", ["baseline", "proposal"])
@pytest.mark.parametrize("failure", [RuntimeError, torch.OutOfMemoryError, KeyboardInterrupt])
def test_record_trial_bridge_restores_on_callback_failure(proposal_evaluation_case, phase, failure):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    calls = []

    def callback(active_model, candidate_id):
        calls.append(candidate_id)
        selected = identity.baseline_candidate_id if phase == "baseline" else identity.proposal_candidate_id
        if candidate_id == selected:
            raise failure("controlled record callback failure")
        return baseline if candidate_id == identity.baseline_candidate_id else candidate

    with pytest.raises(failure):
        collect_proposal_record_pairs(model, identity, callback)
    assert len(calls) == (1 if phase == "baseline" else 2)
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("noop", [False, True])
def test_record_trial_bridge_refuses_stale_actual_baseline_before_callback(proposal_evaluation_case, noop):
    model, _, identity, _, _ = proposal_evaluation_case
    if noop:
        identity = replace(identity, proposal=replace(identity.proposal, alpha=0.0))
    with torch.no_grad():
        model.layers[0].mlp.down_proj.weight.add_(0.25)
    stale = capture_model_baseline(model)
    calls = []

    def forbidden_callback(*args):
        calls.append(True)
        raise AssertionError("stale baseline reached record callback")

    with pytest.raises(ValueError, match="complete registered baseline"):
        collect_proposal_record_pairs(model, identity, forbidden_callback)
    assert calls == [] and capture_model_baseline(model) == stale


@pytest.mark.parametrize("phase", ["baseline", "proposal"])
@pytest.mark.parametrize("changed", ["label", "missing"])
def test_record_trial_bridge_refuses_invalid_callback_records_and_restores(proposal_evaluation_case, phase, changed):
    model, before, identity, baseline, candidate = proposal_evaluation_case

    def callback(active_model, candidate_id):
        records = baseline if candidate_id == identity.baseline_candidate_id else candidate
        selected = identity.baseline_candidate_id if phase == "baseline" else identity.proposal_candidate_id
        if candidate_id == selected:
            return tuple(replace(record, candidate_id="opaque") for record in records) if changed == "label" else records[:-1]
        return records

    with pytest.raises(ValueError):
        collect_proposal_record_pairs(model, identity, callback)
    assert capture_model_baseline(model) == before


def test_record_trial_bridge_refuses_cross_candidate_prompt_drift_and_restores(proposal_evaluation_case):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    candidate = tuple(replace(record, prompt_token_ids=(1, 5)) for record in candidate)
    with pytest.raises(ValueError, match="prompt mismatch"):
        collect_proposal_record_pairs(model, identity, lambda active_model, candidate_id: baseline if candidate_id == identity.baseline_candidate_id else candidate)
    assert capture_model_baseline(model) == before


def test_record_trial_bridge_noop_does_not_change_live_weights(proposal_evaluation_case):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    identity = replace(identity, proposal=replace(identity.proposal, alpha=0.0))
    candidate = tuple(replace(record, candidate_id=identity.proposal_candidate_id) for record in candidate)

    def callback(active_model, candidate_id):
        assert capture_model_baseline(active_model) == before
        return baseline if candidate_id == identity.baseline_candidate_id else candidate

    assert collect_proposal_record_pairs(model, identity, callback) == tuple(zip(baseline, candidate))
    assert capture_model_baseline(model) == before


def test_record_trial_bridge_module_graph_corruption_is_fatal(proposal_evaluation_case):
    model, before, identity, baseline, _ = proposal_evaluation_case
    original = model.layers[1].mlp.down_proj

    def callback(active_model, candidate_id):
        replacement = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            replacement.weight.copy_(original.weight)
        active_model.layers[1].mlp.down_proj = replacement
        return baseline

    try:
        with pytest.raises(RestorationError, match="Fatal"):
            collect_proposal_record_pairs(model, identity, callback)
    finally:
        model.layers[1].mlp.down_proj = original
    assert capture_model_baseline(model) == before


def test_record_trial_bridge_unbacked_corruption_is_fatal_not_claimed_restored(proposal_evaluation_case):
    model, before, identity, baseline, _ = proposal_evaluation_case
    snapshots = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    def callback(active_model, candidate_id):
        with torch.no_grad():
            active_model.layers[0].mlp.down_proj.weight.add_(0.25)
        return baseline

    try:
        with pytest.raises(RestorationError, match="Fatal"):
            collect_proposal_record_pairs(model, identity, callback)
        assert capture_model_baseline(model) != before
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(snapshots[name])
    assert capture_model_baseline(model) == before


def test_record_trial_bridge_restoration_failure_is_fatal(proposal_evaluation_case, monkeypatch):
    model, before, identity, baseline, _ = proposal_evaluation_case

    def failed_restore(*args, **kwargs):
        raise RuntimeError("controlled restoration failure")

    monkeypatch.setattr(verify_module, "restore_weights", failed_restore)
    with pytest.raises(RestorationError, match="Fatal"):
        collect_proposal_record_pairs(model, identity, lambda active_model, candidate_id: baseline)
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("changed", ["model", "identity", "callback", "schedule"])
def test_record_trial_bridge_refuses_invalid_admission_without_callbacks(proposal_evaluation_case, changed):
    model, before, identity, _, _ = proposal_evaluation_case
    calls = []

    def callback(*args):
        calls.append(True)
        raise AssertionError("invalid declaration reached callback")

    supplied_model = model
    if changed == "model":
        supplied_model = object()
    elif changed == "identity":
        identity = object()
    elif changed == "callback":
        callback = None
    else:
        items = tuple(replace(identity.run_identity.dataset.items[0], item_id=str(index), prompt="question " + str(index)) for index in range(9))
        identity = replace(identity, run_identity=replace(identity.run_identity, dataset=DatasetManifest(items)))
    with pytest.raises(ValueError):
        collect_proposal_record_pairs(supplied_model, identity, callback)
    assert calls == [] and capture_model_baseline(model) == before


@pytest.mark.parametrize("phase", ["baseline", "proposal"])
@pytest.mark.parametrize("changed", ["tokens", "thinking"])
def test_prompt_bound_graded_trial_refuses_drift_before_grader_and_restores(proposal_evaluation_case, monkeypatch, phase, changed):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    expected = {"a": ((1, 4), True)}
    callbacks = []
    graders = []
    selected = identity.baseline_candidate_id if phase == "baseline" else identity.proposal_candidate_id

    def callback(active_model, candidate_id):
        callbacks.append(candidate_id)
        expected["a"] = ((1, 5), True) if changed == "tokens" else ((1, 4), False)
        records = baseline if candidate_id == identity.baseline_candidate_id else candidate
        if phase == "baseline" or candidate_id == selected:
            return tuple(replace(record, prompt_token_ids=(1, 5)) if changed == "tokens"
                         else replace(record, thinking_open_from_prompt=False) for record in records)
        return records

    def grade(run, generations, *, candidate_id, registry=None):
        graders.append(candidate_id)
        assert candidate_id != selected, "mismatched prompt must never reach the grader"
        return generations

    monkeypatch.setattr("llm_loop_fix.verify.evaluate_generation_batch", grade)
    with pytest.raises(ValueError, match="independent prompt/context"):
        evaluate_proposal_record_pairs(model, identity, callback, expected_prompt_contexts=expected)
    assert callbacks == ([identity.baseline_candidate_id] if phase == "baseline"
                         else [identity.baseline_candidate_id, identity.proposal_candidate_id])
    assert graders == ([] if phase == "baseline" else [identity.baseline_candidate_id])
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("attached", [False, True])
def test_prompt_bound_graded_trial_preserves_quality_health_and_mapping_isolation(proposal_evaluation_case, monkeypatch, attached):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    if attached:
        baseline = tuple(replace(record, health=observe_generation_health(record, thinking_required=True)) for record in baseline)
        candidate = tuple(replace(record, health=observe_generation_health(record, thinking_required=True)) for record in candidate)
    expected = {"a": ((1, 4), True)}
    calls = []

    def callback(active_model, candidate_id):
        calls.append(candidate_id)
        expected.clear()
        return baseline if candidate_id == identity.baseline_candidate_id else candidate

    pairs = evaluate_proposal_record_pairs(model, identity, callback, expected_prompt_contexts=expected)
    for pair, originals in zip(pairs, zip(baseline, candidate)):
        for result, original in zip(pair, originals):
            assert result.quality.correctness is True and result.health is not None
            assert replace(result, quality=original.quality, health=original.health) == original
            if attached:
                assert result.health is original.health
    assert calls == [identity.baseline_candidate_id, identity.proposal_candidate_id]
    assert not expected and capture_model_baseline(model) == before


    def changed_grade(run, generations, *, candidate_id, registry=None):
        return tuple(replace(record, prompt_token_ids=(1, 5)) for record in generations)

    monkeypatch.setattr("llm_loop_fix.verify.evaluate_generation_batch", changed_grade)
    with pytest.raises(ValueError, match="independent prompt/context"):
        evaluate_proposal_record_pairs(model, identity, callback, expected_prompt_contexts={"a": ((1, 4), True)})
    assert calls[-1] == identity.baseline_candidate_id and len(calls) == 3
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("changed", ["missing", "tokens", "context", "extra"])
def test_prompt_bound_graded_trial_preflights_expectations_before_model_or_grader(proposal_evaluation_case, monkeypatch, changed):
    model, before, identity, _, _ = proposal_evaluation_case
    if changed == "missing":
        expected = {}
    elif changed == "tokens":
        expected = {"a": ([1, 4], True)}
    elif changed == "context":
        expected = {"a": ((1, 4), None)}
    else:
        expected = {"a": ((1, 4), True), "extra": ((1, 4), True)}

    def forbidden(*args, **kwargs):
        raise AssertionError("must preflight before model setup, callback or grader")

    monkeypatch.setattr("llm_loop_fix.verify.capture_model_baseline", forbidden)
    monkeypatch.setattr("llm_loop_fix.verify.patch_trial", forbidden)
    monkeypatch.setattr("llm_loop_fix.verify.evaluate_generation_batch", forbidden)
    with pytest.raises(ValueError, match="[Ii]ndependent prompt/context"):
        evaluate_proposal_record_pairs(model, identity, forbidden, expected_prompt_contexts=expected)
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("phase", ["baseline", "proposal"])
@pytest.mark.parametrize("failure", [RuntimeError, torch.OutOfMemoryError, KeyboardInterrupt])
def test_prompt_bound_graded_trial_restores_after_grader_interruption(proposal_evaluation_case, monkeypatch, phase, failure):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    callbacks = []
    graders = []
    selected = identity.baseline_candidate_id if phase == "baseline" else identity.proposal_candidate_id

    def callback(active_model, candidate_id):
        callbacks.append(candidate_id)
        return baseline if candidate_id == identity.baseline_candidate_id else candidate

    def grade(run, generations, *, candidate_id, registry=None):
        graders.append(candidate_id)
        assert (capture_model_baseline(model) == before) is (candidate_id == identity.baseline_candidate_id)
        if candidate_id == selected:
            raise failure("controlled prompt-bound grader interruption")
        return generations

    monkeypatch.setattr("llm_loop_fix.verify.evaluate_generation_batch", grade)
    with pytest.raises(failure, match="controlled prompt-bound grader interruption"):
        evaluate_proposal_record_pairs(model, identity, callback, expected_prompt_contexts={"a": ((1, 4), True)})
    assert callbacks == graders and len(graders) == (1 if phase == "baseline" else 2)
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("changed", ["identity", "callback", "model"])
def test_prompt_bound_graded_trial_refuses_invalid_arguments_before_callback(proposal_evaluation_case, monkeypatch, changed):
    model, before, identity, _, _ = proposal_evaluation_case

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid arguments must not execute a callback or grader")

    monkeypatch.setattr("llm_loop_fix.verify.evaluate_generation_batch", forbidden)
    with pytest.raises(ValueError):
        evaluate_proposal_record_pairs(None if changed == "model" else model,
                                       object() if changed == "identity" else identity,
                                       None if changed == "callback" else forbidden,
                                       expected_prompt_contexts={"a": ((1, 4), True)})
    assert capture_model_baseline(model) == before


def test_prompt_bound_graded_trial_refuses_incomplete_records_before_grader(proposal_evaluation_case, monkeypatch):
    model, before, identity, baseline, _ = proposal_evaluation_case
    callbacks = []

    def callback(active_model, candidate_id):
        callbacks.append(candidate_id)
        return baseline[:-1]

    def forbidden(*args, **kwargs):
        raise AssertionError("incomplete records must not reach the grader")

    monkeypatch.setattr("llm_loop_fix.verify.evaluate_generation_batch", forbidden)
    with pytest.raises(ValueError):
        evaluate_proposal_record_pairs(model, identity, callback, expected_prompt_contexts={"a": ((1, 4), True)})
    assert callbacks == [identity.baseline_candidate_id]
    assert capture_model_baseline(model) == before


def test_graded_trial_bridge_preserves_complete_records_and_exact_restoration(proposal_evaluation_case):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    pairs = evaluate_proposal_record_pairs(model, identity, lambda active_model, candidate_id: baseline if candidate_id == identity.baseline_candidate_id else candidate)
    assert len(pairs) == 2
    for pair, originals in zip(pairs, zip(baseline, candidate)):
        for record, original in zip(pair, originals):
            assert record.quality.correctness is True and record.quality.grader_status == "graded"
            assert record.health is not None and record.health.thinking.closure_complete
            assert replace(record, quality=None, health=None) == original
    assert all(record.quality is None and record.health is None for record in baseline + candidate)
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("phase", ["baseline", "proposal"])
@pytest.mark.parametrize("failure", [RuntimeError, torch.OutOfMemoryError, KeyboardInterrupt])
def test_graded_trial_bridge_restores_on_grading_interruption(proposal_evaluation_case, monkeypatch, phase, failure):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    original = evaluation_module.evaluate_math_generation
    calls = []

    def interrupted_grader(item, record, profile, **kwargs):
        calls.append(record.candidate_id)
        selected = identity.baseline_candidate_id if phase == "baseline" else identity.proposal_candidate_id
        if record.candidate_id == selected and record.key.sample_index == 1:
            raise failure("controlled trial grader interruption")
        return original(item, record, profile, **kwargs)

    monkeypatch.setattr(evaluation_module, "evaluate_math_generation", interrupted_grader)
    with pytest.raises(failure):
        evaluate_proposal_record_pairs(model, identity, lambda active_model, candidate_id: baseline if candidate_id == identity.baseline_candidate_id else candidate)
    assert len(calls) == (2 if phase == "baseline" else 4)
    assert all(record.quality is None and record.health is None for record in baseline + candidate)
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("changed", ["key", "health"])
def test_graded_trial_bridge_preflights_candidate_before_its_grader(proposal_evaluation_case, monkeypatch, changed):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    if changed == "key":
        candidate = candidate[:-1] + (replace(candidate[-1], key=replace(candidate[-1].key, seed=1)),)
    else:
        health = observe_generation_health(candidate[-1], thinking_required=True)
        candidate = candidate[:-1] + (replace(candidate[-1], health=health, generated_text=r"\boxed{4}"),)
    original = evaluation_module.evaluate_math_generation
    calls = []

    def measured_grader(item, record, profile, **kwargs):
        calls.append(record.candidate_id)
        return original(item, record, profile, **kwargs)

    monkeypatch.setattr(evaluation_module, "evaluate_math_generation", measured_grader)
    with pytest.raises(ValueError):
        evaluate_proposal_record_pairs(model, identity, lambda active_model, candidate_id: baseline if candidate_id == identity.baseline_candidate_id else candidate)
    assert calls == [identity.baseline_candidate_id] * 2
    assert capture_model_baseline(model) == before


@pytest.mark.parametrize("outcome", ["missing_reference", "backend_error", "parser_error", "wrong", "empty"])
def test_graded_trial_bridge_retains_independent_quality_without_acceptance(proposal_evaluation_case, monkeypatch, outcome):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    if outcome == "missing_reference":
        item = replace(identity.run_identity.dataset.items[0], reference_answer=None)
        identity = replace(identity, run_identity=replace(identity.run_identity, dataset=DatasetManifest((item,))))
        baseline = tuple(replace(record, experiment_id=identity.run_identity.sha256) for record in baseline)
        candidate = tuple(replace(record, experiment_id=identity.run_identity.sha256) for record in candidate)
    elif outcome == "backend_error":
        candidate = tuple(replace(record, finish_reason="error") for record in candidate)
    elif outcome == "parser_error":
        import math_verify
        from math_verify.errors import TimeoutException

        def failed_parser(*args, **kwargs):
            raise TimeoutException("controlled trial parser timeout")

        monkeypatch.setattr(math_verify, "parse", failed_parser)
    else:
        candidate = tuple(replace(record, generated_text=r"</think>\boxed{5}" if outcome == "wrong" else "</think>") for record in candidate)
    pairs = evaluate_proposal_record_pairs(model, identity, lambda active_model, candidate_id: baseline if candidate_id == identity.baseline_candidate_id else candidate)
    for _, record in pairs:
        assert record.quality.correctness is (False if outcome in ("wrong", "empty") else None)
        assert record.quality.grader_status == ("graded" if outcome in ("wrong", "empty") else "error" if outcome == "parser_error" else "unavailable")
        assert record.quality.usability == ("empty" if outcome == "empty" else "usable")
    assert capture_model_baseline(model) == before


def test_graded_trial_bridge_preserves_attached_health_objects(proposal_evaluation_case):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    baseline = tuple(replace(record, health=observe_generation_health(record, thinking_required=True, repetition_spec=RepetitionSpec(max_tokens=16))) for record in baseline)
    candidate = tuple(replace(record, health=observe_generation_health(record, thinking_required=True, repetition_spec=RepetitionSpec(max_tokens=16))) for record in candidate)
    pairs = evaluate_proposal_record_pairs(model, identity, lambda active_model, candidate_id: baseline if candidate_id == identity.baseline_candidate_id else candidate)
    assert all(original.health is source.health and edited.health is target.health for (original, edited), source, target in zip(pairs, baseline, candidate))
    assert capture_model_baseline(model) == before


def test_graded_trial_bridge_accepts_only_controlled_declared_instruction_registry(proposal_evaluation_case):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    profile = replace(identity.run_identity.profile, task="ifeval", grader="ifeval-strict", grader_version="controlled-v1")
    item = replace(identity.run_identity.dataset.items[0], reference_answer=None, evaluator_id="ifeval-strict",
                   instructions=(InstructionSpec("controlled:contains_four", "{}"),))
    identity = replace(identity, run_identity=replace(identity.run_identity, profile=profile, dataset=DatasetManifest((item,))))
    baseline = tuple(replace(record, experiment_id=identity.run_identity.sha256, profile_id=profile.sha256) for record in baseline)
    candidate = tuple(replace(record, experiment_id=identity.run_identity.sha256, profile_id=profile.sha256) for record in candidate)

    class ControlledChecker:
        def __init__(self, instruction_id):
            assert instruction_id == "controlled:contains_four"

        def get_instruction_args_keys(self):
            return ()

        def get_instruction_args(self):
            return {}

        def build_description(self):
            return "Controlled test constraint"

        def check_following(self, value):
            return "4" in value

    registry = InstructionRegistry("controlled-v1", {"controlled:contains_four": ControlledChecker})
    pairs = evaluate_proposal_record_pairs(model, identity, lambda active_model, candidate_id: baseline if candidate_id == identity.baseline_candidate_id else candidate, registry=registry)
    assert all(record.quality.correctness is True and record.quality.instruction_following == (True,) for pair in pairs for record in pair)
    assert capture_model_baseline(model) == before


def test_graded_trial_bridge_refuses_noncallable_before_trial(proposal_evaluation_case):
    model, before, identity, _, _ = proposal_evaluation_case
    with pytest.raises(ValueError, match="requires a callback"):
        evaluate_proposal_record_pairs(model, identity, None)
    assert capture_model_baseline(model) == before


def test_graded_trial_bridge_refuses_stale_model_before_callback_or_grader(proposal_evaluation_case, monkeypatch):
    model, _, identity, _, _ = proposal_evaluation_case
    with torch.no_grad():
        model.layers[0].mlp.down_proj.weight.add_(0.25)
    stale = capture_model_baseline(model)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("stale model reached callback/grader")

    monkeypatch.setattr(evaluation_module, "evaluate_math_generation", forbidden)
    with pytest.raises(ValueError, match="complete registered baseline"):
        evaluate_proposal_record_pairs(model, identity, forbidden)
    assert calls == [] and capture_model_baseline(model) == stale


def test_graded_trial_bridge_noop_restores_without_selection(proposal_evaluation_case):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    identity = replace(identity, proposal=replace(identity.proposal, alpha=0.0))
    candidate = tuple(replace(record, candidate_id=identity.proposal_candidate_id) for record in candidate)

    def callback(active_model, candidate_id):
        assert capture_model_baseline(active_model) == before
        return baseline if candidate_id == identity.baseline_candidate_id else candidate

    pairs = evaluate_proposal_record_pairs(model, identity, callback)
    assert len(pairs) == 2 and all(record.quality.correctness is True for pair in pairs for record in pair)
    assert capture_model_baseline(model) == before


def test_graded_trial_bridge_refuses_grader_output_changes_and_restores(proposal_evaluation_case, monkeypatch):
    model, before, identity, baseline, candidate = proposal_evaluation_case
    original = evaluation_module.evaluate_math_generation

    def corrupt_grader(item, record, profile, **kwargs):
        result = original(item, record, profile, **kwargs)
        return replace(result, backend_request_id="tampered") if record.candidate_id == identity.proposal_candidate_id else result

    monkeypatch.setattr(evaluation_module, "evaluate_math_generation", corrupt_grader)
    with pytest.raises(ValueError, match="grading result"):
        evaluate_proposal_record_pairs(model, identity, lambda active_model, candidate_id: baseline if candidate_id == identity.baseline_candidate_id else candidate)
    assert capture_model_baseline(model) == before


def test_detect_loop_failure_modes():
    tokens = [1, 2, 3, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9]
    res = detect_loop_onset(tokens, min_ngram=3, min_repeats=3)
    assert res.is_looping is True
    assert res.failure_mode == "ngram_loop"
    assert res.onset_index == 3
    assert res.period_length == 3
    assert res.repeat_count == 4

    # Test whitespace-tail degeneration guard
    ws_tokens = [1, 2, 3, 4] + [0] * 16
    ws_res = detect_loop_onset(
        ws_tokens, whitespace_token_ids={0}, whitespace_tail_threshold=16
    )
    assert ws_res.is_looping is True
    assert ws_res.has_whitespace_tail is True
    assert ws_res.failure_mode == "whitespace_tail"
    assert ws_res.onset_index == 4

    # Test token-ceiling truncation detection
    cap_tokens = list(range(20))
    cap_res = detect_loop_onset(cap_tokens, max_new_tokens=20, eos_token_ids={99})
    assert cap_res.is_looping is True
    assert cap_res.hit_token_cap is True
    assert cap_res.failure_mode == "cap_truncation"


@pytest.fixture
def numerical_patch_case():
    model = ToyLoopingCausalLM(vocab_size=4, d_model=2, d_ff=2)
    culprit = CulpritMatrix("layers.1.mlp.down_proj", 1, [2, 2], 1.0, 1.0, 1.0, [0], [1.0])
    report = LocalizationReport({1: torch.tensor([1.0, 0.0])}, {1: 1.0}, [culprit])
    return model, report


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("alpha", [0.001, 2.0])
def test_patch_metrics_measure_actual_stored_cast_and_restore(numerical_patch_case, dtype, alpha):
    model, report = numerical_patch_case
    model.to(dtype=dtype)
    original = model.layers[1].mlp.down_proj.weight.detach().clone()
    baseline_hashes = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    backup = {}
    try:
        result = apply_surgical_patch(model, report, PatchConfig(alpha=alpha, top_k_matrices=1, top_k_columns=1), backup)
        stored = model.layers[1].mlp.down_proj.weight.detach()
        measured = float((torch.linalg.vector_norm(stored.double() - original.double()) / torch.linalg.vector_norm(original.double())).item())
        metrics = result.matrix_metrics[0]
        assert metrics.relative_frobenius_change == measured
        assert result.bitwise_changed_params_count == _bitwise_changed_entries(original, stored)
        assert result.modified_params_count == 2
        assert metrics.pre_edit_sha256 == baseline_hashes["layers.1.mlp.down_proj.weight"]
        assert metrics.post_edit_sha256 == _tensor_sha256(stored)
        assert _tensor_sha256(stored[:, 1:]) == _tensor_sha256(original[:, 1:])
        if dtype == torch.bfloat16 and alpha == 0.001:
            assert metrics.relative_frobenius_change == 0.0
            assert result.bitwise_changed_params_count == 0
            assert metrics.pre_edit_sha256 == metrics.post_edit_sha256
    finally:
        restore_weights(model, backup)
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == baseline_hashes


@pytest.mark.parametrize("kind", ["alpha_zero", "cast_erased", "orthogonal"])
def test_noop_patch_is_explicit_and_performs_no_copy(numerical_patch_case, monkeypatch, kind):
    model, report = numerical_patch_case
    alpha = 0.0 if kind == "alpha_zero" else 0.001 if kind == "cast_erased" else 1.0
    if kind == "cast_erased":
        model.to(dtype=torch.bfloat16)
    if kind == "orthogonal":
        report.layer_directions[1] = torch.tensor([0.0, 1.0])
    if kind == "alpha_zero":
        with torch.no_grad():
            model.layers[1].mlp.down_proj.weight[1, 0] = -0.0
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}

    def forbid_copy(*args, **kwargs):
        pytest.fail("No-op patch reached tensor mutation")

    monkeypatch.setattr(torch.Tensor, "copy_", forbid_copy)
    backup = {}
    result = apply_surgical_patch(model, report, PatchConfig(alpha=alpha, top_k_matrices=1, top_k_columns=1), backup)
    assert result.is_noop is True
    assert result.is_effective is False
    assert result.bitwise_changed_params_count == 0
    assert backup == {}
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before
    if kind == "alpha_zero":
        assert result.modified_modules == []
        assert result.modified_params_count == 0
    else:
        assert result.matrix_metrics[0].relative_frobenius_change == 0.0
        assert result.matrix_metrics[0].pre_edit_sha256 == result.matrix_metrics[0].post_edit_sha256


def test_effective_float32_change_is_not_rounded_to_zero(numerical_patch_case):
    model, report = numerical_patch_case
    backup = {}
    before = _tensor_sha256(model.layers[1].mlp.down_proj.weight)
    try:
        result = apply_surgical_patch(model, report, PatchConfig(alpha=1e-7, top_k_matrices=1, top_k_columns=1), backup)
        assert 0 < result.matrix_metrics[0].relative_frobenius_change < 1e-6
        assert result.bitwise_changed_params_count == 1
    finally:
        restore_weights(model, backup)
    assert _tensor_sha256(model.layers[1].mlp.down_proj.weight) == before


def test_small_bf16_weight_norm_does_not_underflow(numerical_patch_case):
    model, report = numerical_patch_case
    model.to(dtype=torch.bfloat16)
    with torch.no_grad():
        model.layers[1].mlp.down_proj.weight[0, 0] = 2.0 ** -100
    backup = {}
    before = _tensor_sha256(model.layers[1].mlp.down_proj.weight)
    try:
        result = apply_surgical_patch(model, report, PatchConfig(alpha=1.0, top_k_matrices=1, top_k_columns=1), backup)
        assert result.matrix_metrics[0].relative_frobenius_change == 1.0
        assert result.bitwise_changed_params_count == 1
    finally:
        restore_weights(model, backup)
    assert _tensor_sha256(model.layers[1].mlp.down_proj.weight) == before


def test_late_missing_patch_target_fails_before_any_mutation(numerical_patch_case):
    model, report = numerical_patch_case
    report.ranked_culprits.append(CulpritMatrix("missing.target", 0, [2, 2], 1.0, 1.0, 1.0, [0], [1.0]))
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    backup = {}
    try:
        with pytest.raises(ValueError, match="Unknown patch target"):
            apply_surgical_patch(model, report, PatchConfig(alpha=2.0, top_k_matrices=2, top_k_columns=1), backup)
        assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before
        assert backup == {}
    finally:
        restore_weights(model, backup)


@pytest.mark.parametrize("kind", ["missing_direction", "nonunit", "zero", "nan", "direction_shape", "layer", "matrix_shape", "duplicate_columns", "negative_column", "out_of_range", "nonfinite_weight", "unsupported_dtype"])
def test_late_invalid_proposal_is_rejected_before_any_copy(numerical_patch_case, monkeypatch, kind):
    model, report = numerical_patch_case
    later = CulpritMatrix("layers.0.mlp.down_proj", 0, [2, 2], 1.0, 1.0, 1.0, [0], [1.0])
    report.ranked_culprits.append(later)
    report.layer_directions[0] = torch.tensor([1.0, 0.0])
    if kind == "missing_direction":
        del report.layer_directions[0]
    elif kind == "nonunit":
        report.layer_directions[0] *= 2
    elif kind == "zero":
        report.layer_directions[0].zero_()
    elif kind == "nan":
        report.layer_directions[0][0] = float("nan")
    elif kind == "direction_shape":
        report.layer_directions[0] = torch.ones(3)
    elif kind == "layer":
        later.layer_idx = 1
    elif kind == "matrix_shape":
        later.shape = [3, 2]
    elif kind == "duplicate_columns":
        later.top_column_indices = [0, 0]
        later.top_column_scores = [1.0, 1.0]
    elif kind == "negative_column":
        later.top_column_indices = [-1]
    elif kind == "out_of_range":
        later.top_column_indices = [2]
    elif kind == "nonfinite_weight":
        with torch.no_grad():
            model.layers[0].mlp.down_proj.weight[0, 0] = float("inf")
    else:
        model.layers[0].mlp.down_proj.to(dtype=torch.float64)
    snapshots = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    original_copy = torch.Tensor.copy_
    copies = []

    def track_copy(tensor, source, *args, **kwargs):
        copies.append(tensor.data_ptr())
        return original_copy(tensor, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", track_copy)
    backup = {}
    try:
        with pytest.raises(ValueError):
            apply_surgical_patch(model, report, PatchConfig(alpha=2.0, top_k_matrices=2, top_k_columns=1), backup)
        assert copies == []
        assert backup == {}
        assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                original_copy(parameter, snapshots[name])


@pytest.mark.parametrize("changes", [{"alpha": -1.0}, {"alpha": 7.0}, {"alpha": float("nan")}, {"alpha": float("inf")}, {"alpha": True}, {"top_k_matrices": -1}, {"top_k_matrices": 1.5}, {"top_k_matrices": True}, {"top_k_columns": 0}, {"top_k_columns": True}, {"sparse_column_fraction": 0.0}, {"sparse_column_fraction": 1.1}, {"sparse_column_fraction": float("nan")}, {"sparse_column_fraction": True}, {"mode": "unknown"}])
def test_invalid_patch_config_never_mutates(numerical_patch_case, monkeypatch, changes):
    model, report = numerical_patch_case
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    config = PatchConfig(alpha=2.0, top_k_matrices=1, top_k_columns=1)
    for name, value in changes.items():
        setattr(config, name, value)

    def forbid_copy(*args, **kwargs):
        pytest.fail("Invalid config reached tensor mutation")

    monkeypatch.setattr(torch.Tensor, "copy_", forbid_copy)
    backup = {}
    with pytest.raises(ValueError):
        apply_surgical_patch(model, report, config, backup)
    assert backup == {}
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before


@pytest.mark.parametrize("mode", ["top_k_columns", "sparse_columns"])
def test_insufficient_support_never_falls_back_to_static_alignment(numerical_patch_case, monkeypatch, mode):
    model, report = numerical_patch_case
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}

    def forbid_selection(*args, **kwargs):
        pytest.fail("Insufficient support reached static alignment selection")

    monkeypatch.setattr(torch, "topk", forbid_selection)
    backup = {}
    with pytest.raises(ValueError, match="Insufficient supplied column support"):
        apply_surgical_patch(model, report, PatchConfig(alpha=2.0, top_k_matrices=1, mode=mode, top_k_columns=2, sparse_column_fraction=1.0), backup)
    assert backup == {}
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_late_nonfinite_materialization_is_rejected_before_copy(numerical_patch_case, monkeypatch, dtype):
    model, report = numerical_patch_case
    model.layers[0].mlp.down_proj.to(dtype=dtype)
    with torch.no_grad():
        model.layers[0].mlp.down_proj.weight.zero_()
        model.layers[0].mlp.down_proj.weight[0, 0] = torch.finfo(dtype).max
    report.ranked_culprits.append(CulpritMatrix("layers.0.mlp.down_proj", 0, [2, 2], 1.0, 1.0, 1.0, [0], [1.0]))
    report.layer_directions[0] = torch.tensor([1.0, 0.0])
    snapshots = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    original_copy = torch.Tensor.copy_
    copies = []

    def track_copy(tensor, source, *args, **kwargs):
        copies.append(tensor.data_ptr())
        return original_copy(tensor, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", track_copy)
    backup = {}
    try:
        with pytest.raises(ValueError, match="Nonfinite stored proposal"):
            apply_surgical_patch(model, report, PatchConfig(alpha=6.0, top_k_matrices=2, top_k_columns=1), backup)
        assert copies == []
        assert backup == {}
        assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                original_copy(parameter, snapshots[name])


@pytest.mark.parametrize("limit,value", [("max_alpha", 1.0), ("max_matrices", 1), ("max_distinct_layers", 1), ("max_touched_entries", 3), ("max_bitwise_changed_entries", 0), ("max_matrix_relative_frobenius_change", 1.99), ("max_model_relative_frobenius_change", 0.0)])
def test_declared_patch_budget_rejects_before_any_copy(numerical_patch_case, monkeypatch, limit, value):
    model, report = numerical_patch_case
    report.ranked_culprits.append(CulpritMatrix("layers.0.mlp.down_proj", 0, [2, 2], 1.0, 1.0, 1.0, [0], [1.0]))
    report.layer_directions[0] = torch.tensor([1.0, 0.0])
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    budget = patch_module.PatchBudget(6.0, 2, 2, 4, 3, 6.0, 6.0)
    budget = replace(budget, **{limit: value})

    def forbid_copy(*args, **kwargs):
        pytest.fail("Over-budget proposal reached live mutation")

    monkeypatch.setattr(torch.Tensor, "copy_", forbid_copy)
    backup = {}
    with pytest.raises(ValueError, match="declared budget"):
        apply_surgical_patch(model, report, PatchConfig(alpha=2.0, top_k_matrices=2, top_k_columns=1, budget=budget), backup)
    assert backup == {}
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before


def test_budgeted_patch_records_actual_global_norm_and_restores(numerical_patch_case):
    model, report = numerical_patch_case
    snapshots = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    budget = patch_module.PatchBudget(2.0, 1, 1, 2, 1, 2.0, 2.0)
    with pytest.raises(FrozenInstanceError):
        budget.max_alpha = 3.0
    backup = {}
    try:
        result = apply_surgical_patch(model, report, PatchConfig(alpha=2.0, top_k_matrices=1, top_k_columns=1, budget=budget), backup)
        baseline_squared = sum(float(torch.linalg.vector_norm(tensor.double()).item()) ** 2 for tensor in snapshots.values())
        delta_squared = sum(float(torch.linalg.vector_norm(parameter.detach().double() - snapshots[name].double()).item()) ** 2 for name, parameter in model.named_parameters())
        assert result.model_relative_frobenius_change == (delta_squared / baseline_squared) ** 0.5
        assert result.is_effective
    finally:
        restore_weights(model, backup)
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before


@pytest.mark.parametrize("changes", [{"max_alpha": float("inf")}, {"max_alpha": 7.0}, {"max_matrices": True}, {"max_distinct_layers": -1}, {"max_touched_entries": 1, "max_bitwise_changed_entries": 2}, {"max_matrix_relative_frobenius_change": float("nan")}, {"max_model_relative_frobenius_change": -1.0}])
def test_invalid_budget_declarations_fail_closed(changes):
    budget = patch_module.PatchBudget(2.0, 1, 1, 2, 1, 2.0, 2.0)
    with pytest.raises(ValueError):
        replace(budget, **changes)


def test_zero_budget_baseline_is_a_measured_noop(numerical_patch_case, monkeypatch):
    model, report = numerical_patch_case
    budget = PatchBudget(0.0, 0, 0, 0, 0, 0.0, 0.0)
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}

    def forbid_copy(*args, **kwargs):
        pytest.fail("Zero-budget baseline reached tensor mutation")

    monkeypatch.setattr(torch.Tensor, "copy_", forbid_copy)
    result = apply_surgical_patch(model, report, PatchConfig(alpha=0.0, top_k_columns=1, budget=budget))
    assert result.is_noop
    assert result.model_relative_frobenius_change == 0.0
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before


@pytest.mark.parametrize("kind", ["parameter", "buffer"])
def test_patch_rejects_storage_tied_to_unselected_state(numerical_patch_case, kind):
    model, report = numerical_patch_case
    alias = model.layers[1].mlp.down_proj.weight.detach()
    if kind == "parameter":
        model.register_parameter("untouched_alias", nn.Parameter(alias))
    else:
        model.register_buffer("untouched_alias", alias)
    before = {name: _tensor_sha256(tensor) for name, tensor in model.state_dict().items()}
    with pytest.raises(ValueError, match="share weight storage with another state tensor"):
        apply_surgical_patch(model, report, PatchConfig(alpha=2.0, top_k_matrices=1, top_k_columns=1))
    assert {name: _tensor_sha256(tensor) for name, tensor in model.state_dict().items()} == before


def test_synthetic_projection_name_is_not_a_canonical_target(numerical_patch_case):
    model, report = numerical_patch_case
    wrapper = nn.Module()
    wrapper.model = model
    before = {name: _tensor_sha256(parameter) for name, parameter in wrapper.named_parameters()}
    backup = {}
    try:
        with pytest.raises(ValueError, match="Unknown patch target: layers.1.mlp.down_proj"):
            apply_surgical_patch(wrapper, report, PatchConfig(alpha=2.0, top_k_matrices=1, top_k_columns=1), backup)
        assert {name: _tensor_sha256(parameter) for name, parameter in wrapper.named_parameters()} == before
        assert backup == {}
    finally:
        restore_weights(wrapper, backup)


def test_projection_discovery_returns_registered_nested_paths(numerical_patch_case):
    model, _report = numerical_patch_case
    wrapper = nn.Module()
    wrapper.model = model
    targets = find_output_projections(wrapper)
    assert set(targets) == {"model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"}
    for name, (layer_idx, module) in targets.items():
        assert wrapper.get_submodule(name) is module
        assert wrapper.state_dict()[name + ".weight"].data_ptr() == module.weight.data_ptr()
        assert module is model.layers[layer_idx].mlp.down_proj


@pytest.mark.parametrize("error", [RuntimeError("injected copy failure"), torch.OutOfMemoryError("injected OOM"), KeyboardInterrupt()])
def test_partial_copy_failure_restores_every_tensor(numerical_patch_case, monkeypatch, error):
    model, report = numerical_patch_case
    report.ranked_culprits.append(CulpritMatrix("layers.0.mlp.down_proj", 0, [2, 2], 1.0, 1.0, 1.0, [0], [1.0]))
    report.layer_directions[0] = torch.tensor([1.0, 0.0])
    snapshots = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    second_pointer = model.layers[0].mlp.down_proj.weight.data_ptr()
    original_copy = torch.Tensor.copy_
    failed = False

    def fail_once(tensor, source, *args, **kwargs):
        nonlocal failed
        if tensor.data_ptr() == second_pointer and not failed:
            failed = True
            raise error
        return original_copy(tensor, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_once)
    backup = {}
    try:
        with pytest.raises(type(error)):
            apply_surgical_patch(model, report, PatchConfig(alpha=2.0, top_k_matrices=2, top_k_columns=1), backup)
        assert failed
        assert backup == {}
        assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                original_copy(parameter, snapshots[name])


def test_post_copy_audit_failure_restores_original_bits(numerical_patch_case, monkeypatch):
    model, report = numerical_patch_case
    weight = model.layers[1].mlp.down_proj.weight
    snapshot = weight.detach().clone()
    original_hash = patch_module._tensor_sha256
    before = original_hash(weight)
    failed = False

    def fail_after_change(tensor):
        nonlocal failed
        digest = original_hash(tensor)
        if tensor.data_ptr() == weight.data_ptr() and digest != before and not failed:
            failed = True
            raise RuntimeError("injected post-copy audit failure")
        return digest

    monkeypatch.setattr(patch_module, "_tensor_sha256", fail_after_change)
    try:
        with pytest.raises(RuntimeError, match="post-copy audit"):
            apply_surgical_patch(model, report, PatchConfig(alpha=2.0, top_k_matrices=1, top_k_columns=1))
        assert failed
        assert original_hash(weight) == before
    finally:
        with torch.no_grad():
            weight.copy_(snapshot)


def test_patch_rejects_stale_or_aliased_baselines(numerical_patch_case):
    model, report = numerical_patch_case
    weight = model.layers[1].mlp.down_proj.weight
    name = report.ranked_culprits[0].module_name
    cfg = PatchConfig(alpha=2.0, top_k_matrices=1, top_k_columns=1)
    with pytest.raises(ValueError, match="aliases"):
        apply_surgical_patch(model, report, cfg, {name: weight.detach()})
    backup = {}
    try:
        apply_surgical_patch(model, report, cfg, backup)
        accepted_hash = _tensor_sha256(weight)
        with pytest.raises(ValueError, match="supplied baseline"):
            apply_surgical_patch(model, report, cfg, backup)
        assert _tensor_sha256(weight) == accepted_hash
    finally:
        restore_weights(model, backup)


@pytest.mark.parametrize("kind", ["missing", "shape", "dtype", "alias"])
def test_restore_never_silently_skips_invalid_backups(numerical_patch_case, kind):
    model, report = numerical_patch_case
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    name = report.ranked_culprits[0].module_name
    original = model.layers[1].mlp.down_proj.weight.detach().clone()
    if kind == "missing":
        name = "missing.target"
    elif kind == "shape":
        original = original[:1]
    elif kind == "dtype":
        original = original.to(dtype=torch.float16)
    else:
        original = model.layers[1].mlp.down_proj.weight.detach()
    with pytest.raises(RestorationError):
        restore_weights(model, {name: original})
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before


def test_shared_target_storage_is_rejected_before_mutation(numerical_patch_case):
    model, report = numerical_patch_case
    model.layers[0].mlp.down_proj.weight = model.layers[1].mlp.down_proj.weight
    report.ranked_culprits.append(CulpritMatrix("layers.0.mlp.down_proj", 0, [2, 2], 1.0, 1.0, 1.0, [0], [1.0]))
    report.layer_directions[0] = torch.tensor([1.0, 0.0])
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    with pytest.raises(ValueError, match="share weight storage"):
        apply_surgical_patch(model, report, PatchConfig(alpha=2.0, top_k_matrices=2, top_k_columns=1))
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before


def test_restoration_failure_is_fatal_not_a_rejected_candidate(numerical_patch_case, monkeypatch):
    model, report = numerical_patch_case
    weight = model.layers[1].mlp.down_proj.weight
    snapshot = weight.detach().clone()
    original_hash = patch_module._tensor_sha256
    original_copy = torch.Tensor.copy_
    baseline_hash = original_hash(weight)
    interrupted = False

    def interrupt_after_copy(tensor):
        nonlocal interrupted
        digest = original_hash(tensor)
        if tensor.data_ptr() == weight.data_ptr() and digest != baseline_hash and not interrupted:
            interrupted = True
            raise RuntimeError("injected audit interruption")
        return digest

    def prevent_restore(tensor, source, *args, **kwargs):
        if interrupted and tensor.data_ptr() == weight.data_ptr():
            raise RuntimeError("injected restoration failure")
        return original_copy(tensor, source, *args, **kwargs)

    monkeypatch.setattr(patch_module, "_tensor_sha256", interrupt_after_copy)
    monkeypatch.setattr(torch.Tensor, "copy_", prevent_restore)
    try:
        with pytest.raises(RestorationError, match="Fatal patch restoration"):
            apply_surgical_patch(model, report, PatchConfig(alpha=2.0, top_k_matrices=1, top_k_columns=1))
        assert interrupted
        assert original_hash(weight) != baseline_hash
    finally:
        with torch.no_grad():
            original_copy(weight, snapshot)
    assert original_hash(weight) == baseline_hash


@pytest.mark.parametrize("stage", ["generation", "health", "perplexity"])
@pytest.mark.parametrize("error_type", [RuntimeError, torch.OutOfMemoryError, KeyboardInterrupt])
def test_sweep_failure_restores_trial_weights(numerical_patch_case, monkeypatch, stage, error_type):
    model, report = numerical_patch_case
    snapshots = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    original_copy = torch.Tensor.copy_
    observed_trial = False

    def fail(*args, **kwargs):
        nonlocal observed_trial
        observed_trial = _tensor_sha256(model.layers[1].mlp.down_proj.weight) != before["layers.1.mlp.down_proj.weight"]
        raise error_type("injected evaluation failure")

    generate_fn = fail if stage == "generation" else lambda current: [[1, 2, 3]]
    if stage == "health":
        monkeypatch.setattr(verify_module, "detect_loop_onset", fail)
    monkeypatch.setattr(verify_module, "compute_sequence_perplexity", fail if stage == "perplexity" else lambda *args: 1.0)
    try:
        with pytest.raises(error_type, match="injected evaluation failure"):
            run_alpha_sweep(model, report, generate_fn, [], alphas=(2.0,), top_k_matrices=1, top_k_columns=1)
        assert observed_trial
        assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                original_copy(parameter, snapshots[name])


def test_successful_sweep_returns_evidence_without_accepting_edit(numerical_patch_case, monkeypatch):
    model, report = numerical_patch_case
    snapshots = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    observed = []

    def generate(current):
        edited = _tensor_sha256(current.layers[1].mlp.down_proj.weight) != before["layers.1.mlp.down_proj.weight"]
        observed.append(edited)
        return [[1, 2, 3]] if edited else [[1, 2, 3] * 3]

    monkeypatch.setattr(verify_module, "compute_sequence_perplexity", lambda *args: 1.0)
    try:
        result = run_alpha_sweep(model, report, generate, [], alphas=(0.0, 2.0), top_k_matrices=1, top_k_columns=1)
        assert observed == [False, True]
        assert result.best_alpha == 2.0
        assert result.candidates[1].patch_result.bitwise_changed_params_count > 0
        assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(snapshots[name])


@pytest.mark.parametrize("evaluation_fails", [False, True])
def test_sweep_restoration_failure_is_fatal(numerical_patch_case, monkeypatch, evaluation_fails):
    model, report = numerical_patch_case
    weight = model.layers[1].mlp.down_proj.weight
    snapshot = weight.detach().clone()
    before = _tensor_sha256(weight)
    original_copy = torch.Tensor.copy_
    evaluated = False

    def generate(current):
        nonlocal evaluated
        evaluated = True
        assert _tensor_sha256(weight) != before
        if evaluation_fails:
            raise RuntimeError("injected generation failure")
        return [[1, 2, 3]]

    def fail_restore(tensor, source, *args, **kwargs):
        if evaluated and tensor.data_ptr() == weight.data_ptr():
            raise RuntimeError("injected restore failure")
        return original_copy(tensor, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_restore)
    monkeypatch.setattr(verify_module, "compute_sequence_perplexity", lambda *args: 1.0)
    try:
        with pytest.raises(RestorationError, match="Fatal weight restoration"):
            run_alpha_sweep(model, report, generate, [], alphas=(2.0,), top_k_matrices=1, top_k_columns=1)
        assert evaluated
        assert _tensor_sha256(weight) != before
    finally:
        with torch.no_grad():
            original_copy(weight, snapshot)
    assert _tensor_sha256(weight) == before


@pytest.mark.parametrize("kind", ["module", "parameter", "storage"])
@pytest.mark.parametrize("evaluation_fails", [False, True])
def test_sweep_target_identity_change_is_fatal(numerical_patch_case, monkeypatch, kind, evaluation_fails):
    model, report = numerical_patch_case
    parent = model.layers[1].mlp
    original_module = parent.down_proj
    original_parameter = original_module.weight
    original_storage_view = original_parameter.detach()
    snapshot = original_parameter.detach().clone()
    before = _tensor_sha256(snapshot)
    observed_trial = False

    def generate(current):
        nonlocal observed_trial
        observed_trial = _tensor_sha256(original_parameter) != before
        if kind == "module":
            replacement = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                replacement.weight.copy_(original_parameter)
            parent.down_proj = replacement
        elif kind == "parameter":
            original_module.weight = nn.Parameter(original_parameter.detach().clone())
        else:
            original_parameter.data = original_parameter.detach().clone()
        if evaluation_fails:
            raise RuntimeError("injected error after identity replacement")
        return [[1, 2, 3]]

    monkeypatch.setattr(verify_module, "compute_sequence_perplexity", lambda *args: 1.0)
    try:
        with pytest.raises(RestorationError, match="identity"):
            run_alpha_sweep(model, report, generate, [], alphas=(2.0,), top_k_matrices=1, top_k_columns=1)
        assert observed_trial
        assert _tensor_sha256(original_storage_view) == before
    finally:
        original_parameter.data = original_storage_view
        original_module.weight = original_parameter
        parent.down_proj = original_module
        with torch.no_grad():
            original_parameter.copy_(snapshot)
    assert _tensor_sha256(original_parameter) == before


def test_sweep_tampered_baseline_is_never_adopted(numerical_patch_case, monkeypatch):
    model, report = numerical_patch_case
    weight = model.layers[1].mlp.down_proj.weight
    snapshot = weight.detach().clone()
    before = _tensor_sha256(snapshot)
    original_apply = verify_module.apply_surgical_patch

    def corrupt_handoff(*args, **kwargs):
        result = original_apply(*args, **kwargs)
        for baseline in kwargs["backup_store"].values():
            baseline.add_(1.0)
        return result

    monkeypatch.setattr(verify_module, "apply_surgical_patch", corrupt_handoff)
    monkeypatch.setattr(verify_module, "compute_sequence_perplexity", lambda *args: 1.0)
    try:
        with pytest.raises(RestorationError, match="baseline identity mismatch"):
            run_alpha_sweep(model, report, lambda current: [[1, 2, 3]], [], alphas=(2.0,), top_k_matrices=1, top_k_columns=1)
        assert _tensor_sha256(weight) != before
        assert not torch.equal(weight, snapshot + 1.0)
    finally:
        with torch.no_grad():
            weight.copy_(snapshot)
    assert _tensor_sha256(weight) == before


@pytest.mark.parametrize("alpha", [0.0, 2.0])
def test_sweep_unbacked_target_change_is_fatal(numerical_patch_case, monkeypatch, alpha):
    model, report = numerical_patch_case
    snapshots = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    before = {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}

    def generate(current):
        current.layers[0].mlp.down_proj.weight.add_(1.0)
        return [[1, 2, 3]]

    monkeypatch.setattr(verify_module, "compute_sequence_perplexity", lambda *args: 1.0)
    try:
        with pytest.raises(RestorationError, match="baseline"):
            run_alpha_sweep(model, report, generate, [], alphas=(alpha,), top_k_matrices=1, top_k_columns=1)
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(snapshots[name])
    assert {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()} == before


def test_sweep_projection_registry_change_is_fatal(numerical_patch_case, monkeypatch):
    model, report = numerical_patch_case
    snapshot = model.layers[1].mlp.down_proj.weight.detach().clone()
    before = _tensor_sha256(snapshot)

    def generate(current):
        current.layers[0].mlp.out_proj = nn.Linear(2, 2, bias=False)
        return [[1, 2, 3]]

    monkeypatch.setattr(verify_module, "compute_sequence_perplexity", lambda *args: 1.0)
    try:
        with pytest.raises(RestorationError, match="identity"):
            run_alpha_sweep(model, report, generate, [], alphas=(2.0,), top_k_matrices=1, top_k_columns=1)
        assert _tensor_sha256(model.layers[1].mlp.down_proj.weight) == before
    finally:
        if hasattr(model.layers[0].mlp, "out_proj"):
            del model.layers[0].mlp.out_proj
        with torch.no_grad():
            model.layers[1].mlp.down_proj.weight.copy_(snapshot)


@pytest.mark.parametrize("error_type", [RuntimeError, torch.OutOfMemoryError, KeyboardInterrupt])
def test_sweep_restoration_audit_interruption_is_fatal(numerical_patch_case, monkeypatch, error_type):
    model, report = numerical_patch_case
    weight = model.layers[1].mlp.down_proj.weight
    snapshot = weight.detach().clone()
    original_hash = verify_module._tensor_sha256
    before = original_hash(snapshot)
    evaluated = False

    def generate(current):
        nonlocal evaluated
        evaluated = True
        return [[1, 2, 3]]

    def interrupt_audit(tensor):
        if evaluated:
            raise error_type("injected restoration audit failure")
        return original_hash(tensor)

    monkeypatch.setattr(verify_module, "_tensor_sha256", interrupt_audit)
    monkeypatch.setattr(verify_module, "compute_sequence_perplexity", lambda *args: 1.0)
    try:
        with pytest.raises(RestorationError, match="identity audit/restoration failure"):
            run_alpha_sweep(model, report, generate, [], alphas=(2.0,), top_k_matrices=1, top_k_columns=1)
        assert evaluated
        assert original_hash(weight) != before
    finally:
        with torch.no_grad():
            weight.copy_(snapshot)
    assert original_hash(weight) == before


@pytest.mark.parametrize("kind", ["parameter_bytes", "parameter_object", "parameter_storage", "parameter_alias", "buffer_bytes", "buffer_object", "buffer_storage", "buffer_alias", "buffer_removed", "module"])
@pytest.mark.parametrize("evaluation_fails", [False, True])
def test_sweep_registered_model_state_change_is_fatal(numerical_patch_case, monkeypatch, kind, evaluation_fails):
    model, report = numerical_patch_case
    model.register_buffer("audit_counter", torch.arange(2, dtype=torch.int64))
    model.identity_marker = nn.Identity()
    original_marker = model.identity_marker
    original_parameter = model.embed.weight
    parameter_view = original_parameter.detach()
    parameter_snapshot = original_parameter.detach().clone()
    original_buffer = model.audit_counter
    buffer_view = original_buffer.detach()
    buffer_snapshot = original_buffer.clone()
    trial_weight = model.layers[1].mlp.down_proj.weight
    trial_snapshot = trial_weight.detach().clone()
    trial_hash = _tensor_sha256(trial_snapshot)
    evaluated = False

    def generate(current):
        nonlocal evaluated
        evaluated = True
        assert _tensor_sha256(trial_weight) != trial_hash
        if kind == "parameter_bytes":
            original_parameter.add_(1.0)
        elif kind == "parameter_object":
            current.embed.weight = nn.Parameter(parameter_snapshot.clone())
        elif kind == "parameter_storage":
            original_parameter.data = parameter_snapshot.clone()
        elif kind == "parameter_alias":
            current.register_parameter("extra_parameter", original_parameter)
        elif kind == "buffer_bytes":
            original_buffer.add_(1)
        elif kind == "buffer_object":
            current.audit_counter = buffer_snapshot.clone()
        elif kind == "buffer_storage":
            original_buffer.data = buffer_snapshot.clone()
        elif kind == "buffer_alias":
            current.register_buffer("extra_buffer", original_buffer)
        elif kind == "buffer_removed":
            del current.audit_counter
        else:
            current.identity_marker = nn.Identity()
        if evaluation_fails:
            raise RuntimeError("injected error after unrelated model state change")
        return [[1, 2, 3]]

    monkeypatch.setattr(verify_module, "compute_sequence_perplexity", lambda *args: 1.0)
    try:
        with pytest.raises(RestorationError, match="model|identity|baseline"):
            run_alpha_sweep(model, report, generate, [], alphas=(2.0,), top_k_matrices=1, top_k_columns=1)
        assert evaluated
        assert _tensor_sha256(trial_weight) == trial_hash
    finally:
        for name in ("extra_parameter", "extra_buffer"):
            if hasattr(model, name):
                delattr(model, name)
        model.identity_marker = original_marker
        model.embed.weight = original_parameter
        original_parameter.data = parameter_view
        model.audit_counter = original_buffer
        original_buffer.data = buffer_view
        with torch.no_grad():
            original_parameter.copy_(parameter_snapshot)
            original_buffer.copy_(buffer_snapshot)
            trial_weight.copy_(trial_snapshot)
    assert _tensor_sha256(original_parameter) == _tensor_sha256(parameter_snapshot)
    assert _tensor_sha256(original_buffer) == _tensor_sha256(buffer_snapshot)


def test_model_baseline_is_complete_canonical_and_immutable(numerical_patch_case):
    model, _report = numerical_patch_case
    model.register_parameter("embed_alias", model.embed.weight)
    model.register_buffer("signed_zero", torch.tensor([-0.0]), persistent=False)
    baseline = capture_model_baseline(model)
    expected_keys = {("parameter", name) for name, _parameter in model.named_parameters(remove_duplicate=False)} | {("buffer", "signed_zero")}
    assert {(tensor.kind, tensor.name) for tensor in baseline.tensors} == expected_keys
    signed_zero = next(tensor for tensor in baseline.tensors if tensor.name == "signed_zero")
    assert signed_zero.sha256 == hashlib.sha256(struct.pack("<f", -0.0)).hexdigest()
    assert signed_zero.shape == (1,)
    assert signed_zero.dtype == "torch.float32"
    assert len(baseline.sha256) == 64
    assert baseline == capture_model_baseline(model)
    assert baseline == ModelBaseline(tuple(reversed(baseline.tensors)))
    assert baseline.sha256 == ModelBaseline(tuple(reversed(baseline.tensors))).sha256
    with pytest.raises(FrozenInstanceError):
        baseline.encoding = "different"
    with pytest.raises(FrozenInstanceError):
        signed_zero.sha256 = "0" * 64
    with torch.no_grad():
        model.signed_zero.zero_()
    assert capture_model_baseline(model).sha256 != baseline.sha256


@pytest.mark.parametrize("change", [{"kind": "param"}, {"name": ""}, {"shape": [1]}, {"shape": (True,)}, {"shape": (-1,)}, {"dtype": "torch.complex64"}, {"sha256": "A" * 64}, {"sha256": "0" * 63}])
def test_model_baseline_refuses_malformed_tensor_identity(change):
    identity = BaselineTensorIdentity("buffer", "counter", (1,), "torch.int64", "0" * 64)
    with pytest.raises(ValueError):
        replace(identity, **change)


def test_model_baseline_refuses_duplicate_or_mutable_records():
    identity = BaselineTensorIdentity("buffer", "counter", (1,), "torch.int64", "0" * 64)
    for tensors in ((identity, identity), [identity], ()):
        with pytest.raises(ValueError):
            ModelBaseline(tensors)
    with pytest.raises(ValueError, match="encoding"):
        ModelBaseline((identity,), encoding="unversioned")


@pytest.mark.parametrize("kind", ["parameter_bytes", "buffer_bytes", "parameter_shape", "parameter_dtype", "alias_added", "buffer_removed"])
@pytest.mark.parametrize("alpha", [0.0, 2.0])
def test_declared_baseline_rejects_stale_model_before_any_copy(numerical_patch_case, monkeypatch, kind, alpha):
    model, report = numerical_patch_case
    model.register_buffer("audit_counter", torch.arange(2, dtype=torch.int64), persistent=False)
    baseline = capture_model_baseline(model)
    parameter = model.embed.weight
    original_view = parameter.detach()
    parameter_snapshot = parameter.detach().clone()
    original_buffer = model.audit_counter
    buffer_snapshot = original_buffer.clone()
    original_copy = torch.Tensor.copy_
    copied = []

    def refuse_copy(tensor, *args, **kwargs):
        copied.append(tensor)
        raise AssertionError("Live copy occurred before stale baseline rejection")

    backup = {}
    try:
        with torch.no_grad():
            if kind == "parameter_bytes":
                parameter.add_(1.0)
            elif kind == "buffer_bytes":
                original_buffer.add_(1)
            elif kind == "parameter_shape":
                parameter.data = parameter.reshape(2, 4)
            elif kind == "parameter_dtype":
                parameter.data = parameter.to(dtype=torch.float16)
            elif kind == "alias_added":
                model.register_parameter("extra_parameter", parameter)
            else:
                del model.audit_counter
        mutated_hashes = {name: _tensor_sha256(tensor) for name, tensor in model.state_dict().items()}
        monkeypatch.setattr(torch.Tensor, "copy_", refuse_copy)
        config = PatchConfig(alpha=alpha, top_k_matrices=1, top_k_columns=1, budget=PatchBudget(2.0, 1, 1, 2, 2, 2.0, 2.0), baseline=baseline)
        with pytest.raises(ValueError, match="baseline"):
            apply_surgical_patch(model, report, config, backup)
        assert copied == []
        assert backup == {}
        assert {name: _tensor_sha256(tensor) for name, tensor in model.state_dict().items()} == mutated_hashes
    finally:
        if hasattr(model, "extra_parameter"):
            del model.extra_parameter
        parameter.data = original_view
        model.audit_counter = original_buffer
        with torch.no_grad():
            original_copy(parameter, parameter_snapshot)
            original_copy(original_buffer, buffer_snapshot)


def test_declared_baseline_requires_explicit_budget(numerical_patch_case):
    model, report = numerical_patch_case
    config = PatchConfig(alpha=0.0, top_k_matrices=1, top_k_columns=1, baseline=capture_model_baseline(model))
    with pytest.raises(ValueError, match="budget"):
        apply_surgical_patch(model, report, config)


@pytest.mark.parametrize("alpha", [0.0, 2.0])
def test_declared_baseline_permitted_trial_restores_without_compounding(numerical_patch_case, alpha):
    model, report = numerical_patch_case
    baseline = capture_model_baseline(model)
    trial_weight = model.layers[1].mlp.down_proj.weight
    trial_snapshot = trial_weight.detach().clone()
    original_copy = torch.Tensor.copy_
    config = PatchConfig(alpha=alpha, top_k_matrices=1, top_k_columns=1, budget=PatchBudget(2.0, 1, 1, 2, 2, 2.0, 2.0), baseline=baseline)
    backup = {}
    try:
        result = apply_surgical_patch(model, report, config, backup)
        assert result.is_noop == (alpha == 0.0)
        if alpha != 0.0:
            edited_identity = capture_model_baseline(model)
            assert edited_identity != baseline
            with pytest.raises(ValueError, match="baseline"):
                apply_surgical_patch(model, report, config, backup)
            assert capture_model_baseline(model) == edited_identity
        else:
            assert backup == {}
        restore_weights(model, backup)
        assert capture_model_baseline(model) == baseline
        repeated = apply_surgical_patch(model, report, config, backup)
        assert repeated.matrix_metrics == result.matrix_metrics
    finally:
        try:
            restore_weights(model, backup)
        finally:
            with torch.no_grad():
                original_copy(trial_weight, trial_snapshot)
    assert capture_model_baseline(model) == baseline


@pytest.mark.parametrize("kind", ["nonfinite", "meta", "sparse", "byteorder"])
def test_model_baseline_refuses_unsupported_capture(numerical_patch_case, monkeypatch, kind):
    model, _report = numerical_patch_case
    original_parameter = model.embed.weight
    if kind == "nonfinite":
        model.embed.weight = nn.Parameter(torch.full_like(original_parameter, float("nan")))
    elif kind == "meta":
        model.embed.weight = nn.Parameter(torch.empty(original_parameter.shape, device="meta"))
    elif kind == "sparse":
        model.embed.weight = nn.Parameter(original_parameter.detach().to_sparse())
    else:
        monkeypatch.setattr(patch_module.sys, "byteorder", "big")
    try:
        with pytest.raises(ValueError):
            capture_model_baseline(model)
    finally:
        model.embed.weight = original_parameter


def test_patch_proposal_is_deeply_immutable_and_digest_bound(numerical_patch_case):
    model, report = numerical_patch_case
    baseline = capture_model_baseline(model)
    identity = next(tensor for tensor in baseline.tensors if tensor.name == "layers.1.mlp.down_proj.weight")
    target = PatchTarget(identity, 1, tuple(report.layer_directions[1].tolist()), tuple(report.ranked_culprits[0].top_column_indices))
    budget = PatchBudget(2.0, 1, 1, 2, 2, 2.0, 2.0)
    proposal = PatchProposal(baseline, budget, 2.0, (target,))
    digest = proposal.sha256
    assert len(digest) == 64
    assert replace(proposal, alpha=2).sha256 == digest
    assert replace(proposal, alpha=1.0).sha256 != digest
    assert replace(proposal, targets=(replace(target, columns=(1,)),)).sha256 != digest
    report.layer_directions[1].zero_()
    report.ranked_culprits[0].top_column_indices[0] = 1
    report.ranked_culprits[0].shape[0] = 99
    assert proposal.sha256 == digest
    assert target.direction == (1.0, 0.0)
    assert target.columns == (0,)
    with pytest.raises(FrozenInstanceError):
        proposal.alpha = 1.0
    with pytest.raises(FrozenInstanceError):
        target.columns = (1,)


@pytest.mark.parametrize("mode", ["top_k_columns", "sparse_columns", "rank1"])
def test_patch_proposal_freeze_applies_owned_exact_support(numerical_patch_case, mode):
    model, report = numerical_patch_case
    baseline = capture_model_baseline(model)
    trial_weight = model.layers[1].mlp.down_proj.weight
    snapshot = trial_weight.detach().clone()
    original_copy = torch.Tensor.copy_
    config = PatchConfig(alpha=2.0, top_k_matrices=1, mode=mode, top_k_columns=1, sparse_column_fraction=0.5, budget=PatchBudget(2.0, 1, 1, 4, 4, 2.0, 2.0), baseline=baseline)
    proposal = freeze_patch_proposal(report, config)
    expected_columns = (0, 1) if mode == "rank1" else (0,)
    assert proposal.targets[0].columns == expected_columns
    digest = proposal.sha256
    config.alpha = 0.0
    config.top_k_matrices = 0
    report.layer_directions[1].zero_()
    report.ranked_culprits[0].top_column_indices[0] = 1
    report.ranked_culprits.clear()
    backup = {}
    try:
        result = apply_patch_proposal(model, proposal, backup)
        assert result.is_effective
        assert result.baseline_sha256 == baseline.sha256
        assert result.proposal_sha256 == digest
        assert result.matrix_metrics[0].selected_columns == list(expected_columns)
        assert _bitwise_changed_entries(trial_weight[:, 1:], snapshot[:, 1:]) == 0
    finally:
        try:
            restore_weights(model, backup)
        finally:
            with torch.no_grad():
                original_copy(trial_weight, snapshot)
    assert capture_model_baseline(model) == baseline


@pytest.mark.parametrize("change", [{"layer_idx": True}, {"layer_idx": -1}, {"direction": [1.0, 0.0]}, {"direction": (0.0, 0.0)}, {"direction": (float("nan"), 0.0)}, {"direction": (1.0,)}, {"columns": [0]}, {"columns": ()}, {"columns": (0, 0)}, {"columns": (2,)}, {"columns": (True,)}])
def test_patch_proposal_refuses_invalid_target(change):
    identity = BaselineTensorIdentity("parameter", "layers.1.mlp.down_proj.weight", (2, 2), "torch.float32", "0" * 64)
    target = patch_module.PatchTarget(identity, 1, (1.0, 0.0), (0,))
    with pytest.raises(ValueError):
        replace(target, **change)


def test_patch_proposal_refuses_baseline_mismatch_and_mutable_targets(numerical_patch_case):
    model, report = numerical_patch_case
    config = PatchConfig(alpha=2.0, top_k_matrices=1, top_k_columns=1, baseline=capture_model_baseline(model), budget=PatchBudget(2.0, 1, 1, 2, 2, 2.0, 2.0))
    proposal = patch_module.freeze_patch_proposal(report, config)
    target = proposal.targets[0]
    with pytest.raises(ValueError, match="baseline"):
        replace(proposal, targets=(replace(target, parameter=replace(target.parameter, sha256="0" * 64)),))
    with pytest.raises(ValueError, match="immutable"):
        replace(proposal, targets=[target])
    with pytest.raises(ValueError, match="Duplicate"):
        replace(proposal, targets=(target, target))
    with pytest.raises(ValueError, match="alpha"):
        replace(proposal, alpha=3.0)
    with pytest.raises(ValueError, match="matrix/layer"):
        replace(proposal, budget=replace(proposal.budget, max_distinct_layers=0))
    with pytest.raises(ValueError, match="support"):
        replace(proposal, budget=replace(proposal.budget, max_touched_entries=1, max_bitwise_changed_entries=1))
    with pytest.raises(ValueError, match="dtype"):
        replace(proposal, working_dtype="torch.float64")


def test_patch_proposal_applies_different_exact_supports(numerical_patch_case):
    model, report = numerical_patch_case
    baseline = capture_model_baseline(model)
    identities = {tensor.name: tensor for tensor in baseline.tensors}
    targets = (patch_module.PatchTarget(identities["layers.1.mlp.down_proj.weight"], 1, (1.0, 0.0), (0,)), patch_module.PatchTarget(identities["layers.0.mlp.down_proj.weight"], 0, (1.0, 0.0), (1, 0)))
    proposal = patch_module.PatchProposal(baseline, PatchBudget(2.0, 2, 2, 6, 6, 2.0, 2.0), 2.0, targets)
    snapshots = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    original_copy = torch.Tensor.copy_
    backup = {}
    try:
        result = patch_module.apply_patch_proposal(model, proposal, backup)
        assert [metric.selected_columns for metric in result.matrix_metrics] == [[0], [1, 0]]
        assert result.modified_params_count == 6
        assert result.proposal_sha256 == proposal.sha256
    finally:
        try:
            restore_weights(model, backup)
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    original_copy(parameter, snapshots[name])
    assert capture_model_baseline(model) == baseline


def test_patch_proposal_zero_budget_is_noop_without_copy(numerical_patch_case, monkeypatch):
    model, report = numerical_patch_case
    baseline = capture_model_baseline(model)
    config = PatchConfig(alpha=0.0, top_k_matrices=1, top_k_columns=1, baseline=baseline, budget=PatchBudget(0.0, 0, 0, 0, 0, 0.0, 0.0))
    proposal = patch_module.freeze_patch_proposal(report, config)

    def refuse_copy(*args, **kwargs):
        pytest.fail("Zero-budget proposal made a live copy")

    monkeypatch.setattr(torch.Tensor, "copy_", refuse_copy)
    backup = {}
    result = patch_module.apply_patch_proposal(model, proposal, backup)
    assert result.is_noop
    assert backup == {}
    assert result.baseline_sha256 == baseline.sha256
    assert result.proposal_sha256 == proposal.sha256
    assert capture_model_baseline(model) == baseline


def test_patch_proposal_stored_budget_refuses_before_copy(numerical_patch_case, monkeypatch):
    model, report = numerical_patch_case
    baseline = capture_model_baseline(model)
    config = PatchConfig(alpha=2.0, top_k_matrices=1, top_k_columns=1, baseline=baseline, budget=PatchBudget(2.0, 1, 1, 2, 0, 2.0, 2.0))
    proposal = patch_module.freeze_patch_proposal(report, config)

    def refuse_copy(*args, **kwargs):
        pytest.fail("Over-budget proposal made a live copy")

    monkeypatch.setattr(torch.Tensor, "copy_", refuse_copy)
    backup = {}
    with pytest.raises(ValueError, match="support exceeds the declared budget"):
        patch_module.apply_patch_proposal(model, proposal, backup)
    assert backup == {}
    assert capture_model_baseline(model) == baseline


def test_patch_proposal_copy_interruption_restores_owned_baseline(numerical_patch_case, monkeypatch):
    model, report = numerical_patch_case
    baseline = capture_model_baseline(model)
    config = PatchConfig(alpha=2.0, top_k_matrices=1, top_k_columns=1, baseline=baseline, budget=PatchBudget(2.0, 1, 1, 2, 2, 2.0, 2.0))
    proposal = patch_module.freeze_patch_proposal(report, config)
    weight = model.layers[1].mlp.down_proj.weight
    snapshot = weight.detach().clone()
    original_copy = torch.Tensor.copy_
    interrupted = False

    def interrupt_once(tensor, source, *args, **kwargs):
        nonlocal interrupted
        result = original_copy(tensor, source, *args, **kwargs)
        if tensor.data_ptr() == weight.data_ptr() and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("injected proposal copy interruption")
        return result

    monkeypatch.setattr(torch.Tensor, "copy_", interrupt_once)
    backup = {}
    try:
        with pytest.raises(KeyboardInterrupt, match="proposal copy"):
            patch_module.apply_patch_proposal(model, proposal, backup)
        assert interrupted
        assert backup == {}
        assert capture_model_baseline(model) == baseline
    finally:
        with torch.no_grad():
            original_copy(weight, snapshot)


@pytest.mark.parametrize("alpha", [0.0, 2.0])
def test_frozen_patch_trial_restores_after_normal_exit(numerical_patch_case, alpha):
    model, report = numerical_patch_case
    baseline = capture_model_baseline(model)
    config = PatchConfig(alpha=alpha, top_k_matrices=1, top_k_columns=1, baseline=baseline, budget=PatchBudget(2.0, 1, 1, 2, 2, 2.0, 2.0))
    proposal = freeze_patch_proposal(report, config)
    weight = model.layers[1].mlp.down_proj.weight
    snapshot = weight.detach().clone()
    original_copy = torch.Tensor.copy_
    try:
        with patch_trial(model, proposal) as result:
            assert result.proposal_sha256 == proposal.sha256
            assert result.is_noop == (alpha == 0.0)
            assert (capture_model_baseline(model) == baseline) == (alpha == 0.0)
        assert capture_model_baseline(model) == baseline
    finally:
        with torch.no_grad():
            original_copy(weight, snapshot)


@pytest.mark.parametrize("error_type", [RuntimeError, torch.OutOfMemoryError, KeyboardInterrupt])
def test_frozen_patch_trial_restores_after_evaluation_error(numerical_patch_case, error_type):
    model, report = numerical_patch_case
    baseline = capture_model_baseline(model)
    config = PatchConfig(alpha=2.0, top_k_matrices=1, top_k_columns=1, baseline=baseline, budget=PatchBudget(2.0, 1, 1, 2, 2, 2.0, 2.0))
    proposal = freeze_patch_proposal(report, config)
    weight = model.layers[1].mlp.down_proj.weight
    snapshot = weight.detach().clone()
    original_copy = torch.Tensor.copy_
    try:
        with pytest.raises(error_type, match="frozen evaluation"):
            with patch_trial(model, proposal):
                assert capture_model_baseline(model) != baseline
                raise error_type("injected frozen evaluation failure")
        assert capture_model_baseline(model) == baseline
    finally:
        with torch.no_grad():
            original_copy(weight, snapshot)


def test_frozen_patch_trial_stale_baseline_never_yields(numerical_patch_case):
    model, report = numerical_patch_case
    baseline = capture_model_baseline(model)
    config = PatchConfig(alpha=2.0, top_k_matrices=1, top_k_columns=1, baseline=baseline, budget=PatchBudget(2.0, 1, 1, 2, 2, 2.0, 2.0))
    proposal = freeze_patch_proposal(report, config)
    parameter = model.embed.weight
    snapshot = parameter.detach().clone()
    original_copy = torch.Tensor.copy_
    try:
        with torch.no_grad():
            parameter.add_(1.0)
        stale_baseline = capture_model_baseline(model)
        with pytest.raises(ValueError, match="baseline"):
            with patch_trial(model, proposal):
                pytest.fail("Stale frozen proposal yielded an active trial")
        assert capture_model_baseline(model) == stale_baseline
    finally:
        with torch.no_grad():
            original_copy(parameter, snapshot)


@pytest.mark.parametrize("kind", ["unbacked", "parameter"])
def test_frozen_patch_trial_model_corruption_is_fatal(numerical_patch_case, kind):
    model, report = numerical_patch_case
    baseline = capture_model_baseline(model)
    config = PatchConfig(alpha=2.0, top_k_matrices=1, top_k_columns=1, baseline=baseline, budget=PatchBudget(2.0, 1, 1, 2, 2, 2.0, 2.0))
    proposal = freeze_patch_proposal(report, config)
    parameter = model.embed.weight
    snapshot = parameter.detach().clone()
    trial_weight = model.layers[1].mlp.down_proj.weight
    trial_snapshot = trial_weight.detach().clone()
    original_copy = torch.Tensor.copy_
    try:
        with pytest.raises(RestorationError, match="baseline|identity"):
            with patch_trial(model, proposal):
                if kind == "unbacked":
                    parameter.add_(1.0)
                else:
                    model.embed.weight = nn.Parameter(snapshot.clone())
        assert _tensor_sha256(trial_weight) == _tensor_sha256(trial_snapshot)
    finally:
        model.embed.weight = parameter
        with torch.no_grad():
            original_copy(parameter, snapshot)
            original_copy(trial_weight, trial_snapshot)
    assert capture_model_baseline(model) == baseline


def test_frozen_patch_trial_restoration_failure_is_fatal(numerical_patch_case, monkeypatch):
    model, report = numerical_patch_case
    baseline = capture_model_baseline(model)
    config = PatchConfig(alpha=2.0, top_k_matrices=1, top_k_columns=1, baseline=baseline, budget=PatchBudget(2.0, 1, 1, 2, 2, 2.0, 2.0))
    proposal = freeze_patch_proposal(report, config)
    weight = model.layers[1].mlp.down_proj.weight
    snapshot = weight.detach().clone()
    original_copy = torch.Tensor.copy_
    evaluated = False

    def fail_restore(tensor, source, *args, **kwargs):
        if evaluated and tensor.data_ptr() == weight.data_ptr():
            raise RuntimeError("injected frozen-context restoration failure")
        return original_copy(tensor, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_restore)
    try:
        with pytest.raises(RestorationError, match="Fatal weight restoration"):
            with patch_trial(model, proposal):
                evaluated = True
                raise RuntimeError("injected evaluation failure before fatal cleanup")
        assert _tensor_sha256(weight) != _tensor_sha256(snapshot)
    finally:
        with torch.no_grad():
            original_copy(weight, snapshot)
    assert capture_model_baseline(model) == baseline


def test_legacy_run_refuses_export_before_model_loading(tmp_path, monkeypatch, capsys):
    output = tmp_path / "never-created"
    diagnostic = tmp_path / "never-written.json"
    monkeypatch.setattr(sys, "argv", ["llm-loop-fix", "run", "--model", "not-a-model", "--prompts", "missing.jsonl", "--output-dir", str(output), "--report-out", str(diagnostic)])
    original_import = __import__

    def refuse_transformers(name, *args, **kwargs):
        if name == "transformers":
            pytest.fail("disabled run attempted to load model dependencies")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", refuse_transformers)
    with pytest.raises(SystemExit) as error:
        cli_main()
    assert error.value.code == 2
    assert "run is disabled" in capsys.readouterr().err
    assert not output.exists()
    assert not diagnostic.exists()


def test_end_to_end_top_k_columns_and_manifest(tmp_path):
    model = ToyLoopingCausalLM()
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
    assert loop_seq.numel() == 16
    assert clean_seq.tolist() == [2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]

    acts, analyses = collect_contrastive_activations(
        model, [loop_seq, clean_seq], prompt_lengths=[1, 1], min_ngram=3, min_repeats=3
    )
    assert analyses[0].is_looping is True
    assert analyses[1].is_looping is False

    report = localize_loop_culprits(model, acts)
    top_culprit = report.ranked_culprits[0]
    assert top_culprit.module_name == "layers.1.mlp.down_proj"
    assert top_culprit.top_column_indices[0] == 0

    # Test exact top_k_columns=1 patch and bitwise audit metrics
    backup = {}
    cfg = PatchConfig(
        alpha=2.0, top_k_matrices=1, mode="top_k_columns", top_k_columns=1
    )
    patch_res = apply_surgical_patch(model, report, cfg, backup_store=backup)
    assert patch_res.modified_modules == ["layers.1.mlp.down_proj"]
    assert patch_res.modified_params_count == 8  # 1 column * d_model=8
    assert patch_res.bitwise_changed_params_count > 0
    assert len(patch_res.matrix_metrics) == 1
    assert patch_res.matrix_metrics[0].pre_edit_sha256 != patch_res.matrix_metrics[0].post_edit_sha256

    fixed_out = model.greedy_generate([1], max_new_tokens=15)
    fixed_analysis = detect_loop_onset(fixed_out[1:], min_ngram=3, min_repeats=3)
    assert fixed_analysis.is_looping is False
    assert fixed_out[-1] == 15
    assert model.greedy_generate([2], max_new_tokens=15) == clean_seq.tolist()

    restore_weights(model, backup)

    # Test automated alpha sweep with paired rescue/regression tracking
    sweep = run_alpha_sweep(
        model=model,
        report=report,
        generate_fn=lambda m: [
            m.greedy_generate([1], max_new_tokens=15)[1:],
            m.greedy_generate([2], max_new_tokens=15)[1:],
        ],
        reference_sequences=[clean_seq],
        alphas=(0.0, 1.0, 2.0),
        top_k_matrices=1,
        mode="top_k_columns",
        top_k_columns=1,
    )
    assert sweep.baseline_loop_rate == 0.5
    assert sweep.best_loop_rate == 0.0
    assert sweep.best_rescued_count == 1
    assert sweep.best_regressed_count == 0

    # Verify manifest serialization and cryptographic validation
    manifest = build_edit_manifest(
        model_id="toy-looping-lm",
        config=cfg,
        patch_result=patch_res,
        localization=report,
        sweep=sweep,
    )
    manifest_path = tmp_path / "edit_manifest.json"
    write_edit_manifest(manifest_path, manifest)
    loaded = load_and_validate_manifest(manifest_path)
    assert loaded["summary"]["touched_params"] == 8
    assert loaded["verification"]["rescued_count"] == 1


def test_wp05_qwen_text_adapter_validates_contract_and_refuses_mismatches(tiny_qwen35_model):
    from llm_loop_fix.probe import inspect_qwen_text_adapter

    adapter = inspect_qwen_text_adapter(tiny_qwen35_model, position_convention="predictive_next_token")
    assert adapter.model_type == "qwen3_5"
    assert adapter.layer_prefix == "model.language_model.layers"
    assert adapter.num_layers == 2
    assert adapter.hidden_size == 16
    assert adapter.device == "cpu"
    assert adapter.position_convention == "predictive_next_token"
    assert adapter.backend == "pytorch_forward_hooks"
    assert adapter.projection_layers == {
        "model.language_model.layers.0.linear_attn.out_proj": 0,
        "model.language_model.layers.0.mlp.down_proj": 0,
        "model.language_model.layers.1.self_attn.o_proj": 1,
        "model.language_model.layers.1.mlp.down_proj": 1,
    }
    assert adapter.projection_shapes["model.language_model.layers.1.mlp.down_proj"] == (16, 32)
    assert all(dtype == "torch.float32" for dtype in adapter.weight_dtypes.values())

    untyped = ToyLoopingCausalLM().eval()
    with pytest.raises(ValueError, match="supported Qwen config.model_type"):
        inspect_qwen_text_adapter(untyped)

    with pytest.raises(ValueError, match="position convention"):
        inspect_qwen_text_adapter(tiny_qwen35_model, position_convention="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("convention", ["observed_token", "predictive_next_token"])
def test_wp05_explicit_convention_excludes_prompt_boundary_and_matches_reference(tiny_qwen35_model, convention):
    model = tiny_qwen35_model
    layers = find_transformer_layers(model)
    projs = find_output_projections(model)
    # Prompt length 3, 5 generated tokens (clean), plus a looping sequence with 2 clean gen tokens then 9 looping tokens
    clean_seq = torch.tensor([1, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64)
    loop_seq = torch.tensor([1, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9], dtype=torch.int64)

    raw_residuals = {0: [], 1: []}
    handles = [
        layers[0].register_forward_hook(lambda _m, _i, o: raw_residuals[0].append((o[0] if isinstance(o, tuple) else o).detach().float().cpu())),
        layers[1].register_forward_hook(lambda _m, _i, o: raw_residuals[1].append((o[0] if isinstance(o, tuple) else o).detach().float().cpu())),
    ]
    try:
        with torch.inference_mode():
            model(clean_seq.unsqueeze(0))
            model(loop_seq.unsqueeze(0))
    finally:
        for h in handles:
            h.remove()

    acts, analyses = collect_contrastive_activations(
        model, [clean_seq, loop_seq], [3, 3], convention=convention
    )
    assert acts.position_convention == convention
    assert len(analyses) == 2 and not analyses[0].is_looping and analyses[1].is_looping
    assert analyses[1].onset_index == 2
    assert acts.num_clean_prompts == 1 and acts.num_looping_prompts == 1
    assert set(acts.module_input_clean_sum) == set(projs)

    shift = 1 if convention == "predictive_next_token" else 0
    # Clean sequence: gen [0, 5) -> [3 - shift, 8 - shift) -> 5 tokens
    # Loop sequence: clean gen [0, 2) -> [3 - shift, 5 - shift) -> 2 tokens; loop gen [2, 11) -> [5 - shift, 14 - shift) -> 9 tokens
    assert acts.residual_clean_count == {0: 7, 1: 7}
    assert acts.residual_loop_count == {0: 9, 1: 9}
    for idx in (0, 1):
        expected_clean = (
            raw_residuals[idx][0][0, 3 - shift : 8 - shift].sum(dim=0)
            + raw_residuals[idx][1][0, 3 - shift : 5 - shift].sum(dim=0)
        )
        expected_loop = raw_residuals[idx][1][0, 5 - shift : 14 - shift].sum(dim=0)
        torch.testing.assert_close(acts.residual_clean_sum[idx], expected_clean, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(acts.residual_loop_sum[idx], expected_loop, atol=1e-6, rtol=1e-5)


def test_wp05_predictive_convention_refuses_zero_prompt_before_model_access(monkeypatch):
    model = ToyLoopingCausalLM().eval()

    def forbidden_eval():
        raise AssertionError("zero-prompt predictive probe must fail before touching model")

    monkeypatch.setattr(model, "eval", forbidden_eval)
    with pytest.raises(ValueError, match="Predictive span requires a preceding token context"):
        collect_contrastive_activations(
            model,
            [torch.tensor([1, 3, 4, 5, 6, 7, 8, 9])],
            [0],
            convention="predictive_next_token",
        )


def test_wp05_recorded_probe_uses_attached_spans_and_prompt_balancing(tiny_qwen35_model):
    from llm_loop_fix.detector import RepetitionSpec, observe_generation_health
    from llm_loop_fix.probe import collect_recorded_contrastive_activations
    from llm_loop_fix.records import GenerationRecord, SampleKey

    model = tiny_qwen35_model
    rep_spec = RepetitionSpec(min_period=3, max_period=12, min_repeats=3, max_tokens=64)
    profile_id = "b" * 64

    clean_rec = GenerationRecord(
        experiment_id="wp05-test",
        candidate_id="base",
        profile_id=profile_id,
        key=SampleKey("item-clean", 0, 11),
        prompt_tokens=2,
        prompt_token_ids=(1, 3),
        generated_token_ids=(4, 5, 6, 7, 8, 9),
        generated_text="clean",
        finish_reason="stop",
    )
    clean_rec = replace(
        clean_rec,
        health=observe_generation_health(clean_rec, thinking_required=False, repetition_spec=rep_spec),
    )

    loop_rec = GenerationRecord(
        experiment_id="wp05-test",
        candidate_id="base",
        profile_id=profile_id,
        key=SampleKey("item-loop", 0, 11),
        prompt_tokens=2,
        prompt_token_ids=(1, 3),
        generated_token_ids=(4, 5, 7, 8, 9, 7, 8, 9, 7, 8, 9),
        generated_text="loop",
        finish_reason="length",
    )
    loop_rec = replace(
        loop_rec,
        health=observe_generation_health(loop_rec, thinking_required=False, repetition_spec=rep_spec),
    )

    target_proj = "model.language_model.layers.1.mlp.down_proj"
    balanced = collect_recorded_contrastive_activations(
        model,
        [clean_rec, loop_rec],
        convention="predictive_next_token",
        target_layers=[1],
        target_projections=[target_proj],
        prompt_balanced=True,
    )
    assert balanced.position_convention == "predictive_next_token"
    assert balanced.num_clean_prompts == 1 and balanced.num_looping_prompts == 1
    assert set(balanced.residual_clean_sum) == {1}
    assert set(balanced.residual_loop_sum) == {1}
    assert set(balanced.module_input_clean_sum) == {target_proj}
    assert set(balanced.module_input_loop_sum) == {target_proj}
    # Prompt-balanced weighting: 2 sequences contribute clean spans (1 each) and 1 sequence contributes loop span
    assert balanced.residual_clean_count == {1: 2}
    assert balanced.residual_loop_count == {1: 1}
    assert balanced.module_input_clean_count == {target_proj: 2}
    assert balanced.module_input_loop_count == {target_proj: 1}

    token_weighted = collect_recorded_contrastive_activations(
        model,
        [clean_rec, loop_rec],
        convention="predictive_next_token",
        target_layers=[1],
        target_projections=[target_proj],
        prompt_balanced=False,
    )
    # Clean sequence has 6 generated tokens; loop sequence has 2 pre-loop tokens and 9 loop tokens
    assert token_weighted.residual_clean_count == {1: 8}
    assert token_weighted.residual_loop_count == {1: 9}

    # Missing health observation must fail closed without recomputing repetition
    unhealthy_rec = replace(clean_rec, health=None)
    with pytest.raises(ValueError, match="complete attached repetition observations"):
        collect_recorded_contrastive_activations(
            model, [unhealthy_rec], convention="observed_token"
        )
