"""End-to-end unit tests for loop detection, over-projection patching, and provenance manifests."""

import hashlib
from dataclasses import FrozenInstanceError, replace
import struct
import sys
import pytest
import torch
import torch.nn as nn
import llm_loop_fix.patch as patch_module
import llm_loop_fix.verify as verify_module

from llm_loop_fix import BaselineTensorIdentity, ModelBaseline, PatchBudget, capture_model_baseline
from llm_loop_fix.cli import main as cli_main
from llm_loop_fix.detector import detect_loop_onset
from llm_loop_fix.localize import localize_loop_culprits, CulpritMatrix, LocalizationReport
from llm_loop_fix.manifest import (
    build_edit_manifest,
    load_and_validate_manifest,
    write_edit_manifest,
)
from llm_loop_fix.patch import PatchConfig, RestorationError, apply_surgical_patch, restore_weights, _tensor_sha256, _bitwise_changed_entries
from llm_loop_fix.probe import collect_contrastive_activations, find_output_projections
from llm_loop_fix.verify import run_alpha_sweep


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
