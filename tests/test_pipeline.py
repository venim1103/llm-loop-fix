"""End-to-end unit tests for loop detection, over-projection patching, and provenance manifests."""

import hashlib
import struct
import pytest
import torch
import torch.nn as nn
import llm_loop_fix.patch as patch_module

from llm_loop_fix.detector import detect_loop_onset
from llm_loop_fix.localize import localize_loop_culprits, CulpritMatrix, LocalizationReport
from llm_loop_fix.manifest import (
    build_edit_manifest,
    load_and_validate_manifest,
    write_edit_manifest,
)
from llm_loop_fix.patch import PatchConfig, RestorationError, apply_surgical_patch, restore_weights, _tensor_sha256, _bitwise_changed_entries
from llm_loop_fix.probe import collect_contrastive_activations
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


def test_synthetic_projection_name_is_not_a_canonical_target(numerical_patch_case):
    model, report = numerical_patch_case
    wrapper = nn.Module()
    wrapper.model = model
    before = {name: _tensor_sha256(parameter) for name, parameter in wrapper.named_parameters()}
    backup = {}
    try:
        with pytest.raises(ValueError, match="canonical model path"):
            apply_surgical_patch(wrapper, report, PatchConfig(alpha=2.0, top_k_matrices=1, top_k_columns=1), backup)
        assert {name: _tensor_sha256(parameter) for name, parameter in wrapper.named_parameters()} == before
        assert backup == {}
    finally:
        restore_weights(wrapper, backup)


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
