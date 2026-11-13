"""Surgical top-K column, sparse fractional, and rank-1 weight patching with bitwise auditing."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import sys
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from .localize import CulpritMatrix, LocalizationReport
from .probe import find_output_projections


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Compute deterministic SHA-256 digest of a tensor's raw bytes on CPU."""
    cpu_contig = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu_contig.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _bitwise_changed_entries(original: torch.Tensor, edited: torch.Tensor) -> int:
    if original.shape != edited.shape or original.dtype != edited.dtype or original.device != edited.device:
        raise ValueError("stored-bit comparison requires identical shape, dtype and device")
    original_bytes = original.detach().contiguous().reshape(-1).view(torch.uint8).reshape(original.numel(), original.element_size())
    edited_bytes = edited.detach().contiguous().reshape(-1).view(torch.uint8).reshape(edited.numel(), edited.element_size())
    return int((original_bytes != edited_bytes).any(dim=-1).sum().item())


class RestorationError(RuntimeError):
    """Fatal failure to recover the original stored weight identity."""


@dataclass(frozen=True)
class BaselineTensorIdentity:
    kind: str
    name: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    def __post_init__(self) -> None:
        if self.kind not in ("parameter", "buffer") or not isinstance(self.name, str) or not self.name:
            raise ValueError("Baseline tensor requires a registered kind and name")
        if not isinstance(self.shape, tuple) or any(type(size) is not int or size < 0 for size in self.shape):
            raise ValueError("Baseline shape must be an immutable nonnegative integer tuple")
        if self.dtype not in ("torch.bfloat16", "torch.float16", "torch.float32", "torch.float64", "torch.int8", "torch.int16", "torch.int32", "torch.int64", "torch.uint8", "torch.bool"):
            raise ValueError("Unsupported baseline tensor dtype")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("Baseline tensor requires a lowercase SHA256 digest")


@dataclass(frozen=True)
class ModelBaseline:
    """Portable registered-tensor identity, not model or experiment provenance."""

    tensors: tuple[BaselineTensorIdentity, ...]
    encoding: str = "torch-contiguous-little-endian-v1"

    def __post_init__(self) -> None:
        if self.encoding != "torch-contiguous-little-endian-v1":
            raise ValueError("Unsupported baseline byte encoding")
        if not isinstance(self.tensors, tuple) or not self.tensors or any(not isinstance(tensor, BaselineTensorIdentity) for tensor in self.tensors):
            raise ValueError("Baseline requires immutable registered tensor identities")
        keys = [(tensor.kind, tensor.name) for tensor in self.tensors]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate baseline tensor registration")
        object.__setattr__(self, "tensors", tuple(sorted(self.tensors, key=lambda tensor: (tensor.kind, tensor.name))))

    @property
    def sha256(self) -> str:
        payload = {"encoding": self.encoding, "tensors": [{"kind": tensor.kind, "name": tensor.name, "shape": list(tensor.shape), "dtype": tensor.dtype, "sha256": tensor.sha256} for tensor in self.tensors]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def capture_model_baseline(model: nn.Module) -> ModelBaseline:
    """Hash registered aliases and nonpersistent buffers without retaining copies."""
    if sys.byteorder != "little":
        raise ValueError("Baseline raw encoding requires a little-endian host")
    identities = []
    for kind, registry in (("parameter", model.named_parameters(remove_duplicate=False)), ("buffer", model.named_buffers(remove_duplicate=False))):
        for name, tensor in registry:
            if tensor.layout != torch.strided or tensor.device.type == "meta":
                raise ValueError("Unsupported baseline tensor layout/device")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError("Nonfinite baseline tensor")
            identities.append(BaselineTensorIdentity(kind, name, tuple(tensor.shape), str(tensor.dtype), _tensor_sha256(tensor)))
    return ModelBaseline(tuple(identities))


@dataclass
class MatrixEditMetrics:
    """Detailed audit metrics for a single modified weight matrix."""

    module_name: str
    layer_idx: int
    shape: List[int]
    selected_columns: List[int]
    touched_entries: int
    bitwise_changed_entries: int
    relative_frobenius_change: float
    pre_edit_sha256: str
    post_edit_sha256: str
    baseline_frobenius_norm: Optional[float] = None
    stored_delta_frobenius_norm: Optional[float] = None


@dataclass(frozen=True)
class PatchBudget:
    """Explicit numerical limits; these are not utility acceptance thresholds."""

    max_alpha: float
    max_matrices: int
    max_distinct_layers: int
    max_touched_entries: int
    max_bitwise_changed_entries: int
    max_matrix_relative_frobenius_change: float
    max_model_relative_frobenius_change: float

    def __post_init__(self) -> None:
        for value in (self.max_alpha, self.max_matrix_relative_frobenius_change, self.max_model_relative_frobenius_change):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Budget numerical limits must be finite and nonnegative")
        if self.max_alpha > 6:
            raise ValueError("Budget alpha cannot exceed the supported [0, 6] envelope")
        for value in (self.max_matrices, self.max_distinct_layers, self.max_touched_entries, self.max_bitwise_changed_entries):
            if type(value) is not int or value < 0:
                raise ValueError("Budget counts must be nonnegative integers")
        if self.max_bitwise_changed_entries > self.max_touched_entries:
            raise ValueError("Budget changed-entry limit exceeds touched-entry limit")


@dataclass(frozen=True)
class PatchTarget:
    parameter: BaselineTensorIdentity
    layer_idx: int
    direction: tuple[float, ...]
    columns: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parameter, BaselineTensorIdentity) or self.parameter.kind != "parameter" or not self.parameter.name.endswith(".weight"):
            raise ValueError("Proposal target requires a registered baseline weight")
        if len(self.parameter.shape) != 2 or min(self.parameter.shape) < 1 or self.parameter.dtype not in ("torch.bfloat16", "torch.float16", "torch.float32"):
            raise ValueError("Proposal target requires a supported nonempty matrix")
        if type(self.layer_idx) is not int or self.layer_idx < 0:
            raise ValueError("Proposal target requires a nonnegative layer index")
        if not isinstance(self.direction, tuple) or len(self.direction) != self.parameter.shape[0] or any(type(value) not in (int, float) or not math.isfinite(value) for value in self.direction):
            raise ValueError("Proposal direction must be an immutable finite vector")
        working_direction = torch.tensor(self.direction, dtype=torch.float32, device="cpu")
        if not bool(torch.isfinite(working_direction).all()) or not math.isclose(float(torch.linalg.vector_norm(working_direction).item()), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("Proposal direction must be unit length in the working dtype")
        object.__setattr__(self, "direction", tuple(working_direction.tolist()))
        if not isinstance(self.columns, tuple) or not self.columns or any(type(column) is not int or not 0 <= column < self.parameter.shape[1] for column in self.columns) or len(set(self.columns)) != len(self.columns):
            raise ValueError("Proposal support must be immutable unique in-range columns")


@dataclass(frozen=True)
class PatchProposal:
    """Frozen numerical intervention; confirmation and profile binding are separate."""

    baseline: ModelBaseline
    budget: PatchBudget
    alpha: float
    targets: tuple[PatchTarget, ...]
    working_dtype: str = "torch.float32"

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, ModelBaseline) or not isinstance(self.budget, PatchBudget):
            raise ValueError("Proposal requires an immutable baseline and explicit budget")
        if self.working_dtype != "torch.float32":
            raise ValueError("Unsupported proposal working dtype")
        if type(self.alpha) not in (int, float) or not math.isfinite(self.alpha) or not 0 <= self.alpha <= self.budget.max_alpha:
            raise ValueError("Proposal alpha exceeds its finite declared budget")
        object.__setattr__(self, "alpha", float(self.alpha) if self.alpha else 0.0)
        if not isinstance(self.targets, tuple) or any(not isinstance(target, PatchTarget) for target in self.targets):
            raise ValueError("Proposal targets must be deeply immutable")
        baseline_parameters = {tensor.name: tensor for tensor in self.baseline.tensors if tensor.kind == "parameter"}
        names = [target.parameter.name for target in self.targets]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate proposal target")
        directions = {}
        for target in self.targets:
            if baseline_parameters.get(target.parameter.name) != target.parameter:
                raise ValueError("Proposal target does not match its complete baseline")
            digest = _tensor_sha256(torch.tensor(target.direction, dtype=torch.float32, device="cpu"))
            if target.layer_idx in directions and directions[target.layer_idx] != digest:
                raise ValueError("Proposal targets disagree on their layer direction")
            directions[target.layer_idx] = digest
        if self.alpha:
            if len(self.targets) > self.budget.max_matrices or len(directions) > self.budget.max_distinct_layers:
                raise ValueError("Proposal matrix/layer count exceeds its budget")
            if sum(target.parameter.shape[0] * len(target.columns) for target in self.targets) > self.budget.max_touched_entries:
                raise ValueError("Proposal intended support exceeds its budget")

    @property
    def sha256(self) -> str:
        budget = asdict(self.budget)
        for key in ("max_alpha", "max_matrix_relative_frobenius_change", "max_model_relative_frobenius_change"):
            budget[key] = float(budget[key])
        payload = {"encoding": "directional-projection-float32-v1", "baseline_sha256": self.baseline.sha256, "budget": budget, "alpha": self.alpha, "working_dtype": self.working_dtype, "targets": [{"parameter_name": target.parameter.name, "layer_idx": target.layer_idx, "direction": list(target.direction), "columns": list(target.columns)} for target in self.targets]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


@dataclass
class PatchConfig:
    """Configuration for surgical weight-space editing.

    Modes:
      - `top_k_columns`: Edit exactly `top_k_columns` (e.g. 64) per culprit matrix:
            W_new = W - alpha * outer(u, mask_K * (u @ W))
      - `sparse_columns`: Edit `sparse_column_fraction` of columns per culprit matrix.
      - `rank1`: Full rank-1 directional projection across all columns.
    - `declared_columns`: Use all supplied columns as exact immutable support.
    """

    alpha: float = 1.0
    top_k_matrices: int = 2
    mode: str = "top_k_columns"  # "top_k_columns", "sparse_columns", or "rank1"
    top_k_columns: int = 64
    sparse_column_fraction: float = 0.05
    budget: Optional[PatchBudget] = None
    baseline: Optional[ModelBaseline] = None


@dataclass
class PatchResult:
    """Metadata describing the applied weight edit across the model."""

    modified_modules: List[str]
    total_model_params: int
    modified_params_count: int
    bitwise_changed_params_count: int
    modified_fraction: float
    matrix_metrics: List[MatrixEditMetrics] = field(default_factory=list)
    model_relative_frobenius_change: Optional[float] = None
    baseline_sha256: Optional[str] = None
    proposal_sha256: Optional[str] = None

    @property
    def is_effective(self) -> bool:
        return self.bitwise_changed_params_count > 0

    @property
    def is_noop(self) -> bool:
        return not self.is_effective


def _validate_patch_config(config: PatchConfig) -> None:
    if isinstance(config.alpha, bool) or not isinstance(config.alpha, (int, float)) or not math.isfinite(config.alpha) or not 0 <= config.alpha <= 6:
        raise ValueError("Patch alpha must be finite and in [0, 6]")
    if type(config.top_k_matrices) is not int or config.top_k_matrices < 0:
        raise ValueError("Patch matrix count must be a nonnegative integer")
    if type(config.top_k_columns) is not int or config.top_k_columns < 1:
        raise ValueError("Patch column count must be a positive integer")
    if isinstance(config.sparse_column_fraction, bool) or not isinstance(config.sparse_column_fraction, (int, float)) or not math.isfinite(config.sparse_column_fraction) or not 0 < config.sparse_column_fraction <= 1:
        raise ValueError("Sparse column fraction must be finite and in (0, 1]")
    if config.mode not in ("rank1", "top_k_columns", "sparse_columns", "declared_columns"):
        raise ValueError(f"Unsupported patch mode: {config.mode}")
    if config.budget is not None:
        if not isinstance(config.budget, PatchBudget):
            raise ValueError("Patch budget must be an immutable PatchBudget")
        if config.alpha > config.budget.max_alpha:
            raise ValueError("Patch alpha exceeds the declared budget")
    if config.baseline is not None:
        if not isinstance(config.baseline, ModelBaseline):
            raise ValueError("Declared baseline must be an immutable ModelBaseline")
        if config.budget is None:
            raise ValueError("Baseline-bound patch requires an explicit numerical budget")


def freeze_patch_proposal(report: LocalizationReport, config: PatchConfig) -> PatchProposal:
    """Copy numerical inputs into frozen records without model mutation."""
    _validate_patch_config(config)
    if config.baseline is None or config.budget is None:
        raise ValueError("Freezing requires a declared baseline and numerical budget")
    baseline_parameters = {tensor.name: tensor for tensor in config.baseline.tensors if tensor.kind == "parameter"}
    targets = []
    for culprit in report.ranked_culprits[:config.top_k_matrices]:
        identity = baseline_parameters.get(culprit.module_name + ".weight")
        if identity is None or len(identity.shape) != 2 or min(identity.shape) < 1 or not isinstance(culprit.shape, list) or any(type(size) is not int for size in culprit.shape) or tuple(culprit.shape) != identity.shape:
            raise ValueError("Localization target does not match the declared baseline")
        direction = report.layer_directions.get(culprit.layer_idx)
        if not isinstance(direction, torch.Tensor) or not direction.is_floating_point() or direction.layout != torch.strided or direction.device.type == "meta" or direction.shape != (identity.shape[0],):
            raise ValueError("Missing or incompatible proposal direction")
        columns = culprit.top_column_indices
        if not isinstance(columns, list) or any(type(column) is not int or not 0 <= column < identity.shape[1] for column in columns) or len(set(columns)) != len(columns):
            raise ValueError("Invalid localization column support")
        if config.mode == "rank1":
            support = tuple(range(identity.shape[1]))
        else:
            count = len(columns) if config.mode == "declared_columns" else min(identity.shape[1], config.top_k_columns) if config.mode == "top_k_columns" else max(1, round(identity.shape[1] * config.sparse_column_fraction))
            if len(columns) < count:
                raise ValueError("Insufficient supplied support for the frozen proposal")
            support = tuple(columns[:count])
        targets.append(PatchTarget(identity, culprit.layer_idx, tuple(direction.detach().to(device="cpu", dtype=torch.float32).tolist()), support))
    return PatchProposal(config.baseline, config.budget, config.alpha, tuple(targets))


def apply_patch_proposal(model: nn.Module, proposal: PatchProposal, backup_store: Optional[Dict[str, torch.Tensor]] = None) -> PatchResult:
    """Apply a frozen numerical trial; acceptance/export are never implied."""
    if not isinstance(proposal, PatchProposal):
        raise ValueError("Expected an immutable PatchProposal")
    proposal_digest = proposal.sha256
    directions = {target.layer_idx: torch.tensor(target.direction, dtype=torch.float32, device="cpu") for target in proposal.targets}
    culprits = [CulpritMatrix(target.parameter.name.removesuffix(".weight"), target.layer_idx, list(target.parameter.shape), 0.0, 0.0, 0.0, list(target.columns), []) for target in proposal.targets]
    config = PatchConfig(alpha=proposal.alpha, top_k_matrices=len(proposal.targets), mode="declared_columns", budget=proposal.budget, baseline=proposal.baseline)
    result = apply_surgical_patch(model, LocalizationReport(directions, {}, culprits), config, backup_store)
    result.proposal_sha256 = proposal_digest
    return result


@torch.no_grad()
def apply_surgical_patch(
    model: nn.Module,
    report: LocalizationReport,
    config: PatchConfig,
    backup_store: Optional[Dict[str, torch.Tensor]] = None,
) -> PatchResult:
    """Apply atomically on Python failures; this is not repair acceptance."""
    _validate_patch_config(config)
    if config.baseline is not None and capture_model_baseline(model) != config.baseline:
        raise ValueError("Declared model baseline does not match registered tensor identity")
    targets = find_output_projections(model)
    state_tensors = model.state_dict(keep_vars=True)
    original_parameters = {name: module.weight for name, (_idx, module) in targets.items()}
    selected = report.ranked_culprits[:config.top_k_matrices]
    names = set()
    storage_ids = set()
    for culprit in selected:
        name = culprit.module_name
        if name in names:
            raise ValueError("Duplicate patch target")
        names.add(name)
        if name not in targets:
            continue
        weight = targets[name][1].weight
        storage_id = (weight.device, weight.untyped_storage().data_ptr())
        if storage_id in storage_ids:
            raise ValueError("Patch targets share weight storage")
        storage_ids.add(storage_id)
        for state_name, state_tensor in state_tensors.items():
            if state_name != name + ".weight" and isinstance(state_tensor, torch.Tensor) and (state_tensor.device, state_tensor.untyped_storage().data_ptr()) == storage_id:
                raise ValueError("Patch targets share weight storage with another state tensor")
        if backup_store is not None and name in backup_store:
            baseline = backup_store[name]
            if not isinstance(baseline, torch.Tensor) or baseline.shape != weight.shape or baseline.dtype != weight.dtype:
                raise ValueError("Backup does not identify the current baseline tensor")
            if baseline.device == weight.device and baseline.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr():
                raise ValueError("Backup aliases live weight storage")
            if _tensor_sha256(baseline) != _tensor_sha256(weight):
                raise ValueError("Patch must start from the immutable supplied baseline")
    owned_backup: Dict[str, torch.Tensor] = {}
    try:
        result = _apply_surgical_patch(model, report, config, owned_backup)
        if config.baseline is not None:
            result.baseline_sha256 = config.baseline.sha256
        if backup_store is not None:
            for name, baseline in owned_backup.items():
                backup_store.setdefault(name, baseline)
        return result
    except BaseException:
        try:
            for name in owned_backup:
                if model.get_submodule(name) is not targets[name][1] or targets[name][1].weight is not original_parameters[name]:
                    raise RestorationError("Target identity changed during the failed mutation")
            restore_weights(model, owned_backup)
        except BaseException as error:
            raise RestorationError("Fatal patch restoration failure; model is not validated") from error
        raise


def _materialize_stored_update(
    weight: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
    columns: Optional[List[int]] = None,
) -> torch.Tensor:
    working = weight.detach().to(dtype=torch.float32)
    direction = direction.to(device=weight.device, dtype=torch.float32)
    if columns is None:
        updated = working - alpha * torch.outer(direction, torch.mv(working.t(), direction))
    else:
        updated = working.clone()
        indices = torch.tensor(columns, device=weight.device, dtype=torch.long)
        selected = working[:, indices]
        updated[:, indices] = selected - alpha * torch.outer(direction, torch.mv(selected.t(), direction))
    return updated.to(dtype=weight.dtype)


@torch.no_grad()
def _apply_surgical_patch(
    model: nn.Module,
    report: LocalizationReport,
    config: PatchConfig,
    backup_store: Optional[Dict[str, torch.Tensor]] = None,
) -> PatchResult:
    """Apply directional projection or top-K column over-projection to culprit matrices.

    Implements the MIT CSAIL Qwen3.5-0.8B intervention formula:
        W_new = W_orig - alpha * outer(u, mask_K * (u @ W_orig))
    and records post-cast relative change as well as bitwise parameter changes
    after casting back to the module's target dtype (e.g. bfloat16/float16/float32).
    """
    proj_modules = find_output_projections(model)
    total_params = sum(p.numel() for p in model.parameters())
    modified_params = 0
    bitwise_changed_total = 0
    modified_names: List[str] = []
    metrics_list: List[MatrixEditMetrics] = []
    prepared = []
    delta_norm_squared = 0.0
    model_relative_change = None

    selected = report.ranked_culprits[: config.top_k_matrices]
    for culprit in selected:
        name = culprit.module_name
        if name not in proj_modules:
            raise ValueError(f"Unknown patch target: {name}")
        try:
            actual_module = model.get_submodule(name)
        except AttributeError as error:
            raise ValueError(f"Patch target lacks a canonical model path: {name}") from error
        if actual_module is not proj_modules[name][1]:
            raise ValueError(f"Patch target module identity mismatch: {name}")
        layer_idx, mod = proj_modules[name]
        weight = mod.weight
        if type(culprit.layer_idx) is not int or culprit.layer_idx != layer_idx:
            raise ValueError("Patch layer metadata does not match the target")
        if not isinstance(culprit.shape, list) or any(type(size) is not int for size in culprit.shape) or culprit.shape != list(weight.shape) or weight.ndim != 2 or min(weight.shape) < 1:
            raise ValueError("Patch shape metadata does not match the target")
        if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32) or weight.layout != torch.strided or weight.device.type == "meta":
            raise ValueError("Unsupported stored weight dtype/layout/device")
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("Nonfinite baseline weight")
        direction = report.layer_directions.get(layer_idx)
        if not isinstance(direction, torch.Tensor) or not direction.is_floating_point() or direction.shape != (weight.shape[0],) or direction.layout != torch.strided or direction.device.type == "meta":
            raise ValueError("Missing or incompatible patch direction")
        working_direction = direction.to(device=weight.device, dtype=torch.float32)
        if not bool(torch.isfinite(working_direction).all()) or not math.isclose(float(torch.linalg.vector_norm(working_direction).item()), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("Patch direction must be finite and unit length")
        columns = culprit.top_column_indices
        if not isinstance(columns, list) or any(type(column) is not int or not 0 <= column < weight.shape[1] for column in columns) or len(set(columns)) != len(columns):
            raise ValueError("Patch columns must be unique in-range integers")
        if config.mode != "rank1":
            required_columns = len(columns) if config.mode == "declared_columns" else min(weight.shape[1], config.top_k_columns) if config.mode == "top_k_columns" else max(1, round(weight.shape[1] * config.sparse_column_fraction))
            if required_columns < 1:
                raise ValueError("Declared support cannot be empty")
            if len(columns) < required_columns:
                raise ValueError("Insufficient supplied column support; no fallback is allowed")
    if config.alpha == 0:
        return PatchResult([], total_params, 0, 0, 0.0, model_relative_frobenius_change=0.0 if config.budget is not None else None)
    for culprit in selected:
        name = culprit.module_name
        layer_idx, mod = proj_modules[name]

        orig_weight = mod.weight.detach()
        if not bool(torch.isfinite(orig_weight).all()):
            raise ValueError("Nonfinite baseline weight")
        pre_sha = _tensor_sha256(orig_weight)
        d_out, d_in = orig_weight.shape

        if config.mode == "rank1":
            cols = list(range(d_in))
            touched = orig_weight.numel()
        elif config.mode in ("top_k_columns", "sparse_columns", "declared_columns"):
            if config.mode == "declared_columns":
                num_cols = len(culprit.top_column_indices)
            elif config.mode == "top_k_columns":
                num_cols = min(d_in, max(1, config.top_k_columns))
            else:
                num_cols = max(1, int(round(d_in * config.sparse_column_fraction)))

            cols = culprit.top_column_indices[:num_cols]

            touched = d_out * len(cols)
        else:
            raise ValueError(f"Unsupported patch mode: {config.mode}")

        w_new_cast = _materialize_stored_update(
            orig_weight, report.layer_directions[layer_idx], config.alpha,
            None if config.mode == "rank1" else cols,
        )
        if not bool(torch.isfinite(w_new_cast).all()):
            raise ValueError("Nonfinite stored proposal")
        stored_original = orig_weight.to(dtype=torch.float64)
        baseline_norm = float(torch.linalg.vector_norm(stored_original).item())
        delta_norm = float(torch.linalg.vector_norm(w_new_cast.to(dtype=torch.float64) - stored_original).item())
        rel_frob = delta_norm / baseline_norm if baseline_norm else (0.0 if delta_norm == 0.0 else float("inf"))
        if not math.isfinite(rel_frob):
            raise ValueError("Nonfinite stored perturbation metric")
        delta_norm_squared += delta_norm ** 2
        bitwise_diff = _bitwise_changed_entries(orig_weight, w_new_cast)

        post_sha = _tensor_sha256(w_new_cast)

        modified_params += touched
        bitwise_changed_total += bitwise_diff
        modified_names.append(name)
        metrics_list.append(
            MatrixEditMetrics(
                module_name=name,
                layer_idx=layer_idx,
                shape=[d_out, d_in],
                selected_columns=cols,
                touched_entries=touched,
                bitwise_changed_entries=bitwise_diff,
                relative_frobenius_change=rel_frob,
                pre_edit_sha256=pre_sha,
                post_edit_sha256=post_sha,
                baseline_frobenius_norm=baseline_norm,
                stored_delta_frobenius_norm=delta_norm,
            )
        )
        if bitwise_diff:
            prepared.append((mod, w_new_cast, metrics_list[-1]))

    if config.budget is not None:
        budget = config.budget
        if len(selected) > budget.max_matrices or len({culprit.layer_idx for culprit in selected}) > budget.max_distinct_layers:
            raise ValueError("Patch matrix/layer count exceeds the declared budget")
        if modified_params > budget.max_touched_entries or bitwise_changed_total > budget.max_bitwise_changed_entries:
            raise ValueError("Patch support exceeds the declared budget")
        if any(metrics.relative_frobenius_change > budget.max_matrix_relative_frobenius_change for metrics in metrics_list):
            raise ValueError("Patch matrix perturbation exceeds the declared budget")
        model_norm_squared = sum(float(torch.linalg.vector_norm(parameter.detach().to(dtype=torch.float64)).item()) ** 2 for parameter in model.parameters())
        if not math.isfinite(model_norm_squared) or not math.isfinite(delta_norm_squared):
            raise ValueError("Nonfinite model perturbation measurement")
        model_relative_change = math.sqrt(delta_norm_squared / model_norm_squared) if model_norm_squared else (0.0 if not delta_norm_squared else float("inf"))
        if not math.isfinite(model_relative_change) or model_relative_change > budget.max_model_relative_frobenius_change:
            raise ValueError("Patch model perturbation exceeds the declared budget")

    for mod, _stored_candidate, metrics in prepared:
        if backup_store is not None:
            backup_store[metrics.module_name] = mod.weight.detach().clone()
    for mod, stored_candidate, _metrics in prepared:
        mod.weight.copy_(stored_candidate)
    for mod, _stored_candidate, metrics in prepared:
        if _tensor_sha256(mod.weight) != metrics.post_edit_sha256:
            raise ValueError("Stored proposal hash differs after the complete copy batch")

    return PatchResult(
        modified_modules=modified_names,
        total_model_params=total_params,
        modified_params_count=modified_params,
        bitwise_changed_params_count=bitwise_changed_total,
        modified_fraction=modified_params / max(1, total_params),
        matrix_metrics=metrics_list,
        model_relative_frobenius_change=model_relative_change,
    )


@torch.no_grad()
def restore_weights(model: nn.Module, backup_store: Dict[str, torch.Tensor]) -> None:
    """Prevalidate all backups and verify exact restored bytes, or fail fatally."""
    if not backup_store:
        return
    try:
        proj_modules = find_output_projections(model)
        restores = []
        for name, orig_w in backup_store.items():
            if name not in proj_modules:
                raise RestorationError(f"Cannot restore missing target: {name}")
            _idx, mod = proj_modules[name]
            canonical = model.get_submodule(name)
            if canonical is not mod or not isinstance(orig_w, torch.Tensor) or orig_w.shape != mod.weight.shape or orig_w.dtype != mod.weight.dtype:
                raise RestorationError(f"Cannot restore incompatible target: {name}")
            if orig_w.device == mod.weight.device and orig_w.untyped_storage().data_ptr() == mod.weight.untyped_storage().data_ptr():
                raise RestorationError("Cannot restore from a live-alias backup")
            restores.append((mod, orig_w, _tensor_sha256(orig_w)))
        for mod, orig_w, expected_hash in restores:
            mod.weight.copy_(orig_w.to(device=mod.weight.device))
        for mod, _orig_w, expected_hash in restores:
            if _tensor_sha256(mod.weight) != expected_hash:
                raise RestorationError("Restored tensor bits do not match the baseline")
    except BaseException as error:
        raise RestorationError("Fatal weight restoration failure") from error
