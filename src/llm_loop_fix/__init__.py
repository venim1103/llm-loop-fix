"""llm-loop-fix: Surgical weight-space de-looping for small language models."""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import detector as detector, localize as localize, manifest as manifest, patch as patch, probe as probe, verify as verify
    from .detector import LoopAnalysis, detect_loop_onset
    from .localize import CulpritMatrix, LocalizationReport, localize_loop_culprits
    from .manifest import (
        MANIFEST_SCHEMA,
        build_edit_manifest,
        load_and_validate_manifest,
        validate_edit_manifest,
        write_edit_manifest,
    )
    from .patch import (
        BaselineTensorIdentity,
        MatrixEditMetrics,
        ModelBaseline,
        PatchBudget,
        PatchConfig,
        PatchProposal,
        PatchResult,
        PatchTarget,
        apply_patch_proposal,
        apply_surgical_patch,
        capture_model_baseline,
        freeze_patch_proposal,
        restore_weights,
    )
    from .probe import TrajectoryActivations, collect_contrastive_activations
    from .verify import SweepCandidate, SweepResult, patch_trial, run_alpha_sweep

__all__ = [
    "LoopAnalysis",
    "detect_loop_onset",
    "TrajectoryActivations",
    "collect_contrastive_activations",
    "CulpritMatrix",
    "LocalizationReport",
    "localize_loop_culprits",
    "BaselineTensorIdentity",
    "ModelBaseline",
    "capture_model_baseline",
    "MatrixEditMetrics",
    "PatchBudget",
    "PatchConfig",
    "PatchProposal",
    "PatchResult",
    "PatchTarget",
    "apply_patch_proposal",
    "apply_surgical_patch",
    "freeze_patch_proposal",
    "restore_weights",
    "SweepCandidate",
    "SweepResult",
    "patch_trial",
    "run_alpha_sweep",
    "MANIFEST_SCHEMA",
    "build_edit_manifest",
    "write_edit_manifest",
    "load_and_validate_manifest",
    "validate_edit_manifest",
]

_EXPORT_MODULES = {
    "LoopAnalysis": "detector",
    "detect_loop_onset": "detector",
    "TrajectoryActivations": "probe",
    "collect_contrastive_activations": "probe",
    "CulpritMatrix": "localize",
    "LocalizationReport": "localize",
    "localize_loop_culprits": "localize",
    "BaselineTensorIdentity": "patch",
    "ModelBaseline": "patch",
    "capture_model_baseline": "patch",
    "MatrixEditMetrics": "patch",
    "PatchBudget": "patch",
    "PatchConfig": "patch",
    "PatchProposal": "patch",
    "PatchResult": "patch",
    "PatchTarget": "patch",
    "apply_patch_proposal": "patch",
    "apply_surgical_patch": "patch",
    "freeze_patch_proposal": "patch",
    "restore_weights": "patch",
    "SweepCandidate": "verify",
    "SweepResult": "verify",
    "patch_trial": "verify",
    "run_alpha_sweep": "verify",
    "MANIFEST_SCHEMA": "manifest",
    "build_edit_manifest": "manifest",
    "write_edit_manifest": "manifest",
    "load_and_validate_manifest": "manifest",
    "validate_edit_manifest": "manifest",
}
_SUBMODULES = frozenset(_EXPORT_MODULES.values())


def __getattr__(name: str):
    if name in _SUBMODULES:
        value = _import_module("." + name, __name__)
        globals()[name] = value
        return value
    owner = _EXPORT_MODULES.get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module("." + owner, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
