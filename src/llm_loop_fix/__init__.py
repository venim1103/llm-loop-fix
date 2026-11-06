"""llm-loop-fix: Surgical weight-space de-looping for small language models."""

from .detector import LoopAnalysis, detect_loop_onset
from .localize import CulpritMatrix, LocalizationReport, localize_loop_culprits
from .patch import PatchConfig, PatchResult, apply_surgical_patch, restore_weights
from .probe import TrajectoryActivations, collect_contrastive_activations
from .verify import SweepCandidate, SweepResult, run_alpha_sweep

__all__ = [
    "LoopAnalysis",
    "detect_loop_onset",
    "TrajectoryActivations",
    "collect_contrastive_activations",
    "CulpritMatrix",
    "LocalizationReport",
    "localize_loop_culprits",
    "PatchConfig",
    "PatchResult",
    "apply_surgical_patch",
    "restore_weights",
    "SweepCandidate",
    "SweepResult",
    "run_alpha_sweep",
]
