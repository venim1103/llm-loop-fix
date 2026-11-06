# llm-loop-fix

**Surgical Weight-Space De-Looping & Behavioral Repair Toolkit for Small Language Models**

Inspired by recent research on *Weight-Space Programming* (Takehi et al., MIT CSAIL, Nov 2026), `llm-loop-fix` is a training-free, gradient-free Python toolkit that diagnoses and repairs pathological generation loops (such as endless `<think>` reasoning chains, n-gram repetition traps, and token-ceiling truncation) in small language models by directly analyzing and editing model weights.

---

## Core Mechanism

Small language models (0.5B–3B parameters) frequently possess latent domain knowledge (e.g., math, logic, or instruction following) but fail on benchmarks because their hidden state trajectories fall into cyclical attractors (`"Wait, let me double check..."` loops) until hitting the token limit.

Rather than running expensive RL/SFT or heuristic decoding penalties, `llm-loop-fix` executes a 5-stage weight-space pipeline using only ~20–50 forward passes:

1. **Multi-Mode Loop & Degeneration Detection (`detector.py`):** Identifies contiguous/tail n-gram loops, unclosed `<think>` / token-ceiling truncations, and whitespace-tail degeneration.
2. **Contrastive Trajectory Probing (`probe.py`):** Captures layer-wise residual stream activations and pre-projection MLP/Attention inputs for looping vs. productive token spans.
3. **Repetition Direction & Column Attribution (`localize.py`):** Computes the unit repetition direction $u^{(\ell)} \in \mathbb{R}^{d_{\text{model}}}$ per layer and ranks every `mlp.down_proj` and `self_attn.o_proj` matrix and neuron column by how strongly it writes along $u^{(\ell)}$.
4. **Surgical Top-$K$ Column Over-Projection (`patch.py`):** Directly edits the top $K$ culprit columns (e.g., $K=64$ columns in 2 layers = 131,072 parameters, or $0.016\%$ of an 800M model) using scaled directional projection:
   $$W_{\text{new}} = W - \alpha \, u \left(\text{mask}_K \odot (u^\top W)\right)$$
   Supporting both orthogonal projection ($\alpha = 1.0$) and active anti-loop over-projection ($\alpha \in [2.0, 6.0]$, matching the MIT CSAIL Qwen3.5-0.8B intervention).
5. **Paired Rescue/Regression Sweep & Provenance Auditing (`verify.py`, `manifest.py`):** Sweeps candidate $\alpha$ values, tracks item-level **Rescues** vs. **Regressions**, guards against perplexity drift and whitespace degeneration, and emits a cryptographic SHA-256 + bitwise `bfloat16` edit manifest adapted from `micro_classifier`.

---

## Quickstart

### Installation

```bash
pip install -e .
```

### 1. Run Full End-to-End De-Looping Pipeline

```bash
llm-loop-fix run \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts examples/sample_prompts.jsonl \
  --output-dir ./patched_model \
  --mode top_k_columns \
  --top-k-columns 64 \
  --top-k-layers 2
```

### 2. Step-by-Step Diagnostic & Provenance Report

```bash
llm-loop-fix diagnose \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts examples/sample_prompts.jsonl \
  --report-out diagnostic_report.json
```

---

## Repository Structure

```text
llm-loop-fix/
├── .devcontainer/            # Reproducible CPU, CUDA/Docker, and CUDA/Podman/WSL containers
├── README.md                 # Project overview & quickstart
├── docs/
│   ├── DESIGN.md             # Mathematical formulation & Qwen3.5-0.8B intervention analysis
│   └── PLAN.md               # Engineering roadmap & validation milestones
├── examples/
│   └── sample_prompts.jsonl  # Contrastive math & reasoning loop test prompts
├── src/
│   └── llm_loop_fix/
│       ├── __init__.py
│       ├── detector.py       # N-gram loop, cap-truncation & whitespace-tail detector
│       ├── probe.py          # Forward-hook residual stream & module input collector
│       ├── localize.py       # Contrastive vector extraction & column attribution
│       ├── patch.py          # Top-K column over-projection & bitwise diff metrics
│       ├── verify.py         # Alpha sweep with paired rescue/regression accounting
│       ├── manifest.py       # Deterministic JSON provenance & SHA-256 tensor auditor
│       └── cli.py            # Unified CLI entrypoint
└── tests/
    └── test_pipeline.py      # Unit tests for localization, over-projection & manifests
```

## License

MIT
