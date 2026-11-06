# llm-loop-fix

**Surgical Weight-Space De-Looping & Behavioral Repair Toolkit for Small Language Models**

Inspired by recent research on *Weight-Space Programming* (Takehi et al., MIT CSAIL, 2026), `llm-loop-fix` is a training-free, gradient-free Python toolkit that diagnoses and repairs pathological generation loops (such as endless reasoning chains, n-gram repetition traps, and EOS suppression) in small language models by directly analyzing and editing model weights.

---

## Core Mechanism

Small language models (0.5B–3B parameters) frequently possess latent domain knowledge (e.g., math or coding) but fail on benchmarks because their hidden state trajectories fall into cyclical attractors ("Wait, let me double check..." loops).

Rather than running expensive RL/SFT or heuristic decoding penalties that degrade reasoning quality, `llm-loop-fix` executes a 4-stage pipeline using only ~20–50 forward passes:

1. **Contrastive Trajectory Probing (`probe.py`):** Runs target prompts, automatically detects repetition onset via token n-gram / cyclical period analysis, and captures layer-wise residual stream activations for looping vs. terminating states.
2. **Repetition Direction Extraction (`localize.py`):** Computes the principal contrastive unit vector $u \in \mathbb{R}^{d_{\text{model}}}$ separating looping attractors from productive generation, and scores every MLP `down_proj` and Attention `o_proj` matrix by how strongly its columns write along $u$.
3. **Surgical Weight Patching (`patch.py`):** Applies a rank-1 directional projection or sparse column attenuation to the top culprit weight matrices (typically modifying $<0.05\%$ of total parameters):
   $$W_{\text{new}} = W - \alpha (u u^\top) W$$
4. **Automated Verification & Alpha Sweep (`verify.py`):** Evaluates patched weights across validation prompts to minimize loop rate while preserving baseline perplexity and coherent generation.

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
  --top-k-layers 2
```

### 2. Step-by-Step Diagnostic & Agentic Inspection

Generate a compact JSON diagnostic report (designed for human review or LLM agent weight-space reasoning):

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
├── README.md                 # Project overview & quickstart
├── docs/
│   ├── DESIGN.md             # Mathematical formulation & architecture design
│   └── PLAN.md               # Engineering roadmap & validation milestones
├── examples/
│   └── sample_prompts.jsonl  # Contrastive math & reasoning loop test prompts
├── src/
│   └── llm_loop_fix/
│       ├── __init__.py
│       ├── detector.py       # N-gram & phrase loop onset detection
│       ├── probe.py          # Forward-hook residual stream activation collector
│       ├── localize.py       # Contrastive vector extraction & weight matrix attribution
│       ├── patch.py          # Rank-1 projection & sparse column surgical weight editor
│       ├── verify.py         # Alpha sweep, loop rate & perplexity evaluator
│       └── cli.py            # Unified CLI entrypoint
└── tests/
    └── test_pipeline.py      # Unit tests on synthetic looping transformer architectures
```

## License

MIT
