# Project Roadmap & Execution Plan

## Phase 1: Core Infrastructure & DevContainer Setup (Completed)
- [x] Mathematical specification & architecture design (`docs/DESIGN.md`)
- [x] Token & phrase loop onset detector (`src/llm_loop_fix/detector.py`)
- [x] Forward-hook activation & residual stream probe (`src/llm_loop_fix/probe.py`)
- [x] Contrastive direction & matrix/column attribution engine (`src/llm_loop_fix/localize.py`)
- [x] Surgical rank-1 and sparse column weight patcher (`src/llm_loop_fix/patch.py`)
- [x] Automated alpha sweep & perplexity guard verifier (`src/llm_loop_fix/verify.py`)
- [x] Reproducible `.devcontainer` setup (CPU, CUDA/Docker, CUDA/Podman/WSL, CA trust, hash-locked deps, smoke tests)

## Phase 2: Exact MIT/Qwen3.5 Parity & `micro_classifier` Audit Discipline (Completed in Current Update)
- [x] **Exact Top-$K$ Column Masking & Over-Projection ($\alpha \in [1, 6]$):** Support exact `top_k_columns=64` surgical edits (`W_new = W - alpha * outer(u, mask_K * (u @ W))`) matching the 131,072-parameter Qwen3.5-0.8B intervention.
- [x] **Multi-Mode Failure & Degeneration Detector:** Detect unclosed `<think>` / token-ceiling truncation and guard against whitespace-tail degeneration caused by excessive $\alpha$.
- [x] **Paired Rescue vs. Regression Accounting:** Track item-level rescues and regressions across validation sweeps instead of coarse averages alone.
- [x] **Deterministic Edit Manifest & Bitwise BFloat16 Auditing (`manifest.py`):** Adopt `micro_classifier`'s strict JSON schema, SHA-256 tensor hashing, relative Frobenius change reporting, and bitwise dtype change counting.

## Phase 3: Real-Model Experiments & Agentic Weight-Space Loop (Next Steps)
- [ ] **GitHub Actions CI:** Add automated pytest + ruff + shellcheck workflow on push/PR.
- [ ] **Small Reasoning Model Live Validation:** Run `llm-loop-fix` inside the CUDA DevContainer on Qwen2.5-0.5B / Qwen3-0.6B / DeepSeek-R1-Distill-Qwen-1.5B on GSM8K & MATH practice prompts.
- [ ] **Lightweight Safetensors Delta Exporter:** Save/load only the modified 64-column slices (~256 KB diff) alongside the provenance manifest so patches can be distributed without re-uploading gigabyte checkpoints.
- [ ] **Attention Periodicity Head Probing:** Extend localization to `q_proj`/`k_proj` induction heads that lock onto cyclic periods.
