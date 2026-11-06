# Project Roadmap & Execution Plan

## Phase 1: Core Infrastructure & Synthetic Validation (Completed)
- [x] Mathematical specification & architecture design (`docs/DESIGN.md`)
- [x] Token & phrase loop onset detector (`src/llm_loop_fix/detector.py`)
- [x] Forward-hook activation & residual stream probe (`src/llm_loop_fix/probe.py`)
- [x] Contrastive direction & matrix/column attribution engine (`src/llm_loop_fix/localize.py`)
- [x] Surgical rank-1 and sparse column weight patcher (`src/llm_loop_fix/patch.py`)
- [x] Automated alpha sweep & perplexity guard verifier (`src/llm_loop_fix/verify.py`)
- [x] CLI tool & unit tests on synthetic looping models (`tests/test_pipeline.py`)

## Phase 2: Real-Model Diagnostic & Agent-Assisted Loop Repair (Next Steps)
- [ ] **GitHub Actions CI:** Add automated pytest workflow on push/PR.
- [ ] **Multi-Architecture Adapter:** Ensure seamless layer discovery across Qwen2/2.5/3, Llama-3, Gemma-2/3, and SmolLM architectures.
- [ ] **Agentic Diagnostic JSON Exporter:** Emit compact weight summary tables (top-20 anomalous neuron columns, cosine similarities, EOS logit projections) so an LLM agent can inspect and prescribe custom weight edits directly.
- [ ] **Attention Head Sink Analysis:** Extend localization to `q_proj`/`k_proj` periodicity heads that lock onto cycle period $P$.

## Phase 3: Benchmark Evaluation Suite
- [ ] Evaluate loop rate & accuracy delta on GSM8K / MATH-500 subset with small reasoning models (0.5B–1.5B).
- [ ] Add Safetensors delta-patch export/import for lightweight sharing of surgical edits.
