# Plan and Implementation Audit (`PLAN_REVIEW.md`)

Date: 2026-11-06  
Audited Head: `4efac3c` (`fix(patch): contain mutation failures and verify restoration`)  
Scope: Independent review of `docs/PLAN.md`, `docs/DESIGN.md`, and current implementation skeletons (`patch.py`, `verify.py`, `probe.py`, `localize.py`, `detector.py`, `manifest.py`) to guide the next work-package slices.

---

## Executive Summary

The repository's current architecture and documentation (`docs/REVIEW.md`, `docs/DESIGN.md`, `docs/PLAN.md`, and `AGENTS.md`) establish a strong, evidence-gated structure:
- **WP00** (baseline, dual Python 3.10/3.12 CPU locks, CUDA lock, CI, and container builds) and **WP01** (exact BF16 published-artifact replay with verified SHA-256 hashes across modified and 486 untouched tensors) are cleanly completed and verified.
- **WP02** (records, complete-identity joins, math extraction/grading wrapper, strict fixed-argument IFEval controller, native vLLM output adapter) and **WP04** (raw BF16/FP16/FP32 byte hashing, post-cast float64 Frobenius norm auditing, canonical target preflight, and atomic mutation rollback) have solid offline test coverage through Section 23 of `docs/PLAN.md`.

However, cross-checking the remaining requirements in `docs/DESIGN.md` against the current code in `src/llm_loop_fix/` reveals **seven concrete technical and planning gaps** that should be addressed as WP04–WP08 proceed.

---

## Detailed Findings & Recommendations

### Finding 1: Evaluation-Time Exception Leak & Order-Dependent Baseline in `verify.py` (WP04 / WP07)

- **Location:** `src/llm_loop_fix/verify.py` (`run_alpha_sweep`, lines 93–199)
- **Issue:**
  1. **Uncaught Evaluation Exceptions Leave Mutated Weights Active:** While `apply_surgical_patch` (`4efac3c`) now rolls back cleanly if an error occurs *inside* the patch call, `run_alpha_sweep` executes `generate_fn(model)` (line 118) and `compute_sequence_perplexity(model, reference_sequences)` (line 138) *outside* any `try...finally` restoration block. If generation or perplexity evaluation raises an exception (OOM, timeout, keyboard interrupt, or bad output) on a candidate $\alpha > 0$, `run_alpha_sweep` exits with the trial weight mutation still live on `model`.
  2. **Final Selection Leaves Model Mutated Before Confirmation:** At lines 190–198, `run_alpha_sweep` automatically re-applies `best.alpha` to `model` as a side effect of running the sweep. This conflicts with `docs/DESIGN.md` Section 7 ("Final commit/export occurs only after confirmation and fresh-load checks, not as a side effect of choosing an alpha").
  3. **Grid-Order Baseline Assumption:** `baseline = candidates[0]` (line 167) and `if baseline_analyses is None: baseline_analyses = analyses` (line 130) assume `alphas[0] == 0.0`. If a caller passes a grid without `0.0` first (e.g., `(1.0, 2.0, 0.0)` or `(1.0, 2.0)`), baseline comparisons and perplexity ratios are computed against a mutated candidate.
- **Recommendation:**
  - Wrap the entire sweep body in `try...except BaseException` that calls `restore_weights(model, backup)` before re-raising, and always restore to the immutable baseline at the end of a development sweep unless an explicit `apply_winner: bool = False` flag is set.
  - Evaluate the explicit $\alpha = 0.0$ baseline first (independently of the user-supplied `alphas` ordering) and reject empty `reference_sequences` (`total_tokens == 0` in `compute_sequence_perplexity` currently returns `1.0` instead of raising an explicit error as required by `DESIGN.md` Section 3.1).

---

### Finding 2: Incomplete Proposal Prevalidation & Undocumented Fallback in `patch.py` (WP04)

- **Location:** `src/llm_loop_fix/patch.py` (`apply_surgical_patch` / `_apply_surgical_patch`, lines 78–235)
- **Issue:**
  1. **Missing Pre-Mutation Validation of Directions and Config:** In `_apply_surgical_patch`, `report.layer_directions[layer_idx]` is accessed inside the second loop (line 172) *after* earlier culprits in `selected` may have already been copied. Furthermore, it does not prevalidate that:
     - `config.alpha` is finite and within declared bounds,
     - `config.top_k_matrices >= 0`, `config.top_k_columns >= 1`, and `0.0 < config.sparse_column_fraction <= 1.0`,
     - `layer_idx` exists in `report.layer_directions`,
     - `u` has shape `(d_out,)`, is finite, and is a unit vector ($\|\|u\|_2 - 1.0\| \le \epsilon$).
  2. **Silent Fallback to Static Alignment (`u_t_w.abs()`):** At lines 190–193, if `num_cols > len(culprit.top_column_indices)`, `_apply_surgical_patch` silently falls back to `_, top_idxs = torch.topk(u_t_w.abs(), k=num_cols)`. This directly violates `docs/DESIGN.md` Section 6 ("do not silently replace signed ranking with absolute alignment or include nonpositive writers merely to fill K. Fewer than K supported columns leads to a recorded smaller candidate or abstention").
  3. **Duplicate / Out-of-Range Column Indices:** If `culprit.top_column_indices` contains duplicate or out-of-range indices, `w[:, col_tensor]` either crashes mid-mutation or under-counts unique support.
  4. **Cast-Erased No-Op Candidates:** When $\alpha > 0$ produces `bitwise_changed_params_count == 0` (all updates rounded to zero in BF16/FP16), `PatchResult` does not expose an explicit `is_effective` / `is_noop` status so the verifier can skip redundant re-evaluation.
- **Recommendation:**
  - Move *all* config, direction (`u`), shape, column-index uniqueness/range, and perturbation-budget checks into the preflight pass before any matrix is backed up or modified.
  - Remove the `u_t_w.abs()` fallback in `patch.py`; require `culprit.top_column_indices` to supply validated, unique, in-range indices (or fail closed).

---

### Finding 3: Canonical Module Path Mismatch Between `probe.py` and `patch.py` (WP04 / WP05 Bridge)

- **Location:** `src/llm_loop_fix/probe.py` (`find_transformer_layers` & `find_output_projections`, lines 30–72) vs. `src/llm_loop_fix/patch.py` (lines 155–159)
- **Issue:**
  - Commit `4efac3c` updated `patch.py` to enforce canonical submodule resolution: `actual_module = model.get_submodule(name)` and `if actual_module is not proj_modules[name][1]: raise ValueError(...)`.
  - However, `find_output_projections` in `probe.py` (line 70) still synthesizes names as `full_name = f"layers.{layer_idx}.{sub_name}"` regardless of where the layer list actually lives on `model`.
  - On any model where decoder layers live under `model.layers` (Llama, Qwen2.5) or `model.language_model.layers` (Qwen3.5-0.8B), `find_output_projections` returns `"layers.15.mlp.down_proj"`, which causes `model.get_submodule(name)` in `patch.py` and `restore_weights` to fail with `AttributeError` ("Patch target lacks a canonical model path").
- **Recommendation:**
  - Update `probe.py` (or the WP05 adapter) so `find_transformer_layers` returns both the canonical layer prefix (e.g., `"model.language_model.layers"`) and the layer list, or look up the exact canonical path from `dict(model.named_modules())`.
  - Remove the dangerous fallback in `find_transformer_layers` (lines 50–53) that grabs the first `nn.ModuleList` in `model.modules()`, as `DESIGN.md` Section 5 explicitly requires unsupported stacks to fail closed rather than guessing the first `ModuleList`.

---

### Finding 4: Unbounded $\mathcal{O}(L \cdot T \cdot d)$ Hook Memory Retention in `probe.py` (WP05)

- **Location:** `src/llm_loop_fix/probe.py` (`collect_contrastive_activations`, lines 99–188)
- **Issue:**
  - `make_layer_hook` and `make_proj_hook` currently copy and retain the full sequence activation tensor `hidden.detach().float().cpu()` (`[1, seq_len, d_model]`) and `x.detach().float().cpu()` (`[1, seq_len, d_in]`) for *every* layer and *every* target projection simultaneously during `model(input_ids)`, and only slice `clean_slice` / `loop_slice` after the forward pass completes.
  - As calculated in `docs/DESIGN.md` Section 9, for Qwen3.5-0.8B at 32K tokens, holding full-sequence float32 residuals and MLP inputs across all layers consumes ~13.5 GiB per sequence.
  - Additionally, `clean_slice = slice(max(0, p_len - 1), ...)` (line 139) includes the last prompt token `p_len - 1` as part of clean generated reasoning, which conflicts with `DESIGN.md` Section 5 ("avoid treating prompt-boundary tokens as clean generated reasoning").
- **Recommendation:**
  - Compute `clean_slice` and `loop_slice` *before* calling `model(input_ids)` (since `analysis` is already computed at line 125 before `model(input_ids)` at line 134!).
  - Pass those slices into the hook state so each hook immediately indexes `h[clean_slice].sum(dim=0)` and `h[loop_slice].sum(dim=0)` inside the hook and discards the full sequence tensor immediately. This reduces peak probe memory from $\mathcal{O}(L \cdot T \cdot d)$ to $\mathcal{O}(L \cdot d)$.

---

### Finding 5: Token-Weighted Imbalance & Protective-Neuron Ablation Hazard in `localize.py` (WP06)

- **Location:** `src/llm_loop_fix/localize.py` (`localize_loop_culprits`, lines 58–151)
- **Issue:**
  1. **Token-Pooled vs. Prompt-Balanced Direction Estimation:** `probe.py` and `localize.py` pool raw token sums across all prompts (`residual_loop_sum[idx] / residual_loop_count[idx]`). A single prompt that loops for 30,000 tokens has 100× the weight of a prompt that loops for 300 tokens, allowing one outlier sequence to dominate $u^{(\ell)}$. `DESIGN.md` Section 6 specifies prompt-balanced direction estimates.
  2. **Selection of Negative (Protective) Columns:** At line 131, `top_vals, top_idxs = torch.topk(col_contributions, k=k)` selects the top $k$ columns even when their signed contribution $c_j = (u^\top W_{:, j}) \Delta \bar{m}_j \le 0$. If a matrix has only 20 positive writer columns and $K=64$ is requested, `topk` selects 44 zero or *negative* (protective) columns and `patch.py` projects them out or reverses them.
  3. **Silent Static-Alignment Fallback:** Lines 126–128 fall back to `col_contributions = u_t_w.abs()` when module input activations are absent, and line 133 blends `static_score * 0.1` when `act_score <= 0`. `DESIGN.md` Section 6 requires positive signed writers and explicit abstention rather than silent static fallback.
- **Recommendation:**
  - Accumulate per-prompt normalized mean vectors (or equal-weighted prompt means) when computing $\mu_{\text{loop}}^{(\ell)}$ and $\mu_{\text{productive}}^{(\ell)}$.
  - Filter candidate columns strictly to `col_contributions > 0` before calling `topk`. If fewer than $K$ columns have positive contribution, either record the smaller supported set or abstain according to the experiment's `ProbeSpec` / `PatchProposal` policy.

---

### Finding 6: Decouple WP02 Math Pilot from the Quarantined IFEval Blocker in `docs/PLAN.md` (WP02)

- **Location:** `docs/PLAN.md` (Section 2 Dependency Map, WP02 description lines 85–101, and Section 19 lines 830–870)
- **Issue:**
  - Section 19 of `docs/PLAN.md` properly documents that full official IFEval execution is isolated-blocked on upstream NLTK `punkt` archive licensing clarity, `langdetect` Python-3 build review, and normalized-kwargs compatibility.
  - Meanwhile, the source math grading stack (`sympy`, `latex2sympy2_extended`, `word2number`, `timeout-decorator`) and timed math evaluation wrapper (`evaluation.py`) are already locked, tested, and verified in CI (`92f7e25`, `670969d`).
  - Keeping WP02 as a single monolithic gate risks blocking the bounded hardware pilot and MATH-500 / GSM8K behavioral reproduction on IFEval's third-party NLP resource licensing.
- **Recommendation:**
  - Split WP02 in `docs/PLAN.md` into:
    - **WP02a (Core Contracts, Hardware Pilot & Math Behavioral Reproduction):** Unblocked as soon as the live generation backend adapter is wired; gates WP05–WP07 for math repair discovery.
    - **WP02b (Out-of-Domain IFEval Generalization Reproduction):** Gates only cross-domain instruction-following claims once the NLTK/langdetect resource policy is resolved.

---

### Finding 7: Cross-Field Invariant Gaps in `manifest.py` (WP08)

- **Location:** `src/llm_loop_fix/manifest.py` (`validate_edit_manifest`, lines 142–154)
- **Issue:**
  - `validate_edit_manifest` currently checks only top-level key presence and `manifest_sha256` self-digest integrity.
  - As noted in `docs/PLAN.md` WP08 (lines 194–195), a recomputed `manifest_sha256` will happily validate a manifest with impossible internal counts (e.g., `bitwise_changed_params > touched_params`, `touched_params != sum(m.touched_entries)`, `pre_edit_sha256 == post_edit_sha256` when `bitwise_changed_entries > 0`, negative `alpha`, or missing base model revision/profile digests).
- **Recommendation:**
  - When executing WP08, add strict cross-field invariant checks in `validate_edit_manifest` verifying that matrix shapes, column indices, `touched_entries`, `bitwise_changed_entries`, and SHA-256 pre/post digests are mutually consistent with `summary` and `intervention`.

---

## Suggested Immediate Execution Order (Next Unblocked Slices)

1. **WP04 Slice A (Proposal Prevalidation & No-Op Policy in `patch.py`):**
   - Prevalidate `alpha`, finite unit directions `u`, unique in-range column indices, and non-fallback signed support before any tensor backup or copy.
   - Expose explicit no-op / cast-erased status when `bitwise_changed_params_count == 0`.
2. **WP04 Slice B (Evaluation-Time Exception Containment & Explicit Baseline in `verify.py`):**
   - Wrap `run_alpha_sweep` candidate evaluation in `try...finally: restore_weights(model, backup)`, evaluate $\alpha = 0.0$ independently of grid order, and reject empty reference sequences.
3. **WP03 Slice (Call-Site Health Record Consistency):**
   - Calibration fixtures for recovered repetition vs. pathological loops and call-site alignment between `detector.py`, `probe.py`, and `verify.py`.
4. **WP05 Slice (Canonical Qwen Text Adapter & Streaming Hook Aggregation in `probe.py`):**
   - Resolve canonical module names from root `model.named_modules()`, remove the arbitrary `ModuleList` fallback, and aggregate `clean_slice`/`loop_slice` sums inside the forward hooks to bound memory to $\mathcal{O}(L \cdot d)$.
