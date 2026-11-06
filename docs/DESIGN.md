# Architecture & Mathematical Design (`llm-loop-fix`)

## 1. Problem Formulation & Research Grounding

Let a language model consist of $L$ transformer blocks operating on a residual stream $h^{(\ell)}_t \in \mathbb{R}^d$ at layer $\ell \in \{0, \dots, L-1\}$ and token position $t$. Each block updates the residual stream via output projections from Multi-Head Attention (`o_proj`, denoted $W_{\text{attn}}^{(\ell)}$) and gated MLP (`down_proj`, denoted $W_{\text{mlp}}^{(\ell)}$):

$$h^{(\ell)}_t = h^{(\ell-1)}_t + W_{\text{attn}}^{(\ell)} a^{(\ell)}_t + W_{\text{mlp}}^{(\ell)} m^{(\ell)}_t$$

In pathological looping trajectories (such as Qwen3.5-0.8B's thinking-mode loops documented by Takehi et al., MIT CSAIL, Nov 2026), the residual stream enters a cyclic attractor subspace characterized by a dominant layer-specific repetition unit direction $u^{(\ell)} \in \mathbb{R}^d$ ($\|u^{(\ell)}\|_2 = 1$). When $h^{(\ell)}_t \cdot u^{(\ell)}$ spikes, the model endlessly repeats reasoning transitions (or fails to close its `<think>` block before hitting the 32,768-token ceiling), resulting in empty extracted answers.

---

## 2. Key Findings from the Qwen3.5-0.8B Intervention Analysis

Detailed audit of the MIT CSAIL Experiment 8 weight-space intervention reveals five critical mechanics that our toolkit implements directly:

1. **Exact Top-$K$ Column Masking (`top_k_columns=64`):**
   In Qwen3.5-0.8B ($d_{\text{model}}=1024, d_{\text{ff}}=3584$), the intervention targeted exactly **2 matrices** (`layers.15.mlp.down_proj` and `layers.19.mlp.down_proj`) and masked **64 columns** per matrix:
   $$2 \times 64 \times 1024 = 131,072 \text{ parameters} \approx 0.016\% \text{ of } 800\text{M}$$
2. **Over-Projection Scaling ($\alpha > 1.0$, specifically $\alpha = 6.0$):**
   The exact update formula applied to each target matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is:
   $$W_{\text{new}} = W - \alpha \, u \left(\text{mask}_K \odot (u^\top W)\right)$$
   Because only 64 of the 3,584 columns are edited, setting $\alpha = 1.0$ only removes the repetition projection of those 64 neurons while leaving residual contributions from the remaining 3,520 columns intact. Using an **over-projection factor $\alpha \in [2.0, 8.0]$** (such as $\alpha = 6.0$ in the paper) turns those 64 neurons into active *anti-loop suppressors* when the loop context activates them.
3. **Failure Taxonomy Beyond N-Grams:**
   Real reasoning models fail via four distinct signatures that must be separated during probing and verification:
   - **Exact/Near K-Gram Loops:** Contiguous or high-density n-gram recurrence.
   - **Unclosed Thinking / Token-Cap Truncation:** Hitting `max_new_tokens` without emitting `</think>` or EOS.
   - **Whitespace / Control-Token Tail Degeneration:** Over-aggressive edits ($\alpha$ too high) can cause whitespace floods (observed in 51/1,152 practice items in the Qwen run).
4. **Paired Rescue vs. Regression Accounting:**
   Aggregate loop rates hide whether an edit broke previously working prompts. Every evaluation must track item-level transitions:
   - **Rescued:** Failed/looped in baseline $\rightarrow$ terminated cleanly in patched model.
   - **Regressed:** Terminated cleanly in baseline $\rightarrow$ looped or degenerated in patched model.
5. **Cryptographic Provenance & Bitwise BFloat16 Diff Auditing (Adapted from `micro_classifier`):**
   In `bfloat16` / `float16` checkpoints, small float32 updates can round back to the original value (e.g., 130,188 of 131,072 entries changed bitwise in Qwen3.5-0.8B). Our manifest generator (`manifest.py`) records SHA-256 tensor digests, relative Frobenius norms $\|W_{\text{new}} - W\|_F / \|W\|_F$, and exact bitwise-changed parameter counts in the target dtype.

---

## 3. Five-Stage Weight-Space Pipeline

### Stage 1: Multi-Mode Loop & Degeneration Detection (`detector.py`)
Given a generated token sequence $y = (y_1, \dots, y_T)$, `detect_loop_onset` classifies the trajectory into one of four states:
- `clean`: Terminated normally before `max_new_tokens` with no repetitive attractors or tail degeneration.
- `ngram_loop`: Contiguous periodic block or tail n-gram flood with exact onset index $t_{\text{loop}}$.
- `whitespace_tail`: Pathological run of whitespace/newline/pad tokens in the tail (used to reject over-scaled $\alpha$ values).
- `cap_truncation`: Reached `max_new_tokens` without EOS (or without closing `</think>`), using the final tail window $[T - w, T]$ as the contrastive loop region.

### Stage 2: Contrastive Activation Probing (`probe.py`)
Captures layer-wise residual stream vectors $h^{(\ell)}_t \in \mathbb{R}^{d_{\text{model}}}$ and pre-projection linear inputs $m^{(\ell)}_t \in \mathbb{R}^{d_{\text{in}}}$ across $\mathcal{S}_{\text{loop}}$ and $\mathcal{S}_{\text{clean}}$ token spans in a single forward pass per sequence.

### Stage 3: Contrastive Direction & Culprit Localization (`localize.py`)
1. Computes the unit repetition direction $u^{(\ell)} = \frac{\mu_{\text{loop}}^{(\ell)} - \mu_{\text{clean}}^{(\ell)}}{\|\mu_{\text{loop}}^{(\ell)} - \mu_{\text{clean}}^{(\ell)}\|_2}$ per layer.
2. Scores every residual-writing matrix (`mlp.down_proj`, `self_attn.o_proj`) by its activation-weighted directional contribution:
   $$c_j = (u^{(\ell)\top} W_{:, j}) \cdot (\bar{m}_{\text{loop}, j} - \bar{m}_{\text{clean}, j}), \quad S_{\text{act}}(W, u^{(\ell)}) = \sum_{j=1}^{d_{\text{in}}} c_j$$
3. Ranks columns by $c_j$ (or $|u^{(\ell)\top} W_{:, j}|$) to isolate the exact $K$ culprit columns (default $K = 64$).

### Stage 4: Surgical Weight Editing (`patch.py`)
Supports three intervention modes on the top-$M$ culprit matrices:
- `top_k_columns` (Default, $K=64$):
  $$W_{\text{new}} = W - \alpha \, u \left(\text{mask}_K \odot (u^\top W)\right)$$
- `sparse_columns` (Fractional column selection $p \in (0, 1]$)
- `rank1` (Full-matrix rank-1 projection across all columns)

Also computes exact `MatrixEditMetrics` per modified module (touched entries, bitwise-changed entries in target dtype, relative Frobenius norm, and SHA-256 before/after digests).

### Stage 5: Paired Rescue/Regression Sweep & Provenance Manifest (`verify.py` & `manifest.py`)
Sweeps across candidate scales $\alpha \in \{0.0, 1.0, 2.0, 4.0, 6.0\}$:
- Tracks **Loop Rate**, **Cap-Hit Rate**, **Whitespace Degeneration Count**, **Paired Rescues**, **Paired Regressions**, and **Clean Reference Perplexity**.
- Rejects any $\alpha$ that exceeds `max_perplexity_ratio`, introduces whitespace tail degeneration, or causes net regressions.
- Emits a strict, schema-validated JSON audit manifest (`llm-loop-fix/edit-manifest/1`) modeled after `micro_classifier/teacher/provenance.py`.
