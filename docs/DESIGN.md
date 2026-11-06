# Architecture & Mathematical Design (`llm-loop-fix`)

## 1. Problem Formulation

Let a language model consist of $L$ transformer blocks operating on a residual stream $h^{(\ell)}_t \in \mathbb{R}^d$ at layer $\ell \in \{1, \dots, L\}$ and token position $t$. Each block updates the residual stream via output projections from the Multi-Head Attention (`o_proj`, denoted $W_{\text{attn}}^{(\ell)}$) and gated MLP (`down_proj`, denoted $W_{\text{mlp}}^{(\ell)}$):

$$h^{(\ell)}_t = h^{(\ell-1)}_t + W_{\text{attn}}^{(\ell)} a^{(\ell)}_t + W_{\text{mlp}}^{(\ell)} m^{(\ell)}_t$$

In pathological looping trajectories, the residual stream enters a low-dimensional invariant subspace characterized by a dominant direction $u \in \mathbb{R}^d$ ($\|u\|_2 = 1$). When $h^{(\ell)}_t \cdot u$ exceeds a critical threshold, downstream attention heads repeatedly attend to the previous cycle's tokens and suppress the `<|endoftext|>` (EOS) logit.

---

## 2. Four-Stage Weight-Space Pipeline

### Stage 1: Loop Onset Detection (`detector.py`)
Given a generated token sequence $y = (y_1, \dots, y_T)$, we identify whether a trajectory is trapped in a loop and pinpoint the exact onset index $t_{\text{loop}}$:
- **Exact K-gram Repetition:** Detects when a contiguous span of length $k \in [4, 64]$ repeats at least $r \ge 3$ times consecutively or semi-consecutively.
- **Reasoning Prefix Recurrence:** Detects high-frequency repetition of transition phrases (e.g., `"Wait,"`, `"Let's recalculate"`, `"Alternatively,"`) combined with stagnant answer progression.

Tokens generated in $[t_{\text{loop}}, T]$ are labeled $\mathcal{S}_{\text{loop}}$, while productive tokens in non-looping trajectories (or pre-onset regions $[1, t_{\text{loop}} - w]$) are labeled $\mathcal{S}_{\text{clean}}$.

### Stage 2: Contrastive Direction & Culprit Localization (`probe.py` & `localize.py`)
1. **Residual Stream Capture:** Using PyTorch forward hooks, we collect mean-pooled hidden states at each layer $\ell$ for $\mathcal{S}_{\text{loop}}$ and $\mathcal{S}_{\text{clean}}$.
2. **Contrastive Direction ($u^{(\ell)}$):**
   $$\Delta \mu^{(\ell)} = \mathbb{E}_{t \in \mathcal{S}_{\text{loop}}}\left[h^{(\ell)}_t\right] - \mathbb{E}_{t \in \mathcal{S}_{\text{clean}}}\left[h^{(\ell)}_t\right], \quad u^{(\ell)} = \frac{\Delta \mu^{(\ell)}}{\|\Delta \mu^{(\ell)}\|_2}$$
   Optionally refined via the top principal component (SVD) of mean-centered differences to isolate the sharpest variance axis.
3. **Direct Weight Attribution:** Without running gradients, we measure how strongly each output matrix $W \in \mathbb{R}^{d_{\text{model}} \times d_{\text{in}}}$ (`mlp.down_proj` or `self_attn.o_proj`) aligns with $u^{(\ell)}$:
   - **Static Matrix Alignment Score:**
     $$S_{\text{static}}(W, u) = \frac{\|u^\top W\|_2}{\|W\|_F}$$
   - **Activation-Weighted Contribution Score:** Using recorded intermediate MLP/Attention activations $m_t \in \mathbb{R}^{d_{\text{in}}}$:
     $$S_{\text{act}}(W, u) = \mathbb{E}_{t \in \mathcal{S}_{\text{loop}}}\left[ u^\top (W m_t) \right] - \mathbb{E}_{t \in \mathcal{S}_{\text{clean}}}\left[ u^\top (W m_t) \right]$$
   - **Column-Level Attribution:** Each column $W_{:, j}$ represents the output vector written into the residual stream by intermediate neuron $j$. Its loop contribution is $c_j = (u^\top W_{:, j}) \cdot \Delta \bar{m}_j$.

### Stage 3: Surgical Weight Editing (`patch.py`)
We support two non-destructive, training-free weight modifications on the top-$K$ culprit matrices:

1. **Rank-1 Directional Projection:**
   Removes or attenuates the component of $W$ that writes along $u$:
   $$W_{\text{new}} = W - \alpha \, u (u^\top W)$$
   where $\alpha \in [0, 1.5]$ controls suppression strength ($\alpha = 1.0$ completely orthogonalizes $W$'s output column space against $u$).

2. **Sparse Top-Column Attenuation (Ultra-Surgical Mode):**
   Modifies only the top $p\%$ (e.g., $0.01\% - 0.1\%$) of columns $j \in \mathcal{J}_{\text{culprit}}$ with the highest positive loop contribution $c_j$:
   $$W_{\text{new}, :, j} = W_{:, j} - \alpha (u^\top W_{:, j}) u \quad \forall j \in \mathcal{J}_{\text{culprit}}$$
   This matches the MIT CSAIL Qwen3.5-0.8B finding where editing just $0.016\%$ of weights resolved loop traps without hurting general fluency.

### Stage 4: Closed-Loop Verification (`verify.py`)
Iterates over candidate scales $\alpha \in \{0.25, 0.5, 0.75, 1.0, 1.25\}$:
- Measures **Loop Rate** (% of prompts hitting `max_new_tokens` or triggering `detector.py`).
- Measures **Reference Perplexity Drift** ($\Delta \text{PPL}$) on clean reference text to ensure knowledge retention.
- Selects the Pareto-optimal $\alpha^*$ and exports the patched checkpoint or JSON diff.
