# MMWM Next-Step Integration Roadmap

This document translates high-impact components already present in this repository into a staged plan for MMWM.

## 1) Kendall Uncertainty-Weighted Multi-Task Loss (Highest ROI, lowest risk)

**Source reference:** `omnilatent/training/losses.py`

### Why first
- MMWM currently uses fixed scalar coefficients (`LossWeights` in `MMWM/losses.py`).
- Fixed weights are brittle when gradients across heads have different scales.

### Formulation
For task losses \(\{\mathcal{L}_i\}_{i=1}^K\), learn log-variances \(s_i = \log \sigma_i^2\):

\[
\mathcal{L}_{\text{kendall}} = \sum_{i=1}^K \left( \tfrac{1}{2} e^{-s_i} \mathcal{L}_i + \tfrac{1}{2} s_i \right)
\]

This yields dynamic weighting while regularizing the learned uncertainties.

### Integration points
- Add optional `learned_uncertainty: bool` flag to `WorldModelLoss`.
- Keep current fixed-weight path as fallback.
- Expose per-task \(\sigma_i\) in TensorBoard via `MMWM/monitoring.py` hooks.

### Complexity
- Time: \(O(K)\) extra scalar ops per step.
- Space: \(O(K)\) parameters.

---

## 2) Curriculum Training (5-phase transfer from OmniLatent)

**Source reference:** `curriculum_train.py`

### Why second
- MMWM has a standard loop in `MMWM/trainer.py`, but no difficulty schedule.
- Curriculum directly reduces early optimization instability.

### Practical schedule
1. **Latent reconstruction warmup** (short horizons, no hard negatives).
2. **Single-step transition fit** (action-conditioned next latent only).
3. **Short rollout consistency** (2–4 steps, memory active).
4. **Decoder alignment** (text/vector/image heads reintroduced with stronger supervision).
5. **Full objective + robustness perturbations** (masking/noise/domain shifts).

### Integration points
- Add phase config in `MMWM/config.py`.
- Add per-phase dataloader/loss-mask controls in `MMWM/trainer.py`.
- Persist phase metadata in checkpoints.

### Complexity
- Time: unchanged asymptotically, but fewer wasted steps in early epochs.
- Space: negligible metadata overhead.

---

## 3) Mamba SSM replacement for GRU memory

**Source reference:** `hpwm/components/temporal_state.py`

### Why third
- This is the largest architectural lift but best long-context payoff.
- Current recurrent memory can bottleneck long-range propagation.

### Target behavior
Replace recurrent update with selective state-space dynamics over latent sequence:

\[
x_{t+1} = A(\Delta_t) x_t + B(\Delta_t) u_t, \qquad y_t = C_t x_t + D_t u_t
\]

with input-conditioned discretization for selective retention.

### Integration points
- New `MMWM/components/memory_mamba.py` implementation.
- Adapter layer to preserve existing `MemoryState` interface in `MMWM/containers.py`.
- Backward-compat switch in model config.

### Complexity (expected)
- Time: near-linear in sequence length, \(O(Td)\)-style streaming behavior.
- Space: linear in state dimension and batch.

---

## 4) MoD Surprise Routing

**Source reference:** `hpwm/components/mod_router.py`

### Why fourth
- After memory stabilizes, route compute to surprising latent channels.
- Improves compute efficiency under fixed budget.

### Core signal
Use prediction error as surprise:

\[
r_t = \|\hat{z}_{t+1} - z_{t+1}\|_2
\]

and allocate expert depth or attention budget where \(r_t\) is high.

### Integration points
- Add `router_score` to transition auxiliary outputs.
- Gate optional expert blocks in decoder/transition using top-k surprise dimensions.
- Log routing entropy and utilization for collapse detection.

### Complexity
- Time: routing score \(O(d)\), top-k \(O(d \log k)\) or approximate selection.
- Space: small per-batch routing buffers.

---

## 5) Slot Attention + Token Merging for object-centric memory

**Source reference:** `hpwm/components/slot_encoder.py`

### Why fifth
- Best added after training stability and memory upgrades.
- Introduces stronger structured representations for scenes and multi-object dynamics.

### Formulation sketch
Given feature tokens \(\{f_n\}_{n=1}^N\) and slots \(\{s_k\}_{k=1}^K\), iterative soft assignment:

\[
a_{nk} = \mathrm{softmax}_k\left(\frac{q(s_k)^\top k(f_n)}{\sqrt{d}}\right),
\quad
s_k \leftarrow \mathrm{GRU}\left(s_k, \sum_n a_{nk} v(f_n)\right)
\]

then token merging reduces \(N\) before/after slot updates.

### Integration points
- New optional encoder path in `MMWM/encoders.py`.
- Keep current latent interface unchanged by projecting slots to fixed latent dims.

### Complexity
- Time: slot attention \(O(NK)\), reduced by token merge pre-pass.
- Space: \(O(NK)\) assignments.

---

## Suggested execution order
1. Kendall uncertainty weighting.
2. Curriculum training phases.
3. Mamba memory backend.
4. MoD surprise routing.
5. Slot attention + token merge.

This order maximizes incremental learning stability while minimizing simultaneous moving parts.
