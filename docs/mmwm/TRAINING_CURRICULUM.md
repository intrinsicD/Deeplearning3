# MMWM Agentic Training Curriculum

A staged plan for training the Multimodal Latent World Model on real datasets toward agentic use, with integrated critical review.

---

## Part 1: Critical Architecture Review

### Strengths

- **Modular registry design** (`interfaces.py`): swap any component via config string; no if/else branching.
- **Role-based latent decomposition** (sem/dyn/ctrl/mem): interpretable specialization with selective loss weighting.
- **Kendall learned uncertainty weighting** (`losses.py:39-50`): automatic multi-task balancing with regularizer floor clamp (line 155) to prevent anti-collapse from being disabled.
- **Adaptive curriculum scheduler** (`curriculum.py:104`): loss-plateau detection avoids manual phase boundary tuning.
- **Progressive transition cores**: MLP → GRU → AttnRes → RecurrentAttnRes → MoD routing — a clean complexity ladder.
- **Tool ecosystem skeleton** (`tools.py`): LatentRouter, ToolExecutionEngine, lazy-loaded tools provide an agentic inference path.
- **Comprehensive monitoring** (`monitoring.py`): losses, gradients, latent stats, text predictions, embeddings in TensorBoard.

### Critical Gaps (Blockers)

> **April 29, 2026 status correction:** Several items in the original blocker table below have since been implemented: checkpoint save/load, memory propagation through `train_step`, LR scheduling, LayerNorm-based projector normalization, sequence-level BPTT, gradient checkpointing for recurrent transitions, and contrastive alignment loss. A minimal deterministic trainer-compatible dataset now exists in `MMWM/data.py`, and audio is supported as an input modality.
>
> **July 7, 2026 update:** the vector/offline-RL adapter path is now reusable: `TransitionTupleDataset` handles D4RL-style mappings, `D4RLTransitionDataset` is an optional loader, and the Minari script uses the same shared batch contract. The structured smoke path now uses `GridWorldTransitionDataset` plus `run_gridworld_smoke()` to cover DataLoader -> train -> checkpoint save/load -> rollout metrics with a decreasing vector reconstruction metric. The still-active blockers are visual/text real dataset adapters (DM Control/TextWorld), text-action encoding, empirical validation on learnable environments, `LatentRouter` training, and full RL infrastructure.

| # | Gap | Location | Severity |
|---|-----|----------|----------|
| 1 | **No Dataset/DataLoader** — trainer accepts `DataLoader` but nothing produces the required batch dict format | `trainer.py:167`, `_to_packet():61-71` | BLOCKER |
| 2 | **Memory reset every step** — `memory_state=None` on every `train_step()` defeats all recurrent learning | `trainer.py:88` | BLOCKER |
| 3 | **No checkpoint save/load** — zero `torch.save`/`torch.load` in codebase; multi-stage training impossible | entire codebase | BLOCKER |
| 4 | **No sequence-level training (BPTT)** — single `(obs_t, action, obs_tp1)` per step; no unrolled training | `trainer.py:73-128` | BLOCKER for Stage 3+ |
| 5 | **LatentRouter untrained** — `ToolExecutionEngine.iterate()` runs under `@torch.no_grad()`, no loss signal | `tools.py:288` | BLOCKER for Stage 5 |
| 6 | **No LR scheduler** — only grad clipping; flat LR across all stages | `trainer.py:__init__` | MODERATE |
| 7 | **Toy encoders** — 3-layer CNN (32→64→128), bag-of-words text; won't scale to real data | `encoders.py:50-65`, `25-38` | MODERATE |
| 8 | **BatchNorm in latent projectors** — fails with small batches (RL, generation) | `components.py:42,75` | MODERATE |
| 9 | **No contrastive loss** — cross-modal alignment (Stage 4) has no implementation | `losses.py` | BLOCKER for Stage 4 |
| 10 | **No RL infrastructure** — no rollout buffer, advantage estimation, PPO, env interface | entire codebase | BLOCKER for Stage 5 |

---

## Part 2: Infrastructure Prerequisites (Stage 0)

**Must be completed before any real training. Estimated: 4-6 weeks for one developer.**

### 0A. Minimal Viable Training Pipeline (Priority 1-4, ~1 week)

1. **Checkpoint save/load** (~50 lines)
   - Save: model state_dict, optimizer state_dict, scheduler state, global_step, curriculum phase
   - Load: resume from checkpoint with full state recovery
   - Best-model tracking by validation loss

2. **TransitionTupleDataset** (~100 lines)
   - Generic wrapper: `(obs_t, action, obs_tp1)` → trainer's expected batch dict
   - Per-dataset adapters: start with D4RL (vector), DM Control (image), TextWorld (text)
   - Handles modality-specific preprocessing (tokenization, normalization, resizing)

3. **Fix memory propagation in train_step** (~20 lines)
   - Accept and return `memory_state` parameter
   - Detach at episode boundaries to prevent infinite BPTT
   - Keep original single-step path as fallback

4. **LR scheduler** (~10 lines)
   - `CosineAnnealingWarmRestarts` or `OneCycleLR`
   - Warmup for first 5% of steps per stage
   - Expose in Trainer constructor

### 0B. Replace BatchNorm with LayerNorm (Priority 5, ~5 lines)

Replace `nn.BatchNorm1d` in `components.py:42,75` with `nn.LayerNorm`. BatchNorm causes problems with small batch sizes (RL, inference), mixed datasets, and train/eval distribution shift.

### 0C. End-to-End Smoke Test (Priority 6, ~30 lines)

Extend `demo.py` into a structured synthetic environment:
- Simple gridworld with known dynamics
- Image = rendered grid, text = state description, vector = (x,y) position
- Verify: dataset → dataloader → train_step → checkpoint save → load → eval rollout → metrics decrease
- This catches integration bugs before touching real data

### 0D. Pre-trained Backbone Wrappers (Priority 9, ~1 week)

- `PretrainedVisionEncoder(ModalitySubEncoder)`: wraps CLIP ViT or DINOv2
  - CLS token extraction → linear projection to `hidden_dim`
  - Configurable freeze/unfreeze
- `PretrainedTextEncoder(ModalitySubEncoder)`: wraps sentence-transformers
  - Frozen embedding extraction → MLP projection
- Register as new encoder options in the registry

### 0E. Sequence-Level Training (Priority 7, ~1 week)

- `EpisodeDataset`: yields `(obs_seq[0..T], action_seq[0..T-1])` windows
- `Trainer.train_sequence_step(batch_sequence, window_length)`: BPTT with truncated backprop
- Detach memory at window boundaries
- Keep original `train_step()` for single-step stages

### 0F. Gradient Checkpointing (Priority 10)

Add `torch.utils.checkpoint.checkpoint()` in `RecurrentAttnResTransformerTransitionCore.forward()` at `transitions.py:184` — checkpoint each recurrent step for 4x memory savings in backward pass. Required before Stage 3 BPTT.

---

## Part 3: Training Curriculum

### Design Principles

**Reviewer-corrected decisions:**
1. **Commit to one encoder architecture throughout** — use `structured_multimodal` from the start; disable unused modality paths via data rather than swapping architectures. Eliminates weight transfer problems between stages.
2. **Limit to 2-3 datasets per stage** — drop complex-infra datasets (Habitat, WebArena, SWE-bench, NetHack) until approach is validated on simpler domains.
3. **Merge vector unimodal + transitions** — no reason to separate representation learning from transition learning for MLP-encoded vectors.
4. **Defer agentic RL (Stage 5) as a separate project** — requires building an entire RL framework; the world model must work first.

---

### Stage 1: Unimodal Representation + Single-Step Transitions (Weeks 5-9)

*(Weeks 1-4 consumed by Stage 0 infrastructure)*

**Goal:** Learn per-modality latent representations AND action-conditioned transitions jointly.

#### Phase 1A: Vector Domain (D4RL)

**Dataset:** D4RL (walker2d-medium-v2, hopper-medium-v2, halfcheetah-medium-v2)
- Easy to obtain: `pip install d4rl`, HDF5 download
- Clean (state, action, next_state) tuples with continuous actions
- Known baselines for comparison

**Config:**
```python
encoder_name = "structured_multimodal"   # use final architecture from start
transition_core_name = "mlp"             # simplest core
action_encoder_name = "mlp"
action_encoder_kwargs = {"action_dim": 6, "action_embed_dim": 128}  # match walker2d
learned_uncertainty = True
curriculum = relative_curriculum_phases(total_steps=30_000)
```

**Training:**
- Full (obs_t, action, obs_tp1) with latent prediction losses
- Phases 1-2 of curriculum (sem + dyn + ctrl ramp-up)
- Vector reconstruction decoder active throughout
- 30K steps, batch_size=256, lr=3e-4 with cosine warmup

**Success Criteria:**
- latent_sem_cosine > 0.85
- 5-step rollout mean MSE < 0.1
- Vector reconstruction R² > 0.90

#### Phase 1B: Visual Domain (DM Control Suite)

**Dataset:** DM Control Suite (cartpole-swingup, walker-walk, cheetah-run)
- Standard visual RL benchmark with Python API
- Continuous actions, 64x64 pixel observations
- Collect 100K transitions via random policy + replay buffer

**Config:**
```python
transition_core_name = "attnres_transformer"    # upgrade for visual complexity
decoder_configs = [("image_reconstruction", {"latent_dim": 512, "image_channels": 3, "image_size": 64})]
```

**Training:**
- Initialize from Phase 1A checkpoint (shared architecture)
- Curriculum phases 1-3 (gradual image decoder introduction)
- 50K steps, batch_size=64, lr=1e-4

**Success Criteria:**
- 5-step rollout latent cosine > 0.75
- Image reconstruction PSNR > 18 dB (realistic target for 128-dim bottleneck; 25 dB was unrealistic per reviewer)
- Latent z_sem variance > 0.1 (no collapse)

#### Phase 1C: Text Domain (TextWorld)

**Dataset:** TextWorld
- Programmatic generation via Python API — no download needed
- Natural language observations + text action commands
- Generate 50K (obs, action_text, next_obs) transitions

**Note:** Text actions are neither discrete IDs nor continuous vectors. Requires:
- Tokenize action text → encode via text sub-encoder → use as action representation
- Or: map actions to a discrete vocabulary and use `DiscreteActionEncoder`

**Config:**
```python
transition_core_name = "attnres_transformer"
decoder_configs = [("text_autoregressive_head", {...})]
```

**Training:**
- Text-specific curriculum: sem → dyn → text_ce decoder
- 30K steps, lr=3e-4

**Success Criteria:**
- Text perplexity < 80 on next-state descriptions (reviewer noted < 30 is unrealistic for tiny decoder)
- Semantic similarity between predicted and actual > 0.70

---

### Stage 2: Multi-Step Rollout & Memory (Weeks 9-14)

**Requires:** `EpisodeDataset` + `train_sequence_step` + gradient checkpointing (Stage 0E, 0F)

**Goal:** Train recurrent memory and accurate multi-step prediction via BPTT.

#### Phase 2A: Short-Horizon Memory (window=4-8)

**Dataset:** MiniGrid (MemoryS7, KeyCorridor, ObstructedMaze)
- Partially observable gridworlds requiring memory
- Simple image observations (7x7 grid rendered to 56x56)
- Discrete actions (7 actions)
- Python API: `pip install minigrid`

**Config:**
```python
memory_name = "gru"
transition_core_name = "recurrent_attnres_transformer"
curriculum = relative_curriculum_phases(total_steps=50_000)
```

**Training:**
- BPTT: start window=2, ramp to 4, then 8 over training
- Memory warmup: first 5K steps with window=2 only
- 50K steps total

**Success Criteria:**
- Rollout error growth rate < 2.0 at horizon=8
- Memory-dependent task accuracy > 60% (KeyCorridor requires remembering key location)
- `recurrent_steps_mean` from aux shows adaptive halting is being used (< recurrent_steps max)

#### Phase 2B: Long-Horizon Memory (window=16)

**Dataset:** Crafter (or continue with MiniGrid long-episode variants)
- Long-horizon survival with crafting dependency chains
- Requires remembering resource locations and recipes
- Diverse observation types

**Config:**
```python
memory_name = "mamba_ssm"                                        # upgrade for long-range
transition_core_name = "mod_recurrent_attnres_transformer"       # MoD routing
```

**Training:**
- Initialize from Phase 2A checkpoint
- BPTT window=16, gradient checkpointing enabled
- 100K steps, gradient accumulation (effective batch 128, micro-batch 32)

**Success Criteria:**
- Rollout error growth rate < 3.0 at horizon=16
- MoD routing metrics: `mod_surprise_mean` decreases over training, `mod_routed_dims` converges to ~30% of total dims

**Memory Budget Warning (from reviewer):**
> BPTT window=16 with MoD-RecurrentAttnRes means 16 × 4 × 6 = 384 transformer layer forward passes stored for backprop. With hidden_dim=512 and batch_size=32, this approaches A100 80GB limits. Gradient checkpointing is mandatory. Consider reducing batch size or window length if OOM.

---

### Stage 3: Cross-Modal Grounding (Weeks 14-18)

**Requires:** Contrastive alignment loss added to `losses.py` and `curriculum.py:_ALL_TASK_KEYS`

**Goal:** Align representations across modalities for transfer.

#### Phase 3A: Vision-Language Grounding

**Dataset:** ALFWorld
- Text instructions + visual observations (rendered scenes)
- Step-by-step interaction with paired modalities
- Moderate complexity, well-established benchmark

**New Loss Required:**
```python
# Add to losses.py: InfoNCE contrastive loss between text z_sem and image z_sem
# Add "contrastive_alignment_loss" to _ALL_TASK_KEYS in curriculum.py
# Integrate into WorldModelLoss.forward() with Kendall weighting
```

**Config:**
```python
encoder_name = "structured_multimodal"    # cross-modal attention fusion active
decoder_configs = [text_head, image_head]  # both active
# Full 5-phase curriculum with contrastive loss introduced in Phase 3
```

**Training:**
- Initialize from best Stage 2 checkpoint
- All modality paths active (text + image + vector proprioception)
- Contrastive alignment introduced at 0.25 weight, ramped to 1.0
- 80K steps

**Success Criteria:**
- Cross-modal retrieval (text → image) > 40% R@5
- Instruction-conditioned rollout accuracy > 50%
- Tri-modal latent cosine > 0.7 for aligned observations

---

### Stage 4: Agentic Tool Training (Weeks 18-22, or separate project)

**Reviewer recommendation: defer this stage until Stages 1-3 are validated. It requires building an entire RL framework from scratch.**

If proceeding:

#### Phase 4A: Supervised Tool Selection

**Prerequisite:** Make `ToolExecutionEngine.iterate()` differentiable (remove `@torch.no_grad()`), or collect traces for offline supervised learning.

**Expert Traces:** Hand-design heuristic oracles for each tool:
- `transition.internal`: always used when action is available
- `memory.read`: used when observation mentions previously-seen entity
- `memory.write`: used after significant state changes
- `decoder.text`: used when text output is requested

**Training:**
- Cross-entropy on expert action labels for LatentRouter
- 30K steps on oracle traces

#### Phase 4B: RL Fine-Tuning (requires external RL library)

**Recommendation:** Use CleanRL or Stable Baselines3 as RL backbone, write MMWM adapter.

---

## Part 4: Revised Compute Estimates

The original 1,250 A100-hours estimate was significantly low. Revised:

| Stage | GPU-hours (A100) | Notes |
|-------|------------------|-------|
| 0: Infrastructure | 0 | Code only |
| 1A: Vector (D4RL) | ~5 | Small model, vector-only |
| 1B: Visual (DM Control) | ~40 | AttnRes transformer on images |
| 1C: Text (TextWorld) | ~20 | Moderate text processing |
| 2A: Short memory | ~100 | BPTT window=8, RecurrentAttnRes |
| 2B: Long memory | ~400 | BPTT window=16, MoD routing, gradient checkpointing |
| 3A: Cross-modal | ~200 | Full multimodal, all decoders |
| 4: Agentic (if done) | ~500+ | RL overhead, environment interaction |
| **Total (Stages 0-3)** | **~765** | |
| **Total (with Stage 4)** | **~1,265+** | |

**Note:** These estimates assume single-GPU A100 80GB. Scaling to hidden_dim > 512 or adding distributed training (Stage 6C of original plan) would require 5,000-15,000 GPU-hours.

---

## Part 5: Implementation Priority Ordering

From the critical review, prioritized by impact and dependency:

| Priority | Task | Effort | Blocks | Status |
|----------|------|--------|--------|--------|
| 1 | Checkpoint save/load | ~50 lines | Everything | Done |
| 2 | TransitionTupleDataset (D4RL adapter) | ~100 lines | All training | Done: reusable tuple wrapper, D4RL-style mapping support, optional D4RL loader, shared Minari wrapper |
| 3 | Fix memory propagation in train_step | ~20 lines | Stage 2+ | Done |
| 4 | LR scheduler | ~10 lines | Training quality | Done |
| 5 | Replace BatchNorm -> LayerNorm | ~5 lines | RL, small batches | Done |
| 6 | End-to-end smoke test | ~30 lines | Confidence | Done: structured gridworld dataset plus train/checkpoint/load/rollout smoke test |
| 7 | EpisodeDataset + train_sequence_step | ~200 lines | Stage 2 | Partial: `train_sequence_step` exists; reusable `EpisodeDataset` remains open |
| 8 | Gradient checkpointing | ~20 lines | Stage 2B | Done |
| 9 | Pre-trained backbone wrappers | ~200 lines | Real data quality | Open |
| 10 | Contrastive alignment loss | ~50 lines | Stage 3 | Done |
| 11 | DM Control adapter | ~100 lines | Stage 1B | Open |
| 12 | TextWorld adapter | ~100 lines | Stage 1C | Open |
| Defer | RL infrastructure | ~1000+ lines | Stage 4 | Deferred |
| Defer | Distributed training | ~500+ lines | Scaling | Deferred |

**Items 1-6 should be completed as a single PR before starting any training stage.**

---

## Part 6: Key Reviewer Warnings

1. **Do not switch encoder architectures between stages.** Use `structured_multimodal` throughout to avoid weight transfer failures. The plan originally proposed simple→slot→structured, which would break checkpoint loading since parameter shapes differ.

2. **The autoencoder-only Phase 1A was self-contradictory.** The plan said "freeze all decoders" but expected autoencoder reconstruction signal from a frozen (randomly initialized) decoder. Fix: train encoder + decoder jointly from the start.

3. **Text action encoding is an unsolved problem in this codebase.** TextWorld actions are natural language strings, but the action encoder expects either discrete IDs or continuous vectors. Need to either tokenize actions and use text sub-encoder, or map to a discrete action vocabulary.

4. **Success criteria from the original plan were miscalibrated:**
   - PSNR > 25 dB through 128-dim bottleneck with tiny CNN is unrealistic (VQGAN gets ~22-25 dB with orders of magnitude more capacity)
   - Text perplexity < 30 for *next-state prediction* with a 2-layer 256-dim decoder is unrealistic
   - Cross-modal retrieval > 50% without contrastive loss implementation is not measurable

5. **22 weeks for everything including agentic RL from zero infrastructure is not credible.** Stages 0-3 alone are realistically 18-22 weeks for a single developer. Stage 4 (agentic) is a separate multi-month project.
