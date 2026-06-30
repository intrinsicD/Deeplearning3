# Tracked Work Plan

This is the sequenced execution plan derived from [`../Audit.md`](../Audit.md).
It converts the raw findings (**A1–A10**) and the wish-2/3 capability proposal
(**P1/P2**) into ordered work items with explicit dependencies, acceptance
criteria, and live status.

**How to read it.** Phases are strictly ordered: a phase may not start until
every item in its declared dependencies is `done`. Within a phase, items may
run in parallel unless an intra-phase dependency is noted. Each item has a
stable ID — reference these IDs in commits and PRs (e.g. `fix(W0.1): ...`).

**Status legend:** `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked

**Why this order.** Correctness before capability before growth. A
self-improving system amplifies whatever loop it is given; wiring selection
(P1/P2) or growth on top of a loop that trains on zero-tensors (A10) or with a
dead optimizer (A4/A7) would optimize the defect. So Phase 0 gates everything,
Phase 1 installs the measurement floor, and only then do the capability lanes
open.

---

## Dependency overview

```
Phase 0  Loop correctness (A1–A10)
   │            ┌─────────────────────────────┐
   ▼            ▼                             │
Phase 1  Quality floor (probes/vault/rollback)│
   │            │                             │
   ▼            ▼                             │
Phase 2  Wish 2 — selection (P1) ◄────────────┘ (uses probes to measure routing)
   │
   ▼
Phase 3  Wish 3 — use (P2-v1)
   │
   ├────────────► Phase 4  Growth (hook expansion on plateau)
   │
   └────────────► Phase 5  Research lane (P2-v2/v3, OOD, composition) — parallel, ongoing
```

---

## Phase 0 — Close the loop (blocks everything)

Goal: one fully-correct training step — every loss parameter optimized, every
data path delivering real tensors, every failure raising instead of
zero-filling, inference sharing the training forward path.

**Exit criterion for the phase:** a new `tests/integration/test_loop_integrity.py`
passes, asserting (a) all loss `log_vars` receive gradients and appear in the
optimizer, (b) a mixed real batch reaches the model with ≥2 modalities and
non-zero per-modality loss, (c) a corrupt input file raises rather than
returning zeros, (d) `generate()` and the training forward path produce
identical hidden states for the same input with hooks registered.

| ID | Item | Findings | Depends on | Files | Acceptance | Status |
|----|------|----------|-----------|-------|------------|--------|
| **W0.1** | Put learnable loss params in the optimizer | A4, A7 | — | `omnilatent/training/trainer.py`, `omnilatent/training/losses.py`, `scripts/training/curriculum_train.py`, `scripts/training/train_mmwm_minari.py`, `MMWM/losses.py` | Probe asserts `criterion_params_in_optimizer == 4/4` (Omni) and `8/8` (temporal) and MMWM uncertainty params update; build optimizer from `chain(model.parameters(), criterion.parameters())` **after** all criteria exist | `[ ]` |
| **W0.2** | Real data path: collate bridge + tensorized samples | A1, A2 | — | `omnilatent/data/collate/__init__.py`, `omnilatent/data/sources/local.py`, `omnilatent/data/datasets/streaming.py`, `omnilatent/data/sources/manifest.py` | `StreamingMultiModalDataset` → default DataLoader yields `dict[str,Tensor]`; local audio/video samples carry decoded tensors not paths; `streaming:"false"` parses to `False` (fix `bool(...)` coercion) | `[ ]` |
| **W0.3** | Collator must not silently drop modalities / emit zero-loss steps | A3 | W0.2 | `omnilatent/training/data.py`, `omnilatent/training/trainer.py` | Collator preserves all present modalities; a step with <2 usable modalities is skipped+counted, never returns `total: 0.0`; unit test on a mixed batch | `[ ]` |
| **W0.4** | Fail loud on bad media (no zero-tensor fallback) | A10 | — | `omnilatent/training/video_dataset.py`, `omnilatent/data/datasets/coco_dataset.py`, `datasets/adapters/mmwm_adapter.py` | Failed image/audio/video load raises a typed `MediaDecodeError` (or is dropped by the dataset), never returns zeros; test feeds a corrupt file and asserts raise/skip | `[ ]` |
| **W0.5** | Inference parity: hooks + true multimodal fusion | A9 | — | `omnilatent/model/omnilatent.py` | `generate()` runs through the hook manager (identical hidden states to training forward); `forward_multimodal()` fuses all provided inputs instead of picking one; tests assert both | `[ ]` |
| **W0.6** | Fix direction-blind / collapse-inducing temporal losses | A5 | W0.1 | `omnilatent/training/losses.py`, `scripts/training/curriculum_train.py` | `TemporalOrderLoss` gives different logits for `(A,B)` vs `(B,A)` (use the abandoned `combined` classifier input); distant-clip objective models change, not `MSE(z_anchor,z_context)` collapse; unit tests for asymmetry and non-collapse | `[ ]` |
| **W0.7** | Repair broken MMWM AV training script | A6 | — | `scripts/training/train_mmwm_av.py`, `MMWM/decoders.py`, `MMWM/losses.py` | Script runs end-to-end on synthetic data: define `include_text`/`include_image`, use `output_channels`/`output_size`, pass `weights` first to `WorldModelLoss`; smoke test | `[ ]` |
| **W0.8** | HPWM: refuse to freeze a random DINO; honor `--ssv2-dir` | A8 | — | `hpwm/model.py`, `hpwm/train.py`, `hpwm/data.py` | If pretrained DINO load fails, **raise** (or require `--allow-random-dino`); `--ssv2-dir` is threaded into `create_dataloaders`; test asserts both | `[ ]` |
| **W0.9** | Fix broken `curriculum_train` test imports | (test debt noted in Verification) | — | `tests/...` | The 3 failing temporal/video tests import from `scripts.training.curriculum_train`; suite green | `[ ]` |
| **W0.10** | Loop-integrity gate test | — | W0.1–W0.5 | `tests/integration/test_loop_integrity.py` | New test encodes the four phase-exit assertions above; CI runs it | `[ ]` |

---

## Phase 1 — Quality floor (measurement before self-change)

Goal: a hard, frozen-probe quality floor with rollback, so any later change
(selection, growth, self-training) is *measured* and cannot silently regress.
Implements `docs/self_improvement.md` §4.6 / §6.1 ahead of the rest of the
harness.

**Depends on:** Phase 0 complete.

| ID | Item | Depends on | Files | Acceptance | Status |
|----|------|-----------|-------|------------|--------|
| **W1.1** | Content-addressed checkpoint vault | Phase 0 | `scripts/training/self_improve/vault.py` | save→load round-trips identical metrics; SHA-256 addressing; LRU cap | `[ ]` |
| **W1.2** | Frozen probe sets + eval registry | Phase 0 | `scripts/training/self_improve/eval_registry.py`, `data/probes/<component>/` | per-component probe (Omni recon+caption acc, MMWM rollout, HPWM 3 signals, LGQ PSNR/LPIPS, Gauss MSE); hash-pinned; refuses start if probe file changed | `[ ]` |
| **W1.3** | Rollback-on-regression gate | W1.1, W1.2 | `scripts/training/self_improve/orchestrator.py` (gate only) | inject a regression batch → best snapshot restored, LR halved, event logged with provenance; test | `[ ]` |

---

## Phase 2 — Wish 2: input-conditioned selection (P1)

Goal: the system looks at an input and decides *which* learned structure is
relevant — a learned router over an expert registry, with abstention.

**Depends on:** Phase 0 (correct loop). **Uses:** Phase 1 probes to measure
routing quality (W2.4). Can begin once Phase 0 is done; W2.4 needs W1.2.

| ID | Item | Depends on | Files | Acceptance | Status |
|----|------|-----------|-------|------------|--------|
| **W2.1** | Expert registry (hooks ∪ tools ∪ KB-query) with learnable keys | Phase 0 | `omnilatent/agent/`, `omnilatent/model/hooks.py` (read `NeuralPortSpec.tags`) | every registered port/tool gets a key vector; registry enumerable; unit test | `[ ]` |
| **W2.2** | `LearnedLatentRouter(BaseRouter)` with sparse top-k gating | W2.1 | `omnilatent/agent/router.py` | maps pooled `LatentPacket.state` → top-k expert weights; emits `RouteDecision` + `metadata["expert_weights"]`; drop-in for `StaticRouter` in `runtime.py`; test | `[ ]` |
| **W2.3** | Retrieval-as-routing + calibrated abstention | W2.2 | `omnilatent/agent/router.py`, `omnilatent/kb/retrieval.py` | KB query derived from same summary via `retrieve_top_k`; below-threshold `confidence` routes to `KB_READ`/`MEMORY_READ` or backbone-only instead of forcing a skill; test | `[ ]` |
| **W2.4** | Routing probe in the eval suite | W2.2, W1.2 | `scripts/training/self_improve/eval_registry.py`, `tests/...` | synthetic tasks each solvable by exactly one hook → **routing accuracy**; **abstention calibration** (ECE/reliability); fails CI if accuracy ≤ chance | `[ ]` |

---

## Phase 3 — Wish 3: conditional use + credit (P2-v1)

Goal: actually deploy the selected skills for *this* input, and learn which
selection was correct (in-distribution).

**Depends on:** Phase 2.

| ID | Item | Depends on | Files | Acceptance | Status |
|----|------|-----------|-------|------------|--------|
| **W3.1** | Content-conditioned hook gates | W2.2 | `omnilatent/model/hooks.py` (`NeuralPortManager`) | effective gate = `sigmoid(static[h,L]) * route_weight[h](x)`; `route_weight=0` ⇒ exact prior behaviour (additive-safety preserved); effective gates logged to `AgentTraceStep.hook_gates`; test asserts exact recovery at weight 0 and parity at weight 1 | `[ ]` |
| **W3.2** | Learned router execution loop | W3.1, W2.3 | `omnilatent/agent/runtime.py` | runtime drives `LearnedLatentRouter`; top-k experts co-activate and compose in attention; side-effect guard intact; e2e test | `[ ]` |
| **W3.3** | Credit assignment v1 (end-to-end + load-balancing) | W3.2 | `omnilatent/agent/router.py`, training loop | router trained with task loss + Switch-style load-balancing aux; **counterfactual lift > 0** on the routing probe vs random-expert and backbone-only baselines | `[ ]` |

---

## Phase 4 — Growth (capacity on demand)

Goal: grow the network only when justified, via hook expansion — the repo's
native parameter-isolation method.

**Depends on:** Phase 1 (floor) + Phase 3 (so new capacity is selectable).

| ID | Item | Depends on | Files | Acceptance | Status |
|----|------|-----------|-------|------------|--------|
| **W4.1** | Plateau detector → `LatentNeuralHook` registration | W1.3, W3.3 | `scripts/training/self_improve/orchestrator.py`, `omnilatent/model/hooks.py` | manufactured plateau triggers a fresh hook (gate≈0); backbone weights unchanged; new hook trains and becomes routable by W2.2; test (mirrors `self_improvement.md` phase 8) | `[ ]` |

---

## Phase 5 — Research lane (parallel, ongoing, explicitly uncertain)

These are **not** scheduled deliverables — they are tracked experiments with
no guaranteed completion. They may run in parallel with Phases 2–4 once the
floor (Phase 1) exists to measure them. Each must report a metric, not a
vibe.

| ID | Item | Depends on | Acceptance (per experiment) | Status |
|----|------|-----------|------------------------------|--------|
| **W5.1** | Credit v2: outcome-based (probe-delta reward) | W3.3, W1.3 | router credit from frozen-probe deltas; beats v1 counterfactual lift on a 2-domain toy | `[ ]` |
| **W5.2** | Credit v3: counterfactual attribution on replay | W5.1 | marginal-probe-improvement credit; measurably better OOD selection than v1/v2 | `[ ]` |
| **W5.3** | OOD selection + abstention study | W2.4 | held-out-novel set; report whether abstention calibration holds off-distribution (it is expected to degrade — quantify it) | `[ ]` |
| **W5.4** | Compositional use (novel skill composition) | W3.3 | benchmark requiring composing 2 known hooks into an unseen behaviour; report gap vs single-skill | `[ ]` |
| **W5.5** | Router credit under pseudo-label self-training | W5.1, `self_improvement.md` §5 | confirm `§5` divergence guards cover router credit; poisoned-edge test severs+heals | `[ ]` |

---

## Milestones

- **M1 — Trustworthy loop:** Phase 0 done (`W0.10` green). The repo trains on
  real signal and fails loudly. *Prerequisite for any claim in the README.*
- **M2 — Measured & safe:** Phase 1 done. Capability can rise with a hard
  floor under it.
- **M3 — Selects (Wish 2):** Phase 2 done. The system identifies the relevant
  pattern and *abstains* when it has none, with a routing accuracy number.
- **M4 — Uses (Wish 3):** Phase 3 done. Selected skills are applied
  per-input with positive counterfactual lift.
- **M5 — Grows:** Phase 4 done. Capacity expands on demand without regression.
- *(Research)* **R*** — Phase 5 items report results as they land.

---

## Changelog

- *2026-06-30* — Plan created from `Audit.md` (A1–A10) + wish-2/3 proposal
  (P1/P2). All items `todo`.
