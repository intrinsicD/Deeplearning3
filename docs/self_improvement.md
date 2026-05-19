# Self-Improvement Protocol

A continual self-supervised learning system that lets each model in this
repo (`omnilatent`, `MMWM`, `hpwm`, `lgq`, `gaussian_encoder`) improve
itself indefinitely from unlabeled data — individually or jointly — without
catastrophic forgetting.

This document is the design. Implementation lands on
`claude/self-supervised-learning-FaPcc` in incremental, reviewable PRs
(see [Phases](#phases)).

---

## 1. Goals and non-goals

**Goals**

1. Run unattended on a stream of unlabeled inputs (raw video, audio,
   images, text). No human labels at any point in the loop.
2. Allow any single component to self-improve in isolation.
3. Allow all five components to co-train, sharing data, pseudo-labels, and
   distillation signals.
4. Guarantee monotonic-or-better performance on a frozen evaluation suite:
   never silently regress.
5. Be killable and resumable at any step — full state lives on disk.

**Non-goals**

- Online learning with real-time guarantees (we operate in epoch-batches).
- Reinforcement-learning style reward shaping. The signal is purely
  self-supervised reconstruction / prediction quality.
- Architecture search. Capacity is grown only through `LatentNeuralHook`
  additions (see §6.3); the backbone itself is fixed per run.

---

## 2. High-level architecture

```
                       ┌──────────────────────────────────────┐
                       │      SelfImproveOrchestrator         │
                       │  (scripts/training/self_improve.py)  │
                       └───────────────┬──────────────────────┘
                                       │
       ┌───────────────┬───────────────┼───────────────┬───────────────┐
       ▼               ▼               ▼               ▼               ▼
  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
  │  Omni   │    │  MMWM   │    │  HPWM   │    │   LGQ   │    │ Gauss.  │
  │ Plugin  │    │ Plugin  │    │ Plugin  │    │ Plugin  │    │ Plugin  │
  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘
       │              │              │              │              │
       ▼              ▼              ▼              ▼              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                    Shared services (one process)                  │
  │  • ReplayBank          — per-component reservoir buffers          │
  │  • EMATeacher          — slow-EMA snapshot for distillation       │
  │  • FisherStore         — diagonal-Fisher matrices for EWC         │
  │  • EvalRegistry        — frozen probe sets per component          │
  │  • CheckpointVault     — versioned, hash-addressed weights        │
  │  • PseudoLabelBroker   — routes labels between components         │
  │  • DataStream          — adapters for raw inputs → batch dicts    │
  └──────────────────────────────────────────────────────────────────┘
```

The orchestrator owns the outer loop. Each **plugin** is a thin adapter
around an existing trainer (no rewrites — see the file:line citations in
§7). Plugins implement a small interface:

```python
class ComponentPlugin(Protocol):
    name: str
    def make_batch(self, raw: RawSample, replay: list[BatchDict]) -> BatchDict: ...
    def train_step(self, batch: BatchDict) -> StepReport: ...
    def evaluate(self, probe_set: ProbeSet) -> EvalReport: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
    def teacher_distill_loss(self, batch, teacher_out) -> Tensor: ...   # optional
    def emit_pseudo_label(self, raw: RawSample) -> PseudoLabel | None:  # optional
```

That's the whole contract. Everything else (replay, EWC penalties,
rollback, scheduling) is implemented once in the orchestrator and inherits
to every component.

---

## 3. The continual-learning loop

```
loop forever:
    # 3.1 pick what to train this round
    component = scheduler.pick()              # see §3.1

    # 3.2 build a mixed batch
    raw       = data_stream.next(component.modality_filter)
    replay    = replay_bank.sample(component.name, k=B_replay)
    batch     = component.make_batch(raw, replay)

    # 3.3 attach forgetting-mitigation losses
    losses    = component.train_step(batch)
    losses   += ewc.penalty(component)                          # §4.3
    losses   += distill.loss(component, teacher, batch)         # §4.2
    losses   += pseudo_label_broker.consistency_loss(component) # §5

    losses.backward()
    optimizer.step()

    # 3.4 maintain auxiliary state
    ema_teacher.update(component)                               # §4.4
    replay_bank.insert(component.name, raw, reservoir=True)     # §4.1
    fisher.online_update(component, batch)                      # §4.3

    # 3.5 periodic gating
    if step % EVAL_EVERY == 0:
        report = component.evaluate(eval_registry[component.name])
        vault.snapshot(component, report)
        if regression(report, baseline=vault.best(component)):
            vault.rollback(component)                           # §4.6
        scheduler.record(component, report)
```

### 3.1 Scheduler

A bandit over components. State per component: best score `b_i`, latest
score `s_i`, steps-since-last-eval `t_i`, "stagnation pressure" `p_i`.

- **Round-robin warmup** for the first `N_warm` rounds (all five touched
  equally) so we have a baseline for every component.
- After warmup, score each candidate by

  `priority_i = α · regression_gap_i + β · staleness_i + γ · uncertainty_i`

  where `regression_gap_i = max(0, b_i − s_i)` (favor recovering losses),
  `staleness_i = t_i / EVAL_EVERY` (favor unrun components), and
  `uncertainty_i` is the std of recent eval scores (favor noisy ones).
- A fixed quota (e.g. 10%) is always spent on the *worst-performing*
  component to prevent abandonment.

### 3.2 Modality filter

Each component declares which modalities it can consume. The `DataStream`
demuxes a single raw video frame-sequence into:

- video clip tensors → HPWM, MMWM, OmniLatent
- audio mel spectrograms → MMWM, OmniLatent
- single frames → LGQ, Gaussian Encoder (resized to 28×28 grayscale)
- transcripts (if present) → OmniLatent, MMWM

One raw sample fans out to many plugins; this is where co-training gets
its data efficiency.

---

## 4. Forgetting-mitigation strategy (defense in depth)

No single technique is sufficient. We stack five, ordered from cheapest to
most invasive. Each one is independently togglable in the config so we can
ablate.

### 4.1 Experience replay (always on)

The simplest, most-validated method (Chaudhry et al. 2019, Buzzega et al.
2020 "Dark Experience Replay"). For each component we keep a
**reservoir-sampled** buffer of fixed capacity `M` (default 10k samples).

- **Storage form is component-aware.** Raw video is expensive, so:
  - HPWM/MMWM store **LGQ tokens + audio mel + transcript** (≈1 KB/sample
    after tokenization, versus ≈2 MB raw). This is the cross-component
    payoff: LGQ pays for everyone else's replay.
  - LGQ/Gaussian Encoder store **raw image bytes** (cheap).
  - OmniLatent stores the (modality, tensor-hash) tuple and rematerializes
    from the shared `DataStream` cache.
- **Mix ratio** at each step: 50% fresh, 50% replay (configurable). Dark
  Experience Replay (DER++) additionally stores the **logits** the model
  produced for each replay sample at insertion time and adds an MSE
  consistency term — this is dramatically more effective than vanilla
  replay and we use it as the default.
- **Eviction:** reservoir sampling gives uniform coverage of all
  historical data without storing it all.

### 4.2 EMA teacher distillation (always on for OmniLatent / MMWM / HPWM)

We keep a slow-moving exponential moving average of each component's
weights (decay `0.999`, akin to BYOL/MoCo/DINO). On every step:

- Run the EMA teacher in `eval()` mode on the *same batch*.
- Add `λ_kd · KL(student || teacher)` for distributional outputs
  (OmniLatent text logits, MMWM latent Gaussian) or
  `λ_kd · MSE(student, teacher)` for regression outputs (HPWM
  next-frame, LGQ reconstruction).
- This is "Learning without Forgetting" (Li & Hoiem 2017) implemented
  with self-distillation — no separate teacher checkpoint needed.

The EMA also serves as the **rollback target** (§4.6) and the **export
checkpoint** (we publish EMA weights, not raw weights, because they
generalize better — well established in DINO/MAE/MoCo).

### 4.3 Online EWC + Synaptic Intelligence (optional, on for backbone)

Elastic Weight Consolidation (Kirkpatrick et al. 2017) anchors important
parameters with a quadratic penalty:

`L_ewc = (λ_ewc / 2) · Σ_i F_i · (θ_i − θ_i*)²`

where `F_i` is the diagonal Fisher and `θ_i*` is a snapshot. We use
**Online EWC** (Schwarz et al. 2018):

- Fisher is updated incrementally: `F ← γ · F + (1−γ) · F_batch`, `γ=0.95`.
- The anchor `θ*` is the most recent EMA snapshot (not the original init).
- We compute Fisher only on parameters with `requires_grad` and with norm
  above a small threshold — keeps the matrix sparse and storage bounded.

We additionally maintain **Synaptic Intelligence** path integrals
(Zenke et al. 2017) — strictly cheaper than Fisher because the importance
estimate is accumulated during the normal backward pass, no second pass
needed. SI and online-EWC compose well; we average their importances.

### 4.4 Parameter-isolation via Latent Neural Hooks (the repo's killer feature)

The single best forgetting defense available here, because the
architecture already supports it. When the scheduler decides a component
needs a **new skill** (e.g. a new language, a new visual domain), we do
not fine-tune the backbone. Instead:

1. Register a fresh `LatentNeuralHook` with the gate biased near zero
   (`gate_bias_init=-4.0`, exactly the README default).
2. Freeze the backbone (`requires_grad = False`).
3. Train only the hook. The sigmoid gate lets the model recover the
   original behaviour exactly (gate ≈ 0) and any improvement is purely
   additive.
4. After convergence, optionally **distill the hook back into the
   backbone** with a slow LwF schedule, then drop the hook.

This is essentially LoRA / Progressive Networks (Rusu et al. 2016)
implemented natively in this codebase. It is the path of last resort for
domains where the other defenses can't prevent regression.

### 4.5 Gradient surgery (A-GEM, optional)

Averaged Gradient Episodic Memory (Chaudhry et al. 2019): when the
current-batch gradient `g` would increase loss on the replay buffer
gradient `g_ref`, project it:

`g' = g − (g·g_ref / ‖g_ref‖²) · g_ref   if g·g_ref < 0`

A-GEM costs one extra forward+backward on a small replay batch per step.
Off by default (the DER++ replay + EWC + EMA stack is usually sufficient)
but switchable on per-component if we see persistent regression.

### 4.6 Snapshot-and-rollback gating (always on)

The hard guarantee. Every `EVAL_EVERY` steps:

- Evaluate the student on the frozen `ProbeSet` (§3.1, §7) for this
  component.
- Compare against `vault.best(component)`. If the score is **worse by more
  than `tol`** on *any* probe metric, restore weights from the best
  snapshot, halve the learning rate, and continue. Log the regression
  with full provenance (which data, which step).
- If the score is **better than best**, atomically promote the new
  snapshot to `best` and persist Fisher / SI / EMA / replay-buffer
  state alongside it.

`tol` is per-metric and small (e.g. 1% relative for PSNR, 0.5pp absolute
for token accuracy). The vault is content-addressed (SHA-256 of the
state dict) so duplicate snapshots are free.

### 4.7 What we explicitly do *not* use

- **L2-SP / vanilla L2 to init** — strictly worse than EWC in every
  benchmark I've seen.
- **Pure replay-only** — works but converges slowly without EMA/EWC
  helpers; we use it as the *substrate*, not the whole solution.
- **PackNet / iterative pruning** — too invasive; replaced by hooks.
- **Generative replay from the model's own decoder** — appealing but
  drifts (mode collapse over many iterations). LGQ-compressed real data
  is strictly better than synthetic replay here.

---

## 5. Cross-component bootstrapping

The point of co-training is that each model can label data for the
others. The `PseudoLabelBroker` mediates this so cycles don't form.

| Producer | Consumer | Pseudo-label |
|---|---|---|
| LGQ | HPWM, MMWM | Image / frame token IDs — used as discrete targets and replay storage |
| HPWM | OmniLatent | Predicted next-frame slot states — used as a "future" target for video→video |
| OmniLatent (text decoder) | MMWM, HPWM | ASR-style transcripts for silent video clips, enabling grounding without real transcripts |
| MMWM (latent) | OmniLatent | Cross-modal anchor latents — InfoNCE alignment target |
| Gaussian Encoder | LGQ | A second-opinion reconstruction signal for grayscale images (sanity check) |

**Rules to prevent collapse**:

1. **Direction is acyclic.** A graph of `(producer, consumer)` edges is
   checked at orchestrator startup; cycles are an error.
2. **Pseudo-labels are versioned.** A consumer trains only on labels
   produced by a *frozen* snapshot of the producer (vault commit hash).
   The producer is allowed to keep improving in parallel; the consumer
   picks up new labels only when its scheduler chooses to refresh.
3. **Confidence-gated.** Each producer attaches a scalar confidence
   (e.g. softmax entropy for text, codebook perplexity for LGQ, slot
   binding stability for HPWM). Consumers drop pseudo-labels below
   threshold.
4. **Real data always mixed in.** Pseudo-label batches are capped at
   30% of the training mix.

---

## 6. Safeguards against silent failure

### 6.1 Frozen probe sets

Each component owns a small (≈500-sample), **never-trained-on** probe set
held out at orchestrator init. These are the gold-standard scores used
for rollback decisions. Stored under `data/probes/<component>/` and
hashed; if the file changes, the orchestrator refuses to start.

| Component | Probe |
|---|---|
| OmniLatent | 16-modality-pair reconstruction MSE + caption token accuracy on a held-out COCO slice |
| MMWM | Multi-step latent rollout MSE + cosine on held-out video |
| HPWM | All three validation signals from `hpwm/evaluate.py` |
| LGQ | PSNR / SSIM / LPIPS on held-out images |
| Gaussian Encoder | MSE on held-out MNIST + Gaussian σ distribution KL |

### 6.2 Compute / drift budget

- A hard step budget per orchestrator run (`--max-steps`). No infinite
  loops by default.
- A "drift budget": if cumulative EMA-vs-init weight L2 distance crosses
  a threshold, the orchestrator pauses and emits a report. This catches
  runaway pseudo-label loops before they corrupt weights.
- Disk budget on the vault (LRU eviction of non-`best` snapshots once
  storage exceeds `--vault-cap-gb`).

### 6.3 Capacity-aware expansion

When a component's score plateaus for `N_plateau` evals despite scheduler
priority, the orchestrator **adds a hook** (§4.4) rather than continuing
to bash the backbone. This is the only allowed form of growth.

### 6.4 Reward-hacking detection

Self-supervised losses can be gamed (e.g. a degenerate decoder that emits
the mean image scores well on MSE but terribly on LPIPS). We mitigate by:

- Always evaluating with **at least two uncorrelated metrics** per
  component (the probe sets in §6.1 are designed for this — PSNR + LPIPS,
  MSE + cosine, etc.).
- Including a small adversarial probe per component (random transforms
  that should not change semantic identity).

---

## 7. Per-component plan

For each component I list the existing surfaces the plugin will wrap. All
line numbers are stable as of the inventory dated 2026-05-19.

### 7.1 OmniLatent  (`omnilatent/`)

- **Trainer wrap**: `omnilatent/training/trainer.py:47` (`Trainer.__init__`),
  step at `:150` (`_train_step`), validation at `:286`.
- **Loss surface**: `omnilatent/training/losses.py` — already
  label-free for self-reconstruction (text CE on its own tokens, audio
  L1+MSE, image L1+FFT, video L1+temporal).
- **Forgetting stack**: DER++ replay + EMA teacher + online EWC on
  backbone + hooks for new modalities.
- **Pseudo-label production**: caption generation via `model.generate`
  for HPWM/MMWM consumers.
- **Probe set**: held-out COCO subset (the existing `train_coco.py`
  already pulls COCO; we slice 500 unseen captions).

### 7.2 MMWM  (`MMWM/`)

- **Trainer wrap**: `MMWM/trainer.py:41` (`Trainer.__init__`), step at
  `:226` (`train_step`), sequence step at `:298`.
- **Eval**: `MMWM/evaluation.py:252` (`EvaluationSuite`) — already
  returns multi-step rollout MSE / cosine / R²; perfect for the probe.
- **Loss surface**: `MMWM/losses.py:52` — fully self-supervised when
  reconstruction losses are disabled; we keep them on, weighted, with
  uncertainty-weighting already implemented.
- **Forgetting stack**: DER++ replay (storing LGQ tokens of frames) +
  EMA teacher + online EWC on world-model core. Curriculum scheduler
  inside MMWM is preserved and runs *inside* whatever step the
  orchestrator picks.

### 7.3 HPWM  (`hpwm/`)

- **Trainer wrap**: `hpwm/train.py:32` (`Trainer.__init__`), main loop
  at `:128`.
- **Eval**: `hpwm/evaluate.py:34` — three validation signals, all three
  must pass. Encode this as `min(signal_i)` for the scheduler priority.
- **Forgetting stack**: DER++ replay + EMA teacher. EWC is *off* because
  HPWM's MoD router is too sparse for diagonal Fisher to be meaningful;
  we rely on replay + EMA + the frozen DINO backbone for stability.
- **Pseudo-label production**: next-frame slot states → OmniLatent
  video→video target.

### 7.4 LGQ  (`lgq/`)

- **Trainer wrap**: `lgq/train.py:111` (`train(config)`). We refactor
  the inner loop body into a `step(batch)` method so the orchestrator
  can drive one step at a time (small, mechanical change).
- **Eval**: `lgq/train.py:306` (`evaluate`) — returns PSNR/SSIM/FID +
  codebook stats.
- **Forgetting stack**: replay (raw images) + EMA teacher on the
  generator. Discriminator is *re-initialized* periodically (every
  N steps) rather than EMA'd — standard GAN practice avoids the
  discriminator becoming a memory bottleneck.
- **Pseudo-label production**: image / frame tokens consumed by all
  video models. This is the most-used producer in the system.

### 7.5 Gaussian Encoder  (`gaussian_encoder/`)

- **Trainer wrap**: refactor `gaussian_encoder/train.py:45` `main()`
  into `make_trainer()` + `step(batch)`. Currently no checkpoint code —
  we add a minimal `state_dict()/load_state_dict()` round-trip.
- **Forgetting stack**: replay + EMA only. It's small enough that EWC
  is overkill.
- **Role**: scientific control / interpretability probe. We track the
  evolution of its Gaussian σ distribution as a sanity signal — if the
  whole system is healthy, the σ distribution should drift slowly.

---

## 8. File layout

New code lives under `scripts/training/self_improve/` (a package, not a
single file — the orchestrator is too big to live in one module).

```
scripts/training/self_improve/
    __init__.py
    orchestrator.py          # main loop; thin
    scheduler.py             # bandit over components
    plugins/
        __init__.py
        base.py              # ComponentPlugin protocol + StepReport / EvalReport
        omnilatent.py
        mmwm.py
        hpwm.py
        lgq.py
        gaussian_encoder.py
    forgetting/
        __init__.py
        replay.py            # ReplayBank, DER++ logic
        ema.py               # EMATeacher
        ewc.py               # online EWC + Synaptic Intelligence
        agem.py              # gradient projection (optional)
        hooks.py             # helpers to register / freeze / distill hooks
    vault.py                 # content-addressed checkpoint storage
    eval_registry.py         # frozen probe sets + scoring
    pseudo_labels.py         # cross-component label broker
    data_stream.py           # raw input → per-component batch demux
    cli.py                   # entry point: `omnilatent-self-improve`

scripts/training/self_improve.py  # one-line shim → self_improve.cli:main

tests/self_improve/
    test_replay.py
    test_ewc.py
    test_ema.py
    test_vault_rollback.py
    test_scheduler.py
    test_plugins_smoke.py    # one tiny step per plugin on synthetic data
    test_end_to_end.py       # 50 steps across all 5 components, verify no regression
```

CLI:

```bash
# Train one component, self-supervised, on a video directory
omnilatent-self-improve --components hpwm --video-dir /data/videos --max-steps 50000

# Train all five jointly with cross-component pseudo-labels
omnilatent-self-improve --components all --video-dir /data/videos \
    --enable-pseudo-labels --max-steps 200000

# Resume
omnilatent-self-improve --resume runs/2026-05-19_03-22/

# Dry-run: validate config, build probe sets, evaluate every component once, exit
omnilatent-self-improve --dry-run --components all
```

Config is YAML, schema in `scripts/training/self_improve/config_schema.py`
(pydantic). Sane defaults so the CLI works with zero flags.

---

## 9. Phases (concrete PRs)

Each phase is a self-contained PR that leaves the repo in a working state.

| # | PR | Adds | Verifies |
|---|---|---|---|
| 1 | **Plugin scaffolding** | `plugins/base.py`, plugin wrappers around the five existing trainers, no orchestration yet. Refactor `lgq/train.py` and `gaussian_encoder/train.py` to expose `step(batch)`. | New unit test per plugin: 1 step on synthetic data, weights changed, no NaN. |
| 2 | **Vault + eval registry** | `vault.py`, `eval_registry.py`, probe-set loaders for each component. | Round-trip test: save → load → identical metrics. |
| 3 | **Replay + EMA** | `forgetting/replay.py` (DER++), `forgetting/ema.py`. Wire into all five plugins behind a flag. | Train a plugin for 100 steps with replay disabled vs. enabled on a two-domain toy task; replay version retains domain-1 accuracy. |
| 4 | **EWC + SI** | `forgetting/ewc.py`. Wire into OmniLatent/MMWM plugins. | Same two-domain test, this time with EWC; should outperform replay-only on the OmniLatent backbone. |
| 5 | **Orchestrator + scheduler** | `orchestrator.py`, `scheduler.py`, `data_stream.py`, `cli.py`. Single-component mode only. | End-to-end test: 50 steps on HPWM via the orchestrator matches direct training within tolerance. |
| 6 | **Multi-component co-training** | Enable all five at once. No pseudo-labels yet. | E2E test: 5 components × 10 steps each, all probes pass. |
| 7 | **Pseudo-label broker** | `pseudo_labels.py`, integration with LGQ→HPWM, OmniLatent→MMWM, etc. Cycle check at startup. | Test that pseudo-labels actually flow and that the cycle detector rejects a bad config. |
| 8 | **Hook-based capacity expansion** | Plateau detector triggers `LatentNeuralHook` registration on OmniLatent. | Manufactured plateau triggers expansion; new hook trains; backbone weights unchanged. |
| 9 | **A-GEM + drift budget + rollback hardening** | `forgetting/agem.py`, vault rollback on regression. | Inject a regression-inducing batch; rollback fires; final metrics match pre-regression baseline. |
| 10 | **Control-center integration** | Endpoints in `apps/control_center/server.py` to launch / monitor / pause a self-improve run. | UI smoke test. |

PRs 1–6 are the minimum viable system. 7–10 are the "all together" /
production hardening half.

---

## 10. Open questions (decide before phase 5)

1. **Single process vs. multi-process?** A single process is simpler and
   gets us shared GPU memory for the EMA teachers. Multi-process is
   safer for blast radius (one component crashing won't kill the run).
   Recommendation: **single process** for v1, with each plugin in its
   own CUDA stream; revisit if we see contention.
2. **Where does the data stream live?** Pulling from disk every step is
   slow on cold cache. Recommendation: **persistent dataloader workers**
   per component, shared `DataStream` cache in shared memory.
3. **How do we test forgetting without weeks of compute?** Build a
   synthetic two-domain regression toy in `tests/self_improve/` (the
   harness from phase 3) that runs in <60s. Real regression tests run
   nightly on the control-center machine.
4. **Pseudo-label confidence calibration.** The thresholds in §5 are
   guesses. Phase 7 should include a small calibration script that
   sweeps the threshold against held-out consumer probe scores.
5. **License of pseudo-labels.** If we train on user-supplied video and
   then publish weights, the pseudo-labels we generated may inherit the
   video's license. Document this in the CLI's `--help`.

---

## 11. Summary

The system is built around three claims:

1. **Replay + EMA + EWC + isolation + rollback** is the strongest known
   defense-in-depth for catastrophic forgetting. Each layer covers what
   the others miss.
2. **The repo's `LatentNeuralHook` mechanism is a first-class
   parameter-isolation method** — we get LoRA-style capacity expansion
   for free, with no third-party dependencies.
3. **Pseudo-label co-training is safe iff the graph is acyclic,
   confidence-gated, and capped against real data** — these three rules,
   enforced at orchestrator startup, prevent the failure modes that
   sink most self-training papers.

If all three claims hold, the five models in this repository can run
indefinitely on a stream of unlabeled video and image data, individually
or jointly, with a hard floor on quality and an open ceiling.
