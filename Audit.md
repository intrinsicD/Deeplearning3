> **Status:** This file is the raw inventory. The sequenced, tracked execution
> plan derived from it lives in [`docs/work_plan.md`](docs/work_plan.md).
> Findings below are referenced from there as **A1–A10**; the capability
> proposal for wishes 2 & 3 is referenced as **P1/P2**.

I did not modify files. Findings below are ordered by impact on real learning/generalization.
Findings
1.
Blocker: manifest data cannot feed training as-is.
StreamingMultiModalDataset yields MultiModalSample objects directly, while the trainer expects dict[str, Tensor] batches. Default DataLoader fails with TypeError: default_collate ... found MultiModalSample. See streaming.py, sample.py, trainer.py. There is no real collate/tokenize/resize bridge in collate/init.py.
2.
Blocker: local audio/video manifest samples are metadata-only.
Local audio/video sources return paths in metadata, not tensors, so the model has nothing to learn from. See local.py. Also, streaming: "false" in JSON/YAML becomes truthy because of bool(...) coercion in manifest.py.
3.
High: collators silently drop modalities and can create zero-loss steps.
The main multimodal collator keeps only modalities present in every sample. Mixed real batches can lose image/audio/video fields entirely; if fewer than two modalities remain, training returns total: 0.0. See data.py and trainer.py.
4.
High: learnable loss parameters are not optimized.
MultiModalLoss has learnable log_vars, but optimizers are created from model.parameters() only. Probe result: criterion_params_in_optimizer 0 of 4. See losses.py and trainer.py. Curriculum training also omits both MultiModalLoss and TemporalContextLoss parameters; probe result: 0/4 and 0/8. See curriculum_train.py and losses.py.
5.
High: temporal order loss cannot learn direction.
TemporalOrderLoss computes a symmetric dot product, so (A, B) and (B, A) produce identical logits. The unused combined = torch.cat(...) suggests the intended classifier input was abandoned. See losses.py. Separately, distant-clip prediction directly minimizes MSE(z_anchor, z_context), encouraging temporal collapse rather than change modeling. See curriculum_train.py.
6.
High: public MMWM AV training script is broken.
include_text and include_image are undefined in train_mmwm_av.py. Image decoder kwargs use out_channels/out_size, but the decoder expects output_channels/output_size in decoders.py. The script also passes ModelConfig as the first positional arg to WorldModelLoss, whose first arg is weights in losses.py.
7.
High: MMWM learned uncertainty is also not optimized.
train_mmwm_minari.py creates the optimizer before WorldModelLoss(learned_uncertainty=True), so loss log_vars never update. See train_mmwm_minari.py and MMWM/losses.py.
8.
Medium: HPWM can train on frozen random visual features.
If pretrained DINO loading fails, HPWM falls back to a random DINO implementation, then freezes it. That can make training appear functional while preventing visual generalization. See model.py. Also, --ssv2-dir is not passed into create_dataloaders, so the CLI path can be ignored. See train.py and data.py.
9.
Medium: OmniLatent advertised multimodal paths bypass learned behavior.
generate() calls the backbone without the hook manager, so trained latent hooks are ignored during generation. See omnilatent.py. forward_multimodal() does not actually fuse multiple modalities; it picks one input and ignores the rest. See omnilatent.py.
10.
Medium: many real-data failures become zero tensors instead of hard failures.
Failed video/audio/image loads often return zeros, which keeps losses finite while teaching the model corrupted targets. Examples: video_dataset.py, coco_dataset.py, mmwm_adapter.py.
Verification
Focused tests: 29 passed for gradient/model/streaming coverage. Temporal/video focused tests: 3 failed, 45 passed; all three failures are broken imports of curriculum_train from tests, while implementation lives under scripts/training/curriculum_train.py. Full suite was attempted but did not complete cleanly in this environment.

---

# Capability proposal — wishes 2 & 3 (pattern identification and pattern use)

The findings above (A1–A10) are *correctness* defects: the learning loop does
not yet optimize the signal it claims to. This section is different — it is a
*capability* proposal for the two sub-goals the repository does not yet
address at all:

- **Wish 2 — "identify the right pattern in the data."** Given an input,
  decide *which* learned structure (skill / memory / tool) is relevant.
- **Wish 3 — "use the right pattern."** Actually deploy the selected
  structure to produce output, and learn which choice was correct.

## Current gap (grounded in code)

- The only router is `StaticRouter` (`omnilatent/agent/router.py:37`): it
  replays a scripted sequence of `RouteDecision`s. There is **no
  input-conditioned selection** anywhere in the system.
- Hook influence is a **static, always-on learned scalar**: the per-layer
  sigmoid gate in `NeuralPortManager` / `LatentNeuralHook.gate_values()`
  (`omnilatent/model/hooks.py`) does not depend on the current input. Once a
  hook is trained it fires for *every* packet, whether relevant or not.
- KB retrieval exists (`omnilatent/kb/retrieval.py:retrieve_top_k`) but is
  not wired to any decision about *whether* the retrieved pattern should be
  used.
- The plumbing for observability is already present and unused for this
  purpose: `RouteDecision.confidence` / `stop_prob`
  (`router.py:11`) and `AgentTraceStep.hook_gates`
  (`omnilatent/protocol.py:254`).

So both wishes reduce to one missing component: **an input-conditioned
selection layer over a registry of "experts"** (hooks, tools, KB queries),
plus a **credit-assignment signal** that teaches it.

## P1 — Wish 2: a learned, input-conditioned router-over-experts

Replace/augment `StaticRouter` with `LearnedLatentRouter(BaseRouter)`:

1. **Expert registry.** Treat every registered `NeuralPort`/hook, every
   `ToolDescriptor`, and "query the KB" as a selectable *expert*. Each gets a
   learnable key vector (seeded from `NeuralPortSpec.tags` /
   `ToolDescriptor.tags` where available).
2. **Router head.** A small module maps a pooled summary of the encoded
   `LatentPacket.state` (after the ENCODE node) to routing logits over the
   registry. Use **sparse top-k gating** (Switch/MoE style) so only a few
   experts fire per input — this keeps a *small* model cheap while letting the
   registry grow.
3. **Retrieval as routing.** Derive the KB query from the same summary and
   reuse `retrieve_top_k`; retrieved chunks become candidate patterns scored
   alongside the hooks.
4. **Abstention / "do I even have the right pattern?"** Emit a calibrated
   `confidence`. Below threshold, the router does **not** force a skill — it
   routes to `KB_READ`/`MEMORY_READ` to gather evidence, or falls back to
   backbone-only. This abstention path is the part that matters for novel
   (out-of-distribution) input, and is where honest measurement is required
   (see "open problems").

`LearnedLatentRouter` emits the existing `RouteDecision` (action, confidence,
stop_prob) plus a `metadata["expert_weights"]` map, so the runtime and trace
are unchanged.

## P2 — Wish 3: content-conditioned activation + execution + credit

1. **Content-conditioned gates.** Extend `NeuralPortManager` so the effective
   gate for hook *h* at layer *L* becomes
   `sigmoid(static_gate[h,L]) * route_weight[h](x)`.
   When `route_weight = 0` the hook is *fully* off — zero cost, zero
   interference — which **preserves the additive-safety property** that makes
   hooks valuable (gate≈0 ⇒ exact recovery of prior behaviour). When `1`, the
   current behaviour is recovered exactly. Log effective gates into the
   already-present `AgentTraceStep.hook_gates`.
2. **Composition.** The router's top-k experts run simultaneously; hooks
   already compose in attention ("Composable" in `hooks.py`). No new
   composition code — only conditioned activation.
3. **Execution.** The agent runtime (`omnilatent/agent/runtime.py`) already
   executes routed actions (TOOL_CALL/KB_READ/DECODE) under the side-effect
   guard. We only swap the *static* router for the *learned* one.
4. **Credit assignment** — the hard part, ranked by honesty:
   - **v1 (engineering, tractable):** train the router end-to-end with the
     task loss + a Switch-style **load-balancing** auxiliary loss. Gives
     correct selection **in-distribution** only.
   - **v2 (research):** **outcome-based credit** — use the
     self-improvement harness's frozen-probe deltas (see
     `docs/self_improvement.md` §4.6/§6.1) as a reward signal: which expert,
     when activated, improved the probe? Connects P2 to the existing harness.
   - **v3 (research, principled):** **counterfactual attribution** — on
     replay samples, run with and without each candidate expert and assign
     credit by marginal probe improvement. Expensive but the only path to
     *correct* selection rather than *plausible* selection.

## Measurement (non-negotiable, consistent with the harness philosophy)

Add a **routing probe** to the frozen eval suite so selection quality is
gated like everything else:

- **Routing accuracy** on a held-out set where the correct expert is known
  (e.g. synthetic tasks each solvable only by one registered hook).
- **Abstention calibration** — does low `confidence` actually correlate with
  wrong/again-novel selection? (reliability curve / ECE).
- **Counterfactual lift** — mean probe improvement from the router's choice
  vs. a random expert and vs. backbone-only. If lift ≤ 0 the router is noise.

## Open problems this proposal does NOT solve (stated plainly)

- **Out-of-distribution selection.** v1 gating generalizes only as far as the
  representation does. Choosing the right pattern on *genuinely novel* input
  is unsolved field-wide; the abstention path bounds the damage, it does not
  remove the problem.
- **Compositional/systematic use.** Top-k composition handles *co-activation*,
  not *novel composition* of skills into a skill the registry has never seen.
- **Stable credit under self-training.** v2/v3 feed on probe deltas, which the
  pseudo-label co-training loop (`self_improvement.md` §5) can bias. The
  divergence guards there must cover router credit too.

The engineering core (P1 + P2-v1 + the routing probe) is buildable now and is
sequenced in `docs/work_plan.md`. v2/v3 and the open problems are tracked
there as a separate, explicitly-research lane.