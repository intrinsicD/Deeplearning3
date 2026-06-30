# Toward a Small, Self-Improving General Model — An Honest Architecture Design

This document answers four questions that motivate this repository:

1. What is the best architecture and harness for a small "AGI-capable" model?
2. What is AGI, exactly?
3. How can we build/grow a network that processes any input, finds the right
   pattern, uses it, improves itself over time, and produces high-quality
   output of any kind?
4. Are these the right questions?

It is written to be useful, not flattering. Where the field has no answer,
it says so. Where this repository already has the pieces, it cites them.
Where this repository is currently broken, it cites `Audit.md`.

---

## 0. The most important reframe

**AGI is a property of a learning loop, not of a network architecture.**

If there were a single "best architecture," the field would be finished.
There isn't. Every operational definition of general intelligence that is
worth using is *behavioral* — about what a system can learn and do — not
*structural*. So the productive target is not "the best network." It is:

> the **minimal substrate** that can keep getting more capable on its own,
> wrapped in a **harness** that grows it without regressing.

This repository already contains a substrate (`omnilatent`, `MMWM`, `hpwm`,
`lgq`) and the beginnings of that harness (`docs/self_improvement.md`,
`omnilatent/agent/`, `omnilatent/kb/`). The architecture is not the
bottleneck. The *correctness and closure of the loop* is.

---

## 1. What AGI is (operational definitions)

There is no consensus definition. The useful ones are all behavioral:

- **Capability coverage.** Matches competent humans across the *breadth* of
  cognitive tasks, including unseen ones. Breadth, not superhuman depth.
- **Transfer & sample efficiency** *(the definition that should drive this
  project)*. The ability to acquire **new** skills from few examples and
  **transfer** across domains. Generality lives in the *learning*, not the
  frozen weights.
- **Closed-loop autonomy.** Perceive → model → decide → act → learn from the
  outcome, over long horizons, without being re-engineered per task.

None of these names a topology. This is why "what is the best architecture"
is the wrong first question: the architecture is a substrate; the
intelligence is the loop.

### 1.1 Why "small + AGI-capable + any input/any output/any quality" is overloaded

Three properties are being asked for at once:

- **Generality** (any input, any output) — *achievable at small scale* as
  *general-purpose breadth*. `OmniLatent` already spans 4 modalities × 16
  directions.
- **Competence** (high quality of any kind) — *not* free at small scale.
  Quality is bought with parameters, data, and compute. A small model can be
  a generalist or a specialist-quality expert, rarely both at once.
- **Omnicompetence** (high quality at *everything*) — not a small-model
  property at all, and arguably not a current-era property for anyone.

The defensible target is: **a small, general-purpose model that improves its
quality over time on the domains it is actually exposed to.** That is what
the harness in §5 is for.

---

## 2. The five sub-wishes, sorted by real difficulty

The motivating request — a network that can (1) process any input,
(2) identify the right pattern, (3) use the right pattern, (4) improve
itself over time, (5) produce high-quality output of any kind — decomposes
into tasks of *very* different difficulty:

| Wish | Status | Where it lives | Honest note |
|---|---|---|---|
| (1) Process any input | **Solved engineering** | `omnilatent` encoders + unified latent | New modality = encoder + `LatentNeuralHook`. |
| (5) Produce any output | **Solved engineering; hard quality** | `omnilatent` decoders | Quality is scale/data, not topology. |
| (2) Identify the right pattern | **Partly open** | representations + `kb/` retrieval + world model | In-distribution: works. Out-of-distribution: open for everyone. |
| (3) Use the right pattern | **Open (control)** | `agent/router.py` | Choosing *which* skill to deploy is a policy problem, not a perception one. |
| (4) Improve over time | **The frontier** | `self_improve/` | This is where AGI lives — and where this repo is most fragile. |

Takeaway: wishes 1 and 5 are essentially done as *plumbing*. The project's
real work is 2–4, and the center of gravity is 4.

---

## 3. Reference architecture (six subsystems, one loop)

Think in subsystems in a loop, not in "a network." This repository already
has a named module for each.

```
        ┌──────────── self-improvement harness (the AGI part) ────────────┐
        │                                                                  │
   Perception → Unified Latent → World Model → Controller → Action → Outcome
   (encoders)   (OmniLatent      (MMWM/HPWM)   (agent/      (decoders/  │
        │        backbone)                      router)      tools)      │
        │                            ↑                                   │
        └──────── Memory (KB + replay + EMA + vault) ←───────────────────┘
                              (learns from the outcome)
```

| Subsystem | Purpose | Module(s) | Maturity |
|---|---|---|---|
| Perception → latent | map any modality into one space | `omnilatent` encoders | mature |
| Unified representation | a shared space where patterns are comparable | `omnilatent` backbone (RMSNorm/SwiGLU/RoPE) | mature |
| World model | predict next state → *understanding*, not matching | `MMWM`, `hpwm` | partial |
| Memory | explicit (retrievable) + implicit (weights) | `kb/`, replay/EMA/vault in `self_improve/` | partial |
| Controller / agency | select skill, choose tool, decide to stop | `omnilatent/agent/` | early |
| Self-improvement harness | grow capability without regression | `self_improve/` | designed, partial |

### 3.1 Why the world model matters most for "understanding"

Pattern-*matching* asks "what have I seen like this?" A world model asks
"what happens next if I act?" Predictive world models (`MMWM`, `hpwm`) are
the part of this stack that turns correlation into something usable for
planning and counterfactual reasoning. If you want the system to *use* a
pattern (wish 3) rather than just *recognize* it (wish 2), the world model +
controller pair is the locus — not the encoders.

### 3.2 Growth mechanism: hooks, not retraining

The repository's distinctive asset is `LatentNeuralHook`: trainable latent
tokens injected into self-attention behind a near-zero sigmoid gate. This is
a native parameter-isolation method (LoRA / Progressive-Networks family).
It is the *correct* answer to "how do we grow the network":

- New skill → freeze backbone, add a hook, train only the hook.
- Gate ≈ 0 recovers prior behavior exactly → improvements are **additive**,
  not catastrophic.
- Capacity grows on demand (`self_improvement.md §6.3`) instead of by
  retraining a monolith.

This is how a *small* model stays small while still *growing*: it adds narrow
capacity where needed rather than scaling everything.

---

## 4. The harness (this is the actual "how to build AGI" answer)

The harness is the loop that makes the substrate improve. The design in
`docs/self_improvement.md` is the right shape. Its load-bearing ideas:

1. **Self-supervised signal** — reconstruction/prediction quality, no human
   labels in the loop. Lets the system learn from raw streams indefinitely.
2. **Defense-in-depth against forgetting** — DER++ replay + EMA teacher +
   online EWC/SI + hook isolation + snapshot-rollback. No single method
   suffices; the stack does.
3. **A hard quality floor** — frozen probe sets + content-addressed vault +
   rollback-on-regression. *This is the single most AGI-relevant safety
   property in the repo:* it lets capability rise with a guarantee it won't
   silently fall.
4. **Cross-component bootstrapping** — each model labels data for the others
   (`§5`), with snapshot-mediated reads, staleness budgets, confidence gates,
   a real-data majority cap, and an edge-severing divergence guard to keep
   recursive self-labeling from collapsing.

If you internalize one sentence from this document, make it this:

> **A self-improving system amplifies whatever loop it is given. If the loop
> has silent failure modes, self-improvement optimizes the failure.**

That is why §6 comes before any new architecture work.

---

## 5. Closing the loop: agency

Perception + world model + harness produce a system that *learns*. To make
it *act* (wishes 2–3 in deployment), the agent layer must:

- **Select** which learned skill/hook applies to the current latent packet
  (`agent/router.py` — currently deterministic/static; this is where a
  learned policy belongs).
- **Retrieve** relevant explicit memory (`kb/retrieval.py`).
- **Decide to stop** (the router already models `stop_prob`).
- **Constrain side effects** (`runtime.py` already gates side-effecting nodes
  — a genuinely good safety primitive to have early).

The open problem here is not the plumbing; it is the **objective**. The
controller currently acts toward hand-specified goals. *Where the goal comes
from* is the deepest unsolved question in the whole stack (see §7.4). Do not
mistake a working agent runtime for solved agency.

---

## 6. The unglamorous prerequisite: fix the loop (`Audit.md`)

`Audit.md` documents that the current loop does not close cleanly. Until
these are fixed, self-improvement will confidently optimize garbage:

- **Dead loss parameters.** `MultiModalLoss`/`TemporalContextLoss` learnable
  `log_vars` are **not in the optimizer** (`0 of 4`, `0 of 8`). Uncertainty
  weighting is inert.
- **Data cannot reach training.** `StreamingMultiModalDataset` yields objects
  the trainer's collate can't consume; local audio/video samples are
  metadata-only. The harness has nothing real to learn from on those paths.
- **Silent zero-tensor fallbacks.** Failed image/audio/video loads return
  zeros → finite loss on **corrupted targets**. A self-improving system will
  not notice; it will *reinforce* the corruption.
- **Inference bypasses learned skills.** `generate()` runs the backbone
  without the hook manager → trained hooks are ignored at generation time.
- **Direction-blind temporal loss.** `TemporalOrderLoss` is symmetric;
  `(A,B)` and `(B,A)` are identical → it cannot learn temporal direction,
  which is foundational for the world model.

**Recommended sequencing:** make one fully-correct, fully-wired training step
— every loss parameter optimized, every data path delivering real tensors,
every failure raising instead of zero-filling, inference using the same code
path as training — *before* adding any new architecture. A boring correct
loop beats a clever broken one.

---

## 7. What no architecture buys you (open research, not engineering)

These are unsolved for the *entire field*. A harness cannot close them; it
can only *measure* them and prevent regression on them.

1. **Out-of-distribution generalization** — choosing the right pattern on
   genuinely novel input. Interpolation is solved; extrapolation is not.
2. **Compositional / systematic reasoning** — reliably composing known
   skills into new ones rather than memorizing combinations.
3. **Stable open-ended self-improvement** — recursive self-training (the
   `§5` pseudo-label cycles) drifting into mode collapse is an open risk.
   The divergence guards are a hypothesis, not a proof.
4. **Grounded objectives** — *where the goal comes from.* The agent acts, but
   its reward/objective is hand-specified. Autonomy without a principled
   objective is the deepest gap in the stack.

Honesty here is not pessimism. It is what lets you spend effort on the
harness (which you *can* advance) instead of chasing an architecture that
would supposedly "solve intelligence" (which no one knows how to build).

---

## 8. Recommended path (concrete, ordered)

1. **Close the loop.** Burn down `Audit.md`. Definition of done: one training
   step where every loss param is optimized, every data path yields real
   tensors, every load failure raises, and `generate()` shares the training
   forward path. *(Prerequisite for everything below.)*
2. **Instrument the floor.** Land the vault + frozen probe sets +
   rollback-on-regression from `self_improvement.md` §4.6/§6.1 first. You
   want the guarantee in place *before* the system starts changing itself.
3. **Single-component self-improvement.** Replay + EMA on one component
   (`self_improvement.md` phases 3–5). Verify retention on a two-domain toy
   that runs in <60s (`§10.3`).
4. **Hook-based growth.** Wire the plateau detector → `LatentNeuralHook`
   expansion (phase 8). This is the repo's real "grow the network" story.
5. **Multi-component co-training, guards on.** Enable the pseudo-label broker
   with cycles *and* the divergence guard (phase 7). Treat the guard firing
   as a first-class experiment, not an error.
6. **Strengthen the controller.** Replace the static router with a learned
   policy over skills/hooks. This is where wishes 2–3 become real at
   inference time.
7. **Only then** consider architectural changes. By this point you can
   *measure* whether any change helps, because the floor and probes exist.

---

## 9. Answering the four questions directly

1. **Best architecture/harness for a small AGI-capable model?** There is no
   "best" architecture. The right *substrate* is a unified-latent multimodal
   backbone with on-demand hook growth (you have it). The right *harness* is
   self-supervised continual learning with a hard quality floor (you have the
   design). The harness, not the network, is the AGI-relevant artifact.
2. **What is AGI?** A behavioral property — breadth of competence, transfer,
   and closed-loop autonomy — not a topology. Use the transfer/sample-
   efficiency definition to discipline the work.
3. **How to build/grow such a network?** Encoders into one latent space for
   "any input"; decoders for "any output"; a world model + controller to
   *use* patterns; hooks to *grow*; a self-improvement harness with a hard
   floor to *improve without regressing*. Then fix the loop so it optimizes
   signal, not corruption.
4. **Are these the right questions?** Three of four, yes. The fourth — "the
   best architecture for a small omnicompetent model" — contains a category
   error: it asks for an architectural answer to a learning-loop problem, and
   conflates generality (cheap) with omnicompetence (not a small-model
   property). The sharper question is: *"What is the minimal substrate that
   can keep getting more capable on its own, and how do I build the loop that
   grows it without regressing?"* — which this repository is already, mostly,
   trying to answer.

---

## 10. One-paragraph summary

You do not need a new architecture. You need to (a) make the existing
training loop *correct* (`Audit.md`), (b) put a *hard quality floor* under it
(vault + probes + rollback), (c) let it *grow on demand* via
`LatentNeuralHook`, and (d) let it *teach itself* via the self-supervised,
guard-protected co-training loop already designed in
`docs/self_improvement.md`. "AGI-capable" is not a property you install; it
is a trajectory you earn by closing that loop and keeping it from regressing.
The architecture is a substrate. The loop is the intelligence.
