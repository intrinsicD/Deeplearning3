# Does input-conditioned routing actually help? — an honest measurement

This documents the result of work plan **W6.2**: integrating the learned router
into real training (`RoutedTrainer`) and measuring whether selecting *which*
hooks fire per input beats firing all of them. The harness is
`scripts/diagnostics/routing_ablation.py`; rerun it to reproduce.

**TL;DR — at this scale, on this task, routing does NOT beat always-on.** That
is a real, useful finding, not a failure: it shows the mechanism is correct and
identifies exactly the conditions under which routing would pay off.

## Setup

- Real `OmniLatentModel` (hidden 64, 2 layers), 4 hooks, image self-recon.
- Multi-domain data: 4 domains, each a fixed prototype + noise (specialization
  is *possible* — a hook can specialize per domain).
- Three arms, identical backbone init, backbone trained (`--no-freeze`):
  - `no_hooks` — frozen-hook baseline (what hooks must beat).
  - `always_on` — every hook fires on every input.
  - `routed` — learned router fires the top-k hooks per input.
- 400 steps, lr 3e-3, gate-bias 2.0.

## Result (final self-reconstruction loss, lower is better)

| arm | top_k=1 | top_k=2 | top_k=4 (= all) |
|---|---|---|---|
| `no_hooks` | 1.006 | 1.006 | 1.006 |
| `always_on` | 1.001 | 1.001 | 1.001 |
| `routed` | 1.051 | 1.012 | 0.998 |
| **routed vs always_on** | **+5.0%** | **+1.1%** | **−0.3%** |

The trend is monotonic and exactly what mixture-of-experts theory predicts: the
fewer hooks routing fires, the more it loses to always-on; when it fires all of
them (`top_k = n_hooks`) it converges to always-on (they become equivalent).

## Interpretation (honest)

1. **Hooks barely matter here.** `always_on` beats `no_hooks` by only ~0.5%.
   With a small model and an easy task, the backbone already does most of the
   work, so there is little for hooks — and therefore little for *routing* — to
   contribute.
2. **Sparsity costs capacity with no offsetting benefit.** Routing's win
   condition is avoiding *interference* between experts that genuinely conflict.
   These hooks do not conflict (the model learns to use all of them fine), so
   turning hooks off (top-k < n) just removes useful capacity → routed loses.
3. **The router is not broken.** At `top_k=4` routed ties always-on, confirming
   the gap at low k is the sparsity, not a defective router. The synthetic
   routing probe (W2.4) separately shows the router *can* learn to select the
   correct expert with >0.8 accuracy when selection actually matters.

## What would change the verdict

Routing is expected to help only when one of these holds — none do in this toy:

- **Capacity / compute pressure:** many experts, but you can only afford to run
  a few per input (the real MoE regime). Here all 4 hooks are cheap to run.
- **Genuinely conflicting experts:** a hook that *helps* domain A but *hurts*
  domain B, so firing it everywhere (always-on) is actively harmful. Our hooks
  are benign-or-helpful everywhere.
- **Scale:** larger models/data where interference and capacity limits bite.

## Follow-up: the capacity regime (where routing *does* pay off)

The single-hook-count result above said routing doesn't beat always-on. The
deeper reason is architectural: **LatentNeuralHooks act through attention, which
is already input-conditioned**, so always-on hooks can self-adapt per input —
explicit routing is redundant *for quality*. That points to where routing
*should* help: **compute**. Scaling the hook pool while keeping `top_k` small:

| n_hooks | no_hooks | always_on (all active) | routed top-2 (2 active) | routed vs always_on |
|---|---|---|---|---|
| 12 | 1.270 | 1.295 (+2.0%) | 1.286 | **−0.7%**, 2/12 hooks |
| 16 | 1.270 | 1.298 (+2.2%) | 1.287 | **−0.8%**, 2/16 hooks |

Two consistent effects:

1. **always-on degrades as the pool grows** (1.295 → 1.298, both *worse* than
   no-hooks): firing every hook on every input piles up interference.
2. **routed stays flat** (~1.286) by firing only the top-2 — so it **matches or
   slightly beats** always-on while *each input* uses only 2 of 12–16 hooks.

So routing's win in this codebase is **efficiency and robustness, not quality**
— but the size of the compute win depends on batch size, and the honest number
is smaller than the per-input count suggests.

### How big is the compute win, really? (batch-size matters)

A hook is injected for the **whole batch** if *any* sample selects it. So the
hooks actually run per batch is the *union* of the per-sample top-k picks, not
`top_k`. Measuring the hooks injected per batch (16-hook pool, top-2):

| batch size | hooks injected / batch (routed) | vs always-on (16) |
|---|---|---|
| 1 (per-request serving) | 2.0 | **88% fewer** |
| 16 (batched) | 11.3 | **29% fewer** |

The full `top_k/N` saving only holds at **batch = 1** (per-request inference).
As the batch grows and spans many domains, the injected set approaches the full
pool and the saving erodes toward always-on. My first draft of this doc claimed
a flat "6–8× reduction" — that was the per-input count and overstated the
batched reality; the corrected, batch-aware numbers are above.

(The per-sample masking fix — bug 2 — makes this clean: a sample whose weight
for an injected hook is 0 gets *exact* no-hook output, so "injected for the
batch" never silently changes a non-selecting sample.)

Caveat, stated plainly: the absolute quality gaps are small (~1–2%) at this toy
scale. The **direction** (always-on degrades with pool size; routed holds; real
compute saving at low batch) is consistent and matches MoE theory; the
magnitude would need real scale and per-request serving to matter.

## Bottom line for the project

Phases 2–3 delivered a *correct and trainable* selection/use mechanism, and W6
shows it runs in real training. But the value of routing is **not yet
demonstrated** on this codebase — and now we know precisely what to build to
demonstrate it: a task with capacity pressure or conflicting hooks (the natural
next experiment), or scale. Until then, `always_on` is the honest default and
routing is an available, measured option rather than a proven win.
