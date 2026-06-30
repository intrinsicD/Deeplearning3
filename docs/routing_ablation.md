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

## Bottom line for the project

Phases 2–3 delivered a *correct and trainable* selection/use mechanism, and W6
shows it runs in real training. But the value of routing is **not yet
demonstrated** on this codebase — and now we know precisely what to build to
demonstrate it: a task with capacity pressure or conflicting hooks (the natural
next experiment), or scale. Until then, `always_on` is the honest default and
routing is an available, measured option rather than a proven win.
