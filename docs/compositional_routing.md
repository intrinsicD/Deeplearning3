# Do two singly-learned hooks compose on an unseen task?

Work plan **W5.4** (research lane). Harness:
`scripts/diagnostics/compositional_routing.py`.

## Setup

Three image→image *transform* tasks over a shared base prototype P, differing by
an additive output bias:

- **red**: input `P + red_signal`, target `P + red_bias`
- **blue**: input `P + blue_signal`, target `P + blue_bias`
- **compose** (held out): input `P + red_signal + blue_signal`,
  target `P + red_bias + blue_bias`

Hook **A** is trained only on red, hook **B** only on blue — neither ever sees
the compose task. The question: on **unseen compose inputs**, does activating
*both* hooks beat the best single hook, and do both hooks carry positive
counterfactual credit (W5.2)?

## Result

| | trainable backbone (default) | frozen backbone (hooks must carry the skill) |
|---|---|---|
| loss_none | 0.2754 | 0.5473 |
| best single (A or B) | 0.27525 | 0.54464 |
| both active | 0.27517 | 0.54375 |
| counterfactual credit A / B | +0.0003 / +0.0009 | **+0.0005 / +0.0006** |
| composition gap (best-single − both) | +0.00008 (≈0) | **+0.00089 (both wins)** |
| verdict | no composition signal | **composition helps (directionally)** |

## Interpretation (honest)

1. **Composition works directionally — but only when hooks must carry the
   skill.** With the backbone frozen (after a base-reconstruction warmup), both
   independently-trained hooks carry *positive* counterfactual credit on the
   unseen compose task, and firing both beats either alone. Two skills learned
   in isolation combine on a task neither was trained for. That is real
   compositional generalization.
2. **A trainable backbone erases the effect.** In the default setup the shared
   backbone simply *absorbs* both transforms during single-skill training, so
   the hooks become vestigial and the compose gap collapses to ~0. The skill
   doesn't live in the hooks, so there's nothing to compose.
3. **The magnitude is tiny either way** (~0.1–0.2% of loss; credits ~5e-4).
   This is the same weak-output-leverage of `LatentNeuralHook`s seen in
   `routing_ablation.md`: attention-injected tokens move the decoded output
   only slightly at this scale.

## Bottom line

The *capability* — composing independently-acquired skills — is present in the
architecture and measurable, but **weak and fragile**: it shows up only when the
backbone is prevented from absorbing the skills, and even then the effect is
fractions of a percent. To make composition a load-bearing property you would
need hooks with real output leverage (larger hooks / deeper injection / a
backbone that genuinely defers to them) and real scale. As-is, treat
compositional routing as a demonstrated-but-marginal behaviour, not a
dependable mechanism.
