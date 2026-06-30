"""Compositional use: do two singly-learned hooks combine on an unseen task?

Work plan W5.4 (research lane). Three image→image *transform* tasks share a base
prototype P and differ by an additive output bias:

    red    : input = P + red_signal,              target = P + red_bias
    blue   : input = P + blue_signal,             target = P + blue_bias
    compose: input = P + red_signal + blue_signal, target = P + red_bias + blue_bias

Hook A is trained ONLY on red, hook B ONLY on blue (each sees its own signal/bias
and never the compose task). The question: on **unseen compose inputs**, does
activating BOTH hooks beat the best single hook? If the two skills compose, both
hooks should carry positive counterfactual credit on compose inputs and
``loss(both) < min(loss(A), loss(B))``.

The harness assumes nothing — it reports the gap, whatever its sign.

Run::

    python -m scripts.diagnostics.compositional_routing --steps 250
"""

from __future__ import annotations

import argparse

import torch

from omnilatent.config import OmniLatentConfig
from omnilatent.model.hooks import LatentNeuralHook
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.routed_trainer import counterfactual_hook_credit


def _bias(seed: int, size: int, channel: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    b = torch.zeros(3, size, size)
    b[channel] = 0.8 + 0.2 * torch.rand(size, size, generator=g)
    return b


def make_domains(size: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(3, size, size, generator=g)
    red_sig = torch.zeros(3, size, size); red_sig[0] = 0.3
    blue_sig = torch.zeros(3, size, size); blue_sig[2] = 0.3
    red_bias = _bias(1, size, channel=0)
    blue_bias = _bias(2, size, channel=2)
    return {
        "base": base, "red_sig": red_sig, "blue_sig": blue_sig,
        "red_bias": red_bias, "blue_bias": blue_bias,
    }


def _sample(d, kind: str, b: int, noise: float, gen: torch.Generator):
    size = d["base"].shape[-1]
    x = d["base"].unsqueeze(0) + noise * torch.randn(b, 3, size, size, generator=gen)
    if kind == "red":
        return x + d["red_sig"], x + d["red_bias"]
    if kind == "blue":
        return x + d["blue_sig"], x + d["blue_bias"]
    # compose
    return x + d["red_sig"] + d["blue_sig"], x + d["red_bias"] + d["blue_bias"]


def _model(size: int) -> OmniLatentModel:
    cfg = OmniLatentConfig(hidden_dim=64, num_layers=2, num_heads=4, image_size=size, image_patch_size=8)
    model = OmniLatentModel(cfg)
    for name in ("A", "B"):
        model.register_hook(
            LatentNeuralHook(name=name, num_tokens=4, dim=cfg.hidden_dim,
                             target_layers=[0, 1], gate_bias_init=2.0)
        )
    return model


def _train_hook(model, d, kind, hook_name, steps, batch, noise, seed, freeze_backbone=False):
    """Train one hook (and optionally the backbone) on one single-skill domain."""
    model.train()
    model.hook_manager.set_route_weights({"A": 0.0, "B": 0.0, hook_name: 1.0})
    if freeze_backbone:
        hook_ids = {id(p) for h in model.hook_manager.hooks.values() for p in h.parameters()}
        for p in model.parameters():
            p.requires_grad_(id(p) in hook_ids)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=3e-3)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x, y = _sample(d, kind, batch, noise, gen)
        out = model("image", x, "image", y)["output"]
        opt.zero_grad(); ((out - y) ** 2).mean().backward(); opt.step()
    model.hook_manager.set_route_weights(None)


@torch.no_grad()
def _loss(model, x, y, weights) -> float:
    model.hook_manager.set_route_weights(weights)
    try:
        out = model("image", x, "image", y)["output"]
        return float(((out - y) ** 2).mean().item())
    finally:
        model.hook_manager.set_route_weights(None)


def _warmup_backbone(model, d, steps, batch, noise, seed):
    """Pre-train the backbone on base reconstruction (no hooks), then freeze it,
    so the hooks must carry the domain-specific transforms."""
    model.train()
    model.hook_manager.set_route_weights({"A": 0.0, "B": 0.0})
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(seed)
    size = d["base"].shape[-1]
    for _ in range(steps):
        x = d["base"].unsqueeze(0) + noise * torch.randn(batch, 3, size, size, generator=gen)
        out = model("image", x, "image", x)["output"]
        opt.zero_grad(); ((out - x) ** 2).mean().backward(); opt.step()
    model.hook_manager.set_route_weights(None)


def run(steps: int = 250, batch: int = 16, size: int = 16, noise: float = 0.05,
        seed: int = 0, freeze_backbone: bool = False) -> dict:
    torch.manual_seed(seed)
    d = make_domains(size, seed)
    model = _model(size)
    if freeze_backbone:
        _warmup_backbone(model, d, steps, batch, noise, seed)
    # Train A on red, then B on blue (each only on its own domain).
    _train_hook(model, d, "red", "A", steps, batch, noise, seed + 1, freeze_backbone)
    _train_hook(model, d, "blue", "B", steps, batch, noise, seed + 2, freeze_backbone)

    # Evaluate on UNSEEN compose inputs.
    gen = torch.Generator().manual_seed(seed + 99)
    x, y = _sample(d, "compose", 64, noise, gen)
    none = _loss(model, x, y, {"A": 0.0, "B": 0.0})
    a = _loss(model, x, y, {"A": 1.0, "B": 0.0})
    b = _loss(model, x, y, {"A": 0.0, "B": 1.0})
    both = _loss(model, x, y, {"A": 1.0, "B": 1.0})
    credit, names = counterfactual_hook_credit(model, "image", x)  # vs no-hook baseline
    return {
        "loss_none": none, "loss_A": a, "loss_B": b, "loss_both": both,
        "credit_A": float(credit[:, names.index("A")].mean()),
        "credit_B": float(credit[:, names.index("B")].mean()),
        "best_single": min(a, b),
        "composition_gap": min(a, b) - both,  # >0 ⇒ both beats best single
    }


def _format(r: dict) -> str:
    lines = ["", "Compositional routing — loss on UNSEEN compose task (lower better)", "-" * 64]
    for k in ("loss_none", "loss_A", "loss_B", "loss_both"):
        lines.append(f"  {k:<12} {r[k]:.5f}")
    lines += [
        "-" * 64,
        f"  counterfactual credit on compose:  A={r['credit_A']:+.5f}  B={r['credit_B']:+.5f}",
        f"  best single = {r['best_single']:.5f}   both = {r['loss_both']:.5f}",
    ]
    gap = r["composition_gap"]
    if gap > 1e-4 and r["credit_A"] > 0 and r["credit_B"] > 0:
        verdict = f"COMPOSITION HELPS: both-active beats best-single by {gap:+.5f}; both hooks contribute."
    elif gap > 1e-4:
        verdict = f"both-active beats best-single by {gap:+.5f}, but only one hook carries positive credit."
    else:
        verdict = f"NO composition benefit (gap {gap:+.5f}): combining the skills does not beat the best single."
    lines += ["-" * 64, f"  verdict: {verdict}", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Compositional routing benchmark (W5.4)")
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--freeze-backbone", action="store_true",
                   help="warm up + freeze the backbone so hooks must carry the skill")
    args = p.parse_args()
    print(_format(run(steps=args.steps, batch=args.batch, size=args.size, seed=args.seed,
                      freeze_backbone=args.freeze_backbone)))


if __name__ == "__main__":
    main()
