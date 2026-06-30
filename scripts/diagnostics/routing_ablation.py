"""Routing ablation: does input-conditioned hook selection beat always-on?

Work plan W6.2. Trains three arms of the *same* frozen-backbone OmniLatent model
on a controlled multi-domain image task and reports final self-reconstruction
loss per arm:

* ``no_hooks``   — frozen backbone only (baseline: what hooks must beat).
* ``always_on``  — every hook fires on every input (pre-routing default).
* ``routed``     — a learned router picks the top-k hooks per input.

The task is built so specialization is *possible* (each domain is a distinct
prototype + noise; a domain-specialized hook can bias reconstruction toward its
prototype). Always-on must fire every hook on every input, risking
interference; routing can fire just the relevant one. Whether that advantage
actually materializes at this scale is exactly what the numbers report — the
harness assumes nothing.

Run::

    python -m scripts.diagnostics.routing_ablation --domains 4 --hooks 4 --steps 400
"""

from __future__ import annotations

import argparse

import torch

from omnilatent.agent.registry import ExpertRegistry
from omnilatent.agent.router import LearnedLatentRouter
from omnilatent.config import OmniLatentConfig
from omnilatent.model.hooks import LatentNeuralHook
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.routed_trainer import RoutedTrainer


def make_multidomain_dataset(
    n_domains: int,
    image_size: int,
    n_per_domain: int = 64,
    noise: float = 0.1,
    seed: int = 0,
) -> list[dict]:
    """Fixed prototype per domain; samples are prototype + small noise."""
    gen = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(n_domains, 3, image_size, image_size, generator=gen)
    samples = []
    for d in range(n_domains):
        for _ in range(n_per_domain):
            img = prototypes[d] + noise * torch.randn(3, image_size, image_size, generator=gen)
            samples.append(img)
    return samples


def _batches(samples: list, batch_size: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    while True:
        idx = torch.randint(0, len(samples), (batch_size,), generator=gen)
        yield {"image": torch.stack([samples[int(i)] for i in idx])}


def _fresh_model(cfg: OmniLatentConfig, n_hooks: int, seed: int, gate_bias: float) -> OmniLatentModel:
    torch.manual_seed(seed)  # identical backbone init across arms
    model = OmniLatentModel(cfg)
    for i in range(n_hooks):
        model.register_hook(
            LatentNeuralHook(
                name=f"skill{i}", num_tokens=4, dim=cfg.hidden_dim,
                target_layers=list(range(cfg.num_layers)), gate_bias_init=gate_bias,
            )
        )
    return model


def run_arm(mode, cfg, samples, n_hooks, steps, batch_size, top_k, seed, freeze, gate_bias) -> float:
    model = _fresh_model(cfg, n_hooks, seed, gate_bias)
    router = None
    if mode == "routed":
        reg = ExpertRegistry(key_dim=cfg.hidden_dim)
        reg.sync_hooks(model.hook_manager)
        router = LearnedLatentRouter(reg, input_dim=cfg.hidden_dim, top_k=top_k)
    trainer = RoutedTrainer(
        model, cfg, modality="image", mode=mode, router=router, lr=3e-3,
        freeze_backbone=freeze,
    )

    train_iter = _batches(samples, batch_size, seed=seed + 1)
    for _ in range(steps):
        trainer.step(next(train_iter))

    eval_iter = _batches(samples, batch_size, seed=seed + 999)
    loss = trainer.evaluate(eval_iter, batches=16)
    return loss, trainer.injected_per_batch


def run_ablation(
    n_domains: int = 4,
    n_hooks: int = 4,
    steps: int = 400,
    batch_size: int = 16,
    top_k: int = 1,
    image_size: int = 32,
    seed: int = 0,
    freeze_backbone: bool = True,
    gate_bias: float = 0.0,
) -> dict[str, float]:
    cfg = OmniLatentConfig(
        hidden_dim=64, num_layers=2, num_heads=4,
        image_size=image_size, image_patch_size=8,
    )
    samples = make_multidomain_dataset(n_domains, image_size, seed=seed)
    results = {}
    for mode in ("no_hooks", "always_on", "routed"):
        loss, injected = run_arm(
            mode, cfg, samples, n_hooks, steps, batch_size, top_k, seed,
            freeze_backbone, gate_bias,
        )
        results[mode] = {"loss": loss, "injected": injected}
    return results


def _format(results: dict[str, dict], n_hooks: int, top_k: int) -> str:
    base = results["no_hooks"]["loss"]
    logical = {"no_hooks": 0, "always_on": n_hooks, "routed": min(top_k, n_hooks)}
    lines = ["", "Routing ablation — final self-reconstruction loss (lower is better)", "-" * 78]
    lines.append(f"  {'arm':<12} {'loss':>9}   {'hooks/input':>11}   {'hooks injected/batch':>20}")
    for mode in ("no_hooks", "always_on", "routed"):
        v = results[mode]["loss"]
        inj = results[mode]["injected"]
        delta = "" if mode == "no_hooks" else f"  ({(v - base) / base * 100:+.1f}% vs none)"
        lines.append(f"  {mode:<12} {v:>9.5f}   {logical[mode]:>11}   {inj:>20.1f}{delta}")
    r, a = results["routed"]["loss"], results["always_on"]["loss"]
    r_inj, a_inj = results["routed"]["injected"], results["always_on"]["injected"]
    quality = (
        "routed BEATS always_on" if r < a * 0.99
        else "routed LOSES to always_on" if r > a * 1.01
        else "routed ~ always_on (tie)"
    )
    lines += ["-" * 78, f"  quality: {quality}  (routed {(r - a) / a * 100:+.1f}% vs always_on)"]
    # Honest compute verdict: compare ACTUAL hooks injected per batch, not the
    # logical per-input count.
    if r <= a * 1.01 and r_inj < a_inj * 0.95:
        lines.append(
            f"  COMPUTE WIN: equal/better quality injecting {r_inj:.1f} vs {a_inj:.1f} "
            f"hooks/batch ({(1 - r_inj / a_inj) * 100:.0f}% fewer)."
        )
    elif r <= a * 1.01:
        lines.append(
            f"  NO batched-compute win: routed injects {r_inj:.1f} hooks/batch "
            f"(~{a_inj:.1f} for always_on) — the per-input top_k saving collapses "
            f"because some sample selects most hooks. Real saving needs batch=1."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Routing ablation (W6.2)")
    p.add_argument("--domains", type=int, default=4)
    p.add_argument("--hooks", type=int, default=4)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-freeze", action="store_true", help="train the backbone too")
    p.add_argument("--gate-bias", type=float, default=0.0, help="hook gate bias init")
    args = p.parse_args()
    results = run_ablation(
        n_domains=args.domains, n_hooks=args.hooks, steps=args.steps,
        batch_size=args.batch_size, top_k=args.top_k, image_size=args.image_size, seed=args.seed,
        freeze_backbone=not args.no_freeze, gate_bias=args.gate_bias,
    )
    print(_format(results, n_hooks=args.hooks, top_k=args.top_k))


if __name__ == "__main__":
    main()
