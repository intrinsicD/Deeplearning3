"""Phase 3 acceptance test: two-domain non-forgetting on the
gaussian_encoder plugin.

Design doc verification line for phase 3:
    "Train a plugin for 100 steps with replay disabled vs. enabled on a
     two-domain toy task; replay version retains domain-1 accuracy."

The toy task is reconstruction of two disjoint distributions:

- Domain A: images with intensity centered at 0.25.
- Domain B: images with intensity centered at 0.75.

Both are 1-channel 28×28 (matches the gaussian autoencoder's input
shape). Without replay, training for 50 steps on A then 50 steps on B
should noticeably degrade A's reconstruction. With replay enabled, A's
loss should stay close to its pre-B baseline.
"""

from __future__ import annotations

import torch

from scripts.training.self_improve import ReplayBank
from scripts.training.self_improve.plugins import get_plugin

GaussianEncoderPlugin = get_plugin("gaussian_encoder")


def _make_domain(seed: int, center: float, n: int = 64, batch: int = 8) -> list[torch.Tensor]:
    """Return ``n`` mini-batches of shape ``(batch, 1, 28, 28)`` drawn
    from a uniform distribution centered at ``center`` with width 0.4.
    """
    g = torch.Generator().manual_seed(seed)
    half_width = 0.2
    lo = max(0.0, center - half_width)
    hi = min(1.0, center + half_width)
    return [
        lo + (hi - lo) * torch.rand(batch, 1, 28, 28, generator=g)
        for _ in range(n)
    ]


@torch.no_grad()
def _eval_loss(plugin: GaussianEncoderPlugin, batches: list[torch.Tensor]) -> float:
    model = plugin.model
    was_training = model.training
    model.eval()
    try:
        total = 0.0
        n = 0
        for x in batches:
            x = x.to(plugin.device)
            x_hat, _ = model(x)
            total += float(torch.nn.functional.mse_loss(x_hat, x).item()) * x.shape[0]
            n += x.shape[0]
        return total / max(n, 1)
    finally:
        model.train(was_training)


def _train_two_domain(
    *,
    attach_replay: bool,
    steps_per_domain: int = 50,
) -> tuple[float, float]:
    """Run the two-domain protocol. Returns ``(domain_a_loss_final,
    domain_b_loss_final)`` after training A then B.
    """
    torch.manual_seed(0)

    plugin = GaussianEncoderPlugin(in_ch=1, image_size=28, lr=2e-3)
    if attach_replay:
        bank = ReplayBank(capacity=256, seed=0)
        plugin.attach_replay(bank, weight=1.0, der_weight=1.0, batch_size=8)
        # NB: distill_weight defaults to 0.5; for the toy we leave EMA off
        # so the test isolates the replay effect.

    domain_a = _make_domain(seed=1, center=0.25, n=steps_per_domain)
    domain_b = _make_domain(seed=2, center=0.75, n=steps_per_domain)

    # Phase 1: train on A only.
    for batch in domain_a:
        plugin.train_step(batch)

    # Phase 2: train on B only. With replay, domain-A samples that were
    # inserted during phase 1 keep flowing into the gradient.
    for batch in domain_b:
        plugin.train_step(batch)

    # Hold-out eval batches (different seeds from training).
    eval_a = _make_domain(seed=101, center=0.25, n=4)
    eval_b = _make_domain(seed=102, center=0.75, n=4)
    return _eval_loss(plugin, eval_a), _eval_loss(plugin, eval_b)


def test_replay_reduces_forgetting_on_two_domain_toy() -> None:
    """The headline phase-3 assertion: replay-enabled training retains
    significantly more of domain A's reconstruction quality after the
    domain-B phase than replay-disabled training.

    Tolerances are loose so the test isn't flaky on toolchain or seed
    perturbations; the qualitative effect (replay << no-replay on
    domain-A loss) is what we're checking.
    """
    a_no_replay, _ = _train_two_domain(attach_replay=False)
    a_replay, _ = _train_two_domain(attach_replay=True)

    # Replay must improve domain-A retention by a meaningful margin.
    # In practice on this toy the no-replay degradation is several×
    # worse than the replay version.
    assert a_replay < a_no_replay, (
        f"replay did not reduce forgetting: a_replay={a_replay:.4f} "
        f"a_no_replay={a_no_replay:.4f}"
    )
    # And the relative improvement should be substantial (>=25%).
    improvement = (a_no_replay - a_replay) / a_no_replay
    assert improvement >= 0.25, (
        f"replay improvement only {improvement:.1%}; expected >= 25%"
    )
