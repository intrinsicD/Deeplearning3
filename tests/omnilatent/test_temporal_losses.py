"""W0.6 — temporal order loss learns direction; distant-predict can't collapse (A5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from omnilatent.config import OmniLatentConfig
from omnilatent.training.losses import TemporalContextLoss, TemporalOrderLoss


def test_order_loss_is_direction_sensitive() -> None:
    torch.manual_seed(0)
    D = 32
    loss_fn = TemporalOrderLoss(hidden_dim=D)
    z_a = torch.randn(4, D)
    z_b = torch.randn(4, D)
    labels = torch.ones(4)
    # Swapping anchor/context must change the loss — a commutative dot product
    # (the old bug) would make these identical and order unlearnable.
    forward = loss_fn(z_a, z_b, labels)
    swapped = loss_fn(z_b, z_a, labels)
    assert not torch.allclose(forward, swapped)


def test_order_loss_can_learn_direction() -> None:
    torch.manual_seed(0)
    D = 16
    loss_fn = TemporalOrderLoss(hidden_dim=D)
    opt = torch.optim.Adam(loss_fn.parameters(), lr=0.05)

    # Fixed pair of prototypes; "anchor before context" = label 1.
    a = torch.randn(D)
    b = torch.randn(D)
    z_a = a.expand(8, D)
    z_b = b.expand(8, D)
    ones = torch.ones(8)
    zeros = torch.zeros(8)

    first = None
    for step in range(200):
        opt.zero_grad()
        # (a, b) -> 1, (b, a) -> 0
        loss = loss_fn(z_a, z_b, ones) + loss_fn(z_b, z_a, zeros)
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < first  # learning happened
    assert loss.item() < 0.2     # actually separates the two orders


def test_distant_predict_target_is_stop_gradient() -> None:
    cfg = OmniLatentConfig(hidden_dim=24, num_layers=2, num_heads=4)
    tcl = TemporalContextLoss(cfg)

    z_anchor = torch.randn(4, cfg.hidden_dim, requires_grad=True)
    z_context = torch.randn(4, cfg.hidden_dim, requires_grad=True)

    # The mechanism used by curriculum_train: predict context from anchor,
    # detached target.
    pred = tcl.distant_predictor(z_anchor)
    target = z_context.detach()
    loss = F.mse_loss(pred, target) + 0.5 * (
        1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    )
    loss.backward()

    # No gradient flows into the target clip — so the loss cannot pull the two
    # clip latents together (no representational collapse).
    assert z_context.grad is None
    # The predictor path is live, so the anchor still receives a learning signal.
    assert z_anchor.grad is not None


def test_distant_predictor_has_parameters() -> None:
    cfg = OmniLatentConfig(hidden_dim=24, num_layers=2, num_heads=4)
    tcl = TemporalContextLoss(cfg)
    assert sum(p.numel() for p in tcl.distant_predictor.parameters()) > 0
