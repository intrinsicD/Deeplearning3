"""W5.2 — counterfactual attribution credit (credit v3)."""

from __future__ import annotations

import torch

from omnilatent.agent.registry import ExpertRegistry
from omnilatent.agent.router import LearnedLatentRouter
from omnilatent.config import OmniLatentConfig
from omnilatent.model.hooks import LatentNeuralHook
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.routed_trainer import (
    RoutedTrainer,
    _per_sample_recon_loss,
    counterfactual_hook_credit,
)


def _model(n_hooks: int = 2) -> OmniLatentModel:
    cfg = OmniLatentConfig(hidden_dim=64, num_layers=2, num_heads=4)
    model = OmniLatentModel(cfg).eval()
    for i in range(n_hooks):
        model.register_hook(
            LatentNeuralHook(
                name=f"skill{i}", num_tokens=4, dim=cfg.hidden_dim,
                target_layers=[0, 1], gate_bias_init=1.0,
            )
        )
    return model


def _img(model: OmniLatentModel, b: int = 4) -> torch.Tensor:
    return torch.randn(b, 3, model.config.image_size, model.config.image_size)


def test_credit_equals_measured_loss_deltas() -> None:
    # The credit for hook j must equal (no-hook loss) - (only-hook-j loss),
    # per sample. Recompute the deltas independently and compare.
    torch.manual_seed(0)
    model = _model(2)
    img = _img(model)
    credit, names = counterfactual_hook_credit(model, "image", img)
    assert names == ["skill0", "skill1"]
    assert credit.shape == (img.shape[0], 2)

    mgr = model.hook_manager
    mgr.set_route_weights({"skill0": 0.0, "skill1": 0.0})
    base = _per_sample_recon_loss(model, "image", img)
    for j, name in enumerate(names):
        w = {"skill0": 0.0, "skill1": 0.0}
        w[name] = 1.0
        mgr.set_route_weights(w)
        delta = base - _per_sample_recon_loss(model, "image", img)
        torch.testing.assert_close(credit[:, j], delta)
    mgr.set_route_weights(None)


def test_no_hooks_gives_empty_credit() -> None:
    cfg = OmniLatentConfig(hidden_dim=64, num_layers=2, num_heads=4)
    model = OmniLatentModel(cfg).eval()
    credit, names = counterfactual_hook_credit(model, "image", _img(model))
    assert names == [] and credit.shape == (4, 0)


def test_helpful_hook_gets_positive_credit() -> None:
    # Overfit hook skill0 on a fixed batch so it genuinely reduces that batch's
    # loss; its counterfactual credit must then exceed the untrained hook's.
    torch.manual_seed(0)
    model = _model(2)
    reg = ExpertRegistry(key_dim=model.config.hidden_dim)
    reg.sync_hooks(model.hook_manager)
    tr = RoutedTrainer(model, model.config, mode="routed", router=LearnedLatentRouter(reg, input_dim=model.config.hidden_dim, top_k=1))
    img = _img(model, b=4)

    # Train ONLY skill0.
    model.hook_manager.set_route_weights({"skill0": 1.0, "skill1": 0.0})
    opt = torch.optim.Adam([p for h in model.hook_manager.hooks.values() for p in h.parameters()], lr=5e-3)
    for _ in range(80):
        opt.zero_grad()
        out = model.reconstruct("image", img)["output"]
        ((out - img) ** 2).mean().backward()
        opt.step()
    model.hook_manager.set_route_weights(None)

    credit, names = counterfactual_hook_credit(model, "image", img)
    # skill0 reduces loss (positive, larger) more than the untrained skill1.
    assert credit[:, 0].mean() > credit[:, 1].mean()
    assert credit[:, 0].mean() > 0


def test_step_counterfactual_routes_toward_helpful_hook() -> None:
    torch.manual_seed(0)
    model = _model(2)
    reg = ExpertRegistry(key_dim=model.config.hidden_dim)
    reg.sync_hooks(model.hook_manager)
    router = LearnedLatentRouter(reg, input_dim=model.config.hidden_dim, top_k=1)
    tr = RoutedTrainer(model, model.config, mode="routed", router=router, lr=5e-3)
    img = _img(model, b=4)

    # Make skill0 the genuinely-helpful hook on this batch.
    model.hook_manager.set_route_weights({"skill0": 1.0, "skill1": 0.0})
    hp = [p for h in model.hook_manager.hooks.values() for p in h.parameters()]
    opt = torch.optim.Adam(hp, lr=5e-3)
    for _ in range(80):
        opt.zero_grad()
        out = model.reconstruct("image", img)["output"]
        ((out - img) ** 2).mean().backward()
        opt.step()
    model.hook_manager.set_route_weights(None)

    # Train the router via counterfactual credit on this batch.
    for _ in range(60):
        tr.step_counterfactual({"image": img})

    # The router now prefers skill0 (the counterfactually-better hook).
    with torch.no_grad():
        w = router.forward(model.encode("image", img)[:, 1:].mean(dim=1))["weights"].mean(dim=0)
    ids = reg.ids()
    w0 = w[ids.index("hook:skill0")]
    w1 = w[ids.index("hook:skill1")]
    assert w0 > w1
