"""W6.1 — RoutedTrainer integrates the router into real model training."""

from __future__ import annotations

import math

import torch

from omnilatent.agent.registry import ExpertRegistry
from omnilatent.agent.router import LearnedLatentRouter
from omnilatent.config import OmniLatentConfig
from omnilatent.model.hooks import LatentNeuralHook
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.routed_trainer import RoutedTrainer


def _model(n_hooks: int = 3) -> OmniLatentModel:
    cfg = OmniLatentConfig(hidden_dim=64, num_layers=2, num_heads=4)
    model = OmniLatentModel(cfg)
    for i in range(n_hooks):
        model.register_hook(
            LatentNeuralHook(
                name=f"skill{i}", num_tokens=4, dim=cfg.hidden_dim,
                target_layers=[0, 1], gate_bias_init=0.0,
            )
        )
    return model


def _router(model: OmniLatentModel, top_k: int = 1) -> LearnedLatentRouter:
    reg = ExpertRegistry(key_dim=model.config.hidden_dim)
    reg.sync_hooks(model.hook_manager)
    return LearnedLatentRouter(reg, input_dim=model.config.hidden_dim, top_k=top_k)


def _batch(cfg: OmniLatentConfig, b: int = 4) -> dict:
    return {"image": torch.randn(b, 3, cfg.image_size, cfg.image_size)}


def test_routed_step_runs_and_is_finite() -> None:
    model = _model()
    trainer = RoutedTrainer(model, model.config, modality="image", mode="routed", router=_router(model))
    loss = trainer.step(_batch(model.config))
    assert math.isfinite(loss)


def test_router_receives_gradients_in_real_training() -> None:
    model = _model()
    router = _router(model)
    trainer = RoutedTrainer(model, model.config, modality="image", mode="routed", router=router)
    trainer.step(_batch(model.config))
    # The router is trained by the task loss through the gate scaling.
    assert router.query_proj.weight.grad is not None
    assert any(p.grad is not None for p in router.registry.parameters())


def test_backbone_is_frozen() -> None:
    model = _model()
    RoutedTrainer(model, model.config, mode="routed", router=_router(model))
    # Encoders/decoders frozen; hooks trainable.
    enc_grad = [p.requires_grad for p in model.encoders.parameters()]
    assert not any(enc_grad)
    hook_grad = [p.requires_grad for h in model.hook_manager.hooks.values() for p in h.parameters()]
    assert all(hook_grad)


def test_all_modes_run() -> None:
    for mode in ("routed", "always_on", "no_hooks"):
        model = _model()
        router = _router(model) if mode == "routed" else None
        trainer = RoutedTrainer(model, model.config, mode=mode, router=router)
        loss = trainer.step(_batch(model.config))
        assert math.isfinite(loss)


def test_routed_training_reduces_loss_on_fixed_batch() -> None:
    torch.manual_seed(0)
    model = _model()
    trainer = RoutedTrainer(model, model.config, mode="routed", router=_router(model), lr=3e-3)
    batch = _batch(model.config, b=4)
    first = trainer.step(batch)
    for _ in range(60):
        last = trainer.step(batch)
    # Overfitting a single batch must drive the loss down.
    assert last < first
