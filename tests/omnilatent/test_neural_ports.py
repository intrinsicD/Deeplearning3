from __future__ import annotations

import torch

from omnilatent.config import OmniLatentConfig
from omnilatent.model.hooks import HookManager, LatentNeuralHook, NeuralPort, NeuralPortManager, NeuralPortSpec
from omnilatent.model.omnilatent import OmniLatentModel


def _config() -> OmniLatentConfig:
    return OmniLatentConfig(
        hidden_dim=64,
        num_layers=4,
        num_heads=4,
        gradient_checkpointing=False,
        vocab_size=256,
        image_size=32,
        image_patch_size=8,
        video_size=32,
        video_patch_size=8,
        video_temporal_patch=2,
        video_max_frames=4,
        audio_n_mels=32,
    )


def test_hook_manager_is_neural_port_manager_compatible() -> None:
    manager = HookManager()

    assert isinstance(manager, NeuralPortManager)
    assert not manager.has_ports()


def test_neural_port_spec_manifest_roundtrip_fields() -> None:
    spec = NeuralPortSpec(
        name="kb_retrieval_v1",
        kind="retrieval",
        version="1",
        latent_dim=64,
        hook_tokens=4,
        target_layers=[1, 2],
        reads=("z_sem", "z_ctx"),
        writes=("z_ctx",),
        side_effects=False,
        trainable=("hook_tokens", "input_projector"),
        compatibility_loss={"latent_alignment": True, "behavior_distillation": True},
        metadata={"owner": "unit"},
    )

    payload = spec.to_dict()

    assert payload["name"] == "kb_retrieval_v1"
    assert payload["reads"] == ["z_sem", "z_ctx"]
    assert payload["writes"] == ["z_ctx"]
    assert payload["compatibility_loss"]["latent_alignment"] is True
    assert payload["metadata"] == {"owner": "unit"}


def test_neural_port_freeze_unfreeze_and_gate_logging() -> None:
    cfg = _config()
    manager = NeuralPortManager()
    port = manager.register_port(NeuralPortSpec(
        name="logger",
        kind="diagnostic",
        latent_dim=cfg.hidden_dim,
        hook_tokens=2,
        target_layers=[0],
        gate_bias_init=0.0,
    ))

    manager.freeze_all()
    assert all(not p.requires_grad for p in port.parameters())
    manager.unfreeze_all()
    assert all(p.requires_grad for p in port.parameters())

    x = torch.randn(2, 5, cfg.hidden_dim)
    manager.begin_forward(batch_size=2)
    y = manager.pre_layer(0, x)
    z = manager.post_layer(0, y)

    assert y.shape[1] == x.shape[1] + 2
    assert z.shape == x.shape
    assert manager.gate_log()["logger.0"] == 0.5


def test_neural_port_and_legacy_hook_compose_in_model_forward() -> None:
    cfg = _config()
    model = OmniLatentModel(cfg)
    port = NeuralPort(NeuralPortSpec(
        name="port",
        kind="retrieval",
        latent_dim=cfg.hidden_dim,
        hook_tokens=2,
        target_layers=[0, 1],
    ))
    legacy = LatentNeuralHook("legacy", 2, cfg.hidden_dim, [2, 3])

    model.hook_manager.register_port(port)
    model.register_hook(legacy)
    text = torch.randint(1, cfg.vocab_size, (2, 8))
    result = model.reconstruct("text", text)

    assert result["output"].shape[:2] == text.shape
    assert not torch.isnan(result["output"]).any()
    assert set(model.list_hooks()) == {"port", "legacy"}


def test_removing_and_readding_neural_port_preserves_non_hook_state_dict() -> None:
    cfg = _config()
    model = OmniLatentModel(cfg)
    before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("hook_manager.")
    }
    port = NeuralPort(NeuralPortSpec(
        name="swap",
        kind="extension",
        latent_dim=cfg.hidden_dim,
        hook_tokens=2,
        target_layers=[0],
    ))

    model.hook_manager.register_port(port)
    removed = model.hook_manager.remove_port("swap")
    assert removed is port
    model.hook_manager.register_port(port)

    after = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("hook_manager.")
    }
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(after[name], before[name]), name

