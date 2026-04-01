from __future__ import annotations

import torch

from MMWM.config import ModelConfig, build_model
from MMWM.containers import LatentState, ModelOutput
from MMWM.curriculum import default_curriculum_phases
from MMWM.losses import WorldModelLoss


def _dummy_output(batch_size: int = 2, latent_dim: int = 8) -> ModelOutput:
    pred = LatentState(
        z_sem=torch.randn(batch_size, latent_dim),
        z_dyn=torch.randn(batch_size, latent_dim),
        z_ctrl=torch.randn(batch_size, latent_dim),
        z_mem=torch.randn(batch_size, latent_dim),
        extras={},
    )
    tgt = LatentState(
        z_sem=torch.randn(batch_size, latent_dim),
        z_dyn=torch.randn(batch_size, latent_dim),
        z_ctrl=torch.randn(batch_size, latent_dim),
        z_mem=torch.randn(batch_size, latent_dim),
        extras={},
    )
    return ModelOutput(
        current_latent=pred,
        predicted_next_latent=pred,
        target_next_latent=tgt,
        next_memory=None,  # type: ignore[arg-type]
        decoder_outputs={},
        aux={"regularizer_total": torch.tensor(0.1)},
    )


def test_kendall_uncertainty_loss_produces_logvars() -> None:
    loss_fn = WorldModelLoss(learned_uncertainty=True)
    losses = loss_fn(_dummy_output(), {})
    assert "total_loss" in losses
    assert "latent_sem_loss_log_var" in losses


def test_curriculum_phases_are_ordered() -> None:
    phases = default_curriculum_phases()
    assert len(phases) == 5
    assert all(phases[i].until_step < phases[i + 1].until_step for i in range(len(phases) - 1))


def test_slot_mamba_mod_model_builds_and_runs_forward() -> None:
    cfg = ModelConfig(
        encoder_name="slot_multimodal",
        encoder_kwargs={
            "text_vocab_size": 256,
            "text_embed_dim": 32,
            "vector_input_dim": 16,
            "image_channels": 3,
            "hidden_dim": 32,
            "n_slots": 4,
            "slot_dim": 16,
            "slot_iters": 2,
            "merge_threshold": 0.8,
        },
        latent_projector_kwargs={"input_dim": 32, "latent_dim": 16, "use_batchnorm": False},
        memory_name="mamba_ssm",
        memory_kwargs={"input_dim": 32, "hidden_dim": 16},
        action_encoder_kwargs={"action_dim": 8, "action_embed_dim": 16},
        conditioner_kwargs={"latent_dim": 64, "action_dim": 16, "memory_dim": 16, "out_dim": 64},
        transition_core_name="mod_recurrent_attnres_transformer",
        transition_core_kwargs={"input_dim": 64, "hidden_dim": 64, "recurrent_steps": 2},
        prediction_head_kwargs={"hidden_dim": 64, "latent_dim": 16},
        decoder_configs=[],
    )
    model = build_model(cfg)

    obs = {
        "text": torch.randint(0, 256, (2, 6)),
        "vector": torch.randn(2, 16),
        "image": torch.randn(2, 3, 32, 32),
    }
    from MMWM.containers import ObservationPacket

    packet = ObservationPacket(modalities=obs)
    action = torch.randn(2, 8)
    out = model(packet, action, obs_tp1=packet)
    assert out.predicted_next_latent.z_sem.shape == (2, 16)
