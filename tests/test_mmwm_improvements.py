from __future__ import annotations

import torch

from MMWM.config import ModelConfig, build_model
from MMWM.containers import LatentState, ModelOutput, ObservationPacket
from MMWM.curriculum import (
    AdaptiveCurriculumScheduler,
    default_curriculum_phases,
    relative_curriculum_phases,
)
from MMWM.evaluation import (
    EvaluationSuite,
    LatentPredictionMetrics,
    ReconstructionMetrics,
    RolloutMetrics,
)
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
    packet = ObservationPacket(modalities=obs)
    action = torch.randn(2, 8)
    out = model(packet, action, obs_tp1=packet)
    assert out.predicted_next_latent.z_sem.shape == (2, 16)


# ============================================================
# New tests for concern fixes
# ============================================================


def test_adaptive_role_split_projector() -> None:
    """Concern 1: Adaptive role split with learnable capacity gating."""
    cfg = ModelConfig(
        encoder_name="simple_multimodal",
        encoder_kwargs={"text_vocab_size": 256, "text_embed_dim": 32, "vector_input_dim": 16, "image_channels": 3, "hidden_dim": 32},
        latent_projector_name="adaptive_role_split_mlp",
        latent_projector_kwargs={"input_dim": 32, "latent_dim": 16, "intermediate_dim": 64, "use_batchnorm": False},
        memory_kwargs={"input_dim": 32, "hidden_dim": 16},
        action_encoder_kwargs={"action_dim": 8, "action_embed_dim": 16},
        conditioner_kwargs={"latent_dim": 64, "action_dim": 16, "memory_dim": 16, "out_dim": 64},
        transition_core_name="mlp",
        transition_core_kwargs={"input_dim": 64, "hidden_dim": 64},
        prediction_head_kwargs={"hidden_dim": 64, "latent_dim": 16},
        decoder_configs=[],
    )
    model = build_model(cfg)
    packet = ObservationPacket(modalities={"text": torch.randint(0, 256, (2, 6)), "vector": torch.randn(2, 16)})
    action = torch.randn(2, 8)
    out = model(packet, action, obs_tp1=packet)
    assert out.predicted_next_latent.z_sem.shape == (2, 16)
    assert "capacity_weights" in out.current_latent.extras


def test_structured_multimodal_encoder() -> None:
    """Concern 2: Structured encoder with cross-modal attention fusion."""
    cfg = ModelConfig(
        encoder_name="structured_multimodal",
        encoder_kwargs={
            "text_vocab_size": 256, "text_embed_dim": 32, "text_transformer_layers": 1,
            "text_nhead": 2, "vector_input_dim": 16, "image_channels": 3,
            "hidden_dim": 32, "n_slots": 4, "slot_dim": 16, "slot_iters": 2,
            "merge_threshold": 0.8, "fusion_nhead": 2,
        },
        latent_projector_kwargs={"input_dim": 32, "latent_dim": 16, "use_batchnorm": False},
        memory_kwargs={"input_dim": 32, "hidden_dim": 16},
        action_encoder_kwargs={"action_dim": 8, "action_embed_dim": 16},
        conditioner_kwargs={"latent_dim": 64, "action_dim": 16, "memory_dim": 16, "out_dim": 64},
        transition_core_name="mlp",
        transition_core_kwargs={"input_dim": 64, "hidden_dim": 64},
        prediction_head_kwargs={"hidden_dim": 64, "latent_dim": 16},
        decoder_configs=[],
    )
    model = build_model(cfg)
    packet = ObservationPacket(modalities={
        "text": torch.randint(0, 256, (2, 6)),
        "vector": torch.randn(2, 16),
        "image": torch.randn(2, 3, 32, 32),
    })
    action = torch.randn(2, 8)
    out = model(packet, action, obs_tp1=packet)
    assert out.predicted_next_latent.z_sem.shape == (2, 16)


def test_regularizer_min_weight_with_kendall() -> None:
    """Concern 3: Regularizer weight is clamped when using Kendall uncertainty."""
    loss_fn = WorldModelLoss(learned_uncertainty=True, regularizer_min_weight=0.5)
    # Set the regularizer log_var to a large positive value (would normally suppress it)
    with torch.no_grad():
        loss_fn.log_vars["regularizer_loss"].fill_(10.0)
    losses = loss_fn(_dummy_output(), {})
    assert "regularizer_effective_weight" in losses
    assert losses["regularizer_effective_weight"].item() >= 0.5


def test_relative_curriculum_phases() -> None:
    """Concern 4: Relative curriculum boundaries."""
    phases = relative_curriculum_phases(total_steps=100_000)
    assert len(phases) == 5
    assert phases[0].until_step == 5_000   # 5%
    assert phases[1].until_step == 15_000  # 15%
    assert phases[2].until_step == 35_000  # 35%
    assert phases[3].until_step == 60_000  # 60%
    assert phases[4].until_step == 100_000


def test_adaptive_curriculum_scheduler() -> None:
    """Concern 4: Adaptive curriculum transitions on loss plateau."""
    phases = default_curriculum_phases()
    scheduler = AdaptiveCurriculumScheduler(phases, patience=5, min_improvement=0.01, window_size=3)
    assert scheduler.current_phase.phase_id == 1

    # Feed plateau losses to trigger phase transition
    for _ in range(20):
        scheduler.step(1.0)
    assert scheduler.current_phase.phase_id >= 2


def test_image_decoder_head() -> None:
    """Concern 5: Image decoder reconstructs from latent."""
    cfg = ModelConfig(
        encoder_name="simple_multimodal",
        encoder_kwargs={"text_vocab_size": 256, "text_embed_dim": 32, "vector_input_dim": 16, "image_channels": 3, "hidden_dim": 32},
        latent_projector_kwargs={"input_dim": 32, "latent_dim": 16, "use_batchnorm": False},
        memory_kwargs={"input_dim": 32, "hidden_dim": 16},
        action_encoder_kwargs={"action_dim": 8, "action_embed_dim": 16},
        conditioner_kwargs={"latent_dim": 64, "action_dim": 16, "memory_dim": 16, "out_dim": 64},
        transition_core_name="mlp",
        transition_core_kwargs={"input_dim": 64, "hidden_dim": 64},
        prediction_head_kwargs={"hidden_dim": 64, "latent_dim": 16},
        decoder_configs=[
            ("image_reconstruction", {"latent_dim": 64, "base_channels": 32, "output_channels": 3, "output_size": 32}),
        ],
    )
    model = build_model(cfg)
    packet = ObservationPacket(modalities={"vector": torch.randn(2, 16)})
    action = torch.randn(2, 8)
    out = model(packet, action, obs_tp1=packet)
    img_key = next(k for k in out.decoder_outputs if k.endswith("image_recon"))
    assert out.decoder_outputs[img_key].shape == (2, 3, 32, 32)


def test_audio_decoder_head() -> None:
    """Concern 6: Audio decoder reconstructs from latent."""
    cfg = ModelConfig(
        encoder_name="simple_multimodal",
        encoder_kwargs={"text_vocab_size": 256, "text_embed_dim": 32, "vector_input_dim": 16, "image_channels": 3, "hidden_dim": 32},
        latent_projector_kwargs={"input_dim": 32, "latent_dim": 16, "use_batchnorm": False},
        memory_kwargs={"input_dim": 32, "hidden_dim": 16},
        action_encoder_kwargs={"action_dim": 8, "action_embed_dim": 16},
        conditioner_kwargs={"latent_dim": 64, "action_dim": 16, "memory_dim": 16, "out_dim": 64},
        transition_core_name="mlp",
        transition_core_kwargs={"input_dim": 64, "hidden_dim": 64},
        prediction_head_kwargs={"hidden_dim": 64, "latent_dim": 16},
        decoder_configs=[
            ("audio_reconstruction", {"latent_dim": 64, "base_channels": 32, "output_channels": 1, "output_length": 256}),
        ],
    )
    model = build_model(cfg)
    packet = ObservationPacket(modalities={"vector": torch.randn(2, 16)})
    action = torch.randn(2, 8)
    out = model(packet, action, obs_tp1=packet)
    audio_key = next(k for k in out.decoder_outputs if k.endswith("audio_recon"))
    assert out.decoder_outputs[audio_key].shape == (2, 1, 256)


def test_latent_prediction_metrics() -> None:
    """Concern 7: Latent prediction quality metrics."""
    pred = LatentState(z_sem=torch.randn(4, 16), z_dyn=torch.randn(4, 16))
    target = LatentState(z_sem=pred.z_sem + 0.1 * torch.randn(4, 16), z_dyn=torch.randn(4, 16))
    metrics = LatentPredictionMetrics.compute(pred, target)
    assert "latent_sem_mse" in metrics
    assert "latent_sem_cosine" in metrics
    assert "latent_sem_r2" in metrics
    assert "latent_dyn_mse" in metrics
    # Sem should have high cosine similarity (pred ~ target + small noise)
    assert metrics["latent_sem_cosine"] > 0.8


def test_reconstruction_metrics_image() -> None:
    """Concern 7: PSNR and SSIM for images."""
    pred = torch.rand(2, 3, 16, 16)
    target = pred + 0.01 * torch.randn_like(pred)
    target = target.clamp(0, 1)
    psnr = ReconstructionMetrics.psnr(pred, target)
    assert psnr > 20.0  # Should be high for nearly identical images
    ssim = ReconstructionMetrics.ssim(pred, target)
    assert ssim > 0.5


def test_reconstruction_metrics_text_perplexity() -> None:
    """Concern 7: Text perplexity metric."""
    logits = torch.randn(2, 10, 100)  # [B, T, vocab]
    targets = torch.randint(0, 100, (2, 10))
    ppl = ReconstructionMetrics.text_perplexity(logits, targets)
    assert ppl > 1.0  # Perplexity is always >= 1


def test_evaluation_suite_step() -> None:
    """Concern 7: EvaluationSuite produces metrics from model output."""
    output = _dummy_output()
    suite = EvaluationSuite()
    result = suite.evaluate_step(output, {})
    assert len(result.latent_metrics) > 0
    assert "latent_sem_mse" in result.latent_metrics


def test_image_recon_loss_in_world_model_loss() -> None:
    """Image and audio reconstruction losses are included in total."""
    loss_fn = WorldModelLoss(learned_uncertainty=False)
    output = _dummy_output()
    output.decoder_outputs["image_reconstruction.image_recon"] = torch.rand(2, 3, 8, 8)
    batch = {"image_target": torch.rand(2, 3, 8, 8)}
    losses = loss_fn(output, batch)
    assert "image_recon_loss" in losses
    assert losses["image_recon_loss"].item() > 0


def test_curriculum_includes_image_audio_multipliers() -> None:
    """Curriculum phases include image/audio multipliers."""
    phases = default_curriculum_phases()
    phase5 = phases[-1]
    assert "image_recon_loss" in phase5.task_multipliers
    assert "audio_recon_loss" in phase5.task_multipliers
    assert phase5.task_multipliers["image_recon_loss"] == 1.0
