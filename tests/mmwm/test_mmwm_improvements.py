from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

from MMWM.config import ModelConfig, build_model
from MMWM.containers import LatentState, MemoryState, ModelOutput, ObservationPacket
from MMWM.curriculum import (
    AdaptiveCurriculumScheduler,
    default_curriculum_phases,
    relative_curriculum_phases,
)
from MMWM.data import (
    DeterministicTransitionDataset,
    EpisodeDataset,
    GridWorldTransitionDataset,
    TransitionTupleDataset,
    collate_transition_batch,
    flatten_transition_value,
)
from MMWM.demo import run_gridworld_smoke
from MMWM.evaluation import (
    EvaluationSuite,
    LatentPredictionMetrics,
    ReconstructionMetrics,
)
from MMWM.encoders import PretrainedMultimodalEncoder, PretrainedTextEncoder, PretrainedVisionEncoder
from MMWM.losses import ContrastiveAlignmentLoss, WorldModelLoss
from MMWM.trainer import Trainer, build_lr_scheduler


def _dummy_output(batch_size: int = 2, latent_dim: int = 8) -> ModelOutput:
    pred = LatentState(
        z_sem=torch.randn(batch_size, latent_dim),
        z_dyn=torch.randn(batch_size, latent_dim),
        z_ctrl=torch.randn(batch_size, latent_dim),
        z_ctx=torch.randn(batch_size, latent_dim),
        extras={},
    )
    tgt = LatentState(
        z_sem=torch.randn(batch_size, latent_dim),
        z_dyn=torch.randn(batch_size, latent_dim),
        z_ctrl=torch.randn(batch_size, latent_dim),
        z_ctx=torch.randn(batch_size, latent_dim),
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


def _small_model_cfg() -> ModelConfig:
    """Minimal config for fast test execution."""
    return ModelConfig(
        encoder_name="simple_multimodal",
        encoder_kwargs={"text_vocab_size": 256, "text_embed_dim": 32, "vector_input_dim": 16, "image_channels": 3, "hidden_dim": 32},
        latent_projector_kwargs={"input_dim": 32, "latent_dim": 16, "use_norm": False},
        memory_kwargs={"input_dim": 32, "hidden_dim": 16},
        action_encoder_kwargs={"action_dim": 8, "action_embed_dim": 16},
        conditioner_kwargs={"latent_dim": 64, "action_dim": 16, "memory_dim": 16, "out_dim": 64},
        transition_core_name="mlp",
        transition_core_kwargs={"input_dim": 64, "hidden_dim": 64},
        prediction_head_kwargs={"hidden_dim": 64, "latent_dim": 16},
        decoder_configs=[],
    )


def _small_batch(batch_size: int = 2) -> dict:
    return {
        "text_t": torch.randint(0, 256, (batch_size, 6)),
        "text_tp1": torch.randint(0, 256, (batch_size, 6)),
        "vector_t": torch.randn(batch_size, 16),
        "vector_tp1": torch.randn(batch_size, 16),
        "action": torch.randn(batch_size, 8),
    }


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
        latent_projector_kwargs={"input_dim": 32, "latent_dim": 16, "use_norm": False},
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
# Tests for concern fixes
# ============================================================


def test_adaptive_role_split_projector() -> None:
    """Concern 1: Adaptive role split with learnable capacity gating."""
    cfg = ModelConfig(
        encoder_name="simple_multimodal",
        encoder_kwargs={"text_vocab_size": 256, "text_embed_dim": 32, "vector_input_dim": 16, "image_channels": 3, "hidden_dim": 32},
        latent_projector_name="adaptive_role_split_mlp",
        latent_projector_kwargs={"input_dim": 32, "latent_dim": 16, "intermediate_dim": 64, "use_norm": False},
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
        latent_projector_kwargs={"input_dim": 32, "latent_dim": 16, "use_norm": False},
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
    cfg = _small_model_cfg()
    cfg.decoder_configs = [
        ("image_reconstruction", {"latent_dim": 64, "base_channels": 32, "output_channels": 3, "output_size": 32}),
    ]
    model = build_model(cfg)
    packet = ObservationPacket(modalities={"vector": torch.randn(2, 16)})
    action = torch.randn(2, 8)
    out = model(packet, action, obs_tp1=packet)
    img_key = next(k for k in out.decoder_outputs if k.endswith("image_recon"))
    assert out.decoder_outputs[img_key].shape == (2, 3, 32, 32)


def test_audio_decoder_head() -> None:
    """Concern 6: Audio decoder reconstructs from latent."""
    cfg = _small_model_cfg()
    cfg.decoder_configs = [
        ("audio_reconstruction", {"latent_dim": 64, "base_channels": 32, "output_channels": 1, "output_length": 256}),
    ]
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


def test_curriculum_includes_image_audio_contrastive_multipliers() -> None:
    """Curriculum phases include image/audio/contrastive multipliers."""
    phases = default_curriculum_phases()
    phase5 = phases[-1]
    assert "image_recon_loss" in phase5.task_multipliers
    assert "audio_recon_loss" in phase5.task_multipliers
    assert "contrastive_alignment_loss" in phase5.task_multipliers
    assert phase5.task_multipliers["image_recon_loss"] == 1.0
    assert phase5.task_multipliers["contrastive_alignment_loss"] == 1.0
    # Contrastive loss is 0 in early phases
    assert phases[0].task_multipliers["contrastive_alignment_loss"] == 0.0
    assert phases[1].task_multipliers["contrastive_alignment_loss"] == 0.0
    # Introduced in phase 3
    assert phases[2].task_multipliers["contrastive_alignment_loss"] == 0.25


# ============================================================
# New tests for review blocker fixes
# ============================================================


def test_memory_propagation_in_train_step() -> None:
    """Blocker 2: Memory state is propagated across train_step calls."""
    model = build_model(_small_model_cfg())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = WorldModelLoss()
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=torch.device("cpu"), mixed_precision=False)

    batch = _small_batch()
    metrics1, memory1 = trainer.train_step(batch)
    assert memory1 is not None
    # Second step uses the returned memory
    metrics2, memory2 = trainer.train_step(batch, memory_state=memory1.detach())
    assert memory2 is not None
    assert "total_loss" in metrics2


def test_checkpoint_save_load() -> None:
    """Blocker 3: Checkpoint save and load restores full state."""
    cfg = _small_model_cfg()
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = WorldModelLoss(learned_uncertainty=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=torch.device("cpu"), run_dir=tmpdir, mixed_precision=False)
        batch = _small_batch()
        trainer.train_step(batch)
        trainer.train_step(batch)
        assert trainer.global_step == 2

        path = trainer.save_checkpoint()
        assert os.path.exists(path)

        # Create a fresh trainer and load
        model2 = build_model(cfg)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        loss_fn2 = WorldModelLoss(learned_uncertainty=True)
        trainer2 = Trainer(model=model2, optimizer=optimizer2, loss_fn=loss_fn2, device=torch.device("cpu"), run_dir=tmpdir, mixed_precision=False)
        trainer2.load_checkpoint(path)
        assert trainer2.global_step == 2


def test_lr_scheduler_integration() -> None:
    """Blocker 6: LR scheduler steps with training."""
    model = build_model(_small_model_cfg())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_lr_scheduler(optimizer, total_steps=100, warmup_fraction=0.1)
    loss_fn = WorldModelLoss()
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=torch.device("cpu"), mixed_precision=False, lr_scheduler=scheduler)

    initial_lr = optimizer.param_groups[0]["lr"]
    batch = _small_batch()
    # During warmup, LR should increase
    for _ in range(5):
        trainer.train_step(batch)
    warmup_lr = optimizer.param_groups[0]["lr"]
    # After a few warmup steps, LR should be higher than initial (which starts at 0)
    assert warmup_lr > 0


def test_layernorm_replaces_batchnorm() -> None:
    """Blocker 8: LayerNorm works with batch_size=1."""
    cfg = _small_model_cfg()
    cfg.latent_projector_kwargs = {"input_dim": 32, "latent_dim": 16, "use_norm": True}
    model = build_model(cfg)
    # batch_size=1 should work (would fail with BatchNorm)
    packet = ObservationPacket(modalities={"vector": torch.randn(1, 16)})
    action = torch.randn(1, 8)
    out = model(packet, action, obs_tp1=packet)
    assert out.predicted_next_latent.z_sem.shape == (1, 16)


def test_contrastive_alignment_loss() -> None:
    """Blocker 9: InfoNCE contrastive loss for cross-modal alignment."""
    loss_fn = ContrastiveAlignmentLoss(temperature=0.07)
    feat_a = torch.randn(8, 32)
    feat_b = torch.randn(8, 32)
    loss = loss_fn(feat_a, feat_b)
    assert loss.shape == ()
    assert loss.item() > 0
    # Same features should have lower loss
    loss_same = loss_fn(feat_a, feat_a)
    assert loss_same.item() < loss.item()


def test_contrastive_loss_in_world_model_loss() -> None:
    """Contrastive loss is computed when multiple modality features are present."""
    loss_fn = WorldModelLoss(learned_uncertainty=False)
    output = _dummy_output(batch_size=4)
    # Add per-modality features to extras
    output.current_latent.extras["text_feat"] = torch.randn(4, 8)
    output.current_latent.extras["vector_feat"] = torch.randn(4, 8)
    losses = loss_fn(output, {})
    assert "contrastive_alignment_loss" in losses
    assert losses["contrastive_alignment_loss"].item() > 0


def test_memory_state_detach() -> None:
    """MemoryState.detach() cuts the computation graph."""
    ctx = torch.randn(2, 8, requires_grad=True)
    hidden = torch.randn(2, 8, requires_grad=True)
    ms = MemoryState(context=ctx, hidden=hidden, extras={"extra_t": torch.randn(2, 4, requires_grad=True)})
    detached = ms.detach()
    assert not detached.context.requires_grad
    assert not detached.hidden.requires_grad
    assert not detached.extras["extra_t"].requires_grad


def test_sequence_level_training() -> None:
    """Blocker 4: Sequence-level training (BPTT) runs end-to-end."""
    model = build_model(_small_model_cfg())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = WorldModelLoss()
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=torch.device("cpu"), mixed_precision=False)

    T = 4  # 4 transitions → 5 timesteps
    B = 2
    batch_seq = {
        "vector_seq": torch.randn(B, T + 1, 16),
        "action_seq": torch.randn(B, T, 8),
    }
    metrics = trainer.train_sequence_step(batch_seq, window_length=2)
    assert "total_loss" in metrics
    assert metrics["sequence_length"] == T


def test_episode_dataset_windows_sequences_for_bptt() -> None:
    observations = torch.arange(6 * 16, dtype=torch.float32).reshape(6, 16)
    actions = torch.arange(5 * 8, dtype=torch.float32).reshape(5, 8)
    episode = {
        "observations": observations,
        "actions": actions,
        "terminals": torch.tensor([False, False, True, False, False]),
    }
    dataset = EpisodeDataset([episode], window_length=3, stride=2)

    assert len(dataset) == 2
    item = dataset[0]
    assert item["vector_seq"].shape == (4, 16)
    assert item["action_seq"].shape == (3, 8)
    assert item["vector_target_seq"].shape == (4, 16)
    assert item["done_seq"].tolist() == [False, False, False, True]


def test_episode_dataset_feeds_train_sequence_step() -> None:
    torch.manual_seed(0)
    episodes = [
        {
            "observations": torch.randn(6, 16),
            "actions": torch.randn(5, 8),
            "timeouts": torch.tensor([False, False, False, False, True]),
        }
        for _ in range(2)
    ]
    dataset = EpisodeDataset(episodes, window_length=3, stride=1)
    batch = collate_transition_batch([dataset[0], dataset[1]])

    model = build_model(_small_model_cfg())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=WorldModelLoss(), device=torch.device("cpu"), mixed_precision=False)
    metrics = trainer.train_sequence_step(batch, window_length=2)

    assert "total_loss" in metrics
    assert metrics["sequence_length"] == 3.0


# ============================================================
# Tests for review fixes (claude/review-mmwm-folder-L2YyA)
# ============================================================


def test_mse_raises_on_pred_target_mismatch() -> None:
    """Review #1: silently returning zero when only one of (pred, target) is None
    hides a class of bugs where a head drops a role the target still has."""
    import pytest

    pred = LatentState(
        z_sem=torch.randn(2, 8),
        z_dyn=None,  # head dropped this role
        z_ctrl=torch.randn(2, 8),
        z_ctx=torch.randn(2, 8),
        extras={},
    )
    tgt = LatentState(
        z_sem=torch.randn(2, 8),
        z_dyn=torch.randn(2, 8),  # but the target still has it
        z_ctrl=torch.randn(2, 8),
        z_ctx=torch.randn(2, 8),
        extras={},
    )
    out = ModelOutput(
        current_latent=pred,
        predicted_next_latent=pred,
        target_next_latent=tgt,
        next_memory=None,  # type: ignore[arg-type]
        decoder_outputs={},
        aux={"regularizer_total": torch.tensor(0.0)},
    )
    loss_fn = WorldModelLoss()
    with pytest.raises(ValueError, match="pred"):
        loss_fn(out, {})


def test_mse_zero_when_both_roles_absent() -> None:
    """Review #1: when both prediction and target genuinely lack a role,
    returning zero is the correct behavior (role is just disabled)."""
    pred = LatentState(z_sem=torch.randn(2, 8), z_dyn=None, z_ctrl=None, z_ctx=None, extras={})
    tgt = LatentState(z_sem=torch.randn(2, 8), z_dyn=None, z_ctrl=None, z_ctx=None, extras={})
    out = ModelOutput(
        current_latent=pred,
        predicted_next_latent=pred,
        target_next_latent=tgt,
        next_memory=None,  # type: ignore[arg-type]
        decoder_outputs={},
        aux={"regularizer_total": torch.tensor(0.0)},
    )
    losses = WorldModelLoss()(out, {})
    assert losses["latent_dyn_loss"].item() == 0.0
    assert losses["latent_ctrl_loss"].item() == 0.0
    assert losses["latent_ctx_loss"].item() == 0.0


def test_uncertainty_skips_inactive_losses() -> None:
    """Review #2: with no decoder targets in the batch, modality losses are
    inactive and must not contribute their +0.5*log_var penalty."""
    loss_fn = WorldModelLoss(learned_uncertainty=True)
    losses = loss_fn(_dummy_output(), batch={})
    # No text/vector/image/audio target in batch and no decoder outputs ->
    # those log_var keys must be absent from the per-loss diagnostic dict.
    for inactive in ("text_ce_loss", "vector_recon_loss", "image_recon_loss",
                     "audio_recon_loss", "contrastive_alignment_loss"):
        assert f"{inactive}_log_var" not in losses, (
            f"{inactive} has no signal but its log_var was still recorded "
            "(meaning it contributed to the total)"
        )
    # Active losses must still record their log_var.
    for active in ("latent_sem_loss", "latent_dyn_loss",
                   "latent_ctrl_loss", "latent_ctx_loss", "regularizer_loss"):
        assert f"{active}_log_var" in losses


def test_effective_weight_upper_clamp() -> None:
    """Review #18: effective_weight must be clamped from above so a single
    log_var << 0 cannot dominate the total loss."""
    loss_fn = WorldModelLoss(
        learned_uncertainty=True,
        regularizer_max_weight=2.5,
    )
    # Force the regularizer log_var very negative -> exp(-log_var) huge.
    with torch.no_grad():
        loss_fn.log_vars["regularizer_loss"].fill_(-10.0)
    losses = loss_fn(_dummy_output(), batch={})
    assert losses["regularizer_effective_weight"].item() <= 2.5 + 1e-6


def test_text_ce_respects_pad_token_id() -> None:
    """Review #4: pad tokens should be ignored when computing text CE."""
    loss_fn = WorldModelLoss(text_pad_token_id=0)
    pred = LatentState(z_sem=torch.randn(2, 8), z_dyn=None, z_ctrl=None, z_ctx=None, extras={})
    tgt = LatentState(z_sem=torch.randn(2, 8), z_dyn=None, z_ctrl=None, z_ctx=None, extras={})
    # All targets are pad (=0). With ignore_index=0 the loss should be NaN-free
    # zero-elements; PyTorch returns NaN for "all ignored" but with at least
    # one non-pad token it must be finite. Mix in one real token:
    text_target = torch.zeros(2, 4, dtype=torch.long)
    text_target[0, 0] = 5
    text_logits = torch.randn(2, 4, 32)
    out = ModelOutput(
        current_latent=pred,
        predicted_next_latent=pred,
        target_next_latent=tgt,
        next_memory=None,  # type: ignore[arg-type]
        decoder_outputs={"text_logits": text_logits},
        aux={"regularizer_total": torch.tensor(0.0)},
    )
    losses = loss_fn(out, {"text_target": text_target})
    assert torch.isfinite(losses["text_ce_loss"])
    # And without ignore_index, every pad position would also contribute,
    # generally pushing the loss higher.
    loss_fn_no_pad = WorldModelLoss()
    losses_no_pad = loss_fn_no_pad(out, {"text_target": text_target})
    assert losses["text_ce_loss"].item() != losses_no_pad["text_ce_loss"].item()


def test_text_ce_seq_len_mismatch_raises() -> None:
    """Review #4: silently reshaping mismatched logits/targets is dangerous."""
    import pytest

    loss_fn = WorldModelLoss()
    pred = LatentState(z_sem=torch.randn(2, 8), z_dyn=None, z_ctrl=None, z_ctx=None, extras={})
    tgt = LatentState(z_sem=torch.randn(2, 8), z_dyn=None, z_ctrl=None, z_ctx=None, extras={})
    out = ModelOutput(
        current_latent=pred,
        predicted_next_latent=pred,
        target_next_latent=tgt,
        next_memory=None,  # type: ignore[arg-type]
        decoder_outputs={"text_logits": torch.randn(2, 5, 32)},
        aux={"regularizer_total": torch.tensor(0.0)},
    )
    with pytest.raises(ValueError, match="seq lengths disagree"):
        loss_fn(out, {"text_target": torch.zeros(2, 4, dtype=torch.long)})


def test_text_decoder_pos_embed_overflow_raises() -> None:
    """Review #5: TextAutoregressiveHead must reject sequences longer than its
    learned position embedding instead of silently indexing out of range."""
    import pytest

    from MMWM.decoders import TextAutoregressiveHead

    head = TextAutoregressiveHead(
        vocab_size=64, latent_dim=16, text_embed_dim=32, hidden_dim=32,
        num_layers=1, nhead=2, max_seq_len=8,
    )
    latent = LatentState(z_sem=torch.randn(1, 16), z_dyn=None, z_ctrl=None, z_ctx=None, extras={})
    # 1 latent token + 8 prefix tokens = 9 > max_seq_len.
    with pytest.raises(ValueError, match="max_seq_len"):
        head(latent, context={"prefix_tokens": torch.zeros(1, 8, dtype=torch.long)})


def test_simple_multimodal_encoder_rejects_empty_modalities() -> None:
    """Review #14: an empty ObservationPacket should fail loud, not silently
    propagate a zero feature."""
    import pytest

    from MMWM.encoders import SimpleMultimodalEncoder

    enc = SimpleMultimodalEncoder(
        text_vocab_size=64, text_embed_dim=16, vector_input_dim=8,
        image_channels=3, hidden_dim=16,
    )
    with pytest.raises(ValueError, match="no modalities"):
        enc(ObservationPacket(modalities={}))


def test_simple_multimodal_encoder_rejects_unknown_modalities() -> None:
    """Review #14: a packet whose modalities are all unknown to the encoder
    should also fail rather than producing a zero embedding."""
    import pytest

    from MMWM.encoders import SimpleMultimodalEncoder

    enc = SimpleMultimodalEncoder(
        text_vocab_size=64, text_embed_dim=16, vector_input_dim=8,
        image_channels=3, hidden_dim=16,
    )
    # "proprio_extra" is not a registered sub-encoder of SimpleMultimodalEncoder.
    with pytest.raises(ValueError, match="match its registered sub-encoders"):
        enc(ObservationPacket(modalities={"proprio_extra": torch.randn(2, 16)}))


def test_build_model_catches_dim_mismatch() -> None:
    """Review #10: dimension mismatches across components should fail at
    build time, not deep inside a forward pass."""
    import pytest

    cfg = _small_model_cfg()
    # Conditioner.latent_dim must equal 4 * latent_projector.latent_dim.
    cfg.conditioner_kwargs["latent_dim"] = 99
    with pytest.raises(ValueError, match="conditioner.latent_dim"):
        build_model(cfg)

    cfg2 = _small_model_cfg()
    cfg2.prediction_head_kwargs["hidden_dim"] = 99
    with pytest.raises(ValueError, match="prediction_head.hidden_dim"):
        build_model(cfg2)


def test_build_model_skip_validation_works() -> None:
    """Review #10: users with custom components should be able to opt out."""
    cfg = _small_model_cfg()
    cfg.conditioner_kwargs["latent_dim"] = 99  # would normally fail
    # Building with skip_validation=True must not raise at validate time
    # (it may still fail at forward if dims are truly wrong, but that's the
    # user's problem once they opt out).
    import pytest

    with pytest.raises(ValueError):
        build_model(cfg)
    # Explicit opt-out proceeds past validation:
    try:
        build_model(cfg, skip_validation=True)
    except ValueError as e:
        if "ModelConfig dimension mismatch" in str(e):
            pytest.fail("skip_validation=True should bypass dim validation")


def test_transition_aux_reserved_key_raises() -> None:
    """Review #8: only the documented '_transition_hidden' aux key is allowed;
    typos that start with '_' must surface instead of silently disappearing."""
    import pytest
    import torch.nn as nn

    from MMWM.containers import ObservationPacket
    from MMWM.interfaces import ITransitionCore, TRANSITION_CORES

    class BadCore(ITransitionCore):
        def __init__(self, input_dim: int = 64, hidden_dim: int = 64) -> None:
            super().__init__()
            self.net = nn.Linear(input_dim, hidden_dim)

        def forward(self, conditioned_input, memory_state):
            return self.net(conditioned_input), {"_typo_key": torch.tensor(0.0)}

    # Register under a unique name so we don't collide with other tests.
    name = "bad_core_for_aux_test"
    if name not in TRANSITION_CORES.names():
        TRANSITION_CORES.register(name)(BadCore)

    cfg = _small_model_cfg()
    cfg.transition_core_name = name
    model = build_model(cfg)
    packet = ObservationPacket(modalities={"vector": torch.randn(2, 16)})
    action = torch.randn(2, 8)
    with pytest.raises(ValueError, match="reserved aux keys"):
        model(packet, action, obs_tp1=packet)


def test_memory_read_default_implementation() -> None:
    """Review #9: IMemory.read has a default implementation that validates
    context is not None and returns it."""
    import pytest

    from MMWM.interfaces import MEMORIES

    # All registered memories should inherit the default or override consistently.
    for mem_name in MEMORIES.names():
        mem = MEMORIES.build(mem_name, **({"input_dim": 32, "hidden_dim": 16}
                                          if mem_name != "identity"
                                          else {"latent_dim": 16}))
        state = mem.init_state(batch_size=2, device=torch.device("cpu"))
        # Default read() should return a non-None tensor equal to state.context.
        out = mem.read(state)
        assert out is state.context

    # If context is None, read() must raise clearly.
    from MMWM.containers import MemoryState

    mem = MEMORIES.build("identity", latent_dim=16)
    with pytest.raises(RuntimeError, match="state.context is None"):
        mem.read(MemoryState(context=None, hidden=None))


def test_checkpoint_save_load_roundtrip() -> None:
    """Review #19: save/load should round-trip model + optimizer + loss_fn
    state exactly, including global_step and learned log_vars."""
    import tempfile

    cfg = _small_model_cfg()
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = WorldModelLoss(learned_uncertainty=True)
    device = torch.device("cpu")

    trainer_a = Trainer(
        model=model, optimizer=optimizer, loss_fn=loss_fn, device=device,
        run_dir=tempfile.mkdtemp(prefix="mmwm_ckpt_"), mixed_precision=False,
    )
    # Mutate some state so the round-trip is meaningful.
    trainer_a.global_step = 123
    trainer_a.best_eval_loss = 0.5
    with torch.no_grad():
        trainer_a.loss_fn.log_vars["latent_sem_loss"].fill_(-1.5)
    ckpt_path = trainer_a.save_checkpoint()

    # Build a fresh trainer B and load.
    model_b = build_model(cfg)
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    loss_fn_b = WorldModelLoss(learned_uncertainty=True)
    trainer_b = Trainer(
        model=model_b, optimizer=optimizer_b, loss_fn=loss_fn_b, device=device,
        run_dir=tempfile.mkdtemp(prefix="mmwm_ckpt_b_"), mixed_precision=False,
    )
    trainer_b.load_checkpoint(ckpt_path)

    assert trainer_b.global_step == 123
    assert trainer_b.best_eval_loss == 0.5
    # log_vars must round-trip.
    assert torch.allclose(
        trainer_b.loss_fn.log_vars["latent_sem_loss"],
        trainer_a.loss_fn.log_vars["latent_sem_loss"],
    )
    # Model parameters must match bit-for-bit.
    for p_a, p_b in zip(trainer_a.model.parameters(), trainer_b.model.parameters()):
        assert torch.equal(p_a, p_b)


def test_gradient_checkpointing_flag() -> None:
    """Blocker: Gradient checkpointing option on RecurrentAttnRes."""
    cfg = ModelConfig(
        encoder_name="simple_multimodal",
        encoder_kwargs={"text_vocab_size": 256, "text_embed_dim": 32, "vector_input_dim": 16, "image_channels": 3, "hidden_dim": 32},
        latent_projector_kwargs={"input_dim": 32, "latent_dim": 16, "use_norm": False},
        memory_kwargs={"input_dim": 32, "hidden_dim": 16},
        action_encoder_kwargs={"action_dim": 8, "action_embed_dim": 16},
        conditioner_kwargs={"latent_dim": 64, "action_dim": 16, "memory_dim": 16, "out_dim": 64},
        transition_core_name="recurrent_attnres_transformer",
        transition_core_kwargs={"input_dim": 64, "hidden_dim": 64, "recurrent_steps": 2, "gradient_checkpointing": True},
        prediction_head_kwargs={"hidden_dim": 64, "latent_dim": 16},
        decoder_configs=[],
    )
    model = build_model(cfg)
    packet = ObservationPacket(modalities={"vector": torch.randn(2, 16)})
    action = torch.randn(2, 8)
    out = model(packet, action, obs_tp1=packet)
    assert out.predicted_next_latent.z_sem.shape == (2, 16)
    # Verify gradient flows
    loss = out.predicted_next_latent.z_sem.sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_latent_state_exposes_z_ctx_not_z_mem() -> None:
    latent = LatentState(
        z_sem=torch.randn(2, 8),
        z_dyn=torch.randn(2, 8),
        z_ctrl=torch.randn(2, 8),
        z_ctx=torch.randn(2, 8),
        extras={},
    )
    assert latent.z_ctx is not None
    assert not hasattr(latent, "z_mem")


def test_latent_state_checkpoint_roundtrip_preserves_z_ctx() -> None:
    latent = LatentState(
        z_sem=torch.randn(2, 8),
        z_dyn=torch.randn(2, 8),
        z_ctrl=torch.randn(2, 8),
        z_ctx=torch.randn(2, 8),
        extras={"tag": torch.ones(2, 8)},
    )
    restored = LatentState.from_dict(latent.to_dict())
    assert torch.equal(restored.z_sem, latent.z_sem)
    assert restored.z_ctx is not None
    assert latent.z_ctx is not None
    assert torch.equal(restored.z_ctx, latent.z_ctx)


def test_latent_state_backward_compat_accepts_z_mem_checkpoint() -> None:
    legacy_z_mem = torch.randn(2, 8)
    legacy_payload = {
        "z_sem": torch.randn(2, 8),
        "z_dyn": torch.randn(2, 8),
        "z_ctrl": torch.randn(2, 8),
        "z_mem": legacy_z_mem,
        "extras": {},
    }
    restored = LatentState.from_dict(legacy_payload)
    assert restored.z_ctx is not None
    assert torch.equal(restored.z_ctx, legacy_z_mem)


def test_loss_dict_uses_latent_ctx_key() -> None:
    losses = WorldModelLoss()(_dummy_output(), {})
    assert "latent_ctx_loss" in losses
    assert "latent_mem_loss" not in losses


def test_audio_input_modality_runs_forward() -> None:
    """Audio is a first-class input modality, not just a decoder target."""
    cfg = _small_model_cfg()
    cfg.encoder_kwargs["audio_channels"] = 2
    model = build_model(cfg)
    packet = ObservationPacket(modalities={"audio": torch.randn(2, 2, 64)})
    action = torch.randn(2, 8)
    out = model(packet, action, obs_tp1=packet)
    assert out.predicted_next_latent.z_sem.shape == (2, 16)
    assert "audio_feat" in out.current_latent.extras


def test_structured_encoder_accepts_audio() -> None:
    cfg = ModelConfig(
        encoder_name="structured_multimodal",
        encoder_kwargs={
            "text_vocab_size": 256, "text_embed_dim": 32, "text_transformer_layers": 1,
            "text_nhead": 2, "vector_input_dim": 16, "image_channels": 3,
            "audio_channels": 2, "hidden_dim": 32, "n_slots": 4, "slot_dim": 16,
            "slot_iters": 2, "merge_threshold": 0.8, "fusion_nhead": 2,
        },
        latent_projector_kwargs={"input_dim": 32, "latent_dim": 16, "use_norm": False},
        memory_kwargs={"input_dim": 32, "hidden_dim": 16},
        action_encoder_kwargs={"action_dim": 8, "action_embed_dim": 16},
        conditioner_kwargs={"latent_dim": 64, "action_dim": 16, "memory_dim": 16, "out_dim": 64},
        transition_core_name="mlp",
        transition_core_kwargs={"input_dim": 64, "hidden_dim": 64},
        prediction_head_kwargs={"hidden_dim": 64, "latent_dim": 16},
        decoder_configs=[],
    )
    model = build_model(cfg)
    packet = ObservationPacket(modalities={"audio": torch.randn(2, 2, 64), "vector": torch.randn(2, 16)})
    out = model(packet, torch.randn(2, 8), obs_tp1=packet)
    assert "audio_feat" in out.current_latent.extras


def test_recurrent_transition_supports_input_hidden_dim_mismatch() -> None:
    cfg = _small_model_cfg()
    cfg.conditioner_kwargs["out_dim"] = 32
    cfg.transition_core_name = "recurrent_attnres_transformer"
    cfg.transition_core_kwargs = {"input_dim": 32, "hidden_dim": 64, "recurrent_steps": 2}
    cfg.prediction_head_kwargs["hidden_dim"] = 64
    model = build_model(cfg)
    packet = ObservationPacket(modalities={"vector": torch.randn(2, 16)})
    out = model(packet, torch.randn(2, 8), obs_tp1=packet)
    assert out.predicted_next_latent.z_sem.shape == (2, 16)


def test_mod_recurrent_transition_supports_input_hidden_dim_mismatch() -> None:
    cfg = _small_model_cfg()
    cfg.conditioner_kwargs["out_dim"] = 32
    cfg.transition_core_name = "mod_recurrent_attnres_transformer"
    cfg.transition_core_kwargs = {"input_dim": 32, "hidden_dim": 64, "recurrent_steps": 2}
    cfg.prediction_head_kwargs["hidden_dim"] = 64
    model = build_model(cfg)
    packet = ObservationPacket(modalities={"vector": torch.randn(2, 16)})
    out = model(packet, torch.randn(2, 8), obs_tp1=packet)
    assert out.predicted_next_latent.z_sem.shape == (2, 16)


def test_reconstruction_loss_shape_mismatch_raises() -> None:
    import pytest

    loss_fn = WorldModelLoss()
    output = _dummy_output()
    output.decoder_outputs["vector_reconstruction.vector_recon"] = torch.randn(2, 8)
    with pytest.raises(ValueError, match="vector_recon_loss"):
        loss_fn(output, {"vector_target": torch.randn(2, 9)})


def test_text_perplexity_can_ignore_padding() -> None:
    logits = torch.randn(2, 4, 16)
    targets = torch.zeros(2, 4, dtype=torch.long)
    assert ReconstructionMetrics.text_perplexity(logits, targets, pad_token_id=0) == 1.0


def test_deterministic_transition_dataset_trainer_batch() -> None:
    dataset = DeterministicTransitionDataset(length=4, vector_dim=16, action_dim=8, include_audio=True, audio_channels=2)
    batch = collate_transition_batch([dataset[0], dataset[1]])
    assert batch["vector_t"].shape == (2, 16)
    assert batch["audio_t"].shape == (2, 2, 128)

    cfg = _small_model_cfg()
    cfg.encoder_kwargs["audio_channels"] = 2
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, WorldModelLoss(), torch.device("cpu"), mixed_precision=False)
    metrics, memory = trainer.train_step(batch)
    assert "total_loss" in metrics
    assert memory is not None


def test_transition_tuple_dataset_from_d4rl_mapping() -> None:
    mapping = {
        "observations": np.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
            dtype=np.float32,
        ),
        "actions": np.array([[0.5], [1.5], [2.5]], dtype=np.float32),
        "next_observations": np.array(
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
            dtype=np.float32,
        ),
        "terminals": np.array([False, True, False]),
    }
    dataset = TransitionTupleDataset.from_mapping(mapping, max_transitions=2)

    assert len(dataset) == 2
    assert dataset.vector_dim == 2
    assert dataset.action_dim == 1
    assert torch.equal(dataset[0]["vector_target"], dataset[0]["vector_tp1"])
    assert dataset[1]["done"].item() is True

    batch = collate_transition_batch([dataset[0], dataset[1]])
    assert batch["vector_t"].shape == (2, 2)
    assert batch["action"].shape == (2, 1)
    assert batch["done"].dtype == torch.bool


def test_transition_tuple_dataset_trains_one_step() -> None:
    mapping = {
        "observations": torch.randn(6, 16),
        "actions": torch.randn(5, 8),
        "next_observations": torch.randn(5, 16),
        "timeouts": torch.tensor([False, False, True, False, False]),
    }
    dataset = TransitionTupleDataset.from_mapping(mapping, max_transitions=4)
    batch = collate_transition_batch([dataset[0], dataset[1]])

    model = build_model(_small_model_cfg())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, WorldModelLoss(), torch.device("cpu"), mixed_precision=False)
    metrics, memory = trainer.train_step(batch)

    assert "total_loss" in metrics
    assert memory is not None


def test_flatten_transition_value_uses_stable_dict_key_order() -> None:
    value = {
        "z": np.array([3, 4], dtype=np.int64),
        "a": torch.tensor([[1.0, 2.0]]),
    }
    flat = flatten_transition_value(value)
    assert flat.dtype == torch.float32
    assert torch.allclose(flat, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_gridworld_transition_dataset_emits_structured_modalities() -> None:
    dataset = GridWorldTransitionDataset(length=4, grid_size=5, episode_len=3)
    item = dataset[0]

    assert item["vector_t"].shape == (2,)
    assert item["action"].shape == (4,)
    assert item["text_t"].shape == (dataset.text_len,)
    assert item["image_t"].shape == (3, dataset.grid_size, dataset.grid_size)
    assert item["vector_target"].shape == item["vector_tp1"].shape

    obs_sequence, action_sequence = dataset.rollout_sequence(horizon=2, modalities=("vector", "text", "image"))
    assert len(obs_sequence) == 3
    assert len(action_sequence) == 2
    assert obs_sequence[0].modalities["vector"].shape == (1, 2)
    assert obs_sequence[0].modalities["image"].shape == (1, 3, 5, 5)


def test_gridworld_smoke_train_checkpoint_load_and_rollout() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = run_gridworld_smoke(steps=40, batch_size=8, run_dir=tmpdir, device=torch.device("cpu"))

    assert metrics["vector_mse_after"] < metrics["vector_mse_before"]
    assert metrics["loaded_global_step"] == 40.0
    assert metrics["rollout_horizon"] == 4.0
    assert metrics["rollout_mean_mse"] >= 0.0


class _FakeBackboneOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _FakeVisionBackbone(nn.Module):
    def __init__(self, output_dim: int = 6) -> None:
        super().__init__()
        self.proj = nn.Linear(3, output_dim)

    def forward(self, pixel_values: torch.Tensor) -> _FakeBackboneOutput:
        pooled = pixel_values.mean(dim=(-2, -1))
        tokens = self.proj(pooled).unsqueeze(1).repeat(1, 2, 1)
        return _FakeBackboneOutput(tokens)


class _FakeTextBackbone(nn.Module):
    def __init__(self, vocab_size: int = 32, output_dim: int = 6) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, output_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> _FakeBackboneOutput:
        return _FakeBackboneOutput(self.embedding(input_ids))


def test_pretrained_vision_encoder_wraps_frozen_backbone() -> None:
    backbone = _FakeVisionBackbone(output_dim=6)
    encoder = PretrainedVisionEncoder(hidden_dim=8, backbone=backbone, output_dim=6, freeze=True, normalize=False)
    out = encoder(torch.randn(2, 3, 8, 8))

    assert out.shape == (2, 8)
    assert all(not p.requires_grad for p in backbone.parameters())


def test_pretrained_text_encoder_pools_with_attention_mask() -> None:
    backbone = _FakeTextBackbone(output_dim=6)
    encoder = PretrainedTextEncoder(hidden_dim=8, backbone=backbone, output_dim=6, freeze=True)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    out = encoder(tokens, mask=mask)

    assert out.shape == (2, 8)
    assert all(not p.requires_grad for p in backbone.parameters())


def test_pretrained_multimodal_encoder_builds_in_model() -> None:
    cfg = _small_model_cfg()
    cfg.encoder_name = "pretrained_multimodal"
    cfg.encoder_kwargs = {
        "text_vocab_size": 32,
        "text_embed_dim": 16,
        "vector_input_dim": 16,
        "image_channels": 3,
        "hidden_dim": 32,
        "text_backbone": _FakeTextBackbone(output_dim=6),
        "image_backbone": _FakeVisionBackbone(output_dim=6),
        "text_output_dim": 6,
        "image_output_dim": 6,
        "normalize_images": False,
    }
    model = build_model(cfg)
    packet = ObservationPacket(modalities={
        "text": torch.randint(0, 32, (2, 5)),
        "vector": torch.randn(2, 16),
        "image": torch.randn(2, 3, 8, 8),
    })
    out = model(packet, torch.randn(2, 8), obs_tp1=packet)

    assert out.predicted_next_latent.z_sem.shape == (2, 16)
    assert isinstance(model.encoder, PretrainedMultimodalEncoder)
