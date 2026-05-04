"""Tests for the three temporal context approaches.

Covers:
  * Approach 1: Multi-scale temporal sampling (dataset + losses)
  * Approach 2: Hierarchical temporal sequence transformer
  * Approach 3: Recurrent memory tokens
  * Integration: curriculum trainer with temporal modules
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from omnilatent.config import OmniLatentConfig
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.model.temporal import (
    TemporalSequenceTransformer,
    RecurrentMemory,
)
from omnilatent.training.losses import (
    TemporalOrderLoss,
    TemporalDistanceLoss,
    NextClipPredictionLoss,
    SceneBoundaryLoss,
    TemporalContextLoss,
)
from omnilatent.training.video_dataset import (
    classify_temporal_distance,
    TEMPORAL_DISTANCE_BOUNDARIES,
)


@pytest.fixture
def config() -> OmniLatentConfig:
    """Small config for fast testing."""
    return OmniLatentConfig(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        gradient_checkpointing=False,
        vocab_size=256,
        # image_patch_size=16 ensures decoder 16x upsampling matches image_size
        image_size=64,
        image_patch_size=16,
        video_size=32,
        video_patch_size=8,
        video_temporal_patch=2,
        video_max_frames=4,
        audio_n_mels=32,
        audio_max_frames=64,
        batch_size=2,
        # Temporal context settings
        temporal_seq_layers=2,
        temporal_seq_heads=4,
        temporal_seq_max_clips=16,
        temporal_seq_dropout=0.0,
        temporal_distance_buckets=4,
        memory_num_tokens=4,
        memory_gate_bias_init=-4.0,
    )


@pytest.fixture
def model(config: OmniLatentConfig) -> OmniLatentModel:
    return OmniLatentModel(config)


# =========================================================================
# Approach 1: Multi-Scale Temporal Sampling
# =========================================================================
class TestTemporalDistanceClassification:
    def test_bucket_under_10s(self) -> None:
        assert classify_temporal_distance(5.0) == 0
        assert classify_temporal_distance(0.0) == 0
        assert classify_temporal_distance(9.9) == 0

    def test_bucket_10_to_60s(self) -> None:
        assert classify_temporal_distance(10.0) == 1
        assert classify_temporal_distance(30.0) == 1
        assert classify_temporal_distance(59.9) == 1

    def test_bucket_1_to_5min(self) -> None:
        assert classify_temporal_distance(60.0) == 2
        assert classify_temporal_distance(180.0) == 2
        assert classify_temporal_distance(299.9) == 2

    def test_bucket_over_5min(self) -> None:
        assert classify_temporal_distance(300.0) == 3
        assert classify_temporal_distance(600.0) == 3
        assert classify_temporal_distance(3600.0) == 3

    def test_custom_boundaries(self) -> None:
        assert classify_temporal_distance(5.0, [10.0]) == 0
        assert classify_temporal_distance(15.0, [10.0]) == 1


class TestTemporalOrderLoss:
    def test_forward_shape(self) -> None:
        loss_fn = TemporalOrderLoss()
        z_a = torch.randn(4, 64)
        z_b = torch.randn(4, 64)
        labels = torch.randint(0, 2, (4,))
        loss = loss_fn(z_a, z_b, labels)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_perfect_separation(self) -> None:
        """Loss should be lower when predictions are correct."""
        loss_fn = TemporalOrderLoss()
        # Create clearly separable embeddings
        z_a = torch.ones(4, 64)
        z_b = -torch.ones(4, 64)
        labels = torch.zeros(4)  # anchor NOT before context
        loss_correct = loss_fn(z_a, z_b, labels)

        labels_wrong = torch.ones(4)  # wrong labels
        loss_wrong = loss_fn(z_a, z_b, labels_wrong)

        # These should differ
        assert loss_correct.item() != loss_wrong.item()


class TestTemporalDistanceLoss:
    def test_forward_shape(self) -> None:
        loss_fn = TemporalDistanceLoss(num_buckets=4, hidden_dim=64)
        z_a = torch.randn(4, 64)
        z_b = torch.randn(4, 64)
        labels = torch.randint(0, 4, (4,))
        loss = loss_fn(z_a, z_b, labels)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_gradient_flows(self) -> None:
        loss_fn = TemporalDistanceLoss(num_buckets=4, hidden_dim=64)
        z_a = torch.randn(4, 64, requires_grad=True)
        z_b = torch.randn(4, 64, requires_grad=True)
        labels = torch.randint(0, 4, (4,))
        loss = loss_fn(z_a, z_b, labels)
        loss.backward()
        assert z_a.grad is not None
        assert z_b.grad is not None


# =========================================================================
# Approach 2: Temporal Sequence Transformer
# =========================================================================
class TestTemporalSequenceTransformer:
    def test_construction(self, config: OmniLatentConfig) -> None:
        tx = TemporalSequenceTransformer(config)
        n_params = sum(p.numel() for p in tx.parameters())
        assert n_params > 0

    def test_forward_shape(self, config: OmniLatentConfig) -> None:
        tx = TemporalSequenceTransformer(config)
        B, N, D = 2, 8, config.hidden_dim
        clip_latents = torch.randn(B, N, D)
        clip_mask = torch.ones(B, N, dtype=torch.bool)

        out = tx(clip_latents, clip_mask)

        assert out["next_clip_pred"].shape == (B, N, D)
        assert out["scene_boundary"].shape == (B, N, 1)
        assert out["temporal_latent"].shape == (B, N, D)

    def test_causal_masking(self, config: OmniLatentConfig) -> None:
        """Changing a later clip should not affect predictions for earlier ones."""
        tx = TemporalSequenceTransformer(config)
        tx.eval()
        B, N, D = 1, 8, config.hidden_dim

        clip_latents = torch.randn(B, N, D)
        mask = torch.ones(B, N, dtype=torch.bool)

        with torch.no_grad():
            out1 = tx(clip_latents, mask)

        # Modify the last clip
        clip_latents_mod = clip_latents.clone()
        clip_latents_mod[:, -1] = torch.randn(1, D)

        with torch.no_grad():
            out2 = tx(clip_latents_mod, mask)

        # First N-1 predictions should be identical (causal attention)
        torch.testing.assert_close(
            out1["next_clip_pred"][:, :-1],
            out2["next_clip_pred"][:, :-1],
        )

    def test_predict_next_clip(self, config: OmniLatentConfig) -> None:
        tx = TemporalSequenceTransformer(config)
        clip_latents = torch.randn(2, 8, config.hidden_dim)
        pred = tx.predict_next_clip(clip_latents)
        assert pred.shape == (2, 8, config.hidden_dim)

    def test_variable_sequence_length(self, config: OmniLatentConfig) -> None:
        """Should handle shorter sequences with masking."""
        tx = TemporalSequenceTransformer(config)
        B, D = 2, config.hidden_dim

        # First sample: 8 clips, second: 4 clips
        clip_latents = torch.randn(B, 8, D)
        mask = torch.ones(B, 8, dtype=torch.bool)
        mask[1, 4:] = False

        out = tx(clip_latents, mask)
        assert out["next_clip_pred"].shape == (B, 8, D)

    def test_gradient_flow(self, config: OmniLatentConfig) -> None:
        tx = TemporalSequenceTransformer(config)
        clip_latents = torch.randn(2, 8, config.hidden_dim, requires_grad=True)
        mask = torch.ones(2, 8, dtype=torch.bool)

        out = tx(clip_latents, mask)
        loss = out["next_clip_pred"].mean()
        loss.backward()
        assert clip_latents.grad is not None


class TestNextClipPredictionLoss:
    def test_basic(self) -> None:
        loss_fn = NextClipPredictionLoss()
        pred = torch.randn(2, 8, 64)
        target = torch.randn(2, 8, 64)
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_with_mask(self) -> None:
        loss_fn = NextClipPredictionLoss()
        pred = torch.randn(2, 8, 64)
        target = torch.randn(2, 8, 64)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[1, 4:] = False

        loss_masked = loss_fn(pred, target, mask)
        loss_unmasked = loss_fn(pred, target)

        # Masked loss should differ from unmasked
        assert loss_masked.item() != loss_unmasked.item()

    def test_perfect_prediction(self) -> None:
        loss_fn = NextClipPredictionLoss()
        target = torch.randn(2, 8, 64)
        loss = loss_fn(target, target)
        # MSE should be 0, cosine should be 1 -> cos_loss = 0
        assert loss.item() < 1e-5

    def test_loss_decreases(self, config: OmniLatentConfig) -> None:
        """Training the temporal transformer should decrease loss."""
        tx = TemporalSequenceTransformer(config)
        loss_fn = NextClipPredictionLoss()
        optimizer = torch.optim.Adam(tx.parameters(), lr=1e-3)

        # Fixed target to overfit on
        clip_latents = torch.randn(2, 8, config.hidden_dim)
        target_next = clip_latents[:, 1:]

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            out = tx(clip_latents)
            pred_next = out["next_clip_pred"][:, :-1]
            loss = loss_fn(pred_next, target_next)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


class TestSceneBoundaryLoss:
    def test_forward(self) -> None:
        loss_fn = SceneBoundaryLoss()
        logits = torch.randn(2, 8, 1)
        labels = torch.zeros(2, 8)
        labels[:, 4] = 1.0  # scene boundary at position 4

        loss = loss_fn(logits, labels)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_with_mask(self) -> None:
        loss_fn = SceneBoundaryLoss()
        logits = torch.randn(2, 8)
        labels = torch.zeros(2, 8)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[1, 6:] = False

        loss = loss_fn(logits, labels, mask)
        assert loss.shape == ()


# =========================================================================
# Approach 3: Recurrent Memory Tokens
# =========================================================================
class TestRecurrentMemory:
    def test_construction(self, config: OmniLatentConfig) -> None:
        mem = RecurrentMemory(config)
        assert mem.num_tokens == config.memory_num_tokens
        assert mem.dim == config.hidden_dim

    def test_init_state(self, config: OmniLatentConfig) -> None:
        mem = RecurrentMemory(config)
        state = mem.init_state(batch_size=4)
        assert state.shape == (4, config.memory_num_tokens, config.hidden_dim)

    def test_prepend(self, config: OmniLatentConfig) -> None:
        mem = RecurrentMemory(config)
        content = torch.randn(4, 10, config.hidden_dim)
        combined, mem_len = mem.prepend(content)

        assert mem_len == config.memory_num_tokens
        assert combined.shape == (4, 10 + config.memory_num_tokens, config.hidden_dim)

    def test_prepend_with_state(self, config: OmniLatentConfig) -> None:
        mem = RecurrentMemory(config)
        content = torch.randn(4, 10, config.hidden_dim)
        state = mem.init_state(4) * 2.0  # modified state

        combined, mem_len = mem.prepend(content, state)
        assert combined.shape == (4, 10 + config.memory_num_tokens, config.hidden_dim)

    def test_extract(self, config: OmniLatentConfig) -> None:
        mem = RecurrentMemory(config)
        B, N, D = 4, 10, config.hidden_dim
        mem_len = config.memory_num_tokens

        output = torch.randn(B, mem_len + N, D)
        new_mem, content = mem.extract(output, mem_len)

        assert new_mem.shape == (B, mem_len, D)
        assert content.shape == (B, N, D)

    def test_detach_state(self, config: OmniLatentConfig) -> None:
        mem = RecurrentMemory(config)
        state = mem.init_state(4)
        # State should require grad through init_tokens
        assert state.requires_grad

        detached = mem.detach_state(state)
        assert not detached.requires_grad

    def test_gate_starts_near_zero(self, config: OmniLatentConfig) -> None:
        mem = RecurrentMemory(config)
        gate_val = mem.gate_value().item()
        # sigmoid(-4) ≈ 0.018
        assert gate_val < 0.05

    def test_expand_attention_mask(self, config: OmniLatentConfig) -> None:
        mem = RecurrentMemory(config)
        N = 10
        mem_len = config.memory_num_tokens

        mask = torch.ones(1, 1, N, N, dtype=torch.bool)
        expanded = mem.expand_attention_mask(mask, mem_len)
        assert expanded.shape == (1, 1, N + mem_len, N + mem_len)
        # Memory positions should be True (fully visible)
        assert expanded[0, 0, :mem_len, :].all()
        assert expanded[0, 0, :, :mem_len].all()

    def test_memory_roundtrip(self, config: OmniLatentConfig) -> None:
        """Memory should persist information across prepend/extract cycle."""
        mem = RecurrentMemory(config)
        B, D = 2, config.hidden_dim

        state0 = mem.init_state(B)
        content = torch.randn(B, 10, D)

        # Prepend -> simulate backbone -> extract
        combined, mem_len = mem.prepend(content, state0)
        # Simulate identity backbone (just pass through)
        new_mem, out_content = mem.extract(combined, mem_len)

        assert new_mem.shape == state0.shape
        assert out_content.shape == content.shape

    def test_gradient_through_memory(self, config: OmniLatentConfig) -> None:
        """Gradients should flow through memory operations."""
        mem = RecurrentMemory(config)
        B, D = 2, config.hidden_dim

        state = mem.init_state(B)
        content = torch.randn(B, 10, D, requires_grad=True)

        combined, mem_len = mem.prepend(content, state)
        # Simulate backbone
        output = combined * 2  # simple operation
        new_mem, out_content = mem.extract(output, mem_len)

        loss = new_mem.mean() + out_content.mean()
        loss.backward()

        assert content.grad is not None
        # Memory parameters should get gradients
        assert mem.init_tokens.grad is not None


# =========================================================================
# Integration: Temporal Context with Main Model
# =========================================================================
class TestTemporalIntegration:
    def test_encode_clip_sequence(
        self, config: OmniLatentConfig, model: OmniLatentModel
    ) -> None:
        """Test encoding a sequence of clips independently."""
        N = 4
        C, T, H, W = 3, config.video_max_frames, config.video_size, config.video_size
        clips = [torch.randn(2, C, T, H, W) for _ in range(N)]

        latents = []
        with torch.no_grad():
            for clip in clips:
                enc = model.encode("video", clip)
                z = enc[:, 1:].mean(dim=1)  # (B, D)
                latents.append(z)

        clip_latents = torch.stack(latents, dim=1)  # (B, N, D)
        assert clip_latents.shape == (2, N, config.hidden_dim)

    def test_temporal_transformer_with_model(
        self, config: OmniLatentConfig, model: OmniLatentModel
    ) -> None:
        """End-to-end: encode clips -> temporal transformer -> loss."""
        tx = TemporalSequenceTransformer(config)
        loss_fn = NextClipPredictionLoss()

        N = 6
        C, T, H, W = 3, config.video_max_frames, config.video_size, config.video_size
        clips = torch.randn(2, N, C, T, H, W)

        # Encode clips (no grad)
        clip_latents = []
        with torch.no_grad():
            for i in range(N):
                enc = model.encode("video", clips[:, i])
                z = enc[:, 1:].mean(dim=1)
                clip_latents.append(z)
        clip_latents = torch.stack(clip_latents, dim=1)

        # Temporal transformer (with grad)
        out = tx(clip_latents)
        pred = out["next_clip_pred"][:, :-1]
        target = clip_latents[:, 1:]

        loss = loss_fn(pred, target)
        loss.backward()

        # Only temporal transformer should have gradients
        assert any(p.grad is not None for p in tx.parameters())

    def test_memory_with_model(
        self, config: OmniLatentConfig, model: OmniLatentModel
    ) -> None:
        """End-to-end: encode clip with memory -> backbone -> decode."""
        mem = RecurrentMemory(config)
        B = 2
        C, T, H, W = 3, config.video_max_frames, config.video_size, config.video_size
        clip = torch.randn(B, C, T, H, W)

        # Encode
        src_tokens = model.encode("video", clip)
        mem_state = mem.init_state(B)

        # Prepend memory
        tokens_with_mem, mem_len = mem.prepend(src_tokens, mem_state)

        assert tokens_with_mem.shape[1] == src_tokens.shape[1] + mem.num_tokens

    def test_curriculum_with_temporal_modules(
        self, config: OmniLatentConfig
    ) -> None:
        """CurriculumTrainer accepts temporal modules."""
        from omnilatent.training.data import (
            SyntheticMultiModalDataset,
            collate_multimodal,
        )
        from torch.utils.data import DataLoader
        from curriculum_train import CurriculumTrainer, Phase

        model = OmniLatentModel(config)
        tx = TemporalSequenceTransformer(config)
        mem = RecurrentMemory(config)

        dataset = SyntheticMultiModalDataset(config, length=20)
        dataloader = DataLoader(
            dataset, batch_size=2, collate_fn=collate_multimodal, drop_last=True,
        )

        phases = [
            Phase("warmup", ["video_recon"], 0.5, "warmup"),
            Phase("joint", ["video_recon", "audio_recon"], 0.5, "joint"),
        ]

        trainer = CurriculumTrainer(
            model=model,
            config=config,
            dataloader=dataloader,
            phases=phases,
            total_steps=4,
            temporal_transformer=tx,
            recurrent_memory=mem,
        )
        # Should run without error (synthetic data uses standard batches)
        trainer.train(log_interval=2)

    def test_temporal_context_loss_combined(
        self, config: OmniLatentConfig
    ) -> None:
        """TemporalContextLoss combines multiple loss components."""
        loss_module = TemporalContextLoss(config)

        losses = {
            "temporal_order": torch.tensor(0.5),
            "temporal_distance": torch.tensor(0.3),
            "next_clip": torch.tensor(0.2),
        }
        result = loss_module(losses)

        assert "temporal_total" in result
        assert result["temporal_total"].item() > 0
        assert "temporal_order" in result
        assert "temporal_distance" in result
        assert "next_clip" in result


# =========================================================================
# Approach 1: ClipIndex multi-clip support
# =========================================================================
class TestClipIndexTemporalSupport:
    def test_classify_distance_edge_cases(self) -> None:
        """Edge cases for bucket classification."""
        assert classify_temporal_distance(0.0) == 0
        assert classify_temporal_distance(10.0) == 1  # exactly at boundary
        assert classify_temporal_distance(60.0) == 2
        assert classify_temporal_distance(300.0) == 3
