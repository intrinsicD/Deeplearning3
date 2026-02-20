"""Tests for the video-watching training pipeline.

Since we can't rely on actual video files in CI, these tests cover:
  * Transcript parsing (SRT, plain text)
  * Simple tokenizer
  * Collation logic (now uses standard multi-modal collation)
  * CurriculumTrainer with synthetic data
  * Integration: curriculum training runs for a few steps
"""

from __future__ import annotations

import pytest
import torch

from omnilatent.config import OmniLatentConfig
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.video_dataset import (
    _parse_srt,
    _simple_tokenize,
    collate_video_watching,
)


@pytest.fixture
def config() -> OmniLatentConfig:
    return OmniLatentConfig(
        hidden_dim=64,
        num_layers=2,
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
        audio_max_frames=64,
        batch_size=2,
    )


class TestTranscriptParsing:
    def test_parse_srt(self) -> None:
        srt_text = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:03,000 --> 00:00:05,000
This is a test
"""
        result = _parse_srt(srt_text)
        assert "Hello world" in result
        assert "This is a test" in result
        assert "-->" not in result

    def test_parse_srt_with_tags(self) -> None:
        srt_text = """1
00:00:01,000 --> 00:00:03,000
<i>Italic text</i>
"""
        result = _parse_srt(srt_text)
        assert "Italic text" in result
        assert "<i>" not in result

    def test_parse_empty_srt(self) -> None:
        result = _parse_srt("")
        assert result == ""


class TestSimpleTokenizer:
    def test_basic_tokenization(self) -> None:
        tokens = _simple_tokenize("hello", max_len=10, vocab_size=256)
        assert tokens.dtype == torch.long
        assert tokens.shape[0] == 5  # "hello" is 5 bytes
        assert (tokens >= 1).all()
        assert (tokens < 256).all()

    def test_max_length_truncation(self) -> None:
        tokens = _simple_tokenize("a" * 100, max_len=10, vocab_size=256)
        assert tokens.shape[0] == 10

    def test_no_zero_tokens(self) -> None:
        """Token 0 is reserved for padding."""
        tokens = _simple_tokenize("test string", max_len=50, vocab_size=256)
        assert (tokens > 0).all()


class TestCollation:
    def test_collate_multimodal_format(self) -> None:
        """Collation works with the standard multi-modal dict format."""
        batch = [
            {
                "video": torch.randn(3, 4, 32, 32),
                "audio": torch.randn(32, 64),
                "image": torch.randn(3, 32, 32),
            },
            {
                "video": torch.randn(3, 4, 32, 32),
                "audio": torch.randn(32, 64),
                "image": torch.randn(3, 32, 32),
            },
        ]
        result = collate_video_watching(batch)
        assert "video" in result
        assert "audio" in result
        assert "image" in result
        assert result["video"].shape[0] == 2
        assert result["audio"].shape[0] == 2

    def test_collate_with_variable_audio(self) -> None:
        """Audio with different lengths gets padded."""
        batch = [
            {"audio": torch.randn(32, 32)},
            {"audio": torch.randn(32, 64)},
        ]
        result = collate_video_watching(batch)
        assert result["audio"].shape == (2, 32, 64)  # padded to max


class TestCurriculumTrainer:
    def test_curriculum_runs_synthetic(self, config: OmniLatentConfig) -> None:
        """Curriculum trainer can run a few steps with synthetic data."""
        from omnilatent.training.data import (
            SyntheticMultiModalDataset,
            collate_multimodal,
        )
        from torch.utils.data import DataLoader
        from curriculum_train import CurriculumTrainer, Phase

        model = OmniLatentModel(config)
        dataset = SyntheticMultiModalDataset(config, length=20)
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_multimodal,
            drop_last=True,
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
            total_steps=6,
        )
        trainer.train(log_interval=3)

    def test_phase_transitions(self, config: OmniLatentConfig) -> None:
        """Verify correct phase boundary computation."""
        from curriculum_train import CurriculumTrainer, Phase
        from omnilatent.training.data import (
            SyntheticMultiModalDataset,
            collate_multimodal,
        )
        from torch.utils.data import DataLoader

        model = OmniLatentModel(config)
        dataset = SyntheticMultiModalDataset(config, length=20)
        dataloader = DataLoader(
            dataset, batch_size=2, collate_fn=collate_multimodal, drop_last=True,
        )

        phases = [
            Phase("p1", ["video_recon"], 0.5, "first"),
            Phase("p2", ["audio_recon"], 0.5, "second"),
        ]

        trainer = CurriculumTrainer(
            model=model,
            config=config,
            dataloader=dataloader,
            phases=phases,
            total_steps=100,
        )

        # Step 0 should be in phase 0
        trainer.global_step = 0
        idx, phase = trainer._get_current_phase()
        assert idx == 0
        assert phase.name == "p1"

        # Step 50 should be in phase 1
        trainer.global_step = 50
        idx, phase = trainer._get_current_phase()
        assert idx == 1
        assert phase.name == "p2"

    def test_loss_decreases_curriculum(self, config: OmniLatentConfig) -> None:
        """Loss should decrease when overfitting on a fixed batch."""
        import torch.nn.functional as TF

        model = OmniLatentModel(config)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Fixed batch of images to overfit on
        images = torch.randn(2, 3, config.image_size, config.image_size)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            result = model.reconstruct("image", images)
            loss = TF.mse_loss(result["output"], images)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )
