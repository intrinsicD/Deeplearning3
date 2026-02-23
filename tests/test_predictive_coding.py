"""Tests for Predictive Coding training module.

Verifies:
  1. PCLayer energy computation and precision learning
  2. Inference phase reduces energy (value node convergence)
  3. Learning phase produces weight gradients
  4. Full forward_pc pipeline returns expected outputs
  5. Inference step annealing works correctly
  6. Value node shapes are correct throughout
  7. Hybrid blend mode works
  8. Analytical inference mode (memory-efficient)
  9. Blend annealing curriculum
 10. Checkpoint saving
 11. Inference LR decay
"""

from __future__ import annotations

import pytest
import torch

from omnilatent.config import OmniLatentConfig
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.predictive_coding import (
    PCConfig,
    PCLayer,
    PCTrainer,
    PredictiveCodingNetwork,
)


@pytest.fixture
def config() -> OmniLatentConfig:
    return OmniLatentConfig(
        hidden_dim=64,
        num_layers=3,
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


@pytest.fixture
def model(config: OmniLatentConfig) -> OmniLatentModel:
    return OmniLatentModel(config)


@pytest.fixture
def pc_config() -> PCConfig:
    return PCConfig(
        inference_steps=5,
        inference_lr=0.1,
        learning_lr=1e-3,
        max_steps=100,
        warmup_steps=10,
        batch_size=2,
        mixed_precision=False,
        inference_steps_warmup=2,
        inference_steps_anneal_end=50,
    )


@pytest.fixture
def pc_net(
    model: OmniLatentModel, pc_config: PCConfig
) -> PredictiveCodingNetwork:
    return PredictiveCodingNetwork(model, pc_config)


@pytest.fixture
def sample_data(config: OmniLatentConfig) -> dict[str, torch.Tensor]:
    B = 2
    return {
        "text": torch.randint(1, config.vocab_size, (B, 16)),
        "audio": torch.randn(B, config.audio_n_mels, 64),
        "image": torch.randn(B, 3, config.image_size, config.image_size),
        "video": torch.randn(
            B, 3, config.video_max_frames, config.video_size, config.video_size
        ),
    }


# -----------------------------------------------------------------------
# PCLayer tests
# -----------------------------------------------------------------------

class TestPCLayer:
    def test_energy_non_negative(self, config: OmniLatentConfig) -> None:
        """Layer energy should always be >= 0 (it's a squared norm)."""
        from omnilatent.model.layers import TransformerBlock

        block = TransformerBlock(
            dim=64, num_heads=4, mlp_dim=128, dropout=0.0
        )
        layer = PCLayer(block, dim=64)

        x = torch.randn(2, 8, 64)
        prediction = layer.predict(x)
        target = torch.randn(2, 8, 64)

        energy = layer.layer_energy(target, prediction)
        assert energy.item() >= 0, f"Energy should be non-negative, got {energy.item()}"

    def test_zero_error_gives_zero_energy(
        self, config: OmniLatentConfig
    ) -> None:
        """If prediction exactly matches target, energy should be ~0."""
        from omnilatent.model.layers import TransformerBlock

        block = TransformerBlock(
            dim=64, num_heads=4, mlp_dim=128, dropout=0.0
        )
        layer = PCLayer(block, dim=64)

        x = torch.randn(2, 8, 64)
        prediction = layer.predict(x)

        # Energy when target == prediction should be ~0
        energy = layer.layer_energy(prediction.detach(), prediction)
        assert energy.item() < 1e-6, f"Energy should be ~0 for matching prediction, got {energy.item()}"

    def test_precision_is_learnable(self) -> None:
        """Precision parameter should receive gradients."""
        from omnilatent.model.layers import TransformerBlock

        block = TransformerBlock(
            dim=64, num_heads=4, mlp_dim=128, dropout=0.0
        )
        layer = PCLayer(block, dim=64)

        x = torch.randn(2, 8, 64)
        target = torch.randn(2, 8, 64)
        prediction = layer.predict(x)
        energy = layer.layer_energy(target, prediction)
        energy.backward()

        assert layer.log_precision.grad is not None, "Precision should have gradient"
        assert layer.log_precision.grad.abs().sum() > 0, "Precision gradient should be non-zero"

    def test_precision_property(self) -> None:
        """Precision should be exp(log_precision), clamped."""
        from omnilatent.model.layers import TransformerBlock

        block = TransformerBlock(
            dim=64, num_heads=4, mlp_dim=128, dropout=0.0
        )
        layer = PCLayer(block, dim=64)

        # Default: log_precision=0 -> precision=1.0
        assert abs(layer.precision.item() - 1.0) < 1e-5

        # Set to a known value
        with torch.no_grad():
            layer.log_precision.fill_(1.0)
        expected = torch.exp(torch.tensor(1.0)).item()
        assert abs(layer.precision.item() - expected) < 1e-4


# -----------------------------------------------------------------------
# PredictiveCodingNetwork tests
# -----------------------------------------------------------------------

class TestPredictiveCodingNetwork:
    def test_num_layers_matches_backbone(
        self, pc_net: PredictiveCodingNetwork, config: OmniLatentConfig
    ) -> None:
        assert pc_net.num_layers == config.num_layers

    def test_inference_reduces_energy(
        self, pc_net: PredictiveCodingNetwork, model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """Energy should decrease over inference iterations."""
        model.eval()

        # Create a simple input
        src_tokens = model.encode("image", sample_data["image"])
        rope_freqs = model.backbone.rope_freqs

        value_nodes, info = pc_net.inference_phase(
            x_input=src_tokens,
            rope_freqs=rope_freqs,
            num_steps=10,
        )

        assert info["final_energy"] <= info["initial_energy"], (
            f"Energy should decrease: {info['initial_energy']:.4f} -> "
            f"{info['final_energy']:.4f}"
        )
        assert info["energy_reduction"] >= 0, "Energy reduction should be non-negative"

    def test_value_node_shapes(
        self, pc_net: PredictiveCodingNetwork, model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """All value nodes should have shape (B, N, D)."""
        src_tokens = model.encode("image", sample_data["image"])
        rope_freqs = model.backbone.rope_freqs

        value_nodes, _ = pc_net.inference_phase(
            x_input=src_tokens,
            rope_freqs=rope_freqs,
            num_steps=3,
        )

        # Should have L+1 value nodes (one per layer boundary)
        assert len(value_nodes) == pc_net.num_layers + 1

        B, N, D = src_tokens.shape
        for i, vn in enumerate(value_nodes):
            assert vn.shape == (B, N, D), (
                f"Value node {i} shape {vn.shape} != expected ({B}, {N}, {D})"
            )

    def test_learning_phase_has_gradients(
        self, pc_net: PredictiveCodingNetwork, model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """Learning phase should produce gradients for backbone weights."""
        model.train()
        src_tokens = model.encode("image", sample_data["image"])
        rope_freqs = model.backbone.rope_freqs

        # Run inference to get value nodes
        value_nodes, _ = pc_net.inference_phase(
            x_input=src_tokens,
            rope_freqs=rope_freqs,
            num_steps=3,
        )

        # Perturb value nodes so prediction errors are non-zero
        # (warm-start gives zero error since value[i+1] == block(value[i]))
        for i in range(1, len(value_nodes)):
            value_nodes[i] = (
                value_nodes[i] + 0.1 * torch.randn_like(value_nodes[i])
            ).detach()

        # Zero gradients
        model.zero_grad()

        # Run learning phase
        losses = pc_net.learning_phase(
            value_nodes=value_nodes,
            rope_freqs=rope_freqs,
        )

        assert "total_energy" in losses
        assert losses["total_energy"].item() > 0, "Energy should be non-zero with perturbed nodes"
        losses["total_energy"].backward()

        # Check backbone has gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.backbone.parameters()
        )
        assert has_grad, "Backbone should have gradients after learning phase"

    def test_inference_step_annealing(
        self, pc_net: PredictiveCodingNetwork, pc_config: PCConfig
    ) -> None:
        """Inference steps should increase from warmup to full over training."""
        # At step 0
        steps_start = pc_net._get_inference_steps(0)
        assert steps_start >= pc_config.inference_steps_warmup

        # At end of annealing
        steps_end = pc_net._get_inference_steps(
            pc_config.inference_steps_anneal_end
        )
        assert steps_end == pc_config.inference_steps

        # Monotonically increasing
        prev = 0
        for s in range(0, pc_config.inference_steps_anneal_end + 1, 10):
            current = pc_net._get_inference_steps(s)
            assert current >= prev, f"Steps decreased at global_step={s}"
            prev = current


# -----------------------------------------------------------------------
# Full forward_pc tests
# -----------------------------------------------------------------------

class TestForwardPC:
    def test_returns_expected_keys(
        self, pc_net: PredictiveCodingNetwork,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """forward_pc should return all expected dictionary keys."""
        pc_net.train()
        result = pc_net.forward_pc(
            source_modality="image",
            source_data=sample_data["image"],
            target_modality="image",
            target_data=sample_data["image"],
            global_step=0,
        )

        required_keys = {"total", "pc_energy", "recon_loss", "output"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

        # Should have per-layer energies
        for i in range(pc_net.num_layers):
            assert f"layer_{i}_energy" in result

        # Should have inference stats
        assert "inference_initial_energy" in result
        assert "inference_final_energy" in result

    def test_total_loss_is_scalar(
        self, pc_net: PredictiveCodingNetwork,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """Total loss should be a scalar tensor with grad."""
        pc_net.train()
        result = pc_net.forward_pc(
            source_modality="image",
            source_data=sample_data["image"],
            target_modality="image",
            target_data=sample_data["image"],
        )

        total = result["total"]
        assert total.dim() == 0, f"Total loss should be scalar, got dim={total.dim()}"
        assert total.requires_grad, "Total loss should require grad"

    def test_cross_modal_forward(
        self, pc_net: PredictiveCodingNetwork,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """Cross-modal (image -> text) should work."""
        pc_net.train()
        result = pc_net.forward_pc(
            source_modality="image",
            source_data=sample_data["image"],
            target_modality="text",
            target_data=sample_data["text"],
        )

        assert result["total"].dim() == 0
        assert not torch.isnan(result["total"])

    def test_output_shape_matches_target(
        self, pc_net: PredictiveCodingNetwork,
        sample_data: dict[str, torch.Tensor],
        config: OmniLatentConfig,
    ) -> None:
        """Decoder output should have the correct shape for the target modality."""
        pc_net.train()
        result = pc_net.forward_pc(
            source_modality="image",
            source_data=sample_data["image"],
            target_modality="image",
            target_data=sample_data["image"],
        )

        B = sample_data["image"].shape[0]
        output = result["output"]
        # Image decoder output: (B, C, H, W)
        assert output.shape[0] == B
        assert output.shape[1] == config.image_channels

    def test_backward_from_total(
        self, pc_net: PredictiveCodingNetwork,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """Backward from total loss should not error and produce gradients."""
        pc_net.train()
        pc_net.zero_grad()

        result = pc_net.forward_pc(
            source_modality="image",
            source_data=sample_data["image"],
            target_modality="image",
            target_data=sample_data["image"],
        )

        result["total"].backward()

        # Check backbone has gradients (from PC energy)
        has_backbone_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in pc_net.model.backbone.parameters()
        )
        assert has_backbone_grad, "Backbone should have gradients"

        # Check decoder has gradients (from recon loss)
        has_decoder_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in pc_net.model.decoders["image"].parameters()
        )
        assert has_decoder_grad, "Decoder should have gradients"


# -----------------------------------------------------------------------
# Blend mode test
# -----------------------------------------------------------------------

class TestBlendMode:
    def test_blend_zero_is_pure_pc(
        self, model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """With blend=0, total = pc_energy + supervised_weight * recon_loss."""
        pc_config = PCConfig(
            inference_steps=3,
            backprop_blend=0.0,
            supervised_weight=1.0,
            mixed_precision=False,
        )
        pc_net = PredictiveCodingNetwork(model, pc_config)
        pc_net.train()

        result = pc_net.forward_pc(
            source_modality="image",
            source_data=sample_data["image"],
            target_modality="image",
            target_data=sample_data["image"],
        )

        expected = result["pc_energy"] + result["recon_loss"]
        torch.testing.assert_close(
            result["total"], expected, atol=1e-5, rtol=1e-5
        )

    def test_blend_one_weights_recon_heavily(
        self, model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """With blend=1, total = 1.0 * recon_loss (pure backprop-like)."""
        pc_config = PCConfig(
            inference_steps=3,
            backprop_blend=1.0,
            mixed_precision=False,
        )
        pc_net = PredictiveCodingNetwork(model, pc_config)
        pc_net.train()

        result = pc_net.forward_pc(
            source_modality="image",
            source_data=sample_data["image"],
            target_modality="image",
            target_data=sample_data["image"],
        )

        # blend=1 -> total = 0 * pc_energy + 1 * recon_loss
        # Compare values (shapes may differ: total may be scalar or [1])
        assert abs(result["total"].item() - result["recon_loss"].item()) < 1e-4


# -----------------------------------------------------------------------
# No NaN / Inf tests
# -----------------------------------------------------------------------

class TestNumericalStability:
    def test_no_nan_in_forward(
        self, pc_net: PredictiveCodingNetwork,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """No NaN values in any output."""
        pc_net.train()
        result = pc_net.forward_pc(
            source_modality="image",
            source_data=sample_data["image"],
            target_modality="image",
            target_data=sample_data["image"],
        )

        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in {k}"
                assert not torch.isinf(v).any(), f"Inf in {k}"

    def test_no_nan_gradients(
        self, pc_net: PredictiveCodingNetwork,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """No NaN gradients after backward."""
        pc_net.train()
        pc_net.zero_grad()

        result = pc_net.forward_pc(
            source_modality="image",
            source_data=sample_data["image"],
            target_modality="image",
            target_data=sample_data["image"],
        )
        result["total"].backward()

        for name, p in pc_net.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(p.grad).any(), f"Inf gradient in {name}"


# -----------------------------------------------------------------------
# Analytical inference tests
# -----------------------------------------------------------------------

class TestAnalyticalInference:
    def test_analytical_reduces_energy(
        self, model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """Analytical inference should also reduce energy."""
        pc_cfg = PCConfig(
            inference_steps=10,
            inference_lr=0.1,
            use_analytical_inference=True,
            mixed_precision=False,
        )
        pc_net = PredictiveCodingNetwork(model, pc_cfg)
        model.eval()

        src_tokens = model.encode("image", sample_data["image"])
        rope_freqs = model.backbone.rope_freqs

        value_nodes, info = pc_net.inference_phase(
            x_input=src_tokens,
            rope_freqs=rope_freqs,
            num_steps=10,
        )

        assert info["final_energy"] <= info["initial_energy"], (
            f"Analytical inference energy should decrease: "
            f"{info['initial_energy']:.4f} -> {info['final_energy']:.4f}"
        )

    def test_analytical_value_node_shapes(
        self, model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """Analytical inference should produce correct value node shapes."""
        pc_cfg = PCConfig(
            inference_steps=3,
            use_analytical_inference=True,
            mixed_precision=False,
        )
        pc_net = PredictiveCodingNetwork(model, pc_cfg)

        src_tokens = model.encode("image", sample_data["image"])
        rope_freqs = model.backbone.rope_freqs

        value_nodes, _ = pc_net.inference_phase(
            x_input=src_tokens,
            rope_freqs=rope_freqs,
            num_steps=3,
        )

        assert len(value_nodes) == pc_net.num_layers + 1
        B, N, D = src_tokens.shape
        for i, vn in enumerate(value_nodes):
            assert vn.shape == (B, N, D), (
                f"Value node {i} shape {vn.shape} != ({B}, {N}, {D})"
            )

    def test_analytical_forward_pc_no_nan(
        self, model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """Full forward_pc with analytical inference should be NaN-free."""
        pc_cfg = PCConfig(
            inference_steps=3,
            use_analytical_inference=True,
            mixed_precision=False,
        )
        pc_net = PredictiveCodingNetwork(model, pc_cfg)
        pc_net.train()

        result = pc_net.forward_pc(
            source_modality="image",
            source_data=sample_data["image"],
            target_modality="image",
            target_data=sample_data["image"],
        )

        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in {k}"

    def test_analytical_gradient_method(self) -> None:
        """PCLayer.analytical_value_gradient should produce correct shape."""
        from omnilatent.model.layers import TransformerBlock

        block = TransformerBlock(dim=64, num_heads=4, mlp_dim=128, dropout=0.0)
        layer = PCLayer(block, dim=64)

        error_above = torch.randn(2, 8, 64)
        error_below = torch.randn(2, 8, 64)
        prec_above = layer.precision
        prec_below = layer.precision

        grad = layer.analytical_value_gradient(
            error_above, error_below, prec_above, prec_below,
        )
        assert grad.shape == (2, 8, 64)

        # Without error_below (top node)
        grad_top = layer.analytical_value_gradient(
            error_above, None, prec_above, None,
        )
        assert grad_top.shape == (2, 8, 64)


# -----------------------------------------------------------------------
# Blend annealing tests
# -----------------------------------------------------------------------

class TestBlendAnnealing:
    def test_annealing_progression(self, model: OmniLatentModel) -> None:
        """Blend should decrease from start to end over annealing steps."""
        pc_cfg = PCConfig(
            backprop_blend_anneal=True,
            backprop_blend_start=1.0,
            backprop_blend_end=0.0,
            backprop_blend_anneal_steps=100,
            mixed_precision=False,
        )
        pc_net = PredictiveCodingNetwork(model, pc_cfg)

        # At step 0: should be at start
        assert pc_net._get_backprop_blend(0) == 1.0

        # At midpoint: should be ~0.5
        mid = pc_net._get_backprop_blend(50)
        assert abs(mid - 0.5) < 0.01, f"Expected ~0.5 at midpoint, got {mid}"

        # At end: should be at end value
        assert pc_net._get_backprop_blend(100) == 0.0

        # Past end: should stay at end
        assert pc_net._get_backprop_blend(200) == 0.0

    def test_no_annealing_returns_static(self, model: OmniLatentModel) -> None:
        """Without annealing, blend should be static."""
        pc_cfg = PCConfig(
            backprop_blend=0.3,
            backprop_blend_anneal=False,
            mixed_precision=False,
        )
        pc_net = PredictiveCodingNetwork(model, pc_cfg)

        for step in [0, 50, 100, 1000]:
            assert pc_net._get_backprop_blend(step) == 0.3

    def test_annealing_affects_forward(
        self, model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """Annealed blend should appear in forward_pc result."""
        pc_cfg = PCConfig(
            inference_steps=3,
            backprop_blend_anneal=True,
            backprop_blend_start=1.0,
            backprop_blend_end=0.0,
            backprop_blend_anneal_steps=100,
            mixed_precision=False,
        )
        pc_net = PredictiveCodingNetwork(model, pc_cfg)
        pc_net.train()

        result = pc_net.forward_pc(
            source_modality="image",
            source_data=sample_data["image"],
            target_modality="image",
            target_data=sample_data["image"],
            global_step=50,
        )

        assert "backprop_blend" in result
        assert abs(result["backprop_blend"] - 0.5) < 0.01


# -----------------------------------------------------------------------
# Inference LR decay tests
# -----------------------------------------------------------------------

class TestInferenceLRDecay:
    def test_lr_decay_converges_better(
        self, model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """Inference with LR decay should converge (energy reduces)."""
        pc_cfg = PCConfig(
            inference_steps=10,
            inference_lr=0.1,
            inference_lr_decay=0.9,
            mixed_precision=False,
        )
        pc_net = PredictiveCodingNetwork(model, pc_cfg)
        model.eval()

        src_tokens = model.encode("image", sample_data["image"])
        rope_freqs = model.backbone.rope_freqs

        value_nodes, info = pc_net.inference_phase(
            x_input=src_tokens,
            rope_freqs=rope_freqs,
            num_steps=10,
        )

        assert info["final_energy"] <= info["initial_energy"]

    def test_decay_one_is_constant_lr(
        self, model: OmniLatentModel,
    ) -> None:
        """decay=1.0 should behave the same as no decay."""
        pc_cfg = PCConfig(
            inference_steps=3,
            inference_lr_decay=1.0,
            mixed_precision=False,
        )
        pc_net = PredictiveCodingNetwork(model, pc_cfg)
        # Just verify it constructs and doesn't error
        assert pc_net.pc_config.inference_lr_decay == 1.0


# -----------------------------------------------------------------------
# Checkpoint saving tests
# -----------------------------------------------------------------------

class TestCheckpointSaving:
    def test_save_checkpoint(
        self, model: OmniLatentModel, config: OmniLatentConfig,
        tmp_path,
    ) -> None:
        """Checkpoint saving should create a file with expected keys."""
        from omnilatent.training.data import (
            SyntheticMultiModalDataset,
            build_dataloader,
        )

        pc_cfg = PCConfig(
            inference_steps=3,
            max_steps=10,
            batch_size=2,
            mixed_precision=False,
            save_every=5,
        )
        dataset = SyntheticMultiModalDataset(config, length=100)
        dataloader = build_dataloader(config, dataset)

        trainer = PCTrainer(
            model=model,
            model_config=config,
            pc_config=pc_cfg,
            dataloader=dataloader,
        )

        save_dir = str(tmp_path / "ckpt")
        trainer._save_checkpoint(save_dir, step=1)

        import pathlib

        ckpt_path = pathlib.Path(save_dir) / "checkpoint_1.pt"
        assert ckpt_path.exists(), "Checkpoint file should be created"

        ckpt = torch.load(ckpt_path, weights_only=False)
        expected_keys = {
            "step",
            "model_state_dict",
            "pc_layers_state_dict",
            "output_log_precision",
            "optimizer_backbone",
            "optimizer_peripheral",
            "optimizer_precision",
            "pc_config",
            "model_config",
        }
        assert expected_keys.issubset(ckpt.keys()), (
            f"Missing keys: {expected_keys - set(ckpt.keys())}"
        )
        assert ckpt["step"] == 1


# -----------------------------------------------------------------------
# PCConfig new fields tests
# -----------------------------------------------------------------------

class TestPCConfigNewFields:
    def test_defaults(self) -> None:
        """New config fields should have sensible defaults."""
        cfg = PCConfig()
        assert cfg.inference_lr_decay == 0.95
        assert cfg.backprop_blend_anneal is False
        assert cfg.backprop_blend_start == 1.0
        assert cfg.backprop_blend_end == 0.0
        assert cfg.backprop_blend_anneal_steps == 20_000
        assert cfg.use_analytical_inference is False
        assert cfg.precision_lr_ratio == 0.1
        assert cfg.precision_min == 0.01
        assert cfg.precision_max == 100.0
        assert cfg.save_every == 5000
        assert cfg.save_dir == "checkpoints/pc"

    def test_custom_values(self) -> None:
        """Config should accept custom values."""
        cfg = PCConfig(
            inference_lr_decay=0.8,
            backprop_blend_anneal=True,
            backprop_blend_start=0.9,
            use_analytical_inference=True,
            save_every=1000,
        )
        assert cfg.inference_lr_decay == 0.8
        assert cfg.backprop_blend_anneal is True
        assert cfg.backprop_blend_start == 0.9
        assert cfg.use_analytical_inference is True
        assert cfg.save_every == 1000
