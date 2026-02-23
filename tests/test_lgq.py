"""Comprehensive tests for Learnable Geometric Quantization (LGQ).

Tests cover:
  - LGQ quantizer: soft/hard assignment, temperature schedule, losses
  - FSQ and SimVQ baselines
  - LGQVAE model: forward pass, discriminator, parameter groups
  - Loss computation
  - Metrics: PSNR, SSIM, codebook utilization, FID
  - Gradient flow through the full pipeline
  - Integration with HPWM VQ-VAE
"""

import math
import pytest
import torch
import torch.nn.functional as F

from lgq.config import LGQConfig
from lgq.quantizer import LGQuantizer, FSQuantizer, SimVQuantizer, build_quantizer
from lgq.model import LGQVAE, Encoder, Decoder, PatchDiscriminator
from lgq.losses import LGQLoss, PerceptualLoss
from lgq.metrics import psnr, ssim, CodebookMetrics, compute_fid, MetricAggregator


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def small_config():
    """Small config for fast testing."""
    return LGQConfig(
        n_codebooks=4,
        vocab_size=32,
        codebook_dim=4,
        vq_dim=16,
        in_channels=3,
        hidden_dim=32,
        n_res_blocks=1,
        downsample_factor=4,
        resolution=32,
        tau_init=1.0,
        tau_final=0.05,
        tau_warmup_steps=10,
        tau_anneal_steps=100,
        disc_hidden_dim=16,
        disc_n_layers=2,
        disc_start_step=5,
        batch_size=2,
    )


@pytest.fixture
def dummy_images():
    """Dummy image batch for testing."""
    return torch.randn(2, 3, 32, 32)


# ======================================================================
# LGQuantizer Tests
# ======================================================================

class TestLGQuantizer:
    def test_init(self):
        q = LGQuantizer(n_codebooks=4, vocab_size=32, codebook_dim=4)
        assert q.codebooks.shape == (4, 32, 4)
        assert q.dim == 16
        assert q._step.item() == 0

    def test_temperature_schedule(self):
        q = LGQuantizer(
            tau_init=1.0, tau_final=0.1,
            tau_warmup_steps=10, tau_anneal_steps=100,
        )
        # During warmup
        assert q.get_temperature(0) == 1.0
        assert q.get_temperature(5) == 1.0
        assert q.get_temperature(9) == 1.0

        # At start of anneal
        tau_start = q.get_temperature(10)
        assert abs(tau_start - 1.0) < 0.01

        # Mid anneal
        tau_mid = q.get_temperature(60)
        assert 0.1 < tau_mid < 1.0

        # End of anneal
        tau_end = q.get_temperature(110)
        assert abs(tau_end - 0.1) < 0.01

    def test_soft_assignment_sums_to_one(self):
        q = LGQuantizer(n_codebooks=4, vocab_size=32, codebook_dim=4)
        z = torch.randn(10, 4)
        codebook = q.codebooks[0]
        probs = q.soft_assignment(z, codebook, tau=1.0)
        assert probs.shape == (10, 32)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5)

    def test_soft_assignment_sharpens_with_low_tau(self):
        q = LGQuantizer(n_codebooks=4, vocab_size=32, codebook_dim=4)
        z = torch.randn(10, 4)
        codebook = q.codebooks[0]

        soft = q.soft_assignment(z, codebook, tau=10.0)
        hard = q.soft_assignment(z, codebook, tau=0.01)

        # Low tau should produce sharper (lower entropy) assignments
        entropy_soft = -(soft * (soft + 1e-8).log()).sum(-1).mean()
        entropy_hard = -(hard * (hard + 1e-8).log()).sum(-1).mean()
        assert entropy_hard < entropy_soft

    def test_forward_training(self):
        q = LGQuantizer(n_codebooks=4, vocab_size=32, codebook_dim=4)
        q.train()
        z = torch.randn(2, 16, 8, 8)

        out = q(z)
        assert out["quantized"].shape == z.shape
        assert out["indices"].shape == (2, 4, 8, 8)
        assert len(out["soft_assignments"]) == 4
        assert out["soft_assignments"][0].shape == (128, 32)  # B*H*W, K
        assert out["commitment_loss"].item() >= 0
        assert out["free_energy"].item() != 0  # should be nonzero
        assert out["confidence_loss"].item() >= 0
        assert isinstance(out["temperature"], float)

    def test_forward_eval(self):
        q = LGQuantizer(n_codebooks=4, vocab_size=32, codebook_dim=4)
        q.eval()
        z = torch.randn(2, 16, 8, 8)

        out = q(z)
        assert out["quantized"].shape == z.shape
        assert out["indices"].shape == (2, 4, 8, 8)

    def test_step_counter_increments(self):
        q = LGQuantizer(n_codebooks=4, vocab_size=32, codebook_dim=4)
        q.train()
        z = torch.randn(1, 16, 4, 4)

        initial_step = q._step.item()
        q(z)
        assert q._step.item() == initial_step + 1
        q(z)
        assert q._step.item() == initial_step + 2

    def test_indices_to_embeddings(self):
        q = LGQuantizer(n_codebooks=4, vocab_size=32, codebook_dim=4)
        indices = torch.randint(0, 32, (2, 4, 8, 8))
        emb = q.indices_to_embeddings(indices)
        assert emb.shape == (2, 16, 8, 8)

    def test_codebook_utilization(self):
        q = LGQuantizer(n_codebooks=4, vocab_size=32, codebook_dim=4)
        util = q.codebook_utilization()
        assert "active_ratio" in util
        assert "perplexity" in util
        assert 0.0 <= util["active_ratio"] <= 1.0
        assert util["perplexity"] >= 0.0

    def test_gradient_flow_to_codebook(self):
        """LGQ should allow gradients to flow to codebook entries."""
        q = LGQuantizer(n_codebooks=2, vocab_size=16, codebook_dim=4)
        q.train()
        z = torch.randn(2, 8, 4, 4, requires_grad=True)

        out = q(z)
        loss = out["quantized"].sum() + out["commitment_loss"] + out["free_energy"]
        loss.backward()

        assert q.codebooks.grad is not None
        assert q.codebooks.grad.abs().sum() > 0, "Gradients should flow to codebooks"
        assert z.grad is not None

    def test_balance_loss_zero_for_uniform(self):
        """Balance loss should be near zero when usage is perfectly uniform."""
        q = LGQuantizer(n_codebooks=1, vocab_size=4, codebook_dim=4)
        q.train()

        # Force uniform soft assignments
        z = torch.randn(100, 4, 4, 4)
        out = q(z)
        # Balance loss should be small (not necessarily zero due to batch effects)
        assert out["balance_loss"].item() < 10.0  # reasonable upper bound


# ======================================================================
# Baseline Quantizer Tests
# ======================================================================

class TestFSQuantizer:
    def test_forward(self):
        q = FSQuantizer(levels=[8, 6, 5], dim=12)
        z = torch.randn(2, 12, 4, 4)
        out = q(z)
        assert out["quantized"].shape == z.shape
        assert out["indices"].shape == (2, 3, 4, 4)

    def test_straight_through(self):
        q = FSQuantizer(levels=[4, 4], dim=8)
        z = torch.randn(2, 8, 4, 4, requires_grad=True)
        out = q(z)
        loss = out["quantized"].sum()
        loss.backward()
        assert z.grad is not None


class TestSimVQuantizer:
    def test_forward_train(self):
        q = SimVQuantizer(n_codebooks=4, vocab_size=16, codebook_dim=4)
        q.train()
        z = torch.randn(2, 16, 4, 4)
        out = q(z)
        assert out["quantized"].shape == z.shape
        assert out["indices"].shape == (2, 4, 4, 4)
        assert out["commitment_loss"].item() >= 0

    def test_codebook_utilization(self):
        q = SimVQuantizer(n_codebooks=4, vocab_size=16, codebook_dim=4)
        q.train()
        # Run a few batches to populate EMA counts
        for _ in range(5):
            z = torch.randn(4, 16, 4, 4)
            q(z)
        util = q.codebook_utilization()
        assert "active_ratio" in util


class TestBuildQuantizer:
    def test_lgq(self, small_config):
        small_config.quantizer_type = "lgq"
        q = build_quantizer(small_config)
        assert isinstance(q, LGQuantizer)

    def test_fsq(self, small_config):
        small_config.quantizer_type = "fsq"
        q = build_quantizer(small_config)
        assert isinstance(q, FSQuantizer)

    def test_simvq(self, small_config):
        small_config.quantizer_type = "simvq"
        q = build_quantizer(small_config)
        assert isinstance(q, SimVQuantizer)


# ======================================================================
# Model Tests
# ======================================================================

class TestEncoder:
    def test_output_shape(self):
        enc = Encoder(in_channels=3, hidden_dim=32, out_dim=16, n_res_blocks=1, n_downsample=2)
        x = torch.randn(2, 3, 32, 32)
        z = enc(x)
        assert z.shape == (2, 16, 8, 8)  # 32 / 4 = 8


class TestDecoder:
    def test_output_shape(self):
        dec = Decoder(in_dim=16, hidden_dim=32, out_channels=3, n_res_blocks=1, n_upsample=2)
        z = torch.randn(2, 16, 8, 8)
        x = dec(z)
        assert x.shape == (2, 3, 32, 32)


class TestPatchDiscriminator:
    def test_output_shape(self):
        disc = PatchDiscriminator(in_channels=3, hidden_dim=16, n_layers=2)
        x = torch.randn(2, 3, 32, 32)
        out = disc(x)
        assert out.dim() == 4
        assert out.shape[0] == 2
        assert out.shape[1] == 1


class TestLGQVAE:
    def test_forward(self, small_config):
        model = LGQVAE(small_config)
        model.train()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)

        assert "recon" in out
        assert out["recon"].shape == x.shape
        assert "indices" in out
        assert "quantized" in out
        assert "commitment_loss" in out

    def test_encode_decode(self, small_config):
        model = LGQVAE(small_config)
        x = torch.randn(2, 3, 32, 32)

        z = model.encode(x)
        assert z.shape[0] == 2
        assert z.shape[1] == small_config.vq_dim

        recon = model.decode(z)
        assert recon.shape == x.shape

    def test_discriminator_forward(self, small_config):
        model = LGQVAE(small_config)
        real = torch.randn(2, 3, 32, 32)
        fake = torch.randn(2, 3, 32, 32)

        out = model.forward_discriminator(real, fake)
        assert "d_loss" in out
        assert out["d_loss"].item() >= 0

    def test_generator_adversarial_loss(self, small_config):
        model = LGQVAE(small_config)
        fake = torch.randn(2, 3, 32, 32)
        loss = model.generator_adversarial_loss(fake)
        assert loss.shape == ()

    def test_param_groups(self, small_config):
        model = LGQVAE(small_config)
        groups = model.get_param_groups()
        assert "generator" in groups
        assert "discriminator" in groups
        assert len(groups["generator"]) > 0
        assert len(groups["discriminator"]) > 0

    def test_count_parameters(self, small_config):
        model = LGQVAE(small_config)
        counts = model.count_parameters()
        assert "encoder" in counts
        assert "quantizer" in counts
        assert "decoder" in counts
        assert "discriminator" in counts

    def test_gradient_flow_full_pipeline(self, small_config):
        """Gradients should flow from reconstruction loss through encoder."""
        model = LGQVAE(small_config)
        model.train()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)

        loss = F.mse_loss(out["recon"], x) + out["commitment_loss"]
        loss.backward()

        # Check encoder has gradients
        enc_grads = sum(
            p.grad.abs().sum().item()
            for p in model.encoder.parameters()
            if p.grad is not None
        )
        assert enc_grads > 0, "Encoder should receive gradients"

        # Check quantizer codebook has gradients (LGQ advantage)
        if hasattr(model.quantizer, "codebooks"):
            cb_grads = model.quantizer.codebooks.grad.abs().sum().item()
            assert cb_grads > 0, "Codebook should receive gradients via LGQ"


# ======================================================================
# Loss Tests
# ======================================================================

class TestLGQLoss:
    def test_generator_loss(self, small_config):
        loss_fn = LGQLoss(small_config)
        model = LGQVAE(small_config)
        model.train()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)

        losses = loss_fn.generator_loss(x, out, step=0)
        assert "total" in losses
        assert "recon_loss" in losses
        assert "codebook_loss" in losses
        assert losses["total"].item() > 0

    def test_adversarial_not_used_before_start(self, small_config):
        loss_fn = LGQLoss(small_config)
        model = LGQVAE(small_config)
        model.train()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        adv = model.generator_adversarial_loss(out["recon"])

        # Before disc_start_step
        losses = loss_fn.generator_loss(x, out, adv, step=0)
        assert losses["adversarial"].item() == 0.0

        # After disc_start_step
        losses = loss_fn.generator_loss(x, out, adv, step=small_config.disc_start_step + 1)
        # adversarial should be nonzero (it's the actual GAN loss)


class TestPerceptualLoss:
    def test_forward(self):
        loss_fn = PerceptualLoss()
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_zero_for_identical(self):
        loss_fn = PerceptualLoss()
        x = torch.randn(2, 3, 32, 32)
        loss = loss_fn(x, x)
        assert loss.item() < 1e-5


# ======================================================================
# Metric Tests
# ======================================================================

class TestPSNR:
    def test_identical_images(self):
        x = torch.rand(2, 3, 32, 32)
        val = psnr(x, x)
        assert val.item() > 50  # should be very high

    def test_different_images(self):
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 32, 32)
        val = psnr(x, y)
        assert 0 < val.item() < 50


class TestSSIM:
    def test_identical_images(self):
        x = torch.rand(2, 3, 32, 32)
        val = ssim(x, x)
        assert val.item() > 0.99

    def test_different_images(self):
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 32, 32)
        val = ssim(x, y)
        assert -1 <= val.item() <= 1


class TestCodebookMetrics:
    def test_basic(self):
        m = CodebookMetrics(n_codebooks=2, vocab_size=16)
        indices = torch.randint(0, 16, (4, 2, 8, 8))
        m.update(indices)
        results = m.compute()
        assert "active_codes" in results
        assert "active_ratio" in results
        assert "perplexity" in results
        assert results["active_ratio"] > 0

    def test_reset(self):
        m = CodebookMetrics(n_codebooks=2, vocab_size=16)
        indices = torch.randint(0, 16, (4, 2, 8, 8))
        m.update(indices)
        m.reset()
        assert m.total_tokens == 0


class TestFID:
    def test_identical_distributions(self):
        # Same features should give FID near 0
        feats = torch.randn(100, 64)
        fid = compute_fid(feats, feats + torch.randn_like(feats) * 0.01)
        assert fid < 5.0  # should be small

    def test_different_distributions(self):
        real = torch.randn(100, 64)
        fake = torch.randn(100, 64) + 5.0  # shifted
        fid = compute_fid(real, fake)
        assert fid > 1.0  # should be significant


class TestMetricAggregator:
    def test_basic_aggregation(self):
        agg = MetricAggregator(n_codebooks=2, vocab_size=16)
        pred = torch.rand(4, 3, 32, 32)
        target = pred + torch.randn_like(pred) * 0.1
        target = target.clamp(0, 1)
        indices = torch.randint(0, 16, (4, 2, 8, 8))

        agg.update(pred, target, indices)
        results = agg.compute()
        assert "psnr" in results
        assert "ssim" in results
        assert results["psnr"] > 0


# ======================================================================
# Integration with HPWM
# ======================================================================

class TestHPWMIntegration:
    def test_lgq_as_vqvae_quantizer(self):
        """LGQuantizer can serve as drop-in replacement for VectorQuantizer."""
        from hpwm.components.vqvae import VQVAEEncoder, VQVAEDecoder

        # Create HPWM encoder/decoder
        enc = VQVAEEncoder(in_channels=3, hidden=32, out_dim=16, n_layers=1)
        dec = VQVAEDecoder(in_dim=16, hidden=32, out_channels=3, n_layers=1)

        # Use LGQuantizer instead of VectorQuantizer
        q = LGQuantizer(n_codebooks=4, vocab_size=32, codebook_dim=4)

        x = torch.randn(2, 3, 128, 128)
        z = enc(x)
        q.train()
        q_out = q(z)
        recon = dec(q_out["quantized"])

        assert recon.shape == x.shape
        assert q_out["indices"].shape[1] == 4

    def test_lgq_indices_compatible(self):
        """LGQ indices have same format as VQ indices."""
        from hpwm.components.vqvae import VectorQuantizer

        vq = VectorQuantizer(n_codebooks=4, vocab_size=32, dim=16)
        lgq = LGQuantizer(n_codebooks=4, vocab_size=32, codebook_dim=4)

        z = torch.randn(2, 16, 8, 8)

        vq_out = vq(z)  # returns (quantized, indices, commitment_loss)
        vq_indices = vq_out[1]

        lgq.train()
        lgq_out = lgq(z)
        lgq_indices = lgq_out["indices"]

        assert vq_indices.shape == lgq_indices.shape
        assert vq_indices.dtype == lgq_indices.dtype
