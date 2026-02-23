"""Loss functions for LGQ training.

Combines:
  1. Reconstruction loss (L1 + perceptual)
  2. Quantizer losses (commitment, free-energy, confidence, balance)
  3. Adversarial loss (hinge GAN)
  4. Perceptual loss (feature-matching via discriminator or VGG-style)

The free-energy objective interprets the quantizer as a variational
inference problem over an isotropic Gaussian mixture, yielding a principled
ELBO-derived loss that updates all codebook entries proportionally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    """Lightweight perceptual loss using multi-scale frequency domain.

    Avoids dependency on pretrained VGG by using multi-scale FFT-based
    feature comparison, similar to the approach in omnilatent/training/losses.py.
    For proper LPIPS, use the evaluation metrics module instead.
    """

    def __init__(self, scales: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.scales = scales

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)
        for s in self.scales:
            if s > 1:
                p = F.avg_pool2d(pred, s)
                t = F.avg_pool2d(target, s)
            else:
                p = pred
                t = target
            # FFT-based perceptual distance
            h, w = p.shape[-2], p.shape[-1]
            fft_h = 1 << (h - 1).bit_length()
            fft_w = 1 << (w - 1).bit_length()
            p_freq = torch.fft.rfft2(p, s=(fft_h, fft_w))
            t_freq = torch.fft.rfft2(t, s=(fft_h, fft_w))
            loss = loss + F.l1_loss(p_freq.abs(), t_freq.abs())
        return loss / len(self.scales)


class LGQLoss(nn.Module):
    """Combined training loss for LGQVAE.

    Manages the interplay between reconstruction, quantizer regularization,
    and adversarial objectives with configurable weights and scheduling.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.perceptual_loss = PerceptualLoss()

    def generator_loss(
        self,
        x: torch.Tensor,
        model_out: dict[str, torch.Tensor],
        adv_loss: torch.Tensor | None = None,
        step: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Compute generator (encoder + quantizer + decoder) loss.

        Args:
            x: [B, C, H, W] input images
            model_out: output dict from LGQVAE.forward()
            adv_loss: adversarial loss from model.generator_adversarial_loss()
            step: current training step (for scheduling)

        Returns dict with individual and total losses.
        """
        recon = model_out["recon"]
        cfg = self.config

        # --- Reconstruction ---
        recon_l1 = F.l1_loss(recon, x)
        recon_l2 = F.mse_loss(recon, x)
        recon_loss = recon_l1 + 0.5 * recon_l2

        # --- Perceptual ---
        perceptual = self.perceptual_loss(recon, x)

        # --- Quantizer losses ---
        commitment = model_out["commitment_loss"]
        free_energy = model_out["free_energy"]
        confidence = model_out["confidence_loss"]
        balance = model_out["balance_loss"]

        codebook_loss = (
            cfg.commitment_weight * commitment
            + cfg.free_energy_weight * free_energy
            + cfg.confidence_weight * confidence
            + cfg.balance_weight * balance
        )

        # --- Adversarial ---
        use_disc = step >= cfg.disc_start_step and adv_loss is not None
        adv_term = adv_loss if use_disc else torch.tensor(0.0, device=x.device)

        # --- Total ---
        total = (
            cfg.recon_weight * recon_loss
            + cfg.perceptual_weight * perceptual
            + cfg.codebook_weight * codebook_loss
            + (cfg.adversarial_weight * adv_term if use_disc else 0.0)
        )

        return {
            "total": total,
            "recon_loss": recon_loss.detach(),
            "recon_l1": recon_l1.detach(),
            "recon_l2": recon_l2.detach(),
            "perceptual": perceptual.detach(),
            "commitment": commitment.detach() if isinstance(commitment, torch.Tensor) else commitment,
            "free_energy": free_energy.detach() if isinstance(free_energy, torch.Tensor) else free_energy,
            "confidence": confidence.detach() if isinstance(confidence, torch.Tensor) else confidence,
            "balance": balance.detach() if isinstance(balance, torch.Tensor) else balance,
            "codebook_loss": codebook_loss.detach(),
            "adversarial": adv_term.detach() if isinstance(adv_term, torch.Tensor) else adv_term,
            "temperature": model_out.get("temperature", 0.0),
        }
