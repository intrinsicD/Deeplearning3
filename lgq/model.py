"""VQGAN-style model with pluggable quantization (LGQ / FSQ / SimVQ).

Architecture follows the standard VQGAN blueprint:
  Encoder -> Quantizer -> Decoder
with an optional PatchGAN discriminator for adversarial training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lgq.config import LGQConfig
from lgq.quantizer import build_quantizer


class ResBlock(nn.Module):
    """Residual block with GroupNorm + SiLU."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Downsample(nn.Module):
    """Strided convolution downsample."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Transposed convolution upsample."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Encoder(nn.Module):
    """Multi-stage convolutional encoder with progressive downsampling."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 128,
        out_dim: int = 64,
        n_res_blocks: int = 3,
        n_downsample: int = 3,  # 8x downsample for n_downsample=3
    ):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, hidden_dim // 2, 3, padding=1)

        ch = hidden_dim // 2
        layers = []
        for i in range(n_downsample):
            out_ch = min(hidden_dim * (2 ** i), hidden_dim * 4)
            layers.append(Downsample(ch, out_ch))
            layers.append(nn.SiLU())
            for _ in range(n_res_blocks):
                layers.append(ResBlock(out_ch))
            ch = out_ch

        layers.append(nn.GroupNorm(8, ch))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(ch, out_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_conv(x)
        return self.net(h)


class Decoder(nn.Module):
    """Multi-stage convolutional decoder with progressive upsampling."""

    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 128,
        out_channels: int = 3,
        n_res_blocks: int = 3,
        n_upsample: int = 3,
    ):
        super().__init__()
        # Match encoder's deepest channel count
        init_ch = min(hidden_dim * (2 ** (n_upsample - 1)), hidden_dim * 4)
        self.input_conv = nn.Conv2d(in_dim, init_ch, 1)

        ch = init_ch
        layers = []
        for i in range(n_upsample):
            out_ch = max(hidden_dim // (2 ** i), hidden_dim // 2)
            for _ in range(n_res_blocks):
                layers.append(ResBlock(ch))
            layers.append(nn.SiLU())
            layers.append(Upsample(ch, out_ch))
            ch = out_ch

        layers.append(nn.GroupNorm(8, ch))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(ch, out_channels, 3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_conv(x)
        return self.net(h)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for adversarial VQGAN training."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        ]

        ch = hidden_dim
        for i in range(1, n_layers):
            out_ch = min(ch * 2, hidden_dim * 8)
            stride = 2 if i < n_layers - 1 else 1
            layers.extend([
                nn.Conv2d(ch, out_ch, 4, stride=stride, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.LeakyReLU(0.2),
            ])
            ch = out_ch

        layers.append(nn.Conv2d(ch, 1, 4, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LGQVAE(nn.Module):
    """Complete VQGAN with pluggable quantization.

    Supports LGQ, FSQ, and SimVQ quantizers through config.quantizer_type.
    """

    def __init__(self, config: LGQConfig):
        super().__init__()
        self.config = config

        # Number of downsample stages (log2 of downsample factor)
        n_stages = 0
        f = config.downsample_factor
        while f > 1:
            f //= 2
            n_stages += 1

        self.encoder = Encoder(
            in_channels=config.in_channels,
            hidden_dim=config.hidden_dim,
            out_dim=config.vq_dim,
            n_res_blocks=config.n_res_blocks,
            n_downsample=n_stages,
        )

        self.quantizer = build_quantizer(config)

        self.decoder = Decoder(
            in_dim=config.vq_dim,
            hidden_dim=config.hidden_dim,
            out_channels=config.in_channels,
            n_res_blocks=config.n_res_blocks,
            n_upsample=n_stages,
        )

        self.discriminator = PatchDiscriminator(
            in_channels=config.in_channels,
            hidden_dim=config.disc_hidden_dim,
            n_layers=config.disc_n_layers,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to continuous latent."""
        return self.encoder(x)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent to images."""
        return self.decoder(z_q)

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Full encode-quantize-decode pass.

        Args:
            x: [B, C, H, W] input images

        Returns dict with:
            recon: [B, C, H, W] reconstructed images
            All quantizer outputs (indices, losses, etc.)
        """
        z = self.encoder(x)
        quant_out = self.quantizer(z)
        recon = self.decoder(quant_out["quantized"])

        result = {"recon": recon, "z_e": z}
        result.update(quant_out)
        return result

    def forward_discriminator(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Discriminator forward pass.

        Args:
            real: [B, C, H, W] real images
            fake: [B, C, H, W] generated/reconstructed images (detached)

        Returns:
            d_loss: discriminator loss
            d_real: real logits
            d_fake: fake logits
        """
        d_real = self.discriminator(real)
        d_fake = self.discriminator(fake.detach())

        # Hinge loss
        loss_real = F.relu(1.0 - d_real).mean()
        loss_fake = F.relu(1.0 + d_fake).mean()
        d_loss = 0.5 * (loss_real + loss_fake)

        return {
            "d_loss": d_loss,
            "d_real": d_real.mean().detach(),
            "d_fake": d_fake.mean().detach(),
        }

    def generator_adversarial_loss(self, fake: torch.Tensor) -> torch.Tensor:
        """Generator's adversarial loss (non-saturating)."""
        d_fake = self.discriminator(fake)
        return -d_fake.mean()

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Separate parameter groups for generator and discriminator."""
        gen_params = (
            list(self.encoder.parameters())
            + list(self.quantizer.parameters())
            + list(self.decoder.parameters())
        )
        disc_params = list(self.discriminator.parameters())
        return {
            "generator": gen_params,
            "discriminator": disc_params,
        }

    def count_parameters(self) -> dict[str, int]:
        """Count parameters per component."""
        counts = {}
        for name, module in [
            ("encoder", self.encoder),
            ("quantizer", self.quantizer),
            ("decoder", self.decoder),
            ("discriminator", self.discriminator),
        ]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            counts[name] = {"total": total, "trainable": trainable}

        counts["generator_total"] = sum(
            counts[k]["trainable"] for k in ("encoder", "quantizer", "decoder")
        )
        counts["total"] = sum(c["total"] for c in counts.values() if isinstance(c, dict))
        return counts
