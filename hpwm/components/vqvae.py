"""
Component 7: VQ-VAE Discrete Tokenizer.

Phases 1-3: VQ-GAN discrete tokenization with next-token prediction.
Phase -1: Tiny VQ-VAE with 8 codebooks for frame tokenization.

Encodes video frames into discrete tokens for the prediction objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResBlock(nn.Module):
    """Simple residual block for encoder/decoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VectorQuantizer(nn.Module):
    """
    Multi-codebook vector quantization with EMA updates.

    Each spatial position is quantized independently across codebooks.
    Final embedding = sum of codebook embeddings (residual quantization).
    """

    def __init__(
        self,
        n_codebooks: int = 8,
        vocab_size: int = 256,
        dim: int = 64,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.dim = dim
        self.dim_per_codebook = dim // n_codebooks

        # Each codebook: [vocab_size, dim_per_codebook]
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(vocab_size, self.dim_per_codebook) * 0.02)
            for _ in range(n_codebooks)
        ])

        # EMA tracking
        self.register_buffer("ema_count", torch.zeros(n_codebooks, vocab_size))
        self.register_buffer(
            "ema_weight",
            torch.randn(n_codebooks, vocab_size, self.dim_per_codebook) * 0.02,
        )
        self.ema_decay = ema_decay

    def quantize_single(
        self, z: torch.Tensor, codebook_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize using a single codebook.

        Args:
            z: [B*H*W, dim_per_codebook]
            codebook_idx: which codebook to use

        Returns:
            quantized, indices, commitment_loss
        """
        codebook = self.codebooks[codebook_idx]  # [V, D_cb]

        # L2 distances
        dists = (
            z.pow(2).sum(-1, keepdim=True)
            - 2 * z @ codebook.t()
            + codebook.pow(2).sum(-1, keepdim=True).t()
        )

        indices = dists.argmin(dim=-1)  # [B*H*W]
        quantized = codebook[indices]   # [B*H*W, D_cb]

        # EMA update (training only)
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.vocab_size).float()
                self.ema_count[codebook_idx] = (
                    self.ema_decay * self.ema_count[codebook_idx]
                    + (1 - self.ema_decay) * one_hot.sum(0)
                )
                self.ema_weight[codebook_idx] = (
                    self.ema_decay * self.ema_weight[codebook_idx]
                    + (1 - self.ema_decay) * (one_hot.t() @ z)
                )
                n = self.ema_count[codebook_idx].sum()
                count = (
                    (self.ema_count[codebook_idx] + 1e-5)
                    / (n + self.vocab_size * 1e-5)
                    * n
                )
                self.codebooks[codebook_idx].data = (
                    self.ema_weight[codebook_idx] / count.unsqueeze(-1)
                )

        # Commitment loss
        commitment_loss = F.mse_loss(z, quantized.detach())

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized, indices, commitment_loss

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, D, H, W] continuous features

        Returns:
            quantized: [B, D, H, W] quantized features
            indices: [B, n_codebooks, H, W] codebook indices
            commitment_loss: scalar
        """
        B, D, H, W = z.shape
        z_flat = rearrange(z, "b d h w -> (b h w) d")

        # Split across codebooks
        z_splits = z_flat.chunk(self.n_codebooks, dim=-1)

        quantized_parts = []
        all_indices = []
        total_commitment = 0.0

        for i, z_part in enumerate(z_splits):
            q, idx, c_loss = self.quantize_single(z_part, i)
            quantized_parts.append(q)
            all_indices.append(idx.view(B, H, W))
            total_commitment = total_commitment + c_loss

        quantized = torch.cat(quantized_parts, dim=-1)  # [B*H*W, D]
        quantized = rearrange(quantized, "(b h w) d -> b d h w", b=B, h=H, w=W)

        indices = torch.stack(all_indices, dim=1)  # [B, n_codebooks, H, W]
        total_commitment = total_commitment / self.n_codebooks

        return quantized, indices, total_commitment

    def codebook_utilization(self) -> dict[str, float]:
        """Compute per-codebook utilization metrics.

        Returns dict with:
            active_ratio: fraction of codes used (averaged over codebooks)
            perplexity: exp(entropy) of code usage (averaged)
        """
        with torch.no_grad():
            total_active = 0.0
            total_perplexity = 0.0
            for i in range(self.n_codebooks):
                counts = self.ema_count[i]
                total = counts.sum()
                if total < 1e-8:
                    continue
                probs = counts / total
                active = (counts > 1.0).float().mean().item()
                # Shannon entropy → perplexity
                log_probs = torch.log(probs + 1e-8)
                entropy = -(probs * log_probs).sum()
                perplexity = entropy.exp().item()
                total_active += active
                total_perplexity += perplexity
            n = max(1, self.n_codebooks)
            return {
                "active_ratio": total_active / n,
                "perplexity": total_perplexity / n,
            }

    def indices_to_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert codebook indices back to embeddings.

        Args:
            indices: [B, n_codebooks, H, W]

        Returns:
            embeddings: [B, D, H, W]
        """
        B, _, H, W = indices.shape
        parts = []
        for i in range(self.n_codebooks):
            idx = indices[:, i].reshape(-1)  # [B*H*W]
            emb = self.codebooks[i][idx]     # [B*H*W, D_cb]
            parts.append(emb)

        combined = torch.cat(parts, dim=-1)  # [B*H*W, D]
        return rearrange(combined, "(b h w) d -> b d h w", b=B, h=H, w=W)


class VQVAEEncoder(nn.Module):
    """Downsamples frames to latent grid and projects to VQ dim."""

    def __init__(self, in_channels: int = 3, hidden: int = 128,
                 out_dim: int = 64, n_layers: int = 3):
        super().__init__()

        layers = [
            # 128x128 -> 64x64
            nn.Conv2d(in_channels, hidden // 2, 4, stride=2, padding=1),
            nn.SiLU(),
            # 64x64 -> 32x32
            nn.Conv2d(hidden // 2, hidden, 4, stride=2, padding=1),
            nn.SiLU(),
            # 32x32 -> 16x16
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1),
            nn.SiLU(),
        ]

        for _ in range(n_layers):
            layers.append(ResBlock(hidden))

        layers.append(nn.Conv2d(hidden, out_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            [B, out_dim, H//8, W//8]
        """
        return self.net(x)


class VQVAEDecoder(nn.Module):
    """Upsamples from latent grid back to pixel space."""

    def __init__(self, in_dim: int = 64, hidden: int = 128,
                 out_channels: int = 3, n_layers: int = 3):
        super().__init__()

        layers = [nn.Conv2d(in_dim, hidden, 1)]

        for _ in range(n_layers):
            layers.append(ResBlock(hidden))

        layers.extend([
            nn.SiLU(),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1),
            nn.SiLU(),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(hidden, hidden // 2, 4, stride=2, padding=1),
            nn.SiLU(),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(hidden // 2, out_channels, 4, stride=2, padding=1),
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_dim, H//8, W//8]
        Returns:
            [B, 3, H, W]
        """
        return self.net(x)


class VQVAE(nn.Module):
    """
    Complete VQ-VAE for frame tokenization.

    Encodes frames to discrete tokens (for prediction objective)
    and decodes back (for reconstruction loss during training).
    """

    def __init__(
        self,
        n_codebooks: int = 8,
        vocab_size: int = 256,
        vq_dim: int = 64,
        hidden: int = 128,
        n_layers: int = 3,
        resolution: int = 128,
    ):
        super().__init__()

        self.encoder = VQVAEEncoder(
            in_channels=3, hidden=hidden, out_dim=vq_dim, n_layers=n_layers,
        )
        self.quantizer = VectorQuantizer(
            n_codebooks=n_codebooks, vocab_size=vocab_size, dim=vq_dim,
        )
        self.decoder = VQVAEDecoder(
            in_dim=vq_dim, hidden=hidden, out_channels=3, n_layers=n_layers,
        )

        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.latent_h = resolution // 8
        self.latent_w = resolution // 8

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode frames to discrete tokens.

        Args:
            x: [B, 3, H, W]

        Returns:
            indices: [B, n_codebooks, H', W'] codebook indices
            z_q: [B, D, H', W'] quantized features
        """
        z = self.encoder(x)
        z_q, indices, _ = self.quantizer(z)
        return indices, z_q

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized features to frames.

        Args:
            z_q: [B, D, H', W']

        Returns:
            [B, 3, H, W] reconstructed frames
        """
        return self.decoder(z_q)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encode-quantize-decode pass.

        Args:
            x: [B, 3, H, W]

        Returns:
            recon: [B, 3, H, W] reconstruction
            indices: [B, n_codebooks, H', W'] discrete tokens
            commitment_loss: scalar
            z_q: [B, D, H', W'] quantized features
        """
        z = self.encoder(x)
        z_q, indices, commitment_loss = self.quantizer(z)
        recon = self.decoder(z_q)
        return recon, indices, commitment_loss, z_q

    def indices_to_features(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert token indices to continuous features.

        Args:
            indices: [B, n_codebooks, H', W']

        Returns:
            [B, D, H', W']
        """
        return self.quantizer.indices_to_embeddings(indices)
