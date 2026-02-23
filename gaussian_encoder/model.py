"""Learnable Gaussian image encoder.

Convolutional filters parameterized as mixtures of oriented 2D Gaussians —
structured, interpretable kernels with fewer free parameters than standard
convolutions.  Train via autoencoder reconstruction to verify that the
Gaussian-parameterized filters learn meaningful features (edges, blobs, etc.).

Default architecture targets 28×28 greyscale images (MNIST).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core primitive: generate a conv kernel from learnable Gaussian parameters
# ---------------------------------------------------------------------------

class GaussianKernel(nn.Module):
    """Generates 2-D conv kernels from learnable Gaussian parameters.

    Each of the ``c_out * c_in`` filters is the sum of *n_gaussians* oriented
    Gaussian blobs, each controlled by six scalars:

        μx, μy      – centre position on the kernel grid
        log σx, log σy – log-scale (keeps σ > 0)
        θ           – rotation angle
        a           – amplitude / weight

    Total learnable params per filter: ``6 * n_gaussians``, compared with
    ``K * K`` for a standard conv (e.g. 18 vs 25 for K=5, n_gaussians=3).
    """

    def __init__(
        self, c_out: int, c_in: int, kernel_size: int, n_gaussians: int = 3,
    ) -> None:
        super().__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.K = kernel_size

        N = c_out * c_in  # total filters
        self.mu = nn.Parameter(torch.randn(N, n_gaussians, 2) * 0.5)
        self.log_sigma = nn.Parameter(torch.zeros(N, n_gaussians, 2))
        self.theta = nn.Parameter(torch.zeros(N, n_gaussians))
        self.amplitude = nn.Parameter(torch.randn(N, n_gaussians) * 0.1)

        # Fixed coordinate grid centred at 0
        half = kernel_size // 2
        ax = torch.arange(kernel_size, dtype=torch.float32) - half
        gy, gx = torch.meshgrid(ax, ax, indexing="ij")
        self.register_buffer(
            "grid", torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1),
        )  # (K*K, 2)

    def forward(self) -> torch.Tensor:
        """Return generated kernels with shape ``(c_out, c_in, K, K)``."""
        sigma = self.log_sigma.exp().clamp(min=0.1)  # (N, G, 2)
        cos_t, sin_t = self.theta.cos(), self.theta.sin()  # (N, G)

        # Offset from each Gaussian centre – (N, G, K*K, 2)
        delta = self.grid[None, None] - self.mu[:, :, None, :]
        dx, dy = delta[..., 0], delta[..., 1]

        # Rotate into the Gaussian's principal axes
        rx = dx * cos_t[..., None] + dy * sin_t[..., None]
        ry = -dx * sin_t[..., None] + dy * cos_t[..., None]

        # Evaluate Gaussian
        sx, sy = sigma[..., 0:1], sigma[..., 1:2]
        g = torch.exp(-0.5 * (rx**2 / sx**2 + ry**2 / sy**2))

        # Weighted sum over Gaussians → (N, K*K) → (c_out, c_in, K, K)
        return (g * self.amplitude[..., None]).sum(1).view(
            self.c_out, self.c_in, self.K, self.K,
        )


# ---------------------------------------------------------------------------
# Conv layer that uses Gaussian kernels
# ---------------------------------------------------------------------------

class GaussianConv2d(nn.Module):
    """Drop-in ``Conv2d`` replacement with Gaussian-parameterized kernels."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 5,
        n_gaussians: int = 3,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.pad = kernel_size // 2
        self.kernel = GaussianKernel(c_out, c_in, kernel_size, n_gaussians)
        self.bias = nn.Parameter(torch.zeros(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel(), self.bias, self.stride, self.pad)


# ---------------------------------------------------------------------------
# Autoencoder (encoder uses Gaussian convs; decoder is plain transposed conv)
# ---------------------------------------------------------------------------

class GaussianAutoencoder(nn.Module):
    """Tiny autoencoder with a Gaussian-kernel encoder.

    Architecture (for 28×28 input, default channels ``(16, 32)``):

    Encoder
        GaussianConv2d 1→16  stride 2  (28→14)
        GaussianConv2d 16→32 stride 2  (14→7)
        Flatten → Linear → latent_dim

    Decoder
        Linear → 7×7 feature map
        ConvTranspose2d 32→16 stride 2 (7→14)
        ConvTranspose2d 16→1  stride 2 (14→28)
    """

    def __init__(
        self,
        in_ch: int = 1,
        latent_dim: int = 32,
        channels: tuple[int, ...] = (16, 32),
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # --- Encoder (Gaussian convolutions) ---
        enc: list[nn.Module] = []
        c = in_ch
        for c_out in channels:
            enc += [GaussianConv2d(c, c_out, kernel_size=5, stride=2), nn.ReLU()]
            c = c_out
        self.enc_conv = nn.Sequential(*enc)
        # 28→14→7 with two stride-2 layers
        self.enc_fc = nn.Linear(channels[-1] * 7 * 7, latent_dim)

        # --- Decoder (standard transposed convolutions) ---
        rev = list(reversed(channels))
        self.dec_fc = nn.Linear(latent_dim, rev[0] * 7 * 7)
        self._dec_ch0 = rev[0]
        dec: list[nn.Module] = []
        for i in range(len(rev) - 1):
            dec += [nn.ConvTranspose2d(rev[i], rev[i + 1], 4, 2, 1), nn.ReLU()]
        dec += [nn.ConvTranspose2d(rev[-1], in_ch, 4, 2, 1), nn.Sigmoid()]
        self.dec_conv = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_fc(self.enc_conv(x).flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z).view(-1, self._dec_ch0, 7, 7)
        return self.dec_conv(h)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z
