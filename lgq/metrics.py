"""Evaluation metrics for image tokenizer comparison.

Implements:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - LPIPS (Learned Perceptual Image Patch Similarity) - lightweight proxy
  - rFID (reconstruction FID) - feature-based
  - Codebook utilization metrics
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio.

    Args:
        pred: [B, C, H, W] predicted images in [0, max_val]
        target: [B, C, H, W] ground truth images
        max_val: maximum pixel value

    Returns:
        Scalar PSNR averaged over batch.
    """
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    psnr_val = 10.0 * torch.log10(max_val ** 2 / (mse + 1e-8))
    return psnr_val.mean()


def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords.pow(2) / (2 * sigma * sigma))
    return g / g.sum()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Structural Similarity Index.

    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
        window_size: Gaussian filter size
        C1, C2: stability constants

    Returns:
        Scalar SSIM averaged over batch.
    """
    channels = pred.shape[1]
    device = pred.device

    # Create separable Gaussian window
    kernel_1d = _gaussian_kernel_1d(window_size, 1.5, device)
    kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
    window = kernel_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)

    pad = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_target = F.conv2d(target, window, padding=pad, groups=channels)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_target_sq
    sigma_cross = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_cross

    ssim_map = (
        (2 * mu_cross + C1) * (2 * sigma_cross + C2)
    ) / (
        (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    )

    return ssim_map.mean()


class LPIPSProxy(nn.Module):
    """Lightweight LPIPS proxy using multi-scale feature distances.

    Uses a simple learned feature extractor rather than pretrained VGG
    to avoid external model dependencies.  For precise LPIPS, install
    the lpips package and use LPIPSOfficial.
    """

    def __init__(self, in_channels: int = 3, hidden: int = 64, n_layers: int = 4):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(n_layers):
            out_ch = hidden * (2 ** min(i, 2))
            layers.append(nn.Sequential(
                nn.Conv2d(ch, out_ch, 3, stride=2, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
            ))
            ch = out_ch
        self.layers = nn.ModuleList(layers)
        self.weights = nn.Parameter(torch.ones(n_layers) / n_layers)

    def extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        h = x
        for layer in self.layers:
            h = layer(h)
            features.append(h)
        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feats = self.extract_features(pred)
        tgt_feats = self.extract_features(target)

        w = F.softmax(self.weights, dim=0)
        loss = torch.tensor(0.0, device=pred.device)
        for i, (pf, tf) in enumerate(zip(pred_feats, tgt_feats)):
            diff = (pf - tf).pow(2).mean(dim=(1, 2, 3))
            loss = loss + w[i] * diff.mean()
        return loss


class InceptionFeatureExtractor(nn.Module):
    """Lightweight feature extractor for FID computation.

    Replaces Inception-v3 with a simple convolutional feature extractor
    for environments without pretrained model access.  Produces 2048-dim
    feature vectors like Inception, enabling standard FID formula.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 2048),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_fid(
    real_features: torch.Tensor, fake_features: torch.Tensor
) -> float:
    """Compute Frechet Inception Distance from pre-extracted features.

    Args:
        real_features: [N, D] features from real images
        fake_features: [M, D] features from generated/reconstructed images

    Returns:
        FID score (lower is better).
    """
    mu_r = real_features.mean(dim=0)
    mu_f = fake_features.mean(dim=0)

    sigma_r = torch.cov(real_features.T)
    sigma_f = torch.cov(fake_features.T)

    diff = mu_r - mu_f
    mean_term = diff @ diff

    # Matrix square root via eigendecomposition
    product = sigma_r @ sigma_f
    eigvals, eigvecs = torch.linalg.eigh(product)
    eigvals = eigvals.clamp(min=0)
    sqrt_product = eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T

    trace_term = sigma_r.trace() + sigma_f.trace() - 2.0 * sqrt_product.trace()

    fid = (mean_term + trace_term).item()
    return max(0.0, fid)


class CodebookMetrics:
    """Tracks codebook utilization over evaluation."""

    def __init__(self, n_codebooks: int, vocab_size: int):
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.reset()

    def reset(self) -> None:
        self.code_counts = torch.zeros(self.n_codebooks, self.vocab_size)
        self.total_tokens = 0

    @torch.no_grad()
    def update(self, indices: torch.Tensor) -> None:
        """Update counts from a batch of indices.

        Args:
            indices: [B, n_codebooks, H, W]
        """
        for i in range(self.n_codebooks):
            idx = indices[:, i].reshape(-1)
            counts = torch.bincount(idx.cpu(), minlength=self.vocab_size)
            self.code_counts[i] += counts.float()
        self.total_tokens += indices[:, 0].numel()

    def compute(self) -> dict[str, float]:
        """Compute utilization metrics."""
        results = {}
        total_active = 0
        total_perplexity = 0.0
        total_entropy = 0.0

        for i in range(self.n_codebooks):
            counts = self.code_counts[i]
            total = counts.sum()
            if total < 1.0:
                continue
            probs = counts / total
            active = (counts > 0).float().sum().item()
            total_active += active

            log_probs = (probs + 1e-8).log()
            entropy = -(probs * log_probs).sum()
            total_perplexity += entropy.exp().item()
            total_entropy += entropy.item()

        n = max(1, self.n_codebooks)
        results["active_codes"] = total_active / n
        results["active_ratio"] = total_active / (n * self.vocab_size)
        results["perplexity"] = total_perplexity / n
        results["mean_entropy"] = total_entropy / n
        results["max_entropy"] = math.log(self.vocab_size)
        results["utilization_pct"] = (
            (total_active / n) / self.vocab_size * 100.0
        )
        return results


class MetricAggregator:
    """Aggregates evaluation metrics over a dataset."""

    def __init__(self, n_codebooks: int = 8, vocab_size: int = 256):
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.n_batches = 0
        self.codebook_metrics = CodebookMetrics(n_codebooks, vocab_size)
        self.real_features: list[torch.Tensor] = []
        self.fake_features: list[torch.Tensor] = []

    def reset(self) -> None:
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.n_batches = 0
        self.codebook_metrics.reset()
        self.real_features.clear()
        self.fake_features.clear()

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        indices: torch.Tensor,
        real_feats: torch.Tensor | None = None,
        fake_feats: torch.Tensor | None = None,
    ) -> None:
        # Clamp to valid range for metric computation
        pred_clamped = pred.clamp(0, 1)
        target_clamped = target.clamp(0, 1)

        self.psnr_sum += psnr(pred_clamped, target_clamped).item()
        self.ssim_sum += ssim(pred_clamped, target_clamped).item()
        self.n_batches += 1
        self.codebook_metrics.update(indices)

        if real_feats is not None:
            self.real_features.append(real_feats.cpu())
        if fake_feats is not None:
            self.fake_features.append(fake_feats.cpu())

    def compute(self) -> dict[str, float]:
        n = max(1, self.n_batches)
        results = {
            "psnr": self.psnr_sum / n,
            "ssim": self.ssim_sum / n,
        }
        results.update(self.codebook_metrics.compute())

        if self.real_features and self.fake_features:
            real = torch.cat(self.real_features, dim=0)
            fake = torch.cat(self.fake_features, dim=0)
            results["rfid"] = compute_fid(real, fake)

        return results
