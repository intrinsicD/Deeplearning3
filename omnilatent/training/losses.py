"""Multi-modal loss functions.

Supports:
  * Per-modality reconstruction losses (CE for text, L1+spectral for audio,
    perceptual L1 for image/video)
  * Uncertainty-weighted multi-task loss (Kendall et al., 2018) -- learns
    the optimal weighting between modality losses automatically
  * Contrastive cross-modal alignment loss (InfoNCE)
  * Temporal context losses:
    - Next-clip prediction loss (MSE + cosine in latent space)
    - Temporal order classification (binary cross-entropy)
    - Temporal distance classification (cross-entropy over buckets)
    - Scene boundary detection (binary cross-entropy)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnilatent.config import OmniLatentConfig
from omnilatent.utils import ALL_MODALITIES


class ReconstructionLoss(nn.Module):
    """Modality-specific reconstruction losses."""

    def text_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy over vocabulary.

        logits:  (B, T_pred, vocab_size)
        targets: (B, T_tgt) long

        Handles mismatched sequence lengths (common in cross-modal
        decoding) by truncating to the shorter length.
        """
        B, T_pred, V = logits.shape
        T_tgt = targets.shape[1]
        T = min(T_pred, T_tgt)
        logits = logits[:, :T].contiguous()
        targets = targets[:, :T].contiguous()
        return F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=0,  # assume 0 = padding
        )

    def audio_loss(
        self, pred_mel: torch.Tensor, target_mel: torch.Tensor
    ) -> torch.Tensor:
        """L1 + MSE loss on mel spectrogram.

        Both: (B, n_mels, T)

        Uses straightforward reconstruction losses rather than spectral
        convergence (FFT on a mel spectrogram is a double frequency
        transform that doesn't add value).
        """
        # Trim to matching length
        min_t = min(pred_mel.shape[-1], target_mel.shape[-1])
        pred_mel = pred_mel[..., :min_t]
        target_mel = target_mel[..., :min_t]

        return F.l1_loss(pred_mel, target_mel) + F.mse_loss(pred_mel, target_mel)

    def image_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """L1 + perceptual-style frequency loss for images.

        Both: (B, C, H, W)
        """
        l1 = F.l1_loss(pred, target)

        # Simple frequency-domain loss (DCT-like via 2D FFT)
        # Pad spatial dims to next power of 2 for cuFFT compatibility
        h, w = pred.shape[-2], pred.shape[-1]
        fft_h = 1 << (h - 1).bit_length()
        fft_w = 1 << (w - 1).bit_length()
        pred_freq = torch.fft.rfft2(pred, s=(fft_h, fft_w))
        tgt_freq = torch.fft.rfft2(target, s=(fft_h, fft_w))
        freq_loss = F.l1_loss(pred_freq.abs(), tgt_freq.abs())

        return l1 + 0.1 * freq_loss

    def video_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """L1 + temporal consistency loss for video.

        Both: (B, C, T, H, W)
        """
        # Trim temporal dimension to match
        min_t = min(pred.shape[2], target.shape[2])
        pred = pred[:, :, :min_t]
        target = target[:, :, :min_t]

        l1 = F.l1_loss(pred, target)

        # Temporal consistency: penalize jitter between consecutive frames
        if min_t > 1:
            pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
            tgt_diff = target[:, :, 1:] - target[:, :, :-1]
            temporal = F.l1_loss(pred_diff, tgt_diff)
        else:
            temporal = torch.tensor(0.0, device=pred.device)

        return l1 + 0.2 * temporal

    def forward(
        self,
        modality: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if modality == "text":
            return self.text_loss(prediction, target)
        elif modality == "audio":
            return self.audio_loss(prediction, target)
        elif modality == "image":
            return self.image_loss(prediction, target)
        elif modality == "video":
            return self.video_loss(prediction, target)
        else:
            raise ValueError(f"Unknown modality: {modality}")


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for cross-modal alignment.

    Aligns the latent representations of different modalities describing
    the same content (e.g. an image and its caption).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, z_a: torch.Tensor, z_b: torch.Tensor
    ) -> torch.Tensor:
        """Bidirectional InfoNCE.

        z_a, z_b: (B, D) -- mean-pooled latent representations of two
        modalities for the same batch of paired samples.
        """
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        logits = z_a @ z_b.T / self.temperature  # (B, B)
        labels = torch.arange(z_a.shape[0], device=z_a.device)

        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)
        return (loss_ab + loss_ba) / 2


# =========================================================================
# Temporal Context Losses
# =========================================================================

class TemporalOrderLoss(nn.Module):
    """Binary classification: did clip A come before clip B?

    Used with Approach 1 (multi-scale temporal sampling) to teach the
    model about temporal ordering without any model changes.
    """

    def forward(
        self, z_anchor: torch.Tensor, z_context: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute temporal order loss.

        Args:
            z_anchor: (B, D) anchor clip latent.
            z_context: (B, D) context clip latent.
            labels: (B,) long — 1 if anchor before context, 0 otherwise.

        Returns scalar loss.
        """
        # Concatenate and project to logit
        combined = torch.cat([z_anchor, z_context, z_anchor - z_context], dim=-1)
        # Use a simple bilinear-like scoring: dot product of difference
        logits = (z_anchor * z_context).sum(dim=-1)  # (B,)
        return F.binary_cross_entropy_with_logits(
            logits, labels.float()
        )


class TemporalDistanceLoss(nn.Module):
    """Classify temporal distance between two clips into buckets.

    Buckets: <10s, 10-60s, 1-5min, >5min (4 classes by default).
    """

    def __init__(self, num_buckets: int = 4, hidden_dim: int = 768) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_buckets),
        )

    def forward(
        self, z_anchor: torch.Tensor, z_context: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute temporal distance classification loss.

        Args:
            z_anchor: (B, D) anchor clip latent.
            z_context: (B, D) context clip latent.
            labels: (B,) long — bucket index.

        Returns scalar loss.
        """
        combined = torch.cat([z_anchor, z_context, z_anchor - z_context], dim=-1)
        logits = self.classifier(combined)  # (B, num_buckets)
        return F.cross_entropy(logits, labels)


class NextClipPredictionLoss(nn.Module):
    """Loss for next-clip prediction in latent space.

    Combines MSE (magnitude) with cosine similarity (direction) for
    robust latent-space prediction.

    Used with Approach 2 (hierarchical temporal transformer).
    """

    def __init__(self, cosine_weight: float = 0.5) -> None:
        super().__init__()
        self.cosine_weight = cosine_weight

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute next-clip prediction loss.

        Args:
            predicted: (B, N, D) predicted next-clip latents.
            target: (B, N, D) actual next-clip latents.
            mask: (B, N) bool — True for valid positions.

        Returns scalar loss.
        """
        mse = F.mse_loss(predicted, target, reduction="none")  # (B, N, D)
        mse = mse.mean(dim=-1)  # (B, N)

        # Cosine similarity loss (1 - cos_sim)
        cos_sim = F.cosine_similarity(predicted, target, dim=-1)  # (B, N)
        cos_loss = 1.0 - cos_sim

        combined = mse + self.cosine_weight * cos_loss  # (B, N)

        if mask is not None:
            combined = combined * mask.float()
            return combined.sum() / mask.float().sum().clamp(min=1.0)

        return combined.mean()


class SceneBoundaryLoss(nn.Module):
    """Binary cross-entropy for scene boundary detection.

    Used with Approach 2: the temporal transformer predicts whether
    each clip position is a scene boundary.
    """

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute scene boundary detection loss.

        Args:
            logits: (B, N, 1) or (B, N) scene boundary logits.
            labels: (B, N) float — 1.0 for scene boundary, 0.0 otherwise.
            mask: (B, N) bool — True for valid positions.

        Returns scalar loss.
        """
        if logits.dim() == 3:
            logits = logits.squeeze(-1)

        loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )  # (B, N)

        if mask is not None:
            loss = loss * mask.float()
            return loss.sum() / mask.float().sum().clamp(min=1.0)

        return loss.mean()


class TemporalContextLoss(nn.Module):
    """Combined loss for all temporal context tasks.

    Handles losses for:
      - Approach 1: temporal_order, temporal_distance, distant_predict
      - Approach 2: next_clip_prediction, scene_boundary
      - Approach 3: memory-augmented reconstruction (uses standard recon loss)

    Each component is weighted and the total is returned.
    """

    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        self.order_loss = TemporalOrderLoss()
        self.distance_loss = TemporalDistanceLoss(
            num_buckets=config.temporal_distance_buckets,
            hidden_dim=config.hidden_dim,
        )
        self.next_clip_loss = NextClipPredictionLoss()
        self.scene_boundary_loss = SceneBoundaryLoss()

        # Learnable weights for each temporal task
        self.log_vars = nn.ParameterDict({
            "temporal_order": nn.Parameter(torch.zeros(1)),
            "temporal_distance": nn.Parameter(torch.zeros(1)),
            "next_clip": nn.Parameter(torch.zeros(1)),
            "scene_boundary": nn.Parameter(torch.zeros(1)),
        })

    def forward(
        self,
        losses_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Combine temporal losses with uncertainty weighting.

        Args:
            losses_dict: dict mapping task names to raw loss tensors.
                Recognized keys: temporal_order, temporal_distance,
                next_clip, scene_boundary, distant_predict.

        Returns dict with individual losses and "temporal_total".
        """
        result: dict[str, torch.Tensor] = {}
        device = next(iter(losses_dict.values())).device
        total = torch.tensor(0.0, device=device)

        for name, raw_loss in losses_dict.items():
            result[name] = raw_loss
            if name in self.log_vars:
                log_var = self.log_vars[name]
                weighted = torch.exp(-log_var) * raw_loss + log_var
                result[f"{name}_weighted"] = weighted
                total = total + weighted
            else:
                # Direct addition for tasks without learned weights
                total = total + raw_loss

        result["temporal_total"] = total
        return result


class MultiModalLoss(nn.Module):
    """Combined multi-modal loss with learned uncertainty weighting.

    Uses the homoscedastic uncertainty approach from Kendall et al. (2018):
    each modality loss is weighted by exp(-log_var) and a regularization
    term log_var is added.  This learns the optimal task weighting
    automatically.
    """

    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        self.recon_loss = ReconstructionLoss()
        self.contrastive_loss = ContrastiveLoss(config.contrastive_temperature)
        self.contrastive_weight = config.contrastive_weight

        # Learnable log-variance per modality (uncertainty weighting)
        self.log_vars = nn.ParameterDict({
            mod: nn.Parameter(torch.zeros(1))
            for mod in ALL_MODALITIES
        })

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        latents: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute total loss.

        Args:
            predictions: modality_name → decoder output
            targets: modality_name → ground truth
            latents: modality_name → mean-pooled latent (for contrastive)

        Returns dict with "total", per-modality losses, and "contrastive".
        """
        losses: dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        # Reconstruction losses with uncertainty weighting
        for mod in predictions:
            if mod not in targets:
                continue
            raw_loss = self.recon_loss(mod, predictions[mod], targets[mod])
            log_var = self.log_vars[mod]
            # precision-weighted loss + regularization
            weighted = torch.exp(-log_var) * raw_loss + log_var
            losses[mod] = raw_loss
            losses[f"{mod}_weighted"] = weighted
            total = total + weighted

        # Contrastive alignment loss
        if latents is not None and len(latents) >= 2:
            modality_names = list(latents.keys())
            contrastive_total = torch.tensor(0.0, device=total.device)
            n_pairs = 0
            for i in range(len(modality_names)):
                for j in range(i + 1, len(modality_names)):
                    c_loss = self.contrastive_loss(
                        latents[modality_names[i]],
                        latents[modality_names[j]],
                    )
                    contrastive_total = contrastive_total + c_loss
                    n_pairs += 1
            if n_pairs > 0:
                contrastive_total = contrastive_total / n_pairs
            losses["contrastive"] = contrastive_total
            total = total + self.contrastive_weight * contrastive_total

        losses["total"] = total
        return losses
