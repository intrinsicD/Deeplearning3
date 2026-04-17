"""Loss computation for the world model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .containers import ModelOutput


class ContrastiveAlignmentLoss(nn.Module):
    """InfoNCE contrastive loss for cross-modal alignment."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """Compute symmetric InfoNCE between two batches of features [B, D].

        Returns zero for batch_size < 2 since InfoNCE needs negatives.
        """
        if feat_a.shape[0] < 2:
            return torch.zeros((), device=feat_a.device, dtype=feat_a.dtype)
        a = F.normalize(feat_a, dim=-1)
        b = F.normalize(feat_b, dim=-1)
        temperature = self.log_temperature.exp().clamp(min=0.01, max=1.0)
        logits = a @ b.T / temperature  # [B, B]
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        return loss


@dataclass
class LossWeights:
    latent_sem: float = 1.0
    latent_dyn: float = 1.0
    latent_ctrl: float = 0.25
    latent_mem: float = 0.25
    regularizer: float = 1.0
    text_ce: float = 1.0
    vector_recon: float = 0.0
    image_recon: float = 1.0
    audio_recon: float = 1.0
    contrastive_alignment: float = 1.0


class WorldModelLoss(nn.Module):
    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        learned_uncertainty: bool = False,
        regularizer_min_weight: float = 0.5,
        regularizer_max_weight: float = 100.0,
        text_pad_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.learned_uncertainty = learned_uncertainty
        self.regularizer_min_weight = regularizer_min_weight
        # Symmetric upper clamp prevents an active task with log_var << 0 from
        # producing an effective weight that overwhelms every other gradient.
        self.regularizer_max_weight = regularizer_max_weight
        # When set, padding tokens are excluded from the text cross-entropy.
        self.text_pad_token_id = text_pad_token_id
        self.contrastive_loss_fn = ContrastiveAlignmentLoss()
        if learned_uncertainty:
            self.log_vars = nn.ParameterDict({
                "latent_sem_loss": nn.Parameter(torch.zeros(())),
                "latent_dyn_loss": nn.Parameter(torch.zeros(())),
                "latent_ctrl_loss": nn.Parameter(torch.zeros(())),
                "latent_mem_loss": nn.Parameter(torch.zeros(())),
                "regularizer_loss": nn.Parameter(torch.zeros(())),
                "text_ce_loss": nn.Parameter(torch.zeros(())),
                "vector_recon_loss": nn.Parameter(torch.zeros(())),
                "image_recon_loss": nn.Parameter(torch.zeros(())),
                "audio_recon_loss": nn.Parameter(torch.zeros(())),
                "contrastive_alignment_loss": nn.Parameter(torch.zeros(())),
            })

    @staticmethod
    def _mse(
        pred: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        *,
        fallback_device: torch.device,
    ) -> torch.Tensor:
        # Both absent => role is genuinely missing for this batch; contribute zero.
        if pred is None and target is None:
            return torch.zeros((), device=fallback_device)
        # Mismatch is almost always a bug: a target exists but the head returned
        # None, or vice versa. Surfacing this loudly avoids silently zero loss.
        if pred is None or target is None:
            missing = "pred" if pred is None else "target"
            raise ValueError(
                f"_mse received only one of (pred, target); '{missing}' is None. "
                "This usually means a prediction head dropped a role that the "
                "target still has (or vice versa). Check RoleSplit configuration."
            )
        return F.mse_loss(pred, target)

    def _safe_multiplier(
        self,
        task_multipliers: Optional[Dict[str, float]],
        key: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if task_multipliers is None:
            return torch.ones((), device=device, dtype=dtype)
        return torch.tensor(float(task_multipliers.get(key, 1.0)), device=device, dtype=dtype)

    def forward(
        self,
        output: ModelOutput,
        batch: Dict[str, Any],
        task_multipliers: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        if output.target_next_latent is None:
            raise ValueError("WorldModelLoss requires target_next_latent. Provide obs_tp1 during forward().")

        base_device = output.aux["regularizer_total"].device
        base_dtype = output.aux["regularizer_total"].dtype

        # Track which losses correspond to a real signal this batch. Losses
        # without a signal are kept as zero placeholders for logging but are
        # excluded from the learned-uncertainty sum so their log_var doesn't
        # drift on noise.
        active: Dict[str, bool] = {}

        def _role_active(pred: Optional[torch.Tensor], target: Optional[torch.Tensor]) -> bool:
            return pred is not None and target is not None

        losses: Dict[str, torch.Tensor] = {}
        losses["latent_sem_loss"] = self._mse(
            output.predicted_next_latent.z_sem, output.target_next_latent.z_sem, fallback_device=base_device,
        )
        losses["latent_dyn_loss"] = self._mse(
            output.predicted_next_latent.z_dyn, output.target_next_latent.z_dyn, fallback_device=base_device,
        )
        losses["latent_ctrl_loss"] = self._mse(
            output.predicted_next_latent.z_ctrl, output.target_next_latent.z_ctrl, fallback_device=base_device,
        )
        losses["latent_mem_loss"] = self._mse(
            output.predicted_next_latent.z_mem, output.target_next_latent.z_mem, fallback_device=base_device,
        )
        active["latent_sem_loss"] = _role_active(output.predicted_next_latent.z_sem, output.target_next_latent.z_sem)
        active["latent_dyn_loss"] = _role_active(output.predicted_next_latent.z_dyn, output.target_next_latent.z_dyn)
        active["latent_ctrl_loss"] = _role_active(output.predicted_next_latent.z_ctrl, output.target_next_latent.z_ctrl)
        active["latent_mem_loss"] = _role_active(output.predicted_next_latent.z_mem, output.target_next_latent.z_mem)

        losses["regularizer_loss"] = output.aux["regularizer_total"]
        active["regularizer_loss"] = True

        sem_w = self._safe_multiplier(task_multipliers, "latent_sem_loss", losses["latent_sem_loss"].device, losses["latent_sem_loss"].dtype)
        dyn_w = self._safe_multiplier(task_multipliers, "latent_dyn_loss", losses["latent_dyn_loss"].device, losses["latent_dyn_loss"].dtype)
        ctrl_w = self._safe_multiplier(task_multipliers, "latent_ctrl_loss", losses["latent_ctrl_loss"].device, losses["latent_ctrl_loss"].dtype)
        mem_w = self._safe_multiplier(task_multipliers, "latent_mem_loss", losses["latent_mem_loss"].device, losses["latent_mem_loss"].dtype)
        reg_w = self._safe_multiplier(task_multipliers, "regularizer_loss", losses["regularizer_loss"].device, losses["regularizer_loss"].dtype)
        text_w = self._safe_multiplier(task_multipliers, "text_ce_loss", base_device, base_dtype)
        vec_w = self._safe_multiplier(task_multipliers, "vector_recon_loss", base_device, base_dtype)

        if "text_target" in batch and any(key.endswith("text_logits") for key in output.decoder_outputs):
            text_logits = next(v for k, v in output.decoder_outputs.items() if k.endswith("text_logits"))
            text_target = batch["text_target"]
            ce_kwargs: Dict[str, Any] = {}
            if self.text_pad_token_id is not None:
                ce_kwargs["ignore_index"] = self.text_pad_token_id
            # text_logits is [B, T, V] from causal decoder; text_target is [B, T]
            if text_logits.ndim == 3:
                if text_logits.size(1) != text_target.size(1):
                    raise ValueError(
                        f"text_logits/text_target seq lengths disagree: "
                        f"{text_logits.size(1)} vs {text_target.size(1)}"
                    )
                losses["text_ce_loss"] = F.cross_entropy(
                    text_logits.reshape(-1, text_logits.size(-1)), text_target.reshape(-1), **ce_kwargs,
                )
            else:
                losses["text_ce_loss"] = F.cross_entropy(text_logits, text_target, **ce_kwargs)
            active["text_ce_loss"] = True
        else:
            losses["text_ce_loss"] = torch.zeros((), device=base_device, dtype=base_dtype)
            active["text_ce_loss"] = False

        if "vector_target" in batch and any(key.endswith("vector_recon") for key in output.decoder_outputs):
            vector_pred = next(v for k, v in output.decoder_outputs.items() if k.endswith("vector_recon"))
            vector_target = batch["vector_target"]
            losses["vector_recon_loss"] = F.mse_loss(vector_pred, vector_target)
            active["vector_recon_loss"] = True
        else:
            losses["vector_recon_loss"] = torch.zeros((), device=base_device, dtype=base_dtype)
            active["vector_recon_loss"] = False

        # Image reconstruction loss
        img_w = self._safe_multiplier(task_multipliers, "image_recon_loss", base_device, base_dtype)
        if "image_target" in batch and any(key.endswith("image_recon") for key in output.decoder_outputs):
            image_pred = next(v for k, v in output.decoder_outputs.items() if k.endswith("image_recon"))
            image_target = batch["image_target"]
            losses["image_recon_loss"] = F.mse_loss(image_pred, image_target)
            active["image_recon_loss"] = True
        else:
            losses["image_recon_loss"] = torch.zeros((), device=base_device, dtype=base_dtype)
            active["image_recon_loss"] = False

        # Audio reconstruction loss
        audio_w = self._safe_multiplier(task_multipliers, "audio_recon_loss", base_device, base_dtype)
        if "audio_target" in batch and any(key.endswith("audio_recon") for key in output.decoder_outputs):
            audio_pred = next(v for k, v in output.decoder_outputs.items() if k.endswith("audio_recon"))
            audio_target = batch["audio_target"]
            losses["audio_recon_loss"] = F.mse_loss(audio_pred, audio_target)
            active["audio_recon_loss"] = True
        else:
            losses["audio_recon_loss"] = torch.zeros((), device=base_device, dtype=base_dtype)
            active["audio_recon_loss"] = False

        # Contrastive alignment loss across modalities
        contrastive_w = self._safe_multiplier(task_multipliers, "contrastive_alignment_loss", base_device, base_dtype)
        extras = output.current_latent.extras
        modality_feats = {k: v for k, v in extras.items()
                         if isinstance(v, torch.Tensor) and k.endswith("_feat") and v.ndim == 2}
        if len(modality_feats) >= 2:
            feat_names = sorted(modality_feats.keys())
            contrastive_total = torch.zeros((), device=base_device, dtype=base_dtype)
            n_pairs = 0
            for i in range(len(feat_names)):
                for j in range(i + 1, len(feat_names)):
                    contrastive_total = contrastive_total + self.contrastive_loss_fn(
                        modality_feats[feat_names[i]], modality_feats[feat_names[j]],
                    )
                    n_pairs += 1
            losses["contrastive_alignment_loss"] = contrastive_total / max(n_pairs, 1)
            active["contrastive_alignment_loss"] = True
        else:
            losses["contrastive_alignment_loss"] = torch.zeros((), device=base_device, dtype=base_dtype)
            active["contrastive_alignment_loss"] = False

        weighted_losses = {
            "latent_sem_loss": self.weights.latent_sem * sem_w * losses["latent_sem_loss"],
            "latent_dyn_loss": self.weights.latent_dyn * dyn_w * losses["latent_dyn_loss"],
            "latent_ctrl_loss": self.weights.latent_ctrl * ctrl_w * losses["latent_ctrl_loss"],
            "latent_mem_loss": self.weights.latent_mem * mem_w * losses["latent_mem_loss"],
            "regularizer_loss": self.weights.regularizer * reg_w * losses["regularizer_loss"],
            "text_ce_loss": self.weights.text_ce * text_w * losses["text_ce_loss"],
            "vector_recon_loss": self.weights.vector_recon * vec_w * losses["vector_recon_loss"],
            "image_recon_loss": self.weights.image_recon * img_w * losses["image_recon_loss"],
            "audio_recon_loss": self.weights.audio_recon * audio_w * losses["audio_recon_loss"],
            "contrastive_alignment_loss": self.weights.contrastive_alignment * contrastive_w * losses["contrastive_alignment_loss"],
        }

        if self.learned_uncertainty:
            total = torch.zeros((), device=losses["regularizer_loss"].device, dtype=losses["regularizer_loss"].dtype)
            for name, task_loss in weighted_losses.items():
                # Skip the uncertainty term entirely when this loss has no
                # signal in the batch. Otherwise the +0.5*log_var penalty
                # would still be applied, dragging log_var around on noise.
                if not active.get(name, True):
                    continue
                # Clamp log_var to [-6, 10] to prevent divergence when task_loss is zero
                log_var = self.log_vars[name].clamp(-6.0, 10.0)
                effective_weight = 0.5 * torch.exp(-log_var)
                # Symmetric clamp: floor for the regularizer (so anti-collapse
                # cannot be silently disabled) and a ceiling for every loss
                # (so a single task with log_var << 0 cannot dominate).
                lower = self.regularizer_min_weight if name == "regularizer_loss" else 0.0
                effective_weight = torch.clamp(effective_weight, min=lower, max=self.regularizer_max_weight)
                total = total + effective_weight * task_loss + 0.5 * log_var
                losses[f"{name}_log_var"] = log_var.detach()
                if name == "regularizer_loss":
                    losses["regularizer_effective_weight"] = effective_weight.detach()
        else:
            total = sum(weighted_losses.values())

        losses["total_loss"] = total
        return losses
