"""Loss computation for the world model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .containers import ModelOutput


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


class WorldModelLoss(nn.Module):
    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        learned_uncertainty: bool = False,
        regularizer_min_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.learned_uncertainty = learned_uncertainty
        self.regularizer_min_weight = regularizer_min_weight
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
            })

    @staticmethod
    def _mse(pred: Optional[torch.Tensor], target: Optional[torch.Tensor]) -> torch.Tensor:
        if pred is None or target is None:
            device = pred.device if pred is not None else target.device if target is not None else torch.device("cpu")
            return torch.zeros((), device=device)
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

        losses: Dict[str, torch.Tensor] = {}
        losses["latent_sem_loss"] = self._mse(output.predicted_next_latent.z_sem, output.target_next_latent.z_sem)
        losses["latent_dyn_loss"] = self._mse(output.predicted_next_latent.z_dyn, output.target_next_latent.z_dyn)
        losses["latent_ctrl_loss"] = self._mse(output.predicted_next_latent.z_ctrl, output.target_next_latent.z_ctrl)
        losses["latent_mem_loss"] = self._mse(output.predicted_next_latent.z_mem, output.target_next_latent.z_mem)

        losses["regularizer_loss"] = output.aux["regularizer_total"]

        base_device = losses["regularizer_loss"].device
        base_dtype = losses["regularizer_loss"].dtype

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
            # text_logits is [B, T, V] from causal decoder; text_target is [B, T]
            if text_logits.ndim == 3:
                losses["text_ce_loss"] = F.cross_entropy(
                    text_logits.reshape(-1, text_logits.size(-1)), text_target.reshape(-1)
                )
            else:
                losses["text_ce_loss"] = F.cross_entropy(text_logits, text_target)
        else:
            losses["text_ce_loss"] = torch.zeros((), device=base_device, dtype=base_dtype)

        if "vector_target" in batch and any(key.endswith("vector_recon") for key in output.decoder_outputs):
            vector_pred = next(v for k, v in output.decoder_outputs.items() if k.endswith("vector_recon"))
            vector_target = batch["vector_target"]
            losses["vector_recon_loss"] = F.mse_loss(vector_pred, vector_target)
        else:
            losses["vector_recon_loss"] = torch.zeros((), device=base_device, dtype=base_dtype)

        # Image reconstruction loss
        img_w = self._safe_multiplier(task_multipliers, "image_recon_loss", base_device, base_dtype)
        if "image_target" in batch and any(key.endswith("image_recon") for key in output.decoder_outputs):
            image_pred = next(v for k, v in output.decoder_outputs.items() if k.endswith("image_recon"))
            image_target = batch["image_target"]
            losses["image_recon_loss"] = F.mse_loss(image_pred, image_target)
        else:
            losses["image_recon_loss"] = torch.zeros((), device=base_device, dtype=base_dtype)

        # Audio reconstruction loss
        audio_w = self._safe_multiplier(task_multipliers, "audio_recon_loss", base_device, base_dtype)
        if "audio_target" in batch and any(key.endswith("audio_recon") for key in output.decoder_outputs):
            audio_pred = next(v for k, v in output.decoder_outputs.items() if k.endswith("audio_recon"))
            audio_target = batch["audio_target"]
            losses["audio_recon_loss"] = F.mse_loss(audio_pred, audio_target)
        else:
            losses["audio_recon_loss"] = torch.zeros((), device=base_device, dtype=base_dtype)

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
        }

        if self.learned_uncertainty:
            total = torch.zeros((), device=losses["regularizer_loss"].device, dtype=losses["regularizer_loss"].dtype)
            for name, task_loss in weighted_losses.items():
                log_var = self.log_vars[name]
                effective_weight = 0.5 * torch.exp(-log_var)
                # Clamp regularizer weight to prevent anti-collapse from being disabled
                if name == "regularizer_loss":
                    effective_weight = torch.clamp(effective_weight, min=self.regularizer_min_weight)
                total = total + effective_weight * task_loss + 0.5 * log_var
                losses[f"{name}_log_var"] = log_var.detach()
                if name == "regularizer_loss":
                    losses["regularizer_effective_weight"] = effective_weight.detach()
        else:
            total = sum(weighted_losses.values())

        losses["total_loss"] = total
        return losses
