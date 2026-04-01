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


class WorldModelLoss(nn.Module):
    def __init__(self, weights: Optional[LossWeights] = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()

    @staticmethod
    def _mse(pred: Optional[torch.Tensor], target: Optional[torch.Tensor]) -> torch.Tensor:
        if pred is None or target is None:
            device = pred.device if pred is not None else target.device if target is not None else torch.device("cpu")
            return torch.zeros((), device=device)
        return F.mse_loss(pred, target)

    def forward(self, output: ModelOutput, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if output.target_next_latent is None:
            raise ValueError("WorldModelLoss requires target_next_latent. Provide obs_tp1 during forward().")

        losses: Dict[str, torch.Tensor] = {}
        losses["latent_sem_loss"] = self._mse(output.predicted_next_latent.z_sem, output.target_next_latent.z_sem)
        losses["latent_dyn_loss"] = self._mse(output.predicted_next_latent.z_dyn, output.target_next_latent.z_dyn)
        losses["latent_ctrl_loss"] = self._mse(output.predicted_next_latent.z_ctrl, output.target_next_latent.z_ctrl)
        losses["latent_mem_loss"] = self._mse(output.predicted_next_latent.z_mem, output.target_next_latent.z_mem)

        losses["regularizer_loss"] = output.aux["regularizer_total"]

        total = (
            self.weights.latent_sem * losses["latent_sem_loss"]
            + self.weights.latent_dyn * losses["latent_dyn_loss"]
            + self.weights.latent_ctrl * losses["latent_ctrl_loss"]
            + self.weights.latent_mem * losses["latent_mem_loss"]
            + self.weights.regularizer * losses["regularizer_loss"]
        )

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
            total = total + self.weights.text_ce * losses["text_ce_loss"]
        else:
            device = total.device
            losses["text_ce_loss"] = torch.zeros((), device=device)

        if "vector_target" in batch and any(key.endswith("vector_recon") for key in output.decoder_outputs):
            vector_pred = next(v for k, v in output.decoder_outputs.items() if k.endswith("vector_recon"))
            vector_target = batch["vector_target"]
            losses["vector_recon_loss"] = F.mse_loss(vector_pred, vector_target)
            total = total + self.weights.vector_recon * losses["vector_recon_loss"]
        else:
            device = total.device
            losses["vector_recon_loss"] = torch.zeros((), device=device)

        losses["total_loss"] = total
        return losses
