"""Evaluation metrics for the Multimodal Latent World Model.

Provides latent prediction quality, reconstruction quality, and multi-step
rollout metrics with TensorBoard integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .containers import LatentState, MemoryState, ModelOutput, ObservationPacket


# ============================================================
# Latent prediction metrics
# ============================================================


class LatentPredictionMetrics:
    """MSE, cosine similarity, and R-squared between predicted and target latents."""

    @staticmethod
    def compute(predicted: LatentState, target: LatentState) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for name, pred_z, tgt_z in [
            ("sem", predicted.z_sem, target.z_sem),
            ("dyn", predicted.z_dyn, target.z_dyn),
            ("ctrl", predicted.z_ctrl, target.z_ctrl),
            ("ctx", predicted.z_ctx, target.z_ctx),
        ]:
            if pred_z is None or tgt_z is None:
                continue
            mse = F.mse_loss(pred_z, tgt_z).item()
            cosine = F.cosine_similarity(pred_z, tgt_z, dim=-1).mean().item()
            # R-squared: 1 - SS_res / SS_tot
            ss_res = (pred_z - tgt_z).pow(2).sum().item()
            ss_tot = (tgt_z - tgt_z.mean(dim=0, keepdim=True)).pow(2).sum().item()
            r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
            metrics[f"latent_{name}_mse"] = mse
            metrics[f"latent_{name}_cosine"] = cosine
            metrics[f"latent_{name}_r2"] = r2
        return metrics


# ============================================================
# Reconstruction metrics
# ============================================================


class ReconstructionMetrics:
    """PSNR and SSIM for images, perplexity for text."""

    @staticmethod
    def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
        mse = F.mse_loss(pred, target).item()
        if mse < 1e-10:
            return 100.0
        return 10.0 * math.log10(max_val ** 2 / mse)

    @staticmethod
    def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> float:
        """Simplified SSIM for evaluation (channel-averaged)."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = window_size // 2

        # Average pooling as window function
        channels = pred.shape[1]
        kernel = torch.ones(channels, 1, window_size, window_size, device=pred.device) / (window_size ** 2)

        mu_pred = F.conv2d(pred, kernel, padding=pad, groups=channels)
        mu_tgt = F.conv2d(target, kernel, padding=pad, groups=channels)
        mu_pred_sq = mu_pred.pow(2)
        mu_tgt_sq = mu_tgt.pow(2)
        mu_cross = mu_pred * mu_tgt

        sigma_pred_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu_pred_sq
        sigma_tgt_sq = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu_tgt_sq
        sigma_cross = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu_cross

        numerator = (2 * mu_cross + C1) * (2 * sigma_cross + C2)
        denominator = (mu_pred_sq + mu_tgt_sq + C1) * (sigma_pred_sq + sigma_tgt_sq + C2)
        ssim_map = numerator / denominator
        return ssim_map.mean().item()

    @staticmethod
    def text_perplexity(logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute perplexity from logits and target token ids."""
        if logits.ndim == 3:
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        else:
            ce = F.cross_entropy(logits, targets)
        return math.exp(min(ce.item(), 100.0))

    @staticmethod
    def compute(
        decoder_outputs: Dict[str, torch.Tensor],
        batch: Mapping[str, Any],
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Image metrics
        image_key = next((k for k in decoder_outputs if k.endswith("image_recon")), None)
        if image_key is not None and "image_target" in batch:
            pred = decoder_outputs[image_key]
            target = batch["image_target"]
            if isinstance(target, torch.Tensor):
                metrics["image_psnr"] = ReconstructionMetrics.psnr(pred, target)
                if pred.shape[-1] >= 7 and pred.shape[-2] >= 7:
                    metrics["image_ssim"] = ReconstructionMetrics.ssim(pred, target)
                metrics["image_mse"] = F.mse_loss(pred, target).item()

        # Text metrics
        text_key = next((k for k in decoder_outputs if k.endswith("text_logits")), None)
        if text_key is not None and "text_target" in batch:
            logits = decoder_outputs[text_key]
            targets = batch["text_target"]
            if isinstance(targets, torch.Tensor):
                metrics["text_perplexity"] = ReconstructionMetrics.text_perplexity(logits, targets)

        # Vector metrics
        vector_key = next((k for k in decoder_outputs if k.endswith("vector_recon")), None)
        if vector_key is not None and "vector_target" in batch:
            pred = decoder_outputs[vector_key]
            target = batch["vector_target"]
            if isinstance(target, torch.Tensor):
                metrics["vector_mse"] = F.mse_loss(pred, target).item()
                metrics["vector_cosine"] = F.cosine_similarity(pred, target, dim=-1).mean().item()

        # Audio metrics
        audio_key = next((k for k in decoder_outputs if k.endswith("audio_recon")), None)
        if audio_key is not None and "audio_target" in batch:
            pred = decoder_outputs[audio_key]
            target = batch["audio_target"]
            if isinstance(target, torch.Tensor):
                metrics["audio_mse"] = F.mse_loss(pred, target).item()

        return metrics


# ============================================================
# Rollout metrics
# ============================================================


class RolloutMetrics:
    """Multi-step prediction quality over autoregressive latent rollouts."""

    @staticmethod
    @torch.no_grad()
    def compute(
        model: nn.Module,
        obs_sequence: List[ObservationPacket],
        action_sequence: List[torch.Tensor],
        max_horizon: int = 10,
    ) -> Dict[str, float]:
        """Evaluate multi-step rollout accuracy.

        Given a sequence of observations and actions, encodes obs[0], then
        autoregressively predicts forward using actions, comparing against
        the ground-truth latents at each step.
        """
        if len(obs_sequence) < 2 or len(action_sequence) < 1:
            return {}

        horizon = min(max_horizon, len(obs_sequence) - 1, len(action_sequence))
        model.eval()

        current_latent = model.encode(obs_sequence[0])
        batch_size = current_latent.z_sem.shape[0]
        device = current_latent.z_sem.device
        memory_state = model.memory.init_state(batch_size=batch_size, device=device)

        step_mses: List[float] = []
        step_cosines: List[float] = []
        accumulated_error = 0.0

        for t in range(horizon):
            transition = model.transition(current_latent, action_sequence[t], memory_state)
            predicted = transition.next_latent
            memory_state = transition.next_memory

            target_latent = model.encode(obs_sequence[t + 1])
            mse = F.mse_loss(predicted.z_sem, target_latent.z_sem).item()
            cosine = F.cosine_similarity(predicted.z_sem, target_latent.z_sem, dim=-1).mean().item()
            step_mses.append(mse)
            step_cosines.append(cosine)
            accumulated_error += mse

            current_latent = predicted

        metrics: Dict[str, float] = {
            "rollout_horizon": float(horizon),
            "rollout_final_mse": step_mses[-1] if step_mses else 0.0,
            "rollout_mean_mse": sum(step_mses) / len(step_mses) if step_mses else 0.0,
            "rollout_accumulated_error": accumulated_error,
            "rollout_mean_cosine": sum(step_cosines) / len(step_cosines) if step_cosines else 0.0,
            "rollout_final_cosine": step_cosines[-1] if step_cosines else 0.0,
        }

        # Error growth rate: ratio of final to initial MSE
        if len(step_mses) >= 2 and step_mses[0] > 1e-8:
            metrics["rollout_error_growth_rate"] = step_mses[-1] / step_mses[0]

        return metrics


# ============================================================
# Evaluation suite
# ============================================================


@dataclass
class EvalResult:
    latent_metrics: Dict[str, float] = field(default_factory=dict)
    recon_metrics: Dict[str, float] = field(default_factory=dict)
    rollout_metrics: Dict[str, float] = field(default_factory=dict)

    def all_metrics(self) -> Dict[str, float]:
        merged = {}
        merged.update(self.latent_metrics)
        merged.update(self.recon_metrics)
        merged.update(self.rollout_metrics)
        return merged


class EvaluationSuite:
    """Runs all evaluation metrics and logs to TensorBoard."""

    def __init__(self, writer: Optional[SummaryWriter] = None) -> None:
        self.writer = writer

    @torch.no_grad()
    def evaluate_step(
        self,
        output: ModelOutput,
        batch: Mapping[str, Any],
    ) -> EvalResult:
        """Evaluate a single forward pass output."""
        result = EvalResult()

        if output.target_next_latent is not None:
            result.latent_metrics = LatentPredictionMetrics.compute(
                output.predicted_next_latent, output.target_next_latent,
            )

        result.recon_metrics = ReconstructionMetrics.compute(
            output.decoder_outputs, batch,
        )

        return result

    @torch.no_grad()
    def evaluate_rollout(
        self,
        model: nn.Module,
        obs_sequence: List[ObservationPacket],
        action_sequence: List[torch.Tensor],
        max_horizon: int = 10,
    ) -> EvalResult:
        """Evaluate multi-step rollout quality."""
        result = EvalResult()
        result.rollout_metrics = RolloutMetrics.compute(
            model, obs_sequence, action_sequence, max_horizon,
        )
        return result

    def log_to_tensorboard(self, result: EvalResult, step: int, split: str = "eval") -> None:
        if self.writer is None:
            return
        for name, value in result.all_metrics().items():
            self.writer.add_scalar(f"{split}/metrics/{name}", value, step)
