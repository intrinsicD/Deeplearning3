"""Training loop with TensorBoard logging and autoregressive generation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .containers import MemoryState, ObservationPacket
from .losses import WorldModelLoss
from .model import ModularLatentWorldModel
from .monitoring import HookManager


class Trainer:
    def __init__(
        self,
        model: ModularLatentWorldModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: WorldModelLoss,
        device: torch.device,
        run_dir: str = "runs/modular_world_model",
        grad_clip_norm: Optional[float] = 1.0,
        mixed_precision: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and device.type == "cuda")
        self.writer = SummaryWriter(log_dir=run_dir)
        self.hooks = HookManager(self.writer)
        self.global_step = 0

    def _to_packet(self, batch: Mapping[str, Any], suffix: str) -> ObservationPacket:
        modalities: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}
        for name in ["text", "vector", "image"]:
            key = f"{name}_{suffix}"
            if key in batch:
                modalities[name] = batch[key].to(self.device)
            mask_key = f"{name}_mask_{suffix}"
            if mask_key in batch:
                masks[name] = batch[mask_key].to(self.device)
        return ObservationPacket(modalities=modalities, masks=masks, meta={})

    def train_step(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        self.model.train()
        obs_t = self._to_packet(batch, "t")
        obs_tp1 = self._to_packet(batch, "tp1")
        action = batch["action"].to(self.device)

        decoder_context: Dict[str, Any] = {}
        if "prefix_tokens" in batch:
            decoder_context["prefix_tokens"] = batch["prefix_tokens"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
            output = self.model(obs_t, action, obs_tp1=obs_tp1, memory_state=None, decoder_context=decoder_context)
            loss_inputs: Dict[str, Any] = {}
            if "text_target" in batch:
                loss_inputs["text_target"] = batch["text_target"].to(self.device)
            if "vector_target" in batch:
                loss_inputs["vector_target"] = batch["vector_target"].to(self.device)
            losses = self.loss_fn(output, loss_inputs)
            total_loss = losses["total_loss"]

        self.scaler.scale(total_loss).backward()
        if self.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.hooks.log_losses(losses, self.global_step, split="train")
        self.hooks.log_aux(output.aux, self.global_step, split="train")
        self.hooks.log_depth_metrics(output.aux, self.global_step, split="train")
        self.hooks.log_latents(output, self.global_step, split="train")
        self.hooks.log_gradient_norms(self.model, self.global_step)
        self.hooks.log_learning_rate(self.optimizer, self.global_step)
        if "text_target" in batch:
            self.hooks.log_text_predictions(output, {"text_target": batch["text_target"].to(self.device)}, self.global_step, split="train")
        self.hooks.log_embeddings(output, self.global_step, split="train")

        result = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
        self.global_step += 1
        return result

    @torch.no_grad()
    def eval_step(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        self.model.eval()
        obs_t = self._to_packet(batch, "t")
        obs_tp1 = self._to_packet(batch, "tp1")
        action = batch["action"].to(self.device)

        decoder_context: Dict[str, Any] = {}
        if "prefix_tokens" in batch:
            decoder_context["prefix_tokens"] = batch["prefix_tokens"].to(self.device)

        with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
            output = self.model(obs_t, action, obs_tp1=obs_tp1, memory_state=None, decoder_context=decoder_context)
            loss_inputs: Dict[str, Any] = {}
            if "text_target" in batch:
                loss_inputs["text_target"] = batch["text_target"].to(self.device)
            if "vector_target" in batch:
                loss_inputs["vector_target"] = batch["vector_target"].to(self.device)
            losses = self.loss_fn(output, loss_inputs)

        self.hooks.log_losses(losses, self.global_step, split="eval")
        self.hooks.log_aux(output.aux, self.global_step, split="eval")
        self.hooks.log_depth_metrics(output.aux, self.global_step, split="eval")
        self.hooks.log_latents(output, self.global_step, split="eval")
        if "text_target" in batch:
            self.hooks.log_text_predictions(output, {"text_target": batch["text_target"].to(self.device)}, self.global_step, split="eval")
        self.hooks.log_embeddings(output, self.global_step, split="eval")

        return {k: float(v.detach().cpu().item()) for k, v in losses.items()}

    def fit(self, train_loader: DataLoader, eval_loader: Optional[DataLoader] = None, epochs: int = 1, eval_every_steps: Optional[int] = None) -> None:
        for epoch in range(epochs):
            for batch in train_loader:
                train_metrics = self.train_step(batch)
                if eval_loader is not None and eval_every_steps is not None and self.global_step % eval_every_steps == 0:
                    eval_batch = next(iter(eval_loader))
                    self.eval_step(eval_batch)
            self.writer.add_text("train/epoch", f"Finished epoch {epoch}", self.global_step)

    @torch.no_grad()
    def generate_next_tokens(
        self,
        obs_t: ObservationPacket,
        action_t: torch.Tensor,
        prefix_tokens: torch.Tensor,
        steps: int = 16,
    ) -> List[int]:
        """Autoregressively generate tokens from the causal text decoder."""
        self.model.eval()
        batch_size = prefix_tokens.shape[0]
        if batch_size != 1:
            raise ValueError("generate_next_tokens currently expects batch size 1.")

        obs_t = ObservationPacket(
            modalities={k: v.to(self.device) for k, v in obs_t.modalities.items()},
            masks={k: v.to(self.device) for k, v in obs_t.masks.items()},
            meta=dict(obs_t.meta),
        )
        action_t = action_t.to(self.device)
        prefix_tokens = prefix_tokens.to(self.device)

        latent = self.model.encode(obs_t)
        memory = self.model.memory.init_state(batch_size=1, device=self.device)
        transition = self.model.transition(latent, action_t, memory)
        pred_latent = transition.next_latent
        generated: List[int] = []

        for _ in range(steps):
            outputs = self.model.decode(pred_latent, context={"prefix_tokens": prefix_tokens})
            logits = next(v for k, v in outputs.items() if k.endswith("text_logits"))
            # logits is [B, T, V]; take last position for next-token prediction
            next_token = int(logits[:, -1, :].argmax(dim=-1).item())
            generated.append(next_token)
            next_token_tensor = torch.tensor([[next_token]], device=self.device, dtype=prefix_tokens.dtype)
            prefix_tokens = torch.cat([prefix_tokens, next_token_tensor], dim=1)

        self.hooks.log_inference_trace(generated, self.global_step, split="inference")
        return generated
