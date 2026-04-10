"""Training loop with TensorBoard logging and autoregressive generation."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .containers import MemoryState, ObservationPacket
from .curriculum import AdaptiveCurriculumScheduler, CurriculumPhase
from .evaluation import EvaluationSuite
from .losses import WorldModelLoss
from .model import ModularLatentWorldModel
from .monitoring import HookManager


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float = 0.05,
    min_lr_ratio: float = 0.01,
) -> LRScheduler:
    """Cosine-decay LR scheduler with linear warmup."""
    warmup_steps = int(total_steps * warmup_fraction)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.run_dir = run_dir
        self.grad_clip_norm = grad_clip_norm
        self.scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and device.type == "cuda")
        self.writer = SummaryWriter(log_dir=run_dir)
        self.hooks = HookManager(self.writer)
        self.eval_suite = EvaluationSuite(writer=self.writer)
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.curriculum: Optional[List[CurriculumPhase]] = None
        self.adaptive_scheduler: Optional[AdaptiveCurriculumScheduler] = None

    def set_curriculum(self, phases: List["CurriculumPhase"]) -> None:
        self.curriculum = phases

    def set_adaptive_curriculum(self, scheduler: AdaptiveCurriculumScheduler) -> None:
        """Use an adaptive curriculum that transitions phases based on loss plateaus."""
        self.adaptive_scheduler = scheduler

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Optional[str] = None, is_best: bool = False) -> str:
        if path is None:
            path = os.path.join(
                self.run_dir,
                "checkpoint_best.pt" if is_best else f"checkpoint_step_{self.global_step}.pt",
            )
        state: Dict[str, Any] = {
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "loss_fn_state_dict": self.loss_fn.state_dict(),
        }
        if self.lr_scheduler is not None:
            state["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        if self.adaptive_scheduler is not None:
            state["adaptive_scheduler"] = {
                "current_phase_idx": self.adaptive_scheduler.current_phase_idx,
                "best_avg": self.adaptive_scheduler._best_avg,
                "steps_without_improvement": self.adaptive_scheduler._steps_without_improvement,
                "loss_window": list(self.adaptive_scheduler._loss_window),
            }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.best_eval_loss = ckpt.get("best_eval_loss", float("inf"))
        if "loss_fn_state_dict" in ckpt:
            self.loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])
        if self.lr_scheduler is not None and "lr_scheduler_state_dict" in ckpt:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        if self.adaptive_scheduler is not None and "adaptive_scheduler" in ckpt:
            sched = ckpt["adaptive_scheduler"]
            self.adaptive_scheduler.current_phase_idx = sched["current_phase_idx"]
            self.adaptive_scheduler._best_avg = sched["best_avg"]
            self.adaptive_scheduler._steps_without_improvement = sched["steps_without_improvement"]
            self.adaptive_scheduler._loss_window.clear()
            self.adaptive_scheduler._loss_window.extend(sched["loss_window"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _active_phase(self) -> Optional["CurriculumPhase"]:
        if self.adaptive_scheduler is not None:
            return self.adaptive_scheduler.current_phase
        if not self.curriculum:
            return None
        for phase in self.curriculum:
            if self.global_step < phase.until_step:
                return phase
        return self.curriculum[-1]

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

    # ------------------------------------------------------------------
    # Single-step training
    # ------------------------------------------------------------------

    def train_step(
        self, batch: Mapping[str, Any], memory_state: Optional[MemoryState] = None,
    ) -> Tuple[Dict[str, float], Optional[MemoryState]]:
        self.model.train()
        obs_t = self._to_packet(batch, "t")
        obs_tp1 = self._to_packet(batch, "tp1")
        action = batch["action"].to(self.device)

        decoder_context: Dict[str, Any] = {}
        if "prefix_tokens" in batch:
            decoder_context["prefix_tokens"] = batch["prefix_tokens"].to(self.device)

        phase = self._active_phase()
        task_multipliers = phase.task_multipliers if phase is not None else None

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
            output = self.model(obs_t, action, obs_tp1=obs_tp1, memory_state=memory_state, decoder_context=decoder_context)
            loss_inputs: Dict[str, Any] = {}
            if "text_target" in batch:
                loss_inputs["text_target"] = batch["text_target"].to(self.device)
            if "vector_target" in batch:
                loss_inputs["vector_target"] = batch["vector_target"].to(self.device)
            if "image_target" in batch:
                loss_inputs["image_target"] = batch["image_target"].to(self.device)
            if "audio_target" in batch:
                loss_inputs["audio_target"] = batch["audio_target"].to(self.device)
            losses = self.loss_fn(output, loss_inputs, task_multipliers=task_multipliers)
            total_loss = losses["total_loss"]

        self.scaler.scale(total_loss).backward()
        if self.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.hooks.log_losses(losses, self.global_step, split="train")
        self.hooks.log_aux(output.aux, self.global_step, split="train")
        self.hooks.log_depth_metrics(output.aux, self.global_step, split="train")
        self.hooks.log_latents(output, self.global_step, split="train")
        self.hooks.log_gradient_norms(self.model, self.global_step)
        self.hooks.log_learning_rate(self.optimizer, self.global_step)
        if "text_target" in batch:
            self.hooks.log_text_predictions(output, {"text_target": batch["text_target"].to(self.device)}, self.global_step, split="train")
        self.hooks.log_embeddings(output, self.global_step, split="train")

        eval_result = self.eval_suite.evaluate_step(output, loss_inputs)
        self.eval_suite.log_to_tensorboard(eval_result, self.global_step, split="train")

        if self.adaptive_scheduler is not None:
            self.adaptive_scheduler.step(float(total_loss.detach().cpu().item()))

        result = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
        result.update(eval_result.all_metrics())
        if phase is not None:
            result["curriculum_phase"] = float(phase.phase_id)
            self.writer.add_scalar("train/curriculum_phase", phase.phase_id, self.global_step)
        self.global_step += 1
        return result, output.next_memory

    # ------------------------------------------------------------------
    # Sequence-level training (BPTT)
    # ------------------------------------------------------------------

    def train_sequence_step(
        self,
        batch_sequence: Mapping[str, Any],
        window_length: int = 8,
    ) -> Dict[str, float]:
        """Train on a temporal window with truncated BPTT.

        batch_sequence keys:
            text_seq:    [B, T, SeqLen]
            vector_seq:  [B, T, D]
            image_seq:   [B, T, C, H, W]
            action_seq:  [B, T-1, ActionDim]
            text_target_seq:   [B, T, SeqLen]  (optional)
            vector_target_seq: [B, T, D]       (optional)
            image_target_seq:  [B, T, C, H, W] (optional)
        """
        self.model.train()
        phase = self._active_phase()
        task_multipliers = phase.task_multipliers if phase is not None else None

        # Determine sequence length from action_seq
        action_seq = batch_sequence["action_seq"].to(self.device)
        T = action_seq.shape[1]  # T-1 transitions → T = num actions

        memory_state: Optional[MemoryState] = None
        accumulated_loss = torch.zeros((), device=self.device)
        step_losses: List[Dict[str, torch.Tensor]] = []

        self.optimizer.zero_grad(set_to_none=True)

        for t in range(T):
            # Detach memory at window boundaries for truncated BPTT
            if t > 0 and t % window_length == 0 and memory_state is not None:
                memory_state = memory_state.detach()

            # Build observation packets for timestep t and t+1
            obs_t_mods: Dict[str, torch.Tensor] = {}
            obs_tp1_mods: Dict[str, torch.Tensor] = {}
            for name, key in [("text", "text_seq"), ("vector", "vector_seq"), ("image", "image_seq")]:
                if key in batch_sequence:
                    seq = batch_sequence[key].to(self.device)
                    obs_t_mods[name] = seq[:, t]
                    obs_tp1_mods[name] = seq[:, t + 1]

            obs_t = ObservationPacket(modalities=obs_t_mods, masks={}, meta={})
            obs_tp1 = ObservationPacket(modalities=obs_tp1_mods, masks={}, meta={})
            action = action_seq[:, t]

            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                output = self.model(obs_t, action, obs_tp1=obs_tp1, memory_state=memory_state)
                loss_inputs: Dict[str, Any] = {}
                if "text_target_seq" in batch_sequence:
                    loss_inputs["text_target"] = batch_sequence["text_target_seq"][:, t + 1].to(self.device)
                if "vector_target_seq" in batch_sequence:
                    loss_inputs["vector_target"] = batch_sequence["vector_target_seq"][:, t + 1].to(self.device)
                if "image_target_seq" in batch_sequence:
                    loss_inputs["image_target"] = batch_sequence["image_target_seq"][:, t + 1].to(self.device)
                losses = self.loss_fn(output, loss_inputs, task_multipliers=task_multipliers)
                accumulated_loss = accumulated_loss + losses["total_loss"]
                step_losses.append(losses)

            memory_state = output.next_memory

        # Average loss over timesteps, then backward
        avg_loss = accumulated_loss / max(T, 1)
        self.scaler.scale(avg_loss).backward()
        if self.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Aggregate metrics
        result: Dict[str, float] = {}
        if step_losses:
            for key in step_losses[0]:
                vals = [sl[key].detach().cpu().item() for sl in step_losses]
                result[key] = sum(vals) / len(vals)
        result["sequence_length"] = float(T)
        if phase is not None:
            result["curriculum_phase"] = float(phase.phase_id)
            self.writer.add_scalar("train/curriculum_phase", phase.phase_id, self.global_step)

        # Log to TensorBoard
        self.hooks.log_losses(
            {k: torch.tensor(v) for k, v in result.items() if k not in ("sequence_length", "curriculum_phase")},
            self.global_step, split="train",
        )
        self.hooks.log_gradient_norms(self.model, self.global_step)
        self.hooks.log_learning_rate(self.optimizer, self.global_step)

        if self.adaptive_scheduler is not None:
            self.adaptive_scheduler.step(float(avg_loss.detach().cpu().item()))

        self.global_step += 1
        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_step(
        self, batch: Mapping[str, Any], memory_state: Optional[MemoryState] = None,
    ) -> Tuple[Dict[str, float], Optional[MemoryState]]:
        self.model.eval()
        obs_t = self._to_packet(batch, "t")
        obs_tp1 = self._to_packet(batch, "tp1")
        action = batch["action"].to(self.device)

        decoder_context: Dict[str, Any] = {}
        if "prefix_tokens" in batch:
            decoder_context["prefix_tokens"] = batch["prefix_tokens"].to(self.device)

        with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
            output = self.model(obs_t, action, obs_tp1=obs_tp1, memory_state=memory_state, decoder_context=decoder_context)
            loss_inputs: Dict[str, Any] = {}
            if "text_target" in batch:
                loss_inputs["text_target"] = batch["text_target"].to(self.device)
            if "vector_target" in batch:
                loss_inputs["vector_target"] = batch["vector_target"].to(self.device)
            if "image_target" in batch:
                loss_inputs["image_target"] = batch["image_target"].to(self.device)
            if "audio_target" in batch:
                loss_inputs["audio_target"] = batch["audio_target"].to(self.device)
            phase = self._active_phase()
            task_multipliers = phase.task_multipliers if phase is not None else None
            losses = self.loss_fn(output, loss_inputs, task_multipliers=task_multipliers)

        self.hooks.log_losses(losses, self.global_step, split="eval")
        self.hooks.log_aux(output.aux, self.global_step, split="eval")
        self.hooks.log_depth_metrics(output.aux, self.global_step, split="eval")
        self.hooks.log_latents(output, self.global_step, split="eval")
        if "text_target" in batch:
            self.hooks.log_text_predictions(output, {"text_target": batch["text_target"].to(self.device)}, self.global_step, split="eval")
        self.hooks.log_embeddings(output, self.global_step, split="eval")

        eval_result = self.eval_suite.evaluate_step(output, loss_inputs)
        self.eval_suite.log_to_tensorboard(eval_result, self.global_step, split="eval")

        result = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
        result.update(eval_result.all_metrics())
        return result, output.next_memory

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        epochs: int = 1,
        eval_every_steps: Optional[int] = None,
    ) -> None:
        for epoch in range(epochs):
            memory_state: Optional[MemoryState] = None
            for batch in train_loader:
                train_metrics, memory_state = self.train_step(batch, memory_state=memory_state)
                # Detach memory to prevent infinite BPTT across batches
                if memory_state is not None:
                    memory_state = memory_state.detach()
                if eval_loader is not None and eval_every_steps is not None and self.global_step % eval_every_steps == 0:
                    eval_batch = next(iter(eval_loader))
                    eval_metrics, _ = self.eval_step(eval_batch)
                    eval_loss = eval_metrics.get("total_loss", float("inf"))
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_checkpoint(is_best=True)
            self.writer.add_text("train/epoch", f"Finished epoch {epoch}", self.global_step)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

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
            next_token = int(logits[:, -1, :].argmax(dim=-1).item())
            generated.append(next_token)
            next_token_tensor = torch.tensor([[next_token]], device=self.device, dtype=prefix_tokens.dtype)
            prefix_tokens = torch.cat([prefix_tokens, next_token_tensor], dim=1)

        self.hooks.log_inference_trace(generated, self.global_step, split="inference")
        return generated
