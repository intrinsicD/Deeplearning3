"""TensorBoard monitoring hooks."""

from __future__ import annotations

import math
from typing import Any, Callable, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .containers import ModelOutput


class HookManager:
    """Central place for train/inference observability via TensorBoard."""

    def __init__(self, writer: SummaryWriter, histogram_every: int = 200, embedding_every: int = 500) -> None:
        self.writer = writer
        self.histogram_every = histogram_every
        self.embedding_every = embedding_every

    def log_losses(self, losses: Mapping[str, torch.Tensor], step: int, split: str) -> None:
        for name, value in losses.items():
            self.writer.add_scalar(f"{split}/loss/{name}", float(value.detach().cpu().item()), step)

    def log_aux(self, aux: Mapping[str, torch.Tensor], step: int, split: str) -> None:
        for name, value in aux.items():
            if value.ndim == 0:
                self.writer.add_scalar(f"{split}/aux/{name}", float(value.detach().cpu().item()), step)
            else:
                self.writer.add_scalar(f"{split}/aux/{name}_mean", float(value.detach().mean().cpu().item()), step)

    def log_latents(self, output: ModelOutput, step: int, split: str) -> None:
        current = output.current_latent
        pred = output.predicted_next_latent
        for tag, latent in [("current", current), ("predicted", pred)]:
            self.writer.add_scalar(f"{split}/latent/{tag}_z_sem_norm", float(latent.z_sem.norm(dim=-1).mean().detach().cpu().item()), step)
            self.writer.add_scalar(f"{split}/latent/{tag}_z_sem_std", float(latent.z_sem.std(dim=0).mean().detach().cpu().item()), step)
            if latent.z_dyn is not None:
                self.writer.add_scalar(f"{split}/latent/{tag}_z_dyn_norm", float(latent.z_dyn.norm(dim=-1).mean().detach().cpu().item()), step)
            if latent.z_ctrl is not None:
                self.writer.add_scalar(f"{split}/latent/{tag}_z_ctrl_norm", float(latent.z_ctrl.norm(dim=-1).mean().detach().cpu().item()), step)
            if latent.z_mem is not None:
                self.writer.add_scalar(f"{split}/latent/{tag}_z_mem_norm", float(latent.z_mem.norm(dim=-1).mean().detach().cpu().item()), step)

        if step % self.histogram_every == 0:
            self.writer.add_histogram(f"{split}/hist/current_z_sem", current.z_sem.detach().cpu(), step)
            self.writer.add_histogram(f"{split}/hist/predicted_z_sem", pred.z_sem.detach().cpu(), step)

    def log_embeddings(self, output: ModelOutput, step: int, split: str, metadata: Optional[List[str]] = None) -> None:
        if step % self.embedding_every != 0:
            return
        z = output.current_latent.z_sem.detach().cpu()
        max_points = min(256, z.shape[0])
        self.writer.add_embedding(z[:max_points], metadata=metadata[:max_points] if metadata else None, tag=f"{split}/embeddings/z_sem", global_step=step)

    def log_text_predictions(self, output: ModelOutput, batch: Mapping[str, Any], step: int, split: str, id_to_token: Optional[Callable[[int], str]] = None) -> None:
        text_key = next((k for k in output.decoder_outputs if k.endswith("text_logits")), None)
        if text_key is None or "text_target" not in batch:
            return
        logits = output.decoder_outputs[text_key].detach()
        pred_ids = logits.argmax(dim=-1)  # [B, T] or [B]
        target_ids = batch["text_target"].detach()

        def tok(i: int) -> str:
            return id_to_token(i) if id_to_token is not None else str(i)

        preview = []
        if pred_ids.ndim == 2:
            for p_seq, t_seq in zip(pred_ids[:4].tolist(), target_ids[:4].tolist()):
                pred_str = " ".join(tok(int(x)) for x in p_seq[:8])
                tgt_str = " ".join(tok(int(x)) for x in (t_seq[:8] if isinstance(t_seq, list) else [t_seq]))
                preview.append(f"pred=[{pred_str}] | target=[{tgt_str}]")
        else:
            for p, t in zip(pred_ids[:8].tolist(), target_ids[:8].tolist()):
                preview.append(f"pred={tok(int(p))} | target={tok(int(t))}")
        text = "\n".join(preview)
        self.writer.add_text(f"{split}/text_predictions", text, step)

    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: int) -> None:
        for idx, group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"train/lr/group_{idx}", group["lr"], step)

    def log_gradient_norms(self, model: nn.Module, step: int) -> None:
        total_sq = 0.0
        count = 0
        for param in model.parameters():
            if param.grad is None:
                continue
            g = param.grad.detach().float()
            total_sq += float(g.norm().item() ** 2)
            count += 1
        total_norm = math.sqrt(total_sq) if total_sq > 0 else 0.0
        self.writer.add_scalar("train/grad/total_norm", total_norm, step)
        self.writer.add_scalar("train/grad/num_tensors", count, step)

    def log_inference_trace(self, predicted_tokens: Sequence[int], step: int, split: str, id_to_token: Optional[Callable[[int], str]] = None) -> None:
        if id_to_token is None:
            rendered = " ".join(str(int(x)) for x in predicted_tokens)
        else:
            rendered = " ".join(id_to_token(int(x)) for x in predicted_tokens)
        self.writer.add_text(f"{split}/inference/generated", rendered, step)

    def log_depth_metrics(self, aux: Mapping[str, torch.Tensor], step: int, split: str) -> None:
        for key in ["recurrent_steps_mean", "recurrent_steps_max", "recurrent_hidden_norm", "attnres_num_blocks", "attnres_hidden_norm"]:
            if key in aux:
                value = aux[key]
                scalar = float(value.detach().mean().cpu().item()) if value.ndim > 0 else float(value.detach().cpu().item())
                self.writer.add_scalar(f"{split}/depth/{key}", scalar, step)
