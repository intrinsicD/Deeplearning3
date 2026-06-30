"""Memory-efficient trainer for OmniLatent.

Key features:
  * Mixed-precision (FP16) via torch.amp
  * Gradient accumulation (optional)
  * Cosine learning rate schedule with warm-up
  * Gradient clipping
  * Multi-modal round-robin training via TaskSampler
  * Structured metrics collection via MetricsAggregator
  * Deterministic seeding for reproducibility
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from omnilatent.config import OmniLatentConfig
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.losses import MultiModalLoss
from omnilatent.training.metrics import MetricsAggregator, StepMetrics
from omnilatent.training.sampler import TaskSampler
from omnilatent.utils import ALL_MODALITIES, Modality, set_seed, count_parameters


def cosine_schedule(
    step: int,
    total_steps: int,
    warmup_steps: int,
    lr: float,
    min_lr: float = 1e-6,
) -> float:
    """Cosine annealing with linear warm-up."""
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


class Trainer:
    """Single-GPU trainer for OmniLatent.

    Trains the model by randomly sampling (source_modality, target_modality)
    pairs each step via a TaskSampler.  This gives uniform coverage of all
    modality combinations over time.

    For self-reconstruction steps, the source and target are the same
    modality.  For cross-modal steps, they differ (requires paired data).
    """

    def __init__(
        self,
        model: OmniLatentModel,
        config: OmniLatentConfig,
        dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader

        # Deterministic seeding
        self.seed = seed
        set_seed(seed)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Loss (built before the optimizer so its learnable uncertainty
        # weights — MultiModalLoss.log_vars — are actually optimized).
        self.criterion = MultiModalLoss(config).to(self.device)

        # Optimizer — must include the criterion's parameters, otherwise the
        # learned task-weighting (Kendall et al. 2018) never updates.
        self._optim_params = list(self.model.parameters()) + list(
            self.criterion.parameters()
        )
        self.optimizer = torch.optim.AdamW(
            self._optim_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and self.device.type == "cuda")
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # Task sampler (replaces raw itertools.product)
        self.task_sampler = TaskSampler(
            modalities=ALL_MODALITIES,
            self_recon_weight=0.5,
        )

        # Metrics aggregator
        self.metrics = MetricsAggregator()

        self.global_step = 0

    def _get_run_info(self) -> dict[str, Any]:
        """Collect run metadata for logging."""
        info: dict[str, Any] = {
            "seed": self.seed,
            "device": str(self.device),
            "config": {
                k: v for k, v in self.config.__dict__.items()
                if not k.startswith("_")
            },
            "model_params_total": count_parameters(self.model),
        }
        # Parameter breakdown by component
        param_breakdown: dict[str, int] = {}
        for name, module in self.model.named_children():
            param_breakdown[name] = sum(
                p.numel() for p in module.parameters()
            )
        info["model_params_breakdown"] = param_breakdown

        # Git hash if available
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                info["git_hash"] = result.stdout.strip()
        except Exception:
            pass

        return info

    def _update_lr(self) -> float:
        lr = cosine_schedule(
            self.global_step,
            self.config.max_steps,
            self.config.warmup_steps,
            self.config.learning_rate,
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """One training step.

        Uses TaskSampler to pick a (source, target) pair from available
        modalities in this batch.
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        available = list(batch.keys())
        if len(available) == 0:
            # No usable modality this step. Report it as *skipped* — never as a
            # zero-loss step, which would silently bias the loss average and
            # mask a broken data path (Audit.md A3).
            return {"skipped": 1.0}

        # Use task sampler for modality pair selection.
        src_mod, tgt_mod = self.task_sampler.sample(available)

        # Cross-modal training needs row-aligned pairs. Union collation
        # (Audit.md A3) can yield different per-modality batch sizes, so demote
        # a misaligned cross-modal task to self-reconstruction instead of
        # forwarding mismatched tensors.
        if src_mod != tgt_mod and batch[src_mod].shape[0] != batch[tgt_mod].shape[0]:
            tgt_mod = src_mod

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.config.mixed_precision):
            # Encode modalities for the contrastive term — only those whose
            # batch dimension matches the source, so InfoNCE sees aligned rows.
            latents = None
            if self.config.contrastive_weight > 0:
                ref_b = batch[src_mod].shape[0]
                aligned = [m for m in available if batch[m].shape[0] == ref_b]
                if len(aligned) >= 2:
                    latents = {}
                    for mod in aligned:
                        enc = self.model.encode(mod, batch[mod])
                        # Mean-pool content tokens (skip modality indicator at pos 0)
                        latents[mod] = enc[:, 1:].mean(dim=1)

            result = self.model(
                source_modality=src_mod,
                source_data=batch[src_mod],
                target_modality=tgt_mod,
                target_data=batch[tgt_mod],
            )

            # Compute loss (include reasoning outputs if present)
            predictions = {tgt_mod: result["output"]}
            targets = {tgt_mod: batch[tgt_mod]}

            loss_dict = self.criterion(
                predictions,
                targets,
                latents,
                reasoning_bottleneck=result.get("reasoning_bottleneck"),
                source_summary=result.get("source_summary"),
            )

        # Backward
        self.scaler.scale(loss_dict["total"]).backward()

        # Gradient clipping (unscale first for correct norm)
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._optim_params, self.config.grad_clip
        ).item()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        loss_values = {k: v.item() for k, v in loss_dict.items()}

        # Log step metrics
        lr = self.optimizer.param_groups[0]["lr"]
        step_metrics = StepMetrics(
            loss_total=loss_values.get("total", 0.0),
            loss_per_modality={
                k: v for k, v in loss_values.items() if k != "total"
            },
            grad_norm=grad_norm,
            source_modality=src_mod,
            target_modality=tgt_mod,
            learning_rate=lr,
        )
        self.metrics.log_step(step_metrics)

        return loss_values

    def train(self, log_interval: int = 50, save_dir: str | None = None) -> None:
        """Main training loop."""
        self.model.train()

        # Save run info
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            run_info = self._get_run_info()
            with open(save_path / "run_info.json", "w") as f:
                json.dump(run_info, f, indent=2, default=str)

        data_iter = iter(self.dataloader)
        running_loss = 0.0
        n_loss_steps = 0
        n_skipped = 0
        t0 = time.time()

        for step in range(self.global_step, self.config.max_steps):
            self.global_step = step
            lr = self._update_lr()

            # Get batch (loop dataloader if exhausted)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            losses = self._train_step(batch)
            if losses.get("skipped"):
                n_skipped += 1
            else:
                running_loss += losses.get("total", 0.0)
                n_loss_steps += 1

            if (step + 1) % log_interval == 0:
                # Average only over steps that actually produced a loss, so
                # skipped (empty-batch) steps cannot deflate the reported loss.
                avg_loss = running_loss / max(n_loss_steps, 1)
                elapsed = time.time() - t0
                steps_per_sec = log_interval / elapsed
                avg_grad = self.metrics.avg_grad_norm()
                skip_note = f" | skipped {n_skipped}" if n_skipped else ""
                print(
                    f"step {step + 1:>6d} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"grad {avg_grad:.2f} | "
                    f"{steps_per_sec:.1f} steps/s"
                    f"{skip_note}"
                )
                running_loss = 0.0
                n_loss_steps = 0
                n_skipped = 0
                t0 = time.time()

            # Validation
            if (
                self.val_dataloader is not None
                and (step + 1) % (log_interval * 10) == 0
            ):
                val_loss = self.validate()
                print(f"  → val_loss {val_loss:.4f}")

        # Print final metrics summary
        summary = self.metrics.summary()
        print(f"\nTraining complete. Final avg loss: {summary['avg_loss']:.4f}")
        print(f"Task distribution: {summary['task_distribution']}")

    @torch.no_grad()
    def validate(self, max_batches: int = 20) -> float:
        """Quick validation pass."""
        self.model.eval()
        total_loss = 0.0
        n = 0

        for i, batch in enumerate(self.val_dataloader):
            if i >= max_batches:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            available = list(batch.keys())
            if not available:
                continue

            src_mod = available[0]
            tgt_mod = available[0]

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.config.mixed_precision):
                result = self.model(src_mod, batch[src_mod], tgt_mod, batch[tgt_mod])
                loss_dict = self.criterion(
                    {tgt_mod: result["output"]},
                    {tgt_mod: batch[tgt_mod]},
                )

            total_loss += loss_dict["total"].item()
            n += 1

        self.model.train()
        return total_loss / max(n, 1)
