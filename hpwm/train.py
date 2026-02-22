"""
HPWM Phase -1 Training Loop.

Single-GPU training with:
- Gradient accumulation (effective batch size 16)
- Gradient checkpointing for memory efficiency
- bf16 mixed precision
- MoD K-ratio annealing
- Periodic evaluation of 3 validation signals
- Tensorboard logging

Usage:
    python -m hpwm.train [--config path/to/config.yaml] [--synthetic]
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler

from hpwm.config import HPWMConfig
from hpwm.model import HPWM
from hpwm.data import create_dataloaders
from hpwm.evaluate import Evaluator


class Trainer:
    """HPWM Phase -1 Trainer."""

    def __init__(
        self,
        config: HPWMConfig,
        use_synthetic: bool = True,
        resume_from: str | None = None,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_synthetic = use_synthetic

        # Create output directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Initialize model
        print("Initializing HPWM model...")
        self.model = HPWM(config).to(self.device)

        # Print parameter counts
        param_counts = self.model.count_parameters()
        print("\nParameter counts:")
        for name, counts in param_counts.items():
            if name != "total":
                print(f"  {name}: {counts['trainable']:,} trainable / {counts['total']:,} total")
        total = param_counts["total"]
        print(f"  TOTAL: {total['trainable']:,} trainable / {total['total']:,} total")
        print(f"  Estimated VRAM for params: {total['total'] * 2 / 1e6:.1f} MB (fp16)")

        # Optimizer
        param_groups = self.model.get_param_groups()
        self.optimizer = torch.optim.AdamW(
            [
                {"params": g["params"], "lr": config.lr * g["lr_scale"]}
                for g in param_groups if g["params"]
            ],
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler (cosine with warmup)
        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / max(1, config.warmup_steps)
            progress = (step - config.warmup_steps) / max(
                1, config.total_steps - config.warmup_steps,
            )
            import math as _math
            return 0.5 * (1 + _math.cos(_math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda,
        )

        # Mixed precision
        self.use_amp = config.precision in ("bf16", "fp16")
        self.amp_dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
        self.scaler = GradScaler(enabled=(config.precision == "fp16"))

        # Data
        print("\nCreating dataloaders...")
        self.train_loader, self.val_loader = create_dataloaders(
            config, use_synthetic=use_synthetic,
        )
        print(f"  Train: {len(self.train_loader.dataset)} clips")
        print(f"  Val: {len(self.val_loader.dataset)} clips")

        # Evaluator
        self.evaluator = Evaluator(config, self.model, self.val_loader, self.device)

        # Logging
        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(config.log_dir)
        except ImportError:
            print("[WARN] tensorboard not available, logging to stdout only")

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Resume
        if resume_from:
            self._load_checkpoint(resume_from)

    def train(self):
        """Main training loop."""
        config = self.config
        print(f"\nStarting training for {config.total_steps} steps...")
        print(f"  Effective batch size: {config.effective_batch_size}")
        print(f"  Device: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print()

        self.model.train()
        accum_loss = 0.0
        accum_metrics = {}
        temporal_states = None
        step_start_time = time.time()

        while self.global_step < config.total_steps:
            self.epoch += 1

            for batch in self.train_loader:
                if self.global_step >= config.total_steps:
                    break

                frames = batch["frames"].to(self.device)  # [B, T, 3, H, W]

                # Forward pass with mixed precision
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=(self.use_amp and self.device.type == "cuda"),
                ):
                    outputs = self.model(frames, temporal_states)
                    loss = outputs["loss"] / config.grad_accum_steps

                # Backward pass
                self.scaler.scale(loss).backward()

                # Track metrics
                accum_loss += outputs["loss"].item() / config.grad_accum_steps
                for key in [
                    "prediction_loss", "vqvae_recon_loss",
                    "fwm_loss", "commitment_loss", "entropy_loss",
                    "slot_consistency_loss", "slot_specialization_loss",
                ]:
                    val = outputs[key].item()
                    accum_metrics[key] = accum_metrics.get(key, 0) + val / config.grad_accum_steps

                # Detach temporal states for next iteration
                temporal_states = [
                    s.detach() if s is not None else None
                    for s in outputs["temporal_states"]
                ]

                # Gradient accumulation step
                if (self.global_step + 1) % config.grad_accum_steps == 0 or True:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), config.max_grad_norm,
                    )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    # Step MoD K-ratio annealing
                    self.model.mod_router.step()
                    # Step VQ-VAE warmup counter
                    self.model._train_step += 1

                    self.global_step += 1

                    # Logging
                    if self.global_step % config.log_every == 0:
                        elapsed = time.time() - step_start_time
                        steps_per_sec = config.log_every / elapsed

                        lr = self.scheduler.get_last_lr()[0]
                        k_ratio = self.model.mod_router.current_k_ratio

                        print(
                            f"Step {self.global_step}/{config.total_steps} | "
                            f"loss={accum_loss:.4f} | "
                            f"pred={accum_metrics.get('prediction_loss', 0):.4f} | "
                            f"vqvae={accum_metrics.get('vqvae_recon_loss', 0):.4f} | "
                            f"fwm={accum_metrics.get('fwm_loss', 0):.4f} | "
                            f"slot_con={accum_metrics.get('slot_consistency_loss', 0):.4f} | "
                            f"k_ratio={k_ratio:.3f} | "
                            f"lr={lr:.2e} | "
                            f"grad_norm={grad_norm:.3f} | "
                            f"{steps_per_sec:.2f} steps/s"
                        )

                        if self.tb_writer:
                            self.tb_writer.add_scalar("train/loss", accum_loss, self.global_step)
                            for k, v in accum_metrics.items():
                                self.tb_writer.add_scalar(f"train/{k}", v, self.global_step)
                            self.tb_writer.add_scalar("train/lr", lr, self.global_step)
                            self.tb_writer.add_scalar("train/k_ratio", k_ratio, self.global_step)
                            self.tb_writer.add_scalar("train/grad_norm", grad_norm, self.global_step)

                        accum_loss = 0.0
                        accum_metrics = {}
                        step_start_time = time.time()

                    # Evaluation
                    if self.global_step % config.eval_every == 0:
                        self._evaluate()

                    # Save checkpoint
                    if self.global_step % config.save_every == 0:
                        self._save_checkpoint()

                    # Reset temporal states periodically (between videos)
                    if self.global_step % 100 == 0:
                        temporal_states = None

        # Final evaluation and checkpoint
        self._evaluate()
        self._save_checkpoint(is_final=True)
        print("\nTraining complete!")

    @torch.no_grad()
    def _evaluate(self):
        """Run evaluation and log validation signals."""
        self.model.eval()

        print(f"\n{'='*60}")
        print(f"Evaluation at step {self.global_step}")
        print(f"{'='*60}")

        results = self.evaluator.evaluate_all()

        # Signal 1: MoD Routing Entropy
        sig1 = results.get("signal_1_routing_entropy", {})
        print(f"\nSignal 1 - MoD Routing Entropy:")
        print(f"  Entropy: {sig1.get('entropy', 'N/A'):.4f}")
        print(f"  Heavy ratio: {sig1.get('heavy_ratio', 'N/A'):.4f}")

        # Signal 2: Slot Binding Stability
        sig2 = results.get("signal_2_slot_stability", {})
        print(f"\nSignal 2 - Slot Binding Stability:")
        print(f"  Mean Jaccard: {sig2.get('mean_jaccard', 'N/A'):.4f}")
        print(f"  Pass (>0.6): {'YES' if sig2.get('mean_jaccard', 0) > 0.6 else 'NO'}")

        # Signal 3: Mamba State Retention
        sig3 = results.get("signal_3_state_retention", {})
        print(f"\nSignal 3 - State Retention:")
        for length, acc in sig3.items():
            print(f"  {length}: {acc:.4f}")

        # VQ-VAE codebook utilization
        cb_metrics = self.model.vqvae.quantizer.codebook_utilization()
        print(f"\nVQ-VAE Codebook Utilization:")
        print(f"  Active ratio: {cb_metrics['active_ratio']:.4f}")
        print(f"  Perplexity: {cb_metrics['perplexity']:.1f} / {self.config.vqvae_vocab_size}")

        # Validation loss
        val_loss = results.get("val_loss", float("inf"))
        print(f"\nValidation loss: {val_loss:.4f}")
        print(f"{'='*60}\n")

        # Log to tensorboard
        if self.tb_writer:
            for k, v in sig1.items():
                self.tb_writer.add_scalar(f"signal_1/{k}", v, self.global_step)
            for k, v in sig2.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f"signal_2/{k}", v, self.global_step)
            for k, v in sig3.items():
                self.tb_writer.add_scalar(f"signal_3/{k}", v, self.global_step)
            self.tb_writer.add_scalar("val/loss", val_loss, self.global_step)
            self.tb_writer.add_scalar("vqvae/active_ratio", cb_metrics["active_ratio"], self.global_step)
            self.tb_writer.add_scalar("vqvae/perplexity", cb_metrics["perplexity"], self.global_step)

        # Track best
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self._save_checkpoint(is_best=True)

        # Save evaluation results
        results_file = Path(self.config.log_dir) / f"eval_step_{self.global_step}.json"
        serializable = {}
        for k, v in results.items():
            if isinstance(v, dict):
                serializable[k] = {
                    sk: float(sv) if isinstance(sv, (int, float)) else str(sv)
                    for sk, sv in v.items()
                }
            else:
                serializable[k] = float(v) if isinstance(v, (int, float)) else str(v)
        with open(results_file, "w") as f:
            json.dump(serializable, f, indent=2)

        self.model.train()

    def _save_checkpoint(
        self, is_best: bool = False, is_final: bool = False,
    ):
        """Save model checkpoint."""
        # Filter out frozen DINO backbone weights — they're pretrained and
        # don't need to be checkpointed.  This also avoids a key mismatch on
        # resume because DINOBackbone uses lazy loading (_model starts as None).
        model_sd = {
            k: v for k, v in self.model.state_dict().items()
            if not k.startswith("dino._model.")
        }
        ckpt = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": model_sd,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_loss": self.best_loss,
            "config": vars(self.config),
        }

        ckpt_dir = Path(self.config.checkpoint_dir)

        if is_final:
            path = ckpt_dir / "checkpoint_final.pt"
        elif is_best:
            path = ckpt_dir / "checkpoint_best.pt"
        else:
            path = ckpt_dir / f"checkpoint_step_{self.global_step}.pt"

        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {path}...")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Filter out any frozen DINO backbone weights that may have been
        # saved by older checkpoints before the save-side filter was added.
        model_sd = {
            k: v for k, v in ckpt["model_state_dict"].items()
            if not k.startswith("dino._model.")
        }
        self.model.load_state_dict(model_sd, strict=False)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.epoch = ckpt["epoch"]
        self.best_loss = ckpt.get("best_loss", float("inf"))

        print(f"Resumed from step {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description="HPWM Phase -1 Training")
    parser.add_argument(
        "--synthetic", action="store_true", default=True,
        help="Use synthetic dataset (default)",
    )
    parser.add_argument(
        "--ssv2-dir", type=str, default=None,
        help="Path to SSv2 dataset directory",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override total steps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--n-frames", type=int, default=None, help="Override frame count")
    parser.add_argument(
        "--no-mamba", action="store_true",
        help="Use Transformer baseline instead of Mamba",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Run evaluation only (requires --resume)",
    )
    args = parser.parse_args()

    config = HPWMConfig()

    # Apply overrides
    if args.steps:
        config.total_steps = args.steps
    if args.lr:
        config.lr = args.lr
    if args.n_frames:
        config.n_frames = args.n_frames
    if args.no_mamba:
        config.use_mamba = False

    use_synthetic = args.ssv2_dir is None

    trainer = Trainer(
        config=config,
        use_synthetic=use_synthetic,
        resume_from=args.resume,
    )

    if args.eval_only:
        if not args.resume:
            print("Error: --eval-only requires --resume")
            return
        trainer._evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
