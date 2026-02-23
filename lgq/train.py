"""Unified training pipeline for LGQ and baseline quantizers.

Supports training VQGAN with:
  - LGQ (Learnable Geometric Quantization)
  - FSQ (Fixed Scalar Quantization)
  - SimVQ (Simple Vector Quantization with EMA)

Usage:
  python -m lgq.train --quantizer lgq --steps 100000
  python -m lgq.train --quantizer fsq --steps 100000
  python -m lgq.train --quantizer simvq --steps 100000
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lgq.config import LGQConfig
from lgq.model import LGQVAE
from lgq.losses import LGQLoss
from lgq.metrics import (
    MetricAggregator,
    InceptionFeatureExtractor,
    psnr,
    ssim,
)


class SyntheticImageDataset(Dataset):
    """Synthetic dataset for testing and development.

    Generates random images with structured patterns (gradients, circles,
    grids) to provide non-trivial reconstruction targets.
    """

    def __init__(self, size: int = 10000, resolution: int = 256, channels: int = 3):
        self.size = size
        self.resolution = resolution
        self.channels = channels

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        r = self.resolution
        img = torch.zeros(self.channels, r, r)

        gen = torch.Generator().manual_seed(idx)

        # Random gradient background
        direction = torch.rand(1, generator=gen).item()
        if direction < 0.33:
            grad = torch.linspace(0, 1, r).unsqueeze(0).expand(r, r)
        elif direction < 0.66:
            grad = torch.linspace(0, 1, r).unsqueeze(1).expand(r, r)
        else:
            x = torch.linspace(-1, 1, r)
            y = torch.linspace(-1, 1, r)
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            grad = (xx.pow(2) + yy.pow(2)).sqrt() / math.sqrt(2)

        for c in range(self.channels):
            scale = torch.rand(1, generator=gen).item() * 0.5 + 0.5
            offset = torch.rand(1, generator=gen).item() * 0.3
            img[c] = grad * scale + offset

        # Random geometric shapes
        n_shapes = int(torch.randint(2, 6, (1,), generator=gen).item())
        for _ in range(n_shapes):
            cx = torch.rand(1, generator=gen).item()
            cy = torch.rand(1, generator=gen).item()
            radius = torch.rand(1, generator=gen).item() * 0.2 + 0.05
            color = torch.rand(self.channels, generator=gen)

            x = torch.linspace(0, 1, r)
            y = torch.linspace(0, 1, r)
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            mask = ((xx - cx).pow(2) + (yy - cy).pow(2)).sqrt() < radius

            for c in range(self.channels):
                img[c][mask] = color[c].item()

        return img.clamp(0, 1)


def create_dataloader(config: LGQConfig, split: str = "train") -> DataLoader:
    """Create dataloader. Uses synthetic data if no real data path exists."""
    size = 10000 if split == "train" else 1000
    dataset = SyntheticImageDataset(
        size=size,
        resolution=config.resolution,
        channels=config.in_channels,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )


def train(config: LGQConfig) -> dict:
    """Main training loop.

    Returns:
        Final evaluation metrics dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training LGQ with quantizer={config.quantizer_type} on {device}")
    print(f"Config: {config}")

    # Model
    model = LGQVAE(config).to(device)
    param_counts = model.count_parameters()
    print(f"Parameters: {param_counts}")

    # Loss
    loss_fn = LGQLoss(config)

    # Optimizers (separate for generator and discriminator)
    param_groups = model.get_param_groups()
    opt_gen = torch.optim.AdamW(
        param_groups["generator"],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.5, 0.9),
    )
    opt_disc = torch.optim.AdamW(
        param_groups["discriminator"],
        lr=config.disc_learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.5, 0.9),
    )

    # LR schedulers
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(
            1, config.total_steps - config.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched_gen = torch.optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda)
    sched_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda)

    # Data
    train_loader = create_dataloader(config, "train")
    val_loader = create_dataloader(config, "val")

    # Metrics
    feature_extractor = InceptionFeatureExtractor().to(device)
    feature_extractor.eval()
    aggregator = MetricAggregator(config.n_codebooks, config.vocab_size)

    # Mixed precision
    use_amp = config.precision == "bf16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp and config.precision != "bf16" else None
    amp_dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16

    # Training loop
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    step = 0
    best_metrics = {}
    log_history = []
    train_iter = iter(train_loader)

    print(f"Starting training for {config.total_steps} steps...")
    t0 = time.time()

    while step < config.total_steps:
        model.train()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x = batch.to(device)

        # --- Generator step ---
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            model_out = model(x)

            # Adversarial loss for generator
            adv_loss = None
            if step >= config.disc_start_step:
                adv_loss = model.generator_adversarial_loss(model_out["recon"])

            gen_losses = loss_fn.generator_loss(x, model_out, adv_loss, step)

        opt_gen.zero_grad()
        if scaler is not None:
            scaler.scale(gen_losses["total"]).backward()
            scaler.unscale_(opt_gen)
            nn.utils.clip_grad_norm_(
                param_groups["generator"], config.max_grad_norm
            )
            scaler.step(opt_gen)
            scaler.update()
        else:
            gen_losses["total"].backward()
            nn.utils.clip_grad_norm_(
                param_groups["generator"], config.max_grad_norm
            )
            opt_gen.step()

        sched_gen.step()

        # --- Discriminator step ---
        if step >= config.disc_start_step:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                disc_out = model.forward_discriminator(x, model_out["recon"].detach())

            opt_disc.zero_grad()
            if scaler is not None:
                scaler.scale(disc_out["d_loss"]).backward()
                scaler.step(opt_disc)
                scaler.update()
            else:
                disc_out["d_loss"].backward()
                opt_disc.step()

            sched_disc.step()

        step += 1

        # --- Logging ---
        if step % config.log_every == 0:
            elapsed = time.time() - t0
            log_entry = {
                "step": step,
                "elapsed_s": round(elapsed, 1),
                "lr": opt_gen.param_groups[0]["lr"],
            }
            for k, v in gen_losses.items():
                if isinstance(v, torch.Tensor):
                    log_entry[k] = round(v.item(), 6)
                elif isinstance(v, float):
                    log_entry[k] = round(v, 6)

            # Codebook utilization
            if hasattr(model.quantizer, "codebook_utilization"):
                util = model.quantizer.codebook_utilization()
                log_entry.update({f"cb_{k}": round(v, 4) for k, v in util.items()})

            log_history.append(log_entry)
            print(
                f"[{step}/{config.total_steps}] "
                f"loss={log_entry.get('total', 0):.4f} "
                f"recon={log_entry.get('recon_loss', 0):.4f} "
                f"cb={log_entry.get('codebook_loss', 0):.4f} "
                f"tau={log_entry.get('temperature', 0):.4f} "
                f"active={log_entry.get('cb_active_ratio', 0):.3f}"
            )

        # --- Evaluation ---
        if step % config.eval_every == 0:
            metrics = evaluate(model, val_loader, aggregator, feature_extractor, device, use_amp, amp_dtype)
            print(f"  EVAL @ {step}: " + " ".join(
                f"{k}={v:.4f}" for k, v in metrics.items()
            ))

            if not best_metrics or metrics.get("psnr", 0) > best_metrics.get("psnr", 0):
                best_metrics = metrics
                torch.save(
                    {"step": step, "model": model.state_dict(), "metrics": metrics},
                    os.path.join(config.checkpoint_dir, "best.pt"),
                )

        # --- Checkpoint ---
        if step % config.save_every == 0:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "opt_gen": opt_gen.state_dict(),
                    "opt_disc": opt_disc.state_dict(),
                },
                os.path.join(config.checkpoint_dir, f"step_{step}.pt"),
            )

    # Final evaluation
    final_metrics = evaluate(model, val_loader, aggregator, feature_extractor, device, use_amp, amp_dtype)
    print(f"\nFinal metrics: {final_metrics}")

    # Save training log
    with open(os.path.join(config.log_dir, "train_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)

    return final_metrics


@torch.no_grad()
def evaluate(
    model: LGQVAE,
    loader: DataLoader,
    aggregator: MetricAggregator,
    feature_extractor: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """Run evaluation over a dataloader."""
    model.eval()
    aggregator.reset()

    for batch in loader:
        x = batch.to(device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(x)

        recon = out["recon"].clamp(0, 1)
        indices = out["indices"]

        # Feature extraction for FID
        real_feats = feature_extractor(x)
        fake_feats = feature_extractor(recon)

        aggregator.update(recon, x, indices, real_feats, fake_feats)

    return aggregator.compute()


def main():
    parser = argparse.ArgumentParser(description="Train LGQ image tokenizer")
    parser.add_argument("--quantizer", type=str, default="lgq",
                        choices=["lgq", "fsq", "simvq", "vq"],
                        help="Quantizer type")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--n-codebooks", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)

    args = parser.parse_args()

    config = LGQConfig(
        quantizer_type=args.quantizer,
        total_steps=args.steps,
        batch_size=args.batch_size,
        resolution=args.resolution,
        vocab_size=args.vocab_size,
        n_codebooks=args.n_codebooks,
        learning_rate=args.lr,
    )

    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        config.log_dir = args.log_dir

    train(config)


if __name__ == "__main__":
    main()
