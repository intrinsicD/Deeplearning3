#!/usr/bin/env python3
"""Train OmniLatent on COCO Captions (text <-> image).

Two-phase curriculum:
  Phase 1 — Warmup:      Self-reconstruction (image->image, text->text)
  Phase 2 — Cross-modal:  image<->text translation

Usage:
    # Basic (assumes data/ has train2014/ and annotations/)
    python train_coco.py --image-dir data/train2014 \
        --annotation-file data/annotations/captions_train2014.json

    # Quick test run
    python train_coco.py --image-dir data/train2014 \
        --annotation-file data/annotations/captions_train2014.json \
        --total-steps 500 --log-interval 10

    # Resume from checkpoint
    python train_coco.py --image-dir data/train2014 \
        --annotation-file data/annotations/captions_train2014.json \
        --resume checkpoints/checkpoint_step5000.pt
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from omnilatent.config import OmniLatentConfig
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.coco_dataset import CocoCaptionsDataset
from omnilatent.training.data import collate_multimodal
from omnilatent.training.losses import MultiModalLoss
from omnilatent.training.trainer import cosine_schedule
from omnilatent.utils import count_parameters, param_size_mb, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train OmniLatent on COCO Captions")
    p.add_argument("--image-dir", type=str, required=True,
                    help="Path to COCO images (e.g., data/train2014)")
    p.add_argument("--annotation-file", type=str, required=True,
                    help="Path to captions JSON (e.g., data/annotations/captions_train2014.json)")
    p.add_argument("--total-steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-frac", type=float, default=0.15,
                    help="Fraction of steps for self-reconstruction warmup")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--save-dir", type=str, default="checkpoints_coco")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-every", type=int, default=5000,
                    help="Save a checkpoint every N steps")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--contrastive-weight", type=float, default=0.0,
                    help="Weight for contrastive loss (try 0.01-0.1 after warmup)")
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    config = OmniLatentConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.total_steps,
        mixed_precision=not args.no_amp,
        gradient_checkpointing=True,
        seed=args.seed,
        contrastive_weight=args.contrastive_weight,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("OmniLatent — COCO Captions Training (Text <-> Image)")
    print("=" * 60)
    print(f"Device:      {device}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name()}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM:        {mem:.1f} GB")

    # Model
    model = OmniLatentModel(config)
    model = model.to(device)
    print(f"Parameters:  {count_parameters(model):,} ({param_size_mb(model):.1f} MB)")

    # Dataset
    dataset = CocoCaptionsDataset(
        image_dir=args.image_dir,
        annotation_file=args.annotation_file,
        config=config,
        augment=not args.no_augment,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_multimodal,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # Optimizer, loss, scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    criterion = MultiModalLoss(config).to(device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.mixed_precision and device.type == "cuda",
    )
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = int(args.total_steps * args.warmup_frac)

    # Resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    # Training info
    print(f"Dataset:     {len(dataset)} image-caption pairs")
    print(f"Batch size:  {args.batch_size}")
    print(f"Total steps: {args.total_steps}")
    print(f"Warmup:      {warmup_steps} steps (self-reconstruction)")
    print(f"Cross-modal: {args.total_steps - warmup_steps} steps (image <-> text)")
    print(f"Save dir:    {save_dir}")
    print("=" * 60)

    model.train()
    data_iter = iter(dataloader)
    running_loss = 0.0
    t0 = time.time()
    current_phase = None

    for step in range(start_step, args.total_steps):
        # Learning rate schedule
        lr = cosine_schedule(step, args.total_steps, config.warmup_steps, config.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        image = batch["image"].to(device)
        text = batch["text"].to(device)

        # Phase selection: warmup = self-reconstruction, then cross-modal
        if step < warmup_steps:
            phase = "warmup"
            # Self-reconstruction: randomly pick image->image or text->text
            if random.random() < 0.5:
                src_mod, tgt_mod = "image", "image"
                source, target = image, image
            else:
                src_mod, tgt_mod = "text", "text"
                source, target = text, text
        else:
            phase = "cross_modal"
            # Cross-modal: image->text, text->image, or self-reconstruction
            r = random.random()
            if r < 0.35:
                src_mod, tgt_mod = "image", "text"
                source, target = image, text
            elif r < 0.70:
                src_mod, tgt_mod = "text", "image"
                source, target = text, image
            elif r < 0.85:
                src_mod, tgt_mod = "image", "image"
                source, target = image, image
            else:
                src_mod, tgt_mod = "text", "text"
                source, target = text, text

        if phase != current_phase:
            current_phase = phase
            print()
            print("=" * 60)
            if phase == "warmup":
                print(f"Phase 1: Warmup — self-reconstruction (steps 0-{warmup_steps})")
            else:
                print(f"Phase 2: Cross-modal — image <-> text (steps {warmup_steps}-{args.total_steps})")
            print("=" * 60)

        # Forward
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=config.mixed_precision):
            result = model(
                source_modality=src_mod,
                source_data=source,
                target_modality=tgt_mod,
                target_data=target,
            )

            predictions = {tgt_mod: result["output"]}
            targets_dict = {tgt_mod: target}

            # Contrastive loss for cross-modal pairs
            latents = None
            if src_mod != tgt_mod and config.contrastive_weight > 0:
                latents = {}
                src_enc = model.encode(src_mod, source)
                tgt_enc = model.encode(tgt_mod, target)
                latents[src_mod] = src_enc[:, 1:].mean(dim=1)
                latents[tgt_mod] = tgt_enc[:, 1:].mean(dim=1)

            loss_dict = criterion(predictions, targets_dict, latents)

        scaler.scale(loss_dict["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss_dict["total"].item()

        # Logging
        if (step + 1) % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            elapsed = time.time() - t0
            steps_per_sec = args.log_interval / elapsed
            print(
                f"  step {step + 1:>6d} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"{steps_per_sec:.1f} it/s | "
                f"phase={phase} task={src_mod}->{tgt_mod}"
            )
            running_loss = 0.0
            t0 = time.time()

        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            ckpt_path = save_dir / f"checkpoint_step{step + 1}.pt"
            state = {
                "step": step + 1,
                "config": config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            }
            torch.save(state, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    final_path = save_dir / "checkpoint_final.pt"
    state = {
        "step": args.total_steps,
        "config": config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(state, final_path)
    print(f"  Saved final checkpoint: {final_path}")
    print()
    print("Training complete.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
