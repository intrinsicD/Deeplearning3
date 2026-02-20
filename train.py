#!/usr/bin/env python3
"""OmniLatent training entry point.

Usage:
    python train.py                     # train with defaults (synthetic data)
    python train.py --steps 1000        # short run
    python train.py --no-amp            # disable mixed precision
    python train.py --dim 512 --layers 8  # smaller model
"""

from __future__ import annotations

import argparse
import sys

import torch

# --- ADDED: Hardware Optimizations ---
torch.backends.cudnn.benchmark = True           # Speeds up audio/video convolutions
torch.backends.cuda.matmul.allow_tf32 = True    # Speeds up matrix multiplications (Ampere+)
torch.backends.cudnn.allow_tf32 = True          # Speeds up cuDNN operations (Ampere+)
# -------------------------------------

from omnilatent.config import OmniLatentConfig
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.data import SyntheticMultiModalDataset, build_dataloader
from omnilatent.training.trainer import Trainer
from omnilatent.utils import count_parameters, param_size_mb, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train OmniLatent")
    p.add_argument("--dim", type=int, default=768, help="Hidden dimension")
    p.add_argument("--layers", type=int, default=12, help="Number of transformer layers")
    p.add_argument("--heads", type=int, default=12, help="Number of attention heads")
    p.add_argument("--steps", type=int, default=100_000, help="Max training steps")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--no-checkpoint", action="store_true", help="Disable gradient checkpointing")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--data-length", type=int, default=10_000, help="Synthetic dataset size")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = OmniLatentConfig(
        hidden_dim=args.dim,
        num_layers=args.layers,
        num_heads=args.heads,
        max_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mixed_precision=not args.no_amp,
        gradient_checkpointing=not args.no_checkpoint,
        seed=args.seed,
    )

    print("=" * 60)
    print("OmniLatent — All-to-All Multimodal AI")
    print("=" * 60)
    print(f"Device:      {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name()}")
        mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"VRAM:        {mem:.1f} GB")
    print(f"Hidden dim:  {config.hidden_dim}")
    print(f"Layers:      {config.num_layers}")
    print(f"Heads:       {config.num_heads}")
    print(f"MLP dim:     {config.mlp_dim}")
    print(f"Mixed prec:  {config.mixed_precision}")
    print(f"Grad ckpt:   {config.gradient_checkpointing}")

    model = OmniLatentModel(config)

    # --- ADDED: Free JIT Compilation Speedup ---
    print("Compiling model with torch.compile... (this takes a minute on startup)")
    model = torch.compile(model)
    # -------------------------------------------

    n_params = count_parameters(model)
    print(f"Parameters:  {n_params:,} ({param_size_mb(model):.1f} MB in FP32)")
    print("=" * 60)

    # Synthetic data for demonstration
    dataset = SyntheticMultiModalDataset(config, length=args.data_length)
    dataloader = build_dataloader(config, dataset)

    trainer = Trainer(model, config, dataloader)
    trainer.train(log_interval=args.log_interval)


if __name__ == "__main__":
    main()
