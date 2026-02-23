#!/usr/bin/env python3
"""Predictive Coding training entry point for OmniLatent.

Trains the OmniLatent backbone using Predictive Coding (Whittington &
Bogacz 2017) instead of standard backpropagation.  Each transformer
layer becomes a level in a predictive hierarchy, with local Hebbian-like
weight updates driven by prediction error minimization.

Usage:
    python train_pc.py                          # train with defaults
    python train_pc.py --steps 1000             # short run
    python train_pc.py --inference-steps 50     # more inference iterations
    python train_pc.py --blend 0.5              # hybrid PC + backprop
    python train_pc.py --dim 256 --layers 4     # smaller model
"""

from __future__ import annotations

import argparse

import torch

# Hardware optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from omnilatent.config import OmniLatentConfig
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.data import SyntheticMultiModalDataset, build_dataloader
from omnilatent.training.predictive_coding import PCConfig, PCTrainer
from omnilatent.utils import count_parameters, param_size_mb, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train OmniLatent with Predictive Coding"
    )

    # Model architecture
    p.add_argument("--dim", type=int, default=768, help="Hidden dimension")
    p.add_argument("--layers", type=int, default=12, help="Transformer layers")
    p.add_argument("--heads", type=int, default=12, help="Attention heads")

    # PC-specific
    p.add_argument(
        "--inference-steps", type=int, default=20,
        help="Number of inference iterations per step (T_infer)",
    )
    p.add_argument(
        "--inference-lr", type=float, default=0.1,
        help="Learning rate for value node updates during inference",
    )
    p.add_argument(
        "--blend", type=float, default=0.0,
        help="Backprop blend ratio: 0.0=pure PC, 1.0=pure backprop",
    )
    p.add_argument(
        "--supervised-weight", type=float, default=1.0,
        help="Weight for supervised reconstruction loss",
    )

    # Training
    p.add_argument("--steps", type=int, default=100_000, help="Max steps")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--no-checkpoint", action="store_true",
                    help="Disable gradient checkpointing")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--data-length", type=int, default=10_000,
                    help="Synthetic dataset size")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Model config
    model_config = OmniLatentConfig(
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

    # PC config
    pc_config = PCConfig(
        inference_steps=args.inference_steps,
        inference_lr=args.inference_lr,
        learning_lr=args.lr,
        backprop_blend=args.blend,
        supervised_weight=args.supervised_weight,
        max_steps=args.steps,
        batch_size=args.batch_size,
        mixed_precision=not args.no_amp,
        seed=args.seed,
    )

    print("=" * 60)
    print("OmniLatent — Predictive Coding Training")
    print("=" * 60)
    print(f"Device:          {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU:             {torch.cuda.get_device_name()}")
        mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"VRAM:            {mem:.1f} GB")
    print(f"Hidden dim:      {model_config.hidden_dim}")
    print(f"Layers:          {model_config.num_layers}")
    print(f"Heads:           {model_config.num_heads}")
    print(f"MLP dim:         {model_config.mlp_dim}")
    print(f"Mixed prec:      {model_config.mixed_precision}")
    print(f"Inference steps: {pc_config.inference_steps}")
    print(f"Inference LR:    {pc_config.inference_lr}")
    print(f"Learning LR:     {pc_config.learning_lr}")
    print(f"Backprop blend:  {pc_config.backprop_blend}")
    print(f"Supervised wt:   {pc_config.supervised_weight}")

    model = OmniLatentModel(model_config)

    n_params = count_parameters(model)
    print(f"Parameters:      {n_params:,} ({param_size_mb(model):.1f} MB FP32)")
    print("=" * 60)

    # Synthetic data
    dataset = SyntheticMultiModalDataset(model_config, length=args.data_length)
    dataloader = build_dataloader(model_config, dataset)

    trainer = PCTrainer(
        model=model,
        model_config=model_config,
        pc_config=pc_config,
        dataloader=dataloader,
        seed=args.seed,
    )
    trainer.train(log_interval=args.log_interval)


if __name__ == "__main__":
    main()
