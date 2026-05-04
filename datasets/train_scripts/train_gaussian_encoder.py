"""Train Gaussian Encoder on public video frame datasets.

The Gaussian Encoder was originally trained on MNIST.  This script replaces
the MNIST DataLoader with frames extracted from any registered AV dataset,
grayscale-converted and resized to 28 × 28.

Usage:
    python -m datasets.train_scripts.train_gaussian_encoder \
        --dataset vggsound \
        --data-dir ./data/vggsound \
        --epochs 50

    python -m datasets.train_scripts.train_gaussian_encoder \
        --dataset ucf101 --data-dir ./data/ucf101 --resolution 64 --no-grayscale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets import build_av_dataset
from datasets.adapters.gaussian_encoder_adapter import GaussianEncoderAdapter
from gaussian_encoder.model import GaussianAutoencoder


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train Gaussian Encoder on a public dataset")
    parser.add_argument("--dataset",      required=True)
    parser.add_argument("--data-dir",     required=True)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--latent-dim",   type=int,   default=32)
    parser.add_argument("--resolution",   type=int,   default=28)
    parser.add_argument("--no-grayscale", action="store_true")
    parser.add_argument("--n-frames",     type=int,   default=8)
    parser.add_argument("--download",     action="store_true")
    parser.add_argument("--run-dir",      default=None)
    args = parser.parse_args(argv)

    grayscale = not args.no_grayscale
    in_ch     = 1 if grayscale else 3
    run_dir   = args.run_dir or f"runs/gaussian_encoder_{args.dataset}"

    # ── Download ──────────────────────────────────────────────────────────────
    if args.download:
        from datasets import DATASET_REGISTRY
        DATASET_REGISTRY[args.dataset].download(args.data_dir)

    # ── Dataset ──────────────────────────────────────────────────────────────
    base_ds = build_av_dataset(
        args.dataset, data_dir=args.data_dir,
        split="train", n_frames=args.n_frames, resolution=args.resolution,
    )
    if len(base_ds) == 0:
        print(f"[ERROR] Dataset '{args.dataset}' is empty. Use --download first.")
        sys.exit(1)

    train_ds = GaussianEncoderAdapter(base_ds, resolution=args.resolution, grayscale=grayscale)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    print(f"[GaussianEncoder] {len(train_ds)} training images  ({in_ch}ch × {args.resolution}²)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GaussianEncoder] Device: {device}")

    model     = GaussianAutoencoder(in_ch=in_ch, latent_dim=args.latent_dim).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"[GaussianEncoder] Parameters: {n_params:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn   = nn.MSELoss()

    # ── TensorBoard ──────────────────────────────────────────────────────────
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir)
    except ImportError:
        pass

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for x in train_loader:
            x = x.to(device)
            x_hat, _ = model(x)
            loss = loss_fn(x_hat, x)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running      += loss.item()
            n_batches    += 1
            global_step  += 1

            if writer and global_step % 200 == 0:
                writer.add_scalar("loss/train", loss.item(), global_step)

        avg = running / max(1, n_batches)
        print(f"  Epoch {epoch}/{args.epochs}  loss={avg:.5f}")

    # ── Save samples ──────────────────────────────────────────────────────────
    try:
        from torchvision.utils import save_image
        out = Path(run_dir) / "samples.png"
        model.eval()
        x_batch = next(iter(train_loader))[:8].to(device)
        with torch.no_grad():
            x_hat, _ = model(x_batch)
        save_image(torch.cat([x_batch.cpu(), x_hat.cpu()]), str(out), nrow=8)
        print(f"[GaussianEncoder] Saved samples → {out}")
    except Exception as e:
        print(f"[GaussianEncoder] Could not save sample images: {e}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()

