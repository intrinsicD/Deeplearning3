"""Train the Gaussian autoencoder on MNIST.

Usage::

    python -m gaussian_encoder.train            # defaults
    python -m gaussian_encoder.train --epochs 10 --latent-dim 16 --lr 3e-3

Requires ``torchvision`` (listed as an optional dep in the repo).
Reconstructed samples are saved to ``gaussian_encoder/samples.png`` after
training so you can visually verify that the Gaussian kernels learned useful
features.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import GaussianAutoencoder


def save_samples(
    model: GaussianAutoencoder,
    loader: DataLoader,
    path: Path,
    n: int = 8,
) -> None:
    """Save a side-by-side grid of originals and reconstructions."""
    from torchvision.utils import save_image

    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(next(model.parameters()).device)
    with torch.no_grad():
        x_hat, _ = model(x)
    # Top row: originals, bottom row: reconstructions
    save_image(torch.cat([x, x_hat]), str(path), nrow=n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gaussian autoencoder")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tf)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
    )

    # --- Model ---
    model = GaussianAutoencoder(in_ch=1, latent_dim=args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.enc_conv.parameters())
    print(f"Parameters  — total: {n_params:,}  encoder-conv: {n_enc:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # --- Train ---
    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for i, (x, _) in enumerate(train_loader, 1):
            x = x.to(device)
            x_hat, _ = model(x)
            loss = loss_fn(x_hat, x)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += loss.item()

        avg = running / i
        print(f"Epoch {epoch}/{args.epochs}  loss={avg:.5f}")

    # --- Save samples ---
    out = Path(__file__).resolve().parent / "samples.png"
    save_samples(model, train_loader, out)
    print(f"Saved reconstruction samples → {out}")

    # --- Print learned Gaussian stats (first layer) ---
    k = model.enc_conv[0].kernel  # GaussianKernel of the first GaussianConv2d
    sigma = k.log_sigma.exp().detach().cpu()
    print(
        f"First-layer Gaussian σ range: "
        f"[{sigma.min():.3f}, {sigma.max():.3f}]  "
        f"mean={sigma.mean():.3f}",
    )


if __name__ == "__main__":
    main()
