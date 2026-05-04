#!/usr/bin/env python3
"""Test that training actually begins (no errors) for each model.

Runs a minimal number of steps per model using tiny configs and synthetic
data, then reports a pass/fail summary.  Avoids torch.compile (slow) and
internet access (no dataset downloads).

Usage:
    python test_training_startup.py
    python test_training_startup.py --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import torch

RESULTS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _ok(name: str, elapsed: float, notes: str = "") -> None:
    RESULTS[name] = {"status": "PASS", "elapsed": elapsed, "notes": notes}
    print(f"  [PASS] {name}  ({elapsed:.1f}s){('  ' + notes) if notes else ''}")


def _fail(name: str, elapsed: float, exc: Exception) -> None:
    tb = traceback.format_exc()
    RESULTS[name] = {"status": "FAIL", "elapsed": elapsed, "error": str(exc), "tb": tb}
    print(f"  [FAIL] {name}  ({elapsed:.1f}s)")
    print(f"         {exc}")


def _print_summary() -> int:
    """Print final summary table and return exit code (0=all pass, 1=any fail)."""
    _banner("SUMMARY")
    passed = sum(1 for r in RESULTS.values() if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS.values() if r["status"] == "FAIL")
    for name, r in RESULTS.items():
        status = r["status"]
        mark = "✓" if status == "PASS" else "✗"
        print(f"  {mark} {name:<40} {status}  {r['elapsed']:.1f}s")
        if status == "FAIL" and args.verbose:
            print(f"      Error: {r['error']}")
            print(r["tb"])
    print(f"\n  {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# 1. OmniLatent
# ---------------------------------------------------------------------------

def test_omnilatent(steps: int = 3) -> None:
    name = "OmniLatent (synthetic, tiny config)"
    _banner(name)
    t0 = time.time()
    try:
        from omnilatent.config import OmniLatentConfig
        from omnilatent.model.omnilatent import OmniLatentModel
        from omnilatent.training.data import SyntheticMultiModalDataset, build_dataloader
        from omnilatent.training.trainer import Trainer

        config = OmniLatentConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,       # 64/4=16 divisible
            batch_size=2,
            max_steps=steps,
            mixed_precision=False,   # avoid AMP issues on CPU
            gradient_checkpointing=False,
            seed=0,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        model = OmniLatentModel(config)
        # NOTE: skip torch.compile — it is optional and very slow on first run.

        dataset = SyntheticMultiModalDataset(config, length=max(steps * config.batch_size, 16))
        dataloader = build_dataloader(config, dataset)

        trainer = Trainer(model, config, dataloader)
        trainer.train(log_interval=1)

        _ok(name, time.time() - t0, f"completed {steps} steps")
    except Exception as exc:
        _fail(name, time.time() - t0, exc)


# ---------------------------------------------------------------------------
# 2. HPWM
# ---------------------------------------------------------------------------

def test_hpwm(steps: int = 2) -> None:
    name = "HPWM (synthetic moving shapes)"
    _banner(name)
    t0 = time.time()
    try:
        from hpwm.config import HPWMConfig
        from hpwm.data import SyntheticMovingShapes
        from hpwm.model import HPWM
        from torch.utils.data import DataLoader

        # Tiny config: very small model, fewer frames
        config = HPWMConfig(
            resolution=64,
            n_frames=4,             # 4 frames per clip (was 120)
            total_steps=steps,
            grad_accum_steps=1,     # no accumulation so 1 micro-step = 1 optimizer step
            batch_size=1,
            log_every=1,
            eval_every=9999,        # skip eval
            save_every=9999,        # skip save
            precision="fp32",       # avoid bf16 on CPU
            d_mamba=32,
            mamba_n_layers=1,
            n_layers_fast=1,
            n_layers_slow=1,
            d_fast=32,
            d_slow=64,
            token_budget=64,
            n_heads=2,
            n_slots=4,
            d_slot=32,
            slot_mlp_hidden=64,
            vqvae_hidden=32,
            vqvae_n_layers=1,
            fwm_channels=32,
            fwm_layers=1,
            n_heavy_layers=1,
            d_heavy=32,
            n_patches=16,           # 4x4 patch grid for 64px / 14px ≈ 4 patches/side
            patch_grid=4,
            d_dino=384,             # keep matching DINOv2-S architecture
            warmup_steps=0,
            vqvae_warmup_steps=0,
            pred_warmup_steps=0,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")

        # Build tiny synthetic dataset
        train_ds = SyntheticMovingShapes(
            n_clips=16,
            n_frames=config.n_frames,
            resolution=config.resolution,
            n_objects=2,
            fps=config.fps,
        )
        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size,
            shuffle=True, num_workers=0, drop_last=True,
        )

        model = HPWM(config).to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        step = 0
        temporal_states = None
        for batch in train_loader:
            if step >= steps:
                break
            frames = batch["frames"].to(device)

            outputs = model(frames, temporal_states)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Detach temporal states
            temporal_states = [
                s.detach() if s is not None else None
                for s in outputs["temporal_states"]
            ]

            print(f"  step {step + 1}/{steps}  loss={loss.item():.4f}")
            step += 1

        assert step == steps, f"Expected {steps} steps, completed {step}"
        _ok(name, time.time() - t0, f"completed {steps} steps")
    except Exception as exc:
        _fail(name, time.time() - t0, exc)


# ---------------------------------------------------------------------------
# 3. LGQ (all 3 quantizer variants)
# ---------------------------------------------------------------------------

def test_lgq_variant(quantizer: str, steps: int = 3) -> None:
    name = f"LGQ (quantizer={quantizer})"
    _banner(name)
    t0 = time.time()
    try:
        from lgq.config import LGQConfig
        from lgq.model import LGQVAE
        from lgq.losses import LGQLoss
        from lgq.train import SyntheticImageDataset
        from torch.utils.data import DataLoader

        # Small resolution to keep it fast
        config = LGQConfig(
            quantizer_type=quantizer,
            resolution=32,
            batch_size=2,
            total_steps=steps,
            log_every=1,
            eval_every=9999,
            save_every=9999,
            disc_start_step=9999,   # skip discriminator during short test
            hidden_dim=32,
            n_res_blocks=1,
            n_codebooks=2,
            codebook_dim=4,
            vq_dim=8,               # must equal n_codebooks * codebook_dim = 2*4
            vocab_size=16,
            warmup_steps=0,
            precision="fp32",       # avoid bf16 on CPU
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")

        dataset = SyntheticImageDataset(size=32, resolution=config.resolution)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True)

        model = LGQVAE(config).to(device)
        loss_fn = LGQLoss(config)

        param_groups = model.get_param_groups()
        optimizer = torch.optim.AdamW(param_groups["generator"], lr=1e-4)

        data_iter = iter(loader)
        for step in range(1, steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            x = batch.to(device)
            model.train()
            out = model(x)
            gen_losses = loss_fn.generator_loss(x, out, adv_loss=None, step=step)
            gen_losses["total"].backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"  step {step}/{steps}  loss={gen_losses['total'].item():.4f}  "
                  f"recon={gen_losses.get('recon_loss', torch.tensor(0)).item():.4f}")

        _ok(name, time.time() - t0, f"completed {steps} steps")
    except Exception as exc:
        _fail(name, time.time() - t0, exc)


# ---------------------------------------------------------------------------
# 4. Gaussian Encoder
# ---------------------------------------------------------------------------

def test_gaussian_encoder(epochs: int = 1) -> None:
    name = "GaussianEncoder (synthetic MNIST-like)"
    _banner(name)
    t0 = time.time()
    try:
        from gaussian_encoder.model import GaussianAutoencoder
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")

        # Create tiny synthetic dataset mimicking MNIST (greyscale 28x28)
        n_samples = 64
        x_data = torch.rand(n_samples, 1, 28, 28)
        dataset = TensorDataset(x_data, torch.zeros(n_samples, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        model = GaussianAutoencoder(in_ch=1, latent_dim=16).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            running = 0.0
            n_batches = 0
            for x, _ in loader:
                x = x.to(device)
                x_hat, _ = model(x)
                loss = loss_fn(x_hat, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item()
                n_batches += 1
            avg = running / max(n_batches, 1)
            print(f"  epoch {epoch}/{epochs}  loss={avg:.5f}")

        _ok(name, time.time() - t0, f"completed {epochs} epoch(s)")
    except Exception as exc:
        _fail(name, time.time() - t0, exc)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test training startup for all models")
    p.add_argument("--verbose", action="store_true", help="Print full tracebacks for failures")
    p.add_argument("--steps", type=int, default=3, help="Steps for step-based trainers")
    p.add_argument("--epochs", type=int, default=1, help="Epochs for epoch-based trainers")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Training Startup Test")
    print(f"  PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # Run all tests
    test_omnilatent(steps=args.steps)
    test_hpwm(steps=args.steps)
    for qtype in ("lgq", "fsq", "simvq"):
        test_lgq_variant(qtype, steps=args.steps)
    test_gaussian_encoder(epochs=args.epochs)

    sys.exit(_print_summary())
