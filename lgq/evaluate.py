"""Evaluation script for comparing quantization schemes.

Loads trained models and computes comprehensive metrics for reproducible
comparison across LGQ, FSQ, and SimVQ quantizers.

Usage:
  python -m lgq.evaluate --checkpoint checkpoints/lgq/best.pt
  python -m lgq.evaluate --compare lgq:path1.pt fsq:path2.pt simvq:path3.pt
"""

import argparse
import json
import os

import torch
import torch.nn as nn

from lgq.config import LGQConfig
from lgq.model import LGQVAE
from lgq.metrics import (
    MetricAggregator,
    InceptionFeatureExtractor,
    LPIPSProxy,
    psnr,
    ssim,
)
from lgq.train import create_dataloader


@torch.no_grad()
def evaluate_model(
    model: LGQVAE,
    config: LGQConfig,
    device: torch.device,
    max_batches: int = 100,
) -> dict[str, float]:
    """Comprehensive evaluation of a single model.

    Returns dict with PSNR, SSIM, LPIPS, rFID, codebook utilization, etc.
    """
    model.eval()
    loader = create_dataloader(config, "val")

    # Metric modules
    feature_extractor = InceptionFeatureExtractor().to(device)
    feature_extractor.eval()
    lpips_model = LPIPSProxy().to(device)
    lpips_model.eval()

    aggregator = MetricAggregator(config.n_codebooks, config.vocab_size)
    total_lpips = 0.0
    n_batches = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        x = batch.to(device)
        out = model(x)
        recon = out["recon"].clamp(0, 1)
        indices = out["indices"]

        # Features for FID
        real_feats = feature_extractor(x)
        fake_feats = feature_extractor(recon)

        aggregator.update(recon, x, indices, real_feats, fake_feats)

        # LPIPS
        lpips_val = lpips_model(recon, x)
        total_lpips += lpips_val.item()
        n_batches += 1

    metrics = aggregator.compute()
    metrics["lpips"] = total_lpips / max(1, n_batches)

    # Effective codebook size (how many codes are actually used)
    if hasattr(model.quantizer, "codebook_utilization"):
        cb_metrics = model.quantizer.codebook_utilization()
        for k, v in cb_metrics.items():
            metrics[f"quantizer_{k}"] = v

    # Effective bitrate
    active_codes = metrics.get("active_codes", config.vocab_size)
    import math
    bits_per_token = math.log2(max(1, active_codes))
    tokens_per_image = config.n_latent_tokens * config.n_codebooks
    total_bits = bits_per_token * tokens_per_image
    pixels = config.resolution ** 2 * config.in_channels * 8
    metrics["bits_per_pixel"] = total_bits / (config.resolution ** 2)
    metrics["compression_ratio"] = pixels / max(1, total_bits)
    metrics["effective_vocab_bits"] = bits_per_token

    return metrics


def load_model(checkpoint_path: str, device: torch.device) -> tuple[LGQVAE, LGQConfig]:
    """Load a trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", LGQConfig())
    model = LGQVAE(config).to(device)
    model.load_state_dict(ckpt["model"])
    return model, config


def compare_models(
    model_specs: list[tuple[str, str]],
    device: torch.device,
    max_batches: int = 100,
) -> dict[str, dict[str, float]]:
    """Compare multiple models.

    Args:
        model_specs: list of (name, checkpoint_path)

    Returns:
        dict mapping model name to metrics.
    """
    results = {}
    for name, path in model_specs:
        print(f"\nEvaluating {name} from {path}...")
        model, config = load_model(path, device)
        metrics = evaluate_model(model, config, device, max_batches)
        results[name] = metrics
        print(f"  {name}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    return results


def print_comparison_table(results: dict[str, dict[str, float]]) -> None:
    """Print formatted comparison table."""
    if not results:
        return

    # Collect all metric names
    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())

    key_metrics = [
        "psnr", "ssim", "lpips", "rfid",
        "active_ratio", "utilization_pct", "perplexity",
        "bits_per_pixel", "compression_ratio",
    ]
    metrics_to_show = [m for m in key_metrics if m in all_metrics]

    # Header
    name_width = max(len(n) for n in results) + 2
    col_width = 12
    header = f"{'Model':<{name_width}}" + "".join(
        f"{m:>{col_width}}" for m in metrics_to_show
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for name, metrics in results.items():
        row = f"{name:<{name_width}}"
        for m in metrics_to_show:
            val = metrics.get(m, float("nan"))
            row += f"{val:>{col_width}.4f}"
        print(row)

    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Evaluate LGQ models")
    parser.add_argument("--checkpoint", type=str, help="Single model checkpoint")
    parser.add_argument("--compare", nargs="+",
                        help="Compare models: name:path name:path ...")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.compare:
        specs = []
        for spec in args.compare:
            name, path = spec.split(":", 1)
            specs.append((name, path))
        results = compare_models(specs, device, args.max_batches)
        print_comparison_table(results)
    elif args.checkpoint:
        model, config = load_model(args.checkpoint, device)
        metrics = evaluate_model(model, config, device, args.max_batches)
        print(f"\nMetrics:")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.6f}")
        results = {"model": metrics}
    else:
        parser.error("Provide --checkpoint or --compare")
        return

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
