#!/usr/bin/env python3
"""Model component benchmarking and diagnostic report.

Runs a comprehensive analysis of the OmniLatent model to identify what needs
to change for real improvements in quality and performance.

Produces a structured report covering:
  1. Component parameter budgets
  2. Per-component forward/backward timing
  3. Encoder information retention (effective rank, norm stats)
  4. Backbone per-layer contribution analysis
  5. Decoder reconstruction fidelity
  6. Hook marginal contribution
  7. Gradient health (per-component, per-layer)
  8. Scaling sensitivity (depth, width sweeps)
  9. Cross-modal latent alignment
  10. Loss attribution per modality

Usage:
    # Quick report with small model
    python benchmark.py

    # Report with custom config
    python benchmark.py --dim 768 --layers 12 --heads 12

    # Report on a trained checkpoint
    python benchmark.py --checkpoint checkpoints/checkpoint_final.pt

    # Save report to file
    python benchmark.py --output report.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnilatent.config import OmniLatentConfig
from omnilatent.model.backbone import UnifiedTransformer
from omnilatent.model.hooks import LatentNeuralHook
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.model.temporal import RecurrentMemory, TemporalSequenceTransformer
from omnilatent.training.losses import MultiModalLoss
from omnilatent.utils import ALL_MODALITIES, count_parameters, param_size_mb


# -------------------------------------------------------------------------
# Report data structures
# -------------------------------------------------------------------------
@dataclass
class ComponentReport:
    name: str
    params: int = 0
    params_mb: float = 0.0
    params_pct: float = 0.0
    forward_ms: float = 0.0
    backward_ms: float = 0.0
    grad_norm: float = 0.0


@dataclass
class EncoderReport:
    modality: str
    effective_rank: int = 0
    mean_norm: float = 0.0
    std_norm: float = 0.0
    forward_ms: float = 0.0


@dataclass
class BackboneLayerReport:
    layer_idx: int
    delta_norm: float = 0.0
    grad_norm: float = 0.0


@dataclass
class ScalingPoint:
    dimension: str
    value: int
    total_params: int = 0
    forward_ms: float = 0.0
    loss: float = 0.0
    effective_rank: int = 0


@dataclass
class DiagnosticReport:
    config: dict = field(default_factory=dict)
    total_params: int = 0
    total_mb: float = 0.0
    components: list[ComponentReport] = field(default_factory=list)
    encoders: list[EncoderReport] = field(default_factory=list)
    backbone_layers: list[BackboneLayerReport] = field(default_factory=list)
    scaling: list[ScalingPoint] = field(default_factory=list)
    cross_modal_alignment: dict[str, float] = field(default_factory=dict)
    loss_magnitudes: dict[str, float] = field(default_factory=dict)
    loss_gradient_norms: dict[str, float] = field(default_factory=dict)
    hook_impact: float = 0.0
    recommendations: list[str] = field(default_factory=list)


# -------------------------------------------------------------------------
# Benchmark functions
# -------------------------------------------------------------------------

def _make_data(config: OmniLatentConfig, B: int = 2) -> dict[str, torch.Tensor]:
    return {
        "text": torch.randint(1, config.vocab_size, (B, 16)),
        "audio": torch.randn(B, config.audio_n_mels, 64),
        "image": torch.randn(B, 3, config.image_size, config.image_size),
        "video": torch.randn(
            B, 3, config.video_max_frames, config.video_size, config.video_size
        ),
    }


def _effective_rank(z: torch.Tensor, dim: int) -> int:
    flat = z.reshape(-1, dim)
    if flat.shape[0] < 2:
        return dim
    _, s, _ = torch.svd(flat.float())
    threshold = s[0] * 0.01
    return int((s > threshold).sum().item())


def _time_fn(fn, n_iters: int = 10) -> float:
    """Time a function in milliseconds (avg over n_iters)."""
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters * 1000


def benchmark_components(model: OmniLatentModel, config: OmniLatentConfig) -> list[ComponentReport]:
    """Parameter counts and timing for each major component."""
    total = count_parameters(model)
    data = _make_data(config, B=2)
    reports = []

    component_map = {
        "text_encoder": model.encoders["text"],
        "audio_encoder": model.encoders["audio"],
        "image_encoder": model.encoders["image"],
        "video_encoder": model.encoders["video"],
        "backbone": model.backbone,
        "text_decoder": model.decoders["text"],
        "audio_decoder": model.decoders["audio"],
        "image_decoder": model.decoders["image"],
        "video_decoder": model.decoders["video"],
        "target_query_gen": model.target_query_gen,
    }

    for name, component in component_map.items():
        p = count_parameters(component)
        r = ComponentReport(
            name=name,
            params=p,
            params_mb=p * 4 / (1024 ** 2),
            params_pct=100.0 * p / total if total > 0 else 0.0,
        )
        reports.append(r)

    # Time forward pass for the full model per modality
    model.eval()
    for mod in ALL_MODALITIES:
        def _fwd():
            with torch.no_grad():
                model.reconstruct(mod, data[mod])
        ms = _time_fn(_fwd, n_iters=10)
        # Add timing to the encoder report
        for r in reports:
            if r.name == f"{mod}_encoder":
                def _enc(m=mod):
                    with torch.no_grad():
                        model.encode(m, data[m])
                r.forward_ms = _time_fn(_enc, n_iters=10)

    # Backbone timing
    enc = model.encode("image", data["image"])
    def _bb():
        with torch.no_grad():
            model.backbone(enc)
    bb_report = [r for r in reports if r.name == "backbone"][0]
    bb_report.forward_ms = _time_fn(_bb, n_iters=10)

    return reports


def benchmark_encoders(model: OmniLatentModel, config: OmniLatentConfig) -> list[EncoderReport]:
    """Encoder information quality: effective rank, norms."""
    model.eval()
    data = _make_data(config, B=4)
    reports = []

    for mod in ALL_MODALITIES:
        with torch.no_grad():
            z = model.encode(mod, data[mod])
        content = z[:, 1:]  # skip modality token
        norms = content.norm(dim=-1)
        rank = _effective_rank(content, config.hidden_dim)

        r = EncoderReport(
            modality=mod,
            effective_rank=rank,
            mean_norm=norms.mean().item(),
            std_norm=norms.std().item(),
        )
        reports.append(r)

    return reports


def benchmark_backbone_layers(model: OmniLatentModel, config: OmniLatentConfig) -> list[BackboneLayerReport]:
    """Per-layer contribution and gradient health."""
    data = _make_data(config, B=2)
    reports = []

    # Forward: measure per-layer representation change
    model.eval()
    with torch.no_grad():
        enc = model.encode("image", data["image"])
    current = enc.clone()
    with torch.no_grad():
        for i, layer in enumerate(model.backbone.layers):
            prev = current
            current = layer(current, model.backbone.rope_freqs)
            delta = (current - prev).norm() / (prev.norm() + 1e-10)
            reports.append(BackboneLayerReport(
                layer_idx=i,
                delta_norm=delta.item(),
            ))

    # Backward: measure per-layer gradient norms
    model.train()
    model.zero_grad()
    result = model.reconstruct("image", data["image"])
    loss = F.mse_loss(result["output"], data["image"])
    loss.backward()

    for i, layer in enumerate(model.backbone.layers):
        total_norm = 0.0
        n = 0
        for p in layer.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
                n += 1
        reports[i].grad_norm = total_norm ** 0.5 if n > 0 else 0.0

    return reports


def benchmark_scaling(config: OmniLatentConfig) -> list[ScalingPoint]:
    """Sweep depth and width to measure scaling behavior."""
    points = []
    base_data = None

    # Depth sweep
    for n_layers in [1, 2, 4]:
        cfg = OmniLatentConfig(
            hidden_dim=config.hidden_dim,
            num_layers=n_layers,
            num_heads=config.num_heads,
            gradient_checkpointing=False,
            vocab_size=config.vocab_size,
            image_size=config.image_size,
            image_patch_size=config.image_patch_size,
            video_size=config.video_size,
            video_patch_size=config.video_patch_size,
            video_temporal_patch=config.video_temporal_patch,
            video_max_frames=config.video_max_frames,
            audio_n_mels=config.audio_n_mels,
            audio_max_frames=config.audio_max_frames,
        )
        model = OmniLatentModel(cfg)
        model.eval()
        data = _make_data(cfg, B=2)

        # Forward time
        def _fwd():
            with torch.no_grad():
                model.reconstruct("image", data["image"])
        fwd_ms = _time_fn(_fwd, n_iters=5)

        # Initial loss
        with torch.no_grad():
            result = model.reconstruct("image", data["image"])
            init_loss = F.mse_loss(result["output"], data["image"]).item()

        # Effective rank
        with torch.no_grad():
            z = model.encode("image", data["image"])
        rank = _effective_rank(z[:, 1:], cfg.hidden_dim)

        points.append(ScalingPoint(
            dimension="num_layers",
            value=n_layers,
            total_params=count_parameters(model),
            forward_ms=fwd_ms,
            loss=init_loss,
            effective_rank=rank,
        ))

    # Width sweep
    for dim in [32, 64, 128]:
        n_heads = max(2, dim // 16)
        cfg = OmniLatentConfig(
            hidden_dim=dim,
            num_layers=2,
            num_heads=n_heads,
            gradient_checkpointing=False,
            vocab_size=config.vocab_size,
            image_size=config.image_size,
            image_patch_size=config.image_patch_size,
            video_size=config.video_size,
            video_patch_size=config.video_patch_size,
            video_temporal_patch=config.video_temporal_patch,
            video_max_frames=config.video_max_frames,
            audio_n_mels=config.audio_n_mels,
            audio_max_frames=config.audio_max_frames,
        )
        model = OmniLatentModel(cfg)
        model.eval()
        data = _make_data(cfg, B=2)

        def _fwd():
            with torch.no_grad():
                model.reconstruct("image", data["image"])
        fwd_ms = _time_fn(_fwd, n_iters=5)

        with torch.no_grad():
            result = model.reconstruct("image", data["image"])
            init_loss = F.mse_loss(result["output"], data["image"]).item()

        with torch.no_grad():
            z = model.encode("image", data["image"])
        rank = _effective_rank(z[:, 1:], dim)

        points.append(ScalingPoint(
            dimension="hidden_dim",
            value=dim,
            total_params=count_parameters(model),
            forward_ms=fwd_ms,
            loss=init_loss,
            effective_rank=rank,
        ))

    return points


def benchmark_cross_modal_alignment(
    model: OmniLatentModel, config: OmniLatentConfig
) -> dict[str, float]:
    """Cosine similarity between modality latent centroids."""
    model.eval()
    data = _make_data(config, B=4)

    latents: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for mod in ALL_MODALITIES:
            enc = model.encode(mod, data[mod])
            latents[mod] = F.normalize(enc[:, 1:].mean(dim=1), dim=-1)

    results = {}
    mods = list(ALL_MODALITIES)
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            sim = F.cosine_similarity(
                latents[mods[i]], latents[mods[j]], dim=-1
            ).mean().item()
            results[f"{mods[i]}_vs_{mods[j]}"] = round(sim, 4)
    return results


def benchmark_loss_attribution(
    model: OmniLatentModel, config: OmniLatentConfig
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-modality loss magnitudes and gradient norms."""
    criterion = MultiModalLoss(config)
    data = _make_data(config, B=2)
    loss_mags: dict[str, float] = {}
    grad_norms: dict[str, float] = {}

    for mod in ALL_MODALITIES:
        model.train()
        model.zero_grad()
        result = model.reconstruct(mod, data[mod])
        loss_dict = criterion(
            {mod: result["output"]},
            {mod: data[mod]},
        )
        loss_mags[mod] = loss_dict[mod].item()

        loss_dict["total"].backward()
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        grad_norms[mod] = round(total_norm ** 0.5, 4)

    return loss_mags, grad_norms


def benchmark_hook_impact(model: OmniLatentModel, config: OmniLatentConfig) -> float:
    """Measure the output difference with vs without hooks."""
    model.eval()
    data = _make_data(config, B=2)

    with torch.no_grad():
        out_no_hook = model.reconstruct("image", data["image"])["output"]

    hook = LatentNeuralHook(
        "bench", 4, config.hidden_dim,
        list(range(config.num_layers)),
        gate_bias_init=0.0,
    )
    model.register_hook(hook)

    with torch.no_grad():
        out_hook = model.reconstruct("image", data["image"])["output"]

    impact = (out_hook - out_no_hook).abs().mean().item()
    model.remove_hook("bench")
    return impact


def generate_recommendations(report: DiagnosticReport) -> list[str]:
    """Analyze the report and generate actionable recommendations."""
    recs = []

    # 1. Parameter budget balance
    bb = next((c for c in report.components if c.name == "backbone"), None)
    if bb and bb.params_pct < 30:
        recs.append(
            f"BACKBONE UNDERWEIGHT: Backbone has only {bb.params_pct:.0f}% of params. "
            f"Consider increasing depth (num_layers) or width (hidden_dim) — "
            f"the backbone is the core compute and should hold 40-70% of params."
        )

    # 2. Encoder quality
    for enc in report.encoders:
        dim = report.config.get("hidden_dim", 64)
        if enc.effective_rank < dim * 0.3:
            recs.append(
                f"LOW-RANK ENCODER ({enc.modality}): Effective rank {enc.effective_rank} "
                f"is <30% of hidden_dim={dim}. This encoder may be collapsing "
                f"representations. Consider adding more capacity or changing the "
                f"architecture (deeper conv stack, more channels)."
            )

    # 3. Layer contribution
    if report.backbone_layers:
        deltas = [l.delta_norm for l in report.backbone_layers]
        min_delta = min(deltas)
        min_idx = deltas.index(min_delta)
        if min_delta < 0.001:
            recs.append(
                f"LAZY LAYER: Backbone layer {min_idx} contributes delta_norm="
                f"{min_delta:.6f} — nearly identity. This layer may be wasted. "
                f"Consider removing it or investigating why it's not learning."
            )

    # 4. Gradient health
    if report.backbone_layers:
        grad_norms = [l.grad_norm for l in report.backbone_layers if l.grad_norm > 0]
        if grad_norms:
            ratio = max(grad_norms) / (min(grad_norms) + 1e-10)
            if ratio > 100:
                recs.append(
                    f"GRADIENT IMBALANCE: Layer gradient norms vary by {ratio:.0f}x. "
                    f"This suggests vanishing or exploding gradients. Consider "
                    f"adding/tuning gradient clipping, or switching normalization."
                )

    # 5. Loss balance
    if report.loss_magnitudes:
        vals = [v for v in report.loss_magnitudes.values() if v > 0]
        if vals:
            ratio = max(vals) / (min(vals) + 1e-10)
            if ratio > 100:
                dominant = max(report.loss_magnitudes, key=report.loss_magnitudes.get)
                recs.append(
                    f"LOSS DOMINATED by {dominant} ({ratio:.0f}x larger). "
                    f"Other modalities may be starved. Consider adjusting "
                    f"modality_loss_weights or the uncertainty weighting init."
                )

    # 6. Cross-modal alignment
    if report.cross_modal_alignment:
        avg_sim = sum(report.cross_modal_alignment.values()) / len(report.cross_modal_alignment)
        if avg_sim > 0.95:
            recs.append(
                "REPRESENTATION COLLAPSE: All modalities map to nearly the same "
                "point (avg cosine > 0.95). Encoders are not differentiating modalities. "
                "Consider stronger contrastive loss or modality-specific architecture."
            )

    # 7. Scaling
    depth_points = [s for s in report.scaling if s.dimension == "num_layers"]
    if len(depth_points) >= 2:
        depth_points.sort(key=lambda s: s.value)
        last = depth_points[-1]
        first = depth_points[0]
        speedup = last.forward_ms / (first.forward_ms + 1e-10)
        if speedup > 5 and last.loss >= first.loss * 0.95:
            recs.append(
                f"DIMINISHING RETURNS on depth: {last.value} layers is "
                f"{speedup:.1f}x slower than {first.value} layers with similar loss. "
                f"Width scaling may be more efficient."
            )

    if not recs:
        recs.append("No major issues detected. Model components appear balanced.")

    return recs


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def run_benchmark(config: OmniLatentConfig, checkpoint: str | None = None) -> DiagnosticReport:
    """Run the full benchmark suite and return a structured report."""
    print("=" * 70)
    print("OmniLatent Component Diagnostic Benchmark")
    print("=" * 70)

    if checkpoint:
        from evaluate import load_checkpoint
        model, config = load_checkpoint(checkpoint, torch.device("cpu"))
    else:
        model = OmniLatentModel(config)

    report = DiagnosticReport(
        config={
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "image_size": config.image_size,
            "image_patch_size": config.image_patch_size,
        },
        total_params=count_parameters(model),
        total_mb=param_size_mb(model),
    )

    print(f"\nModel: {report.total_params:,} params ({report.total_mb:.1f} MB)")
    print(f"Config: dim={config.hidden_dim}, layers={config.num_layers}, "
          f"heads={config.num_heads}")

    # 1. Component parameter budgets & timing
    print("\n--- Component Parameters & Timing ---")
    report.components = benchmark_components(model, config)
    for c in report.components:
        print(f"  {c.name:25s}  {c.params:>10,} ({c.params_pct:5.1f}%)  "
              f"fwd={c.forward_ms:.1f}ms")

    # 2. Encoder quality
    print("\n--- Encoder Information Quality ---")
    report.encoders = benchmark_encoders(model, config)
    for e in report.encoders:
        print(f"  {e.modality:>6s}  rank={e.effective_rank:3d}/{config.hidden_dim}  "
              f"norm={e.mean_norm:.3f} +/- {e.std_norm:.3f}")

    # 3. Backbone per-layer analysis
    print("\n--- Backbone Layer Analysis ---")
    report.backbone_layers = benchmark_backbone_layers(model, config)
    for bl in report.backbone_layers:
        print(f"  layer {bl.layer_idx}  delta_norm={bl.delta_norm:.4f}  "
              f"grad_norm={bl.grad_norm:.4f}")

    # 4. Scaling sensitivity
    print("\n--- Scaling Sensitivity ---")
    report.scaling = benchmark_scaling(config)
    for s in report.scaling:
        print(f"  {s.dimension}={s.value:4d}  params={s.total_params:>10,}  "
              f"fwd={s.forward_ms:.1f}ms  loss={s.loss:.4f}  rank={s.effective_rank}")

    # 5. Cross-modal alignment
    print("\n--- Cross-Modal Alignment ---")
    report.cross_modal_alignment = benchmark_cross_modal_alignment(model, config)
    for pair, sim in report.cross_modal_alignment.items():
        print(f"  {pair:25s}  cosine={sim:.4f}")

    # 6. Loss attribution
    print("\n--- Loss Attribution ---")
    loss_mags, loss_grads = benchmark_loss_attribution(model, config)
    report.loss_magnitudes = loss_mags
    report.loss_gradient_norms = loss_grads
    for mod in ALL_MODALITIES:
        print(f"  {mod:>6s}  loss={loss_mags.get(mod, 0):.4f}  "
              f"grad_norm={loss_grads.get(mod, 0):.4f}")

    # 7. Hook impact
    print("\n--- Hook Impact ---")
    report.hook_impact = benchmark_hook_impact(model, config)
    print(f"  Mean absolute output change with hooks: {report.hook_impact:.6f}")

    # 8. Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    report.recommendations = generate_recommendations(report)
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")

    print()
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OmniLatent component diagnostic benchmark")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    p.add_argument("--dim", type=int, default=64, help="Hidden dim (ignored with --checkpoint)")
    p.add_argument("--layers", type=int, default=2, help="Num layers")
    p.add_argument("--heads", type=int, default=4, help="Num heads")
    p.add_argument("--output", type=str, default=None, help="Save report as JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = OmniLatentConfig(
        hidden_dim=args.dim,
        num_layers=args.layers,
        num_heads=args.heads,
        gradient_checkpointing=False,
        image_size=64,
        image_patch_size=16,
        video_size=32,
        video_patch_size=8,
        video_temporal_patch=2,
        video_max_frames=4,
        audio_n_mels=32,
        audio_max_frames=64,
    )

    report = run_benchmark(config, args.checkpoint)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
