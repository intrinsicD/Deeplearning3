#!/usr/bin/env python3
"""Inference & evaluation for trained OmniLatent checkpoints.

Load a trained checkpoint and probe what the model learned:

    # Self-reconstruction quality (how well does it autoenccode?)
    python evaluate.py --checkpoint checkpoints/checkpoint_final.pt --mode reconstruct

    # Cross-modal translation (does video→audio produce meaningful output?)
    python evaluate.py --checkpoint checkpoints/checkpoint_final.pt --mode translate

    # Probe with a real file
    python evaluate.py --checkpoint checkpoints/checkpoint_final.pt \
        --input-file photo.jpg --source image --target text

    # Latent space analysis (are modalities aligned?)
    python evaluate.py --checkpoint checkpoints/checkpoint_final.pt --mode latent

    # Full evaluation suite
    python evaluate.py --checkpoint checkpoints/checkpoint_final.pt --mode all
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from omnilatent.config import OmniLatentConfig
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.utils import ALL_MODALITIES, count_parameters, param_size_mb


# -------------------------------------------------------------------------
# File loaders (image, audio, video → model-ready tensors)
# -------------------------------------------------------------------------
def load_image(path: str, config: OmniLatentConfig) -> torch.Tensor:
    """Load an image file → (1, C, H, W) float tensor."""
    from torchvision.io import read_image
    from torchvision.transforms.functional import resize

    img = read_image(path).float() / 255.0  # (C, H, W)
    img = resize(img, [config.image_size, config.image_size])
    if img.shape[0] == 1:
        img = img.expand(3, -1, -1)
    elif img.shape[0] == 4:
        img = img[:3]
    return img.unsqueeze(0)


def load_audio(path: str, config: OmniLatentConfig) -> torch.Tensor:
    """Load an audio file → (1, n_mels, T) float tensor."""
    import torchaudio
    from torchaudio.transforms import MelSpectrogram, Resample

    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != config.audio_sample_rate:
        waveform = Resample(sr, config.audio_sample_rate)(waveform)

    mel = MelSpectrogram(
        sample_rate=config.audio_sample_rate,
        n_mels=config.audio_n_mels,
        hop_length=config.audio_hop_length,
        n_fft=1024,
    )(waveform)
    mel = torch.log1p(mel)

    # Make divisible by 4 and truncate
    t = (mel.shape[-1] // 4) * 4
    mel = mel[..., :t]
    if mel.shape[-1] > config.audio_max_frames:
        mel = mel[..., :config.audio_max_frames]
    return mel  # already (1, n_mels, T)


def load_video(path: str, config: OmniLatentConfig) -> torch.Tensor:
    """Load a video file → (1, C, T, H, W) float tensor."""
    from torchvision.io import read_video

    video, _, _ = read_video(path, start_pts=0, end_pts=2.0, pts_unit="sec")
    # (T, H, W, C) → (T, C, H, W)
    frames = video.permute(0, 3, 1, 2).float() / 255.0
    T = frames.shape[0]
    target_t = config.video_max_frames

    if T >= target_t:
        indices = torch.linspace(0, T - 1, target_t).long()
        frames = frames[indices]
    else:
        pad = target_t - T
        frames = torch.cat([frames, frames[-1:].expand(pad, -1, -1, -1)], dim=0)

    frames = F.interpolate(
        frames, size=(config.video_size, config.video_size),
        mode="bilinear", align_corners=False,
    )
    return frames.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)


LOADERS = {
    "image": load_image,
    "audio": load_audio,
    "video": load_video,
}


def load_file(path: str, modality: str, config: OmniLatentConfig) -> torch.Tensor:
    """Load a file as the given modality."""
    if modality == "text":
        text = Path(path).read_text(encoding="utf-8").strip()
        encoded = text.encode("utf-8")[:config.text_max_len]
        ids = [(b % (config.vocab_size - 1)) + 1 for b in encoded]
        return torch.tensor([ids], dtype=torch.long)
    return LOADERS[modality](path, config)


# -------------------------------------------------------------------------
# Save outputs
# -------------------------------------------------------------------------
def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save (1, C, H, W) tensor as PNG."""
    from torchvision.utils import save_image as _save
    img = tensor.squeeze(0).clamp(0, 1)
    _save(img, path)
    print(f"  Saved image: {path}")


def save_audio_mel(tensor: torch.Tensor, path: str) -> None:
    """Save (1, n_mels, T) mel as a numpy file for inspection."""
    mel = tensor.squeeze(0).cpu().numpy()
    np.save(path, mel)
    print(f"  Saved mel spectrogram: {path} (shape {mel.shape})")


def save_text(tensor: torch.Tensor, path: str) -> None:
    """Save (1, T, V) logits as decoded text."""
    tokens = tensor.squeeze(0).argmax(dim=-1)  # (T,)
    # Convert token IDs back to bytes (inverse of simple tokenizer)
    byte_list = [t.item() for t in tokens if t.item() > 0]
    text = bytes(b % 256 for b in byte_list).decode("utf-8", errors="replace")
    Path(path).write_text(text, encoding="utf-8")
    print(f"  Saved text: {path}")
    print(f"  Content preview: {text[:200]!r}")


SAVERS = {
    "image": lambda t, p: save_image(t, p + ".png"),
    "audio": lambda t, p: save_audio_mel(t, p + "_mel.npy"),
    "text": lambda t, p: save_text(t, p + ".txt"),
    "video": lambda t, p: (
        np.save(p + "_video.npy", t.squeeze(0).cpu().numpy()),
        print(f"  Saved video tensor: {p}_video.npy (shape {t.squeeze(0).shape})"),
    ),
}


# -------------------------------------------------------------------------
# Evaluation modes
# -------------------------------------------------------------------------
def infer_config_from_state_dict(state_dict: dict) -> OmniLatentConfig:
    """Infer model config from checkpoint weight shapes."""
    # hidden_dim from backbone norm
    hidden_dim = state_dict["backbone.final_norm.weight"].shape[0]

    # num_layers from how many backbone.layers.N exist
    num_layers = 0
    while f"backbone.layers.{num_layers}.norm1.weight" in state_dict:
        num_layers += 1

    # num_heads: qk_norm.scale has shape (head_dim,), so num_heads = hidden_dim / head_dim
    head_dim = state_dict["backbone.layers.0.attn.qk_norm.scale"].shape[0]
    num_heads = hidden_dim // head_dim

    config = OmniLatentConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    print(f"  (Inferred config: dim={hidden_dim}, layers={num_layers}, heads={num_heads})")
    return config


def load_checkpoint(ckpt_path: str, device: torch.device) -> tuple[OmniLatentModel, OmniLatentConfig]:
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Try to recover config from checkpoint, otherwise infer from weights
    if "config" in ckpt:
        config = ckpt["config"]
    else:
        config = infer_config_from_state_dict(ckpt["model"])

    model = OmniLatentModel(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    step = ckpt.get("step", "?")
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Step:       {step}")
    print(f"  Parameters: {count_parameters(model):,} ({param_size_mb(model):.1f} MB)")
    print(f"  Config:     dim={config.hidden_dim}, layers={config.num_layers}")
    return model, config


def generate_test_data(config: OmniLatentConfig, device: torch.device) -> dict[str, torch.Tensor]:
    """Generate synthetic test tensors for all modalities."""
    B = 2
    return {
        "text": torch.randint(1, config.vocab_size, (B, 64), device=device),
        "audio": torch.randn(B, config.audio_n_mels, 256, device=device),
        "image": torch.randn(B, 3, config.image_size, config.image_size, device=device),
        "video": torch.randn(
            B, 3, config.video_max_frames, config.video_size, config.video_size,
            device=device,
        ),
    }


@torch.no_grad()
def eval_reconstruction(model: OmniLatentModel, config: OmniLatentConfig, device: torch.device) -> dict:
    """Measure self-reconstruction quality for each modality."""
    print("\n=== Self-Reconstruction Quality ===")
    data = generate_test_data(config, device)
    results = {}

    for mod in ALL_MODALITIES:
        result = model.reconstruct(mod, data[mod])
        output = result["output"]
        inp = data[mod]

        if mod == "text":
            # Accuracy: do predicted tokens match input?
            pred_tokens = output.argmax(dim=-1)
            T = min(pred_tokens.shape[1], inp.shape[1])
            acc = (pred_tokens[:, :T] == inp[:, :T]).float().mean().item()
            results[mod] = {"accuracy": acc}
            print(f"  {mod:>6s}: token accuracy = {acc:.4f}")
        else:
            # For continuous modalities: MSE and cosine similarity
            # Decoder may output a different size than the input,
            # so truncate both to the shorter length before comparing.
            flat_inp = inp.flatten(1)
            flat_out = output.flatten(1)
            min_len = min(flat_inp.shape[1], flat_out.shape[1])
            flat_inp = flat_inp[:, :min_len]
            flat_out = flat_out[:, :min_len]
            mse = F.mse_loss(flat_out, flat_inp).item()
            cos = F.cosine_similarity(flat_out, flat_inp, dim=-1).mean().item()
            results[mod] = {"mse": mse, "cosine_sim": cos}
            print(f"  {mod:>6s}: MSE = {mse:.6f}, cosine_sim = {cos:.4f}")

    return results


@torch.no_grad()
def eval_cross_modal(model: OmniLatentModel, config: OmniLatentConfig, device: torch.device) -> dict:
    """Test all cross-modal translation pairs."""
    print("\n=== Cross-Modal Translation ===")
    data = generate_test_data(config, device)
    results = {}

    pairs = [
        ("image", "text"),
        ("text", "image"),
        ("audio", "image"),
        ("image", "audio"),
        ("video", "audio"),
        ("audio", "video"),
        ("video", "text"),
        ("text", "video"),
    ]

    for src, tgt in pairs:
        # Pass target_data for teacher forcing (text targets need it)
        result = model(src, data[src], tgt, data.get(tgt))
        output = result["output"]

        # Check output is valid (no NaN/Inf, reasonable magnitude)
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        mean_abs = output.abs().mean().item()
        std = output.std().item()

        status = "OK" if (not has_nan and not has_inf and mean_abs < 1000) else "WARN"
        results[f"{src}_to_{tgt}"] = {
            "status": status,
            "mean_abs": mean_abs,
            "std": std,
            "shape": list(output.shape),
        }
        print(
            f"  {src:>6s} → {tgt:<6s}: {status}  "
            f"shape={list(output.shape)}  "
            f"mean_abs={mean_abs:.4f}  std={std:.4f}"
        )

    return results


@torch.no_grad()
def eval_latent_space(model: OmniLatentModel, config: OmniLatentConfig, device: torch.device) -> dict:
    """Analyze latent space alignment across modalities."""
    print("\n=== Latent Space Analysis ===")
    data = generate_test_data(config, device)

    # Get source-encoded latent representations for each modality
    latents = {}
    for mod in ALL_MODALITIES:
        enc = model.encode(mod, data[mod])
        # Mean-pool content tokens (skip modality indicator at position 0)
        lat = enc[:, 1:].mean(dim=1)
        latents[mod] = F.normalize(lat, dim=-1)

    # Compute cross-modal cosine similarities
    print("  Cosine similarity between modality latents (higher = more aligned):")
    results = {}
    mods = list(ALL_MODALITIES)
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            sim = F.cosine_similarity(latents[mods[i]], latents[mods[j]], dim=-1).mean().item()
            key = f"{mods[i]}_vs_{mods[j]}"
            results[key] = sim
            print(f"    {mods[i]:>6s} ↔ {mods[j]:<6s}: {sim:.4f}")

    # Latent norm statistics
    print("  Latent norms (should be similar across modalities):")
    for mod in ALL_MODALITIES:
        norm = latents[mod].norm(dim=-1).mean().item()
        print(f"    {mod:>6s}: {norm:.4f}")

    return results


@torch.no_grad()
def eval_file(
    model: OmniLatentModel,
    config: OmniLatentConfig,
    device: torch.device,
    input_file: str,
    source_mod: str,
    target_mod: str,
    output_dir: str,
) -> None:
    """Run inference on a real file."""
    print(f"\n=== Inference: {source_mod} → {target_mod} ===")
    print(f"  Input: {input_file}")

    data = load_file(input_file, source_mod, config)
    data = data.to(device)
    print(f"  Input shape: {list(data.shape)}")

    # Cross-modal translation
    if target_mod == "text":
        # Use autoregressive generation for text output
        generated = model.generate(source_mod, data, max_len=64)
        print(f"  Generated token IDs: {generated[0, :20].tolist()}")
    else:
        result = model(source_mod, data, target_mod)
        output = result["output"]
        print(f"  Output shape: {list(output.shape)}")
        print(f"  Output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}")

        # Save output
        os.makedirs(output_dir, exist_ok=True)
        stem = Path(input_file).stem
        out_path = os.path.join(output_dir, f"{stem}_{source_mod}_to_{target_mod}")
        SAVERS[target_mod](output.cpu(), out_path)

    # Also do self-reconstruction of the input (for non-text modalities)
    if source_mod != "text":
        recon = model.reconstruct(source_mod, data)
        recon_out = recon["output"]
        os.makedirs(output_dir, exist_ok=True)
        stem = Path(input_file).stem
        recon_path = os.path.join(output_dir, f"{stem}_recon_{source_mod}")
        SAVERS[source_mod](recon_out.cpu(), recon_path)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained OmniLatent checkpoint")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    p.add_argument("--mode", type=str, default="all",
                   choices=["reconstruct", "translate", "latent", "file", "all"],
                   help="Evaluation mode")
    p.add_argument("--input-file", type=str, default=None, help="Input file for --mode file")
    p.add_argument("--source", type=str, default=None, choices=ALL_MODALITIES,
                   help="Source modality for --mode file")
    p.add_argument("--target", type=str, default=None, choices=ALL_MODALITIES,
                   help="Target modality for --mode file")
    p.add_argument("--output-dir", type=str, default="eval_output", help="Output directory")
    p.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, config = load_checkpoint(args.checkpoint, device)
    all_results = {}

    if args.mode in ("reconstruct", "all"):
        all_results["reconstruction"] = eval_reconstruction(model, config, device)

    if args.mode in ("translate", "all"):
        all_results["cross_modal"] = eval_cross_modal(model, config, device)

    if args.mode in ("latent", "all"):
        all_results["latent_space"] = eval_latent_space(model, config, device)

    if args.mode == "file":
        if not all([args.input_file, args.source, args.target]):
            print("ERROR: --mode file requires --input-file, --source, and --target")
            return
        eval_file(model, config, device, args.input_file, args.source, args.target, args.output_dir)

    # Save results summary
    if all_results:
        os.makedirs(args.output_dir, exist_ok=True)
        summary_path = os.path.join(args.output_dir, "eval_results.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {summary_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
