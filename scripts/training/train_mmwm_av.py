"""Train MMWM on a public audiovisual dataset.

Usage:
    python -m scripts.training.train_mmwm_av \
        --dataset vggsound \
        --data-dir ./data/vggsound \
        --config datasets/configs/mmwm_vggsound.yaml

    # or use defaults:
    python -m scripts.training.train_mmwm_av --dataset audioset --data-dir ./data/audioset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ── Project root on sys.path ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets import build_av_dataset
from datasets.adapters.mmwm_adapter import MMWMAdapter
from MMWM.config import ModelConfig, build_model
from MMWM.losses import WorldModelLoss
from MMWM.trainer import Trainer, build_lr_scheduler


def _load_yaml(path: str | None) -> dict:
    if path is None:
        return {}
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _default_cfg(dataset_name: str) -> str:
    cfg_dir = _ROOT / "datasets" / "configs"
    path = cfg_dir / f"mmwm_{dataset_name}.yaml"
    if path.exists():
        return str(path)
    # fall back to vggsound defaults
    fallback = cfg_dir / "mmwm_vggsound.yaml"
    return str(fallback) if fallback.exists() else ""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train MMWM on a public AV dataset")
    parser.add_argument("--dataset",   required=True, help="Dataset name, e.g. vggsound")
    parser.add_argument("--data-dir",  required=True, help="Root directory for dataset files")
    parser.add_argument("--config",    default=None,  help="YAML config file (optional)")
    parser.add_argument("--resume",    default=None,  help="Checkpoint path to resume from")
    parser.add_argument("--steps",     type=int,      help="Override total training steps")
    parser.add_argument("--batch-size",type=int,      help="Override batch size")
    parser.add_argument("--download",  action="store_true", help="Download dataset before training")
    args = parser.parse_args(argv)

    # ── Config ──────────────────────────────────────────────────────────────
    cfg_path = args.config or _default_cfg(args.dataset)
    raw = _load_yaml(cfg_path)
    mcfg   = raw.get("model",    {})
    tcfg   = raw.get("training", {})
    dscfg  = raw.get("dataset",  {})
    adpcfg = raw.get("adapter",  {})

    if args.steps:      tcfg["total_steps"] = args.steps
    if args.batch_size: tcfg["batch_size"]  = args.batch_size

    batch_size  = int(tcfg.get("batch_size",  16))
    total_steps = int(tcfg.get("total_steps", 50000))
    lr          = float(tcfg.get("lr",        3e-4))
    run_dir     = tcfg.get("run_dir", f"runs/mmwm_{args.dataset}")
    mixed_prec  = bool(tcfg.get("mixed_precision", True))

    # ── Download ─────────────────────────────────────────────────────────────
    if args.download:
        from datasets import DATASET_REGISTRY
        cls = DATASET_REGISTRY[args.dataset]
        extra = {k: v for k, v in dscfg.items() if k not in ("name","split","n_frames","resolution","audio_sr","audio_duration_s")}
        cls.download(args.data_dir, **extra)

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds_kwargs = {k: v for k, v in dscfg.items() if k not in ("name", "split")}
    base_ds = build_av_dataset(
        args.dataset,
        data_dir=args.data_dir,
        split=dscfg.get("split", "train"),
        **ds_kwargs,
    )
    if len(base_ds) == 0:
        print(f"[ERROR] Dataset '{args.dataset}' is empty at {args.data_dir}.\n"
              "  Run with --download first.")
        sys.exit(1)

    # Build adapter
    adapter_kwargs = {
        "vector_dim":       int(mcfg.get("vector_dim",       16)),
        "action_dim":       int(mcfg.get("action_dim",        8)),
        "text_len":         int(mcfg.get("text_len",         32)),
        "vocab_size":       int(mcfg.get("vocab_size",      256)),
        "image_size":       int(mcfg.get("image_size",       64)),
        "audio_mel_bins":   int(mcfg.get("audio_mel_bins",   80)),
        "audio_mel_frames": int(mcfg.get("audio_mel_frames",128)),
        "include_text":     bool(adpcfg.get("include_text",  True)),
        "include_image":    bool(adpcfg.get("include_image", True)),
        "include_audio":    bool(adpcfg.get("include_audio", True)),
    }
    train_ds = MMWMAdapter(base_ds, **adapter_kwargs)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    # Validation: small split of same dataset
    try:
        val_base = build_av_dataset(
            args.dataset, data_dir=args.data_dir,
            split="val", max_samples=min(200, len(base_ds) // 10 or 100),
            **ds_kwargs,
        )
    except Exception:
        val_base = base_ds
    val_ds = MMWMAdapter(val_base, **adapter_kwargs)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    print(f"[MMWM] Train: {len(train_ds)} samples,  Val: {len(val_ds)} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    vocab_size = int(mcfg.get("vocab_size", 256))
    image_size = int(mcfg.get("image_size", 64))
    hidden_dim = 256
    latent_dim = 128
    action_dim = int(mcfg.get("action_dim", 8))

    enc_kwargs = {
        "text_vocab_size":  vocab_size,
        "text_embed_dim":   hidden_dim,
        "vector_input_dim": int(mcfg.get("vector_dim", 16)),
        "image_channels":   3,
        "audio_channels":   1,
        "hidden_dim":       hidden_dim,
    }
    decoder_configs = []
    if include_text:
        decoder_configs.append(("text_autoregressive_head", {
            "vocab_size": vocab_size, "latent_dim": latent_dim,
            "text_embed_dim": hidden_dim, "hidden_dim": hidden_dim,
        }))
    if include_image:
        decoder_configs.append(("image_reconstruction", {
            "latent_dim": latent_dim * 4,
            "out_channels": 3, "out_size": image_size,
        }))

    model_cfg = ModelConfig(
        encoder_kwargs        = enc_kwargs,
        action_encoder_kwargs = {"action_dim": action_dim, "action_embed_dim": latent_dim},
        decoder_configs       = decoder_configs,
    )

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = build_model(model_cfg, skip_validation=True)
    loss_fn   = WorldModelLoss(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Convert total_steps → epochs
    steps_per_epoch = max(1, len(train_loader))
    epochs    = max(1, (total_steps + steps_per_epoch - 1) // steps_per_epoch)
    scheduler = build_lr_scheduler(optimizer, total_steps)
    eval_every = int(tcfg.get("eval_every", 500))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        run_dir=run_dir,
        grad_clip_norm=float(tcfg.get("grad_clip", 1.0)),
        mixed_precision=mixed_prec,
        lr_scheduler=scheduler,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(
        train_loader,
        eval_loader      = val_loader,
        epochs           = epochs,
        eval_every_steps = eval_every,
    )


if __name__ == "__main__":
    main()

