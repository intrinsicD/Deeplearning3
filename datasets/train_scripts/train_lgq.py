"""Train LGQ on public video datasets (extracts frames as training images).

Usage:
    python -m datasets.train_scripts.train_lgq \
        --dataset ucf101 \
        --data-dir ./data/ucf101 \
        --config datasets/configs/lgq_ucf101.yaml

    python -m datasets.train_scripts.train_lgq \
        --dataset kinetics400 --data-dir ./data/kinetics400 \
        --quantizer lgq --steps 200000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets import build_av_dataset
from datasets.adapters.lgq_adapter import LGQAdapter
from lgq.config import LGQConfig
from lgq.train import train as lgq_train


def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _default_cfg(dataset_name: str) -> str:
    cfg_dir = _ROOT / "datasets" / "configs"
    path = cfg_dir / f"lgq_{dataset_name}.yaml"
    return str(path) if path.exists() else ""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train LGQ on a public video dataset")
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--data-dir",   required=True)
    parser.add_argument("--config",     default=None)
    parser.add_argument("--resume",     default=None)
    parser.add_argument("--steps",      type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--quantizer",  default="lgq",
                        choices=["lgq", "fsq", "simvq", "vq"])
    parser.add_argument("--download",   action="store_true")
    args = parser.parse_args(argv)

    raw  = _load_yaml(args.config or _default_cfg(args.dataset))
    mcfg = raw.get("model",    {})
    tcfg = raw.get("training", {})
    dscfg= raw.get("dataset",  {})

    if args.steps:      tcfg["steps"]       = args.steps
    if args.batch_size: tcfg["batch_size"]  = args.batch_size

    resolution  = int(dscfg.get("resolution") or mcfg.get("resolution") or 256)
    n_frames    = int(dscfg.get("n_frames",   16))
    batch_size  = int(tcfg.get("batch_size",  32))
    total_steps = int(tcfg.get("steps",       100000))
    lr          = float(tcfg.get("lr",        1e-4))
    run_dir     = tcfg.get("run_dir", f"runs/lgq_{args.dataset}")
    quantizer   = args.quantizer or mcfg.get("quantizer", "lgq")

    # ── Download ──────────────────────────────────────────────────────────────
    if args.download:
        from datasets import DATASET_REGISTRY
        DATASET_REGISTRY[args.dataset].download(args.data_dir)

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds_kwargs = {k: v for k, v in dscfg.items() if k not in ("name","split")}
    base_ds = build_av_dataset(
        args.dataset, data_dir=args.data_dir,
        split=dscfg.get("split","train"),
        n_frames=n_frames, resolution=resolution, **ds_kwargs,
    )
    if len(base_ds) == 0:
        print(f"[ERROR] Dataset '{args.dataset}' is empty. Use --download first.")
        sys.exit(1)

    train_ds = LGQAdapter(base_ds, resolution=resolution)
    print(f"[LGQ] Training frames: {len(train_ds)}")

    # LGQConfig uses vocab_size (= codebook entries), n_codebooks, and codebook_dim
    # Map yaml "codebook_size" -> vocab_size for convenience
    vocab_size = int(mcfg.get("codebook_size") or mcfg.get("vocab_size") or 256)
    config = LGQConfig(
        quantizer_type  = quantizer,
        total_steps     = total_steps,
        batch_size      = batch_size,
        resolution      = resolution,
        learning_rate   = lr,
        vocab_size      = vocab_size,
        checkpoint_dir  = f"checkpoints/lgq_{args.dataset}",
        log_dir         = run_dir,
    )

    # Override the dataloader inside lgq.train by monkey-patching create_dataloader
    import lgq.train as _lgq_train_mod

    _original_create = _lgq_train_mod.create_dataloader

    def _patched_create(cfg, split="train"):
        if split == "train":
            return DataLoader(
                train_ds, batch_size=cfg.batch_size, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True,
            )
        else:
            return DataLoader(
                train_ds, batch_size=cfg.batch_size, shuffle=False,
                num_workers=2, pin_memory=True,
            )

    _lgq_train_mod.create_dataloader = _patched_create
    try:
        lgq_train(config)
    finally:
        _lgq_train_mod.create_dataloader = _original_create


if __name__ == "__main__":
    main()

