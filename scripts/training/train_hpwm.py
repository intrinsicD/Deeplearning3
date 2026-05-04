"""Train HPWM on a public video dataset.

Usage:
    python -m scripts.training.train_hpwm \
        --dataset kinetics400 \
        --data-dir ./data/kinetics400 \
        --config datasets/configs/hpwm_kinetics400.yaml

    python -m scripts.training.train_hpwm \
        --dataset ucf101 --data-dir ./data/ucf101 --synthetic-fallback
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
from datasets.adapters.hpwm_adapter import HPWMAdapter
from hpwm.config import HPWMConfig
from hpwm.train import Trainer as HPWMTrainer


def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _default_cfg(dataset_name: str) -> str:
    cfg_dir = _ROOT / "datasets" / "configs"
    path = cfg_dir / f"hpwm_{dataset_name}.yaml"
    return str(path) if path.exists() else ""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train HPWM on a public video dataset")
    parser.add_argument("--dataset",           required=True)
    parser.add_argument("--data-dir",          required=True)
    parser.add_argument("--config",            default=None)
    parser.add_argument("--resume",            default=None)
    parser.add_argument("--steps",             type=int)
    parser.add_argument("--batch-size",        type=int)
    parser.add_argument("--synthetic-fallback",action="store_true",
                        help="Use synthetic data if real dataset unavailable")
    parser.add_argument("--download",          action="store_true")
    args = parser.parse_args(argv)

    raw  = _load_yaml(args.config or _default_cfg(args.dataset))
    mcfg = raw.get("model",    {})
    tcfg = raw.get("training", {})
    dscfg= raw.get("dataset",  {})

    if args.steps:      tcfg["total_steps"] = args.steps
    if args.batch_size: mcfg["batch_size"]  = args.batch_size

    # Build HPWMConfig
    config = HPWMConfig()
    for k, v in mcfg.items():
        if hasattr(config, k):
            setattr(config, k, v)
    if tcfg.get("total_steps"):
        config.total_steps = int(tcfg["total_steps"])
    if tcfg.get("lr"):
        config.lr = float(tcfg["lr"])

    n_frames   = int(dscfg.get("n_frames",   config.n_frames))
    resolution = int(dscfg.get("resolution", config.resolution))

    # ── Download ──────────────────────────────────────────────────────────────
    if args.download:
        from datasets import DATASET_REGISTRY
        cls = DATASET_REGISTRY[args.dataset]
        cls.download(args.data_dir)

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds_kwargs = {k: v for k, v in dscfg.items() if k not in ("name","split")}
    base_ds = build_av_dataset(
        args.dataset, data_dir=args.data_dir,
        split=dscfg.get("split","train"), **ds_kwargs,
    )

    if len(base_ds) == 0:
        if args.synthetic_fallback:
            print("[HPWM] Dataset empty – falling back to SyntheticMovingShapes")
            trainer = HPWMTrainer(config=config, use_synthetic=True, resume_from=args.resume)
            trainer.train()
            return
        else:
            print(f"[ERROR] Dataset '{args.dataset}' is empty.  Use --download or --synthetic-fallback.")
            sys.exit(1)

    train_ds = HPWMAdapter(base_ds, resolution=resolution, n_frames=n_frames)

    # Validation split
    try:
        val_base = build_av_dataset(
            args.dataset, data_dir=args.data_dir,
            split="val", max_samples=min(200, max(10, len(base_ds)//10)),
            **ds_kwargs,
        )
        val_ds = HPWMAdapter(val_base, resolution=resolution, n_frames=n_frames)
    except Exception:
        val_ds = train_ds

    print(f"[HPWM] Train: {len(train_ds)},  Val: {len(val_ds)}")

    batch_size = int(mcfg.get("batch_size", config.batch_size))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Monkey-patch the trainer's dataloaders so we can reuse its train() loop
    trainer = HPWMTrainer(config=config, use_synthetic=True, resume_from=args.resume)
    trainer.train_loader = train_loader
    trainer.val_loader   = val_loader
    trainer.evaluator.val_loader = val_loader
    trainer.train()


if __name__ == "__main__":
    main()

