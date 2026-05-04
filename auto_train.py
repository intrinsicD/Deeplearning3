#!/usr/bin/env python3
"""auto_train.py – One-stop orchestrator for training any model on any public dataset.

Supported models:   mmwm | hpwm | lgq | gaussian_encoder
Supported datasets: ucf101 | hmdb51 | vggsound | audioset | kinetics400 | kinetics700
                    miradata | avspeech

Quick-start examples
--------------------
# List all available datasets:
    python auto_train.py --list-datasets

# Download VGGSound metadata only:
    python auto_train.py --model mmwm --dataset vggsound \
        --data-dir ./data/vggsound --download-only

# Train MMWM on AudioSet (downloads if needed):
    python auto_train.py --model mmwm --dataset audioset \
        --data-dir ./data/audioset --download --steps 50000

# Train HPWM on Kinetics-400 with a custom config:
    python auto_train.py --model hpwm --dataset kinetics400 \
        --data-dir ./data/kinetics400 --config my_config.yaml

# Train LGQ tokenizer on UCF-101:
    python auto_train.py --model lgq --dataset ucf101 \
        --data-dir ./data/ucf101 --download

# Train Gaussian Encoder on VGGSound frames:
    python auto_train.py --model gaussian_encoder --dataset vggsound \
        --data-dir ./data/vggsound
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# ── Model → training script mapping ──────────────────────────────────────────
MODEL_SCRIPTS = {
    "mmwm":              _ROOT / "datasets" / "train_scripts" / "train_mmwm.py",
    "hpwm":              _ROOT / "datasets" / "train_scripts" / "train_hpwm.py",
    "lgq":               _ROOT / "datasets" / "train_scripts" / "train_lgq.py",
    "gaussian_encoder":  _ROOT / "datasets" / "train_scripts" / "train_gaussian_encoder.py",
}

# ── Default configs per (model, dataset) ────────────────────────────────────
_CFG_DIR = _ROOT / "datasets" / "configs"
DEFAULT_CONFIGS = {
    # MMWM
    ("mmwm", "vggsound"):         _CFG_DIR / "mmwm_vggsound.yaml",
    ("mmwm", "audioset"):         _CFG_DIR / "mmwm_audioset.yaml",
    ("mmwm", "miradata"):         _CFG_DIR / "mmwm_miradata.yaml",
    ("mmwm", "avspeech"):         _CFG_DIR / "mmwm_avspeech.yaml",
    # HPWM
    ("hpwm", "kinetics400"):      _CFG_DIR / "hpwm_kinetics400.yaml",
    ("hpwm", "ucf101"):           _CFG_DIR / "hpwm_ucf101.yaml",
    # LGQ
    ("lgq", "ucf101"):            _CFG_DIR / "lgq_ucf101.yaml",
    ("lgq", "kinetics400"):       _CFG_DIR / "lgq_kinetics400.yaml",
    # Gaussian Encoder
    ("gaussian_encoder","vggsound"): _CFG_DIR / "gaussian_encoder_vggsound.yaml",
}


def _resolve_config(model: str, dataset: str, override: str | None) -> str | None:
    if override:
        return override
    key = (model, dataset)
    if key in DEFAULT_CONFIGS and DEFAULT_CONFIGS[key].exists():
        return str(DEFAULT_CONFIGS[key])
    return None


def _download_dataset(dataset: str, data_dir: str, extra_kwargs: dict) -> None:
    """Call the dataset's download() classmethod directly."""
    # Ensure datasets package is on the path
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from datasets import DATASET_REGISTRY, list_datasets
    if dataset not in DATASET_REGISTRY:
        print(f"[ERROR] Unknown dataset '{dataset}'.  Available: {', '.join(list_datasets())}")
        sys.exit(1)

    cls = DATASET_REGISTRY[dataset]
    print(f"\n{'='*60}")
    print(f"  Downloading dataset: {dataset}")
    print(f"  Target directory  : {data_dir}")
    print(f"{'='*60}\n")
    cls.download(data_dir, **extra_kwargs)


def build_subcommand(
    model: str,
    dataset: str,
    data_dir: str,
    config: str | None,
    resume: str | None,
    steps: int | None,
    batch_size: int | None,
    download: bool,
    extra: list[str],
) -> list[str]:
    """Build the subprocess command for the per-model training script."""
    script = MODEL_SCRIPTS[model]
    cmd = [sys.executable, str(script), "--dataset", dataset, "--data-dir", data_dir]

    cfg = _resolve_config(model, dataset, config)
    if cfg:
        cmd += ["--config", cfg]
    if resume:
        cmd += ["--resume", resume]
    if steps:
        cmd += ["--steps", str(steps)]
    if batch_size:
        cmd += ["--batch-size", str(batch_size)]
    if download:
        cmd += ["--download"]

    # HPWM-specific: enable synthetic fallback if real data absent
    if model == "hpwm":
        from datasets import build_av_dataset
        try:
            ds = build_av_dataset(dataset, data_dir=data_dir, auto_prepare=True)
            if len(ds) == 0:
                cmd += ["--synthetic-fallback"]
        except Exception:
            cmd += ["--synthetic-fallback"]

    cmd += extra
    return cmd


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="auto_train.py – Train any model on any public AV dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", choices=list(MODEL_SCRIPTS.keys()),
        help="Model to train",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset name (e.g. vggsound, audioset, kinetics400, ucf101, miradata, avspeech)",
    )
    parser.add_argument("--data-dir",   default="./data",      help="Root data directory")
    parser.add_argument("--config",     default=None,          help="Override YAML config file")
    parser.add_argument("--resume",     default=None,          help="Resume from checkpoint")
    parser.add_argument("--steps",      type=int, default=None,help="Override total training steps")
    parser.add_argument("--batch-size", type=int, default=None,help="Override batch size")
    parser.add_argument("--download",   action="store_true",   help="Download dataset before training")
    parser.add_argument("--download-only", action="store_true",help="Only download, do not train")
    parser.add_argument("--list-datasets", action="store_true",help="List all available datasets and exit")
    parser.add_argument("--list-models",   action="store_true",help="List all available models and exit")
    parser.add_argument("--max-clips",  type=int, default=None,
                        help="Limit number of clips to download (for testing)")
    parser.add_argument("--workers",    type=int, default=4,
                        help="Parallel workers for yt-dlp downloads")

    # Pass-through args for the sub-script
    args, extra = parser.parse_known_args(argv)

    # ── Informational commands ────────────────────────────────────────────────
    if args.list_models:
        print("Available models:")
        for m in sorted(MODEL_SCRIPTS.keys()):
            print(f"  {m}")
        return

    if args.list_datasets:
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        from datasets import list_datasets
        print("Available datasets:")
        for d in list_datasets():
            print(f"  {d}")
        return

    # ── Validate required args ────────────────────────────────────────────────
    if not args.model:
        parser.error("--model is required (or use --list-models to see options)")
    if not args.dataset:
        parser.error("--dataset is required (or use --list-datasets to see options)")
    if args.model not in MODEL_SCRIPTS:
        parser.error(f"Unknown model '{args.model}'")

    data_dir = str(Path(args.data_dir).expanduser().resolve())

    # ── Download ──────────────────────────────────────────────────────────────
    if args.download or args.download_only:
        dl_kwargs: dict = {"workers": args.workers}
        if args.max_clips:
            dl_kwargs["max_clips"] = args.max_clips
        _download_dataset(args.dataset, data_dir, dl_kwargs)

    if args.download_only:
        print("\n[auto_train] Download complete.  Re-run without --download-only to train.")
        return

    # ── Launch training subprocess ────────────────────────────────────────────
    cmd = build_subcommand(
        model=args.model,
        dataset=args.dataset,
        data_dir=data_dir,
        config=args.config,
        resume=args.resume,
        steps=args.steps,
        batch_size=args.batch_size,
        download=False,  # already handled above
        extra=extra,
    )

    print(f"\n{'='*60}")
    print(f"  Model  : {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Data   : {data_dir}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

