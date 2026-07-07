#!/usr/bin/env python
"""Train MMWM on a Minari offline-RL dataset.

This script converts Minari episodes into MMWM transition tuples:
    observation[t] + action[t] -> observation[t + 1]

It is intentionally vector-first. Observations/actions are flattened so it works
with Box observations and many dict observations. For image-rich datasets, write a
custom adapter once the vector baseline is stable.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from MMWM import (
    MinariTransitionDataset,
    ModelConfig,
    WorldModelLoss,
    build_lr_scheduler,
    build_model,
    collate_transition_batch,
)
from MMWM.trainer import Trainer


def _require_minari():
    try:
        import minari  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Minari is not installed. Install it with:\n"
            "  pip install minari\n"
            "or:\n"
            "  pip install -e '.[datasets]'"
        ) from exc
    return minari


def write_status(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(target)


def build_mmwm(vector_dim: int, action_dim: int, latent_dim: int, hidden_dim: int, transition: str):
    primary_dim = latent_dim * 4
    cfg = ModelConfig(
        encoder_name="simple_multimodal",
        encoder_kwargs={
            "text_vocab_size": 256,
            "text_embed_dim": min(128, hidden_dim),
            "vector_input_dim": vector_dim,
            "image_channels": 3,
            "audio_channels": 1,
            "hidden_dim": hidden_dim,
        },
        latent_projector_kwargs={"input_dim": hidden_dim, "latent_dim": latent_dim, "use_norm": True},
        memory_name="gru",
        memory_kwargs={"input_dim": latent_dim * 2, "hidden_dim": latent_dim},
        action_encoder_name="mlp",
        action_encoder_kwargs={"action_dim": action_dim, "action_embed_dim": latent_dim},
        conditioner_name="concat_mlp",
        conditioner_kwargs={
            "latent_dim": primary_dim,
            "action_dim": latent_dim,
            "memory_dim": latent_dim,
            "out_dim": primary_dim,
        },
        transition_core_name=transition,
        transition_core_kwargs={"input_dim": primary_dim, "hidden_dim": primary_dim},
        prediction_head_name="role_split",
        prediction_head_kwargs={"hidden_dim": primary_dim, "latent_dim": latent_dim},
        decoder_configs=[("vector_reconstruction", {"latent_dim": latent_dim, "output_dim": vector_dim})],
    )
    return cfg, build_model(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True, help="Minari dataset id, e.g. one printed by list_remote_datasets().")
    parser.add_argument("--download", action="store_true", help="Download dataset before loading it.")
    parser.add_argument("--force-download", action="store_true", help="Force re-download if supported by installed Minari.")
    parser.add_argument("--max-transitions", type=int, default=50_000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--transition", default="mlp", choices=["mlp", "gru", "attnres_transformer", "recurrent_attnres_transformer", "mod_recurrent_attnres_transformer"])
    parser.add_argument("--run-dir", default="runs/mmwm_minari")
    parser.add_argument("--status-file", default=None, help="Optional JSON status file for dashboard polling.")
    args = parser.parse_args()

    minari = _require_minari()
    if args.download:
        print(f"Downloading Minari dataset: {args.dataset_id}", flush=True)
        try:
            minari.download_dataset(args.dataset_id, force_download=args.force_download)
        except TypeError:
            minari.download_dataset(args.dataset_id)

    dataset = MinariTransitionDataset(args.dataset_id, max_transitions=args.max_transitions)
    print(f"Loaded {args.dataset_id}", flush=True)
    print(f"Transitions: {len(dataset)} vector_dim={dataset.vector_dim} action_dim={dataset.action_dim}", flush=True)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_transition_batch, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model = build_mmwm(dataset.vector_dim, dataset.action_dim, args.latent_dim, args.hidden_dim, args.transition)
    # Build the loss before the optimizer: with learned_uncertainty=True it
    # owns 10 nn.Parameter log_vars that must be optimized, otherwise the
    # learned task-weighting never updates.
    loss_fn = WorldModelLoss(learned_uncertainty=True).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = build_lr_scheduler(optimizer, total_steps=args.steps)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        run_dir=args.run_dir,
        mixed_precision=(device.type == "cuda"),
        lr_scheduler=scheduler,
        reset_memory_each_batch=True,
    )

    started = time.time()
    status: Dict[str, Any] = {
        "state": "running",
        "step": 0,
        "total": args.steps,
        "loss": None,
        "device": str(device),
        "dataset_id": args.dataset_id,
        "vector_dim": dataset.vector_dim,
        "action_dim": dataset.action_dim,
        "metrics": {},
        "log": [f"Loaded {args.dataset_id}"],
    }
    write_status(args.status_file, status)

    step = 0
    try:
        while step < args.steps:
            for batch in loader:
                metrics, _ = trainer.train_step(batch)
                step += 1
                loss = float(metrics["total_loss"])
                status.update({
                    "state": "running",
                    "step": step,
                    "loss": loss,
                    "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                    "elapsed_sec": round(time.time() - started, 2),
                })
                if step == 1 or step % 25 == 0:
                    line = f"step={step} total_loss={loss:.6f}"
                    print(line, flush=True)
                    status["log"].append(line)
                write_status(args.status_file, status)
                if step >= args.steps:
                    break
    except KeyboardInterrupt:
        status["state"] = "stopped"
        status["log"].append("Interrupted by user.")
        write_status(args.status_file, status)
        raise

    checkpoint_path = f"{args.run_dir}/checkpoint_final.pt"
    trainer.save_checkpoint(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt["model_config"] = asdict(cfg)
    ckpt["dataset_id"] = args.dataset_id
    ckpt["vector_dim"] = dataset.vector_dim
    ckpt["action_dim"] = dataset.action_dim
    torch.save(ckpt, checkpoint_path)
    status.update({
        "state": "finished",
        "checkpoint": checkpoint_path,
        "elapsed_sec": round(time.time() - started, 2),
    })
    status["log"].append(f"Saved checkpoint: {checkpoint_path}")
    write_status(args.status_file, status)
    print(f"Saved checkpoint: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
