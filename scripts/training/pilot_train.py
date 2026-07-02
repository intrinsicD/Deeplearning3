#!/usr/bin/env python3
"""Bounded OmniLatent pilot training run with structured telemetry.

This is intentionally a pilot runner, not the main long-run trainer. It keeps
the data synthetic and paired so every modality path is exercised, lowers the
warmup by default so a short run actually takes optimizer steps at useful LR,
and writes JSONL metrics/checkpoints for post-run diagnosis.

Example:
    python -m scripts.training.pilot_train --steps 2000 --batch-size 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from omnilatent.config import OmniLatentConfig
from omnilatent.data.collate import decode_eos_byte_tokens, eos_byte_tokenize
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.coco_dataset import CocoCaptionsDataset
from omnilatent.training.data import ROW_SUFFIX, SyntheticMultiModalDataset, build_dataloader, collate_multimodal
from omnilatent.training.trainer import Trainer
from omnilatent.utils import count_parameters, param_size_mb, set_seed


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _device_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info.update(
            {
                "gpu": torch.cuda.get_device_name(),
                "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            }
        )
    return info


def _save_checkpoint(path: Path, trainer: Trainer, config: OmniLatentConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_step": trainer.global_step,
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "criterion": trainer.criterion.state_dict(),
            "config": config.__dict__,
            "metrics": trainer.metrics.summary(),
        },
        path,
    )


def _changed_task(before: dict[str, int], after: dict[str, int]) -> str | None:
    for task, count in after.items():
        if count != before.get(task, 0):
            return task
    return None


class MNISTLabelDataset(Dataset):
    """Real MNIST images paired with simple label text.

    Each sample carries exactly two modalities:
      * ``image``: RGB tensor resized to the OmniLatent image resolution.
      * ``text``: byte-tokenized label phrase, e.g. ``"digit seven"``.

    This is not a rich image-caption dataset, but it is a small, local,
    deterministic real-data bridge for the image/text training path.
    """

    LABEL_TEXT = [
        "digit zero",
        "digit one",
        "digit two",
        "digit three",
        "digit four",
        "digit five",
        "digit six",
        "digit seven",
        "digit eight",
        "digit nine",
    ]

    def __init__(
        self,
        config: OmniLatentConfig,
        root: str = "data",
        *,
        train: bool = True,
        download: bool = False,
        length: int | None = None,
    ) -> None:
        try:
            from torchvision.datasets import MNIST
            from torchvision.transforms import ToTensor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "MNIST pilot requires torchvision. Install torchvision or use "
                "--dataset synthetic."
            ) from exc

        self.config = config
        self.dataset = MNIST(root=root, train=train, download=download, transform=ToTensor())
        self.length = min(length, len(self.dataset)) if length is not None else len(self.dataset)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image, label = self.dataset[idx]
        image = image.float()
        if image.shape[0] == 1 and self.config.image_channels == 3:
            image = image.expand(3, -1, -1)
        elif image.shape[0] != self.config.image_channels:
            image = image[: self.config.image_channels]
        if image.shape[-2:] != (self.config.image_size, self.config.image_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        text = eos_byte_tokenize(
            self.LABEL_TEXT[int(label)],
            self.config.text_max_len,
            self.config.vocab_size,
            bos_token=self.config.text_bos_token,
            eos_token=self.config.text_eos_token,
        )
        return {"image": image, "text": text}


def _slice_dataset(dataset: Dataset, start_index: int, length: int | None) -> Dataset:
    if start_index < 0 or start_index >= len(dataset):
        raise ValueError(f"start_index {start_index} is outside dataset length {len(dataset)}")
    stop = len(dataset) if length is None else min(len(dataset), start_index + length)
    if stop <= start_index:
        raise ValueError(f"Empty dataset slice: start={start_index}, stop={stop}")
    if start_index == 0 and stop == len(dataset):
        return dataset
    return Subset(dataset, range(start_index, stop))


def _build_dataset(
    args: argparse.Namespace,
    config: OmniLatentConfig,
    *,
    split: str | None = None,
    length: int | None = None,
    start_index: int = 0,
) -> Dataset:
    split = split or args.split
    length = args.data_length if length is None else length
    if args.dataset == "synthetic":
        return SyntheticMultiModalDataset(config, length=args.data_length, paired=True)
    if args.dataset == "mnist-labels":
        dataset = MNISTLabelDataset(
            config,
            root=args.data_root,
            train=split == "train",
            download=args.download,
            length=None,
        )
        return _slice_dataset(dataset, start_index, length)
    if args.dataset == "coco-captions":
        if args.image_dir is None or args.annotation_file is None:
            raise ValueError(
                "--dataset coco-captions requires --image-dir and --annotation-file"
            )
        dataset: Dataset = CocoCaptionsDataset(
            image_dir=args.image_dir,
            annotation_file=args.annotation_file,
            config=config,
            augment=args.augment,
        )
        return _slice_dataset(dataset, start_index, length)
    raise ValueError(f"Unknown dataset: {args.dataset}")


def _image_text_eval_loader(
    args: argparse.Namespace,
    config: OmniLatentConfig,
) -> DataLoader | None:
    wants_eval = args.eval_every > 0 or args.eval_final or args.qualitative_samples > 0
    if not wants_eval or args.dataset == "synthetic":
        return None
    start_index = args.eval_start_index
    if start_index is None:
        start_index = args.data_length if args.dataset == "coco-captions" else 0
    dataset = _build_dataset(
        args,
        config,
        split=args.eval_split,
        length=args.eval_data_length,
        start_index=start_index,
    )
    return DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_multimodal,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _text_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    length = min(logits.shape[1], targets.shape[1])
    predicted = logits[:, :length].argmax(dim=-1)
    target = targets[:, :length]
    mask = target.ne(0)
    correct = int((predicted.eq(target) & mask).sum().item())
    total = int(mask.sum().item())
    return correct, total


def _image_metric_values(output: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = output.float()
    target_f = target.float()
    pred_clamped = pred.clamp(0.0, 1.0)
    mse = F.mse_loss(pred_clamped, target_f).item()
    psnr = 99.0 if mse <= 1e-12 else -10.0 * torch.log10(torch.tensor(mse)).item()
    return {
        "l1_clamped": F.l1_loss(pred_clamped, target_f).item(),
        "mse_clamped": mse,
        "psnr_clamped": psnr,
        "pred_min": pred.min().item(),
        "pred_max": pred.max().item(),
        "pred_mean": pred.mean().item(),
        "pred_std": pred.std(unbiased=False).item(),
    }


@torch.inference_mode()
def _evaluate_image_text_routes(
    trainer: Trainer,
    dataloader: DataLoader,
    *,
    max_batches: int,
) -> dict[str, Any]:
    was_training = trainer.model.training
    trainer.model.eval()
    routes = (("text", "text"), ("text", "image"), ("image", "text"), ("image", "image"))
    stats: dict[str, dict[str, float]] = {
        f"{src}->{tgt}": {
            "loss_sum": 0.0,
            "samples": 0.0,
            "correct": 0.0,
            "tokens": 0.0,
            "image_l1_sum": 0.0,
            "image_mse_sum": 0.0,
            "image_psnr_sum": 0.0,
            "pred_mean_sum": 0.0,
            "pred_std_sum": 0.0,
            "pred_min": float("inf"),
            "pred_max": float("-inf"),
        }
        for src, tgt in routes
    }
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        data = {
            k: v.to(trainer.device, non_blocking=True)
            for k, v in batch.items()
            if not k.endswith(ROW_SUFFIX)
        }
        if "text" not in data or "image" not in data:
            continue
        for src, tgt in routes:
            source = data[src]
            target = data[tgt]
            output = trainer.model(src, source, tgt, target)["output"]
            loss = trainer.criterion.recon_loss(tgt, output, target)
            batch_size = float(target.shape[0])
            route_stats = stats[f"{src}->{tgt}"]
            route_stats["loss_sum"] += float(loss.item()) * batch_size
            route_stats["samples"] += batch_size
            if tgt == "text":
                correct, tokens = _text_accuracy(output, target)
                route_stats["correct"] += float(correct)
                route_stats["tokens"] += float(tokens)
            elif tgt == "image":
                image_metrics = _image_metric_values(output, target)
                route_stats["image_l1_sum"] += image_metrics["l1_clamped"] * batch_size
                route_stats["image_mse_sum"] += image_metrics["mse_clamped"] * batch_size
                route_stats["image_psnr_sum"] += image_metrics["psnr_clamped"] * batch_size
                route_stats["pred_mean_sum"] += image_metrics["pred_mean"] * batch_size
                route_stats["pred_std_sum"] += image_metrics["pred_std"] * batch_size
                route_stats["pred_min"] = min(route_stats["pred_min"], image_metrics["pred_min"])
                route_stats["pred_max"] = max(route_stats["pred_max"], image_metrics["pred_max"])

    if was_training:
        trainer.model.train()

    out: dict[str, Any] = {}
    for route, route_stats in stats.items():
        samples = max(route_stats["samples"], 1.0)
        record: dict[str, Any] = {
            "loss": route_stats["loss_sum"] / samples,
            "samples": int(route_stats["samples"]),
        }
        if route_stats["tokens"] > 0:
            record["text_token_accuracy"] = route_stats["correct"] / route_stats["tokens"]
            record["text_token_count"] = int(route_stats["tokens"])
        if route_stats["image_l1_sum"] > 0:
            record["image_l1_clamped"] = route_stats["image_l1_sum"] / samples
            record["image_mse_clamped"] = route_stats["image_mse_sum"] / samples
            record["image_psnr_clamped"] = route_stats["image_psnr_sum"] / samples
            record["pred_mean"] = route_stats["pred_mean_sum"] / samples
            record["pred_std"] = route_stats["pred_std_sum"] / samples
            record["pred_min"] = route_stats["pred_min"]
            record["pred_max"] = route_stats["pred_max"]
        out[route] = record
    return out


@torch.inference_mode()
def _write_qualitative_artifacts(
    trainer: Trainer,
    dataloader: DataLoader,
    output_dir: Path,
    *,
    max_samples: int,
    generate_len: int,
) -> dict[str, Any]:
    from torchvision.utils import save_image

    was_training = trainer.model.training
    trainer.model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = next(iter(dataloader))
    data = {
        k: v.to(trainer.device, non_blocking=True)
        for k, v in batch.items()
        if not k.endswith(ROW_SUFFIX)
    }
    if "text" not in data or "image" not in data:
        return {}
    n = min(max_samples, data["text"].shape[0], data["image"].shape[0])
    text = data["text"][:n]
    image = data["image"][:n]

    text_to_image = trainer.model("text", text, "image", image)["output"].detach().cpu()
    image_to_image = trainer.model("image", image, "image", image)["output"].detach().cpu()
    grid = torch.cat(
        [image.detach().cpu().clamp(0, 1), text_to_image.clamp(0, 1), image_to_image.clamp(0, 1)],
        dim=0,
    )
    image_grid = output_dir / "image_grid_target_text2image_image2image.png"
    save_image(grid, image_grid, nrow=n)

    image_to_text = trainer.model.generate("image", image, max_len=generate_len).detach().cpu()
    text_to_text = trainer.model.generate("text", text, max_len=generate_len).detach().cpu()
    samples_path = output_dir / "text_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            target_text, target_oov = decode_eos_byte_tokens(
                text[i],
                bos_token=trainer.config.text_bos_token,
                eos_token=trainer.config.text_eos_token,
            )
            image_text, image_oov = decode_eos_byte_tokens(
                image_to_text[i],
                bos_token=trainer.config.text_bos_token,
                eos_token=trainer.config.text_eos_token,
            )
            self_text, self_oov = decode_eos_byte_tokens(
                text_to_text[i],
                bos_token=trainer.config.text_bos_token,
                eos_token=trainer.config.text_eos_token,
            )
            f.write(
                json.dumps(
                    {
                        "sample": i,
                        "target_text": target_text,
                        "target_oov_tokens": target_oov,
                        "image_to_text_generated": image_text,
                        "image_to_text_generated_oov_tokens": image_oov,
                        "text_to_text_generated": self_text,
                        "text_to_text_generated_oov_tokens": self_oov,
                        "image_to_text_generated_token_ids": image_to_text[i].tolist(),
                        "text_to_text_generated_token_ids": text_to_text[i].tolist(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    if was_training:
        trainer.model.train()
    return {"image_grid": str(image_grid), "text_samples": str(samples_path)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a bounded OmniLatent pilot training job")
    p.add_argument("--run-dir", default=None, help="Output directory; defaults to runs/pilot_<timestamp>")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument(
        "--image-warmup-steps",
        type=int,
        default=0,
        help="Run this many fixed image->image autoencoder steps before the main route sampler.",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--image-decoder", choices=["deconv", "patch", "gaussian"], default="deconv")
    p.add_argument("--image-gaussians-per-token", type=int, default=1)
    p.add_argument("--image-gaussian-chunk-size", type=int, default=128)
    p.add_argument("--image-gaussian-min-scale", type=float, default=0.015)
    p.add_argument("--image-gaussian-max-scale", type=float, default=0.35)
    p.add_argument("--image-gaussian-offset-scale", type=float, default=0.75)
    p.add_argument("--image-gaussian-anchor-jitter", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--image-edge-weight", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--data-length", type=int, default=10_000)
    p.add_argument(
        "--dataset",
        choices=["synthetic", "mnist-labels", "coco-captions"],
        default="synthetic",
    )
    p.add_argument("--data-root", default="data")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--download", action="store_true", help="Download/process dataset if missing")
    p.add_argument("--image-dir", default=None, help="Image directory for COCO caption pilots")
    p.add_argument(
        "--annotation-file",
        default=None,
        help="COCO captions JSON for COCO caption pilots",
    )
    p.add_argument(
        "--augment",
        action="store_true",
        help="Enable image augmentation for datasets that support it.",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260702)
    p.add_argument("--eval-every", type=int, default=0, help="Evaluate fixed image/text routes every N steps")
    p.add_argument("--eval-final", action="store_true", help="Run fixed image/text route eval after training")
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--eval-data-length", type=int, default=256)
    p.add_argument("--eval-start-index", type=int, default=None)
    p.add_argument("--eval-split", choices=["train", "test"], default="test")
    p.add_argument("--qualitative-samples", type=int, default=0)
    p.add_argument("--generate-len", type=int, default=64)
    amp_group = p.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Enable mixed precision. Disabled by default for stable gradient telemetry.",
    )
    amp_group.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable mixed precision. This is the default.",
    )
    p.set_defaults(amp=False)
    p.add_argument("--no-checkpointing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if args.image_warmup_steps < 0:
        raise ValueError("--image-warmup-steps must be >= 0")
    set_seed(args.seed)

    if args.run_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / f"pilot_omnilatent_{stamp}"
    else:
        run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    validation_path = run_dir / "validation.jsonl"

    total_steps = args.image_warmup_steps + args.steps
    config = OmniLatentConfig(
        hidden_dim=args.dim,
        num_layers=args.layers,
        num_heads=args.heads,
        max_steps=total_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        mixed_precision=args.amp,
        gradient_checkpointing=not args.no_checkpointing,
        seed=args.seed,
        image_decoder=args.image_decoder,
        image_edge_loss_weight=args.image_edge_weight,
        image_gaussians_per_token=args.image_gaussians_per_token,
        image_gaussian_chunk_size=args.image_gaussian_chunk_size,
        image_gaussian_min_scale=args.image_gaussian_min_scale,
        image_gaussian_max_scale=args.image_gaussian_max_scale,
        image_gaussian_offset_scale=args.image_gaussian_offset_scale,
        image_gaussian_anchor_jitter=args.image_gaussian_anchor_jitter,
    )

    print("=" * 72, flush=True)
    print("OmniLatent pilot training", flush=True)
    print("=" * 72, flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"dataset: {args.dataset}  split: {args.split}  root: {args.data_root}", flush=True)
    if args.dataset == "coco-captions":
        print(f"image_dir: {args.image_dir}", flush=True)
        print(f"annotation_file: {args.annotation_file}", flush=True)
    print(
        f"steps: {args.steps}  image_warmup_steps: {args.image_warmup_steps}  "
        f"batch_size: {args.batch_size}  lr_warmup: {args.warmup_steps}",
        flush=True,
    )
    print(
        f"model: dim={args.dim} layers={args.layers} heads={args.heads} "
        f"image_decoder={args.image_decoder} "
        f"gaussians_per_token={args.image_gaussians_per_token}",
        flush=True,
    )
    if args.image_decoder == "gaussian":
        print(
            f"gaussian_scale=[{args.image_gaussian_min_scale}, {args.image_gaussian_max_scale}] "
            f"offset_scale={args.image_gaussian_offset_scale} "
            f"anchor_jitter={args.image_gaussian_anchor_jitter}",
            flush=True,
        )
    print(f"image_edge_weight: {args.image_edge_weight}", flush=True)
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)

    model = OmniLatentModel(config)
    print(
        f"parameters: {count_parameters(model):,} ({param_size_mb(model):.1f} MB fp32)",
        flush=True,
    )

    dataset = _build_dataset(args, config, length=args.data_length, start_index=0)
    dataloader = build_dataloader(config, dataset, num_workers=args.num_workers)
    trainer = Trainer(model, config, dataloader, seed=args.seed)
    eval_loader = _image_text_eval_loader(args, config)

    run_info = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": vars(args),
        "config": config.__dict__,
        "device": str(trainer.device),
        "device_info": _device_info(),
        "parameters": count_parameters(trainer.model),
        "parameter_size_mb_fp32": param_size_mb(trainer.model),
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))

    data_iter = iter(dataloader)
    started = time.time()
    interval_started = started
    interval_loss = 0.0
    interval_steps = 0
    interval_skipped = 0
    final_validation: dict[str, Any] | None = None
    qualitative_artifacts: dict[str, Any] | None = None

    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for step in range(total_steps):
            trainer.global_step = step
            lr = trainer._update_lr()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            phase = "image_warmup" if step < args.image_warmup_steps else "main"
            task_before = trainer.metrics.task_distribution()
            if phase == "image_warmup":
                losses = trainer.train_fixed_route_step(batch, "image", "image")
            else:
                losses = trainer._train_step(batch)
            task_after = trainer.metrics.task_distribution()
            task = _changed_task(task_before, task_after)

            skipped = bool(losses.get("skipped"))
            if skipped:
                interval_skipped += 1
            else:
                interval_loss += float(losses.get("total", 0.0))
                interval_steps += 1

            step_num = step + 1
            should_log = step_num == 1 or step_num % args.log_every == 0 or step_num == total_steps
            if should_log:
                elapsed = time.time() - started
                interval_elapsed = time.time() - interval_started
                avg_loss = interval_loss / max(interval_steps, 1)
                record: dict[str, Any] = {
                    "step": step_num,
                    "elapsed_sec": elapsed,
                    "lr": lr,
                    "task": task,
                    "phase": phase,
                    "losses": _jsonable(losses),
                    "interval_avg_loss": avg_loss,
                    "interval_steps": interval_steps,
                    "interval_skipped": interval_skipped,
                    "interval_steps_per_sec": args.log_every / max(interval_elapsed, 1e-9),
                    "avg_grad_norm": trainer.metrics.avg_grad_norm(),
                    "task_distribution": trainer.metrics.task_distribution(),
                }
                if torch.cuda.is_available():
                    record["cuda"] = {
                        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                    }
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()

                print(
                    f"step {step_num:>6d}/{total_steps} | "
                    f"{phase} | "
                    f"loss {avg_loss:.4f} | lr {lr:.2e} | "
                    f"grad {record['avg_grad_norm']:.2f} | "
                    f"{record['interval_steps_per_sec']:.1f} steps/s | "
                    f"task {task or '-'}",
                    flush=True,
                )
                interval_started = time.time()
                interval_loss = 0.0
                interval_steps = 0
                interval_skipped = 0

            if eval_loader is not None and args.eval_every > 0 and step_num % args.eval_every == 0:
                final_validation = _evaluate_image_text_routes(
                    trainer,
                    eval_loader,
                    max_batches=args.eval_batches,
                )
                validation_record = {
                    "step": step_num,
                    "phase": phase,
                    "routes": final_validation,
                }
                with validation_path.open("a", encoding="utf-8") as validation_file:
                    validation_file.write(json.dumps(_jsonable(validation_record)) + "\n")
                route_note = " ".join(
                    f"{route}:{vals['loss']:.3f}" for route, vals in final_validation.items()
                )
                print(f"validation step {step_num}: {route_note}", flush=True)

            if args.save_every > 0 and step_num % args.save_every == 0:
                _save_checkpoint(run_dir / "checkpoints" / "checkpoint_latest.pt", trainer, config)

    _save_checkpoint(run_dir / "checkpoints" / "checkpoint_final.pt", trainer, config)
    if eval_loader is not None and args.eval_final:
        final_validation = _evaluate_image_text_routes(
            trainer,
            eval_loader,
            max_batches=args.eval_batches,
        )
        validation_record = {
            "step": total_steps,
            "phase": "final",
            "routes": final_validation,
        }
        with validation_path.open("a", encoding="utf-8") as validation_file:
            validation_file.write(json.dumps(_jsonable(validation_record)) + "\n")
        route_note = " ".join(
            f"{route}:{vals['loss']:.3f}" for route, vals in final_validation.items()
        )
        print(f"validation final: {route_note}", flush=True)
    if eval_loader is not None and args.qualitative_samples > 0:
        qualitative_artifacts = _write_qualitative_artifacts(
            trainer,
            eval_loader,
            run_dir / "qualitative",
            max_samples=args.qualitative_samples,
            generate_len=args.generate_len,
        )
    summary = {
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_sec": time.time() - started,
        "metrics": trainer.metrics.summary(),
        "final_validation": final_validation,
        "qualitative_artifacts": qualitative_artifacts,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("Pilot training complete.", flush=True)
    print(f"summary: {run_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
