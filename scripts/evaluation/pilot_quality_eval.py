#!/usr/bin/env python3
"""Evaluate pilot checkpoints on paired image/text datasets.

This is a focused companion to ``scripts.training.pilot_train``. It reports
fixed route metrics for the paired image/text paths and writes qualitative
artifacts that make failure modes visible: target images beside reconstructions
and generated/teacher-forced text beside decoded targets.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from omnilatent.config import OmniLatentConfig
from omnilatent.data.collate import decode_eos_byte_tokens
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.coco_dataset import CocoCaptionsDataset
from omnilatent.training.data import ROW_SUFFIX, collate_multimodal
from omnilatent.training.losses import MultiModalLoss
from omnilatent.utils import count_parameters, param_size_mb, set_seed
from scripts.training.pilot_train import MNISTLabelDataset

Route = tuple[Literal["text", "image"], Literal["text", "image"]]
ROUTES: tuple[Route, ...] = (
    ("text", "text"),
    ("text", "image"),
    ("image", "text"),
    ("image", "image"),
)


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


def _config_from_checkpoint(raw: Any) -> OmniLatentConfig:
    if isinstance(raw, OmniLatentConfig):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint config type: {type(raw)!r}")
    valid = {f.name for f in fields(OmniLatentConfig)}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    return OmniLatentConfig(**kwargs)


def _strip_compiled_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state_dict):
        return {k.removeprefix(prefix): v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(path: Path, device: torch.device) -> tuple[OmniLatentModel, MultiModalLoss, OmniLatentConfig, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = _config_from_checkpoint(ckpt["config"])
    config.mixed_precision = False

    model = OmniLatentModel(config)
    model.load_state_dict(_strip_compiled_prefix(ckpt["model"]))
    model.to(device)
    model.eval()

    criterion = MultiModalLoss(config).to(device)
    if "criterion" in ckpt:
        criterion.load_state_dict(ckpt["criterion"], strict=False)
    criterion.eval()

    info = {
        "checkpoint": str(path),
        "global_step": ckpt.get("global_step", ckpt.get("step")),
        "parameters": count_parameters(model),
        "parameter_size_mb_fp32": param_size_mb(model),
    }
    return model, criterion, config, info


def build_dataset(args: argparse.Namespace, config: OmniLatentConfig) -> Dataset:
    if args.dataset == "coco-captions":
        if args.image_dir is None or args.annotation_file is None:
            raise ValueError("--dataset coco-captions requires --image-dir and --annotation-file")
        base: Dataset = CocoCaptionsDataset(
            image_dir=args.image_dir,
            annotation_file=args.annotation_file,
            config=config,
            augment=False,
        )
    elif args.dataset == "mnist-labels":
        base = MNISTLabelDataset(
            config,
            root=args.data_root,
            train=args.split == "train",
            download=args.download,
            length=None,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    start = args.start_index
    stop = len(base) if args.data_length is None else min(len(base), start + args.data_length)
    if start < 0 or start >= len(base):
        raise ValueError(f"--start-index {start} is outside dataset length {len(base)}")
    if stop <= start:
        raise ValueError(f"Empty dataset slice: start={start}, stop={stop}, len={len(base)}")
    return Subset(base, range(start, stop))


def _tokens_to_bytes(
    tokens: torch.Tensor,
    mode: str,
    config: OmniLatentConfig | None = None,
) -> tuple[str, int]:
    if mode == "eos":
        if config is None:
            raise ValueError("EOS token decoding requires config")
        return decode_eos_byte_tokens(
            tokens,
            bos_token=config.text_bos_token,
            eos_token=config.text_eos_token,
        )
    values: list[int] = []
    out_of_byte_vocab = 0
    for token in tokens.detach().cpu().flatten().tolist():
        token = int(token)
        if token == 0:
            continue
        if mode == "coco":
            if token == 1:
                continue
            byte = token - 2
        else:
            byte = token - 1
        if 0 <= byte <= 255:
            values.append(byte)
        else:
            out_of_byte_vocab += 1
    return bytes(values).decode("utf-8", errors="replace"), out_of_byte_vocab


def _image_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.float()
    target = target.float()
    pred_clamped = pred.clamp(0.0, 1.0)
    mse = F.mse_loss(pred_clamped, target).item()
    psnr = 99.0 if mse <= 1e-12 else -10.0 * math.log10(mse)
    return {
        "l1_clamped": F.l1_loss(pred_clamped, target).item(),
        "mse_clamped": mse,
        "psnr_clamped": psnr,
        "pred_min": pred.min().item(),
        "pred_max": pred.max().item(),
        "pred_mean": pred.mean().item(),
    }


def _text_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_len = logits.shape[1]
    tgt_len = target.shape[1]
    length = min(pred_len, tgt_len)
    logits = logits[:, :length]
    target = target[:, :length]
    predicted = logits.argmax(dim=-1)
    mask = target.ne(0)
    token_count = int(mask.sum().item())
    correct = int((predicted.eq(target) & mask).sum().item())
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        ignore_index=0,
    ).item()
    return {
        "cross_entropy": ce,
        "perplexity": math.exp(min(ce, 20.0)),
        "token_accuracy": correct / max(token_count, 1),
        "token_count": float(token_count),
    }


def _new_route_stats() -> dict[str, Any]:
    return {
        "batches": 0,
        "samples": 0,
        "loss_sum": 0.0,
        "text_ce_sum": 0.0,
        "text_accuracy_sum": 0.0,
        "text_token_count": 0.0,
        "image_l1_sum": 0.0,
        "image_mse_sum": 0.0,
        "image_psnr_sum": 0.0,
        "pred_min": None,
        "pred_max": None,
        "pred_mean_sum": 0.0,
    }


def _update_route_stats(
    stats: dict[str, Any],
    target_modality: str,
    loss: torch.Tensor,
    output: torch.Tensor,
    target: torch.Tensor,
) -> None:
    batch_size = int(target.shape[0])
    stats["batches"] += 1
    stats["samples"] += batch_size
    stats["loss_sum"] += float(loss.detach().item()) * batch_size
    if target_modality == "text":
        m = _text_metrics(output, target)
        token_count = m["token_count"]
        stats["text_ce_sum"] += m["cross_entropy"] * token_count
        stats["text_accuracy_sum"] += m["token_accuracy"] * token_count
        stats["text_token_count"] += token_count
    elif target_modality == "image":
        m = _image_metrics(output, target)
        stats["image_l1_sum"] += m["l1_clamped"] * batch_size
        stats["image_mse_sum"] += m["mse_clamped"] * batch_size
        stats["image_psnr_sum"] += m["psnr_clamped"] * batch_size
        stats["pred_min"] = m["pred_min"] if stats["pred_min"] is None else min(stats["pred_min"], m["pred_min"])
        stats["pred_max"] = m["pred_max"] if stats["pred_max"] is None else max(stats["pred_max"], m["pred_max"])
        stats["pred_mean_sum"] += m["pred_mean"] * batch_size


def _finalize_stats(stats_by_route: dict[str, dict[str, Any]]) -> dict[str, Any]:
    final: dict[str, Any] = {}
    for route, stats in stats_by_route.items():
        samples = max(int(stats["samples"]), 1)
        route_out: dict[str, Any] = {
            "batches": stats["batches"],
            "samples": stats["samples"],
            "loss": stats["loss_sum"] / samples,
        }
        if stats["text_token_count"] > 0:
            tokens = stats["text_token_count"]
            ce = stats["text_ce_sum"] / tokens
            route_out.update(
                {
                    "text_cross_entropy": ce,
                    "text_perplexity": math.exp(min(ce, 20.0)),
                    "text_token_accuracy": stats["text_accuracy_sum"] / tokens,
                    "text_token_count": tokens,
                }
            )
        if stats["image_l1_sum"] > 0:
            route_out.update(
                {
                    "image_l1_clamped": stats["image_l1_sum"] / samples,
                    "image_mse_clamped": stats["image_mse_sum"] / samples,
                    "image_psnr_clamped": stats["image_psnr_sum"] / samples,
                    "pred_min": stats["pred_min"],
                    "pred_max": stats["pred_max"],
                    "pred_mean": stats["pred_mean_sum"] / samples,
                }
            )
        final[route] = route_out
    return final


@torch.inference_mode()
def evaluate_routes(
    model: OmniLatentModel,
    criterion: MultiModalLoss,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    stats = {f"{src}->{tgt}": _new_route_stats() for src, tgt in ROUTES}
    started = time.time()
    for batch_idx, batch in enumerate(dataloader, start=1):
        data = {
            k: v.to(device, non_blocking=True)
            for k, v in batch.items()
            if not k.endswith(ROW_SUFFIX)
        }
        for src, tgt in ROUTES:
            source = data[src]
            target = data[tgt]
            result = model(src, source, tgt, target)
            output = result["output"]
            loss = criterion.recon_loss(tgt, output, target)
            _update_route_stats(stats[f"{src}->{tgt}"], tgt, loss, output, target)
        if batch_idx % 25 == 0:
            print(f"  evaluated {batch_idx} batches", flush=True)
    return {
        "elapsed_sec": time.time() - started,
        "routes": _finalize_stats(stats),
    }


@torch.inference_mode()
def write_qualitative_artifacts(
    model: OmniLatentModel,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    tokenizer_mode: str,
    config: OmniLatentConfig,
    generate_len: int,
) -> dict[str, Any]:
    from torchvision.utils import save_image

    output_dir.mkdir(parents=True, exist_ok=True)
    batch = next(iter(dataloader))
    data = {
        k: v.to(device, non_blocking=True)
        for k, v in batch.items()
        if not k.endswith(ROW_SUFFIX)
    }
    text = data["text"]
    image = data["image"]
    n = min(text.shape[0], image.shape[0])
    text = text[:n]
    image = image[:n]

    text_to_image = model("text", text, "image", image)["output"].detach().cpu()
    image_to_image = model("image", image, "image", image)["output"].detach().cpu()
    image_to_text_logits = model("image", image, "text", text)["output"].detach().cpu()
    text_to_text_logits = model("text", text, "text", text)["output"].detach().cpu()

    image_cpu = image.detach().cpu()
    grid = torch.cat(
        [
            image_cpu.clamp(0.0, 1.0),
            text_to_image.clamp(0.0, 1.0),
            image_to_image.clamp(0.0, 1.0),
        ],
        dim=0,
    )
    grid_path = output_dir / "image_grid_target_text2image_image2image.png"
    save_image(grid, grid_path, nrow=n)

    generated_from_image = model.generate("image", image, max_len=generate_len).detach().cpu()
    generated_from_text = model.generate("text", text, max_len=generate_len).detach().cpu()

    samples: list[dict[str, Any]] = []
    for i in range(n):
        target_text, target_oov = _tokens_to_bytes(text[i], tokenizer_mode, config)
        image_teacher, image_teacher_oov = _tokens_to_bytes(
            image_to_text_logits[i].argmax(dim=-1), tokenizer_mode, config
        )
        text_teacher, text_teacher_oov = _tokens_to_bytes(
            text_to_text_logits[i].argmax(dim=-1), tokenizer_mode, config
        )
        image_generated, image_generated_oov = _tokens_to_bytes(
            generated_from_image[i], tokenizer_mode, config
        )
        text_generated, text_generated_oov = _tokens_to_bytes(
            generated_from_text[i], tokenizer_mode, config
        )
        samples.append(
            {
                "sample": i,
                "target_text": target_text,
                "target_oov_tokens": target_oov,
                "image_to_text_teacher_forced": image_teacher,
                "image_to_text_teacher_forced_oov_tokens": image_teacher_oov,
                "text_to_text_teacher_forced": text_teacher,
                "text_to_text_teacher_forced_oov_tokens": text_teacher_oov,
                "image_to_text_generated": image_generated,
                "image_to_text_generated_oov_tokens": image_generated_oov,
                "text_to_text_generated": text_generated,
                "text_to_text_generated_oov_tokens": text_generated_oov,
                "image_to_text_generated_token_ids": generated_from_image[i].tolist(),
                "text_to_text_generated_token_ids": generated_from_text[i].tolist(),
            }
        )

    samples_path = output_dir / "text_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Qualitative Artifacts",
                "",
                "`image_grid_target_text2image_image2image.png` has three rows:",
                "1. target/input image",
                "2. `text->image` prediction",
                "3. `image->image` prediction",
                "",
                "`text_samples.jsonl` contains decoded targets, teacher-forced text predictions,",
                "greedy generated text, and generated token IDs.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "image_grid": str(grid_path),
        "text_samples": str(samples_path),
        "readme": str(readme_path),
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pilot checkpoint quality on paired image/text data")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dataset", choices=["coco-captions", "mnist-labels"], required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--download", action="store_true")
    p.add_argument("--image-dir", default=None)
    p.add_argument("--annotation-file", default=None)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--data-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--qualitative-samples", type=int, default=6)
    p.add_argument("--generate-len", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260702)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    checkpoint = Path(args.checkpoint)
    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs") / f"eval_{checkpoint.parent.parent.name}_{args.dataset}_{stamp}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, criterion, config, checkpoint_info = load_checkpoint(checkpoint, device)

    # The checkpoint controls model size; the eval batch size controls loader size.
    config.batch_size = args.batch_size
    dataset = build_dataset(args, config)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_multimodal,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    qual_loader = DataLoader(
        Subset(dataset, range(min(args.qualitative_samples, len(dataset)))),
        batch_size=min(args.qualitative_samples, len(dataset)),
        shuffle=False,
        collate_fn=collate_multimodal,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("=" * 72, flush=True)
    print("Pilot quality evaluation", flush=True)
    print("=" * 72, flush=True)
    print(f"checkpoint: {checkpoint}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    print(f"dataset: {args.dataset}  slice: {args.start_index}:{args.start_index + len(dataset)}", flush=True)
    print(f"device: {device}", flush=True)
    print(
        f"parameters: {checkpoint_info['parameters']:,} "
        f"({checkpoint_info['parameter_size_mb_fp32']:.1f} MB fp32)",
        flush=True,
    )

    metrics = evaluate_routes(model, criterion, dataloader, device)
    tokenizer_mode = "eos"
    qualitative = write_qualitative_artifacts(
        model,
        qual_loader,
        device,
        output_dir / "qualitative",
        tokenizer_mode,
        config,
        args.generate_len,
    )

    summary = {
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": vars(args),
        "checkpoint": checkpoint_info,
        "config": config.__dict__,
        "dataset_size": len(dataset),
        "device": str(device),
        "metrics": metrics,
        "qualitative": {k: v for k, v in qualitative.items() if k != "samples"},
    }
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    (output_dir / "qualitative" / "samples_preview.json").write_text(
        json.dumps(_jsonable(qualitative["samples"]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Evaluation complete.", flush=True)
    print(f"summary: {output_dir / 'summary.json'}", flush=True)
    for route, route_metrics in metrics["routes"].items():
        print(f"{route}: loss={route_metrics['loss']:.4f}", flush=True)


if __name__ == "__main__":
    main()
